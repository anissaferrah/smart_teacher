"""Audio processing pipeline: STT → RAG → LLM → TTS for WebSocket."""

import asyncio
import base64
import logging
import time
import uuid

import numpy as np

from core.config import Config
from handlers.session_manager import detect_subject

log = logging.getLogger("SmartTeacher.AudioPipeline")

# NOTE: ``classify_intent`` was removed. It used four hand-curated keyword
# lists (``quiz_keywords``, ``code_keywords``, ``summary_keywords``,
# ``explain_keywords``) plus a substring-containment search to route audio
# queries into quiz/code/rag branches. The whole approach is rule-based:
# misspellings, paraphrases, and code-switching all bypass it, and the
# four classes were defined by what was easy to keyword-match, not by
# pedagogical need. The agentic Q&A graph (``IntentAgent``) already
# performs LLM-based classification with calibrated confidence; routing
# audio queries through the standard RAG path lets the graph make the
# decision instead. Voice-triggered quiz / code execution can be
# reintroduced later as an LLM-classified ``action`` field on
# ``VoiceIntent`` (learned, not keyword-matched).


async def run_pipeline_streaming(
    audio_data: np.ndarray,
    session_id: str,
    history: list,
    on_text_chunk=None,
    on_transcription=None,
    on_audio_chunk=None,
    on_state_change=None,  # ✅ NOUVEAU: Callback pour les mises à jour d'état
    force_language: str = None,
    course_id: str | None = None,
    ctx=None,
    slide_context: str = None,
    # Injected dependencies
    transcriber=None,
    rag=None,
    voice=None,
    brain=None,
    dialogue=None,
    csv_logger=None,
    stt_logger=None,
):
    """🚀 STREAMING PIPELINE: Real-time LLM → TTS streaming.
    
    Yields audio chunks as they're generated, enabling low-latency responses.
    """
    total_start = time.time()
    utt_id = str(uuid.uuid4())[:8]
    llm_error = None  # ✅ Track LLM failure

    # ── 1. STT ────────────────────────────────────────────────────────
    audio_samples = int(getattr(audio_data, "size", 0) or 0)
    audio_dur_input = audio_samples / 16000.0 if audio_samples else 0.0
    log.info(
        f"[{session_id[:8]}] 🎤 STT START | utt={utt_id} samples={audio_samples} "
        f"input_dur={audio_dur_input:.2f}s force_lang={force_language or 'auto'}"
    )
    stt_t0 = time.time()
    text, stt_time, lang, lang_prob, audio_duration = await asyncio.to_thread(
        transcriber.transcribe,
        audio_data,
        force_language,
        slide_context,
    )
    log.info(
        f"[{session_id[:8]}] 🎤 STT DONE | took={time.time()-stt_t0:.2f}s | "
        f"lang={lang}({lang_prob:.0%}) audio_dur={audio_duration:.2f}s | "
        f"text='{(text or '')[:80]}{'...' if text and len(text) > 80 else ''}'"
    )
    if not text or len(text.strip()) <= 2:
        log.info(f"[{session_id[:8]}] 🔇 STT NO-SPEECH | text='{text}' → abort")
        return {"no_speech": True, "message": "Aucune voix détectée"}

    stt_logger.log(
        session_id=session_id,
        utt_id=utt_id,
        audio_duration_sec=audio_duration,
        language_detected=lang,
        language_prob=lang_prob,
        stt_time=stt_time,
        transcription_text=text,
    )

    if on_transcription:
        await on_transcription(text, lang, round(lang_prob, 2))

    # ── 2. Extraction prosodique (Couche #2) ───────────────────────────
    # ✅ NOUVEAU: Extraire hésitations, vitesse parole pour détecter confusion implicite
    if on_state_change:
        await on_state_change("prosody_analysis", {"speech_rate": audio_duration})
    
    prosody = transcriber.extract_prosody(text, audio_duration)
    log.info(
        f"[{session_id[:8]}] 🎙️  Prosody: "
        f"speech_rate={prosody['speech_rate']} wpm, "
        f"hesitations={prosody['hesitation_count']}, "
        f"confusion_signals={prosody['markers']}"
    )

    # ── 3. RAG retrieval ──────────────────────────────────────────────
    # The previous version had a keyword-based action router here that
    # branched into "quiz" / "code" / "rag" based on substring matches.
    # That router has been removed — see the note above ``run_pipeline_streaming``.
    # Every audio query now flows through the standard RAG retrieval path;
    # higher-level routing (intent type, retrieve-or-skip) is handled by
    # the agentic Q&A graph downstream.
    subject = detect_subject(text)
    if on_state_change:
        await on_state_change("rag_search", {})
    try:
        chunks_with_scores = await asyncio.to_thread(
            rag.retrieve_chunks, text, k=Config.RAG_NUM_RESULTS, course_id=course_id
        )
    except Exception:
        chunks_with_scores = []

    # ── 4. Détection confusion ────────────────────────────────────────
    # ✅ GÉNÉRALISATION: Appel unique qui détecte + met à jour Redis
    # Inclut: mots-clés, répétition, patterns d'historique + SEMANTIC + PROSODY (Couches #1 + #2)
    is_confused, confusion_reason, q_hash, confusion_count = await dialogue.detect_and_track_confusion(
        session_id=session_id,
        question_text=text,
        language=lang,
        history=history,  # ← Inclure l'historique pour pattern detection
        brain=brain,      # ← Embeddings sémantiques (Couche D)
        prosody=prosody,  # ← NOUVEAU: Marqueurs prosodiques (Couche #2)
    )

    # ── 4. LLM STREAMING (avec TTS en parallèle) ─────────────────────
    llm_start = time.time()
    full_response = ""
    tts_engine = "edge"
    tts_voice = ""

    # Construire le prompt : normal ou reformulation si confusion
    last_slide = ctx.last_slide_explained if ctx else ""

    if is_confused:
        # ✅ UPDATE STATE: Confusion Detected
        if on_state_change:
            await on_state_change("confusion_detected", {"reason": confusion_reason})
        
        # Prompt spécial reformulation
        confusion_prompt = dialogue.build_confusion_prompt(
            original_question=text,
            language=lang,
            last_slide_content=last_slide,
        )
        question_for_llm = confusion_prompt
        log.info(f"[{session_id[:8]}] 📝 Prompt reformulation envoyé au LLM (audio)")
    else:
        # Prompt normal
        question_for_llm = text
    
    # ✅ UPDATE STATE: LLM Thinking
    if on_state_change:
        await on_state_change("llm_thinking", {"chunks": len(chunks_with_scores)})

    # Envoyer signal au frontend (pour afficher "Je vais reformuler...")
    if is_confused and on_text_chunk:
        preambles = {
            "fr": "Permettez-moi de réexpliquer autrement. ",
            "en": "Let me explain that differently. ",
        }
        preamble = preambles.get(lang[:2], preambles["fr"])
        await on_text_chunk(preamble, preamble)

        # Synthétiser et envoyer le préambule audio immédiatement
        try:
            pre_audio, _, pre_engine, pre_voice, pre_mime = await voice.generate_audio_async(
                preamble, language_code=lang
            )
            if pre_audio and on_audio_chunk:
                await on_audio_chunk(pre_audio, pre_mime)
                log.info(f"[{session_id[:8]}] 📤 Preamble audio streamed")
            tts_engine = pre_engine
            tts_voice = pre_voice
        except Exception as pre_exc:
            log.warning(f"[{session_id[:8]}] ⚠️  Preamble TTS failed: {pre_exc}")

    # ✅ UPDATE STATE: Streaming LLM → TTS
    if on_state_change:
        await on_state_change("streaming_llm", {})
    
    # Direct LLM call (agentic pipeline removed — to be redesigned)
    try:
        full_response, _ = await asyncio.to_thread(
            brain.ask,
            question_for_llm,
            reply_language=lang,
            session_id=session_id,
        )
        full_response = brain._clean_for_speech(full_response)

        if on_text_chunk:
            await on_text_chunk(full_response, full_response)

        if on_state_change:
            await on_state_change("tts_generating", {"response_length": len(full_response)})

        try:
            tts_t0 = time.time()
            log.info(
                f"[{session_id[:8]}] 🔊 TTS START | chars={len(full_response)} lang={lang} "
                f"text_preview='{full_response[:60]}{'...' if len(full_response) > 60 else ''}'"
            )
            audio_bytes, _, tts_engine, tts_voice, mime = await voice.generate_audio_async(
                full_response, language_code=lang
            )
            tts_elapsed = time.time() - tts_t0
            audio_kb = (len(audio_bytes) / 1024.0) if audio_bytes else 0.0
            log.info(
                f"[{session_id[:8]}] 🔊 TTS DONE | took={tts_elapsed:.2f}s | "
                f"engine={tts_engine} voice={tts_voice} bytes={len(audio_bytes or b'')} "
                f"({audio_kb:.1f}KB) mime={mime}"
            )
            if audio_bytes and on_audio_chunk:
                await on_audio_chunk(audio_bytes, mime)
                log.info(f"[{session_id[:8]}] 📤 Audio streamed: {len(audio_bytes)} bytes")
        except Exception as tts_exc:
            log.error(f"[{session_id[:8]}] ❌ TTS error: {tts_exc}")

    except Exception as llm_exc:
        log.warning(
            f"[{session_id[:8]}] ⚠️  LLM streaming failed ({type(llm_exc).__name__}): "
            f"{str(llm_exc)[:100]} → Fallback direct..."
        )
        llm_error = type(llm_exc).__name__  # ✅ Track error type

        try:
            full_response, _ = await asyncio.to_thread(
                brain.ask,
                text,
                reply_language=lang,
                session_id=session_id,
            )
            full_response = brain._clean_for_speech(full_response)

            if full_response.strip():
                audio_bytes, _, tts_engine, tts_voice, mime = await voice.generate_audio_async(
                    full_response, language_code=lang
                )
                if audio_bytes and on_audio_chunk:
                    await on_audio_chunk(audio_bytes, mime)
                    log.info(
                        f"[{session_id[:8]}] 📤 Fallback audio streamed: {len(audio_bytes)} bytes"
                    )

            if on_text_chunk:
                await on_text_chunk(full_response, full_response)

        except Exception as fallback_exc:
            log.error(f"[{session_id[:8]}] ❌ Fallback also failed: {fallback_exc}")
            # ✅ Return error status instead of generic message
            llm_error = f"LLM unavailable ({type(fallback_exc).__name__})"
            full_response = None  # Mark as failure

    llm_time = time.time() - llm_start

    # NOTE: A "Was that clear?" follow-up used to fire when the response
    # exceeded 60 words. The 60-word cutoff was hand-picked with no UX
    # research; the whole confirmation pattern needs design validation
    # before shipping, so the entire branch was removed.

    # If LLM failed completely, return error immediately

    if llm_error and not full_response:
        total_time = time.time() - total_start
        log.error(f"[{session_id[:8]}] ❌ Pipeline failed: No valid LLM response")
        log.info(f"[{session_id[:8]}] 📊 ERROR METRICS:")
        log.info(f"[{session_id[:8]}]    🎙️  STT: {stt_time:.2f}s | text='{text}'")
        log.info(f"[{session_id[:8]}]    🧠 LLM: ❌ FAILED ({llm_error})")
        log.info(f"[{session_id[:8]}]    ⏱️  TOTAL: {total_time:.2f}s")
        log.info(f"[{session_id[:8]}]    📈 KPI: ❌ FAILED (LLM unavailable)")
        return {
            "error": f"Unable to generate response: {llm_error}"
            ,"transcription": {"text": text, "language": lang, "confidence": round(lang_prob, 2)}
            ,"confusion": {
                "detected": bool(is_confused),
                "reason": confusion_reason,
                "hash": q_hash,
                "count": confusion_count,
            }
        }

    # Mise à jour mémoire (only if we have a valid response)
    if full_response:
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": full_response})
        if len(history) > Config.MAX_HISTORY_TURNS * 2:
            history[:] = history[2:]

    total_time = time.time() - total_start
    kpi_ok = total_time <= Config.MAX_RESPONSE_TIME

    # ── 6. Logging ────────────────────────────────────────────────────
    if full_response:
        csv_logger.log_turn(
            audio_duration_sec=audio_duration,
            stt_time=stt_time,
            llm_time=llm_time,
            tts_time=0,
            total_time=total_time,
            language=lang,
            model_used=Config.WHISPER_MODEL_SIZE,
            tts_engine_used="edge",
            tts_model_used="streaming",
            session_id=session_id,
            transcription=text,
        )

        log.info(
            f"[{session_id[:8]}] ✅ STREAMING | STT={stt_time:.2f}s LLM={llm_time:.2f}s "
            f"TOTAL={total_time:.2f}s {'✅' if kpi_ok else '⚠️'}"
        )
        
        # ✅ Detailed answer metrics with emojis
        log.info(f"[{session_id[:8]}] 📊 ANSWER METRICS:")
        log.info(f"[{session_id[:8]}]    🎙️  STT: {stt_time:.2f}s ({text[:50]}...)")
        log.info(f"[{session_id[:8]}]    🧠 LLM: {llm_time:.2f}s | chunks={len(chunks_with_scores)} | RAG={'✅' if len(chunks_with_scores) > 0 else '❌'}")
        log.info(f"[{session_id[:8]}]    ⏱️  TOTAL: {total_time:.2f}s | RTF={total_time/audio_duration:.2f}x")
        log.info(f"[{session_id[:8]}]    📈 KPI: {'✅ PASS' if kpi_ok else '⚠️  SLOW'} (limit={Config.MAX_RESPONSE_TIME}s)")
        log.info(f"[{session_id[:8]}]    🌍 Language: {lang.upper()} ({lang_prob:.0%})")

    return {
        "transcription": {"text": text, "language": lang, "confidence": round(lang_prob, 2)},
        "answer": full_response,
        "subject": subject,
        "rag_chunks": len(chunks_with_scores),
        "rag_chunks_details": [
            {
                "text": (doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else "")) if doc is not None else "",
                "score": round(score, 3),
                "source": source,
            }
            for (doc, score, source) in (chunks_with_scores or [])
        ],
        "tts_engine": tts_engine,
        "tts_voice": tts_voice,
        "confusion": {
            "detected": bool(is_confused),
            "reason": confusion_reason,
            "hash": q_hash,
            "count": confusion_count,
        },
        "question_text": text,
        "performance": {
            "stt_time": round(stt_time, 2),
            "llm_time": round(llm_time, 2),
            "tts_time": 0,
            "total_time": round(total_time, 2),
            "rtf": round(total_time / audio_duration, 2) if audio_duration > 0 else 0,
            "kpi_ok": kpi_ok,
            "kpi_status": "✅ PASS" if kpi_ok else "⚠️  SLOW",
            "rag_status": "✅" if len(chunks_with_scores) > 0 else "❌",
            "language": lang.upper(),
            "lang_confidence": round(lang_prob, 2),
        },
    }


async def run_pipeline(
    audio_data: np.ndarray,
    session_id: str,
    history: list,
    force_language: str = None,
    course_id: str | None = None,
    slide_context: str = None,
    # Injected dependencies
    transcriber=None,
    rag=None,
    voice=None,
    csv_logger=None,
    stt_logger=None,
    agent=None,
    brain=None,
) -> dict:
    """Pipeline complet : audio numpy → réponse JSON."""
    total_start = time.time()
    utt_id = str(uuid.uuid4())[:8]

    # ── 1. STT ────────────────────────────────────────────────────────
    text, stt_time, lang, lang_prob, audio_duration = await asyncio.to_thread(
        transcriber.transcribe,
        audio_data,
        force_language,
        slide_context,
    )
    if not text or len(text.strip()) <= 2:
        return {"no_speech": True, "message": "Aucune voix détectée"}

    stt_logger.log(
        session_id=session_id,
        utt_id=utt_id,
        audio_duration_sec=audio_duration,
        language_detected=lang,
        language_prob=lang_prob,
        stt_time=stt_time,
        transcription_text=text,
    )

    # ── 2. RAG ────────────────────────────────────────────────────────
    subject = detect_subject(text)
    chunks_with_scores = await asyncio.to_thread(
        rag.retrieve_chunks, text, k=Config.RAG_NUM_RESULTS, course_id=course_id
    )

    # ── 3. Détection confusion ────────────────────────────────────────
    # (skipped in this impl)

    # ── 4. LLM ────────────────────────────────────────────────────────
    llm_start = time.time()
    # Direct LLM call (agentic pipeline removed — to be redesigned)
    ai_response, _ = brain.ask(text, reply_language=lang)
    llm_time = time.time() - llm_start

    # Mise à jour mémoire
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": ai_response})
    if len(history) > Config.MAX_HISTORY_TURNS * 2:
        history[:] = history[2:]

    # ── 5. TTS ────────────────────────────────────────────────────────
    audio_bytes, tts_time, tts_engine, tts_voice, mime = (
        await voice.generate_audio_async(ai_response, language_code=lang)
    )

    total_time = time.time() - total_start
    kpi_ok = total_time <= Config.MAX_RESPONSE_TIME

    # ── 6. Logging ────────────────────────────────────────────────────
    csv_logger.log_turn(
        audio_duration_sec=audio_duration,
        stt_time=stt_time,
        llm_time=llm_time,
        tts_time=tts_time,
        total_time=total_time,
        language=lang,
        model_used=Config.WHISPER_MODEL_SIZE,
        tts_engine_used=tts_engine,
        tts_model_used=tts_voice,
        session_id=session_id,
        transcription=text,
    )

    log.info(
        f"[{session_id[:8]}] ✅ STT={stt_time:.2f}s LLM={llm_time:.2f}s "
        f"TTS={tts_time:.2f}s TOTAL={total_time:.2f}s {'✅' if kpi_ok else '⚠️'}"
    )

    return {
        "transcription": {"text": text, "language": lang, "confidence": round(lang_prob, 2)},
        "answer": ai_response,
        "audio_bytes": audio_bytes,
        "audio_b64": base64.b64encode(audio_bytes).decode() if audio_bytes else None,
        "mime": mime,
        "tts_engine": tts_engine,
        "tts_voice": tts_voice,
        "subject": subject,
        "rag_chunks": len(chunks_with_scores),
        "performance": {
            "stt_time": round(stt_time, 2),
            "llm_time": round(llm_time, 2),
            "tts_time": round(tts_time, 2),
            "total_time": round(total_time, 2),
            "kpi_ok": kpi_ok,
        },
    }
