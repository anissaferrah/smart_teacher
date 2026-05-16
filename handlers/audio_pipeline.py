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
    qa_graph=None,                                  # NEW: routes audio through QA graph
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

    # ── 3. Run the Q&A Graph (same pipeline text questions use) ──────
    # Replaces the previous direct rag.retrieve_chunks + manual confusion
    # detection + direct brain.ask call. The graph now handles:
    #   - intent classification (incl. SIGHT-driven confusion detection)
    #   - Self-RAG retrieve-or-skip routing
    #   - query rewriting when needed
    #   - RAG retrieval + reranker + smart-gate review
    #   - mastery tracking, bandit selection, personalization
    # Audio and text now produce equivalent answers.
    subject = detect_subject(text)
    if on_state_change:
        await on_state_change("rag_search", {})

    llm_start = time.time()
    full_response: str | None = ""
    tts_engine = "edge"
    tts_voice = ""
    chunks_with_scores: list = []  # populated below from graph state (telemetry only)

    # Hook: emit confusion preamble TTS as soon as IntentAgent fires
    # SIGHT, so the student hears "Let me explain differently…" up front
    # instead of waiting for the full reformulated answer.
    _preamble_emitted = False

    async def _on_intent_classified(intent_obj):
        nonlocal _preamble_emitted, tts_engine, tts_voice
        intent_type = getattr(intent_obj, "type", "")
        if on_state_change:
            await on_state_change(
                "intent_classified",
                {
                    "intent_type": intent_type,
                    "confidence": float(getattr(intent_obj, "confidence", 0.0)),
                },
            )
        if intent_type == "confusion_signal" and not _preamble_emitted:
            _preamble_emitted = True
            preambles = {
                "fr": "Permettez-moi de réexpliquer autrement. ",
                "en": "Let me explain that differently. ",
            }
            preamble = preambles.get(lang[:2], preambles["fr"])
            if on_text_chunk:
                await on_text_chunk(preamble, preamble)
            try:
                pre_audio, _, pre_engine, pre_voice, pre_mime = await voice.generate_audio_async(
                    preamble, language_code=lang,
                )
                if pre_audio and on_audio_chunk:
                    await on_audio_chunk(pre_audio, pre_mime)
                    log.info(f"[{session_id[:8]}] 📤 Preamble audio streamed")
                tts_engine = pre_engine
                tts_voice = pre_voice
            except Exception as pre_exc:                                  # noqa: BLE001
                log.warning(f"[{session_id[:8]}] ⚠️ Preamble TTS failed: {pre_exc}")

    try:
        if qa_graph is None:
            raise RuntimeError("qa_graph not injected")
        from agentic.qa.runner import run_qa_graph
        qa_result = await run_qa_graph(
            text=text,
            session_id=session_id,
            course_id=course_id or "",
            student_id=(getattr(ctx, "student_id", None) if ctx else None),
            language=lang,
            chapter_idx=((ctx.chapter_index + 1)
                         if ctx and ctx.chapter_index is not None else None),
            chapter_title=(getattr(ctx, "chapter_title", "") if ctx else ""),
            section_idx=(getattr(ctx, "section_index", 0) if ctx else 0),
            section_title=(getattr(ctx, "section_title", "") if ctx else ""),
            last_slide_content=(ctx.last_slide_explained if ctx else ""),
            history=history,
            student_level=(getattr(ctx, "student_level", "lycée") if ctx else "lycée"),
            qa_graph=qa_graph,
            brain=brain,
            on_intent_classified=_on_intent_classified,
        )
        full_response = qa_result["answer"]
        chunks_with_scores = (qa_result.get("qa_final") or {}).get("retrieved_chunks") or []
        log.info(
            f"[{session_id[:8]}] 🧠 QA Graph (audio) done | "
            f"intent={getattr(qa_result.get('intent'), 'type', '?')} | "
            f"chars={len(full_response)} chunks={len(chunks_with_scores)}"
        )
    except Exception as graph_exc:                                        # noqa: BLE001
        log.warning(
            f"[{session_id[:8]}] ⚠️ QA Graph failed ({type(graph_exc).__name__}): "
            f"{str(graph_exc)[:120]} → Fallback brain.ask..."
        )
        llm_error = type(graph_exc).__name__
        try:
            full_response, _ = await asyncio.to_thread(
                brain.ask, text, reply_language=lang, session_id=session_id,
            )
            full_response = brain._clean_for_speech(full_response or "")
        except Exception as fallback_exc:                                 # noqa: BLE001
            log.error(f"[{session_id[:8]}] ❌ Fallback also failed: {fallback_exc}")
            llm_error = f"LLM unavailable ({type(fallback_exc).__name__})"
            full_response = None

    if on_state_change:
        await on_state_change("streaming_llm", {})

    # TTS — only if the LLM produced a usable response. The previous
    # outer LLM try/except wrapper here ran brain.ask again on TTS
    # failure, which was both redundant (the run_qa_graph block above
    # already handles its own fallback) and incorrect (TTS errors are
    # not LLM errors). Removed.
    if full_response:
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
        except Exception as tts_exc:                                      # noqa: BLE001
            log.error(f"[{session_id[:8]}] ❌ TTS error: {tts_exc}")

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

    # Confusion fields now derived from the graph's intent classification
    # (SIGHT runs inside IntentAgent, sets intent.type=="confusion_signal").
    # The pre-graph dialogue.detect_and_track_confusion call was removed
    # in the audio→graph migration; q_hash and confusion_count were only
    # used by that call's Redis bookkeeping and have no consumers downstream.
    _graph_intent = None
    try:
        _graph_intent = qa_result.get("intent") if "qa_result" in locals() else None
    except Exception:                                                     # noqa: BLE001
        _graph_intent = None
    _is_confused = bool(_graph_intent and getattr(_graph_intent, "type", "") == "confusion_signal")
    _confusion_reason = "sight_model" if _is_confused else ""

    # Chunks now arrive as dicts from the QA retriever (content, score,
    # source, idea_id, ...), not as (Document, score, source) tuples
    # like the old direct rag.retrieve_chunks call returned.
    _chunks_details = []
    for ch in (chunks_with_scores or []):
        if isinstance(ch, dict):
            content = str(ch.get("content", "") or "")
            _chunks_details.append({
                "text": content[:400] + ("..." if len(content) > 400 else ""),
                "score": round(float(ch.get("score", 0.0)), 3),
                "source": str(ch.get("source", "") or ""),
            })

    return {
        "transcription": {"text": text, "language": lang, "confidence": round(lang_prob, 2)},
        "answer": full_response,
        "subject": subject,
        "rag_chunks": len(chunks_with_scores),
        "rag_chunks_details": _chunks_details,
        "tts_engine": tts_engine,
        "tts_voice": tts_voice,
        "confusion": {
            "detected": _is_confused,
            "reason": _confusion_reason,
            "hash": "",
            "count": 0,
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
