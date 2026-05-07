import asyncio
import time
import base64
import logging
import inspect
from typing import Callable

from core.config import Config

from deps import (
    get_brain,
    get_voice,
    get_rag,
    get_transcript_searcher,
    get_analytics_engine,
)

log = logging.getLogger("SmartTeacher.services.text_turns")


async def persist_learning_turn(
    session_id: str,
    learning_session_db_id=None,
    question_text: str = "",
    answer_text: str = "",
    language: str = "fr",
    subject: str = "",
    extra_payload: dict | None = None,
    db=None,
    user_id: str | None = None,
    metadata: dict | None = None,
):
    """Lightweight persistence stub that delegates to DB CRUD if available.

    Keep this function minimal and safe for background tasks.
    """
    try:
        transcript_searcher = get_transcript_searcher()
        analytics = get_analytics_engine()
        # Index minimal interaction for search/analytics
        async def _call_maybe_async(fn, *a, **kw):
            try:
                if inspect.iscoroutinefunction(fn):
                    return await fn(*a, **kw)
                # run blocking sync call in thread
                return await asyncio.to_thread(fn, *a, **kw)
            except Exception:
                raise

        try:
            if hasattr(transcript_searcher, "index_interaction"):
                await _call_maybe_async(
                    transcript_searcher.index_interaction,
                    session_id,
                    question_text,
                    answer_text,
                    language,
                    "",
                    subject,
                )
        except Exception:
            log.debug("transcript indexing skipped")

        try:
            if hasattr(analytics, "record_interaction"):
                await _call_maybe_async(
                    analytics.record_interaction,
                    session_id=session_id,
                    question=question_text,
                    answer=answer_text,
                    stt_time=0.0,
                    llm_time=0.0,
                    tts_time=0.0,
                    language=language,
                    subject=subject,
                )
        except Exception:
            log.debug("analytics record skipped")
    except Exception as exc:
        log.debug(f"persist_learning_turn skipped: {exc}")


async def process_text_question_turn(
    *,
    session_id: str,
    content: str,
    turn_id: int,
    send: Callable[[dict], asyncio.Future],
    course_id: str | None = None,
):
    brain = get_brain()
    voice = get_voice()
    rag = get_rag()

    lang = "fr"
    subj = ""

    try:
        # language detection if available
        try:
            from handlers.session_manager import detect_lang_text, detect_subject

            lang = detect_lang_text(content)
            subj = detect_subject(content)
        except Exception:
            pass

        await send({"type": "state_change", "state": "processing", "turn_id": turn_id})

        # RAG retrieval (off-loaded to thread pool — embedding_cache + Qdrant are sync).
        # Same pipeline as the agentic Q&A retriever : RAG top-K + KG-augmented
        # prereqs/examples/illustrated. The HTTP /ask path was previously running
        # bare RAG, which hid the KG signal from the LLM and made the eval
        # framework underestimate the live system.
        chunks = []
        try:
            chunks = await asyncio.to_thread(
                rag.retrieve_chunks, content, k=Config.RAG_NUM_RESULTS, course_id=course_id
            )
        except Exception:
            chunks = []

        # KG augmentation : turn the (Document, score) tuples into the dict
        # format ``kg_augment_chunks`` expects, augment, then re-merge.
        rag_chunks_dict: list[dict] = []
        for item in chunks or []:
            doc = item[0] if isinstance(item, tuple) else item
            score = item[1] if (isinstance(item, tuple) and len(item) > 1) else 0.5
            meta = getattr(doc, "metadata", {}) or {}
            rag_chunks_dict.append({
                "content":   getattr(doc, "page_content", "") or "",
                "score":     float(score) if score is not None else 0.5,
                "idea_id":   meta.get("idea_id"),
                "idea_label": meta.get("idea_label", ""),
                "chapter":   meta.get("chapter_idx"),
                "section_title": meta.get("section_title", ""),
            })

        kg_augmented: list[dict] = []
        if rag_chunks_dict and getattr(Config, "RAG_USE_GRAPH_EXPANSION", True):
            try:
                from agentic.qa.retriever import kg_augment_chunks
                kg_augmented = await asyncio.to_thread(
                    kg_augment_chunks, rag_chunks_dict, rag,
                )
            except Exception as exc:
                log.debug(f"kg augmentation skipped: {exc}")
                kg_augmented = []

        # ✨ Build course_context : direct RAG hits (top-5) + KG-augmented
        # neighbors (prereqs/examples). Cap at 4000 chars to stay safe of
        # the LLM context window.
        course_context = ""
        try:
            ranked = sorted(rag_chunks_dict, key=lambda c: -c.get("score", 0.0))
            parts: list[str] = []
            for c in ranked[:5]:
                if c["content"]:
                    parts.append(c["content"])
            for c in kg_augmented:
                rel = c.get("_kg_relation") or "kg"
                # Tag KG-augmented chunks so the LLM treats them as
                # supporting context, not as the primary answer.
                parts.append(f"[KG: {rel}] {c.get('content', '')}")
            course_context = "\n\n".join(parts)[:4000]
        except Exception:
            course_context = ""

        llm_start = time.time()
        # Call LLM (support sync or async implementations) — with RAG context
        try:
            if inspect.iscoroutinefunction(getattr(brain, "ask", None)):
                answer, _ = await brain.ask(content, course_context=course_context, reply_language=lang)
            else:
                answer, _ = await asyncio.to_thread(
                    brain.ask, content, course_context=course_context, reply_language=lang,
                )
        except Exception as e:
            log.error(f"brain.ask failed: {e}")
            answer = "Désolé, erreur technique."

        llm_time = time.time() - llm_start

        # Send answer text
        try:
            await send({"type": "answer_text", "text": answer, "turn_id": turn_id})
        except Exception:
            pass

        # Generate TTS asynchronously (best-effort)
        try:
            audio_bytes, tts_time, tts_engine, tts_voice, mime = await voice.generate_audio_async(
                answer, language_code=lang
            )
            if audio_bytes:
                await send({
                    "type": "audio_chunk",
                    "data": base64.b64encode(audio_bytes).decode(),
                    "mime": mime,
                    "final": True,
                    "turn_id": turn_id,
                })
        except Exception:
            log.debug("tts generation skipped")

        # Persist minimal learning info in background
        try:
            asyncio.create_task(
                persist_learning_turn(
                    session_id=session_id,
                    learning_session_db_id=None,
                    question_text=content,
                    answer_text=answer,
                    language=lang,
                    subject=subj,
                )
            )
        except Exception:
            pass

        await send({"type": "performance", "turn_id": turn_id, "llm_time": round(llm_time, 2)})

    except Exception as exc:
        log.exception(f"process_text_question_turn failed: {exc}")
        try:
            await send({"type": "error", "message": str(exc), "turn_id": turn_id})
        except Exception:
            pass
