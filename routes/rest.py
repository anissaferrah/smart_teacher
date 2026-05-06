"""REST core endpoints — session lifecycle, /ask Q&A, /ingest."""

import asyncio
import base64
import logging
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

import deps
from core.config import Config
from handlers.session_manager import (
    detect_lang_text,
    detect_subject,
    get_or_create_http_session,
    set_session_token,
    replace_http_history,
    clear_http_history,
)

router = APIRouter()
log = logging.getLogger("SmartTeacher.routes.rest")


# ── Session lifecycle ─────────────────────────────────────────────────

@router.post("/session")
async def create_session():
    """Create a new session with authentication token.

    Returns: {"session_id": str, "token": str}.
    The client must include this token in the WS start_session message.
    """
    session_id = str(uuid.uuid4())
    token = secrets.token_urlsafe(32)  # 256-bit secure random token
    await set_session_token(session_id, token)

    log.info(f"✅ Session created: {session_id[:8]} with auth token")
    return {"session_id": session_id, "token": token}


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Stats Redis dialogue pour une session WS active."""
    dialogue = deps.get_dialogue()
    ctx = await dialogue.get_session(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session introuvable")
    return await dialogue.get_stats(session_id)


@router.post("/session/clear")
async def clear_session(request: Request):
    sid = request.headers.get("X-Session-ID", "")
    if sid:
        # When the client passes X-Course-ID, only that course's history
        # is dropped. Without it, all per-course buckets for this session
        # are cleared (the user explicitly asked to reset).
        course_id = request.headers.get("X-Course-ID")
        await clear_http_history(sid, course_id)
        await deps.get_dialogue().end_session(sid)
    return {"status": "cleared", "session_id": sid}


# ── /ask — REST Q&A (legacy non-streaming) ────────────────────────────

# ``/ask`` is the legacy non-streaming Q&A path (kept for clients that
# don't open a WebSocket). Unlike the WS path, it does NOT go through the
# agentic responder graph — it calls Brain.ask directly. To preserve the
# course-bound guarantee, we retrieve chunks from the course's RAG index
# and pass them as ``course_context`` so the system prompt's strict
# "answer only from the course material" rule has actual material to
# bind to. Without this, /ask would silently degrade to a generic chatbot.
_ASK_RETRIEVAL_K = 5            # top-K chunks fed to the LLM as context
_ASK_CHUNK_CHAR_CAP = 800        # per-chunk char cap in the prompt


@router.post("/ask")
async def ask_question(
    request: Request,
    question: str = Form(...),
    course_id: str | None = Form(None),
):
    if not course_id:
        course_id = request.headers.get("X-Course-ID")
    session_id, history = await get_or_create_http_session(request, course_id=course_id)

    brain = deps.get_brain()
    voice = deps.get_voice()

    lang = detect_lang_text(question)
    subj = detect_subject(question)

    # Retrieve course-scoped chunks so the LLM has actual course material
    # to answer from. When no course is bound or retrieval fails, we still
    # call the LLM, but the strict course-bound system rule will then make
    # it answer "this is not covered" rather than fall back to general
    # knowledge — which is the desired behaviour.
    course_context = ""
    if course_id:
        try:
            rag = deps.get_rag()
            hits = rag.retrieve_chunks(
                query=question,
                k=_ASK_RETRIEVAL_K,
                course_id=course_id,
            ) or []
            chunks_text: list[str] = []
            for entry in hits:
                doc = entry[0] if isinstance(entry, tuple) else entry
                content = (getattr(doc, "page_content", "") or "")[:_ASK_CHUNK_CHAR_CAP]
                if content:
                    chunks_text.append(content)
            if chunks_text:
                course_context = "\n---\n".join(chunks_text)
        except Exception as exc:                                            # noqa: BLE001
            log.debug(f"/ask retrieval skipped: {exc}")

    llm_start = time.time()
    ai_response, _ = brain.ask(
        question,
        reply_language=lang,
        course_context=course_context,
        session_id=session_id,
    )
    llm_time = time.time() - llm_start

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ai_response})
    await replace_http_history(session_id, history, course_id=course_id)

    audio_bytes, tts_time, tts_engine, tts_voice, mime = \
        await voice.generate_audio_async(ai_response, language_code=lang)
    total_time = llm_time + tts_time

    return {
        "session_id": session_id,
        "question":   question,
        "answer":     ai_response,
        "audio":      base64.b64encode(audio_bytes).decode() if audio_bytes else None,
        "subject":    subj,
        "grounded":   bool(course_context),
        "performance": {
            "llm_time":   round(llm_time,   2),
            "tts_time":   round(tts_time,   2),
            "total_time": round(total_time, 2),
        },
    }


# ── /ingest + background helper ───────────────────────────────────────

async def _run_ingestion_background(
    file_paths: list[str],
    incremental: bool = False,
    domain: str = "general",
    course: str = "uploaded",
    course_id: str | None = None,
) -> None:
    """Unified ingestion — IntelligentIngester → MultiModalRAG.

    Uses IntelligentIngester as the single source of truth for extraction
    (text, slide PNGs, OCR images, tables, captions). The pre-extracted
    unstructured Elements are then indexed by the RAG (no re-parsing).

    Side effects produced by this path:
      - media/courses/{course_id}/slides/*.png       (rendered slides)
      - media/courses/{course_id}/images/*.png       (extracted images)
      - RAG index updated (BM25 + Qdrant)
      - IngestionManager status reflects per-file progress
    """
    from pedagogy.intelligent_ingester import IntelligentIngester

    rag = deps.get_rag()
    ingestion_manager = deps.get_ingestion_manager()
    try:
        await ingestion_manager.start_ingestion(len(file_paths))

        # ── Phase 1 : extraction unifiée via IntelligentIngester ──────────
        ingester = IntelligentIngester(ocr_languages="fra+eng")
        all_elements: list[Any] = []
        ingestion_summary = {
            "files":           0,
            "pages":           0,
            "slides_pngs":     0,
            "images_extracted": 0,
            "tables":          0,
            "captions":        0,
        }

        for fp in file_paths:
            try:
                result = await ingester.ingest_file(
                    file_path=fp,
                    media_root="media/courses",
                    course_id=course_id or "",
                )
                all_elements.extend(result.elements or [])
                ingestion_summary["files"]            += 1
                ingestion_summary["pages"]            += int(result.total_pages or 0)
                ingestion_summary["slides_pngs"]      += len(result.slide_pngs or [])
                ingestion_summary["images_extracted"] += int(result.total_images or 0)
                ingestion_summary["tables"]           += int(result.total_tables or 0)
                ingestion_summary["captions"]         += int(result.total_captions or 0)
                log.info(
                    f"🧠 IntelligentIngester [{Path(fp).name}]: "
                    f"{result.total_pages}p, {len(result.slide_pngs)} slides, "
                    f"{result.total_images}img, {result.total_tables}tbl, "
                    f"{result.total_captions}cap ({result.extraction_time_s}s)"
                )
            except Exception as exc:                                    # noqa: BLE001
                log.warning(f"IntelligentIngester failed on {fp}: {exc}")

        if not all_elements:
            await ingestion_manager.fail_ingestion(
                "IntelligentIngester produced no elements from the uploaded files"
            )
            log.error("❌ Aucun élément extrait — ingestion abandonnée")
            return

        # ── Phase 2 : indexation RAG (sans re-parser) ─────────────────────
        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(
            None,
            lambda: rag.index_from_elements(
                elements=all_elements,
                file_paths=file_paths,
                domain=domain,
                course=course,
                course_id=course_id,
                incremental=incremental,
            ),
        )

        if ok:
            stats = rag.get_stats()
            total_chunks = stats.get("total_docs", 0)
            await ingestion_manager.complete_ingestion(total_chunks)
            log.info(
                f"✅ Ingestion unifiée complétée : "
                f"{ingestion_summary['files']} fichiers, "
                f"{ingestion_summary['pages']} pages, "
                f"{ingestion_summary['slides_pngs']} slides PNG, "
                f"{ingestion_summary['tables']} tables, "
                f"{ingestion_summary['captions']} captions, "
                f"{total_chunks} chunks RAG"
            )
        else:
            await ingestion_manager.fail_ingestion("index_from_elements returned False")
            log.error("❌ Indexation RAG échouée")

    except Exception as exc:                                            # noqa: BLE001
        await ingestion_manager.fail_ingestion(str(exc))
        log.error(f"❌ Erreur ingestion unifiée : {exc}")


@router.post("/ingest")
async def ingest_files(
    files:       list[UploadFile] = File(...),
    incremental: bool             = Form(True),
    course_id:   str | None       = Form(None),
):
    """Upload + indexe des fichiers de cours dans la base vectorielle (async)."""
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni")

    # All persistent uploads live under media/ (legacy "courses/" is read-only).
    upload_dir = Path("media/courses/uploaded")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for f in files:
        dest = upload_dir / Path(f.filename).name
        dest.write_bytes(await f.read())
        saved_paths.append(str(dest.resolve()))

    log.info(f"📤 Ingestion lancée ({len(saved_paths)} fichier(s)) — embeddings locaux BAAI/bge-m3")
    if course_id:
        log.info(f"   📚 course_id={course_id}")

    asyncio.create_task(
        _run_ingestion_background(
            saved_paths,
            incremental=incremental,
            domain="general",
            course="uploaded",
            course_id=course_id,
        )
    )

    return {
        "status": "ingestion_started",
        "files": [f.filename for f in files],
        "message": "Ingestion lancée en arrière-plan. Consultez /ingestion/status pour suivre.",
    }
