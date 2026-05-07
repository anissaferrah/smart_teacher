"""REST API endpoints for Smart Teacher."""

import logging
from typing import Optional
from fastapi import Request, UploadFile, File, Form, HTTPException
from handlers.session_manager import detect_subject, _get_history_raw

log = logging.getLogger("SmartTeacher.RestRoutes")


# Note: These functions will be called from main.py with app.post/app.get decorators
# This module groups the endpoint logic for clarity


async def handle_session_creation():
    """POST /session - Create session and generate auth token."""
    # Implemented in main.py - generates SESSION_TOKENS entry


async def handle_ask(
    request: Request,
    question: str,
    language: Optional[str] = None,
    session_id: Optional[str] = None,
    # Injected
    dialogue=None,
    csv_logger=None,
):
    """POST /ask - Answer text question immediately."""
    if not question or len(question.strip()) < 2:
        raise HTTPException(status_code=400, detail="Question too short")

    log.info(f"[{session_id}] ❓ Question: {question[:50]}...")

    subject = detect_subject(question)

    # Will be orchestrated in main.py with brain.ask()
    return {
        "status": "processing",
        "question": question,
        "subject": subject,
        "language": language,
    }


async def handle_ingest(
    request: Request,
    course_name: str = Form(...),
    files: list[UploadFile] = File(...),
    # Injected
    rag=None,
):
    """POST /ingest - Ingest course materials."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    log.info(f"📂 Ingesting {len(files)} files into course: {course_name}")
    # Orchestration in main.py
    return {"status": "ingesting", "course": course_name, "file_count": len(files)}


async def handle_health_check():
    """GET /health - Health check."""
    return {"status": "ok", "service": "SmartTeacher"}


async def handle_rag_stats(course_id: Optional[str] = None):
    """GET /rag/stats - RAG statistics."""
    return {
        "status": "ok",
        "course_id": course_id,
        "chunks_total": 0,  # Will be calculated in main.py
    }


async def handle_session_get(session_id: str):
    """GET /session/{session_id} - Get session info.

    Note: this handler is currently NOT mounted (the /session/{session_id}
    GET route lives in main.py and uses the Redis-backed dialogue layer).
    Kept here for compatibility if a caller imports it directly.
    """
    # course_id not threaded here (legacy compat shim) — falls back to
    # the _NO_COURSE bucket. Real callers go through routes/rest.py.
    history = await _get_history_raw(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "history_turns": len(history) // 2,
    }


async def handle_session_profile(session_id: str):
    """GET /session/{session_id}/profile - Return student profile.

    Delegates to the modules.student_profile.ProfileManager to fetch the
    Redis-backed profile. Returns a JSON-friendly dict.
    """
    try:
        from pedagogy.personalization.profile import get_or_create_profile

        profile = await get_or_create_profile(session_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return {"status": "ok", "profile": profile}
    except HTTPException:
        raise
    except Exception as exc:
        log.debug("Profile fetch failed for %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch profile")


async def handle_session_profile_update(session_id: str, payload: dict):
    """POST /session/{session_id}/profile - Update student profile (partial).

    Accepts a JSON payload with shallow fields to update (e.g., `level`, `preferences`).
    Delegates to services.personalization.profile_manager.update_profile.
    """
    try:
        from pedagogy.personalization.profile import update_profile, get_or_create_profile

        # ensure profile exists
        await get_or_create_profile(session_id)
        updated = await update_profile(session_id, payload)
        return {"status": "ok", "profile": updated}
    except Exception as exc:
        log.debug("Profile update failed for %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Failed to update profile")


async def handle_session_tts_params(session_id: str, confusion_score: float = 0.0):
    """GET /session/{session_id}/tts_params - Compute TTS params for a session.

    Optional query param `confusion_score` lets the caller simulate runtime signals.
    """
    try:
        from pedagogy.personalization.tts_adapter import compute_tts_params

        params = await compute_tts_params(session_id, base_rate=1.0, confusion_score=float(confusion_score))
        return {"status": "ok", "tts_params": params}
    except Exception as exc:
        log.debug("TTS params computation failed for %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Failed to compute tts params")


async def handle_search_transcripts(
    query: str,
    session_id: Optional[str] = None,
    limit: int = 10,
):
    """GET /search/transcripts - Search transcripts."""
    if not query:
        raise HTTPException(status_code=400, detail="Query required")

    log.info(f"🔍 Searching transcripts: {query[:50]}...")
    return {"query": query, "results": [], "total": 0}
