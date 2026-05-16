"""Shared entry point for the Q&A graph — used by both text and audio paths.

Before this module existed, the text path inlined the qa_state build +
graph streaming in handlers/ws.py:777-836, and the audio path bypassed
the graph entirely (handlers/audio_pipeline.py:193-200, calling brain.ask
directly). That meant audio questions skipped: intent classification,
the query rewriter, the smart-gate reviewer, mastery tracking, the
honest-fallback enforcement, and the bandit-driven personalization
strategy. Two modalities, two different brains — a real consistency bug.

Routing both modalities through this helper unifies behavior: voice
questions and typed questions now produce the same quality of answer,
the same telemetry, and the same student-progress updates.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger("agentic.qa.runner")


async def run_qa_graph(
    *,
    text: str,
    session_id: str,
    course_id: str,
    student_id: str | None,
    language: str,
    chapter_idx: int | None,
    chapter_title: str,
    section_idx: int | None,
    section_title: str,
    last_slide_content: str,
    history: list[dict],
    student_level: str,
    qa_graph,
    brain,
    engagement: Optional[dict] = None,
    on_intent_classified: Optional[Callable[[Any], Awaitable[None]]] = None,
) -> dict[str, Any]:
    """Run the Q&A graph end-to-end and return the cleaned answer.

    Loads persistent history and the student-knowledge snapshot
    internally so callers don't have to. Mirrors what handlers/ws.py
    used to do inline — factored out so the audio path can use the
    exact same pipeline.

    Returns dict with:
        answer       : cleaned spoken text (post-_clean_for_speech)
        answer_raw   : LLM output before cleaning
        intent       : VoiceIntent from IntentAgent
        citations    : list of citation dicts from the responder
        timings      : per-node latency dict
        qa_final     : final graph state (for downstream telemetry)
        confidence   : float in [0, 1] from citation grounding
    """
    # 1. Load persistent chat history (best-effort)
    persistent_history: list[dict] = []
    try:
        from pedagogy.student_history import load_chat_history
        persistent_history = await load_chat_history(
            student_id=student_id or "",
            course_id=course_id or "",
            limit=20,
        )
    except Exception as exc:                                              # noqa: BLE001
        log.debug(f"[{session_id[:8]}] persistent history load skipped: {exc}")

    # 2. Build student-knowledge snapshot (best-effort)
    student_snapshot = None
    try:
        from pedagogy.student_knowledge import build_snapshot
        student_snapshot = await build_snapshot(
            student_id=student_id or None,
            course_id=course_id or None,
        )
    except Exception as exc:                                              # noqa: BLE001
        log.debug(f"[{session_id[:8]}] knowledge snapshot skipped: {exc}")

    # 3. Merge persistent + in-memory history, dedupe by fingerprint
    merged_history: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for h in (persistent_history + list(history or [])):
        if not isinstance(h, dict):
            continue
        fp = (h.get("role", ""), str(h.get("content", ""))[-80:])
        if fp in seen:
            continue
        seen.add(fp)
        merged_history.append({
            "role":    h.get("role", "user"),
            "content": h.get("content", ""),
        })

    # 4. Build qa_state (same shape as ws.py:777-797 used to build inline)
    qa_state = {
        "session_id":         session_id,
        "course_id":          course_id or "",
        "student_id":         student_id or None,
        "language":           (language or "fr")[:2],
        "chapter_idx":        chapter_idx or 0,
        "chapter_title":      chapter_title or "",
        "section_idx":        section_idx or 0,
        "section_title":      section_title or "",
        "last_slide_content": last_slide_content or "",
        "history":            merged_history,
        "student_snapshot":   student_snapshot,
        "engagement":         engagement,
        "event_type":         "student_speech",
        "event_payload":      {"text": text},
        "domain":             None,
        "student_level":      student_level or "lycée",
    }

    # 5. Stream the graph. Fire on_intent_classified as soon as the
    # IntentAgent node finishes — text path uses this to push intent
    # to the frontend, audio path uses it to emit a confusion preamble.
    qa_final: dict = dict(qa_state)
    async for update_chunk in qa_graph.astream(qa_state, stream_mode="updates"):
        if not isinstance(update_chunk, dict):
            continue
        for node_name, updates in update_chunk.items():
            if isinstance(updates, dict):
                qa_final.update(updates)
            if node_name == "intent" and isinstance(updates, dict) and on_intent_classified:
                intent_obj = updates.get("intent")
                if intent_obj is not None:
                    try:
                        await on_intent_classified(intent_obj)
                    except Exception as exc:                              # noqa: BLE001
                        log.debug(f"[{session_id[:8]}] on_intent_classified raised: {exc}")

    # 6. Extract + clean answer
    answer_raw = (qa_final.get("answer") or "").strip()
    answer_clean = brain._clean_for_speech(answer_raw) if answer_raw else ""

    # 7. Confidence from citation grounding (mirrors responder.py:1370-1377)
    citations = qa_final.get("citations") or []
    if citations:
        confidence = round(
            sum(float(c.get("score", 0.0)) for c in citations) / len(citations), 3
        )
    else:
        confidence = 0.0

    return {
        "answer":     answer_clean,
        "answer_raw": answer_raw,
        "intent":     qa_final.get("intent"),
        "citations":  citations,
        "timings":    qa_final.get("timings") or {},
        "qa_final":   qa_final,
        "confidence": confidence,
    }
