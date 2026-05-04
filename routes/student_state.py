"""GET /student/me/state — aggregated "what does the system know about
the student right now" endpoint.

# Why this exists

The Smart Teacher accumulates a lot of state per student — current
position, mastery posteriors, recent chat, engagement signals,
confusion history. Each lives in its own store (Redis SessionContext,
Postgres mastery, Redis chat history, ...). Without a single read API,
debugging "why did the tutor say X?" means tailing logs and querying
4 different backends. This endpoint joins everything into one JSON
payload :

    GET /student/me/state?course_id=<uuid>           # auth-scoped
    GET /student/{student_id}/state?course_id=<uuid>  # admin-scoped

# Security

  - ``/student/me/state`` : only sees the JWT-bearer's own state
  - ``/student/{id}/state`` : requires admin role (TODO once role auth
    is in place — for now we accept any authenticated user, same
    pattern as ``/student/{id}/profile/reset``).
  - All fields that could expose other students (raw chat content
    from a different course, internal mastery deltas) are scoped to
    ``course_id`` when provided.

# Performance

The aggregation runs ~5 lookups in parallel via ``asyncio.gather`` :
KG concepts, mastery scores, chat history, profile, snapshot. Each is
already cached or sub-ms (Redis) so the endpoint should land < 50ms
P50 even on cold cache.

# What's NOT in scope

  - Streaming updates : this is a snapshot REST endpoint, not a
    WebSocket. The frontend dashboard polls on demand (e.g., teacher
    opening a student's progress page).
  - Write operations : reset / annotate live in dedicated endpoints
    (see ``/student/{id}/profile/reset``).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from handlers.auth import get_current_user

router = APIRouter()
log = logging.getLogger("SmartTeacher.routes.student_state")


# ── Helpers ───────────────────────────────────────────────────────────

async def _load_session_context(student_id: str) -> Optional[dict]:
    """Fetch the most recent SessionContext for ``student_id``.

    SessionContext is keyed by session_id, not student_id, in Redis —
    so we scan ``session:*`` and return the entry whose ``student_id``
    matches. For sessions, that's at most a few hundred keys ; SCAN
    is O(n) but bounded.

    Best-effort : returns None on Redis miss / parse error rather
    than 500-ing the endpoint.
    """
    try:
        from pedagogy.dialogue import get_redis
        import json
        r = await get_redis()
        cursor = 0
        latest_ts = 0.0
        latest_payload: Optional[dict] = None
        while True:
            cursor, batch = await r.scan(cursor=cursor, match="session:*", count=200)
            for key in batch:
                try:
                    raw = await r.get(key)
                    if not raw:
                        continue
                    payload = json.loads(raw if isinstance(raw, str) else raw.decode())
                    if not isinstance(payload, dict):
                        continue
                    if str(payload.get("student_id") or "") != str(student_id):
                        continue
                    ts = float(payload.get("last_activity") or 0.0)
                    if ts > latest_ts:
                        latest_ts = ts
                        latest_payload = payload
                except Exception:                                          # noqa: BLE001
                    continue
            if cursor == 0:
                break
        return latest_payload
    except Exception as exc:                                              # noqa: BLE001
        log.debug("load_session_context failed: %s", exc)
        return None


async def _load_chat_preview(student_id: str, course_id: str, limit: int = 5) -> list[dict]:
    """Last ``limit`` chat turns for (student, course)."""
    try:
        from pedagogy.student_history import load_chat_history
        return await load_chat_history(student_id, course_id, limit=limit)
    except Exception as exc:                                              # noqa: BLE001
        log.debug("load_chat_preview failed: %s", exc)
        return []


async def _load_reviews_due(student_id: str, course_id: str) -> dict:
    """Count + preview of FSRS-due reviews for (student, course).

    The aggregate state endpoint surfaces this so a dashboard can show
    "you have 5 concepts to review now" without an extra round-trip.
    Returns ``{"count": int, "items": [...]}`` ; empty dict on miss.
    """
    try:
        from pedagogy.review_scheduler import ReviewScheduler
        items = await ReviewScheduler.list_due(student_id, course_id=course_id, limit=10)
        return {"count": len(items), "items": items}
    except Exception as exc:                                              # noqa: BLE001
        log.debug("load_reviews_due failed: %s", exc)
        return {"count": 0, "items": []}


async def _load_knowledge_snapshot(student_id: str, course_id: str) -> dict:
    """Convert StudentKnowledgeSnapshot to JSON-serializable dict."""
    try:
        from pedagogy.student_knowledge import build_snapshot
        snap = await build_snapshot(student_id, course_id)
        return {
            "mastery_by_concept": snap.mastery_by_concept,
            "strong_concepts":    snap.strong_concepts,
            "weak_concepts":      snap.weak_concepts,
            "never_seen_concepts": snap.never_seen_concepts,
            "recently_confused":  snap.recently_confused,
            "total_attempts":     snap.total_attempts,
            "is_cold_start":      snap.is_cold_start,
        }
    except Exception as exc:                                              # noqa: BLE001
        log.debug("load_knowledge_snapshot failed: %s", exc)
        return {}


async def _compute_engagement_for(ctx_payload: Optional[dict]) -> dict:
    """Run the engagement scorer against a SessionContext blob."""
    if not ctx_payload:
        return {"score": None, "label": "unknown", "reason": "no active session"}
    try:
        from pedagogy.engagement import EngagementSignals, compute_engagement
        # Pull the signals we have in SessionContext blobs
        last_activity = float(ctx_payload.get("last_activity") or 0.0)
        now = time.time()
        sig = EngagementSignals(
            seconds_since_last_interaction=(now - last_activity) if last_activity else None,
            questions_in_session=ctx_payload.get("questions_in_session"),
            confusions_in_session=int(ctx_payload.get("confusion_count") or 0),
            consecutive_passive_slides=ctx_payload.get("consecutive_passive_slides"),
            last_interrupt_latency_ms=ctx_payload.get("last_interrupt_latency_ms"),
            session_age_s=(now - float(ctx_payload.get("session_started_at") or last_activity))
                          if last_activity else None,
        )
        result = compute_engagement(sig)
        return {
            "score":  result.score,
            "label":  result.label,
            "reason": result.reason,
        }
    except Exception as exc:                                              # noqa: BLE001
        log.debug("compute_engagement_for failed: %s", exc)
        return {"score": None, "label": "unknown", "reason": str(exc)[:80]}


def _aggregate(
    student_id: str,
    course_id: Optional[str],
    ctx_payload: Optional[dict],
    chat_preview: list[dict],
    knowledge: dict,
    engagement: dict,
    reviews_due: Optional[dict] = None,
) -> dict:
    """Compose the final JSON response. Pure — no I/O."""
    position = {}
    if ctx_payload:
        position = {
            "course_id":     ctx_payload.get("course_id"),
            "chapter_index": ctx_payload.get("chapter_index"),
            "section_index": ctx_payload.get("section_index"),
            "char_position": ctx_payload.get("char_position"),
            "state":         ctx_payload.get("state"),
            "last_activity": ctx_payload.get("last_activity"),
            "is_paused":     bool((ctx_payload.get("paused_state") or {}).get("is_paused")),
        }
    return {
        "student_id":         student_id,
        "course_id":          course_id,
        "current_position":   position,
        "knowledge_snapshot": knowledge,
        "engagement":         engagement,
        "recent_chat_preview": chat_preview,
        "reviews_due":        reviews_due or {"count": 0, "items": []},
        "generated_at":       time.time(),
    }


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/student/me/state")
async def get_my_state(
    course_id: Optional[str] = Query(None, description="Filter to one course"),
    user: dict = Depends(get_current_user),
):
    """Aggregated state for the authenticated student.

    Returns 5 sections (current_position, knowledge_snapshot,
    engagement, recent_chat_preview, generated_at). Always 200 OK
    even with empty data — frontend can distinguish via
    ``is_cold_start`` and the position fields being null.
    """
    student_id = str(user.get("sub") or "")
    if not student_id:
        raise HTTPException(status_code=401, detail="missing student id in token")

    # Run the 4 lookups in parallel — independent stores
    ctx_payload, chat_preview, knowledge, reviews_due = await asyncio.gather(
        _load_session_context(student_id),
        _load_chat_preview(student_id, course_id or ""),
        _load_knowledge_snapshot(student_id, course_id or ""),
        _load_reviews_due(student_id, course_id or ""),
    )
    engagement = await _compute_engagement_for(ctx_payload)
    return _aggregate(
        student_id=student_id,
        course_id=course_id,
        ctx_payload=ctx_payload,
        chat_preview=chat_preview,
        knowledge=knowledge,
        engagement=engagement,
        reviews_due=reviews_due,
    )


@router.get("/student/{student_id}/state")
async def get_student_state(
    student_id: str,
    course_id: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    """Same payload as ``/student/me/state``, scoped to an arbitrary
    student_id. Currently any authenticated user can view any
    student's state — wire role-based admin auth here once the
    auth layer supports it.
    """
    if not student_id:
        raise HTTPException(status_code=400, detail="student_id required")

    ctx_payload, chat_preview, knowledge, reviews_due = await asyncio.gather(
        _load_session_context(student_id),
        _load_chat_preview(student_id, course_id or ""),
        _load_knowledge_snapshot(student_id, course_id or ""),
        _load_reviews_due(student_id, course_id or ""),
    )
    engagement = await _compute_engagement_for(ctx_payload)
    return _aggregate(
        student_id=student_id,
        course_id=course_id,
        ctx_payload=ctx_payload,
        chat_preview=chat_preview,
        knowledge=knowledge,
        engagement=engagement,
        reviews_due=reviews_due,
    )
