"""Persistence helpers for ConfusionEvent rows.

The ``confusion_events`` table is the row-level archive of every
detector hit (SIGHT classifier, prosody, keyword, LLM heuristic).
Whereas ``student_mistakes.confusions`` is just a counter, this table
records *when, on what slide, with what trigger text, by which detector*
each confusion fired — the data needed for analyses like :

  - "concept X causes 80 % of confusions on this course"
  - "the prosody-based detector has a 30 % false-positive rate vs the SIGHT model"
  - "students confused on slide N tend to disengage by slide N+3"

The helper here is best-effort by design : a Postgres failure must NOT
break the user's request path. The caller logs `WARNING` and continues.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

log = logging.getLogger("pedagogy.confusion.persistence")


def _coerce_uuid(value) -> uuid.UUID | None:
    """Tolerant string→UUID. Returns None on bad input — caller decides."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


async def record_confusion_event(
    *,
    student_id: str | uuid.UUID,
    course_id: str | uuid.UUID | None,
    session_id: str | None = None,
    concept_id: str | uuid.UUID | None = None,
    slide_idx: int | None = None,
    trigger_text: str | None = None,
    source: str = "unknown",
    score: float = 0.0,
    resolved: bool = False,
    resolution_strategy: str | None = None,
) -> str | None:
    """Insert one row into ``confusion_events``. Best-effort, never raises.

    Returns the inserted row id (UUID string) on success, ``None`` on
    any failure (DB unreachable, FK violation, etc.). The caller should
    log at WARNING when the return is None and continue without
    blocking the user request.
    """
    sid = _coerce_uuid(student_id)
    if sid is None:
        log.debug("confusion event : invalid student_id %r — skip", student_id)
        return None

    try:
        from database.init_db import AsyncSessionLocal
        from database.models import ConfusionEvent
    except Exception as exc:                                                # noqa: BLE001
        log.warning("confusion event : DB import failed (%s)", exc)
        return None

    cid = _coerce_uuid(course_id)
    concept_uuid = _coerce_uuid(concept_id)

    # Truncate trigger_text to a safe length so a runaway transcript
    # doesn't bloat the DB. 500 chars is wide enough for any normal
    # student utterance + a margin for prosody-based labels.
    safe_trigger = (trigger_text or "")[:500] or None

    try:
        async with AsyncSessionLocal() as db:
            row = ConfusionEvent(
                student_id=sid,
                course_id=cid,
                session_id=(session_id or None),
                concept_id=concept_uuid,
                slide_idx=slide_idx,
                trigger_text=safe_trigger,
                source=source[:30] if source else "unknown",
                score=float(score),
                resolved=bool(resolved),
                resolution_strategy=(resolution_strategy[:60]
                                     if resolution_strategy else None),
            )
            db.add(row)
            await db.commit()
            log.info(
                "😕 confusion event : student=%s course=%s concept=%s "
                "source=%s score=%.2f strategy=%s",
                str(sid)[:8],
                (str(cid)[:8] if cid else "?"),
                (str(concept_uuid)[:8] if concept_uuid else "?"),
                source, score,
                (resolution_strategy or "?"),
            )
            return str(row.id)
    except Exception as exc:                                                # noqa: BLE001
        log.warning("confusion event : DB write failed (%s)", exc)
        return None
