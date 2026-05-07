"""Extract bandit-ready logged episodes from Postgres + helpers for the
write path.

# Two complementary roles

  1. **Write path helper** — ``extra_payload_from_state(qa_final)`` :
     pulls the bandit fields out of a Q&A graph's final state and returns
     a flat dict ready to be merged into ``services/learning_log.py``'s
     ``extra_payload``. Keeps the call sites in ws.py terse.

  2. **Read path** — ``extract_episodes_from_postgres(...)`` and
     ``extract_to_jsonl(...)`` : query the LearningEvent table for rows
     that contain bandit annotations and convert them into
     ``LoggedEpisode`` objects (or write them to a JSONL on disk for
     ``train_offline`` to consume).

The schema mapping :

    LearningEvent.student_state  → context features (learning_style,
                                    avg_response_time, mastery_score)
    LearningEvent.event_payload  → bandit_strategy, bandit_speech_rate,
                                    propensity (if present)
    LearningEvent.reward         → reward (already a normalized scalar)
    LearningEvent.confusion_score → fallback when reward is missing

Rows that lack bandit annotations are silently skipped — they belong to
older runs (pre-Phase-1) and don't carry the action information needed
for offline training.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from pedagogy.personalization.bandit.offline import EpisodeDataset, LoggedEpisode
from pedagogy.personalization.bandit.strategies import (
    SpeechRate,
    Strategy,
    StrategyAction,
    all_actions,
)
from pedagogy.personalization.bandit.thompson import (
    ContextBucket,
    discretize_mastery,
    discretize_response_time,
)

log = logging.getLogger("personalization.bandit.log_extractor")


# ── Write path : packaging bandit info from QA state ────────────────────


def extra_payload_from_state(qa_final: dict | None) -> dict[str, Any]:
    """Extract bandit-related fields from the Q&A graph's final state.

    Returns a flat dict ready to be merged into ``extra_payload`` of
    ``services.learning_log.persist_learning_turn``. Returns an empty
    dict when the state has no actions or no bandit annotations.

    Usage in ws.py :

        from pedagogy.personalization.bandit.log_extractor import (
            extra_payload_from_state,
        )
        await persist_learning_turn(
            ...,
            extra_payload={
                "source": "streaming",
                **extra_payload_from_state(qa_final),
            },
        )
    """
    if not isinstance(qa_final, dict):
        return {}
    actions = qa_final.get("actions") or []
    if not actions:
        return {}
    payload = getattr(actions[0], "payload", None)
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("bandit_strategy", "bandit_speech_rate", "bandit_context"):
        val = payload.get(key)
        if val is not None:
            out[key] = val
    return out


# ── Read path : Postgres → LoggedEpisode ────────────────────────────────


def _build_context_from_student_state(state: dict) -> Optional[ContextBucket]:
    """Reconstruct a ContextBucket from a student_state JSON snapshot.

    Returns ``None`` if the snapshot lacks the minimum required features.
    """
    if not isinstance(state, dict):
        return None
    style = str(state.get("learning_style", "")).strip().lower()
    if not style:
        return None
    rt = state.get("avg_response_time", state.get("avg_response_time_s", 0.0))
    try:
        rt = float(rt)
    except (TypeError, ValueError):
        rt = 0.0
    mastery = state.get("mastery_score", state.get("recent_mastery", 0.5))
    try:
        mastery = float(mastery)
    except (TypeError, ValueError):
        mastery = 0.5
    return ContextBucket(
        learning_style=style,
        pace=discretize_response_time(rt),
        mastery_level=discretize_mastery(mastery),
    )


def _build_action_from_payload(payload: dict) -> Optional[StrategyAction]:
    """Reconstruct a StrategyAction from event_payload bandit fields."""
    if not isinstance(payload, dict):
        return None
    strategy_str = str(payload.get("bandit_strategy", "")).strip().lower()
    rate_str = str(payload.get("bandit_speech_rate", "")).strip().lower()
    if not strategy_str or not rate_str:
        return None
    try:
        return StrategyAction(
            strategy=Strategy(strategy_str),
            speech_rate=SpeechRate(rate_str),
        )
    except ValueError:
        return None


def event_to_episode(event_row: dict) -> Optional[LoggedEpisode]:
    """Convert one LearningEvent row (as a dict) into a LoggedEpisode.

    Returns ``None`` if the row lacks any required field. The ``event_row``
    dict is expected to follow the LearningEvent column names (see
    ``database/models.py``).
    """
    student_state = event_row.get("student_state") or {}
    event_payload = event_row.get("event_payload") or {}

    context = _build_context_from_student_state(student_state)
    if context is None:
        return None

    action = _build_action_from_payload(event_payload)
    if action is None:
        return None

    # Reward : prefer the explicit reward column, fall back to (1 - confusion_score)
    raw_reward = event_row.get("reward")
    if raw_reward is None:
        raw_reward = 1.0 - float(event_row.get("confusion_score", 0.0) or 0.0)
    try:
        reward = max(0.0, min(1.0, float(raw_reward)))
    except (TypeError, ValueError):
        reward = 0.0

    propensity = event_payload.get("bandit_propensity")
    try:
        propensity = float(propensity) if propensity is not None else 1.0 / sum(1 for _ in all_actions())
    except (TypeError, ValueError):
        propensity = 1.0 / sum(1 for _ in all_actions())

    return LoggedEpisode(
        context=context,
        action=action,
        reward=reward,
        propensity=propensity,
    )


async def extract_episodes_from_postgres(
    since: Optional[datetime] = None,
    limit: int = 100_000,
) -> EpisodeDataset:
    """Query LearningEvent rows and convert them to LoggedEpisode.

    Args :
        since : if given, only events ``created_at >= since`` are pulled.
                None means "all".
        limit : maximum number of rows to read (defensive cap).

    Returns an EpisodeDataset. Rows that can't be converted are silently
    skipped (logged at DEBUG level).

    Best-effort : returns an empty dataset on Postgres error rather than
    raising, so the caller can degrade to simulator-only training.
    """
    ds = EpisodeDataset()
    try:
        from sqlalchemy import select
        from database.init_db import AsyncSessionLocal
        from database.models import LearningEvent
    except Exception as exc:                                              # noqa: BLE001
        log.warning("Postgres unavailable, returning empty dataset: %s", exc)
        return ds

    try:
        async with AsyncSessionLocal() as db:
            stmt = select(LearningEvent).order_by(LearningEvent.created_at.asc())
            if since is not None:
                stmt = stmt.where(LearningEvent.created_at >= since)
            stmt = stmt.limit(limit)
            rows = (await db.execute(stmt)).scalars().all()
            n_total = len(rows)
            n_kept = 0
            for row in rows:
                row_dict = {
                    "id":              str(row.id),
                    "session_id":      str(row.session_id) if row.session_id else None,
                    "student_id":      str(row.student_id) if row.student_id else None,
                    "course_id":       str(row.course_id) if row.course_id else None,
                    "event_type":      row.event_type,
                    "concept":         row.concept,
                    "action_taken":    row.action_taken,
                    "confusion_score": row.confusion_score,
                    "reward":          row.reward,
                    "student_state":   row.student_state or {},
                    "event_payload":   row.event_payload or {},
                    "created_at":      row.created_at,
                }
                episode = event_to_episode(row_dict)
                if episode is not None:
                    ds.append(episode)
                    n_kept += 1
            log.info(
                "extracted %d / %d episodes from LearningEvent (%.1f%% with bandit annotations)",
                n_kept, n_total,
                100.0 * n_kept / max(1, n_total),
            )
    except Exception as exc:                                              # noqa: BLE001
        log.warning("Postgres extraction failed: %s", exc)

    return ds


async def extract_to_jsonl(
    output_path: str | Path,
    days: Optional[int] = None,
    limit: int = 100_000,
) -> int:
    """Convenience : extract recent logs and dump as JSONL on disk.

    ``days`` filters to rows from the last N days (None = all). Returns
    the number of episodes written.
    """
    since = datetime.utcnow() - timedelta(days=days) if days else None
    ds = await extract_episodes_from_postgres(since=since, limit=limit)
    ds.to_jsonl(output_path)
    return len(ds)
