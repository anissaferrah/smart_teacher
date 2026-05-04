"""Persistent per-(student, course) chat history.

# Why this exists

The closure ``history`` variable in ``handlers/ws.py`` is scoped to a
SINGLE WebSocket connection — when the same student reconnects on a
different device or after a browser refresh, their previous Q&A
context is gone. Worse, if the WS connection is reused across
sessions, the history can leak across courses.

This module gives each ``(student_id, course_id)`` pair its own chat
history that survives across connections. The "Smart Teacher" can
then say things like *"comme on l'a vu hier dans le chapitre 2, …"*
because the previous exchanges are persisted.

# Storage

Redis list, key = ``chat:{student_id}:{course_id}``. Each entry is a
JSON-encoded ``{"role": "user"|"assistant", "content": str, "ts":
float}``. We use ``LPUSH`` (newest first) + ``LRANGE`` so loading the
N most recent turns is O(N) and doesn't scan the whole list.

Bounded growth : at every append, we ``LTRIM`` the list to the last
``MAX_HISTORY_TURNS`` entries (config). Old turns drop off.

# Why Redis (not Postgres)

The QA path is on the hot loop — every spoken question reads N
turns. Redis gives sub-ms reads. Long-term archiving (analytics,
revision-after-1-week) lives in Postgres ``learning_turn`` rows
already populated by ``services/learning_log.py``. This Redis layer
is only the *working memory* the LLM consults.

# What's NOT in scope

- Cross-course memory : intentionally omitted. The student's history
  in IR doesn't bleed into ML, even if both are taken by the same
  student. Cross-course knowledge transfer happens at the *concept*
  level (KG + mastery), not at the chat-history level.
- LLM-side dedup : turns are stored verbatim ; consumers should
  format/truncate as needed.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from core.config import Config
from pedagogy.dialogue import get_redis

log = logging.getLogger("pedagogy.student_history")


# Sourced from Config so operators can tune per-deployment :
#   - shorter TTL for high-turnover demo environments
#   - longer history cap for advanced users with bigger LLM context
MAX_HISTORY_TURNS   = Config.CHAT_HISTORY_MAX_TURNS
HISTORY_TTL_SECONDS = Config.CHAT_HISTORY_TTL_S


def _key(student_id: str, course_id: str) -> str:
    """Redis key for the (student, course) chat history.

    We hash neither side — student_id is already a UUID and course_id
    is a slug ; both are safe in keys. Using ``::`` as a separator
    avoids collisions with stray colons inside ids (none expected, but
    defensive).
    """
    sid = (student_id or "anon").strip() or "anon"
    cid = (course_id or "default").strip() or "default"
    return f"chat::{sid}::{cid}"


async def append_chat_turn(
    student_id: str,
    course_id: str,
    role: str,
    content: str,
) -> None:
    """Append one chat turn to the (student, course) history.

    Silently skips empty content so the LLM never sees a blank
    "user said: <empty>" turn that would just waste prompt budget.

    ``role`` is normalised to lowercase ; only "user" and "assistant"
    are stored — anything else is rejected (logged at debug level)
    rather than polluting the history with system / tool turns.
    """
    if not content or not content.strip():
        return
    role_norm = (role or "").strip().lower()
    if role_norm not in ("user", "assistant"):
        log.debug("append_chat_turn: rejecting role=%r (not user/assistant)", role)
        return

    payload = json.dumps({
        "role":    role_norm,
        "content": content.strip(),
        "ts":      time.time(),
    }, ensure_ascii=False)

    try:
        r = await get_redis()
        key = _key(student_id, course_id)
        # LPUSH = newest first. LTRIM keeps only the most recent
        # MAX_HISTORY_TURNS entries — the order is reversed when read.
        await r.lpush(key, payload)
        await r.ltrim(key, 0, MAX_HISTORY_TURNS - 1)
        await r.expire(key, HISTORY_TTL_SECONDS)
    except Exception as exc:                                              # noqa: BLE001
        # Never block the user-facing flow on a Redis hiccup. The
        # global in-memory history in ws.py still carries the turn
        # for the current connection — we just lose persistence.
        log.warning(
            "append_chat_turn failed (student=%s course=%s): %s",
            (student_id or "")[:8], course_id, exc,
        )


async def load_chat_history(
    student_id: str,
    course_id: str,
    limit: int = 10,
) -> list[dict]:
    """Return the ``limit`` most recent turns, OLDEST FIRST.

    The chronological order matters : the LLM reads the history as
    "user said X → assistant said Y → user said Z" and gets confused
    if the order is inverted. We store newest-first (LPUSH) for cheap
    appends, then reverse on read.

    Returns ``[]`` on Redis miss, error, or no history. Callers can
    treat the empty list as "first interaction" — same as the legacy
    in-memory ``history`` variable.
    """
    if limit <= 0:
        return []
    try:
        r = await get_redis()
        key = _key(student_id, course_id)
        # LRANGE 0..N-1 = newest N entries (LPUSH order)
        raw_entries = await r.lrange(key, 0, max(0, limit - 1))
    except Exception as exc:                                              # noqa: BLE001
        log.warning(
            "load_chat_history failed (student=%s course=%s): %s",
            (student_id or "")[:8], course_id, exc,
        )
        return []

    out: list[dict] = []
    for raw in raw_entries:
        try:
            entry = json.loads(raw if isinstance(raw, str) else raw.decode())
        except Exception:                                                 # noqa: BLE001
            # Skip a single malformed entry rather than dropping the
            # whole history — defensive against partial Redis writes.
            continue
        if isinstance(entry, dict) and entry.get("content") and entry.get("role"):
            out.append({
                "role":    str(entry["role"]),
                "content": str(entry["content"]),
                "ts":      float(entry.get("ts", 0.0) or 0.0),
            })

    # Reverse to chronological order (oldest first)
    out.reverse()
    return out


async def clear_chat_history(student_id: str, course_id: str) -> None:
    """Wipe all turns for ``(student_id, course_id)``.

    Used by admin tools / "reset progress" actions, NOT during normal
    course transitions (the user explicitly asked to keep history
    across course switches — cross-course leakage is handled by the
    chunk-level filters in retriever.py and responder.py instead).
    """
    try:
        r = await get_redis()
        await r.delete(_key(student_id, course_id))
    except Exception as exc:                                              # noqa: BLE001
        log.warning(
            "clear_chat_history failed (student=%s course=%s): %s",
            (student_id or "")[:8], course_id, exc,
        )


async def history_length(student_id: str, course_id: str) -> int:
    """Return the number of stored turns. Useful for tests and
    observability."""
    try:
        r = await get_redis()
        return int(await r.llen(_key(student_id, course_id)))
    except Exception as exc:                                              # noqa: BLE001
        log.debug("history_length failed: %s", exc)
        return 0
