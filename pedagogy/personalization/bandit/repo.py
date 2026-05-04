"""Redis-backed persistence for the contextual Thompson bandit.

# What needs to be persisted

  1. **Bandit state** — the dict ``{<bucket>|<arm_id>: ArmPosterior}`` that
     accumulates Beta(α, β) updates across all student interactions.
     Singleton ; one instance shared across the whole tutor process.
     Stored at the key ``bandit:state``.

  2. **Pending decisions** — between two student turns, we hold the
     (context_bucket, chosen_action, mastery_before) tuple so we can
     compute the reward when the next turn arrives. Stored per session
     at ``bandit:pending:{session_id}`` with a 1h TTL (long enough for
     a normal study session, short enough to avoid leaking stale state
     when a session is abandoned).

# Why a 1h TTL on pending decisions

If the student leaves the session, the pending decision is dropped on
its own. The alternative — manually reaping pending entries on session
end — is fragile because session ends can be missed (browser crash,
network drop). TTL keeps the cache self-healing.

# Why singleton bandit (not per-student)

The context bucket already encodes the student-relevant features
(learning_style, pace, mastery_level), so observations from any
student inform any other student in the same bucket. This is the
"contextual bandit with shared statistics" pattern, standard in
recommender systems (Li et al. 2010, *A contextual-bandit approach to
personalized news article recommendation*).

# Why best-effort

Redis is a soft dependency : if it's unavailable, the bandit still
runs in process memory (the in-memory ``ContextualThompsonBandit``
instance), it just won't survive a process restart. Calls to
``save_state`` / ``record_pending`` log a warning and return, so the
graph keeps working without observability.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as aioredis

from pedagogy.personalization.bandit.strategies import StrategyAction
from pedagogy.personalization.bandit.thompson import (
    ContextBucket,
    ContextualThompsonBandit,
)

log = logging.getLogger("personalization.bandit.repo")


_BANDIT_KEY = "bandit:state"
_PENDING_KEY_FMT = "bandit:pending:{session_id}"
_PENDING_TTL_S = 3600  # 1 hour


@dataclass
class PendingDecision:
    """A bandit decision waiting for its outcome on the next turn."""

    context:        ContextBucket
    action:         StrategyAction
    mastery_before: float
    primary_concept: str   # idea_id the answer was about
    timestamp:      float

    def to_dict(self) -> dict:
        return {
            "context":         {
                "learning_style": self.context.learning_style,
                "pace":           self.context.pace,
                "mastery_level":  self.context.mastery_level,
            },
            "action":          self.action.arm_id,
            "mastery_before":  self.mastery_before,
            "primary_concept": self.primary_concept,
            "timestamp":       self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PendingDecision":
        ctx = data.get("context") or {}
        return cls(
            context=ContextBucket(
                learning_style=str(ctx.get("learning_style", "mixed")),
                pace=str(ctx.get("pace", "normal")),
                mastery_level=str(ctx.get("mastery_level", "medium")),
            ),
            action=StrategyAction.from_arm_id(str(data.get("action", "analogy:normal"))),
            mastery_before=float(data.get("mastery_before", 0.5)),
            primary_concept=str(data.get("primary_concept", "")),
            timestamp=float(data.get("timestamp", time.time())),
        )


# ── Redis connection ────────────────────────────────────────────────────


_redis: Optional[aioredis.Redis] = None


async def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        _redis = aioredis.Redis(host=host, port=port, decode_responses=True)
    return _redis


# ── Singleton bandit instance + persistence ─────────────────────────────


_bandit_lock = asyncio.Lock()
_bandit_instance: Optional[ContextualThompsonBandit] = None
_dirty: bool = False  # True when there are unsaved updates


async def get_bandit() -> ContextualThompsonBandit:
    """Return the process-singleton bandit, loading it from Redis on first call.

    Subsequent calls return the same in-memory instance. Updates to the
    instance are only persisted when ``save_state`` is called explicitly
    (or via ``mark_dirty_and_maybe_save`` for periodic saving).
    """
    global _bandit_instance
    if _bandit_instance is not None:
        return _bandit_instance
    async with _bandit_lock:
        if _bandit_instance is not None:
            return _bandit_instance
        loaded = await _load_state()
        _bandit_instance = loaded if loaded is not None else ContextualThompsonBandit()
        log.info(
            "bandit loaded — %d posteriors, %d total pulls",
            len(_bandit_instance.posteriors), _bandit_instance.total_pulls(),
        )
        return _bandit_instance


async def _load_state() -> Optional[ContextualThompsonBandit]:
    """Read the bandit state from Redis. Returns None on miss / error."""
    try:
        r = await _get_redis()
        raw = await r.get(_BANDIT_KEY)
        if not raw:
            return None
        data = json.loads(raw)
        return ContextualThompsonBandit.from_dict(data)
    except Exception as exc:                                              # noqa: BLE001
        log.warning("bandit load failed (continuing with fresh instance): %s", exc)
        return None


async def save_state() -> None:
    """Persist the singleton bandit to Redis. Best-effort."""
    global _dirty
    if _bandit_instance is None:
        return
    try:
        r = await _get_redis()
        payload = json.dumps(_bandit_instance.to_dict())
        await r.set(_BANDIT_KEY, payload)
        _dirty = False
        log.debug(
            "bandit saved (%d posteriors, %d total pulls)",
            len(_bandit_instance.posteriors), _bandit_instance.total_pulls(),
        )
    except Exception as exc:                                              # noqa: BLE001
        log.warning("bandit save failed: %s", exc)


def mark_dirty() -> None:
    """Mark the bandit state as containing unsaved updates."""
    global _dirty
    _dirty = True


async def maybe_save_periodically(every_n_updates: int = 25) -> None:
    """Save the bandit state when the in-memory total pulls cross a multiple
    of ``every_n_updates``. Avoids saving on every single update (Redis I/O
    pressure) while keeping the lag bounded."""
    if _bandit_instance is None or not _dirty:
        return
    pulls = _bandit_instance.total_pulls()
    if pulls > 0 and pulls % every_n_updates == 0:
        await save_state()


# ── Pending decision per session ────────────────────────────────────────


async def record_pending(session_id: str, decision: PendingDecision) -> None:
    """Store a pending decision for ``session_id``. Overwrites any
    existing pending entry — there's only one in-flight decision per
    session at a time. Best-effort."""
    if not session_id:
        return
    try:
        r = await _get_redis()
        await r.setex(
            _PENDING_KEY_FMT.format(session_id=session_id),
            _PENDING_TTL_S,
            json.dumps(decision.to_dict()),
        )
    except Exception as exc:                                              # noqa: BLE001
        log.warning("bandit record_pending failed for %s: %s", session_id[:8], exc)


async def consume_pending(session_id: str) -> Optional[PendingDecision]:
    """Pop and return the pending decision for ``session_id`` (or None
    if there's no pending decision). Atomic read-and-delete via Redis
    GETDEL when available."""
    if not session_id:
        return None
    try:
        r = await _get_redis()
        key = _PENDING_KEY_FMT.format(session_id=session_id)
        # GETDEL is Redis 6.2+. Fall back to GET + DELETE on older versions.
        try:
            raw = await r.execute_command("GETDEL", key)
        except Exception:
            raw = await r.get(key)
            if raw is not None:
                await r.delete(key)
        if not raw:
            return None
        return PendingDecision.from_dict(json.loads(raw))
    except Exception as exc:                                              # noqa: BLE001
        log.warning("bandit consume_pending failed for %s: %s", session_id[:8], exc)
        return None


# ── Test helpers ────────────────────────────────────────────────────────


def _reset_for_tests() -> None:
    """Clear the in-process singleton. Tests only."""
    global _bandit_instance, _dirty
    _bandit_instance = None
    _dirty = False
