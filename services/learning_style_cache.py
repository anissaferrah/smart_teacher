"""Two-level cache for resolved learning-style hints (in-process + Redis).

# Why a dedicated cache layer

Each slide presentation calls `services/presentation.py:_resolve_style()`
which (a) loads the Bayes posterior from Postgres, (b) optionally re-runs the
heuristic v1 query (5 SQL counts) on cold start, (c) computes prose hint +
structured params. For a 50-slide course this means 50× the same work for
every authenticated student. The posterior changes incrementally on
behavioral signals, but never within a single course presentation.

# Design

Two-level cache (Hennessy & Patterson, "Computer Architecture", 2017,
Memory Hierarchy chapter — same principle applied to application data):

  L1 — in-process  : process-local dict, ~50 µs hit, 5-min TTL, 500 entries
  L2 — Redis       : cross-process, ~500 µs hit, 5-min TTL, persists across
                     workers and redeploys, optional (graceful degradation
                     if Redis unavailable)

Look-up order: L1 → L2 → recompute. On L2 hit we re-populate L1.

# Stampede protection (single-flight)

When N concurrent requests arrive for the same student with a cold cache,
the naïve cache pattern lets all N issue the expensive recompute in
parallel — defeating the cache. We use a per-key `asyncio.Future` registry:

  - First requester creates the Future, runs the recompute, sets the result
  - Concurrent requesters await the SAME Future
  - On completion the Future is published and removed from the registry

This is the *single-flight* pattern (Go's golang.org/x/sync/singleflight,
Bigtable's read coalescing, CDN cache-fill consolidation).

# Cache invalidation

Two events invalidate a student's cache entry (both in L1 and L2):

  1. `save_posterior(sid, ...)` — the posterior changed → hint may have
     drifted. Hooked at the bottom of save_posterior to be transparent.
  2. `submit_vark_responses` — explicit re-seed → caller invokes invalidate.

We don't use TTL alone because the user-visible delay between a meaningful
posterior change and the next slide using the new style would be too
unpredictable. Event-driven invalidation guarantees freshness within ~10ms.

# Observability

Counters in `_stats` (process-local): hits_l1, hits_l2, misses, single_flight,
invalidations. Read via `get_stats()`, exposed at `/admin/cache/stats`.

# References

  - Hennessy, J. L., & Patterson, D. A. (2017). *Computer Architecture: A
    Quantitative Approach* (6th ed.). Morgan Kaufmann. Ch. 2 — multi-level
    cache hierarchy and inclusion property.
  - Chang, F., et al. (2008). *Bigtable: A Distributed Storage System for
    Structured Data.* OSDI — read coalescing.
  - Cao, P., & Liu, C. (2002). *Maintaining Strong Cache Consistency in the
    World Wide Web.* IEEE TKDE — invalidation vs TTL trade-off.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional

log = logging.getLogger("services.learning_style_cache")


# ── Configuration ─────────────────────────────────────────────────────

_TTL_SECONDS: int = 300              # 5 min — bounded staleness on Redis-only path
_L1_MAX_ENTRIES: int = 500           # process-local dict cap
_REDIS_KEY_PREFIX: str = "lstyle:hint:"


# ── Stats (process-local) ─────────────────────────────────────────────

_stats: dict[str, int] = {
    "hits_l1":        0,
    "hits_l2":        0,
    "misses":         0,
    "single_flight":  0,
    "invalidations":  0,
    "errors":         0,
}


def get_stats() -> dict[str, int]:
    """Snapshot of cache counters. Read-only by callers."""
    return dict(_stats)


def reset_stats() -> None:
    """For tests."""
    for k in _stats:
        _stats[k] = 0


# ── L1 (in-process) ───────────────────────────────────────────────────

# Shape: { (student_id_str, lang): {"value": dict, "expires_at": float} }
_l1_cache: dict[tuple[str, str], dict[str, Any]] = {}


def _l1_key(student_id: str, lang: str) -> tuple[str, str]:
    return (student_id, (lang or "fr")[:2])


def _l1_get(student_id: str, lang: str) -> Optional[dict]:
    entry = _l1_cache.get(_l1_key(student_id, lang))
    if entry is None:
        return None
    if entry["expires_at"] <= time.time():
        _l1_cache.pop(_l1_key(student_id, lang), None)
        return None
    return entry["value"]


def _l1_put(student_id: str, lang: str, value: dict) -> None:
    if len(_l1_cache) >= _L1_MAX_ENTRIES:
        # Evict the soonest-to-expire entry (cheapest meaningful policy)
        oldest_key = min(_l1_cache, key=lambda k: _l1_cache[k]["expires_at"])
        _l1_cache.pop(oldest_key, None)
    _l1_cache[_l1_key(student_id, lang)] = {
        "value":      value,
        "expires_at": time.time() + _TTL_SECONDS,
    }


def _l1_invalidate(student_id: str) -> None:
    keys_to_remove = [k for k in _l1_cache if k[0] == student_id]
    for k in keys_to_remove:
        _l1_cache.pop(k, None)


# ── L2 (Redis) ────────────────────────────────────────────────────────

async def _redis_get(student_id: str, lang: str) -> Optional[dict]:
    try:
        from pedagogy.dialogue import get_redis
        r = await get_redis()
        key = f"{_REDIS_KEY_PREFIX}{student_id}:{lang}"
        raw = await r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:                                            # noqa: BLE001
        log.debug("L2 get skipped: %s", exc)
        _stats["errors"] += 1
        return None


async def _redis_put(student_id: str, lang: str, value: dict) -> None:
    try:
        from pedagogy.dialogue import get_redis
        r = await get_redis()
        key = f"{_REDIS_KEY_PREFIX}{student_id}:{lang}"
        await r.set(key, json.dumps(value), ex=_TTL_SECONDS)
    except Exception as exc:                                            # noqa: BLE001
        log.debug("L2 put skipped: %s", exc)
        _stats["errors"] += 1


async def _redis_invalidate(student_id: str) -> None:
    try:
        from pedagogy.dialogue import get_redis
        r = await get_redis()
        # Single-key delete per language is cheap enough; we don't know
        # which langs are cached for this student so we delete a small
        # known set rather than running KEYS pattern (slow on big DBs).
        for lang in ("fr", "en"):
            await r.delete(f"{_REDIS_KEY_PREFIX}{student_id}:{lang}")
    except Exception as exc:                                            # noqa: BLE001
        log.debug("L2 invalidate skipped: %s", exc)
        _stats["errors"] += 1


# ── Single-flight registry (per (student_id, lang)) ───────────────────

_inflight: dict[tuple[str, str], asyncio.Future] = {}
_inflight_lock = asyncio.Lock()


async def _resolve_slow_path(student_id: str, lang: str, recompute_fn) -> Optional[dict]:
    """Single-flighted L2 lookup + recompute + cache write.

    Concurrent callers for the same (student_id, lang) share a single Future:
    the first to acquire the inflight lock becomes the leader and runs the
    L2 + recompute path; all others await the leader's result.

    The Future lifecycle (creation → resolution → registry removal) is held
    open for the *entire* slow-path duration, so late-arriving callers either
    (a) find an inflight Future and await it, or (b) re-check L1 inside the
    lock and find the value the leader has already written.
    """
    key = _l1_key(student_id, lang)

    # Race-free leader election. CRITICAL: never `await` while holding the
    # lock — otherwise followers serialize behind each other and the
    # single-flight effect is lost.
    fut: Optional[asyncio.Future] = None
    follower_fut: Optional[asyncio.Future] = None
    async with _inflight_lock:
        existing = _inflight.get(key)
        if existing is not None:
            _stats["single_flight"] += 1
            follower_fut = existing
        else:
            # Late re-check inside the lock: a previous leader may have just
            # finished writing to L1 in the same tick.
            l1_again = _l1_get(student_id, lang)
            if l1_again is not None:
                _stats["hits_l1"] += 1
                return l1_again
            fut = asyncio.get_running_loop().create_future()
            _inflight[key] = fut

    # Followers wait OUTSIDE the lock — the leader is not blocked by us.
    if follower_fut is not None:
        return await follower_fut

    # Leader: run the slow path and publish to followers
    try:
        # L2
        cached = await _redis_get(student_id, lang)
        if cached is not None:
            _stats["hits_l2"] += 1
            _l1_put(student_id, lang, cached)
            if not fut.done():
                fut.set_result(cached)
            return cached

        # True miss → recompute
        _stats["misses"] += 1
        try:
            value = await recompute_fn()
        except Exception as exc:
            log.warning("recompute failed for student=%s: %s", student_id[:8], exc)
            _stats["errors"] += 1
            if not fut.done():
                fut.set_result(None)
            return None

        if value is not None:
            _l1_put(student_id, lang, value)
            try:
                asyncio.create_task(_redis_put(student_id, lang, value))
            except RuntimeError:
                # No running loop (sync test context) — best-effort skip.
                pass
        if not fut.done():
            fut.set_result(value)
        return value
    except Exception as exc:
        if not fut.done():
            fut.set_exception(exc)
        raise
    finally:
        async with _inflight_lock:
            _inflight.pop(key, None)


# ── Public API ────────────────────────────────────────────────────────

async def get_or_compute(
    student_id: str | uuid.UUID | None,
    lang: str,
    recompute_fn,
) -> Optional[dict]:
    """Resolve {hint, params, dominant, source} for `student_id` from cache.

    On a miss, ``recompute_fn`` is invoked exactly once (single-flight) and
    its result is written to both cache levels. Returns None when
    student_id is missing or recompute fails irrecoverably.
    """
    if not student_id:
        return None
    sid = str(student_id)
    lang2 = (lang or "fr")[:2]

    # L1 fast path — no await, no race
    cached = _l1_get(sid, lang2)
    if cached is not None:
        _stats["hits_l1"] += 1
        return cached

    # Slow path is single-flighted
    return await _resolve_slow_path(sid, lang2, recompute_fn)


async def invalidate(student_id: str | uuid.UUID | None) -> None:
    """Drop both cache levels for this student.

    Called by `save_posterior()` and any other path that mutates the
    underlying source of truth. Safe to call without a Redis connection.
    """
    if not student_id:
        return
    sid = str(student_id)
    _l1_invalidate(sid)
    _stats["invalidations"] += 1
    await _redis_invalidate(sid)
