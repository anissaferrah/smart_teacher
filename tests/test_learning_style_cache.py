"""Tests for the 2-level learning-style cache.

Covers:
  - L1 hit / miss / TTL expiry
  - L1 size-bound eviction
  - Single-flight coalescing (concurrent misses for same key → 1 recompute)
  - Invalidation drops L1 (and L2 best-effort)
  - Stats counters
  - get_or_compute returns None on missing student_id

L2 (Redis) is exercised via best-effort import; tests don't assume Redis.
"""
import asyncio
import time

import pytest
import pytest_asyncio

from services import learning_style_cache as cache_mod
from services.learning_style_cache import (
    _l1_cache, _l1_get, _l1_put, _l1_invalidate,
    get_or_compute, get_stats, invalidate, reset_stats,
)


# Test student IDs that may bleed into Redis across runs — we proactively
# evict them from L2 in the fixture below so we don't poison neighbours.
_TEST_SIDS = ["sid-0", "sid-1", "sid-2", "sid-3", "sid-4"]


@pytest_asyncio.fixture(autouse=True)
async def clear_cache_state():
    """Reset L1 + stats + in-flight + L2 (best-effort) before each test."""
    _l1_cache.clear()
    cache_mod._inflight.clear()
    reset_stats()
    # Best-effort L2 cleanup — silently noop if Redis unavailable
    for sid in _TEST_SIDS:
        try:
            await cache_mod._redis_invalidate(sid)
        except Exception:
            pass
    reset_stats()  # zero out errors triggered by the cleanup itself
    yield
    _l1_cache.clear()
    cache_mod._inflight.clear()


# ════════════════════════════════════════════════════════════════════
# L1 (in-process)
# ════════════════════════════════════════════════════════════════════

class TestL1:
    def test_put_then_get(self):
        _l1_put("sid-1", "fr", {"hint": "X"})
        assert _l1_get("sid-1", "fr") == {"hint": "X"}

    def test_separate_lang_keys(self):
        _l1_put("sid-1", "fr", {"v": 1})
        _l1_put("sid-1", "en", {"v": 2})
        assert _l1_get("sid-1", "fr") == {"v": 1}
        assert _l1_get("sid-1", "en") == {"v": 2}

    def test_ttl_expiry(self, monkeypatch):
        # Put with very short TTL by patching time backwards after put
        _l1_put("sid-1", "fr", {"v": 1})
        # Force expiry by mutating the entry's expires_at
        key = ("sid-1", "fr")
        cache_mod._l1_cache[key]["expires_at"] = time.time() - 1
        assert _l1_get("sid-1", "fr") is None

    def test_invalidate_drops_all_langs_for_student(self):
        _l1_put("sid-1", "fr", {"v": 1})
        _l1_put("sid-1", "en", {"v": 2})
        _l1_put("sid-2", "fr", {"v": 3})
        _l1_invalidate("sid-1")
        assert _l1_get("sid-1", "fr") is None
        assert _l1_get("sid-1", "en") is None
        assert _l1_get("sid-2", "fr") == {"v": 3}

    def test_eviction_when_full(self, monkeypatch):
        monkeypatch.setattr(cache_mod, "_L1_MAX_ENTRIES", 3)
        for i in range(5):
            _l1_put(f"sid-{i}", "fr", {"v": i})
        # Some entries must have been evicted (cap = 3)
        assert len(_l1_cache) == 3


# ════════════════════════════════════════════════════════════════════
# get_or_compute
# ════════════════════════════════════════════════════════════════════

class TestGetOrCompute:
    @pytest.mark.asyncio
    async def test_miss_then_hit(self):
        calls = {"n": 0}

        async def recompute():
            calls["n"] += 1
            return {"hint": "X", "params": {}, "dominant": "visual", "source": "bayes"}

        # First call: miss → compute → cached
        v1 = await get_or_compute("sid-1", "fr", recompute)
        assert v1 is not None
        assert calls["n"] == 1
        # Second call: L1 hit → no recompute
        v2 = await get_or_compute("sid-1", "fr", recompute)
        assert v2 == v1
        assert calls["n"] == 1

        s = get_stats()
        assert s["misses"] == 1
        assert s["hits_l1"] == 1

    @pytest.mark.asyncio
    async def test_none_student_id_returns_none(self):
        called = {"n": 0}

        async def recompute():
            called["n"] += 1
            return {}

        assert await get_or_compute(None, "fr", recompute) is None
        assert await get_or_compute("", "fr", recompute) is None
        assert called["n"] == 0          # never invoked

    @pytest.mark.asyncio
    async def test_recompute_failure_returns_none(self):
        async def recompute():
            raise RuntimeError("db down")

        v = await get_or_compute("sid-1", "fr", recompute)
        assert v is None
        s = get_stats()
        assert s["errors"] >= 1

    @pytest.mark.asyncio
    async def test_concurrent_misses_coalesce(self):
        calls = {"n": 0}
        gate = asyncio.Event()

        async def slow_recompute():
            await gate.wait()
            calls["n"] += 1
            return {"hint": "X", "params": {}, "dominant": "visual", "source": "bayes"}

        # Launch 5 concurrent misses for the SAME key with a slow recompute
        tasks = [
            asyncio.create_task(get_or_compute("sid-1", "fr", slow_recompute))
            for _ in range(5)
        ]
        await asyncio.sleep(0.01)        # let them all reach the inflight registry
        gate.set()
        results = await asyncio.gather(*tasks)

        assert all(r is not None for r in results)
        assert calls["n"] == 1           # ONE recompute despite 5 concurrent callers
        s = get_stats()
        assert s["single_flight"] >= 4

    @pytest.mark.asyncio
    async def test_concurrent_misses_different_keys_dont_coalesce(self):
        calls = {"n": 0}

        async def recompute():
            calls["n"] += 1
            await asyncio.sleep(0.01)
            return {"hint": "X", "params": {}, "dominant": "visual", "source": "bayes"}

        tasks = [
            asyncio.create_task(get_or_compute(f"sid-{i}", "fr", recompute))
            for i in range(3)
        ]
        await asyncio.gather(*tasks)
        # 3 distinct students → 3 distinct recomputes (no coalescing across keys)
        assert calls["n"] == 3


# ════════════════════════════════════════════════════════════════════
# Invalidation
# ════════════════════════════════════════════════════════════════════

class TestInvalidate:
    @pytest.mark.asyncio
    async def test_invalidate_drops_l1(self):
        _l1_put("sid-1", "fr", {"v": 1})
        await invalidate("sid-1")
        assert _l1_get("sid-1", "fr") is None

    @pytest.mark.asyncio
    async def test_invalidate_increments_counter(self):
        _l1_put("sid-1", "fr", {"v": 1})
        await invalidate("sid-1")
        assert get_stats()["invalidations"] == 1

    @pytest.mark.asyncio
    async def test_invalidate_with_none_is_noop(self):
        # Must not raise
        await invalidate(None)
        await invalidate("")
        assert get_stats()["invalidations"] == 0


# ════════════════════════════════════════════════════════════════════
# Stats
# ════════════════════════════════════════════════════════════════════

class TestStats:
    def test_initial_stats_zero(self):
        s = get_stats()
        assert s == {
            "hits_l1": 0, "hits_l2": 0, "misses": 0,
            "single_flight": 0, "invalidations": 0, "errors": 0,
        }

    def test_stats_is_a_copy(self):
        s = get_stats()
        s["hits_l1"] = 999
        assert get_stats()["hits_l1"] == 0
