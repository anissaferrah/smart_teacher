"""Unit tests for the agentic resilience layer (breaker + fallbacks).

Coverage:
  - Breaker state transitions (closed → open → half_open → closed/open)
  - Fallback shape: each fallback returns a schema-faithful state update
  - build_resilient_node end-to-end:
      * passes through on success
      * uses fallback on exception
      * uses fallback when breaker is open
      * works for sync and async nodes (function and class instances)
      * non-dict returns are coerced to {}
"""
import asyncio

import pytest

from agentic.resilience.circuit_breaker import Breaker, BreakerRegistry, BreakerState
from agentic.resilience.fallbacks import NODE_FALLBACKS
from agentic.resilience.wrap import build_resilient_node


# ════════════════════════════════════════════════════════════════════
# Circuit breaker
# ════════════════════════════════════════════════════════════════════

class TestBreaker:
    def test_starts_closed_and_allows(self):
        b = Breaker(name="t1", min_samples=3)
        assert b.state() == BreakerState.CLOSED
        assert b.allow() is True

    def test_min_samples_prevents_premature_opening(self):
        b = Breaker(name="t2", min_samples=5, threshold=0.5)
        # 3 failures, but only 3 samples → not enough to open
        for _ in range(3):
            b.record_failure()
        assert b.state() == BreakerState.CLOSED

    def test_opens_above_threshold(self):
        b = Breaker(name="t3", min_samples=5, threshold=0.5)
        for _ in range(3):
            b.record_failure()
        for _ in range(2):
            b.record_success()
        # 3/5 = 60% failure rate, exceeds 50% threshold
        b.record_failure()
        assert b.state() == BreakerState.OPEN
        assert b.allow() is False

    def test_half_open_after_cool_down(self):
        import time as _time
        b = Breaker(name="t4", min_samples=2, threshold=0.5, cool_down_s=0.05)
        b.record_failure()
        b.record_failure()
        assert b.state() == BreakerState.OPEN
        _time.sleep(0.06)
        assert b.state() == BreakerState.HALF_OPEN
        assert b.allow() is True       # probe is allowed

    def test_probe_success_closes(self):
        import time as _time
        b = Breaker(name="t5", min_samples=2, threshold=0.5, cool_down_s=0.05)
        b.record_failure()
        b.record_failure()
        _time.sleep(0.06)
        b.allow()                       # probe permission
        b.record_success()
        assert b.state() == BreakerState.CLOSED

    def test_probe_failure_reopens(self):
        import time as _time
        b = Breaker(name="t6", min_samples=2, threshold=0.5, cool_down_s=0.05)
        b.record_failure()
        b.record_failure()
        _time.sleep(0.06)
        b.allow()
        b.record_failure()
        assert b.state() == BreakerState.OPEN

    def test_registry_returns_singleton_per_node(self):
        BreakerRegistry.reset()
        b1 = BreakerRegistry.get("xx")
        b2 = BreakerRegistry.get("xx")
        assert b1 is b2

    def test_registry_stats_lists_all(self):
        BreakerRegistry.reset()
        BreakerRegistry.get("a")
        BreakerRegistry.get("b")
        stats = BreakerRegistry.all_stats()
        names = {s["name"] for s in stats}
        assert {"a", "b"} <= names


# ════════════════════════════════════════════════════════════════════
# Fallbacks shape
# ════════════════════════════════════════════════════════════════════

class TestFallbacks:
    @pytest.mark.parametrize("node_name", list(NODE_FALLBACKS.keys()))
    def test_each_fallback_returns_dict(self, node_name):
        state = {"language": "fr", "last_slide_content": "Le théorème de Pythagore."}
        fb = NODE_FALLBACKS[node_name]
        out = fb(state)
        assert isinstance(out, dict)
        assert "timings" in out

    def test_planner_fallback_yields_a_plan(self):
        state = {"language": "fr", "last_slide_content": "Slide content here.", "section_title": "Pythagore"}
        out = NODE_FALLBACKS["planner"](state)
        assert "plan" in out
        plan = out["plan"]
        assert len(plan.ideas) >= 1

    def test_responder_fallback_is_localized_fr(self):
        state = {"language": "fr"}
        out = NODE_FALLBACKS["responder"](state)
        assert "Désolé" in out["answer"]

    def test_responder_fallback_is_localized_en(self):
        state = {"language": "en"}
        out = NODE_FALLBACKS["responder"](state)
        assert "Sorry" in out["answer"]

    def test_narrator_fallback_includes_slide_excerpt(self):
        state = {"language": "fr", "last_slide_content": "X" * 1000}
        out = NODE_FALLBACKS["narrator"](state)
        # Echo present, capped to 600 chars
        assert "X" in out["answer"]
        assert len(out["answer"]) < 800

    def test_intent_fallback_extracts_raw_text(self):
        state = {"language": "fr", "event_payload": {"text": "salut"}}
        out = NODE_FALLBACKS["intent"](state)
        assert out["intent"].type == "question"
        assert out["intent"].payload["raw_text"] == "salut"


# ════════════════════════════════════════════════════════════════════
# build_resilient_node — end-to-end behavior
# ════════════════════════════════════════════════════════════════════

class TestResilientNode:
    def setup_method(self):
        BreakerRegistry.reset()

    @pytest.mark.asyncio
    async def test_success_passes_through(self):
        async def node(state):
            return {"answer": "hello"}
        wrapped = build_resilient_node(node, "responder")
        out = await wrapped({"language": "fr"})
        assert out["answer"] == "hello"
        assert "__fallback_used" not in out  # should not be marked fallback

    @pytest.mark.asyncio
    async def test_exception_uses_fallback(self):
        async def boom(state):
            raise RuntimeError("kaboom")
        wrapped = build_resilient_node(boom, "intent")
        out = await wrapped({"language": "fr", "event_payload": {"text": "x"}})
        assert out["intent"].type == "question"
        assert "intent" in out.get("__fallback_used", [])

    @pytest.mark.asyncio
    async def test_open_breaker_short_circuits(self):
        async def boom(state):
            raise RuntimeError("kaboom")
        wrapped = build_resilient_node(boom, "intent")
        # Trigger enough failures to push the breaker open
        for _ in range(6):
            await wrapped({"language": "fr", "event_payload": {"text": "x"}})
        assert BreakerRegistry.get("intent").state() == BreakerState.OPEN

        # Next call: replace node body with a counter to prove breaker bypassed it
        called = {"count": 0}

        async def should_not_run(state):
            called["count"] += 1
            return {}
        wrapped2 = build_resilient_node(should_not_run, "intent")
        out = await wrapped2({"language": "fr", "event_payload": {"text": "y"}})
        assert called["count"] == 0       # breaker prevented invocation
        assert "intent" in out.get("__fallback_used", [])

    @pytest.mark.asyncio
    async def test_sync_node_is_run_in_thread(self):
        def sync_node(state):
            return {"answer": "sync ok"}
        wrapped = build_resilient_node(sync_node, "responder")
        out = await wrapped({"language": "fr"})
        assert out["answer"] == "sync ok"

    @pytest.mark.asyncio
    async def test_class_with_async_call_is_awaited(self):
        # Mirrors the real LangGraph node shape: an instance with `async def __call__`.
        # iscoroutinefunction(instance) returns False, so the wrapper must
        # also probe instance.__call__ — otherwise the coroutine is never awaited.
        class AsyncAgent:
            async def __call__(self, state):
                await asyncio.sleep(0)
                return {"answer": "async-class ok"}

        wrapped = build_resilient_node(AsyncAgent(), "responder")
        out = await wrapped({"language": "fr"})
        assert out["answer"] == "async-class ok"

    @pytest.mark.asyncio
    async def test_class_with_sync_call_runs_in_thread(self):
        class SyncAgent:
            def __call__(self, state):
                return {"answer": "sync-class ok"}

        wrapped = build_resilient_node(SyncAgent(), "responder")
        out = await wrapped({"language": "fr"})
        assert out["answer"] == "sync-class ok"

    @pytest.mark.asyncio
    async def test_non_dict_return_coerced(self):
        async def weird(state):
            return "this is not a dict"
        wrapped = build_resilient_node(weird, "intent")
        out = await wrapped({"language": "fr"})
        assert out == {}
