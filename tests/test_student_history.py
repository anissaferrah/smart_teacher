"""Tests for the persistent per-(student, course) chat history.

Redis is mocked — the helpers under test only depend on
``pedagogy.dialogue.get_redis``, so we patch that with an
``AsyncMock`` returning a fake Redis stub.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from pedagogy.student_history import (
    HISTORY_TTL_SECONDS, MAX_HISTORY_TURNS,
    _key, append_chat_turn, clear_chat_history, history_length, load_chat_history,
)


# ── Test harness ─────────────────────────────────────────────────────

class _FakeRedis:
    """Minimal in-memory Redis stand-in supporting the 5 ops we use :
    LPUSH, LTRIM, LRANGE, LLEN, EXPIRE, DELETE. Stores per-key lists.

    Order matches Redis : LPUSH adds to the LEFT (newest first),
    LRANGE 0..N reads from the LEFT.
    """
    def __init__(self):
        self.store: dict[str, list[str]] = {}
        self.ttls:  dict[str, int] = {}

    async def lpush(self, key, value):
        self.store.setdefault(key, []).insert(0, value)
        return len(self.store[key])

    async def ltrim(self, key, start, stop):
        if key not in self.store:
            return "OK"
        # Redis LTRIM keeps inclusive start..stop
        self.store[key] = self.store[key][start:stop + 1]
        return "OK"

    async def lrange(self, key, start, stop):
        items = self.store.get(key, [])
        # +1 because LRANGE end-index is INCLUSIVE
        return items[start:stop + 1]

    async def llen(self, key):
        return len(self.store.get(key, []))

    async def expire(self, key, ttl):
        self.ttls[key] = ttl
        return 1

    async def delete(self, key):
        self.store.pop(key, None)
        self.ttls.pop(key, None)
        return 1


@pytest.fixture
def fake_redis():
    """Patches pedagogy.dialogue.get_redis with a fresh fake. Yields
    the fake so tests can inspect its state directly."""
    fake = _FakeRedis()

    async def _aget():
        return fake

    with patch("pedagogy.student_history.get_redis", side_effect=_aget):
        yield fake


# ════════════════════════════════════════════════════════════════════
# Key namespacing
# ════════════════════════════════════════════════════════════════════

class TestKey:
    def test_key_separates_student_and_course(self):
        assert _key("alice", "ir") == "chat::alice::ir"
        # Different course → different key
        assert _key("alice", "ir") != _key("alice", "ml")
        # Different student → different key
        assert _key("alice", "ir") != _key("bob", "ir")

    def test_key_handles_empty_inputs(self):
        # Anonymous + default fallback
        assert "anon" in _key("", "")
        assert "default" in _key("", "")


# ════════════════════════════════════════════════════════════════════
# Append / load roundtrip
# ════════════════════════════════════════════════════════════════════

class TestAppendLoad:
    @pytest.mark.asyncio
    async def test_single_turn_roundtrip(self, fake_redis):
        await append_chat_turn("alice", "ir", "user", "What is K-means?")
        history = await load_chat_history("alice", "ir", limit=5)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What is K-means?"

    @pytest.mark.asyncio
    async def test_multi_turn_chronological_order(self, fake_redis):
        """Stored newest-first via LPUSH but loaded OLDEST-first — the
        order the LLM expects to read."""
        await append_chat_turn("alice", "ir", "user", "Q1")
        await append_chat_turn("alice", "ir", "assistant", "A1")
        await append_chat_turn("alice", "ir", "user", "Q2")
        await append_chat_turn("alice", "ir", "assistant", "A2")
        history = await load_chat_history("alice", "ir", limit=10)
        contents = [h["content"] for h in history]
        assert contents == ["Q1", "A1", "Q2", "A2"]

    @pytest.mark.asyncio
    async def test_isolation_between_students(self, fake_redis):
        """Alice's history must NEVER appear in Bob's load."""
        await append_chat_turn("alice", "ir", "user", "Alice's Q")
        await append_chat_turn("bob",   "ir", "user", "Bob's Q")
        alice_h = await load_chat_history("alice", "ir", limit=10)
        bob_h   = await load_chat_history("bob",   "ir", limit=10)
        assert [h["content"] for h in alice_h] == ["Alice's Q"]
        assert [h["content"] for h in bob_h]   == ["Bob's Q"]

    @pytest.mark.asyncio
    async def test_isolation_between_courses(self, fake_redis):
        """Same student, different course → different histories."""
        await append_chat_turn("alice", "ir", "user", "IR question")
        await append_chat_turn("alice", "ml", "user", "ML question")
        ir_h = await load_chat_history("alice", "ir", limit=10)
        ml_h = await load_chat_history("alice", "ml", limit=10)
        assert [h["content"] for h in ir_h] == ["IR question"]
        assert [h["content"] for h in ml_h] == ["ML question"]


# ════════════════════════════════════════════════════════════════════
# Bounded growth via LTRIM
# ════════════════════════════════════════════════════════════════════

class TestBoundedGrowth:
    @pytest.mark.asyncio
    async def test_history_capped_at_max_turns(self, fake_redis):
        """Appending MAX+10 turns must leave only MAX entries stored.
        Without LTRIM, a long-running session would balloon Redis."""
        for i in range(MAX_HISTORY_TURNS + 10):
            await append_chat_turn("alice", "ir", "user", f"Q{i}")
        n = await history_length("alice", "ir")
        assert n == MAX_HISTORY_TURNS

    @pytest.mark.asyncio
    async def test_oldest_turns_evicted_first(self, fake_redis):
        """When trimmed, the OLDEST turns drop off — most recent
        survive (FIFO eviction)."""
        for i in range(MAX_HISTORY_TURNS + 5):
            await append_chat_turn("alice", "ir", "user", f"turn{i}")
        history = await load_chat_history("alice", "ir", limit=MAX_HISTORY_TURNS)
        # The first 5 should be evicted ; we should see turn5..turnN-1
        first_kept = history[0]["content"]
        assert first_kept == "turn5", (
            f"oldest survivor should be turn5, got {first_kept!r}"
        )


# ════════════════════════════════════════════════════════════════════
# Validation : empty content + bad roles
# ════════════════════════════════════════════════════════════════════

class TestValidation:
    @pytest.mark.asyncio
    async def test_empty_content_silently_dropped(self, fake_redis):
        await append_chat_turn("alice", "ir", "user", "")
        await append_chat_turn("alice", "ir", "user", "   ")
        n = await history_length("alice", "ir")
        assert n == 0

    @pytest.mark.asyncio
    async def test_unknown_role_rejected(self, fake_redis):
        await append_chat_turn("alice", "ir", "system", "ignored")
        await append_chat_turn("alice", "ir", "tool",   "ignored")
        n = await history_length("alice", "ir")
        assert n == 0

    @pytest.mark.asyncio
    async def test_role_normalised_to_lowercase(self, fake_redis):
        await append_chat_turn("alice", "ir", "USER",  "Q")
        await append_chat_turn("alice", "ir", "Assistant", "A")
        history = await load_chat_history("alice", "ir", limit=10)
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


# ════════════════════════════════════════════════════════════════════
# TTL + clear
# ════════════════════════════════════════════════════════════════════

class TestTTLAndClear:
    @pytest.mark.asyncio
    async def test_expire_set_on_each_append(self, fake_redis):
        await append_chat_turn("alice", "ir", "user", "Q")
        ttl = fake_redis.ttls.get("chat::alice::ir")
        assert ttl == HISTORY_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_clear_wipes_history(self, fake_redis):
        await append_chat_turn("alice", "ir", "user", "Q")
        assert await history_length("alice", "ir") == 1
        await clear_chat_history("alice", "ir")
        assert await history_length("alice", "ir") == 0

    @pytest.mark.asyncio
    async def test_clear_does_not_touch_other_keys(self, fake_redis):
        await append_chat_turn("alice", "ir", "user", "alice")
        await append_chat_turn("bob",   "ir", "user", "bob")
        await clear_chat_history("alice", "ir")
        # Bob's history must remain
        assert await history_length("bob", "ir") == 1


# ════════════════════════════════════════════════════════════════════
# Defensive : Redis error doesn't break the user flow
# ════════════════════════════════════════════════════════════════════

class TestDefensive:
    @pytest.mark.asyncio
    async def test_redis_failure_does_not_raise_on_append(self):
        """If Redis is down, append must NOT raise — the caller's
        request flow must continue. Same contract for load (returns
        empty)."""
        async def _fail():
            raise RuntimeError("Redis down")

        with patch("pedagogy.student_history.get_redis", side_effect=_fail):
            # Both calls must complete cleanly
            await append_chat_turn("alice", "ir", "user", "Q")
            history = await load_chat_history("alice", "ir", limit=5)
            assert history == []
            n = await history_length("alice", "ir")
            assert n == 0

    @pytest.mark.asyncio
    async def test_malformed_entry_skipped_not_aborted(self, fake_redis):
        """One corrupt JSON entry shouldn't kill the whole load."""
        # Manually inject a good entry then a bad one
        good = json.dumps({"role": "user", "content": "Q1", "ts": 0.0})
        bad  = "not-json-at-all"
        # In Redis, LPUSH adds to LEFT — newest first. We want chrono
        # order on read = oldest-first, so put GOOD first (rightmost),
        # BAD second (leftmost).
        fake_redis.store["chat::alice::ir"] = [bad, good]
        history = await load_chat_history("alice", "ir", limit=10)
        # Only the good entry survives
        assert len(history) == 1
        assert history[0]["content"] == "Q1"
