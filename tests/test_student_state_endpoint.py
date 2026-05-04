"""Tests for the /student/me/state aggregation endpoint.

We test the *aggregation* logic — the endpoint composes 4 backends
(Redis SessionContext + chat history + KG snapshot + engagement
scorer) into one JSON. We mock the 4 sources and verify the response
shape, error handling, and security boundary (auth required).

We don't spin up a real ASGI server here ; we call the endpoint
function directly with a mocked ``user`` dependency. That gives us
unit-test speed with full coverage of the aggregation paths.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from routes.student_state import (
    _aggregate, get_my_state, get_student_state,
)


# ════════════════════════════════════════════════════════════════════
# Pure aggregator (no I/O)
# ════════════════════════════════════════════════════════════════════

class TestAggregator:
    """The ``_aggregate`` helper composes the response. Pure function."""

    def test_full_payload(self):
        ctx = {
            "course_id":     "ir-101",
            "chapter_index": 1,
            "section_index": 4,
            "char_position": 320,
            "state":         "PRESENTING",
            "last_activity": 1700000000.0,
            "paused_state": {"is_paused": False},
        }
        chat = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
        knowledge = {
            "strong_concepts": ["k_means"],
            "weak_concepts":   ["index_inverse"],
            "is_cold_start":   False,
        }
        engagement = {"score": 0.72, "label": "engaged", "reason": "..."}
        out = _aggregate(
            student_id="alice",
            course_id="ir-101",
            ctx_payload=ctx,
            chat_preview=chat,
            knowledge=knowledge,
            engagement=engagement,
        )
        assert out["student_id"] == "alice"
        assert out["course_id"] == "ir-101"
        assert out["current_position"]["chapter_index"] == 1
        assert out["current_position"]["section_index"] == 4
        assert out["current_position"]["is_paused"] is False
        assert out["knowledge_snapshot"]["strong_concepts"] == ["k_means"]
        assert out["engagement"]["score"] == 0.72
        assert len(out["recent_chat_preview"]) == 2

    def test_no_session_context_returns_empty_position(self):
        out = _aggregate(
            student_id="alice",
            course_id="ir",
            ctx_payload=None,
            chat_preview=[],
            knowledge={},
            engagement={"score": None, "label": "unknown", "reason": ""},
        )
        assert out["current_position"] == {}
        assert out["recent_chat_preview"] == []
        assert out["engagement"]["score"] is None

    def test_paused_state_surfaced_correctly(self):
        ctx = {
            "course_id": "ir",
            "paused_state": {"is_paused": True, "slide_id": "s1"},
        }
        out = _aggregate(
            student_id="alice", course_id="ir",
            ctx_payload=ctx, chat_preview=[],
            knowledge={}, engagement={},
        )
        assert out["current_position"]["is_paused"] is True


# ════════════════════════════════════════════════════════════════════
# /student/me/state — happy path (with mocked backends)
# ════════════════════════════════════════════════════════════════════

class TestMeStateEndpoint:

    @pytest.mark.asyncio
    async def test_returns_aggregated_state(self):
        ctx_payload = {
            "course_id": "ir",
            "chapter_index": 0,
            "section_index": 2,
            "state": "PRESENTING",
            "last_activity": 1700000000.0,
            "paused_state": {"is_paused": False},
        }
        chat = [{"role": "user", "content": "What is IR?"}]
        knowledge = {"strong_concepts": ["doc_retrieval"], "is_cold_start": False}
        engagement = {"score": 0.55, "label": "neutral", "reason": "..."}

        reviews_due = {"count": 3, "items": [
            {"concept_name": "k_means", "due": "2024-01-01T00:00:00"},
        ]}
        with patch("routes.student_state._load_session_context",
                   AsyncMock(return_value=ctx_payload)), \
             patch("routes.student_state._load_chat_preview",
                   AsyncMock(return_value=chat)), \
             patch("routes.student_state._load_knowledge_snapshot",
                   AsyncMock(return_value=knowledge)), \
             patch("routes.student_state._load_reviews_due",
                   AsyncMock(return_value=reviews_due)), \
             patch("routes.student_state._compute_engagement_for",
                   AsyncMock(return_value=engagement)):
            user = {"sub": "alice-uuid"}
            out = await get_my_state(course_id="ir", user=user)

        assert out["student_id"] == "alice-uuid"
        assert out["course_id"] == "ir"
        assert out["current_position"]["section_index"] == 2
        assert out["recent_chat_preview"][0]["content"] == "What is IR?"
        assert out["engagement"]["label"] == "neutral"
        assert out["reviews_due"]["count"] == 3

    @pytest.mark.asyncio
    async def test_missing_student_id_in_token_raises_401(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await get_my_state(course_id=None, user={"sub": ""})
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_no_course_id_still_works(self):
        """Endpoint accepts course_id=None — useful for the 'where am I'
        landing screen before a course is picked."""
        with patch("routes.student_state._load_session_context",
                   AsyncMock(return_value=None)), \
             patch("routes.student_state._load_chat_preview",
                   AsyncMock(return_value=[])), \
             patch("routes.student_state._load_knowledge_snapshot",
                   AsyncMock(return_value={})), \
             patch("routes.student_state._load_reviews_due",
                   AsyncMock(return_value={"count": 0, "items": []})), \
             patch("routes.student_state._compute_engagement_for",
                   AsyncMock(return_value={"score": None, "label": "unknown", "reason": ""})):
            out = await get_my_state(course_id=None, user={"sub": "alice"})
        assert out["course_id"] is None
        assert out["current_position"] == {}
        assert out["engagement"]["label"] == "unknown"
        assert out["reviews_due"]["count"] == 0


# ════════════════════════════════════════════════════════════════════
# /student/{id}/state — admin-style lookup
# ════════════════════════════════════════════════════════════════════

class TestAdminStateEndpoint:

    @pytest.mark.asyncio
    async def test_admin_can_view_arbitrary_student(self):
        with patch("routes.student_state._load_session_context",
                   AsyncMock(return_value=None)), \
             patch("routes.student_state._load_chat_preview",
                   AsyncMock(return_value=[])), \
             patch("routes.student_state._load_knowledge_snapshot",
                   AsyncMock(return_value={})), \
             patch("routes.student_state._load_reviews_due",
                   AsyncMock(return_value={"count": 0, "items": []})), \
             patch("routes.student_state._compute_engagement_for",
                   AsyncMock(return_value={"score": None, "label": "unknown"})):
            out = await get_student_state(
                student_id="bob-uuid",
                course_id="ir",
                user={"sub": "admin-uuid"},
            )
        assert out["student_id"] == "bob-uuid"

    @pytest.mark.asyncio
    async def test_empty_student_id_returns_400(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await get_student_state(
                student_id="",
                course_id=None,
                user={"sub": "admin"},
            )
        assert exc.value.status_code == 400


# ════════════════════════════════════════════════════════════════════
# Engagement scoring from a SessionContext blob
# ════════════════════════════════════════════════════════════════════

class TestEngagementFromContext:
    @pytest.mark.asyncio
    async def test_engagement_with_full_signals(self):
        from routes.student_state import _compute_engagement_for
        import time as _time
        now = _time.time()
        ctx = {
            "last_activity": now - 5,
            "session_started_at": now - 600,
            "questions_in_session": 4,
            "confusion_count": 2,
            "consecutive_passive_slides": 0,
            "last_interrupt_latency_ms": 150,
        }
        result = await _compute_engagement_for(ctx)
        assert result["score"] is not None
        assert 0.0 <= result["score"] <= 1.0
        assert result["label"] in ("disengaged", "neutral", "engaged")

    @pytest.mark.asyncio
    async def test_engagement_with_no_context(self):
        from routes.student_state import _compute_engagement_for
        result = await _compute_engagement_for(None)
        assert result["label"] == "unknown"
        assert result["score"] is None


# ════════════════════════════════════════════════════════════════════
# Reviews-due loader (FSRS due-date aggregation)
# ════════════════════════════════════════════════════════════════════

class TestReviewsDueLoader:
    @pytest.mark.asyncio
    async def test_loader_returns_count_and_items(self):
        from routes.student_state import _load_reviews_due
        fake_items = [
            {"concept_name": "k_means", "due": "..."},
            {"concept_name": "vector",  "due": "..."},
        ]
        with patch(
            "pedagogy.review_scheduler.ReviewScheduler.list_due",
            AsyncMock(return_value=fake_items),
        ):
            out = await _load_reviews_due("alice", "ir")
        assert out["count"] == 2
        assert out["items"] == fake_items

    @pytest.mark.asyncio
    async def test_loader_returns_empty_on_failure(self):
        """Scheduler error must not break the endpoint — returns 0/[]."""
        from routes.student_state import _load_reviews_due

        async def _fail(*a, **kw):
            raise RuntimeError("DB down")

        with patch(
            "pedagogy.review_scheduler.ReviewScheduler.list_due",
            side_effect=_fail,
        ):
            out = await _load_reviews_due("alice", "ir")
        assert out["count"] == 0
        assert out["items"] == []


class TestAggregatorWithReviews:
    """The aggregator must surface reviews_due if provided ; absent
    field means the endpoint didn't load it (legacy callers)."""

    def test_reviews_due_passed_through(self):
        from routes.student_state import _aggregate
        rdue = {"count": 5, "items": [{"concept_name": "k_means"}]}
        out = _aggregate(
            student_id="alice", course_id="ir",
            ctx_payload=None, chat_preview=[],
            knowledge={}, engagement={},
            reviews_due=rdue,
        )
        assert out["reviews_due"]["count"] == 5
        assert out["reviews_due"]["items"][0]["concept_name"] == "k_means"

    def test_reviews_due_defaults_to_empty_when_missing(self):
        from routes.student_state import _aggregate
        out = _aggregate(
            student_id="alice", course_id="ir",
            ctx_payload=None, chat_preview=[],
            knowledge={}, engagement={},
            # reviews_due omitted
        )
        assert out["reviews_due"]["count"] == 0
        assert out["reviews_due"]["items"] == []
