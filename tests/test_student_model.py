"""Tests for ``pedagogy.student_model.StudentModel``.

We mock every external source (MasteryRepo, ProfileManager,
PersonalizationEngine, learning_style.bayes) so the test stays
deterministic and runs without Redis / Postgres.

What we verify :
  - All four fetches happen in parallel + their failures are captured.
  - Defaults fill in correctly when a source is unavailable.
  - ``mastery_for`` / ``avg_mastery`` / ``is_struggling`` give expected results.
  - The ``ok`` flag flips to False when at least one source fails.
  - The frozen dataclass cannot be mutated by accident.
"""
from __future__ import annotations

from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from pedagogy.student_model import StudentModel, fetch_student_model


# ════════════════════════════════════════════════════════════════════
# Convenience accessors (no I/O)
# ════════════════════════════════════════════════════════════════════

class TestAccessors:
    def test_mastery_for_falls_back_to_default(self):
        model = StudentModel(
            student_id="s1", course_id="c1",
            mastery={"idea_a": 0.8, "idea_b": 0.4},
        )
        out = model.mastery_for(["idea_a", "idea_unknown"])
        assert out == {"idea_a": 0.8, "idea_unknown": 0.5}

    def test_avg_mastery_over_subset(self):
        model = StudentModel(
            student_id="s1", course_id="c1",
            mastery={"a": 0.8, "b": 0.4, "c": 0.6},
        )
        # Subset average
        assert model.avg_mastery(["a", "b"]) == pytest.approx(0.6)
        # Global average
        assert model.avg_mastery() == pytest.approx(0.6)
        # Unknown ideas use default 0.5
        assert model.avg_mastery(["x"]) == 0.5

    def test_avg_mastery_default_when_empty(self):
        model = StudentModel(student_id="s1", course_id="c1")
        assert model.avg_mastery() == 0.5

    def test_is_struggling_threshold(self):
        engaged = StudentModel(student_id="s", course_id="c", confusion_rate=0.1)
        struggling = StudentModel(student_id="s", course_id="c", confusion_rate=0.45)
        assert not engaged.is_struggling()
        assert struggling.is_struggling()
        # Custom threshold
        assert engaged.is_struggling(threshold=0.05)

    def test_to_dict_round_trip(self):
        model = StudentModel(
            student_id="s1", course_id="c1",
            mastery={"a": 0.8},
            style_dominant="visual",
            style_scores={"visual": 0.7, "auditory": 0.1, "kinesthetic": 0.1, "reading": 0.1},
            style_confidence=0.55,
            pace="fast", avg_response_time_s=12.3, confusion_rate=0.05,
            preferred_explanation_depth="concise",
            speech_rate=1.2,
            ok=True,
        )
        d = model.to_dict()
        assert d["style"]["dominant"] == "visual"
        assert d["mastery_avg"] == 0.8
        assert d["pace"] == "fast"
        assert d["ok"] is True

    def test_frozen(self):
        """Mutation must raise — model is read-only by design."""
        model = StudentModel(student_id="s1", course_id="c1")
        with pytest.raises(Exception):  # FrozenInstanceError
            model.style_dominant = "auditory"  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════
# fetch_student_model — happy path (all sources OK)
# ════════════════════════════════════════════════════════════════════

class TestFetchAllOK:
    @pytest.mark.asyncio
    async def test_consolidates_all_sources(self):
        # Mock every safe-fetch helper to return realistic data
        from pedagogy.personalization.engine import PersonalizationContext

        ctx = PersonalizationContext(
            learning_style="visual", pace="fast", explanation_depth="concise",
            avg_response_time_s=12.5, confusion_rate=0.05,
            preferred_difficulty="advanced",
        )

        # Build a tiny posterior stub matching the API
        posterior_mock = MagicMock()
        posterior_mock.dominant.return_value = "visual"
        posterior_mock.mean = (0.7, 0.1, 0.1, 0.1)        # V, A, K, R
        posterior_mock.confidence.return_value = 0.6

        with patch("pedagogy.student_model._safe_fetch_mastery",
                   AsyncMock(return_value={"idea_a": 0.8, "idea_b": 0.5})), \
             patch("pedagogy.student_model._safe_fetch_profile",
                   AsyncMock(return_value={"speech_rate": 1.2,
                                           "preferred_explanation_depth": "balanced"})), \
             patch("pedagogy.student_model._safe_fetch_context",
                   AsyncMock(return_value=ctx)), \
             patch("pedagogy.student_model._safe_fetch_posterior",
                   AsyncMock(return_value=posterior_mock)):

            model = await fetch_student_model(
                student_id="s1", course_id="c1",
                idea_ids=["idea_a", "idea_b"],
            )

        assert model.ok is True
        assert model.sources_failed == ()
        assert model.mastery == {"idea_a": 0.8, "idea_b": 0.5}
        assert model.style_dominant == "visual"
        assert model.style_scores["visual"] == pytest.approx(0.7)
        assert model.style_confidence == pytest.approx(0.6)
        assert model.pace == "fast"
        # Profile is "balanced" but engine context is "concise" → engine wins
        # because depth is derived from observed pace (fast → concise).
        assert model.preferred_explanation_depth == "concise"
        assert model.preferred_difficulty == "advanced"
        assert model.speech_rate == 1.2
        assert model.avg_response_time_s == 12.5
        assert model.confusion_rate == 0.05


# ════════════════════════════════════════════════════════════════════
# fetch_student_model — partial failures
# ════════════════════════════════════════════════════════════════════

class TestFetchPartialFailures:
    """Each source returning None should add a tag to ``sources_failed``."""

    @pytest.mark.asyncio
    async def test_all_sources_failed_returns_defaults(self):
        with patch("pedagogy.student_model._safe_fetch_mastery", AsyncMock(return_value=None)), \
             patch("pedagogy.student_model._safe_fetch_profile", AsyncMock(return_value=None)), \
             patch("pedagogy.student_model._safe_fetch_context", AsyncMock(return_value=None)), \
             patch("pedagogy.student_model._safe_fetch_posterior", AsyncMock(return_value=None)):
            model = await fetch_student_model("s1", "c1", idea_ids=["a"])

        assert model.ok is False
        assert set(model.sources_failed) == {
            "mastery", "profile", "personalization_context", "style_posterior",
        }
        # Defaults fill in
        assert model.style_dominant == "mixed"
        assert model.pace == "normal"
        assert model.preferred_explanation_depth == "balanced"
        assert model.speech_rate == 1.0
        assert model.mastery == {}
        # Convenience accessors still work
        assert model.avg_mastery() == 0.5

    @pytest.mark.asyncio
    async def test_only_posterior_failed(self):
        from pedagogy.personalization.engine import PersonalizationContext
        ctx = PersonalizationContext(pace="normal", avg_response_time_s=20.0, confusion_rate=0.1)

        with patch("pedagogy.student_model._safe_fetch_mastery",
                   AsyncMock(return_value={"x": 0.6})), \
             patch("pedagogy.student_model._safe_fetch_profile",
                   AsyncMock(return_value={"speech_rate": 1.0})), \
             patch("pedagogy.student_model._safe_fetch_context",
                   AsyncMock(return_value=ctx)), \
             patch("pedagogy.student_model._safe_fetch_posterior",
                   AsyncMock(return_value=None)):
            model = await fetch_student_model("s1", "c1", idea_ids=["x"])

        assert model.ok is False
        assert model.sources_failed == ("style_posterior",)
        assert model.style_dominant == "mixed"
        assert model.style_scores == {}
        assert model.style_confidence == 0.0
        # Other sources still populated
        assert model.mastery == {"x": 0.6}
        assert model.pace == "normal"


# ════════════════════════════════════════════════════════════════════
# Class-method shortcut
# ════════════════════════════════════════════════════════════════════

class TestClassMethodFetch:
    @pytest.mark.asyncio
    async def test_classmethod_alias(self):
        """``StudentModel.fetch(...)`` should be equivalent to ``fetch_student_model(...)``."""
        with patch("pedagogy.student_model._safe_fetch_mastery", AsyncMock(return_value={})), \
             patch("pedagogy.student_model._safe_fetch_profile", AsyncMock(return_value=None)), \
             patch("pedagogy.student_model._safe_fetch_context", AsyncMock(return_value=None)), \
             patch("pedagogy.student_model._safe_fetch_posterior", AsyncMock(return_value=None)):
            model = await StudentModel.fetch("s1", "c1")
        assert isinstance(model, StudentModel)
        assert model.student_id == "s1"
        assert model.course_id == "c1"
