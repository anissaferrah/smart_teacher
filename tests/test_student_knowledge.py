"""Tests for the student knowledge snapshot builder.

The snapshot joins KnowledgeGraph (course concepts) + MasteryRepo
(per-student per-idea posteriors). We mock both with simple test
doubles so the test runs without a live DB or RAG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pedagogy.student_knowledge import (
    MAX_LISTED_CONCEPTS, STRONG_THRESHOLD, WEAK_THRESHOLD,
    StudentKnowledgeSnapshot, build_snapshot, to_prompt_context,
)


# ── Test doubles ─────────────────────────────────────────────────────

@dataclass
class _FakeConcept:
    name: str
    idea_ids: set[str] = field(default_factory=set)


def _make_kg(concepts: list[_FakeConcept]):
    """Mock KG that returns the given concepts."""
    kg = MagicMock()
    kg.list_concepts.return_value = concepts
    return kg


# ════════════════════════════════════════════════════════════════════
# Cold start (no data)
# ════════════════════════════════════════════════════════════════════

class TestColdStart:
    @pytest.mark.asyncio
    async def test_missing_student_returns_empty(self):
        snap = await build_snapshot(student_id=None, course_id="ir")
        assert snap.student_id == ""
        assert snap.mastery_by_concept == {}
        assert snap.is_cold_start

    @pytest.mark.asyncio
    async def test_missing_course_returns_empty(self):
        snap = await build_snapshot(student_id="alice", course_id=None)
        assert snap.mastery_by_concept == {}
        assert snap.is_cold_start

    @pytest.mark.asyncio
    async def test_no_kg_concepts_returns_empty(self):
        kg = _make_kg([])
        snap = await build_snapshot("alice", "ir", kg=kg)
        assert snap.mastery_by_concept == {}
        assert snap.is_cold_start

    @pytest.mark.asyncio
    async def test_to_prompt_context_returns_empty_for_cold_start(self):
        snap = StudentKnowledgeSnapshot(student_id="alice", course_id="ir")
        # total_attempts is 0 by default → cold_start
        assert to_prompt_context(snap, "fr") == ""
        assert to_prompt_context(snap, "en") == ""


# ════════════════════════════════════════════════════════════════════
# Mastery aggregation
# ════════════════════════════════════════════════════════════════════

class TestMasteryAggregation:
    """Concept mastery = mean of its ideas' posteriors."""

    @pytest.mark.asyncio
    async def test_concept_score_is_mean_of_ideas(self):
        kg = _make_kg([
            _FakeConcept(name="k_means", idea_ids={"i1", "i2"}),
            _FakeConcept(name="vector_model", idea_ids={"i3"}),
        ])
        # Mocked idea-level scores
        idea_scores = {"i1": 0.8, "i2": 0.6, "i3": 0.9}
        with patch(
            "pedagogy.mastery_repo.MasteryRepo.get_scores_bulk",
            AsyncMock(return_value=idea_scores),
        ):
            snap = await build_snapshot("alice", "ir", kg=kg)
        # k_means : mean(0.8, 0.6) = 0.7
        assert snap.mastery_by_concept["k_means"] == pytest.approx(0.7)
        # vector_model : mean(0.9) = 0.9
        assert snap.mastery_by_concept["vector_model"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_concept_with_no_attempted_ideas_excluded(self):
        """If none of a concept's ideas were attempted, the concept
        is absent from mastery_by_concept (and goes into never_seen)."""
        kg = _make_kg([
            _FakeConcept(name="seen", idea_ids={"i1"}),
            _FakeConcept(name="never_seen", idea_ids={"i2", "i3"}),
        ])
        with patch(
            "pedagogy.mastery_repo.MasteryRepo.get_scores_bulk",
            AsyncMock(return_value={"i1": 0.5}),  # no entries for i2, i3
        ):
            snap = await build_snapshot("alice", "ir", kg=kg)
        assert "seen" in snap.mastery_by_concept
        assert "never_seen" not in snap.mastery_by_concept
        assert "never_seen" in snap.never_seen_concepts


# ════════════════════════════════════════════════════════════════════
# Bucket classification (strong / weak / never_seen / recently_confused)
# ════════════════════════════════════════════════════════════════════

class TestBuckets:
    @pytest.mark.asyncio
    async def test_strong_bucket_threshold(self):
        kg = _make_kg([
            _FakeConcept(name="strong",  idea_ids={"i1"}),
            _FakeConcept(name="middle",  idea_ids={"i2"}),
            _FakeConcept(name="weak",    idea_ids={"i3"}),
        ])
        with patch(
            "pedagogy.mastery_repo.MasteryRepo.get_scores_bulk",
            AsyncMock(return_value={
                "i1": STRONG_THRESHOLD + 0.1,   # strong
                "i2": 0.5,                       # middle
                "i3": WEAK_THRESHOLD - 0.1,      # weak
            }),
        ):
            snap = await build_snapshot("alice", "ir", kg=kg)
        assert "strong" in snap.strong_concepts
        assert "weak"   in snap.weak_concepts
        # middle is in neither
        assert "middle" not in snap.strong_concepts
        assert "middle" not in snap.weak_concepts

    @pytest.mark.asyncio
    async def test_weak_concepts_sorted_weakest_first(self):
        kg = _make_kg([
            _FakeConcept(name="lessbad", idea_ids={"i1"}),
            _FakeConcept(name="terrible", idea_ids={"i2"}),
        ])
        with patch(
            "pedagogy.mastery_repo.MasteryRepo.get_scores_bulk",
            AsyncMock(return_value={"i1": 0.35, "i2": 0.10}),
        ):
            snap = await build_snapshot("alice", "ir", kg=kg)
        # Weakest first → "terrible" before "lessbad"
        assert snap.weak_concepts == ["terrible", "lessbad"]

    @pytest.mark.asyncio
    async def test_strong_concepts_sorted_best_first(self):
        kg = _make_kg([
            _FakeConcept(name="great", idea_ids={"i1"}),
            _FakeConcept(name="okay",  idea_ids={"i2"}),
        ])
        with patch(
            "pedagogy.mastery_repo.MasteryRepo.get_scores_bulk",
            AsyncMock(return_value={"i1": 0.95, "i2": 0.75}),
        ):
            snap = await build_snapshot("alice", "ir", kg=kg)
        assert snap.strong_concepts == ["great", "okay"]

    @pytest.mark.asyncio
    async def test_recently_confused_uses_very_low_mastery_proxy(self):
        kg = _make_kg([
            _FakeConcept(name="bombed", idea_ids={"i1"}),
            _FakeConcept(name="meh",    idea_ids={"i2"}),
        ])
        with patch(
            "pedagogy.mastery_repo.MasteryRepo.get_scores_bulk",
            AsyncMock(return_value={
                "i1": 0.20,   # ≤ 0.30 → recently confused
                "i2": 0.40,   # > 0.30 → not in recently_confused
            }),
        ):
            snap = await build_snapshot("alice", "ir", kg=kg)
        assert "bombed" in snap.recently_confused
        assert "meh" not in snap.recently_confused

    @pytest.mark.asyncio
    async def test_listed_concepts_capped(self):
        # Build 20 weak concepts ; the snapshot should cap at MAX_LISTED_CONCEPTS
        concepts = [_FakeConcept(name=f"c{i}", idea_ids={f"i{i}"}) for i in range(20)]
        scores = {f"i{i}": 0.1 for i in range(20)}
        kg = _make_kg(concepts)
        with patch(
            "pedagogy.mastery_repo.MasteryRepo.get_scores_bulk",
            AsyncMock(return_value=scores),
        ):
            snap = await build_snapshot("alice", "ir", kg=kg)
        assert len(snap.weak_concepts) == MAX_LISTED_CONCEPTS


# ════════════════════════════════════════════════════════════════════
# Cold-start guard on prompt context
# ════════════════════════════════════════════════════════════════════

class TestPromptContext:
    @pytest.mark.asyncio
    async def test_french_prompt_has_internal_markers(self):
        kg = _make_kg([
            _FakeConcept(name="k_means", idea_ids={"i1"}),
            _FakeConcept(name="vector",  idea_ids={"i2"}),
        ])
        # Lots of attempts so we're not in cold-start
        scores = {f"i{i}": 0.8 for i in range(1, 20)}
        scores["i1"] = 0.85
        scores["i2"] = 0.30  # weak
        kg.list_concepts.return_value.extend(
            [_FakeConcept(name=f"c{i}", idea_ids={f"i{i}"}) for i in range(3, 20)]
        )
        with patch(
            "pedagogy.mastery_repo.MasteryRepo.get_scores_bulk",
            AsyncMock(return_value=scores),
        ):
            snap = await build_snapshot("alice", "ir", kg=kg)
        prompt = to_prompt_context(snap, "fr")
        assert "<<INSTRUCTION_INTERNE_NE_PAS_MENTIONNER" in prompt
        assert "<<FIN_INSTRUCTION_INTERNE>>" in prompt
        assert "État pédagogique" in prompt

    def test_cold_start_returns_empty_string(self):
        snap = StudentKnowledgeSnapshot(
            student_id="alice", course_id="ir",
            total_attempts=2,  # below cold-start threshold
            strong_concepts=["x"],
        )
        assert to_prompt_context(snap, "fr") == ""
        assert to_prompt_context(snap, "en") == ""

    def test_english_prompt_has_internal_markers(self):
        snap = StudentKnowledgeSnapshot(
            student_id="alice", course_id="ir",
            total_attempts=20,
            strong_concepts=["k_means"],
            weak_concepts=["inverted_index"],
        )
        prompt = to_prompt_context(snap, "en")
        assert "<<INTERNAL_INSTRUCTION_DO_NOT_MENTION" in prompt
        assert "<<END_INTERNAL_INSTRUCTION>>" in prompt
        assert "MASTERED" in prompt
        assert "WEAK" in prompt

    def test_no_signals_returns_empty(self):
        """Snapshot with total_attempts > 5 but ALL bucket lists empty
        → no useful prompt → empty string."""
        snap = StudentKnowledgeSnapshot(
            student_id="alice", course_id="ir",
            total_attempts=20,
            mastery_by_concept={"middle": 0.5},
            # All bucket lists empty
        )
        assert to_prompt_context(snap, "fr") == ""
