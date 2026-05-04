"""Tests for the engagement scoring module."""
from __future__ import annotations

import pytest

from pedagogy.engagement import (
    CONFUSION_OPTIMAL, FULLY_STALE_S, RECENT_FULL_S,
    EngagementSignals, compute_engagement,
)


# ════════════════════════════════════════════════════════════════════
# Per-signal sub-scores via end-to-end engagement (easier to read than
# importing private helpers and avoids over-binding to internals).
# ════════════════════════════════════════════════════════════════════

class TestRecency:

    def test_recent_interaction_high_score(self):
        sig = EngagementSignals(
            seconds_since_last_interaction=5.0,
            session_age_s=300.0,
        )
        result = compute_engagement(sig)
        assert result.score >= 0.5

    def test_idle_for_15min_low_score(self):
        sig = EngagementSignals(
            seconds_since_last_interaction=900.0,  # > FULLY_STALE_S
            session_age_s=900.0,
        )
        result = compute_engagement(sig)
        # Recency contribution should be 0, but other neutral signals
        # (no questions, no confusions...) push toward 0.5 — overall
        # should be in disengaged or neutral
        assert result.score < 0.6
        # And combined with passive_slides ≥ 3, fully disengaged
        sig.consecutive_passive_slides = 5
        result2 = compute_engagement(sig)
        assert result2.score < result.score

    def test_at_threshold_recent_full(self):
        sig = EngagementSignals(
            seconds_since_last_interaction=RECENT_FULL_S,
            session_age_s=RECENT_FULL_S,
        )
        # Recency exactly at the boundary should still be full credit
        result = compute_engagement(sig)
        # Other signals are all None → neutral → score depends mostly on recency
        assert result.score >= 0.5


class TestQuestionRate:

    def test_no_questions_in_long_session_low(self):
        sig = EngagementSignals(
            questions_in_session=0,
            session_age_s=600.0,  # 10 min
        )
        result = compute_engagement(sig)
        # questions=0 → 0.0 score on that axis ; combined with neutrals
        # should still be on the low side
        assert result.score < 0.55

    def test_healthy_question_rate_high(self):
        sig = EngagementSignals(
            questions_in_session=5,
            session_age_s=600.0,  # 10 min → 0.5 q/min rate
        )
        result = compute_engagement(sig)
        # questions axis = 1.0, others neutral
        assert result.score > 0.55

    def test_too_many_questions_caps(self):
        sig = EngagementSignals(
            questions_in_session=100,
            session_age_s=600.0,  # 10 q/min rate — overwhelmed
        )
        result = compute_engagement(sig)
        # Overwhelmed should NOT yield max engagement
        assert result.score < 0.7

    def test_fresh_session_one_question_credit(self):
        """Session < 60s with at least one question gets credit."""
        sig = EngagementSignals(
            questions_in_session=1,
            session_age_s=20.0,
        )
        result = compute_engagement(sig)
        assert result.score > 0.55


class TestConfusion:

    def test_zero_confusion_neutral_not_high(self):
        """Zero confusion is ambiguous : could be mastery, could be
        disengagement. Don't reward it like a high score."""
        sig = EngagementSignals(confusions_in_session=0)
        result = compute_engagement(sig)
        # confusion contributes 0.5 (neutral)
        assert 0.40 <= result.score <= 0.60

    def test_optimal_confusion_high(self):
        sig = EngagementSignals(confusions_in_session=CONFUSION_OPTIMAL)
        result = compute_engagement(sig)
        # confusion axis = 1.0
        assert result.score > 0.55

    def test_too_much_confusion_low(self):
        sig = EngagementSignals(confusions_in_session=10)
        result = compute_engagement(sig)
        # confusion axis = 0.3, others neutral → score < 0.5
        assert result.score < 0.50


class TestPassiveReader:

    def test_no_passive_full_credit(self):
        sig = EngagementSignals(consecutive_passive_slides=0)
        result = compute_engagement(sig)
        assert result.score > 0.55

    def test_three_passive_slides_zero_credit(self):
        sig = EngagementSignals(consecutive_passive_slides=3)
        result = compute_engagement(sig)
        # passive axis = 0, others neutral → score < 0.50
        assert result.score < 0.50

    def test_one_passive_slide_partial(self):
        sig = EngagementSignals(consecutive_passive_slides=1)
        result = compute_engagement(sig)
        # passive axis = 0.7
        assert 0.45 <= result.score <= 0.60


class TestInterruptLatency:

    def test_fast_interrupt_high_credit(self):
        sig = EngagementSignals(last_interrupt_latency_ms=100)
        result = compute_engagement(sig)
        assert result.score > 0.50

    def test_slow_interrupt_low_credit(self):
        sig = EngagementSignals(last_interrupt_latency_ms=3000)
        result = compute_engagement(sig)
        assert result.score < 0.55


# ════════════════════════════════════════════════════════════════════
# Combined scenarios
# ════════════════════════════════════════════════════════════════════

class TestCombined:

    def test_engaged_student(self):
        """Recently interacting, asks questions, mild confusion, no passive
        slides, fast interrupts → strongly engaged."""
        sig = EngagementSignals(
            seconds_since_last_interaction=10.0,
            questions_in_session=5,
            confusions_in_session=2,
            consecutive_passive_slides=0,
            last_interrupt_latency_ms=120,
            session_age_s=600.0,
        )
        result = compute_engagement(sig)
        assert result.score >= 0.65
        assert result.label == "engaged"

    def test_disengaged_student(self):
        """Idle for 15min, no questions, 6 confusions, 4 passive slides
        → fully disengaged."""
        sig = EngagementSignals(
            seconds_since_last_interaction=900.0,
            questions_in_session=0,
            confusions_in_session=6,
            consecutive_passive_slides=4,
            last_interrupt_latency_ms=2500,
            session_age_s=1500.0,
        )
        result = compute_engagement(sig)
        assert result.score < 0.30
        assert result.label == "disengaged"

    def test_neutral_student(self):
        """Mid-range across the board — slightly idle, no questions,
        no confusions, no passive slides."""
        sig = EngagementSignals(
            seconds_since_last_interaction=300.0,  # 5 min idle (mid-decay)
            questions_in_session=0,
            confusions_in_session=0,
            consecutive_passive_slides=2,
            last_interrupt_latency_ms=1500,
            session_age_s=600.0,
        )
        result = compute_engagement(sig)
        assert 0.30 <= result.score < 0.65
        assert result.label == "neutral"

    def test_all_none_signals_returns_neutral(self):
        """Cold start : no signals at all. Must not crash, must
        return ~ 0.5 (neutral)."""
        sig = EngagementSignals()
        result = compute_engagement(sig)
        # All sub-scores default to 0.5 → weighted sum = 0.5
        assert 0.45 <= result.score <= 0.55
        assert result.label == "neutral"


# ════════════════════════════════════════════════════════════════════
# Output bounds + labels
# ════════════════════════════════════════════════════════════════════

class TestBoundsAndLabels:
    """Score must always be in [0, 1] regardless of input. Labels
    must match the documented thresholds (0.30, 0.65)."""

    @pytest.mark.parametrize("low_signals", [
        EngagementSignals(seconds_since_last_interaction=99999.0,
                          questions_in_session=0,
                          confusions_in_session=20,
                          consecutive_passive_slides=10,
                          last_interrupt_latency_ms=99999.0,
                          session_age_s=99999.0),
    ])
    def test_score_never_below_zero(self, low_signals):
        result = compute_engagement(low_signals)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.parametrize("high_signals", [
        EngagementSignals(seconds_since_last_interaction=0.0,
                          questions_in_session=3,
                          confusions_in_session=2,
                          consecutive_passive_slides=0,
                          last_interrupt_latency_ms=50,
                          session_age_s=600.0),
    ])
    def test_score_never_above_one(self, high_signals):
        result = compute_engagement(high_signals)
        assert 0.0 <= result.score <= 1.0

    def test_label_thresholds(self):
        """Synthetic engineered scores at boundaries 0.30 / 0.65."""
        # Disengaged: all weights pushed down
        sig_low = EngagementSignals(
            seconds_since_last_interaction=900.0,
            questions_in_session=0,
            confusions_in_session=10,
            consecutive_passive_slides=5,
            last_interrupt_latency_ms=3000,
            session_age_s=900.0,
        )
        assert compute_engagement(sig_low).label == "disengaged"

        sig_high = EngagementSignals(
            seconds_since_last_interaction=5.0,
            questions_in_session=4,
            confusions_in_session=2,
            consecutive_passive_slides=0,
            last_interrupt_latency_ms=80,
            session_age_s=480.0,
        )
        assert compute_engagement(sig_high).label == "engaged"
