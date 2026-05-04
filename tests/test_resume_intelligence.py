"""Tests for the resume-intelligence policy.

Six intents, five strategies, deterministic mapping. We verify :

  - Each intent fires on the right (cursor, pause) combination.
  - The intent → strategy map covers all 6 intents.
  - The thresholds (10s / 60s / 3min / 30s grace) gate correctly at
    the boundary values (e.g. exactly 60s = NORMAL not GAP).
  - Defensive paths : missing pause duration, negative cursor.
"""
from __future__ import annotations

import pytest

from pedagogy.resume_intelligence import (
    QUICK_PAUSE_S, NORMAL_PAUSE_S, LONG_PAUSE_S, SLIDE_DONE_GRACE_S,
    ResumeContext, ResumeIntent, ResumeStrategy,
    compose_resume_action, decide_resume_strategy, detect_resume_intent,
)


# ════════════════════════════════════════════════════════════════════
# Mid-narration intents (cursor < narration_len)
# ════════════════════════════════════════════════════════════════════

class TestMidNarrationIntents:
    """When the cursor is mid-narration, the pause bucket alone
    decides the intent."""

    def test_quick_pause_5s(self):
        ctx = ResumeContext(cursor=300, narration_len=689, interruption_duration_s=5.0)
        assert detect_resume_intent(ctx) == ResumeIntent.QUICK_RESUME

    def test_normal_pause_30s(self):
        ctx = ResumeContext(cursor=300, narration_len=689, interruption_duration_s=30.0)
        assert detect_resume_intent(ctx) == ResumeIntent.NORMAL_RESUME

    def test_gap_pause_90s(self):
        ctx = ResumeContext(cursor=300, narration_len=689, interruption_duration_s=90.0)
        assert detect_resume_intent(ctx) == ResumeIntent.RESUME_AFTER_GAP

    def test_long_pause_5min(self):
        ctx = ResumeContext(cursor=300, narration_len=689, interruption_duration_s=300.0)
        assert detect_resume_intent(ctx) == ResumeIntent.RESUME_AFTER_LONG


# ════════════════════════════════════════════════════════════════════
# End-of-slide intents (cursor >= narration_len)
# ════════════════════════════════════════════════════════════════════

class TestEndOfSlideIntents:
    """When cursor reaches the end, short pause = slide done, long
    pause = student wants to review."""

    def test_short_pause_at_end_means_slide_done(self):
        ctx = ResumeContext(cursor=689, narration_len=689, interruption_duration_s=10.0)
        assert detect_resume_intent(ctx) == ResumeIntent.SLIDE_COMPLETED

    def test_long_pause_at_end_means_review(self):
        ctx = ResumeContext(cursor=689, narration_len=689, interruption_duration_s=120.0)
        assert detect_resume_intent(ctx) == ResumeIntent.REVIEW_REQUEST

    def test_cursor_past_end_treated_as_at_end(self):
        ctx = ResumeContext(cursor=999, narration_len=689, interruption_duration_s=5.0)
        assert detect_resume_intent(ctx) == ResumeIntent.SLIDE_COMPLETED


# ════════════════════════════════════════════════════════════════════
# Threshold boundaries
# ════════════════════════════════════════════════════════════════════

class TestBoundaries:
    """Exact-boundary tests : the thresholds must be unambiguous."""

    def test_exactly_quick_threshold_lands_in_normal(self):
        # < 10s = quick, >= 10s = normal
        ctx = ResumeContext(cursor=100, narration_len=500, interruption_duration_s=QUICK_PAUSE_S)
        assert detect_resume_intent(ctx) == ResumeIntent.NORMAL_RESUME

    def test_just_below_quick_threshold_is_quick(self):
        ctx = ResumeContext(cursor=100, narration_len=500, interruption_duration_s=QUICK_PAUSE_S - 0.1)
        assert detect_resume_intent(ctx) == ResumeIntent.QUICK_RESUME

    def test_exactly_normal_threshold_lands_in_gap(self):
        ctx = ResumeContext(cursor=100, narration_len=500, interruption_duration_s=NORMAL_PAUSE_S)
        assert detect_resume_intent(ctx) == ResumeIntent.RESUME_AFTER_GAP

    def test_exactly_long_threshold_is_long(self):
        ctx = ResumeContext(cursor=100, narration_len=500, interruption_duration_s=LONG_PAUSE_S)
        assert detect_resume_intent(ctx) == ResumeIntent.RESUME_AFTER_LONG

    def test_grace_boundary_at_end(self):
        # At end + < grace → SLIDE_COMPLETED ; >= grace → REVIEW_REQUEST
        ctx = ResumeContext(cursor=500, narration_len=500, interruption_duration_s=SLIDE_DONE_GRACE_S - 1)
        assert detect_resume_intent(ctx) == ResumeIntent.SLIDE_COMPLETED
        ctx2 = ResumeContext(cursor=500, narration_len=500, interruption_duration_s=SLIDE_DONE_GRACE_S)
        assert detect_resume_intent(ctx2) == ResumeIntent.REVIEW_REQUEST


# ════════════════════════════════════════════════════════════════════
# Intent → strategy mapping (all 6 intents covered)
# ════════════════════════════════════════════════════════════════════

class TestIntentToStrategy:

    @pytest.mark.parametrize("intent,expected_strategy", [
        (ResumeIntent.QUICK_RESUME,       ResumeStrategy.CONTINUE),
        (ResumeIntent.NORMAL_RESUME,      ResumeStrategy.REWIND_SENTENCE),
        # Medium / long pauses now route to REEXPLAIN_AND_CONTINUE
        # (LLM rephrases the current sentence then continues).
        (ResumeIntent.RESUME_AFTER_GAP,   ResumeStrategy.REEXPLAIN_AND_CONTINUE),
        (ResumeIntent.RESUME_AFTER_LONG,  ResumeStrategy.REEXPLAIN_AND_CONTINUE),
        (ResumeIntent.SLIDE_COMPLETED,    ResumeStrategy.SKIP_TO_NEXT),
        (ResumeIntent.REVIEW_REQUEST,     ResumeStrategy.REWIND_SENTENCE),
    ])
    def test_each_intent_maps_to_expected_strategy(self, intent, expected_strategy):
        assert decide_resume_strategy(intent) == expected_strategy

    def test_all_intents_have_a_strategy(self):
        """Every defined ResumeIntent must produce a strategy — guards
        against future intents being added without updating the map."""
        for intent in ResumeIntent:
            strategy = decide_resume_strategy(intent)
            assert isinstance(strategy, ResumeStrategy), (
                f"intent {intent} returned non-strategy {strategy!r}"
            )


# ════════════════════════════════════════════════════════════════════
# Defensive : missing or pathological inputs
# ════════════════════════════════════════════════════════════════════

class TestDefensive:
    def test_no_pause_duration_treated_as_normal(self):
        """Caller forgot to pass the duration → fall back to normal
        bucket (safe middle ground), NOT to a guessed extreme."""
        ctx = ResumeContext(cursor=300, narration_len=689, interruption_duration_s=None)
        assert detect_resume_intent(ctx) == ResumeIntent.NORMAL_RESUME

    def test_negative_duration_treated_as_normal(self):
        ctx = ResumeContext(cursor=300, narration_len=689, interruption_duration_s=-5.0)
        assert detect_resume_intent(ctx) == ResumeIntent.NORMAL_RESUME

    def test_zero_narration_len_does_not_short_circuit_to_completed(self):
        """A zero-length narration would falsely trigger the
        cursor>=len branch ; we explicitly require narration_len>0
        for the end-of-slide path."""
        ctx = ResumeContext(cursor=0, narration_len=0, interruption_duration_s=5.0)
        # With narration_len=0 the cursor>=len branch is skipped, falls
        # to mid-narration — intent depends on bucket only.
        intent = detect_resume_intent(ctx)
        assert intent == ResumeIntent.QUICK_RESUME


# ════════════════════════════════════════════════════════════════════
# compose_resume_action — end-to-end + reason string
# ════════════════════════════════════════════════════════════════════

class TestComposeResumeAction:
    def test_composes_intent_strategy_and_reason(self):
        ctx = ResumeContext(cursor=300, narration_len=689, interruption_duration_s=90.0)
        action = compose_resume_action(ctx)
        assert action.intent == ResumeIntent.RESUME_AFTER_GAP
        # New mapping : medium pauses now route to REEXPLAIN_AND_CONTINUE
        assert action.strategy == ResumeStrategy.REEXPLAIN_AND_CONTINUE
        assert "intent=resume_after_gap" in action.reason
        assert "strategy=reexplain_and_continue" in action.reason
        assert "300/689" in action.reason

    def test_reason_handles_missing_pause(self):
        ctx = ResumeContext(cursor=100, narration_len=500, interruption_duration_s=None)
        action = compose_resume_action(ctx)
        # Should not crash, and the reason should indicate the missing value
        assert "?" in action.reason


# ════════════════════════════════════════════════════════════════════
# The user's actual scenarios
# ════════════════════════════════════════════════════════════════════

class TestUserScenarios:

    def test_quick_phone_check(self):
        """Student looks at their phone for 5s, comes back. Don't
        interrupt the flow with a recap — continue."""
        ctx = ResumeContext(cursor=400, narration_len=689, interruption_duration_s=5.0)
        action = compose_resume_action(ctx)
        assert action.strategy == ResumeStrategy.CONTINUE

    def test_question_pause_30s(self):
        """Student paused to ask a question, the Q&A took 30s, now
        resuming — replay the current sentence so they pick up the
        idea, no need for a recap."""
        ctx = ResumeContext(cursor=400, narration_len=689, interruption_duration_s=30.0)
        action = compose_resume_action(ctx)
        assert action.strategy == ResumeStrategy.REWIND_SENTENCE

    def test_long_absence(self):
        """Student left the room for 5 minutes. The LLM re-explains the
        sentence in different words then continues."""
        ctx = ResumeContext(cursor=400, narration_len=689, interruption_duration_s=300.0)
        action = compose_resume_action(ctx)
        assert action.strategy == ResumeStrategy.REEXPLAIN_AND_CONTINUE

    def test_slide_finished_quick_resume(self):
        """Audio finished, student paused right after, resumes 3s
        later — slide is truly done, advance."""
        ctx = ResumeContext(cursor=689, narration_len=689, interruption_duration_s=3.0)
        action = compose_resume_action(ctx)
        assert action.strategy == ResumeStrategy.SKIP_TO_NEXT

    def test_slide_finished_review_request(self):
        """Audio finished, student lingers 1 minute (didn't move
        on) — wants to review the last point."""
        ctx = ResumeContext(cursor=689, narration_len=689, interruption_duration_s=60.0)
        action = compose_resume_action(ctx)
        assert action.strategy == ResumeStrategy.REWIND_SENTENCE
