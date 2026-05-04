"""Tests for the REEXPLAIN_AND_CONTINUE resume strategy.

Two layers :
  1. ``current_sentence_span`` : pure helper that returns (start, end,
     text) of the sentence containing a given cursor.
  2. ``decide_resume_strategy`` mapping for medium / long pauses now
     points at REEXPLAIN_AND_CONTINUE — verify the routing.
"""
from __future__ import annotations

import pytest

from pedagogy.resume_intelligence import (
    ResumeContext, ResumeIntent, ResumeStrategy,
    decide_resume_strategy, detect_resume_intent,
)
from services.presentation import current_sentence_span


# ════════════════════════════════════════════════════════════════════
# current_sentence_span — pure helper
# ════════════════════════════════════════════════════════════════════

class TestCurrentSentenceSpan:

    NARRATION = (
        "Premier sujet abordé. "          # span 0..21
        "Deuxième idée importante. "       # span 21..47
        "Troisième et dernière phrase."    # span 47..76
    )

    def test_cursor_in_first_sentence(self):
        # cursor at "sujet"
        start, end, text = current_sentence_span(self.NARRATION, 8)
        assert start == 0
        assert text.startswith("Premier sujet")
        # end should land right after "Premier sujet abordé. "
        assert end > start
        assert "Deuxième" not in text

    def test_cursor_in_second_sentence(self):
        # cursor at "Deuxième"
        cursor = self.NARRATION.find("Deuxième") + 3
        start, end, text = current_sentence_span(self.NARRATION, cursor)
        assert text.startswith("Deuxième")
        assert "Troisième" not in text

    def test_cursor_in_last_sentence(self):
        cursor = self.NARRATION.find("dernière")
        start, end, text = current_sentence_span(self.NARRATION, cursor)
        assert text.startswith("Troisième")
        # No further terminator → end is end-of-narration
        assert end == len(self.NARRATION)

    def test_cursor_at_boundary(self):
        """Cursor exactly at the start of a sentence — span should
        cover that sentence, not the previous one."""
        cursor = self.NARRATION.find("Deuxième")
        start, end, text = current_sentence_span(self.NARRATION, cursor)
        assert text.startswith("Deuxième")

    def test_empty_narration(self):
        assert current_sentence_span("", 0) == (0, 0, "")

    def test_cursor_negative(self):
        start, end, text = current_sentence_span(self.NARRATION, -10)
        assert start == 0
        assert text.startswith("Premier")

    def test_cursor_past_end(self):
        start, end, text = current_sentence_span(self.NARRATION, 9999)
        # Falls back to the last sentence
        assert "Troisième" in text or "dernière" in text

    def test_no_terminator(self):
        text_no_punct = "no punctuation here at all"
        start, end, text = current_sentence_span(text_no_punct, 5)
        assert start == 0
        assert end == len(text_no_punct)
        assert text.strip() == text_no_punct


# ════════════════════════════════════════════════════════════════════
# Strategy mapping : medium / long pauses → REEXPLAIN_AND_CONTINUE
# ════════════════════════════════════════════════════════════════════

class TestStrategyMapping:
    def test_resume_after_gap_now_uses_reexplain(self):
        """60s ≤ pause < 3min → REEXPLAIN, not REWIND_SENTENCE_RECAP."""
        assert decide_resume_strategy(ResumeIntent.RESUME_AFTER_GAP) == \
            ResumeStrategy.REEXPLAIN_AND_CONTINUE

    def test_resume_after_long_now_uses_reexplain(self):
        """≥ 3min → REEXPLAIN, not REWIND_SENTENCE_LONG_RECAP."""
        assert decide_resume_strategy(ResumeIntent.RESUME_AFTER_LONG) == \
            ResumeStrategy.REEXPLAIN_AND_CONTINUE

    def test_quick_resume_unchanged(self):
        """Quick pause (<10s) keeps CONTINUE — no LLM cost."""
        assert decide_resume_strategy(ResumeIntent.QUICK_RESUME) == \
            ResumeStrategy.CONTINUE

    def test_normal_resume_unchanged(self):
        """Normal pause (10-60s) keeps REWIND_SENTENCE — fast verbatim replay."""
        assert decide_resume_strategy(ResumeIntent.NORMAL_RESUME) == \
            ResumeStrategy.REWIND_SENTENCE

    def test_slide_completed_unchanged(self):
        assert decide_resume_strategy(ResumeIntent.SLIDE_COMPLETED) == \
            ResumeStrategy.SKIP_TO_NEXT


# ════════════════════════════════════════════════════════════════════
# End-to-end : pause durations route correctly
# ════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    @pytest.mark.parametrize("duration_s,expected_strategy", [
        (5.0,    ResumeStrategy.CONTINUE),                  # quick
        (30.0,   ResumeStrategy.REWIND_SENTENCE),            # normal
        (90.0,   ResumeStrategy.REEXPLAIN_AND_CONTINUE),     # gap
        (300.0,  ResumeStrategy.REEXPLAIN_AND_CONTINUE),     # long
    ])
    def test_pause_duration_routes_correctly(self, duration_s, expected_strategy):
        ctx = ResumeContext(
            cursor=300, narration_len=689,
            interruption_duration_s=duration_s,
        )
        intent = detect_resume_intent(ctx)
        strategy = decide_resume_strategy(intent)
        assert strategy == expected_strategy, (
            f"pause={duration_s}s should route to {expected_strategy.value}, "
            f"got {strategy.value} via intent={intent.value}"
        )
