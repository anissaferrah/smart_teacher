"""Tests for the multi-signal confusion fusion.

Two layers :
  1. ``score_from_prosody`` : prosody dict → [0, 1] score
  2. ``fuse_confusion_signals`` : (text_score, prosody_dict) → FusedConfusion

Both are pure ; no DB / Redis / LLM mocks needed.
"""
from __future__ import annotations

import pytest

from pedagogy.confusion.fusion import (
    W_PROSODY, W_TEXT,
    ConfusionSignals, FusedConfusion,
    fuse_confusion_signals, score_from_prosody,
)


# ════════════════════════════════════════════════════════════════════
# Prosody → score
# ════════════════════════════════════════════════════════════════════

class TestScoreFromProsody:

    def test_no_markers_zero_score(self):
        assert score_from_prosody({"markers": []}) == 0.0

    def test_one_marker_mild(self):
        assert score_from_prosody({"markers": ["slow_speech_rate"]}) == 0.30

    def test_two_markers_strong(self):
        out = score_from_prosody({"markers": ["slow_speech_rate", "frequent_hesitations"]})
        assert out == 0.70

    def test_three_markers_max(self):
        out = score_from_prosody({"markers": [
            "slow_speech_rate", "frequent_hesitations", "high_silence_ratio",
        ]})
        assert out == 1.0

    def test_more_than_three_markers_capped_at_one(self):
        """Defensive : if upstream ever emits 4+ markers, score caps at 1.0."""
        out = score_from_prosody({"markers": ["a", "b", "c", "d", "e"]})
        assert out == 1.0

    def test_missing_prosody_returns_zero(self):
        assert score_from_prosody(None) == 0.0
        assert score_from_prosody({}) == 0.0
        assert score_from_prosody({"speech_rate": 100}) == 0.0  # no markers key

    def test_malformed_markers_field(self):
        # markers should be a list ; anything else is ignored
        assert score_from_prosody({"markers": "not-a-list"}) == 0.0
        assert score_from_prosody({"markers": None}) == 0.0


# ════════════════════════════════════════════════════════════════════
# Fusion : weighted sum + renormalisation
# ════════════════════════════════════════════════════════════════════

class TestFusion:

    def test_text_only_no_lift_or_drop(self):
        """Only SIGHT available → fused score = SIGHT score, no
        spurious adjustment from missing prosody."""
        sig = ConfusionSignals(text_score=0.9, prosody_dict=None)
        result = fuse_confusion_signals(sig)
        assert result.score == 0.9
        assert "text(sight)" in result.contributors
        assert "prosody" not in result.contributors

    def test_prosody_only(self):
        """Edge case : SIGHT not available, prosody is the only signal.
        Fused score = prosody score directly."""
        sig = ConfusionSignals(
            text_score=None,
            prosody_dict={"markers": ["slow_speech_rate", "frequent_hesitations"]},
        )
        result = fuse_confusion_signals(sig)
        assert result.score == 0.70

    def test_both_signals_agree_high(self):
        """Both SIGHT and prosody say confused → high fused score."""
        sig = ConfusionSignals(
            text_score=0.8,
            prosody_dict={"markers": ["slow_speech_rate", "frequent_hesitations"]},
        )
        result = fuse_confusion_signals(sig)
        # 0.7 * 0.8 + 0.3 * 0.7 = 0.56 + 0.21 = 0.77
        assert abs(result.score - 0.77) < 0.005

    def test_both_signals_agree_clean(self):
        sig = ConfusionSignals(
            text_score=0.05,
            prosody_dict={"markers": []},
        )
        result = fuse_confusion_signals(sig)
        # 0.7 * 0.05 + 0.3 * 0 = 0.035
        assert result.score < 0.10

    def test_disagreement_text_high_prosody_low(self):
        """SIGHT says confused, voice sounds clean. SIGHT's vote wins
        because it has 70 % of the weight + it's calibrated."""
        sig = ConfusionSignals(text_score=0.9, prosody_dict={"markers": []})
        result = fuse_confusion_signals(sig)
        # 0.7 * 0.9 + 0.3 * 0 = 0.63
        assert result.score > 0.50  # still confidently confused
        assert abs(result.score - 0.63) < 0.005

    def test_disagreement_text_low_prosody_high(self):
        """SIGHT says clean but voice is hesitant. Prosody alone
        doesn't tip it past 0.5 — by design, vocal evidence isn't
        enough to override clean text."""
        sig = ConfusionSignals(
            text_score=0.1,
            prosody_dict={"markers": [
                "slow_speech_rate", "frequent_hesitations", "high_silence_ratio",
            ]},
        )
        result = fuse_confusion_signals(sig)
        # 0.7 * 0.1 + 0.3 * 1.0 = 0.07 + 0.30 = 0.37
        assert abs(result.score - 0.37) < 0.005
        assert result.score < 0.50

    def test_no_signals_returns_safe_zero(self):
        sig = ConfusionSignals(text_score=None, prosody_dict=None)
        result = fuse_confusion_signals(sig)
        assert result.score == 0.0
        assert result.contributors == []

    def test_score_clamped_to_unit_interval(self):
        """If SIGHT returns out-of-range (1.5 / -0.2), fused stays in [0, 1]."""
        sig_high = ConfusionSignals(text_score=1.5, prosody_dict=None)
        assert fuse_confusion_signals(sig_high).score <= 1.0

        sig_low = ConfusionSignals(text_score=-0.2, prosody_dict=None)
        assert fuse_confusion_signals(sig_low).score >= 0.0


# ════════════════════════════════════════════════════════════════════
# Smart-Teacher scenarios
# ════════════════════════════════════════════════════════════════════

class TestRealisticScenarios:
    """End-to-end scenarios the Smart Teacher should handle correctly."""

    def test_clean_question_short_audio(self):
        """Student asks 'What is K-means ?' clearly. SIGHT clean,
        no vocal hesitation. Fused score should be near zero."""
        sig = ConfusionSignals(
            text_score=0.05,  # SIGHT confident-clean
            prosody_dict={
                "speech_rate": 140.0,
                "hesitation_count": 0,
                "silence_ratio": 0.1,
                "markers": [],
            },
        )
        result = fuse_confusion_signals(sig)
        assert result.score < 0.10

    def test_hesitant_long_question_text_clean(self):
        """Student asks a clean question but stumbles vocally :
        'um... how does K-means... uh... compute distances ?'
        SIGHT might miss this (text reads as a normal question)
        but prosody catches it. The fused score lifts but doesn't
        confidently flag — exactly what we want : a soft signal."""
        sig = ConfusionSignals(
            text_score=0.20,   # SIGHT mostly clean
            prosody_dict={"markers": ["frequent_hesitations", "high_silence_ratio"]},
        )
        result = fuse_confusion_signals(sig)
        # 0.7 * 0.2 + 0.3 * 0.7 = 0.14 + 0.21 = 0.35
        # → ambiguous, neither confident-clean nor confident-confused
        assert 0.30 <= result.score <= 0.45

    def test_strong_consensus_confused(self):
        """Both modalities scream confusion : confused text + multiple
        vocal markers. Fused score should be very high."""
        sig = ConfusionSignals(
            text_score=0.95,
            prosody_dict={"markers": [
                "slow_speech_rate", "frequent_hesitations", "high_silence_ratio",
            ]},
        )
        result = fuse_confusion_signals(sig)
        # 0.7 * 0.95 + 0.3 * 1.0 = 0.665 + 0.30 = 0.965
        assert result.score >= 0.95


# ════════════════════════════════════════════════════════════════════
# Output structure (FusedConfusion dataclass) sanity
# ════════════════════════════════════════════════════════════════════

class TestOutputStructure:
    def test_reason_string_includes_components(self):
        sig = ConfusionSignals(text_score=0.5, prosody_dict={"markers": ["x"]})
        result = fuse_confusion_signals(sig)
        # Reason should mention text score, prosody score, and contributors
        assert "text=" in result.reason
        assert "prosody=" in result.reason
        assert "fused=" in result.reason

    def test_contributors_track_what_was_used(self):
        sig = ConfusionSignals(text_score=0.5, prosody_dict={"markers": []})
        result = fuse_confusion_signals(sig)
        assert "text(sight)" in result.contributors
        assert "prosody" in result.contributors

    def test_weights_sum_to_one(self):
        """If someone overrides the weights at module level the
        fail-fast assertion in fusion.py protects us."""
        assert abs((W_TEXT + W_PROSODY) - 1.0) < 1e-6
