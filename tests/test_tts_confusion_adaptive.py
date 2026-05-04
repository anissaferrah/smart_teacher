"""Tests for the confusion-driven TTS slowdown.

The adaptive rate is computed by ``confusion_rate_multiplier`` (pure)
and integrated in ``get_edge_tts_rate_with_bandit`` (composes profile
× bandit × confusion). We test both layers.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pedagogy.personalization.tts_adapter import (
    CONFUSION_TTS_SLOWDOWN_FACTOR, CONFUSION_TTS_SLOWDOWN_THRESHOLD,
    confusion_rate_multiplier, get_edge_tts_rate_with_bandit,
    rate_float_to_edge_str,
)


# ════════════════════════════════════════════════════════════════════
# confusion_rate_multiplier — stepped, predictable
# ════════════════════════════════════════════════════════════════════

class TestConfusionRateMultiplier:

    def test_low_score_no_slowdown(self):
        assert confusion_rate_multiplier(0.0) == 1.0
        assert confusion_rate_multiplier(0.3) == 1.0
        assert confusion_rate_multiplier(0.59) == 1.0

    def test_at_threshold_slowdown_kicks_in(self):
        assert confusion_rate_multiplier(CONFUSION_TTS_SLOWDOWN_THRESHOLD) == CONFUSION_TTS_SLOWDOWN_FACTOR

    def test_high_score_full_slowdown(self):
        assert confusion_rate_multiplier(0.7) == CONFUSION_TTS_SLOWDOWN_FACTOR
        assert confusion_rate_multiplier(0.95) == CONFUSION_TTS_SLOWDOWN_FACTOR

    def test_none_no_change(self):
        assert confusion_rate_multiplier(None) == 1.0

    def test_non_numeric_no_change(self):
        assert confusion_rate_multiplier("high") == 1.0
        assert confusion_rate_multiplier({"score": 0.8}) == 1.0


# ════════════════════════════════════════════════════════════════════
# Integration : profile × bandit × confusion
# ════════════════════════════════════════════════════════════════════

class TestTtsRateComposition:
    """The Edge-TTS rate string composes 3 layers : profile preference,
    bandit-chosen category, confusion slowdown."""

    @pytest.mark.asyncio
    async def test_clean_turn_no_slowdown(self):
        """No bandit category, no confusion → rate = profile rate."""
        with patch(
            "pedagogy.personalization.tts_adapter.compute_tts_params",
            AsyncMock(return_value={"rate": 1.0, "pitch": 1.0, "voice": "x"}),
        ):
            rate = await get_edge_tts_rate_with_bandit(
                "alice", bandit_speech_rate=None, confusion_score=0.1,
            )
        assert rate == "+0%"

    @pytest.mark.asyncio
    async def test_high_confusion_slows_down(self):
        """High fused confusion → final rate scaled by 0.9 → -10%."""
        with patch(
            "pedagogy.personalization.tts_adapter.compute_tts_params",
            AsyncMock(return_value={"rate": 1.0, "pitch": 1.0, "voice": "x"}),
        ):
            rate = await get_edge_tts_rate_with_bandit(
                "alice", bandit_speech_rate=None, confusion_score=0.8,
            )
        # 1.0 × 1.0 × 0.9 = 0.9 → -10%
        assert rate == "-10%"

    @pytest.mark.asyncio
    async def test_bandit_slow_plus_confusion_compounds(self):
        """Bandit chose 'slow' (0.85) AND high confusion (× 0.9) → 0.765 → -23 %."""
        with patch(
            "pedagogy.personalization.tts_adapter.compute_tts_params",
            AsyncMock(return_value={"rate": 1.0, "pitch": 1.0, "voice": "x"}),
        ):
            rate = await get_edge_tts_rate_with_bandit(
                "alice", bandit_speech_rate="slow", confusion_score=0.9,
            )
        # 1.0 × 0.85 × 0.9 = 0.765 → -23 %
        assert rate == rate_float_to_edge_str(0.765)

    @pytest.mark.asyncio
    async def test_bandit_fast_neutralised_by_high_confusion(self):
        """Bandit said 'fast' but the student is confused — confusion
        partially neutralises the bandit's push."""
        with patch(
            "pedagogy.personalization.tts_adapter.compute_tts_params",
            AsyncMock(return_value={"rate": 1.0, "pitch": 1.0, "voice": "x"}),
        ):
            rate_no_conf = await get_edge_tts_rate_with_bandit(
                "alice", bandit_speech_rate="fast", confusion_score=0.0,
            )
            rate_with_conf = await get_edge_tts_rate_with_bandit(
                "alice", bandit_speech_rate="fast", confusion_score=0.9,
            )
        # rate_no_conf = 1.15 → "+15%"
        # rate_with_conf = 1.15 × 0.9 = 1.035 → "+3%"
        # Compare numerically (parsing the percentage), not lexicographically.
        def _pct(s):
            return int(s.replace("+", "").replace("%", "").replace("-", "-"))
        assert _pct(rate_with_conf) < _pct(rate_no_conf)

    @pytest.mark.asyncio
    async def test_compute_failure_safe_default(self):
        """If anything blows up internally we return '+0%' so TTS
        never breaks the user flow."""
        async def _fail(*a, **kw):
            raise RuntimeError("DB down")

        with patch(
            "pedagogy.personalization.tts_adapter.compute_tts_params",
            side_effect=_fail,
        ):
            rate = await get_edge_tts_rate_with_bandit(
                "alice", bandit_speech_rate="slow", confusion_score=0.9,
            )
        assert rate == "+0%"
