"""Tests for compute_text_cursor_from_audio_progress.

Why this exists : the operator-visible bug was "pause near end of slide
→ resume restarts the whole slide". Root cause : ``current_presentation_cursor``
was always set to ``len(narration_text)`` once text streaming finished,
even though TTS audio playback typically lags 5-15 seconds behind text
streaming. So a pause at that moment saved cursor=len, and the resume
hit the legacy "cursor < len else 0" fallback that restarted from char 0.

Fix : the frontend computes ``currentAudio.currentTime / duration``
when sending the interrupt, the backend converts that fraction to a
char offset and overrides the cursor BEFORE saving paused_state.
"""
from __future__ import annotations

import math

import pytest

from services.presentation import compute_text_cursor_from_audio_progress


# A realistic narration : 837 chars (the exact length from the user's
# bug report — "La recherche d'information (RI) est une discipline...").
NARRATION = (
    "La recherche d'information (RI) est une discipline de l'informatique "
    "qui traite de l'acquisition, l'organisation, le stockage, la recherche "
    "et la sélection d'informations pertinentes pour un utilisateur, afin "
    "de répondre à un besoin précis d'information. Le terme \"informatique "
    "documentaire\" est synonyme de RI, et désigne les techniques utilisées "
    "pour l'organisation des documents numériques et leur recherche. "
    "L'information retrieval (IR) et le textual information retrieval (TIR) "
    "sont des sous-ensembles de la RI qui se concentrent sur la recherche "
    "d'informations texte dans un corpus de documents, grâce à l'utilisation "
    "de techniques d'indexation et de recherche de mot-clé."
)


# ════════════════════════════════════════════════════════════════════
# Happy path
# ════════════════════════════════════════════════════════════════════

class TestLinearMapping:
    """Half-progress through audio → roughly half-progress through text."""

    def test_half_progress(self):
        cursor = compute_text_cursor_from_audio_progress(0.5, NARRATION)
        assert cursor is not None
        assert abs(cursor - len(NARRATION) // 2) <= 1

    def test_quarter_progress(self):
        cursor = compute_text_cursor_from_audio_progress(0.25, NARRATION)
        assert cursor is not None
        assert abs(cursor - len(NARRATION) // 4) <= 1

    def test_full_progress_clamps_to_len(self):
        cursor = compute_text_cursor_from_audio_progress(1.0, NARRATION)
        assert cursor == len(NARRATION)

    def test_near_end_no_overshoot(self):
        cursor = compute_text_cursor_from_audio_progress(0.999, NARRATION)
        assert cursor is not None
        # Must not exceed the narration length even at near-1 progress
        assert cursor <= len(NARRATION)


# ════════════════════════════════════════════════════════════════════
# Reject invalid / unusable inputs
# ════════════════════════════════════════════════════════════════════

class TestRejection:
    """Returning None signals the caller to fall back to existing cursor logic."""

    def test_none_progress(self):
        assert compute_text_cursor_from_audio_progress(None, NARRATION) is None

    def test_zero_progress_is_no_op(self):
        # 0.0 means "audio hasn't started" — no override
        assert compute_text_cursor_from_audio_progress(0.0, NARRATION) is None

    def test_negative_progress(self):
        assert compute_text_cursor_from_audio_progress(-0.1, NARRATION) is None

    def test_progress_above_one(self):
        assert compute_text_cursor_from_audio_progress(1.5, NARRATION) is None

    def test_nan_progress(self):
        assert compute_text_cursor_from_audio_progress(float("nan"), NARRATION) is None

    def test_infinite_progress(self):
        assert compute_text_cursor_from_audio_progress(float("inf"), NARRATION) is None

    def test_non_numeric_progress(self):
        assert compute_text_cursor_from_audio_progress("not a number", NARRATION) is None

    def test_empty_narration(self):
        assert compute_text_cursor_from_audio_progress(0.5, "") is None


# ════════════════════════════════════════════════════════════════════
# The user's bug regression
# ════════════════════════════════════════════════════════════════════

class TestUserBugRegression:
    """Audio still playing at 80%, student pauses → cursor must reflect
    that 80% position, not the saved-len-at-text-stream-end position."""

    def test_pause_at_80pct_gives_80pct_cursor(self):
        cursor = compute_text_cursor_from_audio_progress(0.80, NARRATION)
        assert cursor is not None
        # 80% of 837 ≈ 670
        expected = int(0.80 * len(NARRATION))
        assert abs(cursor - expected) <= 1
        # Must NOT be at len (= 837) — that's the buggy old behaviour
        assert cursor < len(NARRATION)

    def test_pause_just_after_audio_started(self):
        """Audio at 0.05 = ~5% played, student pauses to ask question."""
        cursor = compute_text_cursor_from_audio_progress(0.05, NARRATION)
        assert cursor is not None
        # ~5% of 837 ≈ 41
        assert cursor < 100
        # Resume from char ~41 makes sense — student heard the first
        # sentence and is now at the start of the second.

    def test_int_progress_treated_as_float(self):
        """0 (int) and 1 (int) shouldn't crash — still treated as fractions."""
        # 0 (int) → no-op (treated as "audio hasn't started")
        assert compute_text_cursor_from_audio_progress(0, NARRATION) is None
        # 1 (int) → full progress
        assert compute_text_cursor_from_audio_progress(1, NARRATION) == len(NARRATION)


# ════════════════════════════════════════════════════════════════════
# Coverage: the full pause/resume scenario the system protects against
# ════════════════════════════════════════════════════════════════════

class TestEndToEndScenario:
    """Walk through the bug : without audio_progress, cursor=len; with
    audio_progress at 0.85, cursor=~711 → resume from middle, no replay."""

    def test_with_audio_progress_avoids_replay(self):
        # WITHOUT audio_progress : caller would have cursor=len, our rewind
        # fallback would replay the last sentence.
        # WITH audio_progress=0.85 : cursor is at 85% of text → resume
        # from there → student hears only the last 15% (much less repeat).
        len_n = len(NARRATION)
        cursor_with = compute_text_cursor_from_audio_progress(0.85, NARRATION)
        assert cursor_with is not None
        # 85% should land somewhere around 711, well before len.
        assert int(0.84 * len_n) <= cursor_with <= int(0.86 * len_n)
        # And critically, the remainder is nontrivial (the student gets
        # to hear the last ~15% of the slide on resume, no full replay).
        remaining = len_n - cursor_with
        assert remaining > 50, "remaining audio after pause should be substantive"
        assert remaining < len_n // 5, "shouldn't be more than 20% of total"
