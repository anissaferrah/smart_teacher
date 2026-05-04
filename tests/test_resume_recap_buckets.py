"""Tests for the duration-aware build_resume_recap.

Three duration buckets must produce three different recap styles :

  - <30s     : just a continuity opener ("Je continue.")
  - 30s-2min : short topic mention ("On était sur X.")
  - >=2min   : full last-sentence read (legacy long recap)

Plus the legacy contract : char_offset==0 returns "" (first run, nothing
to recap), and a None duration falls back to the safe long recap.
"""
from __future__ import annotations

import pytest

from services.presentation import build_resume_recap


# A narration long enough to have multiple complete sentences with a
# meaningful "last sentence" at the chosen offset. The first sentence is
# 13 words ; the second is 9 words ; offset=120 falls into sentence #3.
NARRATION = (
    "Le modèle vectoriel représente chaque document comme un vecteur dans un espace multidimensionnel. "
    "Cette représentation permet de calculer la similarité par produit scalaire. "
    "Les vecteurs sont normalisés pour rendre la mesure indépendante de la longueur. "
    "C'est une technique classique en recherche d'information."
)
# offset 200 falls inside the third sentence (so the last *complete*
# sentence boundary is the second one, "Cette représentation permet…").
OFFSET_MID = 200


# ════════════════════════════════════════════════════════════════════
# Quick bucket — <30s
# ════════════════════════════════════════════════════════════════════

class TestQuickBucket:
    """Brief interruptions get only a continuity opener, no recap content."""

    def test_quick_french_no_recap_content(self):
        out = build_resume_recap(NARRATION, OFFSET_MID, language="fr",
                                 interruption_duration_s=5.0)
        assert out == "Je continue. "

    def test_quick_english(self):
        out = build_resume_recap(NARRATION, OFFSET_MID, language="en",
                                 interruption_duration_s=10.0)
        assert out == "I continue. "

    def test_quick_at_threshold_minus_one(self):
        """29.9s must still be quick (boundary check)."""
        out = build_resume_recap(NARRATION, OFFSET_MID, language="fr",
                                 interruption_duration_s=29.9)
        assert "On était" not in out
        assert "Continuons" not in out
        assert out == "Je continue. "


# ════════════════════════════════════════════════════════════════════
# Medium bucket — 30s ≤ duration < 2min
# ════════════════════════════════════════════════════════════════════

class TestMediumBucket:
    """30s-2min : short topic mention, no full re-read."""

    def test_medium_french(self):
        out = build_resume_recap(NARRATION, OFFSET_MID, language="fr",
                                 interruption_duration_s=60.0)
        # Must use the medium prefix
        assert out.startswith("On était sur :")
        assert "Continuons." in out
        # Topic = first 6 words of the last complete sentence (sentence 2)
        # ≈ "Cette représentation permet de calculer la"
        assert "Cette représentation permet" in out
        # Must NOT include the full sentence (medium is supposed to be
        # short — we cap at 6 words). The full second sentence has 9
        # words including the period ; if all 9 made it through, we'd
        # see "scalaire" in the recap.
        assert "scalaire" not in out

    def test_medium_english(self):
        out = build_resume_recap(NARRATION, OFFSET_MID, language="en",
                                 interruption_duration_s=90.0)
        assert out.startswith("We were on:")
        assert "Let's continue." in out

    def test_medium_at_threshold(self):
        """Exactly 30s → medium (not quick)."""
        out = build_resume_recap(NARRATION, OFFSET_MID, language="fr",
                                 interruption_duration_s=30.0)
        assert out.startswith("On était sur :")


# ════════════════════════════════════════════════════════════════════
# Full bucket — >= 2min
# ════════════════════════════════════════════════════════════════════

class TestFullBucket:
    """≥ 2min OR None : full last-sentence recap (legacy behaviour)."""

    def test_full_french_long_pause(self):
        out = build_resume_recap(NARRATION, OFFSET_MID, language="fr",
                                 interruption_duration_s=300.0)
        assert out.startswith("On était en train de voir :")
        assert "Continuons." in out
        # Full bucket gives the WHOLE last sentence (capped at 25 words).
        assert "scalaire" in out, \
            "full bucket must include the full sentence, not just first words"

    def test_full_english_long_pause(self):
        out = build_resume_recap(NARRATION, OFFSET_MID, language="en",
                                 interruption_duration_s=300.0)
        assert out.startswith("We were just covering:")

    def test_none_duration_uses_full(self):
        """Backwards-compat : if duration is None (caller couldn't compute it),
        fall back to the safe long recap rather than dropping the recap.
        """
        out = build_resume_recap(NARRATION, OFFSET_MID, language="fr",
                                 interruption_duration_s=None)
        assert out.startswith("On était en train de voir :")


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_zero_offset_returns_empty(self):
        """First run (offset=0) → no recap regardless of duration."""
        for dur in (None, 5.0, 60.0, 300.0):
            assert build_resume_recap(NARRATION, 0, language="fr",
                                      interruption_duration_s=dur) == ""

    def test_empty_narration_returns_empty(self):
        assert build_resume_recap("", 100, language="fr",
                                  interruption_duration_s=60.0) == ""

    def test_very_short_last_sentence_falls_back_to_quick(self):
        """A 4-word last sentence isn't worth recapping in any bucket —
        the helper falls back to the continuity opener even on 'full'.
        """
        narration = "OK ! Voilà donc ce qui se passe ici en détail dans le modèle vectoriel."
        # offset inside the second sentence; first sentence has 1 word
        # ("OK !") which is too short to recap.
        out = build_resume_recap(narration, 5, language="fr",
                                 interruption_duration_s=300.0)
        # Should not crash, should return either "" or the continuity opener.
        assert out in ("", "Je continue. ")


# ════════════════════════════════════════════════════════════════════
# Backwards-compat with positional call
# ════════════════════════════════════════════════════════════════════

class TestBackwardsCompat:
    """Old call sites that didn't pass interruption_duration_s must
    still work, falling back to the safe full-recap behaviour."""

    def test_three_arg_call_still_works(self):
        out = build_resume_recap(NARRATION, OFFSET_MID, "fr")
        # Without duration → defaults to "full"
        assert out.startswith("On était en train de voir :")
