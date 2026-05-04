"""Tests for rewind_to_last_sentence_start.

Why this exists : when the saved presentation cursor is at/past the
end of the narration (text streaming finished, but TTS audio may
still be playing on the client), we don't want to :

  - Restart the slide from 0 (the operator-visible 3x repetition bug
    triggered when ``cursor < len`` was strict-`<` and the else branch
    fell back to 0)
  - Skip entirely (the student may not have actually heard the end —
    TTS audio playback can lag the text streaming)

So we rewind to the start of the last sentence and replay only that.
"""
from __future__ import annotations

import pytest

from services.presentation import rewind_to_last_sentence_start


class TestRewindToLastSentence:

    def test_two_sentences_rewinds_to_second(self):
        text = "First sentence. Second sentence."
        # One inter-sentence boundary ". " between them.
        # rewind goes to start of "Second sentence."
        offset = rewind_to_last_sentence_start(text)
        assert text[offset:].startswith("Second")

    def test_three_sentences_rewinds_to_third(self):
        text = "First. Second. Third."
        offset = rewind_to_last_sentence_start(text)
        assert text[offset:].startswith("Third")

    def test_french_with_accents(self):
        text = (
            "Le modèle vectoriel représente les documents. "
            "Cette représentation est utilisée pour la recherche. "
            "Les vecteurs sont normalisés."
        )
        offset = rewind_to_last_sentence_start(text)
        # Should rewind to start of the third (last) sentence
        assert text[offset:].startswith("Les vecteurs")

    def test_french_question_mark(self):
        text = "Premier. Deuxième ? Troisième !"
        offset = rewind_to_last_sentence_start(text)
        # ". " at end of "Premier" + " ? " after "Deuxième"
        # second-to-last terminator ends after "? " → "Troisième !"
        assert text[offset:].startswith("Troisième")

    def test_ellipsis_treated_as_terminator(self):
        text = "Begin… middle. End."
        offset = rewind_to_last_sentence_start(text)
        # Last inter-boundary is ". " before "End"
        assert text[offset:].startswith("End")

    def test_empty_text_returns_zero(self):
        assert rewind_to_last_sentence_start("") == 0

    def test_single_sentence_returns_zero(self):
        # One sentence with terminating punctuation but no inter-boundary
        # → falls back to len-100 path. Since the text is < 100 chars,
        # max(0, len-100) = 0.
        text = "Une seule phrase."
        assert rewind_to_last_sentence_start(text) == 0

    def test_no_punctuation_falls_back_to_tail(self):
        # 200 chars of pure noise, no sentence breaks
        text = "x" * 200
        offset = rewind_to_last_sentence_start(text)
        # Should rewind ~100 chars from the end
        assert 95 <= offset <= 105
        assert offset < len(text)

    def test_short_no_punctuation_returns_zero(self):
        # Less than 100 chars, no breaks → returns 0 (max(0, len-100))
        text = "short text without punctuation"
        offset = rewind_to_last_sentence_start(text)
        assert offset == 0


# ════════════════════════════════════════════════════════════════════
# Regression scenario from the user's bug report
# ════════════════════════════════════════════════════════════════════

class TestUserBugScenario:
    """The user's pause cursor was at offset 837 = len of a 4-sentence
    paragraph about IR. The legacy code restarted from 0, repeating
    the entire 837-char paragraph 3 times. With the rewind, we should
    replay only the 4th sentence."""

    NARRATION = (
        "La recherche d'information (RI) est une discipline de l'informatique "
        "qui traite de l'acquisition, l'organisation, le stockage, la recherche "
        "et la sélection d'informations pertinentes pour un utilisateur. "
        "Le terme \"informatique documentaire\" est synonyme de RI. "
        "L'information retrieval (IR) et le textual information retrieval (TIR) "
        "sont des sous-ensembles de la RI. "
        "Enfin, le document retrieval désigne la capacité de récupérer des "
        "documents pertinents en réponse à une requête."
    )

    def test_rewinds_to_last_sentence_only(self):
        offset = rewind_to_last_sentence_start(self.NARRATION)
        # After rewind, the remaining text should START with the last sentence
        remaining = self.NARRATION[offset:]
        assert remaining.startswith("Enfin"), (
            f"expected last sentence (Enfin...), got: {remaining[:40]!r}"
        )

    def test_remaining_is_substantial_but_not_full(self):
        offset = rewind_to_last_sentence_start(self.NARRATION)
        remaining_len = len(self.NARRATION) - offset
        # Should be << full length but >> 0
        assert remaining_len > 50, "last sentence should have meaningful length"
        assert remaining_len < len(self.NARRATION) // 2, (
            "rewind shouldn't grab more than half the narration"
        )
