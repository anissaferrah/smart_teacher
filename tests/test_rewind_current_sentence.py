"""Tests for rewind_to_current_sentence_start.

Why this exists : when the student pauses mid-sentence, we want to
replay the FULL sentence from its start on resume — not skip past the
partial sentence (the previous forward-skip behaviour). Hearing
"Le modèle vectoriel représente les… [pause] → Cette représentation
permet…" felt jarring because the cut sentence's idea was lost.

The new flow rewinds backward to the start of the CURRENT sentence so
the student hears the whole idea again from the beginning.
"""
from __future__ import annotations

import pytest

from services.presentation import rewind_to_current_sentence_start


# ════════════════════════════════════════════════════════════════════
# Mid-sentence cursor → start of current sentence
# ════════════════════════════════════════════════════════════════════

class TestMidSentenceRewind:

    def test_pause_in_middle_of_second_sentence(self):
        text = (
            "Le modèle vectoriel représente les documents. "
            "Cette représentation permet de calculer la similarité. "
            "Les vecteurs sont normalisés."
        )
        # Find a position inside the second sentence
        cursor = text.find("permet")
        assert cursor > 0
        start = rewind_to_current_sentence_start(text, cursor)
        # Must rewind to the start of "Cette représentation..."
        assert text[start:].startswith("Cette représentation")

    def test_pause_in_first_sentence_returns_zero(self):
        text = "Le modèle vectoriel représente les documents. Cette représentation..."
        cursor = text.find("représente")
        start = rewind_to_current_sentence_start(text, cursor)
        # Inside first sentence → no prior terminator → start from 0
        assert start == 0

    def test_pause_right_after_terminator(self):
        text = "First sentence. Second sentence. Third sentence."
        # Pause exactly after "First sentence. "
        cursor = text.find("Second")
        start = rewind_to_current_sentence_start(text, cursor)
        # Should rewind to start of "Second"
        assert text[start:].startswith("Second")

    def test_pause_at_terminator_itself(self):
        text = "First. Second. Third."
        # Pause at the period of "Second."
        cursor = text.find("Second.") + len("Second")
        start = rewind_to_current_sentence_start(text, cursor)
        # The terminator before that point ends after "First. " → start = 7
        assert text[start:].startswith("Second")


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_narration(self):
        assert rewind_to_current_sentence_start("", 50) == 0

    def test_cursor_zero(self):
        text = "Hello. World."
        assert rewind_to_current_sentence_start(text, 0) == 0

    def test_negative_cursor(self):
        text = "Hello. World."
        assert rewind_to_current_sentence_start(text, -5) == 0

    def test_cursor_past_end_delegates_to_last_sentence(self):
        """Past-end cursor reuses the rewind-to-last-sentence helper —
        this happens when text streaming finished before the audio."""
        text = "First. Second. Third."
        # cursor at len(text) → must rewind to start of last sentence
        start = rewind_to_current_sentence_start(text, len(text))
        assert text[start:].startswith("Third")

    def test_no_terminators_returns_zero(self):
        text = "no punctuation at all just words"
        cursor = 15
        start = rewind_to_current_sentence_start(text, cursor)
        assert start == 0

    def test_french_question_mark_terminator(self):
        text = "Premier ? Deuxième. Troisième !"
        # Pause inside "Deuxième" — should rewind to its start
        cursor = text.find("Deux")
        start = rewind_to_current_sentence_start(text, cursor)
        assert text[start:].startswith("Deuxième")

    def test_ellipsis_treated_as_terminator(self):
        text = "Maybe… Then this. Or that."
        cursor = text.find("Then")
        start = rewind_to_current_sentence_start(text, cursor)
        assert text[start:].startswith("Then")


# ════════════════════════════════════════════════════════════════════
# Cross-check : the helper used in the NEW resume flow
# ════════════════════════════════════════════════════════════════════

class TestResumeFlowIntegration:
    """The new resume flow uses this helper to give the student the
    full sentence they were on. Verify the intuition end-to-end."""

    NARRATION = (
        "Le modèle vectoriel représente les documents comme des vecteurs. "
        "Cette représentation permet de calculer la similarité par produit scalaire. "
        "Les vecteurs sont normalisés pour rendre la mesure indépendante de la longueur. "
        "C'est une technique classique en recherche d'information."
    )

    def test_pause_at_60pct_replays_third_sentence_in_full(self):
        # 60% of ~280 chars ≈ ~168 — somewhere in the third sentence
        cursor = int(0.60 * len(self.NARRATION))
        start = rewind_to_current_sentence_start(self.NARRATION, cursor)
        replayed = self.NARRATION[start:]
        # Resume must START with a sentence beginning, NOT mid-word
        first_word = replayed.split()[0]
        assert first_word[0].isupper(), (
            f"replay should start with a capitalised word (sentence start), "
            f"got: {first_word!r}"
        )

    def test_pause_at_98pct_avoids_orphan_tail(self):
        """Pause near end → cursor is in the LAST sentence → start is
        beginning of that sentence, not a few chars before the end."""
        cursor = int(0.98 * len(self.NARRATION))
        start = rewind_to_current_sentence_start(self.NARRATION, cursor)
        replayed = self.NARRATION[start:]
        # Replay should be a meaningful sentence, not a 5-char fragment
        assert len(replayed) > 30
        # And should start with a capital
        assert replayed[0].isupper()
