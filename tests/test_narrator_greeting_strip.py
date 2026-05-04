"""Tests for the greeting-opener stripper in NarratorAgent.

Why this exists : weak LLMs (Ollama 7-8B) ignore the explicit "no
greetings" instruction in the narrator prompt and still open every
slide with "Bonjour", "Bienvenue dans la partie III", etc. Prompt-only
enforcement is unreliable, so we post-process the output to strip
these openers deterministically.

The stripper must :
  - Remove common French + English greeting openers from the START only
  - Leave mid-text occurrences untouched ("le serveur dit 'bonjour'")
  - Capitalise the first letter of the remaining text
  - Be idempotent (applying twice yields the same result)
  - Not nuke the entire narration if the strip would leave it empty
"""
from __future__ import annotations

import pytest

from agentic.teaching.narrator import _strip_greeting_opener


# ════════════════════════════════════════════════════════════════════
# French openers
# ════════════════════════════════════════════════════════════════════

class TestFrenchOpeners:
    """Each user-reported opener must be detected and removed."""

    @pytest.mark.parametrize("opener", [
        "Bonjour, bienvenue dans l'introduction à la recherche d'information (RI). ",
        "Bienvenue dans ce cours sur la recherche d'information. ",
        "Bienvenue dans la partie III du cours sur l'Introduction à la RI. ",
        "Dans le cadre de cette introduction à la recherche d'information (RI), ",
        "Bonjour, dans le cadre de cette introduction. ",
        "Allez, let's go ! ",
        "Allez, c'est parti ! ",
        "Bonjour à tous ! ",
        "Salut ! ",
        "Pour ce chapitre, nous verrons les définitions clés. ",
        "Dans ce chapitre, nous allons voir l'IR. ",
    ])
    def test_opener_stripped(self, opener):
        body = "Le modèle vectoriel représente les documents comme des vecteurs."
        text = opener + body
        stripped, removed = _strip_greeting_opener(text, "fr")
        assert stripped == body, (
            f"opener {opener!r} not fully stripped — left: {stripped!r}"
        )
        assert removed, "removed_opener should not be empty when something matched"

    def test_multiple_opener_clauses_stripped(self):
        """LLM stacks two openers : 'Bonjour ! Bienvenue dans...'"""
        text = (
            "Bonjour à tous ! Bienvenue dans la partie III ! "
            "Le moteur de recherche Web est l'application phare."
        )
        stripped, removed = _strip_greeting_opener(text, "fr")
        assert stripped.startswith("Le moteur de recherche")
        assert "Bonjour" in removed
        assert "Bienvenue" in removed


# ════════════════════════════════════════════════════════════════════
# English openers
# ════════════════════════════════════════════════════════════════════

class TestEnglishOpeners:
    @pytest.mark.parametrize("opener", [
        "Welcome to part III of the course. ",
        "Welcome back to the lecture. ",
        "Hello everyone! ",
        "Good morning! ",
        "In this introduction, we will see IR. ",
        "Let's start with the basics. ",
        "Let's get started! ",
        "Today, we'll cover the vector model. ",
    ])
    def test_opener_stripped(self, opener):
        body = "The vector model represents documents as vectors."
        text = opener + body
        stripped, removed = _strip_greeting_opener(text, "en")
        assert stripped == body
        assert removed


# ════════════════════════════════════════════════════════════════════
# Don't strip mid-text occurrences
# ════════════════════════════════════════════════════════════════════

class TestPreserveContent:
    """Greeting words inside the narration body must NOT be touched."""

    def test_bonjour_in_quote_preserved(self):
        text = (
            "Le serveur répond avec un message comme 'Bonjour' à chaque "
            "connexion HTTP."
        )
        stripped, removed = _strip_greeting_opener(text, "fr")
        assert stripped == text, "mid-text 'Bonjour' must not be stripped"
        assert removed == ""

    def test_clean_narration_unchanged(self):
        text = (
            "Voyons maintenant les applications de la RI. Le moteur de "
            "recherche Web est l'application la plus connue."
        )
        stripped, removed = _strip_greeting_opener(text, "fr")
        assert stripped == text
        assert removed == ""

    def test_welcome_inside_text_preserved(self):
        text = (
            "The HTTP server returns a 'Welcome' page on first visit."
        )
        stripped, removed = _strip_greeting_opener(text, "en")
        assert stripped == text


# ════════════════════════════════════════════════════════════════════
# Capitalisation + idempotence
# ════════════════════════════════════════════════════════════════════

class TestCapitalisation:
    def test_first_letter_capitalised_after_strip(self):
        # If the body starts with a lowercase word after the opener,
        # capitalise it so the narration doesn't sound mid-sentence.
        text = "Bienvenue dans ce cours. le modèle vectoriel..."
        stripped, _ = _strip_greeting_opener(text, "fr")
        assert stripped.startswith("Le modèle vectoriel")


class TestIdempotence:
    def test_stripping_twice_same_result(self):
        text = (
            "Bonjour à tous ! Le modèle vectoriel représente les "
            "documents comme des vecteurs."
        )
        first, _ = _strip_greeting_opener(text, "fr")
        second, _ = _strip_greeting_opener(first, "fr")
        assert first == second


# ════════════════════════════════════════════════════════════════════
# Defensive — empty / pure-greeting / very short
# ════════════════════════════════════════════════════════════════════

class TestDefensive:
    def test_empty_input(self):
        stripped, removed = _strip_greeting_opener("", "fr")
        assert stripped == ""
        assert removed == ""

    def test_pure_greeting_keeps_original(self):
        """If stripping would leave empty content, keep the original
        (downstream code expects non-empty narration)."""
        text = "Bonjour à tous !"
        stripped, removed = _strip_greeting_opener(text, "fr")
        assert stripped == text  # kept as-is
        assert removed == ""

    def test_huge_opener_refuses_to_strip(self):
        """If the 'opener' is suspiciously long (>400 chars), don't strip
        — the LLM probably ignored the slide entirely, better to keep
        the bad output than truncate to nothing."""
        # Build a fake 500-char "Bienvenue dans..." opener
        long_opener = "Bienvenue dans " + ("la partie merveilleuse " * 30) + "."
        body = "Le contenu principal de la slide."
        text = long_opener + " " + body
        stripped, removed = _strip_greeting_opener(text, "fr")
        # The opener is over the cap, so we keep the original text
        assert stripped == text
        assert removed == ""

    def test_unknown_language_treated_as_english(self):
        """Unknown lang code falls through to the EN regex (matches the
        narrator's own ``lang == 'fr'`` else-branch convention)."""
        text = "Welcome back. The vector model..."
        stripped, _ = _strip_greeting_opener(text, "ar")
        # Should still detect the EN opener
        assert stripped.startswith("The vector model")


# ════════════════════════════════════════════════════════════════════
# Regression — the user's actual example
# ════════════════════════════════════════════════════════════════════

class TestUserReportedRegression:
    """The exact opener the user pasted from production logs."""

    def test_bienvenue_dans_ce_cours(self):
        # Truncated version of the user's example
        text = (
            "Bienvenue dans ce cours sur la récherche d'information "
            "(Information Retrieval, IR). Nous allons commencer par une "
            "introduction à la RI. Dans ce chapitre, nous verrons que "
            "la Récherche d'Information est un domaine de l'informatique."
        )
        stripped, removed = _strip_greeting_opener(text, "fr")
        # The first sentence (the greeting) must be gone
        assert "Bienvenue" not in stripped[:50]
        # The actual content must remain
        assert "domaine de l'informatique" in stripped

    def test_bonjour_bienvenue_dans_introduction(self):
        text = (
            "Bonjour, bienvenue dans l'introduction à la recherche "
            "d'information (RI). Avec l'explosion des données numériques "
            "et du Big Data, nous sommes confrontés à une grande quantité."
        )
        stripped, _ = _strip_greeting_opener(text, "fr")
        assert "Bonjour" not in stripped[:30]
        assert "explosion des données" in stripped
