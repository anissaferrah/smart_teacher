"""Tests for the parenthetical-duplicate stripper.

The user reported narrations like :
    "Notation de Bohlen-Muller (Notation de Bohlen-Muller)"
    "L'analyse de termes-clés (Keyword Analysis)"
    "Aspects Pratiques de l'IR (Aspects Pratiques de l'IR)"

The first and third are pure duplicates → must be stripped.
The second is a FR -> EN translation pair → kept (informative).

Plus the function must NEVER touch :
  - acronym expansions ("RI (Recherche d'Information)")
  - genuine clarifications ("k-means (an unsupervised algorithm)")
"""
from __future__ import annotations

import pytest

from agentic.teaching.narrator import _strip_parenthetical_duplicates


# ════════════════════════════════════════════════════════════════════
# Cases that MUST be stripped (the bug)
# ════════════════════════════════════════════════════════════════════

class TestDuplicatesStripped:

    def test_exact_duplicate(self):
        text = "Notation de Bohlen-Muller (Notation de Bohlen-Muller) sert à mesurer."
        cleaned, n = _strip_parenthetical_duplicates(text)
        assert n == 1
        assert "(Notation de Bohlen-Muller)" not in cleaned
        assert "Notation de Bohlen-Muller sert à mesurer." in cleaned

    def test_case_fold_duplicate(self):
        text = "Le modèle vectoriel (Modèle Vectoriel) est utilisé."
        cleaned, n = _strip_parenthetical_duplicates(text)
        assert n == 1
        assert "(Modèle Vectoriel)" not in cleaned

    def test_accent_fold_duplicate(self):
        text = "Aspects Pratiques de l'IR (Aspects Pratiques de l'IR) en pratique."
        cleaned, n = _strip_parenthetical_duplicates(text)
        assert n == 1
        assert "(Aspects Pratiques" not in cleaned

    def test_multiple_duplicates_in_one_text(self):
        text = (
            "L'analyse de termes (L'analyse de termes) est utile. "
            "Le modèle vectoriel (Modèle Vectoriel) aussi."
        )
        cleaned, n = _strip_parenthetical_duplicates(text)
        assert n == 2


# ════════════════════════════════════════════════════════════════════
# Cases that MUST be preserved (genuine information)
# ════════════════════════════════════════════════════════════════════

class TestPreservedParentheticals:

    def test_acronym_expansion_kept(self):
        """RI (Recherche d'Information) is a real expansion."""
        text = "On parle de la RI (Recherche d'Information) en cours."
        cleaned, n = _strip_parenthetical_duplicates(text)
        assert n == 0
        assert "(Recherche d'Information)" in cleaned

    def test_translation_pair_kept(self):
        """FR + EN translation is informative ; keep it.
        'L'analyse de termes-clés (Keyword Analysis)' has DIFFERENT
        tokens — keyword/analysis vs analyse/termes — so no overlap
        and the parens stays."""
        text = "L'analyse de termes-clés (Keyword Analysis) est centrale."
        cleaned, n = _strip_parenthetical_duplicates(text)
        assert n == 0
        assert "(Keyword Analysis)" in cleaned

    def test_genuine_clarification_kept(self):
        text = "K-means (un algorithme non supervisé) partitionne les données."
        cleaned, n = _strip_parenthetical_duplicates(text)
        assert n == 0
        assert "(un algorithme non supervisé)" in cleaned

    def test_short_inline_definition_kept(self):
        text = "Le rappel (recall) mesure la couverture."
        cleaned, n = _strip_parenthetical_duplicates(text)
        # "rappel" and "recall" share no tokens (different roots, both 6 chars)
        # — preserved.
        assert n == 0


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_text(self):
        cleaned, n = _strip_parenthetical_duplicates("")
        assert cleaned == ""
        assert n == 0

    def test_no_parentheses(self):
        text = "Aucune parenthèse ici."
        cleaned, n = _strip_parenthetical_duplicates(text)
        assert cleaned == text
        assert n == 0

    def test_empty_parentheses(self):
        """Edge case : ``()`` — minimum content is 2 chars per regex."""
        text = "Vide () ici."
        cleaned, n = _strip_parenthetical_duplicates(text)
        assert n == 0
        assert cleaned == text  # untouched

    def test_idempotent(self):
        """Running twice produces the same output."""
        text = "Foo (Foo) bar."
        once, _ = _strip_parenthetical_duplicates(text)
        twice, _ = _strip_parenthetical_duplicates(once)
        assert once == twice


# ════════════════════════════════════════════════════════════════════
# The user's actual reported case
# ════════════════════════════════════════════════════════════════════

class TestUserReportedCase:
    """End-to-end on the exact narration the user showed."""

    NARRATION = (
        "Bienvenue dans ce cours sur la recherche d'information (Information Retrieval). "
        "Nous aborderons : "
        "L'analyse de termes-clés (Keyword Analysis) "
        "Les méthodes de recherche de texte simple (Simple Text Retrieval Methods) "
        "La notation de Bohlen-Muller (Notation de Bohlen-Muller) "
        "Le modèle du vecteur de représentation (Vecteur de Représentation Model) "
        "Les aspects pratiques de l'IR (Aspects Pratiques de l'IR)."
    )

    def test_strips_pure_duplicates_only(self):
        cleaned, n = _strip_parenthetical_duplicates(self.NARRATION)
        # "Notation de Bohlen-Muller (Notation de Bohlen-Muller)" → DUP, stripped
        # "Aspects pratiques de l'IR (Aspects Pratiques de l'IR)" → DUP (case fold), stripped
        # FR-EN pairs (Information Retrieval, Keyword Analysis, Simple Text..., Vecteur de Représentation Model) → kept
        # Expect at least 2 strips.
        assert n >= 2
        assert "(Notation de Bohlen-Muller)" not in cleaned
        assert "(Aspects Pratiques de l'IR)" not in cleaned
        # Translation pairs preserved
        assert "(Information Retrieval)" in cleaned
        assert "(Keyword Analysis)" in cleaned
