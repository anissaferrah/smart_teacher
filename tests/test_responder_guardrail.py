"""Tests for the off-topic / leak guardrail in ResponderAgent.

Two failure modes the guardrail catches :

  1. **Meta-instruction leak** — small LLMs (Ollama 7-8B) sometimes echo
     the personalization / strategy fragment as if it were the question.
     We saw "Voici comment expliquer la stratégie pédagogique que tu as
     demandée, à l'aide d'un exemple concret..." on a slide about IR.
     Always rejected, regardless of grounding.

  2. **Off-topic hallucination** — when retrieval comes back empty (very
     short or typo-laden student questions like "ce qoui ri?"), the LLM
     can pull from general knowledge and answer with concepts unrelated
     to the course (e.g. supervised learning + house prices on an IR
     course slide). Rejected when the answer has no citations AND no
     lexical anchor on the slide / chunks.

What the guardrail must NOT block :

  - Cited answers (trust grounding evidence)
  - Slide-grounded paraphrases without citations (legitimate)
  - Cross-slide answers that recover *any* of the course vocabulary
    (the student is allowed to ask about ANY part of the course, not
    only the current slide)
  - Already-honest fallback answers ("ce point n'est pas abordé...")
"""
from __future__ import annotations

import pytest

from agentic.qa.responder import (
    _detect_leak_or_offtopic,
    _GROUNDING_OVERLAP_THRESHOLD,
    _HONEST_FALLBACK,
    _content_words,
)


# ════════════════════════════════════════════════════════════════════
# _content_words — tokeniser
# ════════════════════════════════════════════════════════════════════

class TestContentWords:
    def test_drops_short_words(self):
        # "le", "un", "à" are < 4 chars and must be dropped
        words = _content_words("le chat est sur un canapé")
        assert "chat" in words
        assert "canape" in words  # accent folded
        assert "le" not in words
        assert "un" not in words

    def test_accent_fold(self):
        words = _content_words("récupération")
        assert "recuperation" in words

    def test_lowercase(self):
        words = _content_words("RECHERCHE Information")
        assert "recherche" in words
        assert "information" in words

    def test_empty(self):
        assert _content_words("") == set()


# ════════════════════════════════════════════════════════════════════
# Leak phrase detection — always rejects
# ════════════════════════════════════════════════════════════════════

class TestLeakPhraseDetection:

    SLIDE = "La Recherche d'Information traite de l'organisation des documents."
    CHUNKS = [{"content": "Recherche d'Information IR document"}]

    @pytest.mark.parametrize("leak", [
        "Voici comment expliquer la stratégie pédagogique que tu as demandée, "
        "avec un exemple sur les vecteurs de recherche.",
        "voici comment expliquer la stratégie SOCRATIQUE pour t'aider.",
        "Bien sûr, à propos de la personnalisation cognitive : l'IR consiste à...",
        "<<INSTRUCTION_INTERNE_NE_PAS_MENTIONNER>> ceci est une fuite",
    ])
    def test_french_leak_rejected(self, leak):
        rejected, reason = _detect_leak_or_offtopic(
            answer=leak, slide=self.SLIDE, chunks=self.CHUNKS,
            lang="fr", has_citations=True,  # even WITH citations, leak rejects
        )
        assert rejected is True
        assert "leak" in reason.lower()

    @pytest.mark.parametrize("leak", [
        "Here's how to explain the strategy you asked about: IR is...",
        "Sure, about cognitive personalization, this is how IR works.",
        "<<INTERNAL_INSTRUCTION_DO_NOT_MENTION>> leaked",
    ])
    def test_english_leak_rejected(self, leak):
        rejected, reason = _detect_leak_or_offtopic(
            answer=leak, slide=self.SLIDE, chunks=self.CHUNKS,
            lang="en", has_citations=True,
        )
        assert rejected is True
        assert "leak" in reason.lower()


# ════════════════════════════════════════════════════════════════════
# Citations are trusted — no off-topic check when LLM cited chunks
# ════════════════════════════════════════════════════════════════════

class TestCitedAnswersPass:
    """When the LLM cites at least one chunk, we trust the grounding
    evidence and skip the overlap check. Avoids over-rejecting answers
    where the LLM paraphrased heavily (low surface overlap, high
    semantic grounding)."""

    def test_cited_with_low_surface_overlap_passes(self):
        # Answer uses synonyms that don't surface-overlap the source
        slide = "Le modèle vectoriel représente les documents comme des vecteurs."
        chunks = [{"content": "Modèle vectoriel : représentation des documents."}]
        # Answer paraphrases heavily — but it IS grounded (citations[] non-empty)
        answer = "C'est une approche numérique pour comparer textuellement."
        rejected, _ = _detect_leak_or_offtopic(
            answer=answer, slide=slide, chunks=chunks,
            lang="fr", has_citations=True,
        )
        assert rejected is False


# ════════════════════════════════════════════════════════════════════
# Slide-grounded answers (no citations, but on-topic) pass
# ════════════════════════════════════════════════════════════════════

class TestSlideGroundedPasses:
    """Even without citations, an answer that recovers slide vocabulary
    is considered grounded — the LLM may have answered from the slide
    directly (the qa prompt allows that)."""

    def test_slide_grounded_no_citations_passes(self):
        slide = (
            "La Recherche d'Information (RI) traite de l'organisation, "
            "indexation et récupération des documents textuels en réponse "
            "à une requête d'utilisateur."
        )
        # Answer uses vocab from the slide : "recherche", "information",
        # "documents", "récupération", "indexation"
        answer = (
            "La Recherche d'Information consiste à organiser, indexer et "
            "récupérer des documents pour répondre à une requête."
        )
        rejected, _ = _detect_leak_or_offtopic(
            answer=answer, slide=slide, chunks=[],
            lang="fr", has_citations=False,
        )
        assert rejected is False

    def test_cross_slide_grounded_via_chunks_passes(self):
        """Student asks about something on a different slide ; RAG retrieved
        the right chunk. The answer recovers chunk vocabulary, no citation
        but passes."""
        slide = "Slide actuelle : indexation des documents textuels."
        # Chunk from a DIFFERENT slide of the same course
        chunks = [{
            "content": "Le modèle vectoriel représente chaque document "
                       "comme un vecteur dans un espace multidimensionnel."
        }]
        answer = (
            "Le modèle vectoriel représente les documents comme des "
            "vecteurs dans un espace multidimensionnel."
        )
        rejected, _ = _detect_leak_or_offtopic(
            answer=answer, slide=slide, chunks=chunks,
            lang="fr", has_citations=False,
        )
        assert rejected is False, "cross-slide grounded answer must not be blocked"


# ════════════════════════════════════════════════════════════════════
# Off-topic answers (no citations, no overlap) get rejected
# ════════════════════════════════════════════════════════════════════

class TestOffTopicRejected:
    """The smoking-gun case from the user's report : IR course, slide on
    'DEFINITIONS ET TERMINOLOGIE', student asks "ce qoui ri?", LLM
    returns an answer about supervised learning and house prices. The
    overlap with the IR slide is essentially zero — must be rejected."""

    def test_supervised_learning_on_ir_slide_rejected(self):
        slide = (
            "Recherche d'Information (RI) — DÉFINITIONS ET TERMINOLOGIE. "
            "Document Retrieval, Textual Information Retrieval, requête, "
            "indexation, pertinence."
        )
        chunks = [{"content": "Information Retrieval : domaine de l'informatique."}]
        # The actual hallucinated answer from the bug report
        offtopic = (
            "Bien sûr ! Voici un exemple concret. L'apprentissage supervisé "
            "consiste à prédire le prix d'une maison à partir de sa taille, "
            "son âge et son nombre de pièces. On entraîne l'algorithme sur "
            "des données puis on prédit le prix."
        )
        rejected, reason = _detect_leak_or_offtopic(
            answer=offtopic, slide=slide, chunks=chunks,
            lang="fr", has_citations=False,
        )
        # This one ALSO contains "Voici comment" pattern? Let's be precise :
        # the actual leak phrases are about "stratégie pédagogique", which
        # this offtopic example doesn't include. So the rejection should
        # come from the OVERLAP check, not the leak check.
        assert rejected is True
        assert "overlap" in reason.lower() or "leak" in reason.lower()


# ════════════════════════════════════════════════════════════════════
# Honest-fallback already present → don't re-reject
# ════════════════════════════════════════════════════════════════════

class TestHonestFallbackPasses:
    """When the LLM already correctly refused (course_bound_rule worked),
    the guardrail must not replace its honest answer with another honest
    answer — that'd be a no-op at best, and double-logging at worst."""

    def test_honest_fallback_fr_passes(self):
        rejected, _ = _detect_leak_or_offtopic(
            answer="Ce point n'est pas abordé dans ce cours.",
            slide="Slide non liée", chunks=[],
            lang="fr", has_citations=False,
        )
        assert rejected is False

    def test_honest_fallback_en_passes(self):
        rejected, _ = _detect_leak_or_offtopic(
            answer="This point is not covered in this course.",
            slide="Unrelated slide", chunks=[],
            lang="en", has_citations=False,
        )
        assert rejected is False


# ════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_answer_not_rejected(self):
        rejected, _ = _detect_leak_or_offtopic(
            answer="", slide="anything", chunks=[],
            lang="fr", has_citations=False,
        )
        assert rejected is False

    def test_no_source_at_all_does_not_overreject(self):
        """Cold start : no slide, no chunks. Don't reject — the prompt-level
        rule must do the work (we have no signal here)."""
        rejected, _ = _detect_leak_or_offtopic(
            answer="Some content", slide="", chunks=[],
            lang="fr", has_citations=False,
        )
        assert rejected is False

    def test_threshold_boundary(self):
        """Construct an answer that has *exactly* the threshold overlap."""
        # 10 content words in the answer, k of them appear in slide.
        slide = "alpha beta gamma delta epsilon"
        # 10 distinct >= 4-char words : need ~1 to be at 8% threshold.
        # Build answer with 1 overlapping ("alpha") + 9 unique.
        answer = (
            "alpha xxxxx yyyyy zzzzz wwwww qqqqq rrrrr ttttt uuuuu vvvvv"
        )
        rejected, _ = _detect_leak_or_offtopic(
            answer=answer, slide=slide, chunks=[],
            lang="fr", has_citations=False,
        )
        # 1/10 = 0.10 > threshold 0.08 → must NOT reject
        assert rejected is False

    def test_below_threshold_rejects(self):
        slide = "alpha"
        # 0 overlap on 10 content words → 0.0 << threshold
        answer = "xxxxx yyyyy zzzzz wwwww qqqqq rrrrr ttttt uuuuu vvvvv ppppp"
        rejected, reason = _detect_leak_or_offtopic(
            answer=answer, slide=slide, chunks=[],
            lang="fr", has_citations=False,
        )
        assert rejected is True
        assert "overlap" in reason.lower()


# ════════════════════════════════════════════════════════════════════
# Constants sanity check
# ════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_threshold_in_sane_range(self):
        # Too high → blocks legitimate paraphrases ; too low → never fires
        assert 0.03 <= _GROUNDING_OVERLAP_THRESHOLD <= 0.25

    def test_fallback_languages_present(self):
        assert "fr" in _HONEST_FALLBACK
        assert "en" in _HONEST_FALLBACK
        assert _HONEST_FALLBACK["fr"]
        assert _HONEST_FALLBACK["en"]
