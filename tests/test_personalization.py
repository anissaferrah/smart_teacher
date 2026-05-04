"""
Tests unitaires pour la couche de personnalisation (M2 grade).

Couvre :
  - Modèle Bayes Dirichlet : math correctness (mean, variance, HDI)
  - Behavior of dominant() under uncertainty
  - VARK questionnaire scoring
  - Password strength rules
  - Auth helpers (hash/verify, JWT round-trip)
"""
import math
import pytest

from handlers.auth import (
    PasswordStrengthError, check_password_strength,
    create_access_token, decode_access_token, hash_password, verify_password,
)
from pedagogy.personalization.learning_style.bayes import (
    DEFAULT_PRIOR_ALPHA, PosteriorEstimate, SIGNAL_WEIGHTS, STYLES,
    fire_signal,
)
from pedagogy.personalization.learning_style.vark import (
    VARK_QUESTIONS, score_responses, serialize_for_frontend,
)


# ════════════════════════════════════════════════════════════════════
# Bayesian Dirichlet model
# ════════════════════════════════════════════════════════════════════

class TestPosteriorEstimate:
    def test_default_prior_is_uniform(self):
        p = PosteriorEstimate()
        means = p.mean
        # Uniform prior → all means ≈ 0.25
        for m in means:
            assert abs(m - 0.25) < 1e-9

    def test_posterior_concentrates_with_data(self):
        # Heavy visual signal → mean shifts toward visual
        p = PosteriorEstimate(alpha=(20.0, 2.0, 2.0, 2.0), n_observations=30)
        means = p.mean
        assert means[0] > 0.7    # visual dominates
        assert means[1] < 0.15
        # Confidence should be high when concentrated AND many observations
        assert p.confidence() > 0.5

    def test_prior_alone_zero_confidence(self):
        # Prior uniform + n_obs=0 → no confidence at all (cold start)
        p = PosteriorEstimate()
        assert p.confidence() < 0.05

    def test_uniform_high_n_still_low_confidence(self):
        # Many obs but uniform distribution → can't determine style
        p = PosteriorEstimate(alpha=(50.0, 50.0, 50.0, 50.0), n_observations=200)
        # Concentration ≈ 0 even though n is high → confidence ≈ 0
        assert p.confidence() < 0.05

    def test_dominant_returns_mixed_when_uniform(self):
        # Uniform posterior: every mean equals 1/N → no dimension above
        # baseline → "mixed" (mathematically defined, not heuristic).
        p = PosteriorEstimate(alpha=(2.0, 2.0, 2.0, 2.0))
        assert p.dominant() == "mixed"

    def test_dominant_returns_argmax_above_baseline(self):
        # Visual mean ~0.41, all others below baseline → visual wins.
        p = PosteriorEstimate(alpha=(20.0, 5.0, 2.0, 2.0))
        assert p.dominant() == "visual"

    def test_dominant_two_close_above_baseline_picks_argmax(self):
        # Two dimensions essentially tied above baseline: returns the argmax
        # deterministically. Previously a 0.05-gap threshold returned "mixed";
        # that arbitrary parameter has been removed.
        p = PosteriorEstimate(alpha=(10.0, 10.0, 2.0, 2.0))
        # alpha=10 each → mean=10/24 ≈ 0.417 → both above 0.25 baseline
        # tiebreak by index order → visual (first in STYLES tuple)
        assert p.dominant() == "visual"

    def test_hdi_95_contains_mean(self):
        p = PosteriorEstimate(alpha=(15.0, 5.0, 5.0, 5.0))
        for i in range(4):
            lo, hi = p.hdi_95(i)
            mean_i = p.mean[i]
            assert lo <= mean_i <= hi

    def test_means_sum_to_one(self):
        p = PosteriorEstimate(alpha=(7.5, 3.2, 1.8, 4.0))
        assert abs(sum(p.mean) - 1.0) < 1e-9

    def test_to_dict_complete(self):
        p = PosteriorEstimate(alpha=(8.0, 3.0, 2.0, 4.0), n_observations=12)
        d = p.to_dict()
        assert "scores" in d and "alpha" in d and "hdi_95" in d
        assert d["dominant"] in (*STYLES, "mixed")
        assert 0.0 <= d["confidence"] <= 1.0
        assert d["n_observations"] == 12


# ════════════════════════════════════════════════════════════════════
# Signal weights
# ════════════════════════════════════════════════════════════════════

class TestSignalWeights:
    def test_all_signals_have_at_least_one_dimension(self):
        for sig, weights in SIGNAL_WEIGHTS.items():
            total = sum(weights.values())
            assert total > 0, f"Signal '{sig}' has zero total weight"

    def test_vark_signals_strongest(self):
        """VARK self-report should have higher weight than behavioral."""
        for dim in STYLES:
            vark_w = SIGNAL_WEIGHTS[f"vark_{dim}"][dim]
            assert vark_w >= 3.0, f"vark_{dim} weight too low ({vark_w})"

    def test_audio_question_pure_auditory(self):
        w = SIGNAL_WEIGHTS["audio_question"]
        assert w == {"auditory": 1.0}


# ════════════════════════════════════════════════════════════════════
# Reformulation prompt
# ════════════════════════════════════════════════════════════════════

class TestComposeReformulationPrompt:
    def test_basic_fr(self):
        from pedagogy.dialogue import compose_reformulation_prompt
        out = compose_reformulation_prompt(
            original_question="je comprends pas",
            language="fr",
        )
        assert "L'étudiant n'a pas compris" in out
        assert "exemple concret" in out
        assert "je comprends pas" in out  # original question echoed

    def test_basic_en(self):
        from pedagogy.dialogue import compose_reformulation_prompt
        out = compose_reformulation_prompt(
            original_question="i don't get it",
            language="en",
        )
        assert "didn't understand" in out
        assert "concrete example" in out
        assert "i don't get it" in out

    def test_unknown_language_falls_back_to_fr(self):
        from pedagogy.dialogue import compose_reformulation_prompt
        out = compose_reformulation_prompt(
            original_question="?",
            language="zz",
        )
        # Falls back to French (default)
        assert "L'étudiant n'a pas compris" in out

    def test_history_text_included(self):
        from pedagogy.dialogue import compose_reformulation_prompt
        out = compose_reformulation_prompt(
            original_question="???",
            language="fr",
            history_text="Étudiant: hein ?\nProf: La formule est x²",
        )
        assert "Échanges récents" in out
        assert "Prof: La formule" in out

    def test_slide_content_truncated(self):
        from pedagogy.dialogue import compose_reformulation_prompt
        long_slide = "x" * 1000
        out = compose_reformulation_prompt(
            original_question="?",
            language="fr",
            last_slide_content=long_slide,
        )
        # Slide content is capped at 400 chars in the rendered prompt
        assert "x" * 400 in out
        assert "x" * 401 not in out


class TestResponderConfusionDelegates:
    """The QA Graph responder must build the same prompt as the audio pipeline."""

    def test_responder_matches_dialogue(self):
        from agentic.qa.responder import _make_confusion_prompt
        from pedagogy.dialogue import compose_reformulation_prompt

        slide = "Three key elements of K-NN..."
        lang = "fr"
        raw_q = "je comprends pas"
        responder_out = _make_confusion_prompt(
            slide, lang,
            history_text="",
            raw_question=raw_q,
        )
        expected = compose_reformulation_prompt(
            original_question=raw_q,
            language=lang,
            last_slide_content=slide,
            history_text="",
        )
        assert responder_out == expected

    def test_responder_with_history(self):
        from agentic.qa.responder import _make_confusion_prompt
        out = _make_confusion_prompt(
            "slide", "fr",
            history_text="Étudiant: hein ?",
            raw_question="je suis perdu",
        )
        assert "L'étudiant n'a pas compris" in out
        assert "je suis perdu" in out
        assert "Étudiant: hein ?" in out


# ════════════════════════════════════════════════════════════════════
# Definition fast-path (DM/CS terms answered directly by LLM, no RAG)
# ════════════════════════════════════════════════════════════════════

class TestDefinitionPrompt:
    def test_definition_prompt_fr_imposes_format(self):
        from agentic.qa.responder import _make_definition_prompt
        out = _make_definition_prompt("Qu'est-ce que K-NN ?", "fr")
        # Must enforce the strict course-bound rule (no Wikipedia leak)
        # and the 3-element format. The original test asserted on the
        # old "DÉFINITION FORMELLE" label; the prompt now uses just
        # "DÉFINITION" and adds the strict course-bound preamble.
        assert "Wikipédia" in out                # course-bound rule
        assert "DÉFINITION" in out               # 3-element format
        assert "INTUITION" in out
        assert "EXEMPLE" in out
        # Must inject the question
        assert "K-NN" in out
        # Must request JSON output with empty supporting_chunks
        assert "supporting_chunks" in out
        assert "[]" in out

    def test_definition_prompt_en_imposes_format(self):
        from agentic.qa.responder import _make_definition_prompt
        out = _make_definition_prompt("What is gradient descent?", "en")
        assert "Wikipedia" in out                # course-bound rule
        assert "DEFINITION" in out
        assert "INTUITION" in out
        assert "EXAMPLE" in out
        assert "gradient descent" in out

    def test_definition_prompt_includes_history(self):
        from agentic.qa.responder import _make_definition_prompt
        out = _make_definition_prompt(
            "Qu'est-ce que la régression ?", "fr",
            history_text="Étudiant: on parle de quoi maintenant ?",
        )
        assert "Étudiant: on parle de quoi maintenant ?" in out

    def test_definition_prompt_no_markdown_no_latex_in_instruction(self):
        from agentic.qa.responder import _make_definition_prompt
        out = _make_definition_prompt("Qu'est-ce que la fonction sigmoïde ?", "fr")
        # The prompt must instruct the LLM to avoid markdown / LaTeX
        assert "markdown" in out.lower() or "Markdown" in out
        assert "LaTeX" in out


class TestIntentDefinitionFlag:
    @staticmethod
    def _disable_sight(monkeypatch):
        monkeypatch.setattr("agentic.qa.intent._sight_predict", None)

    def test_definition_flag_with_term(self, monkeypatch):
        self._disable_sight(monkeypatch)
        from agentic.qa.intent import IntentAgent

        class FakeBrain:
            def ask(self, *a, **k):
                return ('{"intent": "question", "confidence": 0.9, '
                        '"needs_retrieval": true, "is_definition": true, '
                        '"definition_term": "K-NN"}', 0.1)

        agent = IntentAgent(FakeBrain())
        out = agent({"event_payload": {"text": "qu'est-ce que K-NN ?"}})
        intent = out["intent"]
        assert intent.type == "question"
        assert intent.payload["is_definition"] is True
        assert intent.payload["definition_term"] == "K-NN"
        # needs_retrieval STAYS true — RAG runs first, fallback decided
        # by responder based on whether the term appears in chunks.
        assert intent.needs_retrieval is True

    def test_definition_term_can_be_empty_when_not_definition(self, monkeypatch):
        self._disable_sight(monkeypatch)
        from agentic.qa.intent import IntentAgent

        class FakeBrain:
            def ask(self, *a, **k):
                return ('{"intent": "question", "confidence": 0.9, '
                        '"needs_retrieval": true, "is_definition": false, '
                        '"definition_term": ""}', 0.1)

        agent = IntentAgent(FakeBrain())
        out = agent({"event_payload": {"text": "explique cette slide"}})
        assert out["intent"].payload["is_definition"] is False
        # definition_term not set when is_definition=False
        assert out["intent"].payload.get("definition_term", "") == ""

    def test_non_question_no_definition_flag(self, monkeypatch):
        self._disable_sight(monkeypatch)
        from agentic.qa.intent import IntentAgent

        class FakeBrain:
            def ask(self, *a, **k):
                return ('{"intent": "navigation", "confidence": 0.9, '
                        '"needs_retrieval": false, "is_definition": false}', 0.1)

        agent = IntentAgent(FakeBrain())
        out = agent({"event_payload": {"text": "passe à la suivante"}})
        assert "is_definition" not in out["intent"].payload


class TestTermInChunks:
    """The course-priority routing depends on detecting whether the
    extracted technical term appears in the retrieved chunks."""

    def test_term_found_in_chunk(self):
        from agentic.qa.responder import _term_in_chunks
        chunks = [
            {"content": "K-NN classifies points by majority vote of neighbors."},
            {"content": "Other content about decision trees."},
        ]
        assert _term_in_chunks("K-NN", chunks) is True

    def test_term_not_in_any_chunk(self):
        from agentic.qa.responder import _term_in_chunks
        chunks = [
            {"content": "Decision trees and random forests."},
            {"content": "Linear regression and least squares."},
        ]
        assert _term_in_chunks("K-NN", chunks) is False

    def test_case_insensitive_match(self):
        from agentic.qa.responder import _term_in_chunks
        chunks = [{"content": "K-Nearest Neighbors algorithm explained."}]
        assert _term_in_chunks("k-nearest neighbors", chunks) is True

    def test_accent_insensitive_match(self):
        """LLM might extract 'regression' (no accent) while the course
        uses 'régression' (with accent). Both should match."""
        from agentic.qa.responder import _term_in_chunks
        chunks = [{"content": "La régression linéaire est ..."}]
        assert _term_in_chunks("regression lineaire", chunks) is True
        # Reverse direction also works
        chunks2 = [{"content": "Linear regression is ..."}]
        assert _term_in_chunks("régression", chunks2) is True

    def test_empty_term_returns_false(self):
        from agentic.qa.responder import _term_in_chunks
        assert _term_in_chunks("", [{"content": "anything"}]) is False
        assert _term_in_chunks(None, [{"content": "anything"}]) is False

    def test_short_single_char_does_not_match(self):
        """A 1-char 'term' would match too liberally (any letter appears)."""
        from agentic.qa.responder import _term_in_chunks
        assert _term_in_chunks("A", [{"content": "Anything"}]) is False

    def test_no_chunks_returns_false(self):
        from agentic.qa.responder import _term_in_chunks
        assert _term_in_chunks("K-NN", []) is False
        assert _term_in_chunks("K-NN", None) is False


# ════════════════════════════════════════════════════════════════════
# Grounded citations (responder JSON output parsing)
# ════════════════════════════════════════════════════════════════════
#
# NOTE: ``_wants_enrichment`` and its 14 hand-curated regex patterns
# (``_ENRICH_PATTERNS_FR`` / ``_ENRICH_PATTERNS_EN``) were retired when
# the responder dropped its ``enrich_mode`` branch. The "give me a fresh
# example beyond the slide" behaviour is now handled by the LLM directly
# given the rewritten query — no regex pre-classification needed.

class TestParseQAResponse:
    def test_valid_json_with_supporting_chunks(self):
        from agentic.qa.responder import _parse_qa_response
        id_to_chunk = {
            "concept_knn": {"idea_label": "K-NN", "source": "ch3.pdf", "score": 0.85},
            "concept_majority": {"idea_label": "Majority vote", "source": "ch3.pdf", "score": 0.72},
        }
        raw = '{"answer": "K-NN classifies by majority vote.", "supporting_chunks": ["concept_knn", "concept_majority"]}'
        answer, citations = _parse_qa_response(raw, id_to_chunk)
        assert answer == "K-NN classifies by majority vote."
        assert len(citations) == 2
        assert citations[0]["chunk_id"] == "concept_knn"
        assert citations[0]["idea_label"] == "K-NN"
        assert citations[0]["score"] == 0.85

    def test_empty_supporting_chunks_means_ungrounded(self):
        from agentic.qa.responder import _parse_qa_response
        raw = '{"answer": "From general knowledge: Newton said F=ma.", "supporting_chunks": []}'
        answer, citations = _parse_qa_response(raw, {})
        assert "F=ma" in answer
        assert citations == []

    def test_hallucinated_chunk_id_is_silently_dropped(self):
        from agentic.qa.responder import _parse_qa_response
        id_to_chunk = {"real_id": {"idea_label": "Real", "source": "", "score": 0.5}}
        raw = '{"answer": "x", "supporting_chunks": ["real_id", "fabricated_id"]}'
        answer, citations = _parse_qa_response(raw, id_to_chunk)
        assert len(citations) == 1
        assert citations[0]["chunk_id"] == "real_id"

    def test_malformed_json_falls_back_to_raw_text(self):
        from agentic.qa.responder import _parse_qa_response
        raw = "Not valid JSON, just a sentence."
        answer, citations = _parse_qa_response(raw, {})
        assert answer == "Not valid JSON, just a sentence."
        assert citations == []

    def test_duplicate_chunk_ids_deduplicated(self):
        from agentic.qa.responder import _parse_qa_response
        id_to_chunk = {"id1": {"idea_label": "A", "source": "", "score": 0.6}}
        raw = '{"answer": "x", "supporting_chunks": ["id1", "id1", "id1"]}'
        answer, citations = _parse_qa_response(raw, id_to_chunk)
        assert len(citations) == 1


class TestFormatChunksWithIds:
    def test_uses_idea_id_when_present(self):
        from agentic.qa.responder import _format_chunks_with_ids
        chunks = [{"content": "K-NN definition", "idea_id": "concept_knn", "idea_label": "K-NN"}]
        block, mapping = _format_chunks_with_ids(chunks)
        assert "[id:concept_knn]" in block
        assert "concept_knn" in mapping

    def test_falls_back_to_positional_id_when_idea_id_missing(self):
        from agentic.qa.responder import _format_chunks_with_ids
        chunks = [{"content": "Some content"}]
        block, mapping = _format_chunks_with_ids(chunks)
        assert "[id:c0]" in block
        assert "c0" in mapping

    def test_disambiguates_duplicate_idea_ids(self):
        from agentic.qa.responder import _format_chunks_with_ids
        chunks = [
            {"content": "Part 1", "idea_id": "same_id"},
            {"content": "Part 2", "idea_id": "same_id"},
        ]
        block, mapping = _format_chunks_with_ids(chunks)
        # Both chunks must get distinct keys in the mapping
        assert len(mapping) == 2
        assert "same_id" in mapping
        assert "same_id#2" in mapping


# ════════════════════════════════════════════════════════════════════
# Teaching Graph self-reflection fallback
# ════════════════════════════════════════════════════════════════════

class TestReviewRouter:
    def test_grounded_routes_to_end(self):
        from agentic.schemas import ReviewResult
        from agentic.teaching.reviewer import review_router
        state = {"review": ReviewResult(grounded=True, score=1.0, feedback=""), "narrator_retries": 0}
        assert review_router(state) == "end"

    def test_ungrounded_under_budget_routes_to_retry(self):
        from agentic.schemas import ReviewResult
        from agentic.teaching.reviewer import review_router
        state = {"review": ReviewResult(grounded=False, score=0.0, feedback="off-source"), "narrator_retries": 0}
        assert review_router(state) == "retry"

    def test_ungrounded_at_budget_routes_to_fallback(self):
        from agentic.schemas import ReviewResult
        from agentic.teaching.reviewer import review_router, MAX_RETRIES
        state = {
            "review": ReviewResult(grounded=False, score=0.0, feedback="still hallucinating"),
            "narrator_retries": MAX_RETRIES,
        }
        # Previous behavior: silent end. New behavior: explicit fallback.
        assert review_router(state) == "fallback"

    def test_missing_review_routes_to_end(self):
        from agentic.teaching.reviewer import review_router
        state = {"narrator_retries": 0}
        assert review_router(state) == "end"


class TestFallbackNarrator:
    def test_no_slide_returns_honest_no_source_reply(self):
        from agentic.teaching.fallback import FallbackNarratorAgent

        class FakeBrain:
            def ask(self, *a, **k):
                raise AssertionError("brain.ask should NOT be called when slide is empty")

        agent = FallbackNarratorAgent(FakeBrain())
        out = agent({"language": "fr", "last_slide_content": ""})
        assert "pas assez d'éléments" in out["answer"].lower() or "pas assez" in out["answer"].lower()
        # Action annotated as fallback
        actions = out["actions"]
        assert any(getattr(a, "payload", {}).get("fallback") is True for a in actions)
        # Review marked as grounded by construction
        assert out["review"].grounded is True

    def test_llm_paraphrase_is_used_when_json_valid(self):
        from agentic.teaching.fallback import FallbackNarratorAgent

        class FakeBrain:
            def ask(self, *a, **k):
                return ('{"answer": "K-NN classifies points by majority vote of neighbors."}', 0.1)

        agent = FallbackNarratorAgent(FakeBrain())
        out = agent({"language": "en", "last_slide_content": "K-NN: classify by majority vote of K nearest neighbors."})
        assert "K-NN" in out["answer"]
        assert "majority vote" in out["answer"]
        # Action annotated correctly
        kinds = [getattr(a, "payload", {}).get("kind") for a in out["actions"]]
        assert "llm_paraphrase" in kinds

    def test_verbatim_fallback_when_llm_returns_garbage(self):
        from agentic.teaching.fallback import FallbackNarratorAgent

        class FakeBrain:
            def ask(self, *a, **k):
                return ("not valid json at all", 0.1)

        slide = "Newton's second law: F = m × a."
        agent = FallbackNarratorAgent(FakeBrain())
        out = agent({"language": "en", "last_slide_content": slide})
        # Verbatim fallback prefixes with "Here is what the slide says:" + slide content
        assert "F = m × a" in out["answer"] or "Newton" in out["answer"]
        kinds = [getattr(a, "payload", {}).get("kind") for a in out["actions"]]
        assert "verbatim" in kinds

    def test_verbatim_fallback_when_llm_raises(self):
        from agentic.teaching.fallback import FallbackNarratorAgent

        class FakeBrain:
            def ask(self, *a, **k):
                raise RuntimeError("LLM down")

        slide = "Pythagoras: a² + b² = c²."
        agent = FallbackNarratorAgent(FakeBrain())
        out = agent({"language": "fr", "last_slide_content": slide})
        # The verbatim path triggers; output should contain part of the slide
        assert "Pythagoras" in out["answer"] or "a²" in out["answer"]


# ════════════════════════════════════════════════════════════════════
# Retriever — graph-augmented context expansion
# ════════════════════════════════════════════════════════════════════

class TestRetrieverKGAugmentation:
    """The retriever extends the chunk pool with direct prerequisites from
    the IdeaGraph for the top retrieved chunk(s). Augmented chunks are
    flagged ``_via_kg=True`` so the responder can trace them.
    """

    def _build_kg_with_prereq(self):
        """Build a tiny IdeaGraph: target depends on prereq."""
        from pedagogy.knowledge_graph.graph import IdeaGraph, IdeaNode
        g = IdeaGraph()
        g.add_node(IdeaNode(
            idea_id="prereq_id", label="Prereq", text="Prereq content",
            course_id="c", chapter_idx=1, section_idx=0,
        ))
        g.add_node(IdeaNode(
            idea_id="target_id", label="Target", text="Target content",
            course_id="c", chapter_idx=2, section_idx=0,
            depends_on_ids={"prereq_id"},
        ))
        return g

    def test_augment_pulls_prereqs_for_top_chunk(self, monkeypatch):
        from agentic.qa.retriever import RetrieverAgent
        graph = self._build_kg_with_prereq()
        monkeypatch.setattr(
            "pedagogy.knowledge_graph.get_or_build",
            lambda rag: graph,
        )
        agent = RetrieverAgent(rag=object())
        retrieved = [{"idea_id": "target_id", "content": "...", "score": 0.9}]
        augmented = agent._augment_with_prereqs(retrieved)
        assert len(augmented) == 1
        assert augmented[0]["idea_id"] == "prereq_id"
        assert augmented[0]["_via_kg"] is True
        assert augmented[0]["_augmented_from"] == "target_id"

    def test_augment_skips_prereqs_already_retrieved(self, monkeypatch):
        from agentic.qa.retriever import RetrieverAgent
        graph = self._build_kg_with_prereq()
        monkeypatch.setattr(
            "pedagogy.knowledge_graph.get_or_build",
            lambda rag: graph,
        )
        agent = RetrieverAgent(rag=object())
        # Both target and prereq already in retrieval pool
        retrieved = [
            {"idea_id": "target_id", "content": "...", "score": 0.9},
            {"idea_id": "prereq_id", "content": "...", "score": 0.7},
        ]
        augmented = agent._augment_with_prereqs(retrieved)
        # Prereq is already in the pool → no augmentation needed
        assert augmented == []

    def test_augment_returns_empty_when_no_idea_ids(self, monkeypatch):
        from agentic.qa.retriever import RetrieverAgent
        graph = self._build_kg_with_prereq()
        monkeypatch.setattr(
            "pedagogy.knowledge_graph.get_or_build",
            lambda rag: graph,
        )
        agent = RetrieverAgent(rag=object())
        # Legacy chunks without idea_id (pre-graph era)
        retrieved = [{"content": "...", "score": 0.9}]
        augmented = agent._augment_with_prereqs(retrieved)
        assert augmented == []

    def test_augment_returns_empty_when_kg_empty(self, monkeypatch):
        from agentic.qa.retriever import RetrieverAgent
        from pedagogy.knowledge_graph.graph import IdeaGraph
        empty_graph = IdeaGraph()
        monkeypatch.setattr(
            "pedagogy.knowledge_graph.get_or_build",
            lambda rag: empty_graph,
        )
        agent = RetrieverAgent(rag=object())
        retrieved = [{"idea_id": "target_id", "content": "...", "score": 0.9}]
        augmented = agent._augment_with_prereqs(retrieved)
        assert augmented == []

    def test_augment_caps_at_total_limit(self, monkeypatch):
        from agentic.qa.retriever import RetrieverAgent, _KG_TOTAL_AUGMENT_CAP
        from pedagogy.knowledge_graph.graph import IdeaGraph, IdeaNode

        # Build a graph with many prereqs of the target
        g = IdeaGraph()
        prereq_ids = [f"p{i}" for i in range(_KG_TOTAL_AUGMENT_CAP + 5)]
        for pid in prereq_ids:
            g.add_node(IdeaNode(idea_id=pid, label=pid, text=f"text {pid}"))
        g.add_node(IdeaNode(
            idea_id="target_id", label="Target", text="...",
            depends_on_ids=set(prereq_ids),
        ))
        monkeypatch.setattr(
            "pedagogy.knowledge_graph.get_or_build",
            lambda rag: g,
        )
        agent = RetrieverAgent(rag=object())
        retrieved = [{"idea_id": "target_id", "content": "...", "score": 0.9}]
        augmented = agent._augment_with_prereqs(retrieved)
        # Capped — never exceeds the global limit
        assert len(augmented) <= _KG_TOTAL_AUGMENT_CAP


# ════════════════════════════════════════════════════════════════════
# IntentAgent — feedback_polarity in payload
# ════════════════════════════════════════════════════════════════════
#
# These tests bypass SIGHT (which fires on any text with confusion
# semantics) by patching ``_sight_predict`` to None. The feedback
# polarity field is produced by the LLM tier — that's what we verify.

class TestFeedbackPolarity:
    @staticmethod
    def _disable_sight(monkeypatch):
        monkeypatch.setattr("agentic.qa.intent._sight_predict", None)

    def test_llm_feedback_negative_propagated_to_payload(self, monkeypatch):
        self._disable_sight(monkeypatch)
        from agentic.qa.intent import IntentAgent

        class FakeBrain:
            def ask(self, *a, **k):
                return ('{"intent": "feedback", "confidence": 0.9, "feedback_polarity": "negative"}', 0.1)

        agent = IntentAgent(FakeBrain())
        out = agent({"event_payload": {"text": "non, pas clair"}})
        intent = out["intent"]
        assert intent.type == "feedback"
        assert intent.payload.get("feedback_polarity") == "negative"

    def test_llm_feedback_positive_propagated_to_payload(self, monkeypatch):
        self._disable_sight(monkeypatch)
        from agentic.qa.intent import IntentAgent

        class FakeBrain:
            def ask(self, *a, **k):
                return ('{"intent": "feedback", "confidence": 0.9, "feedback_polarity": "positive"}', 0.1)

        agent = IntentAgent(FakeBrain())
        out = agent({"event_payload": {"text": "ok"}})
        intent = out["intent"]
        assert intent.type == "feedback"
        assert intent.payload.get("feedback_polarity") == "positive"

    def test_invalid_polarity_defaults_to_positive(self, monkeypatch):
        self._disable_sight(monkeypatch)
        from agentic.qa.intent import IntentAgent

        class FakeBrain:
            def ask(self, *a, **k):
                return ('{"intent": "feedback", "confidence": 0.9, "feedback_polarity": "weird_value"}', 0.1)

        agent = IntentAgent(FakeBrain())
        out = agent({"event_payload": {"text": "ok"}})
        # Defaults to "positive" (safer interpretation: don't replay)
        assert out["intent"].payload.get("feedback_polarity") == "positive"

    def test_non_feedback_intent_has_no_polarity(self, monkeypatch):
        self._disable_sight(monkeypatch)
        from agentic.qa.intent import IntentAgent

        class FakeBrain:
            def ask(self, *a, **k):
                return ('{"intent": "question", "confidence": 0.9, "needs_retrieval": true}', 0.1)

        agent = IntentAgent(FakeBrain())
        out = agent({"event_payload": {"text": "what is k-NN?"}})
        # Polarity field should not be set for non-feedback intents
        assert "feedback_polarity" not in out["intent"].payload


# ════════════════════════════════════════════════════════════════════
# Agentic graph observability — OpenTelemetry instrumentation
# ════════════════════════════════════════════════════════════════════

class TestObservability:
    """The OTel instrumentation must be a no-op when the package isn't
    installed (the test environment doesn't ship OTel). The tests verify
    that all public functions remain safe in that mode."""

    def test_node_span_is_no_op_without_otel(self):
        from agentic.observability import node_span, is_available
        # Test environment doesn't ship OTel — the module should detect this
        with node_span("test_node") as span:
            assert span is None or not is_available()

    def test_attach_node_attrs_safe_with_none_span(self):
        from agentic.observability import attach_node_attrs
        # Should never raise, even when span is None
        attach_node_attrs(None, "intent", {"intent": object()})
        attach_node_attrs(None, "retriever", {"retrieved_chunks": []})
        attach_node_attrs(None, "responder", {"answer": "x", "citations": []})

    def test_attach_node_attrs_safe_with_non_dict_result(self):
        from agentic.observability import attach_node_attrs
        # Should never raise on weird inputs
        attach_node_attrs(None, "intent", None)
        attach_node_attrs(None, "intent", "not a dict")
        attach_node_attrs(None, "intent", [])

    def test_setup_tracing_returns_false_without_otel(self):
        from agentic.observability import setup_tracing, is_available
        # When OTel is not installed, setup_tracing must return False
        # (and not crash). When it IS installed, it returns True.
        result = setup_tracing(service_name="smartteacher-test", console=True)
        assert isinstance(result, bool)
        # If OTel is available, setup should succeed; otherwise False
        if is_available():
            assert result is True
        else:
            assert result is False


# ════════════════════════════════════════════════════════════════════
# Math → spoken text conversion (output-time, replaces wordification)
# ════════════════════════════════════════════════════════════════════

class TestMathSpeech:
    """The math_speech module verbalizes math notation at TTS output time
    (replacing the old index-time _wordify_math). Math symbols stay in
    the index for retrieval precision; only the spoken output is
    transformed.
    """

    def test_unicode_greek_french(self):
        from audio.math_speech import to_speech
        out = to_speech("Soit α + β = 1", "fr")
        assert "alpha" in out
        # β is "beta" — French entry is "bêta" with circumflex
        assert "bêta" in out or "beta" in out

    def test_unicode_greek_english(self):
        from audio.math_speech import to_speech
        out = to_speech("Let α + β = 1", "en")
        assert "alpha" in out
        assert "beta" in out

    def test_superscript_squared(self):
        from audio.math_speech import to_speech
        # x² in French → "au carré"
        out_fr = to_speech("x² ≥ 0", "fr")
        assert "carré" in out_fr or "carre" in out_fr.lower()
        # x² in English → "squared"
        out_en = to_speech("x² >= 0", "en")
        assert "squared" in out_en

    def test_set_theory(self):
        from audio.math_speech import to_speech
        out = to_speech("∀ x ∈ ℝ", "fr")
        assert "pour tout" in out.lower()
        assert "appartient" in out.lower()
        # ℝ (R) is verbalized as "R" verbatim
        assert "R" in out

    def test_latex_block_fraction(self):
        from audio.math_speech import to_speech
        out = to_speech("La formule $\\frac{a}{b}$", "fr")
        assert "divisé" in out or "divise" in out.lower()
        assert "\\frac" not in out  # LaTeX command consumed
        assert "a" in out
        assert "b" in out

    def test_latex_block_sqrt(self):
        from audio.math_speech import to_speech
        out = to_speech("$\\sqrt{16}$", "en")
        assert "square root" in out.lower()
        assert "16" in out
        assert "\\sqrt" not in out

    def test_latex_command_sum(self):
        from audio.math_speech import to_speech
        out = to_speech("$\\sum x_i$", "en")
        assert "sum" in out.lower()
        assert "\\sum" not in out

    def test_idempotent_on_plain_text(self):
        from audio.math_speech import to_speech
        plain = "This is a plain English sentence with no math notation."
        assert to_speech(plain, "en") == plain

    def test_unknown_lang_falls_back_to_french(self):
        from audio.math_speech import to_speech
        # Unknown language code → defaults to one of fr/en deterministically
        out = to_speech("α = 1", "zz")
        # Output must still contain a verbalization, not the raw symbol
        assert "α" not in out
        assert "alpha" in out

    def test_empty_input(self):
        from audio.math_speech import to_speech
        assert to_speech("", "fr") == ""
        assert to_speech("", "en") == ""


# NOTE: TestStyleToParams + TestParamsToDirectives removed alongside
# their underlying modules (style_to_params, _params_to_directives).
# Those numerical mappings (max_sentences=5/6/7, must_include arrays) had
# no empirical or theoretical justification and were retired. The narrator
# now uses only the prose hint via style_to_prompt_hint.


# ════════════════════════════════════════════════════════════════════
# fire_signal helper (runtime hooks) — must never raise on bad input
# ════════════════════════════════════════════════════════════════════

class TestFireSignal:
    def test_none_student_id_skipped(self):
        # No event loop required — must short-circuit before any async work
        assert fire_signal(None, "audio_question") is None

    def test_empty_string_skipped(self):
        assert fire_signal("", "audio_question") is None

    def test_unknown_signal_skipped(self):
        assert fire_signal("11111111-1111-1111-1111-111111111111", "unknown_signal") is None

    def test_invalid_uuid_skipped(self):
        assert fire_signal("not-a-uuid", "audio_question") is None

    def test_no_running_loop_returns_none(self):
        # Called from sync test → no running loop → must return None, not raise
        assert fire_signal("11111111-1111-1111-1111-111111111111", "audio_question") is None


# ════════════════════════════════════════════════════════════════════
# VARK questionnaire
# ════════════════════════════════════════════════════════════════════

class TestVARKQuestionnaire:
    def test_eight_questions(self):
        assert len(VARK_QUESTIONS) == 8

    def test_each_question_has_four_options(self):
        for q in VARK_QUESTIONS:
            assert set(q.options.keys()) == set(STYLES)
            for opt in q.options.values():
                assert "fr" in opt and "en" in opt

    def test_score_responses_simple(self):
        responses = {"q1": ["visual"], "q2": ["visual"], "q3": ["auditory"]}
        counts = score_responses(responses)
        assert counts == {"visual": 2, "auditory": 1, "kinesthetic": 0, "reading": 0}

    def test_score_multi_choice(self):
        responses = {"q1": ["visual", "kinesthetic"]}
        counts = score_responses(responses)
        assert counts == {"visual": 1, "auditory": 0, "kinesthetic": 1, "reading": 0}

    def test_score_ignores_invalid_qid(self):
        counts = score_responses({"q_invalid": ["visual"]})
        assert sum(counts.values()) == 0

    def test_serialize_fr_and_en(self):
        fr = serialize_for_frontend(lang="fr")
        en = serialize_for_frontend(lang="en")
        assert len(fr) == 8 and len(en) == 8
        # Texts differ between languages
        assert fr[0]["text"] != en[0]["text"]


# ════════════════════════════════════════════════════════════════════
# Password strength
# ════════════════════════════════════════════════════════════════════

class TestPasswordStrength:
    def test_too_short(self):
        with pytest.raises(PasswordStrengthError):
            check_password_strength("short1")

    def test_no_digits(self):
        with pytest.raises(PasswordStrengthError):
            check_password_strength("OnlyLetters")

    def test_no_letters(self):
        with pytest.raises(PasswordStrengthError):
            check_password_strength("12345678")

    def test_common_password(self):
        with pytest.raises(PasswordStrengthError):
            check_password_strength("Password1")  # case-insensitive

    def test_acceptable(self):
        check_password_strength("MyStrongPass2024")     # OK
        check_password_strength("aZ12bxbb")              # OK


# ════════════════════════════════════════════════════════════════════
# Auth helpers (bcrypt + JWT)
# ════════════════════════════════════════════════════════════════════

class TestAuthHelpers:
    def test_hash_verify_roundtrip(self):
        h = hash_password("myPassword123")
        assert verify_password("myPassword123", h) is True
        assert verify_password("wrong", h) is False

    def test_jwt_roundtrip(self):
        token = create_access_token("uuid-abc", "test@test.fr", "student")
        claims = decode_access_token(token)
        assert claims["sub"] == "uuid-abc"
        assert claims["email"] == "test@test.fr"
        assert claims["account_level"] == "student"

    def test_jwt_invalid_signature_rejected(self):
        from fastapi import HTTPException
        token = create_access_token("u", "e@e.fr", "student")
        # Tamper with the token (flip a char in signature part)
        tampered = token[:-3] + ("xxx" if token[-3:] != "xxx" else "yyy")
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(tampered)
        assert exc_info.value.status_code == 401
