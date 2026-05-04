"""Tests for the Q&A reviewer (self-correction layer).

Covers the contract that matters:

  - The reviewer accepts a clearly grounded answer (no LLM call needed
    for the obvious cases — fast paths).
  - The reviewer rejects an answer that hallucinates content not in
    the slide / chunks.
  - The reviewer accepts an honest "this is not covered in this course"
    refusal as a valid outcome (so the LLM isn't penalised for following
    the course-bound rule).
  - The router maps {grounded, ungrounded+retries-left,
    ungrounded+no-retries} to {end, retry, fallback}.
  - The fallback node produces the deterministic safe answer in the
    expected language.

We mock the LLM (``brain.ask``) so the tests are deterministic and don't
need network / Ollama.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agentic.qa.reviewer import (
    MAX_RETRIES,
    QAReviewAgent,
    qa_fallback_node,
    qa_review_router,
)
from agentic.schemas import ReviewResult, VoiceIntent


# ════════════════════════════════════════════════════════════════════
# Fast paths that DON'T call the LLM
# ════════════════════════════════════════════════════════════════════

class TestReviewerFastPaths:
    """The reviewer should short-circuit on obvious cases without LLM cost."""

    def test_empty_answer_flagged_for_retry(self):
        """An empty answer cannot be grounded — flag for retry, no LLM."""
        brain = MagicMock()
        agent = QAReviewAgent(brain)

        result = agent({
            "answer": "",
            "last_slide_content": "Some slide",
            "language": "fr",
        })

        assert result["review"].grounded is False
        assert result["review"].score == 0.0
        brain.ask.assert_not_called()

    def test_honest_refusal_fr_accepted(self):
        """If the responder says 'pas abordé dans ce cours', accept directly."""
        brain = MagicMock()
        agent = QAReviewAgent(brain)

        result = agent({
            "answer": "Ce point n'est pas abordé dans ce cours.",
            "last_slide_content": "K-means clustering...",
            "language": "fr",
        })

        assert result["review"].grounded is True
        brain.ask.assert_not_called()

    def test_honest_refusal_en_accepted(self):
        """English variant of the honest refusal."""
        brain = MagicMock()
        agent = QAReviewAgent(brain)

        result = agent({
            "answer": "This is not covered in this course.",
            "last_slide_content": "Backpropagation...",
            "language": "en",
        })

        assert result["review"].grounded is True
        brain.ask.assert_not_called()

    def test_no_source_to_verify_passes_through(self):
        """When there's no slide AND no chunks, there's nothing to verify."""
        brain = MagicMock()
        agent = QAReviewAgent(brain)

        result = agent({
            "answer": "Some answer.",
            "last_slide_content": "",
            "retrieved_chunks": [],
            "language": "fr",
        })

        assert result["review"].grounded is True
        brain.ask.assert_not_called()


# ════════════════════════════════════════════════════════════════════
# LLM verdict propagation
# ════════════════════════════════════════════════════════════════════

class TestReviewerLLMVerdict:
    """When the LLM is called, its grounded/feedback verdict flows through."""

    def _state(self, answer: str = "An answer.") -> dict:
        return {
            "answer": answer,
            "last_slide_content": "Slide text.",
            "retrieved_chunks": [{"content": "chunk text"}],
            "language": "fr",
            "intent": VoiceIntent(type="question", confidence=0.9, payload={"raw_text": "what?"}),
        }

    def test_grounded_verdict_passes_through(self):
        brain = MagicMock()
        brain.ask.return_value = (json.dumps({"grounded": True, "feedback": "ok"}), 0.1)
        agent = QAReviewAgent(brain)

        result = agent(self._state())

        assert result["review"].grounded is True
        assert result["review"].score == 1.0
        brain.ask.assert_called_once()

    def test_ungrounded_verdict_propagates_feedback(self):
        brain = MagicMock()
        brain.ask.return_value = (
            json.dumps({"grounded": False, "feedback": "claim X is not in the source"}),
            0.1,
        )
        agent = QAReviewAgent(brain)

        result = agent(self._state())

        assert result["review"].grounded is False
        assert result["review"].score == 0.0
        assert "claim X" in result["review"].feedback

    def test_invalid_json_defaults_to_pass(self):
        """A malformed reviewer response shouldn't block the student.

        The user is waiting for an answer; routing them through fallback
        because the *reviewer* (a secondary check) crashed would punish
        the user for our infra problem. PASS is the safe default.
        """
        brain = MagicMock()
        brain.ask.return_value = ("not json at all", 0.1)
        agent = QAReviewAgent(brain)

        result = agent(self._state())

        assert result["review"].grounded is True

    def test_llm_exception_defaults_to_pass(self):
        """Same reasoning when the LLM call itself raises."""
        brain = MagicMock()
        brain.ask.side_effect = RuntimeError("ollama timeout")
        agent = QAReviewAgent(brain)

        result = agent(self._state())

        assert result["review"].grounded is True


# ════════════════════════════════════════════════════════════════════
# Routing decisions
# ════════════════════════════════════════════════════════════════════

class TestRouter:
    """qa_review_router maps the verdict + retry budget to {end, retry, fallback}."""

    def test_grounded_routes_to_end(self):
        state = {
            "review": ReviewResult(grounded=True, score=1.0, feedback=""),
            "responder_retries": 0,
        }
        assert qa_review_router(state) == "end"

    def test_ungrounded_first_failure_routes_to_retry(self):
        state = {
            "review": ReviewResult(grounded=False, score=0.0, feedback="..."),
            "responder_retries": 0,
        }
        assert qa_review_router(state) == "retry"

    def test_ungrounded_at_max_retries_routes_to_fallback(self):
        """At MAX_RETRIES we stop trying and fall back to a safe answer."""
        state = {
            "review": ReviewResult(grounded=False, score=0.0, feedback="..."),
            "responder_retries": MAX_RETRIES,
        }
        assert qa_review_router(state) == "fallback"

    def test_missing_review_routes_to_end(self):
        """No review → assume the upstream node decided to pass through."""
        assert qa_review_router({}) == "end"


# ════════════════════════════════════════════════════════════════════
# Fallback node
# ════════════════════════════════════════════════════════════════════

class TestFallbackNode:
    """The deterministic safe answer node."""

    def test_fallback_french(self):
        out = qa_fallback_node({"language": "fr"})
        assert "pas sûr" in out["answer"]
        assert out["citations"] == []
        assert out["confidence"] == 0.0

    def test_fallback_english(self):
        out = qa_fallback_node({"language": "en"})
        assert "not sure" in out["answer"].lower()
        assert out["citations"] == []
        assert out["confidence"] == 0.0

    def test_fallback_unknown_lang_falls_back_to_french(self):
        """Defensive: an unsupported language picks the FR default rather
        than crashing — the student gets *some* answer."""
        out = qa_fallback_node({"language": "es"})
        assert out["answer"]  # non-empty
