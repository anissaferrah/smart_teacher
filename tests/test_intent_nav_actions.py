"""Tests for the granular navigation sub-actions in IntentAgent.

Coverage :
  - Each of the 7 valid nav_actions is preserved when the LLM returns it.
  - Unknown nav_action collapses to the safe default "next".
  - Non-navigation intents don't get a nav_action populated.
  - go_to_concept correctly reads + caps nav_target (≤ 80 chars).

The LLM call is mocked — these tests verify the parsing/validation
contract, not the LLM's classification quality.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from agentic.qa.intent import IntentAgent, _VALID_NAV_ACTIONS


def _state(text: str = "passe à la suivante", lang: str = "fr") -> dict:
    return {
        "event_payload": {"text": text},
        "language": lang,
        "last_slide_content": "Slide content",
        "history": [],
    }


def _llm_reply(intent_type: str, **fields) -> str:
    """Build a fake LLM JSON output."""
    base = {
        "intent": intent_type,
        "confidence": 0.92,
        "needs_retrieval": False,
    }
    base.update(fields)
    return json.dumps(base)


@pytest.fixture
def agent_with_mock_llm():
    """IntentAgent wired to a mock brain.ask + SIGHT model disabled."""
    brain = MagicMock()
    agent = IntentAgent(brain)
    # SIGHT may be loaded as a side-effect ; force it off so we always
    # exercise the LLM path that the tests are about.
    with patch("agentic.qa.intent._sight_predict", None):
        yield agent, brain


# ════════════════════════════════════════════════════════════════════
# Each valid nav_action survives parsing
# ════════════════════════════════════════════════════════════════════

class TestValidNavActions:
    """Each of the 7 valid actions returned by the LLM ends up in payload."""

    @pytest.mark.parametrize("action", sorted(_VALID_NAV_ACTIONS))
    def test_action_preserved(self, agent_with_mock_llm, action):
        agent, brain = agent_with_mock_llm
        brain.ask.return_value = (
            _llm_reply("navigation", nav_action=action, nav_target=""),
            0.05,
        )
        result = agent(_state())

        intent = result["intent"]
        assert intent.type == "navigation"
        assert intent.payload["nav_action"] == action


# ════════════════════════════════════════════════════════════════════
# Unknown / missing nav_action → safe default
# ════════════════════════════════════════════════════════════════════

class TestNavActionFallback:
    """Unknown LLM output collapses to "next" rather than crashing."""

    def test_unknown_action_collapses_to_next(self, agent_with_mock_llm):
        agent, brain = agent_with_mock_llm
        brain.ask.return_value = (
            _llm_reply("navigation", nav_action="dance_around", nav_target=""),
            0.05,
        )
        result = agent(_state())

        assert result["intent"].payload["nav_action"] == "next"

    def test_missing_nav_action_collapses_to_next(self, agent_with_mock_llm):
        """LLM forgot the field → also collapses to next."""
        agent, brain = agent_with_mock_llm
        brain.ask.return_value = (_llm_reply("navigation"), 0.05)
        result = agent(_state())

        assert result["intent"].payload["nav_action"] == "next"

    def test_uppercase_action_normalised(self, agent_with_mock_llm):
        """LLM returning 'PREVIOUS' should still be accepted (lowercased)."""
        agent, brain = agent_with_mock_llm
        brain.ask.return_value = (
            _llm_reply("navigation", nav_action="PREVIOUS", nav_target=""),
            0.05,
        )
        result = agent(_state())
        assert result["intent"].payload["nav_action"] == "previous"


# ════════════════════════════════════════════════════════════════════
# go_to_concept reads nav_target + cap
# ════════════════════════════════════════════════════════════════════

class TestNavTarget:
    """go_to_concept needs the named-concept payload."""

    def test_go_to_concept_reads_target(self, agent_with_mock_llm):
        agent, brain = agent_with_mock_llm
        brain.ask.return_value = (
            _llm_reply("navigation",
                       nav_action="go_to_concept",
                       nav_target="K-means clustering"),
            0.05,
        )
        result = agent(_state())

        payload = result["intent"].payload
        assert payload["nav_action"] == "go_to_concept"
        assert payload["nav_target"] == "K-means clustering"

    def test_target_capped_at_80_chars(self, agent_with_mock_llm):
        """A very long target name shouldn't crash or pollute downstream."""
        long_name = "x" * 200
        agent, brain = agent_with_mock_llm
        brain.ask.return_value = (
            _llm_reply("navigation",
                       nav_action="go_to_concept",
                       nav_target=long_name),
            0.05,
        )
        result = agent(_state())

        target = result["intent"].payload["nav_target"]
        assert len(target) <= 80

    def test_non_concept_actions_have_empty_target(self, agent_with_mock_llm):
        """Only go_to_concept uses nav_target ; others get ""."""
        agent, brain = agent_with_mock_llm
        brain.ask.return_value = (
            _llm_reply("navigation", nav_action="next",
                       nav_target="should be ignored"),
            0.05,
        )
        result = agent(_state())

        assert result["intent"].payload["nav_action"] == "next"
        assert result["intent"].payload["nav_target"] == ""


# ════════════════════════════════════════════════════════════════════
# Non-navigation intents don't carry nav_action
# ════════════════════════════════════════════════════════════════════

class TestNonNavigationIntents:
    """Question / feedback / confusion_signal must NOT have nav_action set."""

    def test_question_intent_no_nav_action(self, agent_with_mock_llm):
        agent, brain = agent_with_mock_llm
        brain.ask.return_value = (
            _llm_reply(
                "question",
                anchored_concept="K-means",
                needed_rewrite=False,
                rewritten="What is K-means?",
            ),
            0.05,
        )
        result = agent(_state(text="What is K-means?"))

        payload = result["intent"].payload
        assert "nav_action" not in payload
        assert "nav_target" not in payload

    def test_feedback_intent_no_nav_action(self, agent_with_mock_llm):
        agent, brain = agent_with_mock_llm
        brain.ask.return_value = (
            _llm_reply("feedback", feedback_polarity="positive"),
            0.05,
        )
        result = agent(_state(text="oui d'accord"))

        payload = result["intent"].payload
        assert "nav_action" not in payload
        # but feedback_polarity must be present
        assert payload.get("feedback_polarity") == "positive"
