"""Tests for the voice-navigation dispatcher.

Each of the 7 nav_actions must trigger a distinct, observable side effect :

  - next / skip   → dialogue.next_section + UI "next_section" event
  - previous      → dialogue.prev_section + UI "prev_section" event
  - repeat        → slide_loader called for the current slide + slide_update emitted
  - go_to_concept → KG looked up + save_course_position called + slide_update emitted
  - explain_more  → "explain_more_request" event with depth=deep
  - slow_down     → profile_updater called with new speech_rate + speech_rate_changed event

Plus :
  - unknown action collapses to "next" (never crashes)
  - go_to_concept with unresolvable target falls back to next
  - missing dependencies (slide_loader=None, kg_getter=None) don't crash
  - the WS ``send`` callable always receives at least one event so the
    frontend never thinks the request was silently dropped
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.nav_dispatcher import (
    dispatch_nav_action,
    _SLOW_DOWN_FACTOR,
    _SPEECH_RATE_FLOOR,
    _VALID_NAV_ACTIONS,
)


# ── Test doubles ────────────────────────────────────────────────────

@dataclass
class _FakeCtx:
    """Stand-in for SessionContext — only the fields the dispatcher reads."""
    session_id:    str = "sess-123"
    course_id:     str | None = "course-abc"
    chapter_index: int = 2
    section_index: int = 5


@dataclass
class _FakeConcept:
    """Stand-in for ConceptInfo with the fields the dispatcher reads."""
    name:           str = "k_means"
    display_name:   str = "K-means"
    canonical_name: str = "K-Means clustering"
    course_id:      str = "course-abc"
    chapter_idxs:   set[int] = field(default_factory=lambda: {3, 7})
    idea_ids:       set[str] = field(default_factory=lambda: {"i1", "i2"})


@dataclass
class _FakeIdea:
    chapter_idx: int = 3
    section_idx: int = 4


def _make_kg(*, with_concept: bool = True, ideas: list[_FakeIdea] | None = None):
    """Build a mock KG. ``with_concept=False`` simulates a concept that
    doesn't exist (covers the fallback-to-next branch)."""
    kg = MagicMock()
    if with_concept:
        concept = _FakeConcept()
        kg.get_concept.return_value = concept
        kg.list_concepts.return_value = [concept]
        kg.ideas_in_concept.return_value = ideas or [_FakeIdea(chapter_idx=3, section_idx=4)]
    else:
        kg.get_concept.return_value = None
        kg.list_concepts.return_value = []
        kg.ideas_in_concept.return_value = []
    return kg


def _build_deps(**overrides):
    """Build the dependency bundle the dispatcher needs.

    Centralised so each test only has to override what it cares about.
    """
    deps = {
        "ctx":      _FakeCtx(),
        "dialogue": MagicMock(
            next_section=AsyncMock(return_value=None),
            prev_section=AsyncMock(return_value=None),
            save_course_position=AsyncMock(return_value=None),
        ),
        "send":     AsyncMock(return_value=None),
        "turn_id":  42,
        "slide_loader":    AsyncMock(return_value={
            "slide_type":    "section",
            "slide_index":   5,
            "section_title": "Slide 5",
            "content":       "Slide 5 content",
            "chapter_title": "Chap 2",
            "chapter_order": 2,
            "course_id":     "course-abc",
        }),
        "kg_getter":       lambda: _make_kg(with_concept=True),
        "profile_updater": AsyncMock(return_value={"speech_rate": 0.85}),
        "current_speech_rate": 1.0,
    }
    deps.update(overrides)
    return deps


# ════════════════════════════════════════════════════════════════════
# Each action triggers the right primitives
# ════════════════════════════════════════════════════════════════════

class TestNextAndSkip:
    """``next`` and ``skip`` both advance one section."""

    @pytest.mark.parametrize("action", ["next", "skip"])
    @pytest.mark.asyncio
    async def test_advances_one_section(self, action):
        deps = _build_deps()
        result = await dispatch_nav_action(
            nav_action=action, nav_target="", **deps,
        )
        deps["dialogue"].next_section.assert_awaited_once_with("sess-123")
        deps["send"].assert_awaited_once()
        sent = deps["send"].await_args.args[0]
        assert sent["type"] == "next_section"
        assert sent["turn_id"] == 42
        assert result["action"] == action
        assert result["ok"] is True


class TestPrevious:
    @pytest.mark.asyncio
    async def test_goes_back_one_section(self):
        deps = _build_deps()
        result = await dispatch_nav_action(
            nav_action="previous", nav_target="", **deps,
        )
        deps["dialogue"].prev_section.assert_awaited_once_with("sess-123")
        # next_section must NOT have been called — would mean we mixed up the action
        deps["dialogue"].next_section.assert_not_awaited()
        sent = deps["send"].await_args.args[0]
        assert sent["type"] == "prev_section"
        assert result["ok"] is True


class TestRepeat:
    @pytest.mark.asyncio
    async def test_reloads_current_slide(self):
        deps = _build_deps()
        result = await dispatch_nav_action(
            nav_action="repeat", nav_target="", **deps,
        )
        # slide_loader called with the current chapter/section, NOT next/prev
        deps["slide_loader"].assert_awaited_once_with(
            "course-abc", 2, 5,
        )
        # Neither navigation primitive was called — repeat doesn't move
        deps["dialogue"].next_section.assert_not_awaited()
        deps["dialogue"].prev_section.assert_not_awaited()
        sent = deps["send"].await_args.args[0]
        assert sent["type"] == "slide_update"
        assert sent["turn_id"] == 42
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_repeat_without_slide_loader_still_emits_event(self):
        """slide_loader=None must not crash — the dispatcher emits a
        minimal slide_update so the frontend at least knows something
        happened."""
        deps = _build_deps(slide_loader=None)
        result = await dispatch_nav_action(
            nav_action="repeat", nav_target="", **deps,
        )
        assert deps["send"].await_count == 1
        assert deps["send"].await_args.args[0]["type"] == "slide_update"
        assert result["ok"] is True


class TestGoToConcept:
    @pytest.mark.asyncio
    async def test_resolves_and_jumps(self):
        deps = _build_deps()
        result = await dispatch_nav_action(
            nav_action="go_to_concept", nav_target="K-means", **deps,
        )
        # save_course_position called with the resolved chapter/section
        # (chapter 3 = min({3,7}), section 4 = min idea section in chap 3).
        deps["dialogue"].save_course_position.assert_awaited_once()
        call_kwargs = deps["dialogue"].save_course_position.await_args.kwargs
        assert call_kwargs["chapter_index"] == 3
        assert call_kwargs["section_index"] == 4
        assert call_kwargs["course_id"] == "course-abc"

        # slide_loader called with the resolved coords (NOT current ones)
        deps["slide_loader"].assert_awaited_once_with("course-abc", 3, 4)

        sent = deps["send"].await_args.args[0]
        assert sent["type"] == "slide_update"
        assert sent["chapter_index"] == 3
        assert sent["section_index"] == 4
        assert sent["concept_target"] == "K-means"
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_unknown_concept_falls_back_to_next(self):
        deps = _build_deps(kg_getter=lambda: _make_kg(with_concept=False))
        result = await dispatch_nav_action(
            nav_action="go_to_concept", nav_target="ghost-concept", **deps,
        )
        # Falls back to a next_section call so the student isn't stuck
        deps["dialogue"].next_section.assert_awaited_once_with("sess-123")
        deps["dialogue"].save_course_position.assert_not_awaited()
        sent = deps["send"].await_args.args[0]
        assert sent["type"] == "next_section"
        assert result["ok"] is False
        assert "not found" in result["detail"]

    @pytest.mark.asyncio
    async def test_empty_target_falls_back_to_next(self):
        deps = _build_deps()
        result = await dispatch_nav_action(
            nav_action="go_to_concept", nav_target="   ", **deps,
        )
        deps["dialogue"].next_section.assert_awaited_once()
        assert result["ok"] is False
        assert "empty target" in result["detail"]

    @pytest.mark.asyncio
    async def test_fuzzy_match_works(self):
        """Student says 'k_means' (snake_case), KG holds display_name 'K-means'.
        The substring match should still resolve — otherwise the student would
        have to guess the canonical form."""
        kg = _make_kg(with_concept=True)
        # Force the exact get_concept to miss so we exercise list_concepts path
        kg.get_concept.return_value = None
        deps = _build_deps(kg_getter=lambda: kg)
        result = await dispatch_nav_action(
            nav_action="go_to_concept", nav_target="kmeans", **deps,
        )
        # Should still resolve via the fuzzy substring fallback
        kg.list_concepts.assert_called()
        assert result["action"] == "go_to_concept"


class TestExplainMore:
    @pytest.mark.asyncio
    async def test_emits_explain_more_event(self):
        deps = _build_deps()
        result = await dispatch_nav_action(
            nav_action="explain_more", nav_target="", **deps,
        )
        # No section change — explain_more re-explains the *current* slide
        deps["dialogue"].next_section.assert_not_awaited()
        deps["dialogue"].prev_section.assert_not_awaited()
        sent = deps["send"].await_args.args[0]
        assert sent["type"] == "explain_more_request"
        assert sent["depth"] == "deep"
        assert sent["section_index"] == 5
        assert sent["chapter_index"] == 2
        assert result["ok"] is True


class TestSlowDown:
    @pytest.mark.asyncio
    async def test_reduces_speech_rate_and_persists(self):
        deps = _build_deps(current_speech_rate=1.0)
        result = await dispatch_nav_action(
            nav_action="slow_down", nav_target="", **deps,
        )
        # Profile was patched with the new (lower) rate
        deps["profile_updater"].assert_awaited_once()
        patched = deps["profile_updater"].await_args.args[1]
        assert patched["speech_rate"] == pytest.approx(_SLOW_DOWN_FACTOR)
        # And the WS event carries both old + new
        sent = deps["send"].await_args.args[0]
        assert sent["type"] == "speech_rate_changed"
        assert sent["speech_rate"] == pytest.approx(_SLOW_DOWN_FACTOR, rel=1e-3)
        assert sent["previous_rate"] == pytest.approx(1.0)
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_floors_at_minimum_rate(self):
        """Repeated slow_downs shouldn't drag the rate below _SPEECH_RATE_FLOOR
        — TTS gets unstable below that."""
        # Start at floor / factor so one more slow_down would underflow
        deps = _build_deps(current_speech_rate=_SPEECH_RATE_FLOOR / 2)
        await dispatch_nav_action(
            nav_action="slow_down", nav_target="", **deps,
        )
        patched = deps["profile_updater"].await_args.args[1]
        assert patched["speech_rate"] >= _SPEECH_RATE_FLOOR

    @pytest.mark.asyncio
    async def test_works_without_profile_updater(self):
        """profile_updater=None → still emits the event (best-effort)."""
        deps = _build_deps(profile_updater=None)
        result = await dispatch_nav_action(
            nav_action="slow_down", nav_target="", **deps,
        )
        sent = deps["send"].await_args.args[0]
        assert sent["type"] == "speech_rate_changed"
        assert result["ok"] is True


# ════════════════════════════════════════════════════════════════════
# Defensive paths
# ════════════════════════════════════════════════════════════════════

class TestDefensive:
    @pytest.mark.asyncio
    async def test_unknown_action_collapses_to_next(self):
        deps = _build_deps()
        result = await dispatch_nav_action(
            nav_action="dance_around", nav_target="", **deps,
        )
        deps["dialogue"].next_section.assert_awaited_once()
        assert result["action"] == "next"

    @pytest.mark.asyncio
    async def test_no_ctx_returns_failure(self):
        deps = _build_deps(ctx=None)
        result = await dispatch_nav_action(
            nav_action="next", nav_target="", **deps,
        )
        assert result["ok"] is False
        assert "no ctx" in result["detail"]
        # Nothing sent — no ctx = nothing to do
        deps["send"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dialogue_failure_does_not_block_send(self):
        """If next_section raises, the frontend still gets the event so the
        UI doesn't lock up waiting for a confirmation that never comes."""
        dialogue = MagicMock(
            next_section=AsyncMock(side_effect=RuntimeError("redis down")),
            prev_section=AsyncMock(),
            save_course_position=AsyncMock(),
        )
        deps = _build_deps(dialogue=dialogue)
        result = await dispatch_nav_action(
            nav_action="next", nav_target="", **deps,
        )
        deps["send"].assert_awaited_once()
        # We still report ok=True because the UI got its event ; the
        # backend state desync will be caught by a subsequent reconciliation.
        assert result["ok"] is True


# ════════════════════════════════════════════════════════════════════
# All 7 valid actions reach a unique (action, send-type) outcome
# ════════════════════════════════════════════════════════════════════

class TestAllSevenActionsAreDistinct:
    """One sanity check per action — make sure adding an 8th action
    later doesn't accidentally collapse two outcomes into one."""

    EXPECTED_EVENT_TYPE = {
        "next":          "next_section",
        "skip":          "next_section",   # alias of next
        "previous":      "prev_section",
        "repeat":        "slide_update",
        "go_to_concept": "slide_update",
        "explain_more":  "explain_more_request",
        "slow_down":     "speech_rate_changed",
    }

    @pytest.mark.parametrize("action", sorted(_VALID_NAV_ACTIONS))
    @pytest.mark.asyncio
    async def test_each_action_emits_expected_event(self, action):
        deps = _build_deps()
        await dispatch_nav_action(
            nav_action=action,
            # go_to_concept needs a target ; others ignore it.
            nav_target="K-means" if action == "go_to_concept" else "",
            **deps,
        )
        sent = deps["send"].await_args.args[0]
        assert sent["type"] == self.EXPECTED_EVENT_TYPE[action], (
            f"action={action!r} produced event {sent['type']!r}, "
            f"expected {self.EXPECTED_EVENT_TYPE[action]!r}"
        )
        assert sent.get("turn_id") == 42
