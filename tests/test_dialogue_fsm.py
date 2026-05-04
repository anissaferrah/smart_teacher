"""Tests for the DialogState FSM transition map.

These guard the recently-added WAITING -> PROCESSING transition (the
student paused, then asks a real question, the WS handler must
transition straight into PROCESSING). They also assert the broader
contract :

  - Every state has at least one outbound transition (no dead-ends
    other than terminal IDLE).
  - Self-loops are forbidden (state is updated only on real
    transitions).
"""
from __future__ import annotations

import pytest

from pedagogy.dialogue import DialogState, VALID_TRANSITIONS


# ════════════════════════════════════════════════════════════════════
# WAITING -> PROCESSING (the regression we just fixed)
# ════════════════════════════════════════════════════════════════════

class TestWaitingToProcessing:
    def test_waiting_allows_processing(self):
        """The fix : when the student is paused (WAITING) and asks a
        question, the WS handler kicks off the Q&A pipeline which
        transitions into PROCESSING. Without this entry the operator
        would see ``Transition invalide : WAITING -> PROCESSING``."""
        allowed = VALID_TRANSITIONS[DialogState.WAITING]
        assert DialogState.PROCESSING in allowed, (
            f"WAITING must allow direct jump to PROCESSING ; "
            f"current allowed list: {[s.value for s in allowed]}"
        )

    def test_waiting_still_allows_legacy_targets(self):
        """The fix is additive — make sure we didn't drop pre-existing
        transitions out of WAITING (PRESENTING / LISTENING / IDLE / CLARIFICATION)."""
        allowed = set(VALID_TRANSITIONS[DialogState.WAITING])
        for target in (
            DialogState.PRESENTING,
            DialogState.LISTENING,
            DialogState.IDLE,
            DialogState.CLARIFICATION,
        ):
            assert target in allowed, f"WAITING must keep allowing {target.value}"


# ════════════════════════════════════════════════════════════════════
# Global FSM contract
# ════════════════════════════════════════════════════════════════════

class TestFsmContract:
    def test_no_self_loops(self):
        """A state never transitions to itself — ``transition(s, s)``
        is a no-op the manager rejects, so any self-loop in the map
        would be misleading."""
        for state, targets in VALID_TRANSITIONS.items():
            assert state not in targets, (
                f"Self-loop on {state.value} — remove from VALID_TRANSITIONS"
            )

    def test_every_target_is_a_known_state(self):
        """Type-safety guard : every target enum is part of DialogState."""
        all_states = set(DialogState)
        for state, targets in VALID_TRANSITIONS.items():
            for t in targets:
                assert t in all_states, (
                    f"Unknown state {t!r} listed in transitions from {state.value}"
                )

    def test_every_state_except_terminal_has_outbound(self):
        """Every non-terminal state should have at least one outbound
        transition. If it doesn't, the FSM is stuck."""
        for state, targets in VALID_TRANSITIONS.items():
            assert len(targets) > 0, f"{state.value} has no outbound — dead-end"

    @pytest.mark.parametrize("from_state,to_state", [
        (DialogState.IDLE, DialogState.PRESENTING),
        (DialogState.IDLE, DialogState.LISTENING),
        (DialogState.PRESENTING, DialogState.LISTENING),
        (DialogState.LISTENING, DialogState.PROCESSING),
        (DialogState.PROCESSING, DialogState.RESPONDING),
        (DialogState.RESPONDING, DialogState.PRESENTING),
        (DialogState.WAITING, DialogState.LISTENING),
        (DialogState.WAITING, DialogState.PRESENTING),
        (DialogState.WAITING, DialogState.PROCESSING),    # ← the new one
        (DialogState.CLARIFICATION, DialogState.RESPONDING),
    ])
    def test_essential_transitions_present(self, from_state, to_state):
        """Spot-check the transitions the tutor actually relies on
        in production — fence against accidental removals."""
        allowed = VALID_TRANSITIONS[from_state]
        assert to_state in allowed, (
            f"Essential transition {from_state.value} -> {to_state.value} "
            f"missing from FSM map (allowed: {[s.value for s in allowed]})"
        )
