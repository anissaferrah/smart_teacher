"""Tests for the presentation narration cache reuse decision.

The bug we're fixing : after a pause on slide 4, the user navigates back
to slide 0 (which they'd already seen earlier in the session). The Redis
snapshot for slide 0 IS cached, but the previous version of the decision
gated reuse on ``cached_pause_state.slide_id == requested_slide_id`` —
``cached_pause_state`` always holds the LAST PAUSED slide (slide 4), so
the match failed and the LLM regenerated a 60-300s narration the student
had already heard. Visible in operator logs as "Narration cache MISS"
on every back-navigation.

The fix : trust the per-slide Redis snapshot directly, regardless of
which slide is currently paused.
"""
from __future__ import annotations

import pytest

from services.presentation import decide_narration_cache_reuse


# ════════════════════════════════════════════════════════════════════
# Source priority : in-memory > redis snapshot > paused-state
# ════════════════════════════════════════════════════════════════════

class TestSourcePriority:

    def test_in_memory_hit_wins_over_snapshot(self):
        """If we just generated this slide, prefer the in-memory text
        even if the snapshot is cached too — avoids a useless dict copy
        and keeps the cursor wherever the streamer left it."""
        slide_key = ("course-A", 0, 4)
        reuse, src, text, cur = decide_narration_cache_reuse(
            requested_slide_key=slide_key,
            current_presentation_key=slide_key,
            current_presentation_text="In-memory narration text",
            cached_snapshot={"presentation_text": "redis copy", "presentation_cursor": 50},
            cached_pause_state={},
        )
        assert reuse is True
        assert src == "memory"
        assert text == "In-memory narration text"
        # Memory hit reports cursor = full length (slide just played to end)
        assert cur == len("In-memory narration text")

    def test_snapshot_hit_when_no_memory(self):
        """Backbone case : the slide was generated earlier in the session,
        evicted from RAM (user navigated away), then revisited."""
        snapshot_text = "Slide 0 narration from Redis " * 10  # ~290 chars
        reuse, src, text, cur = decide_narration_cache_reuse(
            requested_slide_key=("course-A", 0, 0),
            current_presentation_key=("course-A", 0, 5),  # in RAM but wrong slide
            current_presentation_text="Some other slide's text",
            cached_snapshot={
                "presentation_text": snapshot_text,
                "presentation_cursor": 120,
            },
            cached_pause_state={
                "presentation_text": "stale paused text from slide 4",
                "slide_id": "course-A:0:4",  # ← the bug : doesn't match :0:0
                "presentation_cursor": 200,
            },
        )
        assert reuse is True
        assert src == "redis", "must trust Redis snapshot, not paused-state mismatch"
        assert text == snapshot_text
        assert cur == 120, "cursor comes from snapshot, not from paused-state"

    def test_paused_state_fallback_when_no_snapshot(self):
        """Cold-Redis case : no snapshot for this slide, but paused_state
        happens to match. Used to be the only working path before the
        snapshot fix — kept for safety."""
        paused_text = "Slide 4 narration in pause " * 10  # ~270 chars
        reuse, src, text, cur = decide_narration_cache_reuse(
            requested_slide_key=("course-A", 0, 4),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot=None,  # Redis miss
            cached_pause_state={
                "presentation_text": paused_text,
                "slide_id": "course-A:0:4",
                "presentation_cursor": 80,
            },
        )
        assert reuse is True
        assert src == "paused_state"
        assert text == paused_text
        assert cur == 80

    def test_full_miss_when_nothing_matches(self):
        """No memory, no snapshot, no matching paused_state → must regenerate."""
        reuse, src, text, cur = decide_narration_cache_reuse(
            requested_slide_key=("course-A", 0, 0),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot=None,
            cached_pause_state={
                "presentation_text": "old text",
                "slide_id": "course-A:0:7",  # different slide
            },
        )
        assert reuse is False
        assert src == "miss"
        assert text == ""
        assert cur == 0


# ════════════════════════════════════════════════════════════════════
# The exact reproduction scenario from the user's bug report
# ════════════════════════════════════════════════════════════════════

class TestRegressionScenario:
    """The user's actual session:
       1. Present slide 0 → snapshot saved for slide :0:0.
       2. Present slides 1, 2, 3, 4 → snapshots saved for each.
       3. Pause on slide 4 → paused_state.slide_id = "...:0:4".
       4. Navigate back to slide 4, 3, 2, 1, 0 in sequence.

    Before fix : every back-navigation produced "Narration cache MISS"
    because (paused_state.slide_id == requested_slide_id) only ever
    held for slide 4. The other 4 slides re-ran the LLM (60-300s each).

    After fix : each back-navigation hits the per-slide Redis snapshot
    and reuses the cached narration with 0 LLM call.
    """

    @pytest.mark.parametrize("slide_idx", [0, 1, 2, 3])
    def test_back_navigation_reuses_per_slide_snapshot(self, slide_idx):
        # Paused state stuck on slide 4 (the last slide we paused on)
        paused_state = {
            "presentation_text": "stale slide 4 text",
            "slide_id": "course-A:0:4",
            "presentation_cursor": 200,
        }
        # But Redis has a snapshot for the slide we're going BACK to.
        # Long enough that the cursor isn't clamped — we want to assert
        # the cursor IS read from the snapshot, not lost to clamping.
        snapshot_text = f"Slide {slide_idx} narration content " * 20  # ~600 chars
        snapshot_for_this_slide = {
            "presentation_text": snapshot_text,
            "presentation_cursor": 50 * slide_idx,  # arbitrary, within text len
        }
        reuse, src, text, cur = decide_narration_cache_reuse(
            requested_slide_key=("course-A", 0, slide_idx),
            current_presentation_key=None,           # navigated away → RAM gone
            current_presentation_text="",
            cached_snapshot=snapshot_for_this_slide,
            cached_pause_state=paused_state,
        )
        assert reuse is True, (
            f"slide {slide_idx} : back-navigation MUST hit the Redis "
            f"snapshot and avoid the LLM regeneration"
        )
        assert src == "redis"
        assert text == snapshot_text
        assert cur == 50 * slide_idx


# ════════════════════════════════════════════════════════════════════
# Cursor clamping
# ════════════════════════════════════════════════════════════════════

class TestCursorClamp:
    def test_snapshot_cursor_clamped_to_text_length(self):
        """Snapshot cursor must always end up inside [0, len(text)] — never
        past the end (which would resume in silence). Either clamped to
        len(text) on a legitimate "finished slide" path, or reset to 0 by
        the revisit-completed guard. Both are valid : both keep the cursor
        within bounds and avoid the silent-resume bug."""
        reuse, _, text, cur = decide_narration_cache_reuse(
            requested_slide_key=("course", 0, 0),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot={
                "presentation_text": "short text",  # 10 chars
                "presentation_cursor": 9999,         # corrupt cursor
            },
            cached_pause_state={},
        )
        assert reuse is True
        assert 0 <= cur <= len(text)
        # Revisit guard kicks in on near-end cursors with no active pause —
        # corrupt over-end cursor counts as "near-end" and resets to 0.
        assert cur == 0

    def test_negative_cursor_clamped_to_zero(self):
        reuse, _, _, cur = decide_narration_cache_reuse(
            requested_slide_key=("course", 0, 0),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot={
                "presentation_text": "ok",
                "presentation_cursor": -50,
            },
            cached_pause_state={},
        )
        assert reuse is True
        assert cur == 0


# ════════════════════════════════════════════════════════════════════
# Revisit-completed-slide guard
# ════════════════════════════════════════════════════════════════════

class TestRevisitCompletedSlide:
    """When the user navigates BACK to a slide they previously finished,
    the snapshot cursor is at/near len(text). Without the guard, the TTS
    would resume at end-of-narration (= silence) and the student would
    perceive the slide as skipped. The guard resets cursor to 0 to replay
    the cached narration from the start (no LLM regeneration)."""

    NARRATION = "x" * 1918  # realistic length matching production logs

    def test_revisit_at_exact_end_resets_to_zero(self):
        """Snapshot says cursor=len, no active pause → replay from 0."""
        reuse, src, text, cur = decide_narration_cache_reuse(
            requested_slide_key=("course", 0, 2),
            current_presentation_key=("course", 0, 5),  # currently on a different slide
            current_presentation_text="",
            cached_snapshot={
                "presentation_text": self.NARRATION,
                "presentation_cursor": len(self.NARRATION),  # finished previously
            },
            # Pause is on a DIFFERENT slide (the one being navigated AWAY from)
            cached_pause_state={
                "presentation_key": "course:0:5",
                "slide_id": "course:0:5",
                "presentation_cursor": 800,
                "presentation_text": "y" * 1500,
            },
        )
        assert reuse is True
        assert src == "redis"
        assert text == self.NARRATION
        assert cur == 0  # ← revisit-completed reset

    def test_revisit_near_end_resets_to_zero(self):
        """Snapshot ≥95% counts as 'finished' for the guard."""
        cursor_at_96pct = int(len(self.NARRATION) * 0.96)
        _, _, _, cur = decide_narration_cache_reuse(
            requested_slide_key=("course", 0, 2),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot={
                "presentation_text": self.NARRATION,
                "presentation_cursor": cursor_at_96pct,
            },
            cached_pause_state={},
        )
        assert cur == 0

    def test_revisit_mid_narration_keeps_cursor(self):
        """Snapshot at 50% means user paused mid-slide — keep that cursor
        (don't trigger replay-from-start, the user clearly intended to
        resume where they left off)."""
        cursor_mid = len(self.NARRATION) // 2  # 50%
        _, _, _, cur = decide_narration_cache_reuse(
            requested_slide_key=("course", 0, 2),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot={
                "presentation_text": self.NARRATION,
                "presentation_cursor": cursor_mid,
            },
            cached_pause_state={},
        )
        assert cur == cursor_mid  # ← preserved, not reset

    def test_active_pause_on_same_slide_overrides_revisit_guard(self):
        """If the user has an ACTIVE pause on the destination slide and
        the pause cursor is BEFORE the snap cursor, the pause wins (resume
        from where they paused, not snap, not 0)."""
        _, _, _, cur = decide_narration_cache_reuse(
            requested_slide_key=("course", 0, 2),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot={
                "presentation_text": self.NARRATION,
                "presentation_cursor": len(self.NARRATION),  # snap = end
            },
            cached_pause_state={
                "presentation_key": "course:0:2",  # SAME slide
                "presentation_cursor": 500,        # pause < snap
                "presentation_text": self.NARRATION,
            },
        )
        assert cur == 500  # ← pause wins (user actively paused here)

    def test_paused_state_uses_char_offset_when_no_presentation_cursor(self):
        """Backwards-compat : older paused_state blobs only had ``char_offset``."""
        reuse, src, _, cur = decide_narration_cache_reuse(
            requested_slide_key=("c", 0, 1),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot=None,
            cached_pause_state={
                "presentation_text": "narration text",
                "slide_id": "c:0:1",
                "char_offset": 7,
            },
        )
        assert reuse is True
        assert src == "paused_state"
        assert cur == 7


# ════════════════════════════════════════════════════════════════════
# Defensive empty-input handling
# ════════════════════════════════════════════════════════════════════

class TestDefensive:
    def test_all_empty_inputs(self):
        reuse, src, text, cur = decide_narration_cache_reuse(
            requested_slide_key=("c", 0, 0),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot=None,
            cached_pause_state=None,
        )
        assert reuse is False
        assert src == "miss"
        assert text == ""
        assert cur == 0

    def test_empty_snapshot_dict_treated_as_miss(self):
        reuse, src, _, _ = decide_narration_cache_reuse(
            requested_slide_key=("c", 0, 0),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot={},  # present but empty
            cached_pause_state={},
        )
        assert reuse is False
        assert src == "miss"

    def test_paused_state_with_presentation_key_alias(self):
        """Some pause snapshots use ``presentation_key`` instead of ``slide_id``."""
        reuse, src, _, _ = decide_narration_cache_reuse(
            requested_slide_key=("c", 0, 2),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot=None,
            cached_pause_state={
                "presentation_text": "text",
                "presentation_key": "c:0:2",  # alias of slide_id
                "presentation_cursor": 5,
            },
        )
        assert reuse is True
        assert src == "paused_state"
