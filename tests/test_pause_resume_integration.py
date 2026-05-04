"""End-to-end integration tests for the pause/resume cursor flow.

These reproduce the exact bug pattern from the operator log :

    13:54:08 Interrupt cursor override 719 → 366    ← saved correctly
    13:54:12 Resume cursor : raw=719                 ← BUT recalled as 719 — bug

The bug was : ``decide_narration_cache_reuse`` returned ``len(text)``
for in-memory hits, ignoring the cursor stored in ``paused_state``.

Plus the navigation_prev / navigation_next interrupts didn't reset
the cursor, so the destination slide started at the previous slide's
end position.

These tests use the PURE helpers (no Redis, no LLM) so they catch
regressions cheaply.
"""
from __future__ import annotations

import pytest

from services.presentation import decide_narration_cache_reuse


# ════════════════════════════════════════════════════════════════════
# Bug regression : audio_progress cursor preserved across resume
# ════════════════════════════════════════════════════════════════════

class TestAudioProgressCursorPreservedOnResume:
    """The exact scenario from the operator log :
       1. Slide 0:0 narrated end-to-end (cursor 0 → 719).
       2. User pauses with audio_progress=0.51 → paused_state has cursor=366.
       3. User resumes : the cache decision must return cursor=366,
          NOT len(text)=719.
    """

    SLIDE_KEY = ("course-A", 0, 0)
    SLIDE_ID  = "course-A:0:0"
    NARRATION = "x" * 719  # placeholder for the 719-char narration

    def test_in_memory_hit_uses_paused_cursor(self):
        # In-memory : the same slide is still in the closure variable
        # AND the paused_state has the audio-progress-derived cursor 366.
        reuse, src, text, cursor = decide_narration_cache_reuse(
            requested_slide_key=self.SLIDE_KEY,
            current_presentation_key=self.SLIDE_KEY,
            current_presentation_text=self.NARRATION,
            cached_snapshot=None,
            cached_pause_state={
                "presentation_text": self.NARRATION,
                "slide_id":          self.SLIDE_ID,
                "presentation_cursor": 366,
            },
        )
        assert reuse is True
        assert src == "memory"
        # Critical : cursor must be 366, NOT 719.
        assert cursor == 366, (
            f"Expected paused cursor 366, got {cursor} — cache decision "
            "ignored paused_state for in-memory hit"
        )

    def test_in_memory_hit_without_paused_state_falls_back_to_len(self):
        """First present_section after a fresh narration : there's no
        paused cursor yet (slide just finished, no pause). The legacy
        len(text) behaviour is preserved."""
        reuse, src, text, cursor = decide_narration_cache_reuse(
            requested_slide_key=self.SLIDE_KEY,
            current_presentation_key=self.SLIDE_KEY,
            current_presentation_text=self.NARRATION,
            cached_snapshot=None,
            cached_pause_state={},
        )
        assert reuse is True
        assert src == "memory"
        assert cursor == len(self.NARRATION)

    def test_paused_state_for_different_slide_ignored(self):
        """User paused on slide 0:0 (cursor 366), then navigated to
        slide 0:1. The paused cursor for 0:0 must NOT be applied to 0:1."""
        reuse, src, text, cursor = decide_narration_cache_reuse(
            requested_slide_key=("course-A", 0, 1),
            current_presentation_key=("course-A", 0, 1),
            current_presentation_text="slide 1 narration",
            cached_snapshot=None,
            cached_pause_state={
                "presentation_text": "old slide 0 text",
                "slide_id":          "course-A:0:0",  # WRONG slide
                "presentation_cursor": 366,
            },
        )
        # Slide doesn't match → fall back to len(text) for slide 0:1
        assert cursor == len("slide 1 narration"), (
            "Paused cursor for a DIFFERENT slide must not leak into the "
            "current slide's resume position"
        )


# ════════════════════════════════════════════════════════════════════
# Redis snapshot path : paused_state can override snapshot when fresher
# ════════════════════════════════════════════════════════════════════

class TestRedisSnapshotWithFresherPause:
    """Snapshot was saved at cursor=719 (slide finished), then the
    user paused mid-slide on a re-listen at cursor=200. The cache
    decision should pick the FRESHER (lower) paused cursor."""

    def test_paused_cursor_takes_precedence_when_lower(self):
        long_text = "y" * 800
        reuse, src, text, cursor = decide_narration_cache_reuse(
            requested_slide_key=("course-A", 0, 0),
            current_presentation_key=None,           # not in RAM
            current_presentation_text="",
            cached_snapshot={
                "presentation_text":  long_text,
                "presentation_cursor": 719,           # snapshot is at end
            },
            cached_pause_state={
                "presentation_text": long_text,
                "slide_id":          "course-A:0:0",
                "presentation_cursor": 200,           # fresher pause mid-slide
            },
        )
        assert reuse is True
        assert src == "redis"
        assert cursor == 200, (
            f"When paused cursor is BEFORE the snapshot cursor, the "
            f"paused cursor wins (it's the fresher state). Got {cursor}."
        )

    def test_snapshot_cursor_wins_when_paused_is_stale(self):
        """If the paused cursor is AT or PAST the snapshot, the snapshot
        is more recent — keep the snapshot's cursor."""
        long_text = "y" * 800
        reuse, src, text, cursor = decide_narration_cache_reuse(
            requested_slide_key=("course-A", 0, 0),
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot={
                "presentation_text":  long_text,
                "presentation_cursor": 200,           # fresher
            },
            cached_pause_state={
                "presentation_text": long_text,
                "slide_id":          "course-A:0:0",
                "presentation_cursor": 800,           # stale (older session)
            },
        )
        assert cursor == 200, (
            "Snapshot cursor is more recent (smaller AND from explicit save) — "
            "should win over the stale paused cursor"
        )


# ════════════════════════════════════════════════════════════════════
# Bug regression : navigation does NOT carry stale cursor
# ════════════════════════════════════════════════════════════════════

class TestNavigationCursorBehaviour:
    """When the student clicks next/prev/repeat, the cursor should
    NOT be carried over from the previous slide's end. Tested at the
    cache-decision layer : a navigation that ends up requesting a
    DIFFERENT slide than the paused one must start fresh.

    The actual cursor=0 reset on navigation interrupt is in ws.py
    (see the elif msg_type == 'interrupt' block) — it's a behavioural
    integration not easily unit-testable without booting the WS handler.
    What we CAN test here is that the cache helper handles the
    "different slide than paused" case correctly."""

    def test_navigate_to_next_slide_starts_fresh(self):
        """Pause on slide 0:0 (cursor 400). Click next → present 0:1.
        Cache helper must NOT apply 400 to slide 0:1."""
        reuse, src, text, cursor = decide_narration_cache_reuse(
            requested_slide_key=("c", 0, 1),    # next slide
            current_presentation_key=None,
            current_presentation_text="",
            cached_snapshot={
                "presentation_text":  "slide 1 text",
                "presentation_cursor": 0,        # fresh snapshot for slide 1
            },
            cached_pause_state={
                "presentation_text": "slide 0 paused text",
                "slide_id":          "c:0:0",    # paused on a DIFFERENT slide
                "presentation_cursor": 400,
            },
        )
        assert reuse is True
        # Snapshot for slide 1 has cursor=0 ; paused_state is for a
        # different slide so it's ignored. Resume cursor = 0.
        assert cursor == 0


# ════════════════════════════════════════════════════════════════════
# All three sources together — the priority must hold
# ════════════════════════════════════════════════════════════════════

class TestSourcePriority:
    """When in-memory + Redis snapshot + paused_state are ALL available
    for the same slide, in-memory wins because it's the freshest."""

    def test_in_memory_wins_over_redis_and_pause(self):
        long_text = "z" * 500
        reuse, src, text, cursor = decide_narration_cache_reuse(
            requested_slide_key=("c", 0, 0),
            current_presentation_key=("c", 0, 0),
            current_presentation_text=long_text,
            cached_snapshot={
                "presentation_text": long_text,
                "presentation_cursor": 100,
            },
            cached_pause_state={
                "presentation_text": long_text,
                "slide_id": "c:0:0",
                "presentation_cursor": 250,
            },
        )
        assert src == "memory"
        # In-memory + same slide + paused cursor exists → use paused (250)
        assert cursor == 250
