"""Pedagogical decision layer for resume-after-pause behaviour.

# Why this module exists

Without this layer, the WS handler does *mechanical replay* :
"cursor at end → rewind to last sentence → play". That's the same
behaviour for a 2-second cough, a 5-minute phone call, and a slide
the student already finished. A real teacher doesn't behave like
that — they read the situation and decide.

This module turns 3 inputs (cursor position, narration length,
pause duration) into a structured *intent* and a *strategy*, then
the WS handler applies the strategy. Pure functions, no side
effects, fully unit-testable.

# Intents

Six intents, picked to be *behaviourally distinct* — each one
warrants a different response :

  - QUICK_RESUME       : pause < 10s, cursor mid-narration.
                         Student tabbed away briefly. Continue without
                         interruption — no recap, no rewind.
  - NORMAL_RESUME      : 10s ≤ pause < 60s, cursor mid-narration.
                         Asked a quick question or paused to think.
                         Replay the current sentence (idea), no recap.
  - RESUME_AFTER_GAP   : 60s ≤ pause < 3min, cursor mid-narration.
                         Long-ish pause — student likely lost the thread.
                         Replay current sentence + short topic recap
                         ("On était sur : <topic>").
  - RESUME_AFTER_LONG  : pause ≥ 3min, cursor mid-narration.
                         Came back from a real interruption. Full recap
                         + replay current sentence.
  - SLIDE_COMPLETED    : cursor at/past end, pause < 30s.
                         Student let the slide finish. Don't replay —
                         signal the FE to move to the next slide.
  - REVIEW_REQUEST     : cursor at/past end, pause ≥ 30s.
                         Slide finished but student didn't move on —
                         likely wants to review. Replay last sentence
                         as a short refresher.

# Strategies

Five output strategies map 1-to-many from the intents above :

  - CONTINUE              : keep cursor where it is, no recap
  - REWIND_SENTENCE       : rewind to start of current sentence, no recap
  - REWIND_SENTENCE_RECAP : rewind + medium recap
  - REWIND_SENTENCE_LONG_RECAP : rewind + long recap
  - SKIP_TO_NEXT          : signal "slide done, advance"

# Future extensions

The signature accepts an optional ``engagement_score`` and
``slide_already_seen`` flag — they're not consumed yet, but kept
in the dataclass so adding them later is purely additive (no
breaking signature change).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.config import Config

log = logging.getLogger("SmartTeacher.ResumeIntelligence")


# ── Time thresholds (sourced from Config, env-tunable) ────────────────
# These were inline magic numbers ; they're now Config attributes so
# operators can recalibrate per-deployment without code changes (e.g.,
# a remedial-class deployment may want longer "normal" buckets to
# accommodate slower question rhythm). The module-level constants are
# kept as aliases for backwards-compat with existing tests + imports.
QUICK_PAUSE_S          = Config.RESUME_QUICK_PAUSE_S
NORMAL_PAUSE_S         = Config.RESUME_NORMAL_PAUSE_S
LONG_PAUSE_S           = Config.RESUME_LONG_PAUSE_S
SLIDE_DONE_GRACE_S     = Config.RESUME_SLIDE_DONE_GRACE_S


class ResumeIntent(str, Enum):
    QUICK_RESUME       = "quick_resume"
    NORMAL_RESUME      = "normal_resume"
    RESUME_AFTER_GAP   = "resume_after_gap"
    RESUME_AFTER_LONG  = "resume_after_long"
    SLIDE_COMPLETED    = "slide_completed"
    REVIEW_REQUEST     = "review_request"


class ResumeStrategy(str, Enum):
    CONTINUE                    = "continue"
    REWIND_SENTENCE             = "rewind_sentence"
    REWIND_SENTENCE_RECAP       = "rewind_sentence_recap"
    REWIND_SENTENCE_LONG_RECAP  = "rewind_sentence_long_recap"
    # Calls the LLM to re-explain the current sentence in different
    # words, then continues with the cached remainder. Used after
    # medium / long pauses where the student likely lost the thread —
    # a verbatim replay isn't enough, they need a fresh angle. Costs
    # one extra LLM call (10-60s on Ollama) so we only apply it on
    # pauses long enough to justify the wait.
    REEXPLAIN_AND_CONTINUE      = "reexplain_and_continue"
    SKIP_TO_NEXT                = "skip_to_next"


@dataclass
class ResumeContext:
    """Inputs to the intent detector. All optional fields default
    to "no signal" so callers that only have basic info still work."""
    cursor: int
    narration_len: int
    interruption_duration_s: Optional[float] = None
    # Future signals (not used yet — kept for additive evolution)
    slide_already_seen: bool = False
    engagement_score: int = 0


@dataclass
class ResumeAction:
    """The composed output : what the caller should DO + a human-
    readable reason for the operator log."""
    intent: ResumeIntent
    strategy: ResumeStrategy
    reason: str


def _pause_bucket(duration_s: Optional[float]) -> str:
    """Map an interruption duration to a coarse bucket name.

    None or negative durations fall back to "normal" rather than to
    one of the extreme buckets — silently extreme-bucketing on a
    missing signal would change behaviour unpredictably for callers
    that simply forgot to pass the timestamp.
    """
    if duration_s is None or duration_s < 0:
        log.info("⏱️  pause_bucket | duration=%s → 'normal' (no signal)", duration_s)
        return "normal"
    if duration_s < QUICK_PAUSE_S:
        log.info(
            "⏱️  pause_bucket | duration=%.1fs < %.0fs (QUICK) → 'quick'",
            duration_s, QUICK_PAUSE_S,
        )
        return "quick"
    if duration_s < NORMAL_PAUSE_S:
        log.info(
            "⏱️  pause_bucket | duration=%.1fs < %.0fs (NORMAL) → 'normal'",
            duration_s, NORMAL_PAUSE_S,
        )
        return "normal"
    if duration_s < LONG_PAUSE_S:
        log.info(
            "⏱️  pause_bucket | duration=%.1fs < %.0fs (GAP) → 'gap'",
            duration_s, LONG_PAUSE_S,
        )
        return "gap"
    log.info(
        "⏱️  pause_bucket | duration=%.1fs ≥ %.0fs → 'long'",
        duration_s, LONG_PAUSE_S,
    )
    return "long"


def detect_resume_intent(ctx: ResumeContext) -> ResumeIntent:
    """Classify the resume context into one of the six intents.

    Decision tree :

      cursor >= len  →  pause < grace ? SLIDE_COMPLETED : REVIEW_REQUEST
      cursor < len   →  pause bucket :
                        quick  → QUICK_RESUME
                        normal → NORMAL_RESUME
                        gap    → RESUME_AFTER_GAP
                        long   → RESUME_AFTER_LONG
    """
    log.info(
        "📝 detect_resume_intent | cursor=%d/%d (%.1f%%) pause=%s",
        ctx.cursor, ctx.narration_len,
        (ctx.cursor / ctx.narration_len * 100.0) if ctx.narration_len else 0.0,
        f"{ctx.interruption_duration_s:.1f}s" if ctx.interruption_duration_s is not None else "?",
    )
    # End-of-slide cases first (cursor at or past the narration end).
    if ctx.narration_len > 0 and ctx.cursor >= ctx.narration_len:
        if ctx.interruption_duration_s is not None and ctx.interruption_duration_s < SLIDE_DONE_GRACE_S:
            log.info(
                "📝 detect_resume_intent → SLIDE_COMPLETED (cursor>=end & pause %.1fs < grace %.0fs)",
                ctx.interruption_duration_s, SLIDE_DONE_GRACE_S,
            )
            return ResumeIntent.SLIDE_COMPLETED
        log.info(
            "📝 detect_resume_intent → REVIEW_REQUEST (cursor>=end & pause %.1fs >= grace)",
            ctx.interruption_duration_s or 0.0,
        )
        return ResumeIntent.REVIEW_REQUEST

    # Mid-narration : decide by pause bucket
    bucket = _pause_bucket(ctx.interruption_duration_s)
    intent = {
        "quick":  ResumeIntent.QUICK_RESUME,
        "normal": ResumeIntent.NORMAL_RESUME,
        "gap":    ResumeIntent.RESUME_AFTER_GAP,
        "long":   ResumeIntent.RESUME_AFTER_LONG,
    }.get(bucket, ResumeIntent.RESUME_AFTER_LONG)
    log.info("📝 detect_resume_intent → %s (bucket=%s)", intent.value, bucket)
    return intent


# Map intent → strategy. Single source of truth so adding a 7th
# intent later doesn't require touching the decision logic.
_INTENT_STRATEGY: dict[ResumeIntent, ResumeStrategy] = {
    ResumeIntent.QUICK_RESUME:      ResumeStrategy.CONTINUE,
    ResumeIntent.NORMAL_RESUME:     ResumeStrategy.REWIND_SENTENCE,
    # After a meaningful gap or a long absence, the student probably
    # lost the thread of the idea they were on. A verbatim replay is
    # less helpful than a fresh re-explanation — call the LLM to
    # rephrase that sentence, then continue with the cached remainder.
    ResumeIntent.RESUME_AFTER_GAP:  ResumeStrategy.REEXPLAIN_AND_CONTINUE,
    ResumeIntent.RESUME_AFTER_LONG: ResumeStrategy.REEXPLAIN_AND_CONTINUE,
    ResumeIntent.SLIDE_COMPLETED:   ResumeStrategy.SKIP_TO_NEXT,
    ResumeIntent.REVIEW_REQUEST:    ResumeStrategy.REWIND_SENTENCE,
}


def decide_resume_strategy(intent: ResumeIntent) -> ResumeStrategy:
    """Map intent → strategy. Defensive default for unknown intents
    is REWIND_SENTENCE — same as a moderate pause, never destructive."""
    return _INTENT_STRATEGY.get(intent, ResumeStrategy.REWIND_SENTENCE)


def compose_resume_action(ctx: ResumeContext) -> ResumeAction:
    """End-to-end : context → intent → strategy + a one-line reason."""
    intent = detect_resume_intent(ctx)
    strategy = decide_resume_strategy(intent)
    reason = (
        f"cursor={ctx.cursor}/{ctx.narration_len} "
        f"pause={ctx.interruption_duration_s if ctx.interruption_duration_s is not None else '?'}"
        f"s → intent={intent.value} → strategy={strategy.value}"
    )
    log.info("📝 compose_resume_action FINAL | %s", reason)
    return ResumeAction(intent=intent, strategy=strategy, reason=reason)
