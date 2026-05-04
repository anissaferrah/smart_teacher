"""Per-session engagement score — a single [0..1] signal the rest of
the system can read to decide "should I slow down? speed up? simplify?".

# Why this exists

Several signals about engagement are already collected (pause durations,
confusion counts, question rate, idle time) but none of them, alone, is
actionable. The bandit picks a strategy ; the personalization engine
picks a tone ; but neither has a single "is this student paying
attention right now ?" input. This module fuses the existing signals
into one number so downstream consumers don't have to invent their own
heuristic.

# Score interpretation

  - 0.00 — 0.30  : disengaged. Reduce speed, simplify, prompt for
                   interaction ("you still with me ?").
  - 0.30 — 0.65  : neutral. Keep doing what you're doing.
  - 0.65 — 1.00  : highly engaged. Can introduce harder material, ask
                   socratic questions, push pace.

# Honest defaults

The weights below are NOT empirically calibrated. They're picked from
intuition + the literature on student engagement (Fredricks et al. 2004
on the behavioral / emotional / cognitive triad). The numbers SHOULD be
recalibrated against real interaction logs once we have them — see
``calibrate_weights`` future work in the README.

# What's NOT in scope

  - Real-time prosody / facial cues : would need vision/audio frame-level
    inputs that don't pass through this module.
  - Cross-session trends ("is this student less engaged this week than
    last") : per-session only ; longitudinal analysis lives in analytics.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from core.config import Config

log = logging.getLogger("pedagogy.engagement")


# ── Signal weights — tunable via Config ──────────────────────────────
# Sourced from Config so operators can rebalance per-deployment.
# MUST sum to 1.0 ; the assertion below fails fast if env-var
# overrides are set inconsistently.
W_RECENCY    = Config.ENGAGEMENT_W_RECENCY
W_QUESTION   = Config.ENGAGEMENT_W_QUESTION
W_CONFUSION  = Config.ENGAGEMENT_W_CONFUSION
W_PASSIVE    = Config.ENGAGEMENT_W_PASSIVE
W_INTERRUPT  = Config.ENGAGEMENT_W_INTERRUPT

assert abs(
    (W_RECENCY + W_QUESTION + W_CONFUSION + W_PASSIVE + W_INTERRUPT) - 1.0
) < 1e-3, (
    "engagement weights must sum to 1.0 — check ENGAGEMENT_W_* env vars"
)


# Time horizons (seconds) — sourced from Config
RECENT_FULL_S      = Config.ENGAGEMENT_RECENT_FULL_S
FULLY_STALE_S      = Config.ENGAGEMENT_FULLY_STALE_S
CONFUSION_OPTIMAL  = Config.ENGAGEMENT_CONFUSION_OPTIMAL
INTERRUPT_FAST_MS  = Config.ENGAGEMENT_INTERRUPT_FAST_MS
INTERRUPT_SLOW_MS  = Config.ENGAGEMENT_INTERRUPT_SLOW_MS

# Bucket label thresholds (config-tunable too)
DISENGAGED_BELOW   = Config.ENGAGEMENT_DISENGAGED_BELOW
ENGAGED_ABOVE      = Config.ENGAGEMENT_ENGAGED_ABOVE


@dataclass
class EngagementSignals:
    """Inputs to the engagement score.

    All fields are optional so callers that don't have a particular
    signal pass ``None`` and that signal contributes neutrally
    (0.5 × its weight).
    """
    seconds_since_last_interaction: Optional[float] = None
    questions_in_session:           Optional[int]   = None
    confusions_in_session:          Optional[int]   = None
    consecutive_passive_slides:     Optional[int]   = None
    last_interrupt_latency_ms:      Optional[float] = None
    # Bookkeeping — included in the returned reason for log diagnosis
    session_age_s:                  Optional[float] = None


@dataclass
class EngagementResult:
    score:  float                   # [0.0, 1.0]
    label:  str                     # "disengaged" | "neutral" | "engaged"
    reason: str                     # one-liner for the operator log


def _recency_score(seconds_since: Optional[float]) -> float:
    """1.0 if recent, decays linearly to 0.0 over (RECENT_FULL_S, FULLY_STALE_S)."""
    if seconds_since is None or seconds_since < 0:
        return 0.5  # neutral when no signal
    if seconds_since <= RECENT_FULL_S:
        return 1.0
    if seconds_since >= FULLY_STALE_S:
        return 0.0
    # Linear decay between the two thresholds
    span = FULLY_STALE_S - RECENT_FULL_S
    return max(0.0, 1.0 - (seconds_since - RECENT_FULL_S) / span)


def _question_score(n_questions: Optional[int], session_age_s: Optional[float]) -> float:
    """Question rate (per minute) : 0.5 q/min ≈ optimal engagement.
    0 q/min on a long session = 0 ; > 5 q/min ≈ overwhelmed."""
    if n_questions is None:
        return 0.5
    # On a fresh session (< 60s) we don't have enough sample size — return
    # 1.0 if any question was asked, else 0.5.
    if session_age_s is None or session_age_s < 60.0:
        return 1.0 if n_questions > 0 else 0.5
    rate = n_questions / (session_age_s / 60.0)
    if rate >= 5.0:
        return 0.5  # too many questions — student lost
    if rate >= 0.5:
        return 1.0  # healthy curiosity
    if rate <= 0.0:
        return 0.0
    return rate / 0.5  # 0..0.5 q/min linearly maps to 0..1.0


def _confusion_score(n_confusions: Optional[int]) -> float:
    """Confusion is U-shaped : zero is suspicious, optimal ≈ 2, too many is bad."""
    if n_confusions is None:
        return 0.5
    if n_confusions == 0:
        return 0.5  # ambiguous — could be mastery or disengagement
    if n_confusions == CONFUSION_OPTIMAL:
        return 1.0
    if n_confusions == 1:
        return 0.85
    if n_confusions <= CONFUSION_OPTIMAL + 1:
        return 0.85  # one above optimal still healthy
    if n_confusions <= 5:
        return 0.6
    return 0.3  # 6+ confusions in a session = very lost


def _passive_score(consecutive_passive: Optional[int]) -> float:
    """0 passive slides = full credit ; 3+ = lost the student."""
    if consecutive_passive is None or consecutive_passive < 0:
        return 0.5
    if consecutive_passive == 0:
        return 1.0
    if consecutive_passive == 1:
        return 0.7
    if consecutive_passive == 2:
        return 0.4
    return 0.0  # 3+ slides without interaction = disengaged


def _interrupt_score(latency_ms: Optional[float]) -> float:
    """Short interrupt latency = quick reaction to TTS = paying attention."""
    if latency_ms is None or latency_ms < 0:
        return 0.5
    if latency_ms <= INTERRUPT_FAST_MS:
        return 1.0
    if latency_ms >= INTERRUPT_SLOW_MS:
        return 0.2
    span = INTERRUPT_SLOW_MS - INTERRUPT_FAST_MS
    return 1.0 - (latency_ms - INTERRUPT_FAST_MS) / span * 0.8


def compute_engagement(signals: EngagementSignals) -> EngagementResult:
    """Fuse the 5 signals into a single [0..1] score + label + reason.

    Always returns a bounded result : missing inputs degrade gracefully
    to neutral 0.5 contributions (so ``score`` never crashes on
    incomplete data — it just becomes less informative).
    """
    rec  = _recency_score(signals.seconds_since_last_interaction)
    qst  = _question_score(signals.questions_in_session, signals.session_age_s)
    cfn  = _confusion_score(signals.confusions_in_session)
    psv  = _passive_score(signals.consecutive_passive_slides)
    itr  = _interrupt_score(signals.last_interrupt_latency_ms)

    score = (
        W_RECENCY    * rec +
        W_QUESTION   * qst +
        W_CONFUSION  * cfn +
        W_PASSIVE    * psv +
        W_INTERRUPT  * itr
    )
    score = round(max(0.0, min(1.0, score)), 3)

    if score < DISENGAGED_BELOW:
        label = "disengaged"
    elif score < ENGAGED_ABOVE:
        label = "neutral"
    else:
        label = "engaged"

    reason = (
        f"score={score:.2f} ({label}) | "
        f"recency={rec:.2f} questions={qst:.2f} "
        f"confusion={cfn:.2f} passive={psv:.2f} interrupt={itr:.2f}"
    )
    return EngagementResult(score=score, label=label, reason=reason)


def signals_from_ctx(ctx, now_ts: Optional[float] = None) -> EngagementSignals:
    """Convenience : pull the 5 signals from a SessionContext.

    Tolerant to missing fields — older SessionContext blobs may not
    have ``last_interaction_at`` or ``consecutive_passive_slides``,
    so we default to None (neutral).
    """
    if now_ts is None:
        now_ts = time.time()

    last_interact = getattr(ctx, "last_interaction_at", None)
    seconds_since = (now_ts - float(last_interact)) if last_interact else None

    session_started = getattr(ctx, "session_started_at", None)
    session_age = (now_ts - float(session_started)) if session_started else None

    return EngagementSignals(
        seconds_since_last_interaction=seconds_since,
        questions_in_session=getattr(ctx, "questions_in_session", None),
        confusions_in_session=getattr(ctx, "confusion_count", None),
        consecutive_passive_slides=getattr(ctx, "consecutive_passive_slides", None),
        last_interrupt_latency_ms=getattr(ctx, "last_interrupt_latency_ms", None),
        session_age_s=session_age,
    )
