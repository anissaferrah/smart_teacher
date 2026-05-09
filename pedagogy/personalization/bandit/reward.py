"""Reward function for the personalization bandit.

# What is a "good" turn ?

Not every signal we collect is a reward. The reward is the **scalar
objective the bandit optimizes** — it has to be (a) computable from
observable signals, (b) aligned with the pedagogical goal.

For Smart Teacher, a "good" turn is one where :

  - The student did **not** signal confusion afterwards.
  - Their mastery on the targeted concept **moved toward 1**.
  - They **continued engaging**.

# The composite reward

We combine three observable signals into a single scalar in [0, 1] :

    reward = 0.50 × (1 − confusion) + 0.40 × Δm/0.30 + 0.10 × engaged

Why bounded in [0, 1] : the Thompson sampling Beta posterior expects
rewards in [0, 1] (Agrawal & Goyal 2013).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("personalization.bandit.reward")


def _load_weights() -> tuple[float, float, float]:
    """Load weights from learned_weights.json if it exists,
    otherwise fall back to hardcoded defaults."""
    import json
    from pathlib import Path

    weights_file = Path(__file__).resolve().parent / "learned_weights.json"
    if weights_file.exists():
        try:
            with weights_file.open(encoding="utf-8") as f:
                data = json.load(f)
            w_c = float(data.get("W_CONFUSION",  0.50))
            w_m = float(data.get("W_MASTERY",    0.40))
            w_e = float(data.get("W_ENGAGEMENT", 0.10))
            total = w_c + w_m + w_e
            if total > 0:
                return w_c / total, w_m / total, w_e / total
        except Exception:
            pass
    return 0.50, 0.40, 0.10


# ── Weights ─────────────────────────────────────────────────────────────
#
# Sum = 1.0. Documented rationale per weight :
#
#   W_CONFUSION = 0.50  : the dominant signal. Pedagogical literature
#                         (Bloom 1968, Sweller 1985 cognitive load) treats
#                         confusion as the strongest negative outcome.
#
#   W_MASTERY   = 0.40  : a successful explanation should move the
#                         posterior on the targeted concept toward 1.
#                         Mastery progress is harder to measure per-turn
#                         (often noisy on a single attempt) so it gets
#                         slightly less than confusion avoidance.
#
#   W_ENGAGEMENT = 0.10 : tie-breaker. A response that's "fine" but not
#                         great should still be slightly preferred over
#                         silence.

W_CONFUSION, W_MASTERY, W_ENGAGEMENT = _load_weights()
# Sanity check
assert abs(W_CONFUSION + W_MASTERY + W_ENGAGEMENT - 1.0) < 1e-9


@dataclass
class TurnOutcome:
    """Observable outcome of one student turn — the input to the reward.

    Attributes
    ----------
    confusion_detected : bool
        SIGHT model verdict for the next student utterance (or for the
        current if it's a follow-up).
    mastery_before, mastery_after : float
        Mastery posterior on the target concept, in [0, 1].
        ``mastery_after - mastery_before`` is the per-turn delta.
    engaged : bool
        Did the student keep interacting after this turn (asked a
        follow-up, requested an example, navigated forward) or did
        they bail out / go silent ?
    """

    confusion_detected: bool
    mastery_before:     float
    mastery_after:      float
    engaged:            bool


# Bound on the per-turn mastery delta we expose to the reward
# function. Empirically, the Beta posterior in mastery_repo can move
# at most ~0.3 in a single attempt (e.g. from 1/3 to 2/4 = +0.17, or
# 0/2 to 1/3 = +0.33). Saturating at 0.3 prevents a single noisy
# attempt from dominating the reward.
_MAX_DELTA = 0.15


def compute_reward(outcome: TurnOutcome) -> float:
    """Map a TurnOutcome to a scalar reward in [0, 1].

    Pure function — no DB I/O, no external state mutation. Logs the
    breakdown for observability.
    """
    confusion_signal = 0.0 if outcome.confusion_detected else 1.0
    raw_delta = max(0.0, outcome.mastery_after - outcome.mastery_before)
    mastery_signal = min(raw_delta / _MAX_DELTA, 1.0)
    engagement_signal = 1.0 if outcome.engaged else 0.0

    reward = (
        W_CONFUSION * confusion_signal
        + W_MASTERY * mastery_signal
        + W_ENGAGEMENT * engagement_signal
    )
    reward = max(0.0, min(1.0, reward))

    log.debug(
        "reward.compute = %.3f | confused=%s Δm=%.3f engaged=%s",
        reward, outcome.confusion_detected, raw_delta, outcome.engaged,
    )
    return reward


@dataclass
class DelayedOutcome:
    """Outcome observed when a concept is revisited later.
    
    Attributes
    ----------
    fsrs_stability_gain : float
        How much FSRS stability increased when the concept
        was reviewed. Positive = student retained it.
        In [0, 1] range.
    turns_since_taught : int
        How many turns passed between teaching and revisiting.
    """
    fsrs_stability_gain: float
    turns_since_taught: int


GAMMA = 0.30  # discount factor for delayed reward


def compute_total_reward(
    outcome: TurnOutcome,
    delayed: DelayedOutcome | None = None,
) -> float:
    """Compute total reward combining immediate and delayed signals.
    
    If delayed outcome is provided (concept was revisited):
        total = immediate_reward + GAMMA * delayed_reward
    Otherwise:
        total = immediate_reward (same as compute_reward)
    
    This implements the two-timescale reward described in the
    personalisation design doc. GAMMA=0.30 follows standard
    practice for episodic RL with short horizons (Sutton & Barto 2018).
    """
    immediate = compute_reward(outcome)
    if delayed is None:
        return immediate
    delayed_reward = max(0.0, min(1.0, delayed.fsrs_stability_gain))
    total = immediate + GAMMA * delayed_reward
    return max(0.0, min(1.0, total))
