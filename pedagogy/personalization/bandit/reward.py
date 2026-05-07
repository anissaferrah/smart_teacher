"""Reward function for the personalization bandit.

# What is a "good" turn ?

Not every signal we collect is a reward. The reward is the **scalar
objective the bandit optimizes** — it has to be (a) computable from
observable signals, (b) aligned with the pedagogical goal.

For Smart Teacher, a "good" turn is one where :

  - The student did **not** signal confusion afterwards (negative penalty
    if they did).
  - Their mastery on the targeted concept **moved toward 1**
    (positive reward proportional to the gain).
  - They **continued engaging** (positive small reward) — staying in the
    session is itself a signal that the response wasn't off-putting.

We combine these into a single scalar in [0, 1] :

    reward = w_confusion · (1 - confusion_indicator)
           + w_mastery   · clip(mastery_delta, 0, 1)
           + w_engagement · engagement_indicator

where the weights sum to 1 and are documented per signal.

# Why bounded in [0, 1]

The Thompson sampling Beta posterior expects rewards in [0, 1]
(Agrawal & Goyal 2013). Unbounded rewards would break the conjugate
update we use in ``ArmPosterior.update``. We therefore clip mastery
delta to a sensible per-turn range and normalize the components
before combining.

# Why these weights

The weights below are explicit and documented. They are NOT magic
numbers in the bad sense — they encode an editorial choice about what
matters most in pedagogical terms (confusion avoidance > mastery
progress > engagement). A future ablation study can compare these
weights empirically once enough offline data is collected.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("personalization.bandit.reward")


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

W_CONFUSION  = 0.50
W_MASTERY    = 0.40
W_ENGAGEMENT = 0.10
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
_MAX_DELTA = 0.30


def compute_reward(outcome: TurnOutcome) -> float:
    """Map a TurnOutcome to a scalar reward in [0, 1].

    Pure function — testable in isolation, no I/O. Logs the breakdown
    at DEBUG level for observability.
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
        "reward.compute confusion=%s Δm=%.3f engaged=%s → %.3f "
        "(C=%.2f M=%.2f E=%.2f)",
        outcome.confusion_detected, raw_delta, outcome.engaged, reward,
        confusion_signal, mastery_signal, engagement_signal,
    )
    return reward
