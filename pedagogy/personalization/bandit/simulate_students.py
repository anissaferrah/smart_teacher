"""Simulated student archetypes for offline bandit testing.

# Why archetypes (not hidden preferences)

``simulator.py`` already provides a hidden-preference simulator that
draws a per-student preference vector from a profile-conditional prior.
That's the right tool for *bootstrapping* a fresh bandit posterior at
scale.

This file is for the complementary use case: **interpretable behaviour
tests**. When you want to ask "does the bandit converge on socratic+fast
for a fast learner?" you need a student whose *best* arm is known up
front — not sampled. Each archetype here pins down its preferred
(strategy, speech_rate) tuple, so a passing test produces a clean
evidence trail: bandit picked X for archetype Y, X happens to be Y's
documented optimum, done.

# The four archetypes

  - ``FastLearner``         — low confusion, large mastery gain, bored
                              by slow speech or repetition.
                              Optimum: socratic + fast.
  - ``DeepThinker``         — needs depth and analogy, actively hurt by
                              simpler-words simplification.
                              Optimum: analogy + normal.
  - ``SlowLearner``         — high baseline confusion, lifted by
                              decomposition delivered slowly.
                              Optimum: decomposition + slow.
  - ``InconsistentLearner`` — non-stationary. Drifts between a sharp
                              mood and a tired mood, so the optimal
                              arm changes mid-episode. Used to test
                              whether the bandit *adapts* rather than
                              merely converges.

# Interface

Every archetype implements::

    react(strategy, speech_rate, current_mastery, turn_number) -> TurnOutcome

The reaction is probabilistic. Three knobs drive it:

  1. ``strategy_fit``   in [0, 1] — how well the chosen strategy matches.
  2. ``rate_fit``       in [0, 1] — how well the speech rate matches.
  3. Gaussian noise (std=0.05) — to mimic real-trial variance.

The combined ``fit = strategy_fit * rate_fit`` controls confusion
probability, mastery gain, and (indirectly) engagement. ``confusion``
is sampled, not deterministic; ``mastery_after = mastery_before +
gain * fit``, capped at 1.0; ``engaged`` blends a confusion-streak
check and a strategy-repetition check, with archetype-specific
overrides.

# Constraints honoured

  - Only stdlib imports (``random``, ``typing``) plus ``TurnOutcome``
    from ``reward.py``. No Redis, Postgres, or FastAPI.
  - All probabilities live as class-level constants — tweak in one
    place, no scattered magic numbers.
  - Self-contained: ``python simulate_students.py`` runs a sanity
    check that the documented optimum produces the expected ordering
    (SlowLearner lowest mastery gain, FastLearner highest).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

try:
    from .reward import TurnOutcome
except ImportError:
    from reward import TurnOutcome


STRATEGIES = (
    "analogy", "example", "decomposition",
    "socratic", "recap", "simpler_words",
)
SPEECH_RATES = ("slow", "normal", "fast")


class _BaseStudent:
    """Shared mechanics. Subclasses override the FIT tables and bounds.

    The base class implements:

      - ``_fit_score(strategy, speech_rate)`` — multiplicative combo of
        the per-strategy and per-rate fit tables.
      - ``_confusion_prob(fit)`` — linear interpolation between
        ``BASE_CONFUSION`` (perfect fit) and ``MAX_CONFUSION`` (zero
        fit), plus Gaussian noise.
      - ``_mastery_gain(fit, ...)`` — linear in fit between
        ``GAIN_MIN`` and ``GAIN_MAX``, scaled by ``CONFUSED_GAIN_FACTOR``
        when confusion fires, then by ``(1 - current_mastery)`` to model
        diminishing returns near the ceiling.
      - ``_engagement(strategy, confused)`` — disengage when confused
        for ``ENGAGE_CONFUSION_STREAK`` consecutive turns, or when the
        same strategy is used ``ENGAGE_REPETITION_STREAK`` turns in a
        row.
    """

    name: str = "base"
    BEST_STRATEGY: str = "analogy"
    BEST_RATE: str = "normal"

    STRATEGY_FIT: Dict[str, float] = {}
    RATE_FIT: Dict[str, float] = {}

    BASE_CONFUSION: float = 0.10
    MAX_CONFUSION: float = 0.55

    GAIN_MIN: float = 0.03
    GAIN_MAX: float = 0.10

    NOISE_STD: float = 0.05
    CONFUSED_GAIN_FACTOR: float = 0.30

    ENGAGE_CONFUSION_STREAK: int = 3
    ENGAGE_REPETITION_STREAK: int = 3

    _UNKNOWN_STRATEGY_FIT: float = 0.30
    _UNKNOWN_RATE_FIT: float = 0.50

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._strategy_history: List[str] = []
        self._confusion_streak: int = 0

    def _fit_score(self, strategy: str, speech_rate: str) -> float:
        s = self.STRATEGY_FIT.get(strategy, self._UNKNOWN_STRATEGY_FIT)
        r = self.RATE_FIT.get(speech_rate, self._UNKNOWN_RATE_FIT)
        return max(0.0, min(1.0, s * r))

    def _confusion_prob(self, fit: float) -> float:
        prob = self.BASE_CONFUSION + (self.MAX_CONFUSION - self.BASE_CONFUSION) * (1.0 - fit)
        prob += self._rng.gauss(0.0, self.NOISE_STD)
        return max(0.0, min(1.0, prob))

    def _mastery_gain(self, fit: float, current_mastery: float, confused: bool) -> float:
        gain = self.GAIN_MIN + (self.GAIN_MAX - self.GAIN_MIN) * fit
        gain += self._rng.gauss(0.0, self.NOISE_STD * 0.5)
        gain = max(0.0, gain)
        if confused:
            gain *= self.CONFUSED_GAIN_FACTOR
        gain *= max(0.0, 1.0 - current_mastery)
        return gain

    def _update_history(self, strategy: str, confused: bool) -> None:
        self._strategy_history.append(strategy)
        if len(self._strategy_history) > 10:
            self._strategy_history.pop(0)
        self._confusion_streak = self._confusion_streak + 1 if confused else 0

    def _engagement(self, strategy: str, confused: bool) -> bool:
        if self._confusion_streak >= self.ENGAGE_CONFUSION_STREAK:
            return False
        recent = self._strategy_history[-self.ENGAGE_REPETITION_STREAK:]
        if (len(recent) >= self.ENGAGE_REPETITION_STREAK
                and all(s == strategy for s in recent)):
            return False
        return True

    def react(
        self,
        strategy: str,
        speech_rate: str,
        current_mastery: float,
        turn_number: int,
    ) -> TurnOutcome:
        fit = self._fit_score(strategy, speech_rate)
        confusion_p = self._confusion_prob(fit)
        confused = self._rng.random() < confusion_p
        gain = self._mastery_gain(fit, current_mastery, confused)
        new_mastery = min(1.0, current_mastery + gain)
        self._update_history(strategy, confused)
        engaged = self._engagement(strategy, confused)
        return TurnOutcome(
            confusion_detected=confused,
            mastery_before=current_mastery,
            mastery_after=new_mastery,
            engaged=engaged,
        )


class FastLearner(_BaseStudent):
    """Quick, confident, gets bored fast.

    Profile: low confusion baseline, large mastery gains, but
    disengages when the same strategy repeats or the speech rate
    drags. Optimum is ``socratic + fast``.
    """

    name = "FastLearner"
    BEST_STRATEGY = "socratic"
    BEST_RATE = "fast"

    STRATEGY_FIT = {
        "socratic":      1.00,
        "analogy":       0.75,
        "example":       0.65,
        "decomposition": 0.45,
        "simpler_words": 0.30,
        "recap":         0.20,
    }
    RATE_FIT = {"fast": 1.00, "normal": 0.75, "slow": 0.35}

    BASE_CONFUSION = 0.07
    MAX_CONFUSION  = 0.45
    GAIN_MIN       = 0.05
    GAIN_MAX       = 0.15

    ENGAGE_REPETITION_STREAK = 3


class DeepThinker(_BaseStudent):
    """Wants depth; simplification feels condescending.

    Profile: low-to-medium confusion on rich strategies (analogy,
    socratic, decomposition); high confusion on ``simpler_words``.
    Stays engaged through complex strategies even when mildly
    confused. Optimum is ``analogy + normal``.
    """

    name = "DeepThinker"
    BEST_STRATEGY = "analogy"
    BEST_RATE = "normal"

    STRATEGY_FIT = {
        "analogy":       1.00,
        "socratic":      0.85,
        "example":       0.75,
        "decomposition": 0.75,
        "recap":         0.55,
        "simpler_words": 0.20,
    }
    RATE_FIT = {"normal": 1.00, "slow": 0.80, "fast": 0.60}

    BASE_CONFUSION = 0.08
    MAX_CONFUSION  = 0.55
    GAIN_MIN       = 0.04
    GAIN_MAX       = 0.10

    _COMPLEX_STRATEGIES = frozenset({"analogy", "socratic", "decomposition"})

    def _engagement(self, strategy: str, confused: bool) -> bool:
        # Complex strategies retain engagement unless the student is
        # in a long confusion streak (4+). Repetition still hurts a
        # bit, but less than for the FastLearner.
        if strategy in self._COMPLEX_STRATEGIES and self._confusion_streak < 4:
            return True
        return super()._engagement(strategy, confused)


class SlowLearner(_BaseStudent):
    """High baseline confusion; needs explicit decomposition + time.

    Profile: confusion explodes on socratic or fast speech, drops
    sharply on ``decomposition + slow``. Mastery moves slowly even
    on the optimal arm. Disengages quickly under sustained
    confusion. Optimum is ``decomposition + slow``.
    """

    name = "SlowLearner"
    BEST_STRATEGY = "decomposition"
    BEST_RATE = "slow"

    STRATEGY_FIT = {
        "decomposition": 1.00,
        "example":       0.85,
        "recap":         0.80,
        "simpler_words": 0.70,
        "analogy":       0.45,
        "socratic":      0.15,
    }
    RATE_FIT = {"slow": 1.00, "normal": 0.55, "fast": 0.20}

    BASE_CONFUSION = 0.15
    MAX_CONFUSION  = 0.75
    GAIN_MIN       = 0.02
    GAIN_MAX       = 0.06

    ENGAGE_CONFUSION_STREAK = 3


class InconsistentLearner(_BaseStudent):
    """Non-stationary. Drifts between a "sharp" and a "tired" mood.

    Each turn flips mood with probability ``MOOD_FLIP_PROB``. The two
    moods have *different* optimal strategies (sharp prefers
    socratic+fast; tired prefers recap+slow), so a bandit that
    converges on a single arm will be wrong roughly half the time.

    The point: this archetype is the regression test for adaptation
    — does the bandit re-explore when reward distribution shifts?
    """

    name = "InconsistentLearner"
    # No truly-best arm exists. ``decomposition + normal`` is the
    # least-bad compromise — fit ≈ 0.40 in sharp mood, ≈ 0.51 in
    # tired mood — so the sanity-test number reflects "what happens
    # when you guess a middle-of-the-road arm against a drifting
    # student", which is the realistic baseline the bandit has to beat.
    BEST_STRATEGY = "decomposition"
    BEST_RATE = "normal"

    SHARP_FIT = {
        "socratic":      1.00,
        "analogy":       0.85,
        "example":       0.70,
        "decomposition": 0.50,
        "recap":         0.30,
        "simpler_words": 0.30,
    }
    SHARP_RATE = {"fast": 1.00, "normal": 0.80, "slow": 0.40}

    TIRED_FIT = {
        "recap":         1.00,
        "simpler_words": 0.90,
        "decomposition": 0.85,
        "example":       0.70,
        "analogy":       0.40,
        "socratic":      0.25,
    }
    TIRED_RATE = {"slow": 1.00, "normal": 0.60, "fast": 0.25}

    MOOD_FLIP_PROB = 0.18

    BASE_CONFUSION_SHARP = 0.10
    BASE_CONFUSION_TIRED = 0.35
    MAX_CONFUSION = 0.65

    GAIN_MIN = 0.02
    GAIN_MAX = 0.10

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._mood = "sharp"

    def _maybe_flip_mood(self) -> None:
        if self._rng.random() < self.MOOD_FLIP_PROB:
            self._mood = "tired" if self._mood == "sharp" else "sharp"

    def _fit_score(self, strategy: str, speech_rate: str) -> float:
        if self._mood == "sharp":
            s = self.SHARP_FIT.get(strategy, self._UNKNOWN_STRATEGY_FIT)
            r = self.SHARP_RATE.get(speech_rate, self._UNKNOWN_RATE_FIT)
        else:
            s = self.TIRED_FIT.get(strategy, self._UNKNOWN_STRATEGY_FIT)
            r = self.TIRED_RATE.get(speech_rate, self._UNKNOWN_RATE_FIT)
        return max(0.0, min(1.0, s * r))

    def _confusion_prob(self, fit: float) -> float:
        base = (
            self.BASE_CONFUSION_SHARP if self._mood == "sharp"
            else self.BASE_CONFUSION_TIRED
        )
        prob = base + (self.MAX_CONFUSION - base) * (1.0 - fit)
        prob += self._rng.gauss(0.0, self.NOISE_STD)
        return max(0.0, min(1.0, prob))

    def react(
        self,
        strategy: str,
        speech_rate: str,
        current_mastery: float,
        turn_number: int,
    ) -> TurnOutcome:
        self._maybe_flip_mood()
        return super().react(strategy, speech_rate, current_mastery, turn_number)


def get_all_archetypes() -> List[_BaseStudent]:
    """One freshly-seeded instance of each archetype, in canonical order.

    Seeds are fixed so the sanity-test output is deterministic across
    runs (modulo Python's random module backwards-compat guarantees).
    """
    return [
        FastLearner(seed=42),
        DeepThinker(seed=43),
        SlowLearner(seed=44),
        InconsistentLearner(seed=45),
    ]


def _run_sanity() -> None:
    """Run each archetype for 10 turns at its documented optimum.

    Prints averaged confusion / mastery gain / engagement. Expected
    ordering on the mastery-gain column: SlowLearner lowest,
    FastLearner highest, DeepThinker in between, InconsistentLearner
    variable (depending on mood drift seeded by RNG).
    """
    n_turns = 10
    header = f"{'Archetype':<22} {'best arm':<26} {'avg conf':>9} {'avg dM':>8} {'engage':>8}"
    print(header)
    print("-" * len(header))

    for student in get_all_archetypes():
        best_arm = f"{student.BEST_STRATEGY}+{student.BEST_RATE}"
        total_conf = 0
        total_dm = 0.0
        total_eng = 0
        mastery = 0.0

        for t in range(n_turns):
            outcome = student.react(
                strategy=student.BEST_STRATEGY,
                speech_rate=student.BEST_RATE,
                current_mastery=mastery,
                turn_number=t,
            )
            total_conf += int(outcome.confusion_detected)
            total_dm += (outcome.mastery_after - outcome.mastery_before)
            total_eng += int(outcome.engaged)
            mastery = outcome.mastery_after

        print(
            f"{student.name:<22} {best_arm:<26} "
            f"{total_conf / n_turns:>9.2f} "
            f"{total_dm / n_turns:>8.3f} "
            f"{total_eng / n_turns:>8.2f}"
        )


if __name__ == "__main__":
    _run_sanity()