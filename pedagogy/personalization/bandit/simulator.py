"""Phase 2 — Synthetic student simulator for bandit bootstrapping.

# Why a simulator

The Phase 1 contextual bandit needs ~50-100 observations per
(context_bucket, arm) pair before its posterior tightens enough to
be useful in deployment (Russo et al. 2018, sample-complexity
analysis). With 24 buckets × 18 arms = 432 cells, that's ~30k
observations — far more than a single tutor instance can collect
during a thesis demo.

The simulator generates **synthetic episodes** that bootstrap the
bandit's posterior to a useful state, so when it goes live the
exploration burden is already paid.

# Design : hidden-preference SimulatedStudent

Each simulated student has :

  - a public **profile** that the bandit observes (learning_style,
    pace, mastery_level) — the same features as a real student;
  - a hidden **preference vector** over (strategy × speech_rate) —
    unknown to the bandit, drawn at student creation time. This
    encodes "what actually works for this student".

The reward function noisily samples the preference value : the
student is more likely to be confused / less likely to advance
mastery when the bandit picks a low-preference arm. Noise level is
configurable per student to model variance.

# Why hidden preferences are correlated with profile

A "visual + medium mastery" student is intentionally **biased** toward
the ANALOGY and EXAMPLE strategies — that's how the Phase 1 bandit can
learn to associate context buckets with arms. The bias is implemented
via a profile-conditional Dirichlet prior on the preference vector
(see ``_preference_priors``).

# Caveats

  - This is a **toy environment** : the simulated rewards are NOT
    a faithful model of real student outcomes. Phase 1 production
    deployment will collect real data ; the simulator is just a
    sample-efficient way to start with non-flat priors.
  - The student preferences are STATIC. Real students' preferences
    drift as they learn (Vygotsky's ZPD shifts). Modeling drift is
    Phase 4 territory.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from pedagogy.personalization.bandit.reward import TurnOutcome, compute_reward
from pedagogy.personalization.bandit.strategies import (
    SpeechRate,
    Strategy,
    StrategyAction,
    all_actions,
)
from pedagogy.personalization.bandit.thompson import ContextBucket

log = logging.getLogger("personalization.bandit.simulator")


# ── Preference priors per profile bucket ────────────────────────────────
#
# Hand-curated bias on which strategies tend to suit which student type.
# Values are unnormalized weights (the simulator normalizes per-student).
# These reflect the standard ITS literature heuristics — they're the
# "ground truth" the bandit is supposed to discover from observations.
#
#   visual    → analogy, decomposition (visual mental schemas)
#   auditory  → socratic, simpler_words (oral conversation)
#   reading   → recap, decomposition (structured / textual)
#   kinesthetic → example (concrete, manipulable)

_STYLE_BIAS: dict[str, dict[Strategy, float]] = {
    "visual":      {Strategy.ANALOGY: 3.0, Strategy.DECOMPOSITION: 2.0, Strategy.EXAMPLE: 1.5},
    "auditory":    {Strategy.SOCRATIC: 3.0, Strategy.SIMPLER_WORDS: 2.0, Strategy.RECAP: 1.5},
    "kinesthetic": {Strategy.EXAMPLE: 3.0, Strategy.DECOMPOSITION: 2.0, Strategy.ANALOGY: 1.5},
    "reading":     {Strategy.RECAP: 3.0, Strategy.DECOMPOSITION: 2.0, Strategy.SIMPLER_WORDS: 1.5},
    "mixed":       {},   # uniform
}

# Pace bias on speech rate
#   slow pace student → prefers "slow" speech
#   fast pace student → prefers "fast" speech
#   normal → "normal"
_PACE_BIAS: dict[str, dict[SpeechRate, float]] = {
    "slow":   {SpeechRate.SLOW: 3.0, SpeechRate.NORMAL: 1.5},
    "normal": {SpeechRate.NORMAL: 3.0},
    "fast":   {SpeechRate.FAST: 3.0, SpeechRate.NORMAL: 1.5},
}

# Mastery bias : low-mastery students benefit more from RECAP / SIMPLER_WORDS,
# high-mastery from SOCRATIC (challenge-oriented)
_MASTERY_BIAS: dict[str, dict[Strategy, float]] = {
    "low":    {Strategy.SIMPLER_WORDS: 2.0, Strategy.RECAP: 2.0},
    "medium": {},   # neutral
    "high":   {Strategy.SOCRATIC: 2.0},
}


def _build_preference(
    bucket: ContextBucket,
    rng: random.Random,
    base_weight: float = 1.0,
) -> dict[StrategyAction, float]:
    """Build a normalized preference distribution over arms for one
    simulated student in ``bucket``."""
    raw: dict[StrategyAction, float] = {}
    style_table = _STYLE_BIAS.get(bucket.learning_style, {})
    pace_table = _PACE_BIAS.get(bucket.pace, {})
    mastery_table = _MASTERY_BIAS.get(bucket.mastery_level, {})

    for action in all_actions():
        weight = base_weight
        weight += style_table.get(action.strategy, 0.0)
        weight += pace_table.get(action.speech_rate, 0.0)
        weight += mastery_table.get(action.strategy, 0.0)
        # Per-student noise so two "visual / medium / normal" students
        # don't have identical preferences
        weight *= rng.uniform(0.7, 1.3)
        raw[action] = max(0.1, weight)

    total = sum(raw.values())
    return {a: w / total for a, w in raw.items()}


@dataclass
class SimulatedStudent:
    """A synthetic student with hidden preferences over the action space."""

    bucket: ContextBucket
    preference: dict[StrategyAction, float] = field(default_factory=dict)
    confusion_noise: float = 0.15   # P(reverse outcome) — irreducible noise
    mastery_step: float = 0.10      # mastery delta on a "good" turn
    seed: int | None = None

    @classmethod
    def random(cls, rng: random.Random | None = None) -> "SimulatedStudent":
        """Create a random student with profile-correlated preferences."""
        rng = rng or random.Random()
        bucket = ContextBucket(
            learning_style=rng.choice(["visual", "auditory", "kinesthetic", "reading", "mixed"]),
            pace=rng.choice(["slow", "normal", "fast"]),
            mastery_level=rng.choice(["low", "medium", "high"]),
        )
        return cls(
            bucket=bucket,
            preference=_build_preference(bucket, rng),
        )

    def react(
        self,
        action: StrategyAction,
        rng: random.Random | None = None,
    ) -> TurnOutcome:
        """Sample an outcome for ``action``. Higher preference → higher
        chance of a good turn; ``confusion_noise`` injects irreducible
        randomness to prevent perfect inference."""
        rng = rng or random.Random()
        pref = self.preference.get(action, 0.0)
        # Map preference (∈ [0, 1] approximately, since it's a normalized
        # distribution over many arms) to a probability of "good turn".
        # We rescale so the maximum preference yields ~0.9 P(good).
        max_pref = max(self.preference.values()) if self.preference else 1.0
        p_good_clean = (pref / max_pref) if max_pref > 0 else 0.5
        # Apply noise : flip outcome with probability confusion_noise
        if rng.random() < self.confusion_noise:
            p_good = 1.0 - p_good_clean
        else:
            p_good = p_good_clean
        good_turn = rng.random() < p_good

        # Build outcome : confusion_detected is the negation of "good turn"
        # for the textual signal ; mastery step is positive only on good turns.
        mastery_before = 0.5  # nominal — bandit doesn't observe absolute level here
        delta = self.mastery_step if good_turn else 0.0
        return TurnOutcome(
            confusion_detected=not good_turn,
            mastery_before=mastery_before,
            mastery_after=mastery_before + delta,
            engaged=True,  # we assume the student stays in the simulated session
        )


@dataclass
class Episode:
    """One simulated turn : (bucket, action, reward)."""

    bucket: ContextBucket
    action: StrategyAction
    reward: float


def simulate_episode(
    student: SimulatedStudent,
    action: StrategyAction,
    rng: random.Random | None = None,
) -> Episode:
    outcome = student.react(action, rng=rng)
    reward = compute_reward(outcome)
    return Episode(bucket=student.bucket, action=action, reward=reward)


def generate_episodes(
    n_students: int,
    turns_per_student: int,
    seed: int = 42,
    bandit_select=None,
) -> Iterator[Episode]:
    """Generate a stream of synthetic episodes.

    Args :
        n_students : how many distinct simulated students to spin up
        turns_per_student : how many turns each student plays
        seed : reproducibility
        bandit_select : optional callable ``(bucket) -> StrategyAction``.
            When provided, the action is chosen by the bandit (so the
            bandit gets to learn from its own decisions). When None,
            actions are sampled uniformly at random — useful for an
            unbiased dataset.
    """
    rng = random.Random(seed)
    for _ in range(n_students):
        student = SimulatedStudent.random(rng=rng)
        for _ in range(turns_per_student):
            if bandit_select is not None:
                action = bandit_select(student.bucket)
            else:
                action = rng.choice(list(all_actions()))
            yield simulate_episode(student, action, rng=rng)


def bootstrap_bandit(
    bandit,
    n_students: int = 200,
    turns_per_student: int = 10,
    seed: int = 42,
    use_bandit_for_select: bool = True,
) -> int:
    """Run the simulator for ``n_students × turns_per_student`` turns and
    feed the outcomes into ``bandit``.

    By default the bandit picks its own actions (``use_bandit_for_select=True``)
    so it learns from a self-consistent on-policy stream. Setting it to
    False gives a uniform-action dataset useful for offline RL training
    (Phase 3) where having coverage of low-preference arms is required.

    Returns the number of episodes generated.
    """
    select_fn = bandit.select if use_bandit_for_select else None
    n = 0
    for ep in generate_episodes(
        n_students=n_students,
        turns_per_student=turns_per_student,
        seed=seed,
        bandit_select=select_fn,
    ):
        bandit.update(ep.bucket, ep.action, ep.reward)
        n += 1
    log.info(
        "bootstrap_bandit: %d episodes (%d students × %d turns)",
        n, n_students, turns_per_student,
    )
    return n
