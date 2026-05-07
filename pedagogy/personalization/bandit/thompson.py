"""Contextual Thompson Sampling bandit for pedagogical strategy selection.

# Why Thompson Sampling

  - **Bayesian** — fits the existing posterior infrastructure of the
    project (Beta posterior in mastery_repo, Dirichlet in
    learning_style/bayes.py).
  - **Optimal regret bounds** — Thompson Sampling matches the LinUCB
    regret rate while being simpler to implement and tune (Russo et al.
    2018, *A Tutorial on Thompson Sampling*).
  - **Exploration-exploitation built in** — the per-arm Beta prior
    naturally explores under-tried arms.

# Why "contextual"

The reward depends on the student's profile (a slow auditive student
won't react the same way to a Socratic prompt as a fast reading
student). We use **disjoint contextual bandits with discretized
context** : each context bucket maintains its own per-arm Beta
posterior. This is :

  - simpler than full LinUCB (no matrix inversion, fast online updates);
  - more expressive than vanilla MAB (context is taken into account);
  - easy to debug (each (context_bucket, arm) has a tractable Beta(α, β));
  - storable in a flat key-value table (Postgres / Redis).

The cost is that the context space is bounded — we hash the continuous
features into a small set of buckets. For Phase 1, this is sufficient.
Phase 3 (offline RL) can replace the discretization with a learned
value function once enough data is collected.

# Reward semantics

The bandit expects rewards in [0, 1]. ``reward = 1.0`` means "success"
(no confusion + concept advanced), ``reward = 0.0`` means "failure"
(confusion detected, no progress). Continuous values in between are
allowed and used directly as Beta updates : α += reward, β += 1 - reward.

# References

  - Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018).
    *A Tutorial on Thompson Sampling.* Foundations and Trends in ML.
  - Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). *A
    contextual-bandit approach to personalized news article
    recommendation.* WWW. (LinUCB — alternative we did NOT pick)
  - Chapelle, O., & Li, L. (2011). *An Empirical Evaluation of Thompson
    Sampling.* NIPS. — established empirical superiority on
    web-personalization tasks.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Iterable

from pedagogy.personalization.bandit.strategies import (
    StrategyAction,
    all_actions,
)

log = logging.getLogger("personalization.bandit.thompson")


# ── Context discretization ──────────────────────────────────────────────


@dataclass(frozen=True)
class ContextBucket:
    """Discretized context — small enough that we can keep one Beta per
    (bucket, arm). The fields are intentionally coarse : Phase 1 needs to
    see meaningful counts per bucket within a few hundred student turns.

    Phase 2 : ajout de `kg_position` (KG-aware bucket).
      - "isolated" : pas de prereqs identifies → choix indep du KG
      - "blocked"  : prereqs <50% mastered → strategie review-prereqs probable
      - "ready"    : prereqs >=50% mastered → strategie new-content efficace

    Pourquoi : permet au bandit d'apprendre "quand un eleve a des prereqs
    faibles, l'arm 'review_prereqs_first' bat 'jump_to_advanced'", au lieu
    de melanger ces 2 contextes dans le meme bucket.

    Trade-off : 4D bucket = 5*3*3*3 = 135 buckets max. Sur 1000 turns,
    chaque bucket voit ~7 obs. Avec 4 arms : ~2 par (bucket, arm). Limite
    statistique mais bandit converge tjrs (Beta priors).
    """

    learning_style: str   # "visual" | "auditory" | "kinesthetic" | "reading" | "mixed"
    pace:           str   # "slow" | "normal" | "fast"
    mastery_level:  str   # "low" | "medium" | "high"
    kg_position:    str = "isolated"   # "isolated" | "blocked" | "ready"

    @property
    def bucket_key(self) -> str:
        return f"{self.learning_style}|{self.pace}|{self.mastery_level}|{self.kg_position}"


def discretize_mastery(mastery_score: float) -> str:
    """Coarse mastery bucket. Boundaries follow the Bloom-1968 mastery
    threshold (0.85) for the upper edge and a midpoint for the lower."""
    if mastery_score < 0.40:
        return "low"
    if mastery_score < 0.85:
        return "medium"
    return "high"


def discretize_response_time(avg_response_time_s: float) -> str:
    """Coarse pace bucket from the average response time. The 60s and
    15s breakpoints come from the existing PersonalizationEngine and
    are documented there as the auto-derive pace logic."""
    if avg_response_time_s > 60.0:
        return "slow"
    if avg_response_time_s < 15.0:
        return "fast"
    return "normal"


def discretize_kg_position(prereqs_mastered_ratio: float | None) -> str:
    """Coarse KG-position bucket selon le ratio de prereqs maitrises pour
    le concept courant.

      - None ou 0 prereqs (concept root) → "isolated"
      - ratio < 0.5 → "blocked"  (eleve trop tot, manque les bases)
      - ratio >= 0.5 → "ready"   (terrain prepare)

    Le seuil 0.5 est arbitraire-mais-defendable : "moitie des prereqs au
    moins" est un standard pedagogique courant (mastery learning ne
    requiert pas tous les prereqs, juste assez).
    """
    if prereqs_mastered_ratio is None:
        return "isolated"
    if prereqs_mastered_ratio < 0.5:
        return "blocked"
    return "ready"


def context_from_profile(
    learning_style: str,
    avg_response_time_s: float,
    mastery_score: float,
    prereqs_mastered_ratio: float | None = None,
) -> ContextBucket:
    """Compose a ContextBucket from raw profile features.

    `prereqs_mastered_ratio` (Phase 2 — KG-aware) :
      - Sur le concept que l'eleve s'apprete a aborder, ratio des prereqs
        (kg.prerequisites_of) deja maitrises (mastery >= MASTERY_THRESHOLD).
      - None si pas de KG dispo ou concept root → bucket "isolated".
    """
    return ContextBucket(
        learning_style=(learning_style or "mixed").lower(),
        pace=discretize_response_time(avg_response_time_s or 0.0),
        mastery_level=discretize_mastery(mastery_score or 0.5),
        kg_position=discretize_kg_position(prereqs_mastered_ratio),
    )


# ── Beta posterior per arm ──────────────────────────────────────────────


@dataclass
class ArmPosterior:
    """Per-arm Beta(α, β) posterior. Default Beta(1, 1) is the uniform
    Bayes-Laplace prior — same convention as mastery_repo.

    The (1, 1) prior is the standard non-informative choice (Laplace's
    rule of succession, 1814; Jaynes 2003, *Probability Theory: The
    Logic of Science*, ch. 6). It encodes "no idea whether this arm is
    good or bad" before any observation.
    """

    alpha: float = 1.0
    beta: float = 1.0

    @property
    def n_pulls(self) -> int:
        """Number of observations integrated (excluding the prior)."""
        return int(round(self.alpha + self.beta - 2))

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng: random.Random | None = None) -> float:
        """Draw one sample from Beta(α, β) — used by Thompson sampling."""
        rng = rng or random
        return rng.betavariate(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        """Apply the Beta-Bernoulli conjugate update.

        For continuous reward in [0, 1] we treat it as a fractional
        Bernoulli observation : α += reward, β += 1 - reward. This is
        the standard handling for Thompson sampling on graded rewards
        (Agrawal & Goyal 2013).
        """
        r = max(0.0, min(1.0, float(reward)))
        self.alpha += r
        self.beta += 1.0 - r


# ── The bandit itself ───────────────────────────────────────────────────


@dataclass
class ContextualThompsonBandit:
    """Disjoint contextual Thompson sampling bandit.

    Maintains one Beta posterior per (context_bucket, arm) pair. Arms
    are ``StrategyAction`` instances; context buckets are
    ``ContextBucket`` instances.

    State is a flat dict keyed by ``"<bucket_key>|<arm_id>"``, easy to
    serialize to JSON / Redis / Postgres. Use ``to_dict`` / ``from_dict``
    for persistence.
    """

    posteriors: dict[str, ArmPosterior] = field(default_factory=dict)
    rng_seed:   int | None = None
    _rng:       random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed) if self.rng_seed is not None else random.Random()

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _key(bucket: ContextBucket, action: StrategyAction) -> str:
        return f"{bucket.bucket_key}|{action.arm_id}"

    def _post(self, bucket: ContextBucket, action: StrategyAction) -> ArmPosterior:
        """Get-or-create the posterior for a (bucket, arm) pair."""
        key = self._key(bucket, action)
        post = self.posteriors.get(key)
        if post is None:
            post = ArmPosterior()
            self.posteriors[key] = post
        return post

    # ── Decision API ────────────────────────────────────────────────

    def select(
        self,
        bucket: ContextBucket,
        candidate_actions: Iterable[StrategyAction] | None = None,
    ) -> StrategyAction:
        """Sample a Beta draw for each candidate arm and return the argmax.

        ``candidate_actions`` defaults to the full action space. Restrict
        it when some actions are unavailable (e.g. SOCRATIC requires the
        student to have spent some time on the concept already).

        Ties are broken randomly — important when Beta draws are equal
        on integer-only data (early stage with no observations).
        """
        candidates = list(candidate_actions) if candidate_actions is not None else list(all_actions())
        if not candidates:
            raise ValueError("ContextualThompsonBandit.select: no candidate actions")

        best: StrategyAction | None = None
        best_sample = -1.0
        # Break ties by random shuffling of the iteration order
        self._rng.shuffle(candidates)
        for action in candidates:
            post = self._post(bucket, action)
            draw = post.sample(self._rng)
            if draw > best_sample:
                best_sample = draw
                best = action
        assert best is not None  # candidates is non-empty
        log.debug(
            "bandit.select bucket=%s arm=%s draw=%.3f n_pulls=%d",
            bucket.bucket_key, best.arm_id, best_sample,
            self._post(bucket, best).n_pulls,
        )
        return best

    def update(self, bucket: ContextBucket, action: StrategyAction, reward: float) -> None:
        """Apply the Beta-Bernoulli update for the (bucket, arm) pair."""
        post = self._post(bucket, action)
        post.update(reward)
        log.debug(
            "bandit.update bucket=%s arm=%s reward=%.3f → α=%.2f β=%.2f",
            bucket.bucket_key, action.arm_id, reward, post.alpha, post.beta,
        )

    # ── Inspection ─────────────────────────────────────────────────

    def n_buckets(self) -> int:
        """Number of distinct context buckets seen so far."""
        return len({k.split("|", 3)[0] + "|" + k.split("|", 3)[1] + "|" + k.split("|", 3)[2]
                    for k in self.posteriors})

    def total_pulls(self) -> int:
        return sum(p.n_pulls for p in self.posteriors.values())

    def best_arm_for(self, bucket: ContextBucket) -> StrategyAction | None:
        """Return the arm with the highest posterior MEAN for ``bucket``.

        Used for monitoring / dashboards — NOT for action selection
        (that's ``select`` with proper Thompson exploration).
        """
        best: StrategyAction | None = None
        best_mean = -1.0
        for action in all_actions():
            post = self.posteriors.get(self._key(bucket, action))
            if post is None:
                continue
            if post.mean > best_mean:
                best_mean = post.mean
                best = action
        return best

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "posteriors": {
                k: {"alpha": p.alpha, "beta": p.beta}
                for k, p in self.posteriors.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextualThompsonBandit":
        bandit = cls()
        for k, v in (data.get("posteriors") or {}).items():
            bandit.posteriors[k] = ArmPosterior(
                alpha=float(v.get("alpha", 1.0)),
                beta=float(v.get("beta", 1.0)),
            )
        return bandit
