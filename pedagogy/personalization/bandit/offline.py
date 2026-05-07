"""Phase 3 — Offline training utilities for the personalization bandit.

# Why offline training

Phase 1 collects (context, action, reward) tuples in production. Once
the corpus is large enough (say 1000+ tuples), we can :

  - **Re-train from scratch** on the union of (Phase-1 logs ∪ Phase-2
    simulator) to get tighter posteriors than what online learning has
    produced so far. This is *offline policy evaluation* in the bandit
    sense (Dudík, Langford & Li 2011).
  - **Compare strategies** : train one bandit on the dataset, train
    another with a different reward function, and look at which one
    would have made better decisions on a held-out slice (importance-
    weighted off-policy evaluation).

# What this module provides

  - ``Episode`` and ``EpisodeDataset`` : the data carriers.
  - ``train_offline`` : feed an iterable of episodes into a bandit
    instance, in either insertion order or shuffled.
  - ``ips_off_policy_value`` : Inverse Propensity Scoring estimator
    (Horvitz & Thompson 1952) for evaluating a candidate policy
    against a logged dataset.
  - ``train_test_split`` : split episodes for honest evaluation.

# Limits / scope

This is **bandit offline learning**, not full reinforcement learning.
The Phase-1 contextual bandit assumes one-step decisions (no transitions
between states). Full RL with state transitions requires the action's
reward to depend on the next state, which would need a different
training algorithm (e.g. Conservative Q-Learning, IQL — Kumar et al. 2020).

Phase 4 (DRL in production) is when the next-state model becomes worth
the complexity. For now, contextual bandits + IPS evaluation cover the
M2 deliverable.

# References

  - Dudík, M., Langford, J., & Li, L. (2011). *Doubly Robust Policy
    Evaluation and Learning.* ICML.
  - Horvitz, D. G., & Thompson, D. J. (1952). *A Generalization of
    Sampling Without Replacement From a Finite Universe.* JASA.
  - Swaminathan, A., & Joachims, T. (2015). *Counterfactual Risk
    Minimization.* ICML — handles the variance of IPS.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from pedagogy.personalization.bandit.strategies import StrategyAction
from pedagogy.personalization.bandit.thompson import (
    ContextBucket,
    ContextualThompsonBandit,
)

log = logging.getLogger("personalization.bandit.offline")


@dataclass
class LoggedEpisode:
    """One logged interaction : (context, action, reward, propensity).

    ``propensity`` is the probability with which the *logging policy*
    selected this action given the context. Used by IPS evaluation. If
    unknown (e.g. logs come from a uniform-random exploration phase),
    set to ``1.0 / n_actions``.
    """

    context: ContextBucket
    action:  StrategyAction
    reward:  float
    propensity: float = 1.0

    def to_dict(self) -> dict:
        return {
            "context":   {
                "learning_style": self.context.learning_style,
                "pace":           self.context.pace,
                "mastery_level":  self.context.mastery_level,
            },
            "action":    self.action.arm_id,
            "reward":    self.reward,
            "propensity": self.propensity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoggedEpisode":
        ctx = data.get("context") or {}
        return cls(
            context=ContextBucket(
                learning_style=str(ctx.get("learning_style", "mixed")),
                pace=str(ctx.get("pace", "normal")),
                mastery_level=str(ctx.get("mastery_level", "medium")),
            ),
            action=StrategyAction.from_arm_id(str(data.get("action"))),
            reward=float(data.get("reward", 0.0)),
            propensity=float(data.get("propensity", 1.0)),
        )


@dataclass
class EpisodeDataset:
    """A dataset of logged episodes with simple I/O + split helpers."""

    episodes: list[LoggedEpisode] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self) -> Iterator[LoggedEpisode]:
        return iter(self.episodes)

    def append(self, episode: LoggedEpisode) -> None:
        self.episodes.append(episode)

    def shuffle(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        rng.shuffle(self.episodes)

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for ep in self.episodes:
                fh.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
        log.info("wrote %d episodes → %s", len(self.episodes), path.name)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "EpisodeDataset":
        path = Path(path)
        ds = cls()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ds.append(LoggedEpisode.from_dict(json.loads(line)))
            except Exception as exc:                                      # noqa: BLE001
                log.warning("skipping malformed line in %s: %s", path.name, exc)
        log.info("loaded %d episodes from %s", len(ds), path.name)
        return ds


def train_offline(
    bandit: ContextualThompsonBandit,
    dataset: Iterable[LoggedEpisode],
    shuffle_seed: int | None = None,
) -> int:
    """Apply the Beta-Bernoulli updates for every logged episode.

    Returns the number of updates applied.

    Note : the order doesn't matter for Thompson sampling's posterior
    (Beta updates are commutative — α and β just accumulate). The
    ``shuffle_seed`` option is provided for parity with stochastic
    optimizers but is a no-op for this bandit's posterior. It does
    affect any tie-breaking in subsequent ``select`` calls, since the
    bandit's RNG is consumed during shuffling.
    """
    eps = list(dataset)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(eps)
    n = 0
    for ep in eps:
        bandit.update(ep.context, ep.action, ep.reward)
        n += 1
    log.info("train_offline : applied %d updates", n)
    return n


def train_test_split(
    dataset: EpisodeDataset,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[EpisodeDataset, EpisodeDataset]:
    """Random shuffle + split. Standard ML practice (Manning et al. 2008)."""
    rng = random.Random(seed)
    eps = list(dataset.episodes)
    rng.shuffle(eps)
    n_test = int(len(eps) * test_ratio)
    return EpisodeDataset(eps[n_test:]), EpisodeDataset(eps[:n_test])


# ── Off-policy evaluation ───────────────────────────────────────────────


def ips_off_policy_value(
    target_policy: Callable[[ContextBucket], StrategyAction],
    dataset: Iterable[LoggedEpisode],
) -> float:
    """Inverse Propensity Scoring estimator of the target policy's value.

    Formula (Horvitz & Thompson 1952) :

        V_IPS = (1 / |D|) Σ  reward · 1[π(c) == a] / propensity

    Where π is the candidate (target) policy, c the logged context, a
    the logged action, and propensity the prob the LOGGING policy
    chose ``a`` given ``c``.

    Returns the estimated expected reward of ``target_policy`` if it had
    been deployed on this dataset's contexts. Higher = better policy.

    Caveats :
      - High variance when propensity is near zero (rare arms).
      - Unbiased only if propensities are correctly logged.
      - Use ``ips_clipped_off_policy_value`` for a lower-variance variant.
    """
    n = 0
    total = 0.0
    for ep in dataset:
        n += 1
        if ep.propensity <= 0:
            continue
        if target_policy(ep.context) == ep.action:
            total += ep.reward / ep.propensity
    if n == 0:
        return 0.0
    return total / n


def ips_clipped_off_policy_value(
    target_policy: Callable[[ContextBucket], StrategyAction],
    dataset: Iterable[LoggedEpisode],
    clip: float = 10.0,
) -> float:
    """Clipped IPS — bound the importance weight to ``clip`` to reduce
    variance at the cost of a small bias (Swaminathan & Joachims 2015)."""
    n = 0
    total = 0.0
    for ep in dataset:
        n += 1
        if ep.propensity <= 0:
            continue
        if target_policy(ep.context) == ep.action:
            weight = min(clip, 1.0 / ep.propensity)
            total += ep.reward * weight
    if n == 0:
        return 0.0
    return total / n


def policy_from_bandit(bandit: ContextualThompsonBandit) -> Callable[[ContextBucket], StrategyAction]:
    """Wrap a trained bandit's deterministic best-arm-by-mean policy."""
    def _pi(bucket: ContextBucket) -> StrategyAction:
        best = bandit.best_arm_for(bucket)
        if best is not None:
            return best
        # Fallback : random-uniform when the bucket has no observation
        from pedagogy.personalization.bandit.strategies import all_actions
        return next(iter(all_actions()))
    return _pi
