"""Offline simulation pipeline for the contextual bandit.

Drives 4 student archetypes x 500 turns each through a SINGLE shared
``ContextualThompsonBandit`` and writes the per-turn record to
``turn_history.json`` in this directory. The output is the input to
the reward-weight tuning step that follows.

Run standalone with::

    python -m pedagogy.personalization.bandit.run_simulation

# Why a single shared bandit
# Each archetype has a unique ``learning_style``, so the disjoint
# context buckets do not actually share posteriors across archetypes —
# but the structure mirrors production (one bandit serves all students)
# and keeps the call site honest about the no-leakage guarantee.

# Why mastery is decayed every 50 turns
# Without decay the simulated student saturates at mastery=1.0 within
# ~30-50 turns; the per-turn ``mastery_delta`` collapses to zero and
# the reward stops differentiating arms. Halving mastery every 50
# turns keeps the gradient alive across the full 500-turn run so the
# bandit gets a meaningful learning signal in every cycle.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    from .simulate_students import get_all_archetypes
    from .thompson import ContextualThompsonBandit, context_from_profile
    from .reward import compute_reward
except ImportError:
    from simulate_students import get_all_archetypes
    from thompson import ContextualThompsonBandit, context_from_profile
    from reward import compute_reward


logging.basicConfig(level=logging.WARNING)


N_TURNS = 500
MASTERY_DECAY_EVERY = 50
MASTERY_DECAY_FACTOR = 0.5
PROGRESS_EVERY = 100
BANDIT_SEED = 42

OUT_FILE = Path(__file__).resolve().parent / "turn_history.json"


# Per-archetype profile inputs to ``context_from_profile``. The
# response-time and prereqs values are calibrated so the discretizers
# in thompson.py land on the documented bucket fields:
#
#   time < 15s   → pace="fast"
#   15-60s       → pace="normal"
#   > 60s        → pace="slow"
#
#   ratio is None → kg_position="isolated"
#   ratio < 0.5   → kg_position="blocked"
#   ratio >= 0.5  → kg_position="ready"
_PROFILE_BY_ARCHETYPE: dict[str, dict] = {
    "FastLearner":         {"learning_style": "visual",   "avg_response_time_s": 10.0, "prereqs_mastered_ratio": 0.8},
    "DeepThinker":         {"learning_style": "reading",  "avg_response_time_s": 30.0, "prereqs_mastered_ratio": 0.8},
    "SlowLearner":         {"learning_style": "auditory", "avg_response_time_s": 80.0, "prereqs_mastered_ratio": 0.3},
    "InconsistentLearner": {"learning_style": "mixed",    "avg_response_time_s": 30.0, "prereqs_mastered_ratio": None},
}


def _make_bucket(archetype_name: str, mastery: float):
    p = _PROFILE_BY_ARCHETYPE[archetype_name]
    return context_from_profile(
        learning_style=p["learning_style"],
        avg_response_time_s=p["avg_response_time_s"],
        mastery_score=mastery,
        prereqs_mastered_ratio=p["prereqs_mastered_ratio"],
    )


def _converged_arm(bandit: ContextualThompsonBandit, learning_style: str) -> str:
    """Best arm in the most-observed bucket for this archetype.

    With mastery decay cycling the student through ``low``/``medium``/
    ``high`` mastery buckets, posteriors land in multiple buckets per
    archetype. Querying ``best_arm_for`` on a single end-state bucket
    can hit a sparsely-populated one and report a noisy winner. We
    pick the bucket with the most total pulls (most reliable
    posterior) and report its argmax-mean arm.
    """
    prefix = learning_style + "|"
    pulls_by_bucket: dict[str, int] = {}
    for key, post in bandit.posteriors.items():
        if not key.startswith(prefix):
            continue
        bucket_key = key.rsplit("|", 1)[0]
        pulls_by_bucket[bucket_key] = pulls_by_bucket.get(bucket_key, 0) + post.n_pulls

    if not pulls_by_bucket:
        return "n/a"

    target_bucket = max(pulls_by_bucket, key=pulls_by_bucket.get)

    best_arm_id = "n/a"
    best_mean = -1.0
    for key, post in bandit.posteriors.items():
        if not key.startswith(target_bucket + "|"):
            continue
        arm_id = key.rsplit("|", 1)[1]
        if post.mean > best_mean:
            best_mean = post.mean
            best_arm_id = arm_id
    return best_arm_id


def run_simulation() -> tuple[list[dict], list[dict]]:
    """Run the full simulation and return (records, per-archetype summary)."""
    bandit = ContextualThompsonBandit(rng_seed=BANDIT_SEED)
    records: list[dict] = []
    summaries: list[dict] = []

    for student in get_all_archetypes():
        mastery = 0.0
        running_reward = 0.0
        n_confused = 0
        last_mastery_after = 0.0

        for t in range(N_TURNS):
            bucket = _make_bucket(student.name, mastery)
            action = bandit.select(bucket)

            outcome = student.react(
                strategy=action.strategy.value,
                speech_rate=action.speech_rate.value,
                current_mastery=mastery,
                turn_number=t,
            )
            reward = compute_reward(outcome)
            bandit.update(bucket, action, reward)

            records.append({
                "turn_number":        t,
                "archetype":          student.name,
                "learning_style":     bucket.learning_style,
                "pace":               bucket.pace,
                "mastery_level":      bucket.mastery_level,
                "kg_position":        bucket.kg_position,
                "strategy":           action.strategy.value,
                "speech_rate":        action.speech_rate.value,
                "arm_id":             action.arm_id,
                "mastery_before":     outcome.mastery_before,
                "mastery_after":      outcome.mastery_after,
                "confusion_detected": outcome.confusion_detected,
                "engaged":            outcome.engaged,
                "reward":             reward,
                "mastery_delta":      outcome.mastery_after - outcome.mastery_before,
            })

            mastery = outcome.mastery_after
            last_mastery_after = mastery
            if (t + 1) % MASTERY_DECAY_EVERY == 0:
                mastery = mastery * MASTERY_DECAY_FACTOR

            running_reward += reward
            n_confused += int(outcome.confusion_detected)

            if (t + 1) % PROGRESS_EVERY == 0:
                avg_reward = running_reward / (t + 1)
                print(
                    f"  [{student.name}] turn {t + 1}/{N_TURNS} "
                    f"| avg_reward={avg_reward:.2f} | mastery={mastery:.2f}"
                )

        learning_style = _PROFILE_BY_ARCHETYPE[student.name]["learning_style"]
        summaries.append({
            "archetype":          student.name,
            "turns":              N_TURNS,
            "avg_reward":         running_reward / N_TURNS,
            "avg_conf":           n_confused / N_TURNS,
            "final_mastery":      last_mastery_after,
            "best_arm_converged": _converged_arm(bandit, learning_style),
        })

    return records, summaries


def _print_summary(n_records: int, summaries: list[dict]) -> None:
    print()
    print(f"Simulation complete -- {n_records} turns saved to {OUT_FILE.name}")
    print()
    header = (
        f"{'Archetype':<22}"
        f"{'turns':>6}"
        f"{'avg_reward':>12}"
        f"{'avg_conf':>10}"
        f"{'final_mastery':>15}"
        f"  {'best_arm_converged'}"
    )
    print(header)
    for s in summaries:
        print(
            f"{s['archetype']:<22}"
            f"{s['turns']:>6}"
            f"{s['avg_reward']:>12.2f}"
            f"{s['avg_conf']:>10.2f}"
            f"{s['final_mastery']:>15.2f}"
            f"  {s['best_arm_converged']}"
        )


def main() -> None:
    records, summaries = run_simulation()

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    _print_summary(len(records), summaries)


if __name__ == "__main__":
    main()
