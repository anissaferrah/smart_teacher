"""Bootstrap the personalization bandit with synthetic episodes.

Usage :
    python -m scripts.bootstrap_bandit                        # default 2000 turns
    python -m scripts.bootstrap_bandit --students 500 --turns-per-student 20
    python -m scripts.bootstrap_bandit --output bandit_state.json --no-redis
    python -m scripts.bootstrap_bandit --uniform              # off-policy dataset
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pedagogy.personalization.bandit import (
    ContextualThompsonBandit,
    all_actions,
)
from pedagogy.personalization.bandit.simulator import bootstrap_bandit


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bootstrap the personalization bandit.")
    p.add_argument(
        "--students",
        type=int,
        default=200,
        help="Number of distinct simulated students (default: 200)",
    )
    p.add_argument(
        "--turns-per-student",
        type=int,
        default=10,
        help="Turns per simulated student (default: 10)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Save the bootstrapped bandit state to this JSON file (overrides --no-redis-save)",
    )
    p.add_argument(
        "--no-redis-save",
        action="store_true",
        help="Do NOT persist to Redis after bootstrapping (only useful with --output)",
    )
    p.add_argument(
        "--uniform",
        action="store_true",
        help="Use uniform random action selection instead of self-bandit (off-policy dataset)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
    )
    return p


async def _amain(args) -> int:
    log = logging.getLogger("scripts.bootstrap_bandit")

    # 1) Build a fresh bandit
    bandit = ContextualThompsonBandit(rng_seed=args.seed)
    log.info(
        "bootstrapping fresh bandit : %d students × %d turns = %d episodes",
        args.students, args.turns_per_student,
        args.students * args.turns_per_student,
    )

    # 2) Run the simulator
    n_episodes = bootstrap_bandit(
        bandit,
        n_students=args.students,
        turns_per_student=args.turns_per_student,
        seed=args.seed,
        use_bandit_for_select=not args.uniform,
    )
    log.info("done : %d episodes integrated", n_episodes)

    # 3) Stats
    n_actions = sum(1 for _ in all_actions())
    pairs_with_data = sum(1 for p in bandit.posteriors.values() if p.n_pulls > 0)
    print()
    print("=== Bootstrap report ===")
    print(f"  total_pulls       : {bandit.total_pulls()}")
    print(f"  posteriors stored : {len(bandit.posteriors)}")
    print(f"  arm-pairs with data: {pairs_with_data}")
    print(f"  action_space      : {n_actions} arms")
    print()

    # 4) Save
    if args.output:
        path = Path(args.output)
        path.write_text(json.dumps(bandit.to_dict(), indent=2), encoding="utf-8")
        log.info("bandit state saved → %s", path)

    if not args.no_redis_save:
        try:
            from pedagogy.personalization.bandit.repo import (
                _bandit_instance, save_state,
            )
            # Inject our bootstrapped bandit as the singleton, then save.
            import pedagogy.personalization.bandit.repo as repo_module
            repo_module._bandit_instance = bandit
            await save_state()
            log.info("bandit state saved → Redis (bandit:state)")
        except Exception as exc:                                          # noqa: BLE001
            log.warning("Redis save failed (use --output to persist locally): %s", exc)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
