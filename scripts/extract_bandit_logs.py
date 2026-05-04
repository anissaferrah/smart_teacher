"""Extract Postgres LearningEvent rows → JSONL of LoggedEpisode.

Usage :
    python -m scripts.extract_bandit_logs                              # all events
    python -m scripts.extract_bandit_logs --days 30                    # last 30 days
    python -m scripts.extract_bandit_logs --output bandit_logs.jsonl
    python -m scripts.extract_bandit_logs --combine-with-simulator     # add 2k synthetic episodes

The JSONL output can be fed to ``train_offline`` :

    from rag.evaluation import load_jsonl   # JSONL helper not specific
    from pedagogy.personalization.bandit import (
        EpisodeDataset, ContextualThompsonBandit, train_offline,
    )

    ds = EpisodeDataset.from_jsonl("bandit_logs.jsonl")
    bandit = ContextualThompsonBandit()
    train_offline(bandit, ds)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pedagogy.personalization.bandit import (
    ContextualThompsonBandit,
    EpisodeDataset,
    LoggedEpisode,
    all_actions,
    generate_episodes,
)
from pedagogy.personalization.bandit.log_extractor import (
    extract_episodes_from_postgres,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract Postgres LearningEvent → JSONL bandit dataset.",
    )
    p.add_argument(
        "--output",
        default="bandit_logs.jsonl",
        help="Output JSONL path (default: bandit_logs.jsonl)",
    )
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only events from the last N days (default: all)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="Defensive cap on rows pulled from Postgres",
    )
    p.add_argument(
        "--combine-with-simulator",
        action="store_true",
        help="Append synthetic episodes to the dataset (useful for cold-start)",
    )
    p.add_argument(
        "--sim-students",
        type=int,
        default=200,
        help="Number of synthetic students if --combine-with-simulator (default: 200)",
    )
    p.add_argument(
        "--sim-turns",
        type=int,
        default=10,
        help="Turns per synthetic student (default: 10)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
    )
    return p


async def _amain(args) -> int:
    log = logging.getLogger("scripts.extract_bandit_logs")

    # 1) Pull from Postgres
    from datetime import datetime, timedelta
    since = datetime.utcnow() - timedelta(days=args.days) if args.days else None
    log.info(
        "extracting from Postgres (since=%s, limit=%d)",
        since or "all", args.limit,
    )
    ds = await extract_episodes_from_postgres(since=since, limit=args.limit)
    n_real = len(ds)
    log.info("real-user episodes : %d", n_real)

    # 2) Optionally combine with simulator
    if args.combine_with_simulator:
        n_actions = sum(1 for _ in all_actions())
        n_added = 0
        for ep in generate_episodes(
            n_students=args.sim_students,
            turns_per_student=args.sim_turns,
            seed=42,
            bandit_select=None,  # uniform random (clean off-policy)
        ):
            ds.append(LoggedEpisode(
                context=ep.bucket,
                action=ep.action,
                reward=ep.reward,
                propensity=1.0 / n_actions,
            ))
            n_added += 1
        log.info("synthetic episodes added : %d", n_added)

    # 3) Write JSONL
    output = Path(args.output)
    ds.to_jsonl(output)

    print()
    print("=== Extraction summary ===")
    print(f"  output      : {output}")
    print(f"  total rows  : {len(ds)}")
    print(f"    - from Postgres : {n_real}")
    if args.combine_with_simulator:
        print(f"    - synthetic     : {len(ds) - n_real}")
    print()
    if len(ds) == 0:
        print("⚠️  Empty dataset. Use --combine-with-simulator to bootstrap.")
        return 1
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
