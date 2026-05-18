"""Pretty-print logs/eval_log.jsonl as a table.

Usage:
    python scripts/show_eval_log.py
    python scripts/show_eval_log.py --path logs/eval_log.jsonl --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _good_marker(g: object) -> str:
    if g is True:
        return "y"
    if g is False:
        return "n"
    return "?"


def _trim(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="logs/eval_log.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="show only the last N records (0 = all)")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"no log at {path}", file=sys.stderr)
        return 1

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        records = records[-args.limit:]

    header = f"{'#':>3}  {'question':<60}  {'chunks':>6}  {'top_idea_id':<24}  {'good':<4}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(records, start=1):
        chunks = r.get("chunks_retrieved") or []
        top_id = (chunks[0].get("idea_id") if chunks else "") or "-"
        print(
            f"{i:>3}  {_trim(r.get('question', ''), 60):<60}  "
            f"{len(chunks):>6}  {_trim(str(top_id), 24):<24}  "
            f"{_good_marker(r.get('good')):<4}"
        )
    print(f"\n{len(records)} record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
