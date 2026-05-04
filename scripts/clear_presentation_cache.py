"""Clear presentation narration caches (Redis) so new prompts take effect.

Run from the project root :

    python -m scripts.clear_presentation_cache

What it deletes :

  1. ``presentation:snapshot:*``
        Per-session snapshots (session_id, slide_id → narration text +
        cursor + slide_title). Saved by
        ``DialogueManager.save_presentation_snapshot``. Used to resume
        on the same slide without regeneration.

  2. ``narration:*``
        Cross-session narration cache keyed on (course_id, chapter,
        section, lang, level). Saved by ``cache.slide_cache.set_narration``.
        This is the one that survives session restarts and serves the
        same narration to multiple students — also the reason the
        "Bonjour/Bienvenue" intros persisted across sessions.

By default this is a *dry run* — set ``--apply`` to actually delete.
"""
from __future__ import annotations

import argparse
import asyncio
import sys


_PATTERNS = ("presentation:snapshot:*", "narration:*")


async def _clear(apply: bool) -> None:
    from pedagogy.dialogue import get_redis
    r = await get_redis()

    total_keys = 0
    for pattern in _PATTERNS:
        # SCAN avoids blocking Redis the way KEYS does on large DBs.
        cursor = 0
        keys: list[str] = []
        while True:
            cursor, batch = await r.scan(cursor=cursor, match=pattern, count=500)
            keys.extend(k.decode() if isinstance(k, bytes) else str(k) for k in batch)
            if cursor == 0:
                break

        print(f"[{pattern}] : {len(keys)} key(s) found")
        for k in keys[:5]:
            print(f"    sample: {k}")
        if len(keys) > 5:
            print(f"    ... +{len(keys) - 5} more")

        total_keys += len(keys)
        if apply and keys:
            # Pipeline the deletes to avoid N round-trips
            pipe = r.pipeline()
            for k in keys:
                pipe.delete(k)
            await pipe.execute()
            print(f"    deleted {len(keys)} key(s)")

    print()
    if apply:
        print(f"Done. Deleted {total_keys} key(s) total.")
    else:
        print(f"Dry run. {total_keys} key(s) would be deleted.")
        print("Re-run with --apply to actually delete.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete (default is dry run).",
    )
    args = parser.parse_args()
    asyncio.run(_clear(apply=args.apply))
    return 0


if __name__ == "__main__":
    sys.exit(main())
