"""Reset script — wipes student data + caches, KEEPS courses.

Use case : start verifying step by step on a clean slate without
losing the time-consuming course ingestion.

What this DELETES :
  - All ``students`` rows (accounts) — JWT tokens become invalid
  - All learning data tied to students :
      learning_sessions, interactions, learning_events,
      student_profiles, student_mistakes, student_mastery,
      practice_attempt, review_queue, learning_gain_tests,
      confusion_events, vark_responses
  - Redis caches : presentation snapshots + narration cross-session
                  + per-student chat history (chat::*)
                  + session contexts (session:*)
                  + seen ideas, paused states

What this KEEPS :
  - ``courses``, ``chapters``, ``sections``, ``concepts``
  - ``ingested_asset`` rows
  - ``rag_chunks`` (RAG vector index)
  - All slide assets on disk (media/)

Run with --apply to actually delete (default is dry run).
"""
from __future__ import annotations

import argparse
import asyncio
import sys


# Tables to truncate. Order matters because of FK cascades — we list
# leaves before parents so the deletes don't conflict.
_STUDENT_TABLES = [
    "vark_responses",
    "confusion_events",
    "learning_gain_tests",
    "review_queue",
    "practice_attempt",
    "student_mastery",
    "student_mistakes",
    "student_profiles",
    "learning_events",
    "interactions",
    "learning_sessions",
    "students",          # last : it's the FK target for many of the above
]

# Redis key patterns to wipe. Each is SCAN-ed and DELETED.
_REDIS_PATTERNS = [
    "chat::*",                 # per-(student, course) chat history
    "presentation:snapshot:*", # per-session per-slide narrations
    "narration:*",             # cross-session narration cache
    "session:*",               # SessionContext blobs
    "seen_ideas:*",            # set of ideas marked "seen"
    "session_token:*",         # one-shot WebSocket auth tokens
]


async def _clear_postgres(apply: bool) -> None:
    print("\n-- Postgres (student-related tables) -------------------------")
    try:
        from sqlalchemy import text
        from database.init_db import AsyncSessionLocal
    except Exception as exc:                                              # noqa: BLE001
        print(f"  Postgres unavailable: {exc} — skipping")
        return

    total = 0
    for table in _STUDENT_TABLES:
        try:
            async with AsyncSessionLocal() as db:
                count = (await db.execute(
                    text(f'SELECT COUNT(*) FROM "{table}"')
                )).scalar() or 0
                print(f"  {table:<30s} : {count:>6} rows")
                total += int(count)
                if apply and count > 0:
                    # TRUNCATE … RESTART IDENTITY CASCADE handles FKs.
                    # We use raw SQL because SQLAlchemy ORM can't
                    # bulk-truncate with FK propagation.
                    await db.execute(
                        text(f'TRUNCATE "{table}" RESTART IDENTITY CASCADE')
                    )
                    await db.commit()
                    print(f"    -> deleted")
        except Exception as exc:                                          # noqa: BLE001
            print(f"  {table:<30s} : SKIP ({exc})")

    print(f"\n  Total student rows: {total}")


async def _clear_redis(apply: bool) -> None:
    print("\n-- Redis (caches + sessions) ---------------------------------")
    try:
        from pedagogy.dialogue import get_redis
        r = await get_redis()
    except Exception as exc:                                              # noqa: BLE001
        print(f"  Redis unavailable: {exc} — skipping")
        return

    total = 0
    for pattern in _REDIS_PATTERNS:
        cursor = 0
        keys: list[str] = []
        while True:
            cursor, batch = await r.scan(cursor=cursor, match=pattern, count=500)
            keys.extend(k.decode() if isinstance(k, bytes) else str(k) for k in batch)
            if cursor == 0:
                break
        print(f"  {pattern:<30s} : {len(keys):>6} keys")
        total += len(keys)
        if apply and keys:
            pipe = r.pipeline()
            for k in keys:
                pipe.delete(k)
            await pipe.execute()
            print(f"    -> deleted")

    print(f"\n  Total Redis keys: {total}")


async def _show_preserved() -> None:
    """Display what's KEPT for confirmation — courses must survive."""
    print("\n-- Preserved (courses + ingestion) --------------------------")
    try:
        from sqlalchemy import text
        from database.init_db import AsyncSessionLocal
        for table in ("courses", "chapters", "sections", "concepts",
                      "ingested_asset", "rag_chunks", "practice_question"):
            try:
                async with AsyncSessionLocal() as db:
                    count = (await db.execute(
                        text(f'SELECT COUNT(*) FROM "{table}"')
                    )).scalar() or 0
                    print(f"  {table:<30s} : {count:>6} rows (kept)")
            except Exception:                                              # noqa: BLE001
                pass
    except Exception:                                                      # noqa: BLE001
        pass


async def _run(apply: bool) -> None:
    if not apply:
        print("=" * 64)
        print("DRY RUN — pass --apply to actually delete")
        print("=" * 64)
    else:
        print("=" * 64)
        print("APPLYING DELETES — this is irreversible")
        print("=" * 64)

    await _show_preserved()
    await _clear_postgres(apply)
    await _clear_redis(apply)

    print("\n" + ("=" * 64))
    if apply:
        print("Done. Students cleared, courses preserved.")
        print("Next step :")
        print("  1. Restart the server : python main.py")
        print("  2. Open http://localhost:8000/  -> redirected to login")
        print("  3. Create a new account (test the new fields)")
    else:
        print("Dry run only. Re-run with --apply to actually delete.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete (default is dry run).",
    )
    args = parser.parse_args()
    asyncio.run(_run(apply=args.apply))
    return 0


if __name__ == "__main__":
    sys.exit(main())
