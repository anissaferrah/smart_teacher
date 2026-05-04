#!/usr/bin/env python
"""One-shot migration : create the 3 Phase 2 persistence tables.

Why this script exists
----------------------
``Base.metadata.create_all()`` only creates tables that don't already
exist — it doesn't TOUCH existing ones. After we added 3 new SQLAlchemy
models (``LearningGainTest``, ``ConfusionEvent``, ``VarkResponse``), an
EXISTING DB still won't have these tables until something triggers
``create_all`` again.

This script :
  1. Calls ``create_all`` so the new tables get created
     (existing tables are untouched).
  2. Reports which tables were newly created vs already present.

Idempotent. Safe to re-run.

Usage
-----
    python scripts/migrate_phase2_persistence.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_NEW_TABLES = [
    ("learning_gain_tests",  "Pre-test / post-test results"),
    ("confusion_events",     "Detailed confusion event log"),
    ("vark_responses",       "Raw VARK questionnaire submissions"),
]


async def run_migration() -> bool:
    try:
        from sqlalchemy import text
        from database.init_db import engine
        from database.models import Base
    except Exception as exc:                                                # noqa: BLE001
        print(f"❌ Cannot import DB engine: {exc}")
        return False

    print("=" * 70)
    print("🔧  Migration : Phase 2 persistence tables")
    print("=" * 70)

    try:
        # 1. Inspect : which of our target tables already exist ?
        existing: set[str] = set()
        async with engine.begin() as conn:
            r = await conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ))
            existing = {row[0] for row in r}

        print("\n  Existing target tables before migration :")
        for name, desc in _NEW_TABLES:
            mark = "✅" if name in existing else "—"
            print(f"    {mark} {name:30s} {desc}")

        # 2. Create the missing tables. ``create_all`` is idempotent —
        # it inspects each table's existence before issuing CREATE.
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # 3. Re-inspect — confirm everything is now present.
        async with engine.begin() as conn:
            r = await conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ))
            after = {row[0] for row in r}

        created = sorted(name for name, _ in _NEW_TABLES if name in after - existing)
        already = sorted(name for name, _ in _NEW_TABLES if name in existing & after)

        print("\n  Result :")
        if created:
            for n in created:
                print(f"    🆕 {n} created")
        if already:
            for n in already:
                print(f"    ⏭️  {n} already existed (untouched)")
        missing = [n for n, _ in _NEW_TABLES if n not in after]
        for n in missing:
            print(f"    ❌ {n} STILL MISSING — investigate")

        ok = not missing
        print("\n" + "=" * 70)
        print(f"{'✅' if ok else '❌'}  Migration {'applied.' if ok else 'FAILED.'}")
        print("=" * 70 + "\n")
        return ok
    finally:
        try:
            await engine.dispose()
        except Exception:
            pass


if __name__ == "__main__":
    ok = asyncio.run(run_migration())
    sys.exit(0 if ok else 1)
