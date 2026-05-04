#!/usr/bin/env python
"""One-shot migration : add ``student_profiles.course_id`` column.

Why this script exists
----------------------
The SQLAlchemy ``StudentProfile`` model was updated to include a new
``course_id`` foreign key (per-course scoping of the profile), and the
old ``UNIQUE(student_id)`` constraint was replaced by
``UNIQUE(student_id, course_id)``.

``Base.metadata.create_all()`` (used by ``database/init_db.py``) only
creates tables that don't already exist — it does NOT update columns or
constraints on tables that pre-date the model change. So existing
deployments hit ``UndefinedColumnError: column student_profiles.course_id
does not exist`` until this migration is applied.

What this script does (idempotent)
----------------------------------
1. ``ALTER TABLE … ADD COLUMN IF NOT EXISTS course_id UUID
   REFERENCES courses(id) ON DELETE CASCADE``
2. ``CREATE INDEX IF NOT EXISTS ix_student_profiles_course_id …``
3. Drop the legacy ``UNIQUE(student_id)`` constraint if it exists
   (PostgreSQL auto-named it on the column ; we look it up dynamically).
4. ``ALTER TABLE … ADD CONSTRAINT IF NOT EXISTS uq_student_course_profile
   UNIQUE(student_id, course_id)`` — wrapped in DO block because PG
   doesn't support ``IF NOT EXISTS`` directly on ADD CONSTRAINT before
   PG 17.

All steps are guarded so re-running the script is safe.

Usage
-----
    python scripts/migrate_student_profiles_course_id.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_MIGRATION_SQL = [
    # 1. Add the column (nullable, FK to courses).
    """
    ALTER TABLE student_profiles
        ADD COLUMN IF NOT EXISTS course_id UUID
        REFERENCES courses(id) ON DELETE CASCADE
    """,
    # 2. Index for the foreign key (matches what SQLAlchemy would create).
    """
    CREATE INDEX IF NOT EXISTS ix_student_profiles_course_id
        ON student_profiles(course_id)
    """,
    # 3. Drop the legacy UNIQUE(student_id) constraint if it exists.
    # PostgreSQL auto-names single-column unique constraints created
    # without an explicit name as "<table>_<col>_key". We probe both
    # that and a few other common variants.
    """
    DO $$
    DECLARE
        constraint_name TEXT;
    BEGIN
        SELECT conname INTO constraint_name
        FROM pg_constraint
        WHERE conrelid = 'student_profiles'::regclass
          AND contype = 'u'
          AND array_length(conkey, 1) = 1
          AND EXISTS (
              SELECT 1 FROM pg_attribute
              WHERE attrelid = conrelid
                AND attnum = conkey[1]
                AND attname = 'student_id'
          )
        LIMIT 1;

        IF constraint_name IS NOT NULL THEN
            EXECUTE format('ALTER TABLE student_profiles DROP CONSTRAINT %I',
                           constraint_name);
            RAISE NOTICE 'Dropped legacy UNIQUE constraint % on student_id', constraint_name;
        END IF;
    END$$
    """,
    # 4. Add the new UNIQUE(student_id, course_id) constraint, idempotent.
    # PG 16 doesn't support IF NOT EXISTS on ADD CONSTRAINT, so we DO
    # block + check pg_constraint first.
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'uq_student_course_profile'
              AND conrelid = 'student_profiles'::regclass
        ) THEN
            ALTER TABLE student_profiles
                ADD CONSTRAINT uq_student_course_profile
                UNIQUE (student_id, course_id);
            RAISE NOTICE 'Added UNIQUE(student_id, course_id) constraint';
        END IF;
    END$$
    """,
]


async def run_migration() -> bool:
    try:
        from sqlalchemy import text
        from database.init_db import engine
    except Exception as exc:                                                # noqa: BLE001
        print(f"❌ Cannot import DB engine: {exc}")
        return False

    print("=" * 70)
    print("🔧  Migration : student_profiles.course_id (idempotent)")
    print("=" * 70)

    try:
        async with engine.begin() as conn:
            for i, sql in enumerate(_MIGRATION_SQL, 1):
                print(f"\n  Step {i}/{len(_MIGRATION_SQL)} :")
                print(f"    {sql.strip().splitlines()[0][:80]}…")
                try:
                    await conn.execute(text(sql))
                    print(f"    ✅ ok")
                except Exception as exc:                                    # noqa: BLE001
                    print(f"    ⚠️  step {i} skipped: {exc}")
        print("\n" + "=" * 70)
        print("✅  Migration applied.")
        print("=" * 70 + "\n")
        return True
    finally:
        try:
            await engine.dispose()
        except Exception:
            pass


if __name__ == "__main__":
    ok = asyncio.run(run_migration())
    sys.exit(0 if ok else 1)
