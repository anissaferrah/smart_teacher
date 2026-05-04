# database/init_db.py
"""
Smart Teacher — Initialisation de la base de données PostgreSQL
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text

from core.config import Config
from database.models import (
    Base,
    Student, Course, Chapter, Section, Concept,
    LearningSession, Interaction,
    LearningEvent,
    StudentProfile, StudentMistake, StudentMastery,
    PracticeQuestion, PracticeAttempt, ReviewQueue,
    IngestedAssetDB,
    RAGChunk, SystemLog, PerformanceMetric,
)

# Re-export models so SQLAlchemy metadata picks them up via this module.
# Stage 3 : ConceptKG / ConceptChunkLink / ConceptCooccurrence retires —
# concepts vivent maintenant dans le KnowledgeGraph (pedagogy/knowledge_graph).
__all__ = [
    "Base", "AsyncSessionLocal", "engine",
    "create_tables", "get_db", "check_db_connection",
    "Student", "Course", "Chapter", "Section", "Concept",
    "LearningSession", "Interaction", "LearningEvent",
    "StudentProfile", "StudentMistake", "StudentMastery",
    "PracticeQuestion", "PracticeAttempt", "ReviewQueue",
    "IngestedAssetDB", "RAGChunk", "SystemLog",
    "PerformanceMetric",
]

log = logging.getLogger("SmartTeacher.Database")

# Engine AsyncPG
engine = create_async_engine(
    Config.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
)
# Mask credentials but show host/db so operators see WHERE we connect
def _safe_db_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        u = urlparse(url)
        return f"{u.scheme}://{u.username or '?'}:***@{u.hostname}:{u.port}{u.path}"
    except Exception:
        return "***"

log.info(
    "🐘 POSTGRES engine init | url=%s | pool_size=10 max_overflow=20",
    _safe_db_url(Config.DATABASE_URL),
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── SQLAlchemy event hooks for per-query logging ─────────────────────
# Logs every executed query at INFO level so operators see EXACTLY what
# Postgres is asked to do, with timings and connection-pool state.
# To enable, set LOG_SQL_QUERIES=1 in env (off by default for noisy
# production but ON for debugging).
import os as _os
if _os.getenv("LOG_SQL_QUERIES", "1") == "1":
    import time as _pgtime
    from sqlalchemy import event as _sa_event

    @_sa_event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        context._query_start_time = _pgtime.time()
        # Truncate long statements to keep logs readable
        stmt_short = " ".join(statement.split())[:200]
        log.debug(
            "🐘 PG QUERY START | %s%s | params=%s",
            stmt_short, "..." if len(statement) > 200 else "",
            str(parameters)[:100] if parameters else "{}",
        )

    @_sa_event.listens_for(engine.sync_engine, "after_cursor_execute")
    def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        elapsed_ms = (_pgtime.time() - getattr(context, "_query_start_time", _pgtime.time())) * 1000
        op = statement.strip().split(None, 1)[0].upper() if statement.strip() else "?"
        rowcount = cursor.rowcount if hasattr(cursor, "rowcount") else "?"
        log.debug(
            "🐘 PG QUERY DONE | op=%s rows=%s took=%.1fms",
            op, rowcount, elapsed_ms,
        )

    @_sa_event.listens_for(engine.sync_engine, "connect")
    def _on_connect(dbapi_connection, connection_record):
        log.info("🐘 PG CONNECT | new connection acquired (pool grew)")

    @_sa_event.listens_for(engine.sync_engine, "checkout")
    def _on_checkout(dbapi_connection, connection_record, connection_proxy):
        log.debug("🐘 PG CHECKOUT | connection borrowed from pool")

    @_sa_event.listens_for(engine.sync_engine, "checkin")
    def _on_checkin(dbapi_connection, connection_record):
        log.debug("🐘 PG CHECKIN | connection returned to pool")


async def create_tables():
    """Crée toutes les tables + applique les migrations additives idempotentes."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await run_lazy_migrations()
    log.info("✅ Tables PostgreSQL créées/vérifiées")


# ── Lazy schema migrations ─────────────────────────────────────────────
#
# Additive ALTER TABLE patches applied at boot. Only handles the case where
# the SQLAlchemy model has columns the DB lacks (forward drift). It NEVER
# drops columns, changes types, or touches data — for those, write a proper
# Alembic migration instead. The list is data-driven and idempotent: each
# run checks `information_schema.columns` and only adds what's missing.
#
# Add an entry here when you add a column to a model and want existing
# dev/staging DBs to pick it up automatically without a manual ALTER.

_LAZY_COLUMN_PATCHES: dict[str, list[tuple[str, str]]] = {
    "student_profiles": [
        # Sprint 6 personnalisation cognitive — added 2025
        ("pace",                        "VARCHAR(20) DEFAULT 'normal'"),
        ("avg_response_time_s",         "DOUBLE PRECISION DEFAULT 0.0"),
        ("confusion_rate",              "DOUBLE PRECISION DEFAULT 0.0"),
        ("preferred_explanation_depth", "VARCHAR(20) DEFAULT 'balanced'"),
        # Personalization JSON bag (VARK self-report, Bayes posterior, …)
        ("preferences",                 "JSONB DEFAULT '{}'::jsonb"),
    ],
    "students": [
        # Academic level used for prompt adaptation (LLM tunes vocabulary
        # and depth). Defaults to "lycée" so existing accounts keep working.
        ("student_level",               "VARCHAR(20) DEFAULT 'lycée'"),
    ],
}


# ── Type patches ──────────────────────────────────────────────────────
# (table, column) → (target_postgres_type, USING expression).
# Applied with ALTER COLUMN ... TYPE ... USING ... only if current type
# differs from target. Idempotent: skipped on subsequent boots.
#
# Use cases:
#   - VARCHAR storing UUID strings → migrate to UUID type for proper indexing
#     and to eliminate runtime "operator does not exist: varchar = uuid"
#     errors when callers pass uuid.UUID objects.
#
# These conversions are STRICTLY destructive in the formal sense (column
# type changes), but data-preserving when the source data parses cleanly to
# the target type. The USING expression performs the cast — if any row has
# unparseable data, Postgres rejects the ALTER and we catch the exception
# without crashing boot. The admin must then clean the data manually.

_LAZY_TYPE_PATCHES: list[tuple[str, str, str, str]] = [
    # (table, column, target_type, USING expression)
    ("learning_sessions",   "student_id", "uuid", '"student_id"::uuid'),
    ("interactions",        "student_id", "uuid", '"student_id"::uuid'),
    ("learning_events",     "student_id", "uuid", '"student_id"::uuid'),
    ("student_mistake",     "student_id", "uuid", '"student_id"::uuid'),
    ("performance_metrics", "student_id", "uuid", '"student_id"::uuid'),
]


async def run_lazy_migrations() -> None:
    """Apply additive + type-conversion patches to keep DB in sync with models.

    Idempotent: safe to call on every boot. Two phases :
      1. ADD COLUMN for each entry in `_LAZY_COLUMN_PATCHES` not already present
      2. ALTER COLUMN TYPE for each entry in `_LAZY_TYPE_PATCHES` whose current
         postgres type differs from the target

    Failures are logged at WARNING and don't crash boot.
    """
    async with engine.begin() as conn:
        # ── Phase 1: additive columns ──
        for table_name, columns in _LAZY_COLUMN_PATCHES.items():
            result = await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = :t"
                ),
                {"t": table_name},
            )
            existing = {row[0] for row in result}

            for col_name, col_def in columns:
                if col_name in existing:
                    log.debug("migration: %s.%s already present — skipping", table_name, col_name)
                    continue
                stmt = (
                    f'ALTER TABLE "{table_name}" '
                    f'ADD COLUMN IF NOT EXISTS "{col_name}" {col_def}'
                )
                try:
                    await conn.execute(text(stmt))
                    log.info("✅ migration: added %s.%s (%s)", table_name, col_name, col_def)
                except Exception as exc:                                # noqa: BLE001
                    log.warning("⚠️ migration: failed to add %s.%s — %s", table_name, col_name, exc)

        # ── Phase 2: type conversions ──
        for table_name, col_name, target_type, using_expr in _LAZY_TYPE_PATCHES:
            # Read the current udt_name (postgres native type name, lowercased)
            result = await conn.execute(
                text(
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = :c"
                ),
                {"t": table_name, "c": col_name},
            )
            row = result.first()
            if row is None:
                log.debug("migration: %s.%s not present — skipping type patch", table_name, col_name)
                continue
            current_type = (row[0] or "").lower()
            if current_type == target_type.lower():
                log.debug(
                    "migration: %s.%s already %s — skipping type patch",
                    table_name, col_name, target_type,
                )
                continue

            stmt = (
                f'ALTER TABLE "{table_name}" '
                f'ALTER COLUMN "{col_name}" TYPE {target_type} USING {using_expr}'
            )
            try:
                await conn.execute(text(stmt))
                log.info(
                    "✅ migration: converted %s.%s from %s → %s",
                    table_name, col_name, current_type, target_type,
                )
            except Exception as exc:                                    # noqa: BLE001
                # Common cause: existing rows contain non-uuid-shaped strings.
                # Surface clearly so an operator can clean and re-run.
                log.warning(
                    "⚠️ migration: failed to convert %s.%s (%s → %s) — %s. "
                    "Underlying data probably contains values that don't cast to %s.",
                    table_name, col_name, current_type, target_type, exc, target_type,
                )


async def get_db():
    """Dépendance FastAPI pour obtenir une session DB."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def check_db_connection() -> bool:
    """Vérifie la connexion à PostgreSQL."""
    import time as _t
    _t0 = _t.time()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        elapsed_ms = (_t.time() - _t0) * 1000
        log.info(f"🐘 PG ping OK | took={elapsed_ms:.0f}ms | url={_safe_db_url(Config.DATABASE_URL)}")
        return True
    except Exception as e:
        log.warning(f"🐘 PG ping FAIL | err={e}")
        return False