#!/usr/bin/env python
"""Reset all caches + course tables so the next ingestion runs clean.

Use this when you want to re-ingest a PDF and verify the title-detection
pipeline from scratch, without any cached vision/concept/embedding
result polluting the run.

What it does
------------
1. **Disk caches** : recursively delete
   - ``cache/vision_descriptions/`` (vision LLM output per slide)
   - ``data/kg_concepts_cache.json`` (KG concept layer)
   - ``data/multimodal_db/docs_cache.json`` (RAG chunks; THIS is why
     "Cache charge (XXX docs)" reappears at boot)
   - ``data/multimodal_db/idea_cache.json`` (LLM idea-chunking cache)
   - ``data/multimodal_db/summary_cache.json`` (per-chunk summaries)

2. **Qdrant** : drop every collection starting with
   ``smart_teacher_multimodal*`` (covers all embedding-model variants).

3. **Redis caches** : delete every key matching the patterns used by
   - ``cache.qa_cache``        (Q&A response cache)
   - ``cache.slide_cache``     (cross-session narration cache)
   - ``rag.embedding_cache``   (BGE-m3 embeddings cache)

4. **Database** : ``TRUNCATE`` (with ``CASCADE``) the COURSE-content
   tables — keeps student accounts, profiles, mastery, learning history.
   Tables truncated:
     - ``rag_chunks``        (vector store metadata)
     - ``concepts``          (extracted concepts)
     - ``ingested_asset``    (uploaded files registry)
     - ``sections``          (course sections)
     - ``chapters``          (course chapters)
     - ``courses``           (course headers)
   Student-side tables (students, student_profiles, student_mastery,
   learning_sessions, interactions, learning_events, practice_question,
   practice_attempt, student_mistakes, review_queue) are PRESERVED.

   To wipe everything (incl. students), use ``database/reset_db_fresh.py``
   — that's the nuclear DROP+CREATE option, this script is the per-course
   re-ingestion option.

Usage
-----
    python scripts/reset_for_reingest.py [--yes] [--keep-titles]
                                          [--keep-disk] [--keep-qdrant]
                                          [--keep-redis] [--keep-db]

    --yes          : skip the confirmation prompt
    --keep-titles  : preserve the slide-title LLM cache (recommended
                     for repeat ingestion tests of the same PDF; saves
                     ~30s per slide on Mistral CPU)
    --keep-disk    : preserve other disk caches (vision, KG, RAG)
    --keep-qdrant  : preserve Qdrant vector collections
    --keep-redis   : preserve Redis caches (Q&A, slide narrations, embeddings)
    --keep-db      : preserve course-content DB tables

By default EVERYTHING is reset; the ``--keep-*`` flags let you target a
subset. Common combo for fast repeat tests :

    python scripts/reset_for_reingest.py --yes --keep-titles
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

# Make the project root importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Disk cache locations ────────────────────────────────────────────────
# Relative to PROJECT_ROOT — no abs paths, so the script is portable.
_DISK_TARGETS: list[tuple[str, Path]] = [
    ("vision LLM cache (descriptions + concepts)",
     PROJECT_ROOT / "cache" / "vision_descriptions"),
    ("KG concepts cache",
     PROJECT_ROOT / "data" / "kg_concepts_cache.json"),
    # ── RAG persistence : these three are why "493 docs" reappear at
    # startup even after a reset. The IdeaGraph is rebuilt from
    # docs_cache.json on every boot, so wiping the JSON wipes the KG too.
    ("RAG docs cache (chunks indexed in Qdrant)",
     PROJECT_ROOT / "data" / "multimodal_db" / "docs_cache.json"),
    ("RAG idea cache (LLM-chunked ideas, avoids re-LLM on re-ingest)",
     PROJECT_ROOT / "data" / "multimodal_db" / "idea_cache.json"),
    ("RAG summary cache (LLM-summarised chunks)",
     PROJECT_ROOT / "data" / "multimodal_db" / "summary_cache.json"),
]

# ── Slide-title cache : tracked separately because it's the most
# expensive to rebuild (one Mistral call per slide, ~30s on CPU).
# Default behaviour : we DO clear it when --yes is passed without any
# flag. Add --keep-titles to preserve it across re-ingestion cycles
# (the cache is keyed by content MD5, so identical slides re-hit it).
_TITLE_CACHE_TARGETS: list[tuple[str, Path]] = [
    ("LLM-text title extraction cache (per slide, expensive: Mistral)",
     PROJECT_ROOT / "cache" / "slide_titles"),
]


# ── Redis key patterns ──────────────────────────────────────────────────
# Each entry : (label, glob pattern) — fed to SCAN/DEL.
_REDIS_PATTERNS: list[tuple[str, str]] = [
    ("Q&A response cache",        "qa_cache:*"),
    ("slide narration cache",     "slide_cache:*"),
    ("embedding cache",           "embedding_cache:*"),
    ("embedding cache (legacy)",  "emb:*"),
    # Conversation history (REST /ask) lives in
    # ``http_history:{sid}:{course_id}`` — NOT cleared here because the
    # student may want to resume a session. Use ``--all-redis`` if you
    # really want to wipe those too.
]


# ── DB tables to truncate (course content only) ─────────────────────────
# Order matters: child tables before parents to avoid FK violations even
# without CASCADE. With CASCADE we could rely on the parent only, but
# being explicit makes the intent visible in the log output.
_DB_TABLES_TO_TRUNCATE: list[str] = [
    "rag_chunks",       # FK → courses, sections
    "concepts",         # FK → courses
    "ingested_asset",   # FK → courses
    "sections",         # FK → chapters
    "chapters",         # FK → courses
    "courses",
]


# ════════════════════════════════════════════════════════════════════════
# DISK
# ════════════════════════════════════════════════════════════════════════

def _delete_targets(targets: list[tuple[str, Path]], verbose: bool = True) -> int:
    """Delete each (label, path) target. Returns count of items removed."""
    removed = 0
    for label, target in targets:
        if not target.exists():
            if verbose:
                print(f"  ⏭️  {label} : nothing at {target}")
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed += 1
            print(f"  🗑️  removed {label} → {target}")
        except Exception as exc:                                            # noqa: BLE001
            print(f"  ⚠️  failed to remove {target}: {exc}")
    return removed


def reset_disk(verbose: bool = True) -> int:
    """Delete the standard disk-cache targets (vision, KG, RAG)."""
    return _delete_targets(_DISK_TARGETS, verbose=verbose)


def reset_title_cache(verbose: bool = True) -> int:
    """Delete the slide-title LLM cache. Separated so callers can
    preserve it across re-ingestion cycles (Mistral CPU cost is high).
    """
    return _delete_targets(_TITLE_CACHE_TARGETS, verbose=verbose)


# ════════════════════════════════════════════════════════════════════════
# QDRANT (vector store collections)
# ════════════════════════════════════════════════════════════════════════

def reset_qdrant(verbose: bool = True) -> int:
    """Delete every Qdrant collection that starts with our project prefix.

    The collection name pattern is :
      ``smart_teacher_multimodal__{embedding_namespace_hash}``

    Each combination (provider, embedding model) gets its own collection
    so we DELETE_ALL matching the prefix in one pass — covers both
    OpenAI and BGE-m3 collections, plus stale ones from older runs.

    Returns count of collections deleted.
    """
    try:
        from qdrant_client import QdrantClient
        from core.config import Config
    except Exception as exc:                                                # noqa: BLE001
        print(f"  ⚠️  Qdrant modules unavailable: {exc} — skipping")
        return 0

    try:
        client = QdrantClient(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT,
            timeout=5.0,
        )
        collections = [c.name for c in client.get_collections().collections]
    except Exception as exc:                                                # noqa: BLE001
        print(f"  ⚠️  Qdrant unreachable ({exc}) — skipping")
        return 0

    target_prefix = "smart_teacher_multimodal"
    targets = [name for name in collections if name.startswith(target_prefix)]
    if not targets:
        print(f"  ⏭️  no Qdrant collections matching '{target_prefix}*'")
        return 0

    deleted = 0
    for name in targets:
        try:
            client.delete_collection(collection_name=name)
            print(f"  🗑️  Qdrant collection '{name}' deleted")
            deleted += 1
        except Exception as exc:                                            # noqa: BLE001
            print(f"  ⚠️  failed to delete '{name}': {exc}")
    return deleted


# ════════════════════════════════════════════════════════════════════════
# REDIS
# ════════════════════════════════════════════════════════════════════════

def reset_redis(verbose: bool = True) -> int:
    """Scan + delete every key matching our cache patterns."""
    try:
        import redis as redis_sync                                          # noqa: F401
        from core.config import Config
    except Exception as exc:                                                # noqa: BLE001
        print(f"  ⚠️  Redis modules unavailable: {exc} — skipping")
        return 0

    try:
        client = redis_sync.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=3,
        )
        client.ping()
    except Exception as exc:                                                # noqa: BLE001
        print(f"  ⚠️  Redis unreachable ({exc}) — skipping")
        return 0

    total_deleted = 0
    for label, pattern in _REDIS_PATTERNS:
        keys = list(client.scan_iter(match=pattern, count=500))
        if not keys:
            if verbose:
                print(f"  ⏭️  {label} ({pattern}) : 0 keys")
            continue
        # DEL in chunks of 1000 to avoid pipeline pressure on huge caches.
        for i in range(0, len(keys), 1000):
            client.delete(*keys[i:i + 1000])
        total_deleted += len(keys)
        print(f"  🗑️  {label} ({pattern}) : {len(keys)} keys deleted")
    return total_deleted


# ════════════════════════════════════════════════════════════════════════
# DB
# ════════════════════════════════════════════════════════════════════════

async def reset_db(verbose: bool = True) -> int:
    """TRUNCATE … CASCADE on the course-content tables."""
    try:
        from sqlalchemy import text
        from database.init_db import engine
    except Exception as exc:                                                # noqa: BLE001
        print(f"  ⚠️  DB modules unavailable: {exc} — skipping")
        return 0

    truncated = 0
    try:
        # One TRUNCATE for all tables — atomic, faster than per-table.
        # CASCADE so any FK-related row in a forgotten table is also
        # cleared, but the only candidates are the tables in our list.
        joined = ", ".join(_DB_TABLES_TO_TRUNCATE)
        sql = f"TRUNCATE {joined} RESTART IDENTITY CASCADE;"
        if verbose:
            print(f"  📋 SQL : {sql}")
        async with engine.begin() as conn:
            await conn.execute(text(sql))
        truncated = len(_DB_TABLES_TO_TRUNCATE)
        print(f"  🗑️  truncated {truncated} table(s) : {joined}")
    except Exception as exc:                                                # noqa: BLE001
        print(f"  ⚠️  DB truncate failed: {exc}")
    finally:
        try:
            await engine.dispose()
        except Exception:
            pass
    return truncated


# ════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════

def _confirm() -> bool:
    print("\n⚠️  This will reset caches and truncate course-content tables.")
    print("   Student accounts / profiles / mastery / learning logs are PRESERVED.")
    answer = input("   Continue ? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation prompt")
    parser.add_argument("--keep-disk", action="store_true",
                        help="don't flush disk caches (vision, KG, RAG)")
    parser.add_argument("--keep-titles", action="store_true",
                        help="don't flush the slide-title LLM cache "
                             "(preserves Mistral title extractions across "
                             "re-ingestion cycles — recommended for repeat "
                             "tests of the same PDF)")
    parser.add_argument("--keep-qdrant", action="store_true",
                        help="don't drop Qdrant collections (keeps vectors)")
    parser.add_argument("--keep-redis", action="store_true",
                        help="don't flush Redis caches")
    parser.add_argument("--keep-db", action="store_true",
                        help="don't truncate DB tables")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔄  RESET — caches + course tables")
    print("=" * 70)

    if not args.yes and not _confirm():
        print("Aborted.")
        return 1

    if not args.keep_disk:
        print("\n🗂️  Disk caches (vision, KG, RAG docs/idea/summary)")
        reset_disk()
    else:
        print("\n⏭️  Disk caches kept (--keep-disk)")

    # Slide-title cache is treated separately so the caller can clear
    # everything else but keep the (expensive) Mistral title cache.
    if not args.keep_titles:
        print("\n🏷️  Slide-title LLM cache")
        reset_title_cache()
    else:
        print("\n⏭️  Slide-title LLM cache kept (--keep-titles)")

    if not args.keep_qdrant:
        print("\n🧬  Qdrant (vector store collections)")
        reset_qdrant()
    else:
        print("\n⏭️  Qdrant kept (--keep-qdrant)")

    if not args.keep_redis:
        print("\n📦  Redis caches")
        reset_redis()
    else:
        print("\n⏭️  Redis caches kept (--keep-redis)")

    if not args.keep_db:
        print("\n🗄️  Database (course-content tables)")
        asyncio.run(reset_db())
    else:
        print("\n⏭️  DB kept (--keep-db)")

    print("\n" + "=" * 70)
    print("✅  Reset complete — ready for a clean re-ingestion.")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
