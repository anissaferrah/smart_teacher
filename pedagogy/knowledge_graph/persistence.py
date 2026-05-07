"""Disk persistence for the concept layer of the KnowledgeGraph.

# Why only the concept layer

The idea-level graph (~384 nodes for a typical course, with depends_on
and illustrates edges) is **fast** to rebuild from ``rag.all_docs`` —
it's a deterministic walk over the ingested chunks. ``builder.py``
already caches it process-local for that reason.

The expensive part is the **concept layer** : ``ConceptFromTitles``
runs ~9 LLM calls (each ~30-60s on Ollama Mistral) to canonicalise
titles, generate descriptions, extract sub-concepts. That's 5-15 min
per course on cold start.

This module persists ONLY the concept layer (``ConceptInfo`` list).
Loading it back at startup skips the LLM calls entirely. The idea
graph is rebuilt from RAG as before — fast and deterministic.

# Cache invalidation

The cache file embeds ``n_docs`` (size of the RAG corpus). On load,
if the current ``n_docs`` differs, we treat the cache as stale and
return ``None`` so the caller re-extracts from scratch. This catches:
  - new course ingested
  - a course re-ingested with different chunking
  - cache file corruption

Granularity is per-course : each ``course_id`` has its own cache
entry, so adding course B doesn't invalidate course A.

# File layout

    data/kg_concepts_cache.json

    {
        "version": 1,
        "courses": {
            "<course_id>": {
                "n_docs": 384,
                "saved_at": 1730000000,
                "concepts": [
                    {"name": "...", "display_name": "...", ...},
                    ...
                ]
            },
            ...
        }
    }
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from core.config import Config
from pedagogy.knowledge_graph.graph import ConceptInfo

log = logging.getLogger("pedagogy.knowledge_graph.persistence")


_CACHE_VERSION = 1
_CACHE_PATH: Path = (
    Path(getattr(Config, "LOGS_DIR", "logs")).parent / "data" / "kg_concepts_cache.json"
)


def _ensure_dir() -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:                                                # noqa: BLE001
        log.debug("kg cache dir create failed: %s", exc)


def _load_file() -> dict:
    if not _CACHE_PATH.exists():
        return {"version": _CACHE_VERSION, "courses": {}}
    try:
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        if data.get("version") != _CACHE_VERSION:
            log.info("kg cache version mismatch — discarding old file")
            return {"version": _CACHE_VERSION, "courses": {}}
        if "courses" not in data:
            data["courses"] = {}
        return data
    except Exception as exc:                                                # noqa: BLE001
        log.warning("kg cache read failed: %s — starting fresh", exc)
        return {"version": _CACHE_VERSION, "courses": {}}


def _write_file(data: dict) -> None:
    _ensure_dir()
    try:
        # Atomic-ish write: tmp + rename
        tmp = _CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_CACHE_PATH)
    except Exception as exc:                                                # noqa: BLE001
        log.warning("kg cache write failed: %s", exc)


def _concept_to_serial(c: ConceptInfo) -> dict:
    return {
        "name":           c.name,
        "display_name":   c.display_name,
        "canonical_name": c.canonical_name,
        "description":    c.description,
        "bloom_level":    c.bloom_level,
        "course_id":      c.course_id,
        "chapter_idxs":   sorted(c.chapter_idxs),
        "score":          float(c.score),
        "idea_ids":       sorted(c.idea_ids),
    }


def _serial_to_concept(d: dict) -> ConceptInfo:
    return ConceptInfo(
        name=d.get("name", ""),
        display_name=d.get("display_name", ""),
        canonical_name=d.get("canonical_name", ""),
        description=d.get("description", ""),
        bloom_level=d.get("bloom_level", ""),
        course_id=d.get("course_id", ""),
        chapter_idxs=set(d.get("chapter_idxs") or []),
        score=float(d.get("score", 0.0)),
        idea_ids=set(d.get("idea_ids") or []),
    )


# ── Public API ────────────────────────────────────────────────────────


def save_concepts(course_id: str, concepts: list[ConceptInfo], n_docs: int) -> None:
    """Persist the concept layer for one course.

    ``n_docs`` is the size of the RAG corpus at the time of extraction —
    used as a cheap invalidation signal: if the corpus grows or shrinks
    later, the cache is considered stale and the caller re-extracts.
    """
    if not course_id:
        return
    data = _load_file()
    data["courses"][course_id] = {
        "n_docs":   int(n_docs),
        "saved_at": int(time.time()),
        "concepts": [_concept_to_serial(c) for c in concepts],
    }
    _write_file(data)
    log.info(
        "💾 kg cache saved : %d concepts for course=%s (n_docs=%d)",
        len(concepts), course_id[:16], n_docs,
    )


def load_concepts(course_id: str, n_docs: int) -> Optional[list[ConceptInfo]]:
    """Return the cached ``ConceptInfo`` list for ``course_id`` if fresh.

    Returns None if :
      - no cache file exists
      - no entry for this course
      - ``n_docs`` mismatch (corpus changed → cache stale)

    On a cache hit, the caller can ``kg.attach_concepts(loaded)`` and
    skip the LLM-heavy extraction entirely.
    """
    if not course_id:
        return None
    data = _load_file()
    entry = data.get("courses", {}).get(course_id)
    if not entry:
        return None
    cached_n = int(entry.get("n_docs", -1))
    if cached_n != int(n_docs):
        log.info(
            "kg cache stale for course=%s (cached n_docs=%d, current=%d)",
            course_id[:16], cached_n, n_docs,
        )
        return None
    raw = entry.get("concepts") or []
    concepts = [_serial_to_concept(d) for d in raw]
    log.info(
        "✅ kg cache hit : %d concepts loaded for course=%s "
        "(saved %ds ago)",
        len(concepts),
        course_id[:16],
        max(0, int(time.time()) - int(entry.get("saved_at", 0))),
    )
    return concepts


def invalidate_course(course_id: str) -> None:
    """Drop the cache entry for one course (e.g. after re-ingestion)."""
    if not course_id:
        return
    data = _load_file()
    if course_id in data.get("courses", {}):
        del data["courses"][course_id]
        _write_file(data)
        log.info("kg cache invalidated for course=%s", course_id[:16])


def invalidate_all() -> None:
    """Drop the entire cache file."""
    try:
        if _CACHE_PATH.exists():
            _CACHE_PATH.unlink()
            log.info("kg cache file deleted")
    except Exception as exc:                                                # noqa: BLE001
        log.debug("kg cache delete failed: %s", exc)
