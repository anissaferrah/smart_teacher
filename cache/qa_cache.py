"""Q&A response cache (Redis).

Stores generated Q&A answers keyed by the deterministic content of the
turn: ``(question_normalised, slide_hash, course_id, language)``.

Why
---
On a CPU-only Ollama setup, each Q&A turn costs 3-5 LLM calls totalling
3-10 minutes (Intent + Rewriter + Responder + sometimes Definition
fallback). Repeating the *same* question on the *same* slide is a common
pattern when a student misses the first answer or asks a teammate to
listen. Caching the final response saves the entire pipeline on hits.

Key design
----------
The hash includes:
  - The *normalised* question (lowercase, whitespace collapsed, trailing
    punctuation stripped) — so "What is RANSAC?", "what is ransac",
    "What is RANSAC?  " all hit the same entry.
  - The *first 500 chars* of slide content — so the same question on a
    different slide gets a different answer (the responder is slide-
    aware).
  - The course_id and language.

We deliberately DON'T include session_id so cache is shared across
sessions/students on the same course.

TTL is 6 hours: long enough to cover a study session, short enough that
content updates eventually invalidate.

What's cached
-------------
Only ``ai_response`` (the text answer) and ``citations`` (RAG sources).
Audio bytes are NOT cached — TTS depends on voice/rate which vary per
session.

Disable via config: ``ENABLE_QA_CACHE=false`` in .env.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Optional

from pedagogy.dialogue import get_redis

log = logging.getLogger("cache.qa_cache")


# 6 hours: covers a typical study session, short enough that content
# edits propagate within a day.
DEFAULT_TTL = int(os.getenv("QA_CACHE_TTL", "21600"))


def _enabled() -> bool:
    return os.getenv("ENABLE_QA_CACHE", "true").lower() == "true"


def _normalise_question(q: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation.

    Two questions that differ only in casing, trailing whitespace, or
    final '?' should hit the same cache entry.
    """
    if not q:
        return ""
    s = q.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip("?!.;:, ")
    return s


def _key(question: str, slide_content: str, course_id: str, language: str) -> str:
    """Deterministic cache key for a Q&A turn."""
    q_norm = _normalise_question(question)
    slide_excerpt = (slide_content or "")[:500]
    raw = f"{q_norm}|{slide_excerpt}|{course_id or ''}|{(language or 'en')[:2]}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
    return f"qa_cache:{digest}"


async def get_qa_response(
    question: str,
    slide_content: str,
    course_id: str = "",
    language: str = "en",
) -> Optional[dict]:
    """Look up a cached Q&A response.

    Returns the cached dict (with ``answer``, ``citations``, ``meta``)
    or ``None`` on miss / cache disabled / Redis unavailable.
    """
    if not _enabled() or not question:
        return None
    key = _key(question, slide_content, course_id, language)
    try:
        r = await get_redis()
        raw = await r.get(key)
        if not raw:
            return None
        data = json.loads(raw)
        log.info(
            "✅ Q&A cache HIT (course=%s, q=%r) → 0 LLM calls",
            (course_id or "?")[:8],
            _normalise_question(question)[:40],
        )
        return data
    except Exception as exc:                                                # noqa: BLE001
        log.debug(f"qa_cache get failed: {exc}")
        return None


async def set_qa_response(
    question: str,
    slide_content: str,
    answer: str,
    *,
    course_id: str = "",
    language: str = "en",
    citations: Optional[list] = None,
    meta: Optional[dict] = None,
    ttl: int = DEFAULT_TTL,
) -> None:
    """Persist a Q&A response in Redis.

    Failures are silent — cache is best-effort, never blocks the
    response path.
    """
    if not _enabled() or not question or not answer:
        return
    key = _key(question, slide_content, course_id, language)
    payload = {
        "answer": answer,
        "citations": citations or [],
        "meta": {
            **(meta or {}),
            "cached_at": time.time(),
            "course_id": course_id or "",
            "language": (language or "en")[:2],
        },
    }
    try:
        r = await get_redis()
        await r.setex(key, ttl, json.dumps(payload, ensure_ascii=False))
        log.debug(
            "💾 Q&A cached (course=%s, q=%r, %d chars, ttl=%ds)",
            (course_id or "?")[:8],
            _normalise_question(question)[:40],
            len(answer),
            ttl,
        )
    except Exception as exc:                                                # noqa: BLE001
        log.debug(f"qa_cache set failed: {exc}")


async def invalidate_course(course_id: str) -> int:
    """Drop all cached Q&A responses for a course (call after re-ingest).

    Returns the number of keys deleted (0 if cache disabled or empty).
    """
    if not _enabled() or not course_id:
        return 0
    try:
        r = await get_redis()
        # Scan for keys with this course's id baked in. Since the key is
        # an opaque digest, we have to scan all qa_cache entries and
        # check the stored payload's meta.course_id. For a small cache
        # this is fine; for larger setups, switch to a secondary index.
        deleted = 0
        async for key in r.scan_iter(match="qa_cache:*"):
            try:
                raw = await r.get(key)
                if not raw:
                    continue
                data = json.loads(raw)
                if (data.get("meta") or {}).get("course_id") == course_id:
                    await r.delete(key)
                    deleted += 1
            except Exception:
                continue
        if deleted:
            log.info("Q&A cache: invalidated %d entries for course %s", deleted, course_id[:8])
        return deleted
    except Exception as exc:                                                # noqa: BLE001
        log.debug(f"qa_cache invalidate failed: {exc}")
        return 0
