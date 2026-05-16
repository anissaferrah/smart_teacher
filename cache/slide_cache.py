"""Cross-session narration cache (Redis).

Stores generated narrations for slides, keyed by:
    (course_id, chapter_idx, section_idx, language, student_level)

This cache is **cross-session** : if student A presents slide X then
student B opens the same slide X, we return the cached narration without
re-calling the LLM. Saves a Teaching Graph round-trip (~2 min on Ollama CPU).

Complementary to `pedagogy.dialogue.save/load_presentation_snapshot` which
is per-session and stores the cursor + paused state.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

from pedagogy.dialogue import get_redis
from core.config import Config

log = logging.getLogger("cache.slide_cache")


# TTL alias (single source of truth in core/config.py)
DEFAULT_TTL = Config.NARRATION_CACHE_TTL


def _normalize(s: str | None) -> str:
    """Lowercase + strip + replace spaces with underscores for stable keys."""
    if not s:
        return ""
    return (s or "").strip().lower().replace(" ", "_")[:64]


def _key(course_id: str, chapter_idx: int, section_idx: int, language: str, student_level: str) -> str:
    """Build a stable Redis key for a slide narration."""
    parts = [
        _normalize(course_id) or "default",
        str(int(chapter_idx)),
        str(int(section_idx)),
        _normalize(language) or "fr",
        _normalize(student_level) or "lycée",
    ]
    return "narration:" + ":".join(parts)


async def get_narration(
    course_id: str,
    chapter_idx: int,
    section_idx: int,
    language: str = "fr",
    student_level: str = "lycée",
) -> Optional[str]:
    """Return cached narration text or None on miss / error / Redis down.

    Returns just the text (not the full payload) for caller convenience.
    """
    if not course_id:
        return None
    try:
        r = await get_redis()
        raw = await r.get(_key(course_id, chapter_idx, section_idx, language, student_level))
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            text = (payload or {}).get("text") if isinstance(payload, dict) else None
            return text or None
        except json.JSONDecodeError:
            return None
    except Exception as exc:
        log.debug("get_narration failed: %s", exc)
        return None


async def set_narration(
    course_id: str,
    chapter_idx: int,
    section_idx: int,
    language: str,
    student_level: str,
    text: str,
    *,
    engine: str = "teaching_graph",
    meta: dict | None = None,
    ttl: int = DEFAULT_TTL,
) -> bool:
    """Persist a generated narration. Returns True on success, False otherwise."""
    if not course_id or not text or not text.strip():
        return False
    payload = {
        "text": text,
        "engine": engine,
        "meta": meta or {},
        "language": language,
        "student_level": student_level,
        "chapter_idx": int(chapter_idx),
        "section_idx": int(section_idx),
        "generated_at": time.time(),
        "content_hash": hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12],
    }
    try:
        r = await get_redis()
        await r.setex(
            _key(course_id, chapter_idx, section_idx, language, student_level),
            int(ttl),
            json.dumps(payload, ensure_ascii=False),
        )
        return True
    except Exception as exc:
        log.debug("set_narration failed: %s", exc)
        return False


async def invalidate(
    course_id: str,
    chapter_idx: int,
    section_idx: int,
    language: str = "fr",
    student_level: str = "lycée",
) -> bool:
    """Delete a single cached narration."""
    try:
        r = await get_redis()
        await r.delete(_key(course_id, chapter_idx, section_idx, language, student_level))
        return True
    except Exception as exc:
        log.debug("invalidate failed: %s", exc)
        return False


async def invalidate_course(course_id: str) -> int:
    """Delete all narrations for a course. Returns count deleted (best-effort)."""
    if not course_id:
        return 0
    try:
        r = await get_redis()
        pattern = f"narration:{_normalize(course_id) or 'default'}:*"
        deleted = 0
        async for key in r.scan_iter(match=pattern, count=100):
            await r.delete(key)
            deleted += 1
        return deleted
    except Exception as exc:
        log.debug("invalidate_course failed: %s", exc)
        return 0
