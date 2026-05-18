"""Append-only JSONL writer for retrieval/answer eval pairs.

Captures (question, retrieved_chunks, answer) per turn so the operator
can manually grade `good: true/false` afterwards and feed the file back
as a ground-truth eval set. Best-effort — exceptions are swallowed so a
disk-full log can't break the QA pipeline.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("observability.eval_log")

_LOG_PATH = Path("logs") / "eval_log.jsonl"


def _source_type(chunk: dict[str, Any]) -> str:
    """Derive chunk provenance for downstream filtering.

    "retrieval"  → direct hybrid-search hit
    "kg:<rel>"   → graph-augmented (prereq / example / illustrated)
    "neighbor"   → N±1 cross-slide context of the top hit
    """
    if chunk.get("_via_kg"):
        return f"kg:{chunk.get('_kg_relation') or '?'}"
    if chunk.get("context_type") == "neighbor":
        return "neighbor"
    return "retrieval"


def append_eval_record(
    *,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    answer: str,
    session_id: str | None = None,
    course_id: str | None = None,
) -> None:
    """Append one eval record to logs/eval_log.jsonl. Never raises."""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "session_id": session_id or "",
            "course_id":  course_id or "",
            "question":   question or "",
            "chunks_retrieved": [
                {
                    "rank":            i + 1,
                    "idea_id":         c.get("idea_id"),
                    "score_rerank":    c.get("score"),
                    "score_cosine":    c.get("_vector_score"),
                    "source_type":     _source_type(c),
                    "content_preview": (c.get("content") or "")[:120],
                }
                for i, c in enumerate(retrieved_chunks or [])
            ],
            "answer": answer or "",
            "good":   None,
        }
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        log.debug("eval log append skipped: %s", exc)
