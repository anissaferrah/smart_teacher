"""Eval dataset loading + schema validation.

# Format (JSONL — one JSON object per line)

Each line is a query with its ground truth :

    {
      "id":              "q001",                          # required, unique
      "query":           "qu'est-ce que la régression linéaire ?",  # required
      "language":        "fr",                            # optional, default "fr"
      "course_id":       "ml_intro",                      # optional
      "chapter":         2,                               # optional, int
      "expected_chunks": ["concept_linear_regression",    # required, list of
                          "concept_least_squares"],       # idea_id strings
      "expected_grades": {                                # optional, for nDCG;
        "concept_linear_regression": 3,                   # higher = more relevant
        "concept_least_squares":     2
      },
      "comment":         "..."                            # optional, free-text
    }

When ``expected_grades`` is absent, all expected_chunks are treated as
binary relevance 1 (sufficient for Recall, P@K, MRR, F1, Hit@K, but
nDCG will be binary too).

# Why JSONL

  - One query per line → trivial to add/edit, easy to grep
  - Streaming-friendly (datasets can grow without rewriting)
  - No comma-trail / outer-array issues
  - Standard format for IR eval datasets (TREC, BEIR, MTEB)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("rag.evaluation.dataset")


@dataclass
class EvalQuery:
    """One labelled query for retrieval evaluation."""

    id:               str
    query:            str
    expected_chunks:  list[str] = field(default_factory=list)
    expected_grades:  dict[str, int] = field(default_factory=dict)
    language:         str = "fr"
    course_id:        Optional[str] = None
    chapter:          Optional[int] = None
    comment:          str = ""

    @property
    def relevance(self) -> dict[str, int]:
        """Effective relevance dict — graded if provided, else binary on
        ``expected_chunks``."""
        if self.expected_grades:
            return dict(self.expected_grades)
        return {cid: 1 for cid in self.expected_chunks}


def parse_query(record: dict) -> EvalQuery:
    """Validate and convert one JSONL record. Raises ``ValueError`` on schema breach."""
    qid = record.get("id")
    text = record.get("query")
    expected = record.get("expected_chunks") or []
    if not qid or not isinstance(qid, str):
        raise ValueError(f"missing/invalid 'id' in record: {record}")
    if not text or not isinstance(text, str):
        raise ValueError(f"missing/invalid 'query' in record id={qid}")
    if not isinstance(expected, list):
        raise ValueError(f"'expected_chunks' must be a list (id={qid})")
    grades = record.get("expected_grades") or {}
    if not isinstance(grades, dict):
        raise ValueError(f"'expected_grades' must be a dict (id={qid})")
    return EvalQuery(
        id=qid,
        query=text,
        expected_chunks=[str(x) for x in expected],
        expected_grades={str(k): int(v) for k, v in grades.items()},
        language=str(record.get("language", "fr")),
        course_id=record.get("course_id"),
        chapter=record.get("chapter"),
        comment=str(record.get("comment", "")),
    )


def load_jsonl(path: str | Path) -> list[EvalQuery]:
    """Read an entire JSONL eval file into a list of EvalQuery objects.

    Skips blank lines. Logs (and skips) lines that fail schema validation
    rather than aborting the whole load — partial datasets are usable.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"eval dataset not found: {p}")
    queries: list[EvalQuery] = []
    seen_ids: set[str] = set()
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            record = json.loads(raw)
            q = parse_query(record)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("skipping line %d in %s: %s", lineno, p.name, exc)
            continue
        if q.id in seen_ids:
            log.warning("duplicate id '%s' at line %d (skipping)", q.id, lineno)
            continue
        seen_ids.add(q.id)
        queries.append(q)
    log.info("loaded %d eval queries from %s", len(queries), p.name)
    return queries


def stream_jsonl(path: str | Path) -> Iterator[EvalQuery]:
    """Streaming variant — one query at a time. For large datasets that
    don't fit comfortably in memory."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"eval dataset not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                record = json.loads(raw)
                yield parse_query(record)
            except (json.JSONDecodeError, ValueError) as exc:
                log.warning("skipping line %d in %s: %s", lineno, p.name, exc)
                continue
