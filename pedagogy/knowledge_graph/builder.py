"""IdeaGraph singleton + lazy build from the RAG instance.

The graph is rebuilt on demand from ``rag.all_docs`` (in-memory list of
Documents the RAG holds after ingestion). It is cached process-local for
fast subsequent access.

# When to invalidate

Call :func:`reset` after :
  - A new course ingestion (new ideas + edges added)
  - A graph schema change (new metadata fields)

Otherwise the graph stays valid for the process lifetime.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from pedagogy.knowledge_graph.graph import IdeaGraph

log = logging.getLogger("pedagogy.knowledge_graph.builder")

_singleton: Optional[IdeaGraph] = None
_singleton_n_docs: int = -1
_lock = threading.Lock()


def build_from_rag(rag: Any) -> IdeaGraph:
    """Eagerly build a fresh IdeaGraph from a RAG instance's ``all_docs``.

    Does NOT use the cache. Use :func:`get_or_build` for the cached version.
    """
    g = IdeaGraph()
    docs = getattr(rag, "all_docs", None) or []
    g.add_documents(docs)
    log.info("IdeaGraph built from RAG: %s", g.stats())
    return g


def get_or_build(rag: Any) -> IdeaGraph:
    """Return the cached IdeaGraph, rebuilding if RAG document count changed.

    Cheap on hot path: only rebuilds when the underlying RAG corpus has
    grown / shrunk (i.e. after a new ingestion). Otherwise returns the
    same instance.
    """
    global _singleton, _singleton_n_docs
    if rag is None:
        return IdeaGraph()      # empty graph — caller still works
    n_docs = len(getattr(rag, "all_docs", []) or [])
    with _lock:
        if _singleton is None or n_docs != _singleton_n_docs:
            _singleton = build_from_rag(rag)
            _singleton_n_docs = n_docs
        return _singleton


def reset() -> None:
    """Drop the cached graph — next call to ``get_or_build`` will rebuild.

    Call after a re-ingestion, schema migration, or in tests.
    """
    global _singleton, _singleton_n_docs
    with _lock:
        _singleton = None
        _singleton_n_docs = -1
