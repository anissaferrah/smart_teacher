"""ContextAgent — retrieves grounding chunks from the RAG for the planned ideas.

Pure RAG, no LLM call. Called between Planner and Narrator so the Narrator
has authoritative definitions and prerequisites available when narrating.

Standardise sur Config.RAG_NUM_RESULTS (meme valeur que Q&A retriever) et
transmet idea_id + idea_label + section_idx pour traceability + concept-aware
prompting downstream.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from agentic.state import TutorState
from core.config import Config

log = logging.getLogger("agentic.teaching.context")


class ContextAgent:
    """Augments the state with relevant chunks retrieved from the course RAG."""

    def __init__(self, rag, k: int | None = None) -> None:
        self.rag = rag
        # Default standardise sur Config.RAG_NUM_RESULTS (5) pour coherence
        # avec qa.retriever et endpoints REST (main.py + audio_pipeline)
        self.k = int(k or Config.RAG_NUM_RESULTS)

    def __call__(self, state: TutorState) -> dict[str, Any]:
        start = time.time()
        plan = state.get("plan")
        course_id = state.get("course_id") or ""
        chapter_idx = state.get("chapter_idx", 0)

        if not plan or not getattr(plan, "ideas", None):
            return {
                "retrieved_chunks": [],
                "timings": {**state.get("timings", {}), "context": 0.0},
            }

        # Build a single query from concept-type ideas (ignore intro/summary)
        concept_briefs = [
            idea.content_brief
            for idea in plan.ideas
            if idea.type in ("concept", "example") and idea.content_brief
        ]
        if not concept_briefs:
            concept_briefs = [idea.content_brief for idea in plan.ideas if idea.content_brief]

        query = " ".join(concept_briefs)[:512]
        if not query:
            return {
                "retrieved_chunks": [],
                "timings": {**state.get("timings", {}), "context": 0.0},
            }

        chunks = []
        try:
            raw = self.rag.retrieve_chunks(
                query,
                k=self.k,
                current_chapter_idx=chapter_idx,
                course_id=course_id or None,
            )
            # Normalize to plain dicts. Transmet la metadata enrichie (idea_id, label,
            # section, mastery hints…) pour que Narrator puisse adapter son prompt.
            for item in raw or []:
                # raw items can be (Document, score, source) tuples or Documents
                doc = item[0] if isinstance(item, tuple) else item
                score = item[1] if (isinstance(item, tuple) and len(item) > 1) else None
                content = getattr(doc, "page_content", None) or str(doc)
                meta = getattr(doc, "metadata", {}) or {}
                chunks.append({
                    "content":       content[:1200],
                    "source":        meta.get("source") or meta.get("source_file", ""),
                    "chapter":       meta.get("chapter_idx", meta.get("chapter")),
                    "chapter_title": meta.get("chapter_title", ""),
                    "section_idx":   meta.get("section_idx"),
                    "section_title": meta.get("section_title", ""),
                    # NEW : idea_id + idea_label transmis pour traceability + memory-aware
                    "idea_id":       meta.get("idea_id"),
                    "idea_label":    meta.get("idea_label", ""),
                    "score":         float(score) if score is not None else None,
                })
        except Exception as exc:
            log.warning("context retrieval failed: %s", exc)

        # KG-augmented context — meme logique que le path QA (RetrieverAgent).
        # Avant : ContextAgent appelait rag.retrieve_chunks directement et
        # bypass-ait l'expansion KG → la narration ne beneficiait pas des
        # prereqs / examples / illustrated. Phase 1 closing-loop.
        kg_augmented: list[dict] = []
        if Config.RAG_USE_GRAPH_EXPANSION and chunks:
            try:
                from agentic.qa.retriever import kg_augment_chunks
                kg_augmented = kg_augment_chunks(chunks, self.rag)
            except Exception as exc:
                log.debug("context: KG augmentation skipped (%s)", exc)

        all_chunks = chunks + kg_augmented

        log.info(
            "context: %d direct + %d kg-augmented (course=%s ch=%s query=%dchars k=%d)",
            len(chunks),
            len(kg_augmented),
            course_id[:16] if course_id else "(none)",
            chapter_idx,
            len(query),
            self.k,
        )
        return {
            "retrieved_chunks": all_chunks,
            "timings": {**state.get("timings", {}), "context": round(time.time() - start, 3)},
        }
