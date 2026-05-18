"""RetrieverAgent — RAG retrieval + graph-augmented context expansion.

Pipeline :
  1. RAG retrieval (BM25 + dense + RRF + cross-encoder rerank, all in
     ``rag.multimodal_rag``).
  2. Memory enrichment : annotate each chunk with ``mastery_score`` and
     ``seen`` (no re-ranking — the previous version had arbitrary
     mastery-penalty / seen-penalty / unseen-boost multipliers, retired
     because they had no empirical justification).
  3. Graph-augmented context expansion : for the top retrieved chunk,
     pull the direct prerequisites of its ``idea_id`` from the
     ``IdeaGraph`` and add them as extra context chunks (marked
     ``_via_kg=True`` for traceability). This gives the responder
     access to the conceptual scaffolding behind the answer — useful
     when the student is missing a prerequisite the answer assumes.

Graph augmentation is bounded by ``_KG_PREREQ_CAP`` (per-chunk) and
``_KG_TOTAL_AUGMENT_CAP`` (global). The references for KG-based query
expansion are Doignon & Falmagne (1985) and the line of work on
knowledge-space-aware ITS (ALEKS, Knewton).

Outputs on state :
  retrieved_chunks : list[dict] with idea_id, idea_label, mastery_score,
                     seen, and (for graph-augmented entries) _via_kg=True
                     plus _augmented_from=<source idea_id>.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agentic.state import TutorState
from core.config import Config
from rag.metadata import RetrievedChunk

log = logging.getLogger("agentic.qa.retriever")


# Operational caps for graph-augmented expansion.
#
# Pourquoi des caps : sans bornes, on pourrait gonfler le contexte LLM avec
# 50 prereqs transitifs et 30 examples → context-window overflow + latence.
# Les chiffres ci-dessous ne sont PAS des seuils comportementaux (rien de
# semantique ne change a 4 vs 5) mais des garde-fous operationnels.
#
# Calibration heuristique :
#   - top-K classique = 5 chunks (Config.RAG_NUM_RESULTS, ~1500 chars chacun)
#   - LLM context window utile : ~6000-8000 tokens reserves au contexte
#   - chunks_text_cap responder = 500 chars/chunk → 5*500 + 4*500 = 4500 chars
#     dans la zone safe pour gpt-4o-mini.
#
# A relacher si vous changez vers un LLM avec un context window plus grand
# (Claude Opus, GPT-4o, etc.) ET si les responses LLM omettent des info
# pertinentes du KG.

_KG_PREREQ_CAP = 3            # Max prereqs (depends_on edges) pulled par chunk source.
                              # 3 = un theoreme typique a 2-3 prereqs explicites.
                              # Si > 3, probablement decomposition LLM trop fine.
_KG_EXAMPLES_CAP = 2          # Max exemples (illustrates_id) pulled par concept.
                              # 2 = "donne-moi un exemple de X" reclame 1-2, pas 5.
_KG_TOTAL_AUGMENT_CAP = 4     # Cap global cumule sur toutes les relations.
                              # < cap pour eviter explosion combinatoire si le top-1
                              # a beaucoup de prereqs ET d'exemples ET un illustrated.
_KG_SOURCE_CHUNKS_TO_EXPAND = 1  # On ne developpe QUE le top-1 du retrieval.
                                 # Justification : le top-2/3 sont souvent
                                 # tangentiels — leurs voisins ajouteraient du bruit.
                                 # Si le retrieval est mauvais, le top-1 l'est aussi
                                 # et l'expansion ne sauvera pas la reponse de toute facon.


class RetrieverAgent:
    """Retrieves top-K chunks, augmente avec mastery + seen state."""

    def __init__(self, rag, k: int | None = None) -> None:
        self.rag = rag
        # Standardise sur Config.RAG_NUM_RESULTS (meme valeur partout)
        self.k = int(k or Config.RAG_NUM_RESULTS)

    async def __call__(self, state: TutorState) -> dict[str, Any]:
        start = time.time()
        if self.rag is None:
            return {
                "retrieved_chunks": [],
                "timings": {**state.get("timings", {}), "retriever": 0.0},
            }

        query = (state.get("rewritten_query") or "").strip()
        if not query:
            payload = state.get("event_payload") or {}
            if isinstance(payload, dict):
                query = (payload.get("text") or payload.get("transcript") or "").strip()

        if not query:
            return {
                "retrieved_chunks": [],
                "timings": {**state.get("timings", {}), "retriever": 0.0},
            }

        course_id = state.get("course_id") or None
        chapter_idx = state.get("chapter_idx", 0)
        session_id = state.get("session_id") or ""
        student_id = state.get("student_id") or None
        review_mode = bool(state.get("review_mode", False))

        # Pre-retrieval query expansion using the rewriter's anchored
        # concept (Zheng et al. 2023). When the question is short or
        # contains pronouns ("et donc ?", "comment ça marche ?"), the
        # concept extracted by step-back reasoning gives the retriever a
        # stronger semantic anchor than the rewritten query alone.
        #
        # We append the concept verbatim — RRF fusion + cross-encoder
        # rerank in the RAG layer can absorb the redundancy without bias
        # (the anchored concept is treated as additional query terms,
        # not a separate query). When the concept is already inside the
        # rewritten query (case-insensitive substring match), we skip
        # the expansion to avoid pure repetition that would inflate
        # term frequency without adding signal.
        anchored_concept = (state.get("anchored_concept") or "").strip()
        expanded_query = query
        if anchored_concept and anchored_concept.lower() not in query.lower():
            expanded_query = f"{query} {anchored_concept}"
            log.info(
                "retriever: query expanded with anchored concept (orig=%d → expanded=%d chars)",
                len(query), len(expanded_query),
            )

        # 1. Retrieval brut (over-fetch pour avoir de la marge apres re-scoring)
        chunks: list[dict[str, Any]] = []
        try:
            raw = await asyncio.to_thread(
                self.rag.retrieve_chunks,
                expanded_query,
                k=self.k * 2,  # over-fetch
                current_chapter_idx=chapter_idx,
                course_id=course_id,
            )
            for item in raw or []:
                doc = item[0] if isinstance(item, tuple) else item
                score = item[1] if (isinstance(item, tuple) and len(item) > 1) else None
                content = getattr(doc, "page_content", None) or str(doc)
                meta = getattr(doc, "metadata", {}) or {}
                chunks.append({
                    "content":     content[:1500],
                    "source":      meta.get("source", ""),
                    "chapter":     meta.get("chapter_idx", meta.get("chapter")),
                    "section_idx": meta.get("section_idx"),
                    "section_title": meta.get("section_title", ""),
                    "idea_id":     meta.get("idea_id"),
                    "idea_label":  meta.get("idea_label", ""),
                    "slide_idx":   meta.get("slide_idx"),
                    "score":       float(score) if score is not None else 0.5,
                    "_vector_score": meta.get("_vector_score"),
                    "mastery_score": None,  # filled below
                    "seen":        False,
                    "context_type": meta.get("context_type"),
                })
        except Exception as exc:
            log.warning("retriever failed: %s", exc)

        # 2. Memory layer : seen + mastered (parallel pour minimiser latence)
        idea_ids = [c["idea_id"] for c in chunks if c.get("idea_id")]
        seen: set[str] = set()
        mastery_map: dict[str, float] = {}
        if idea_ids:
            try:
                # Imports lazy pour eviter cycle (mastery_repo touche init_db)
                from pedagogy.dialogue import get_seen_ideas
                from pedagogy.mastery_repo import MasteryRepo

                tasks = [
                    get_seen_ideas(session_id) if session_id else asyncio.sleep(0, result=set()),
                    MasteryRepo.get_scores_bulk(student_id, course_id, idea_ids) if student_id else asyncio.sleep(0, result={}),
                ]
                seen, mastery_map = await asyncio.gather(*tasks)
            except Exception as exc:
                log.debug(f"memory layer lookup failed: {exc}")

        # 3. Tagging only — chunks keep their natural RAG score.
        # We surface mastery + seen as metadata so the responder/narrator can
        # adapt its tone (e.g. "you've already seen this") but we don't bias
        # ranking with arbitrary multipliers.
        for c in chunks:
            iid = c.get("idea_id")
            if not iid:
                continue
            c["mastery_score"] = mastery_map.get(iid, 0.5)
            c["seen"] = iid in seen

        # 4. Keep top k by raw RAG score
        chunks.sort(key=lambda c: -c.get("score", 0))
        chunks = chunks[: self.k]

        # 5. Graph-augmented context expansion — pull prerequisites + examples
        # + the illustrated concept (if top-1 is itself an example) of the
        # top retrieved chunk(s) from the IdeaGraph. The query is unchanged;
        # we extend the candidate pool, not the search input. Failures here
        # are non-fatal (KG might be empty during early ingest).
        kg_augmented = (
            self._augment_with_kg(chunks)
            if Config.RAG_USE_GRAPH_EXPANSION else []
        )

        # 6. N±1 cross-slide context for the rank-#1 hit only. Pulls the
        # chunks on the slides immediately before and after the strongest
        # match (same course, same chapter) so the LLM sees the
        # surrounding lecture context. Limiting to the top hit keeps the
        # prompt focused; downstream _MAX_CITED_CHUNKS still caps total.
        top_neighbors: list[dict[str, Any]] = []
        if chunks:
            top = chunks[0]
            t_chap = top.get("chapter")
            t_slide = top.get("slide_idx")
            if course_id and t_chap is not None and t_slide is not None:
                try:
                    nbr_docs = self.rag.get_slide_neighbors(
                        course=course_id,
                        chapter_idx=int(t_chap),
                        slide_idx=int(t_slide),
                        exclude_idea_ids={c.get("idea_id") for c in chunks if c.get("idea_id")},
                    )
                except Exception as exc:
                    log.debug("get_slide_neighbors failed: %s", exc)
                    nbr_docs = []
                for d in nbr_docs:
                    md = getattr(d, "metadata", {}) or {}
                    top_neighbors.append({
                        "content":       (getattr(d, "page_content", "") or "")[:1500],
                        "source":        md.get("source", ""),
                        "chapter":       md.get("chapter_idx", md.get("chapter")),
                        "section_idx":   md.get("section_idx"),
                        "section_title": md.get("section_title", ""),
                        "idea_id":       md.get("idea_id"),
                        "idea_label":    md.get("idea_label", ""),
                        "slide_idx":     md.get("slide_idx"),
                        "score":         0.0,
                        "mastery_score": None,
                        "seen":          False,
                        "context_type":  "neighbor",
                    })

        # 7. Final list: hits + KG-augmented + N±1 neighbors of top hit.
        # KG and neighbor entries trail the hits so they don't displace
        # direct retrieval hits in the responder's chunk-formatting top-K.
        all_chunks = chunks + kg_augmented + top_neighbors

        log.info(
            "retriever: %d direct + %d kg-augmented + %d N±1 neighbor(s) of top hit (course=%s ch=%s, seen=%d, review_mode=%s)",
            len(chunks),
            len(kg_augmented),
            len(top_neighbors),
            (course_id or "")[:16],
            chapter_idx,
            sum(1 for c in chunks if c["seen"]),
            review_mode,
        )

        # Detailed log : show each retrieved chunk so the operator can
        # see exactly what the LLM will use for grounding. Truncated
        # to 120 chars per chunk to keep the log readable.
        log.info(
            "🔍 retriever expanded_query: %r",
            expanded_query[:200],
        )
        for i, c in enumerate(all_chunks):
            tag = ""
            if c.get("_via_kg"):
                tag = f"[kg:{c.get('_kg_relation', '?')}]"
            elif c.get("context_type") == "neighbor":
                tag = "[nbr]"
            log.info(
                "🔍   chunk[%d] score=%.3f %s ch=%s sec=%s seen=%s mastery=%.2f | %r",
                i,
                c.get("score", 0.0),
                tag,
                c.get("chapter"),
                c.get("section_idx"),
                "Y" if c.get("seen") else "N",
                c.get("mastery_score") or 0.0,
                (c.get("content") or "")[:120],
            )
        return {
            "retrieved_chunks": all_chunks,
            "timings": {**state.get("timings", {}), "retriever": round(time.time() - start, 3)},
        }

    def _augment_with_kg(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Thin instance method — delegates to the module-level helper so
        ContextAgent (teaching path) can reuse the same logic without
        instantiating a RetrieverAgent."""
        return kg_augment_chunks(chunks, self.rag)

    # Alias retro-compatible — code externe (test_personalization) pourrait
    # l'appeler. Sera supprime apres verification qu'aucun caller direct n'existe.
    def _augment_with_prereqs(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._augment_with_kg(chunks)


# ════════════════════════════════════════════════════════════════════
# Module-level helper — partage entre RetrieverAgent (QA) et ContextAgent
# (teaching). Avant : duplique ou bypasse selon le chemin.
# ════════════════════════════════════════════════════════════════════

def kg_augment_chunks(
    chunks: list[dict[str, Any]],
    rag: Any,
) -> list[dict[str, Any]]:
    """Pull KG neighbors of the top-N retrieved chunks and return them as
    fresh chunk dicts.

    Three relations are expanded :
      - prereqs       : ``depends_on_ids`` of source — what the source
                        requires understanding (cap _KG_PREREQ_CAP)
      - examples      : nodes with ``illustrates_id == source.idea_id``
                        — concrete cases of the source concept (cap
                        _KG_EXAMPLES_CAP)
      - illustrated   : if source is itself an example (has
                        ``illustrates_id``), the concept it illustrates
                        (cap 1 — there's only one)

    Returns empty list if :
      - the IdeaGraph is empty (no ingestion done yet)
      - no top chunk has an idea_id (legacy chunks pre-graph)
      - all neighbors are already in the retrieved pool
    """
    if not chunks:
        return []
    try:
        from pedagogy.knowledge_graph import get_or_build
        graph = get_or_build(rag)
    except Exception as exc:
        log.debug("kg_augment_chunks skipped (%s)", exc)
        return []

    if len(graph) == 0:
        return []

    already_in_pool: set[str] = {c.get("idea_id") for c in chunks if c.get("idea_id")}
    augmented: list[dict[str, Any]] = []

    def _push(node, source_id: str, relation: str) -> bool:
        """Append node as augmented chunk. Returns True if cap reached."""
        if node.idea_id in already_in_pool:
            return False
        already_in_pool.add(node.idea_id)
        # Type-checked chunk dict (rag.metadata.RetrievedChunk)
        chunk: RetrievedChunk = {
            "content":         (node.text or "")[:1500],
            "source":          f"kg_{relation}_of:{source_id}",
            "chapter":         node.chapter_idx,
            "section_idx":     node.section_idx,
            "section_title":   "",
            "idea_id":         node.idea_id,
            "idea_label":      node.label,
            # KG-augmented chunks don't have a RAG cosine/rerank score.
            # Neutral 0.5 ; the ``_via_kg`` flag and ``_kg_relation``
            # let honest consumers (e.g. confidence calibration,
            # narrator that says "as a prerequisite/example...") treat
            # them differently.
            "score":           0.5,
            "mastery_score":   None,
            "seen":            False,
            "_via_kg":         True,
            "_kg_relation":    relation,         # "prereq" | "example" | "illustrated"
            "_augmented_from": source_id,
        }
        augmented.append(chunk)
        return len(augmented) >= _KG_TOTAL_AUGMENT_CAP

    for source_chunk in chunks[:_KG_SOURCE_CHUNKS_TO_EXPAND]:
        source_id = source_chunk.get("idea_id")
        if not source_id:
            continue

        # 1. Prereqs : what the source requires
        for prereq_node in graph.prerequisites_of(source_id, transitive=False)[:_KG_PREREQ_CAP]:
            if _push(prereq_node, source_id, "prereq"):
                return augmented

        # 2. Examples : nodes that illustrate the source concept
        #    (utile pour "donne-moi un exemple de X")
        for example_node in graph.examples_of(source_id)[:_KG_EXAMPLES_CAP]:
            if _push(example_node, source_id, "example"):
                return augmented

        # 3. Illustrated concept : if the source IS an example, surface
        #    the concept it illustrates (utile pour "pourquoi cet exemple ?")
        source_node = graph.get(source_id)
        if source_node and source_node.illustrates_id:
            target = graph.get(source_node.illustrates_id)
            if target is not None:
                if _push(target, source_id, "illustrated"):
                    return augmented

    return augmented
