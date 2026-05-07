"""Drive a RAG (or any retriever) through an eval dataset and score it.

# Decoupled from the live RAG class

The runner takes a ``retrieve_fn(query, language, course_id, k) ->
list[chunk_id]`` callable rather than a specific RAG instance. This
keeps it testable (a fake retrieve_fn can be passed in tests) and
allows comparing different retrieval strategies without code changes:

    # Compare hybrid vs dense-only
    rag_hybrid = MultiModalRAG(...)
    rag_dense  = MultiModalRAG(..., disable_bm25=True)
    report_a = run_evaluation(queries, _wrap(rag_hybrid))
    report_b = run_evaluation(queries, _wrap(rag_dense))
    print(compare(report_a, report_b))

# Output

A ``EvalReport`` dataclass with :
  - per-query scores (full breakdown)
  - aggregated metrics (mean across queries)
  - the dataset+config metadata used (timestamp, k, dataset_path)

Both human-readable (``to_markdown()``) and machine-readable
(``to_dict()`` / ``to_json()``) renderings are provided so the report
can flow into a CI pipeline AND a thesis appendix.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

from rag.evaluation.dataset import EvalQuery
from rag.evaluation.metrics import (
    average_precision,
    f1_at_k,
    hit_at_k,
    mean_metric,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

log = logging.getLogger("rag.evaluation.runner")


# RetrieveFn signature: (query, language, course_id, k) -> list[chunk_id]
# A thin layer over a RAG instance — see ``wrap_rag`` below.
RetrieveFn = Callable[[str, str, Optional[str], int], list[str]]


@dataclass
class QueryResult:
    """Per-query scores + retrieved IDs (for offline inspection)."""

    query_id:        str
    query_text:      str
    retrieved:       list[str]           # ordered chunk IDs returned
    expected:        list[str]           # ground truth IDs
    p_at_k:          float
    r_at_k:          float
    f1_at_k:         float
    hit_at_k:        float
    mrr:             float                # reciprocal rank for this query
    ndcg_at_k:       float
    avg_precision:   float
    elapsed_s:       float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    dataset_path:    str
    k:               int
    n_queries:       int
    timestamp:       float = field(default_factory=time.time)
    per_query:       list[QueryResult] = field(default_factory=list)

    # Aggregated metrics (filled by ``finalize``)
    mean_p_at_k:     float = 0.0
    mean_r_at_k:     float = 0.0
    mean_f1_at_k:    float = 0.0
    mean_hit_at_k:   float = 0.0
    mrr:             float = 0.0           # mean reciprocal rank
    mean_ndcg_at_k:  float = 0.0
    mean_avg_precision: float = 0.0        # MAP
    total_elapsed_s: float = 0.0

    def finalize(self) -> "EvalReport":
        if self.per_query:
            self.mean_p_at_k        = mean_metric([q.p_at_k for q in self.per_query])
            self.mean_r_at_k        = mean_metric([q.r_at_k for q in self.per_query])
            self.mean_f1_at_k       = mean_metric([q.f1_at_k for q in self.per_query])
            self.mean_hit_at_k      = mean_metric([q.hit_at_k for q in self.per_query])
            self.mrr                = mean_metric([q.mrr for q in self.per_query])
            self.mean_ndcg_at_k     = mean_metric([q.ndcg_at_k for q in self.per_query])
            self.mean_avg_precision = mean_metric([q.avg_precision for q in self.per_query])
            self.total_elapsed_s    = sum(q.elapsed_s for q in self.per_query)
        self.n_queries = len(self.per_query)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path":     self.dataset_path,
            "k":                self.k,
            "n_queries":        self.n_queries,
            "timestamp":        self.timestamp,
            "metrics": {
                "P@K":          round(self.mean_p_at_k, 4),
                "R@K":          round(self.mean_r_at_k, 4),
                "F1@K":         round(self.mean_f1_at_k, 4),
                "Hit@K":        round(self.mean_hit_at_k, 4),
                "MRR":          round(self.mrr, 4),
                "nDCG@K":       round(self.mean_ndcg_at_k, 4),
                "MAP":          round(self.mean_avg_precision, 4),
            },
            "total_elapsed_s":  round(self.total_elapsed_s, 2),
            "per_query":        [q.to_dict() for q in self.per_query],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Human-readable report. Defense-ready table format."""
        lines = []
        lines.append(f"# RAG evaluation report")
        lines.append("")
        lines.append(f"- **Dataset** : `{self.dataset_path}`")
        lines.append(f"- **# queries** : {self.n_queries}")
        lines.append(f"- **k** : {self.k}")
        lines.append(f"- **Total time** : {self.total_elapsed_s:.2f}s")
        lines.append("")
        lines.append("## Aggregated metrics")
        lines.append("")
        lines.append("| Metric | Value | Reference |")
        lines.append("|---|---|---|")
        lines.append(f"| **Recall@{self.k}** | {self.mean_r_at_k:.4f} | Manning et al. 2008, §8.4 |")
        lines.append(f"| **Precision@{self.k}** | {self.mean_p_at_k:.4f} | Manning et al. 2008, §8.4 |")
        lines.append(f"| **F1@{self.k}** | {self.mean_f1_at_k:.4f} | Manning et al. 2008 |")
        lines.append(f"| **Hit@{self.k}** | {self.mean_hit_at_k:.4f} | hit-rate convention |")
        lines.append(f"| **MRR** | {self.mrr:.4f} | Voorhees TREC-8 1999 |")
        lines.append(f"| **nDCG@{self.k}** | {self.mean_ndcg_at_k:.4f} | Järvelin & Kekäläinen 2002 |")
        lines.append(f"| **MAP** | {self.mean_avg_precision:.4f} | Manning et al. 2008 |")
        lines.append("")
        lines.append("## Per-query breakdown")
        lines.append("")
        lines.append("| ID | R@K | P@K | MRR | nDCG@K | latency | retrieved |")
        lines.append("|---|---|---|---|---|---|---|")
        for q in self.per_query:
            retrieved_short = ", ".join(q.retrieved[:3]) + ("…" if len(q.retrieved) > 3 else "")
            lines.append(
                f"| `{q.query_id}` | {q.r_at_k:.2f} | {q.p_at_k:.2f} | "
                f"{q.mrr:.2f} | {q.ndcg_at_k:.2f} | {q.elapsed_s:.2f}s | {retrieved_short} |"
            )
        return "\n".join(lines)


def evaluate_query(
    query: EvalQuery,
    retrieve_fn: RetrieveFn,
    k: int = 5,
) -> QueryResult:
    """Run a single query and score the returned ranking."""
    start = time.time()
    try:
        retrieved = list(retrieve_fn(query.query, query.language, query.course_id, k * 2))
    except Exception as exc:
        log.warning("retrieve failed for query %s: %s", query.id, exc)
        retrieved = []
    elapsed = time.time() - start

    relevance = query.relevance
    return QueryResult(
        query_id=query.id,
        query_text=query.query,
        retrieved=retrieved,
        expected=list(query.expected_chunks),
        p_at_k=precision_at_k(retrieved, relevance, k),
        r_at_k=recall_at_k(retrieved, relevance, k),
        f1_at_k=f1_at_k(retrieved, relevance, k),
        hit_at_k=hit_at_k(retrieved, relevance, k),
        mrr=reciprocal_rank(retrieved, relevance),
        ndcg_at_k=ndcg_at_k(retrieved, relevance, k),
        avg_precision=average_precision(retrieved, relevance),
        elapsed_s=round(elapsed, 3),
    )


def run_evaluation(
    queries: Sequence[EvalQuery],
    retrieve_fn: RetrieveFn,
    k: int = 5,
    dataset_path: str = "<unknown>",
) -> EvalReport:
    """Run the full eval and return an aggregated report.

    Sequential by design — retrieval calls hit the same vectorstore /
    LLM and serialising prevents thrashing. Parallelism can be added
    later via ``asyncio.gather`` once the RAG client is async.
    """
    report = EvalReport(dataset_path=dataset_path, k=k, n_queries=0)
    for q in queries:
        result = evaluate_query(q, retrieve_fn, k=k)
        report.per_query.append(result)
        log.info(
            "%s | R@%d=%.2f P@%d=%.2f MRR=%.2f nDCG@%d=%.2f (%.2fs)",
            q.id, k, result.r_at_k, k, result.p_at_k, result.mrr, k, result.ndcg_at_k, result.elapsed_s,
        )
    return report.finalize()


# ── Adapter : MultiModalRAG → RetrieveFn ────────────────────────────────


def wrap_rag(
    rag: Any,
    return_concepts: bool = False,
    use_kg_augment: bool = True,
    use_rewriter: bool = False,
) -> RetrieveFn:
    """Wrap a ``MultiModalRAG`` instance into a ``RetrieveFn``.

    The default behaviour evaluates the **agentic RAG pipeline** (RAG
    retrieval + KG-augmented prereqs/examples), which is what the live
    Q&A handler runs. Set ``use_kg_augment=False`` to evaluate the bare
    RAG only — useful for ablation studies.

    Args:
        rag: MultiModalRAG instance.
        return_concepts: si True, traduit chaque idea_id retrouve en sa
            liste de concept_names via le KnowledgeGraph
            (kg.concepts_of_idea). Utile pour evaluer contre un dataset
            ou expected_chunks = concept_names humains (ex:
            tests/data/rag_eval_supervised.jsonl).

            Si False (default) : retourne les idea_id (md5) — utile pour
            evaluer contre un dataset construit a partir des idea_ids
            reels (ex: capture du retrieval reference).

        use_kg_augment: si True (default), applique l'augmentation
            KG (prereqs/examples/illustrated du top-1) APRÈS le retrieval
            classique, comme le fait l'agentic Q&A graph. Mettre à False
            pour mesurer uniquement la couche RAG (BM25+dense+rerank).

        use_rewriter: si True, passe la query par le ``QueryRewriterAgent``
            (step-back reasoning, Zheng et al. 2023) AVANT le retrieval.
            Utile pour mesurer l'apport du rewriter sur les concepts
            sous-représentés. Force le LLM call (bypasse le fast-path
            self-contained heuristic). Default False (bare query).

    Chunks sans idea_id (legacy pre-graph) : id positionnel c{n}, jamais
    matche dans les datasets typiques.
    """
    # ── Lazy-built rewriter (one Brain instance shared across queries) ──
    rewriter_brain = None
    rewriter_agent = None
    if use_rewriter:
        try:
            from ai.llm import Brain
            from agentic.qa.rewriter import QueryRewriterAgent
            rewriter_brain = Brain()
            rewriter_agent = QueryRewriterAgent(rewriter_brain)
        except Exception as exc:
            log.warning("rewriter unavailable: %s — falling back to raw queries", exc)
            rewriter_agent = None

    def _retrieve(query: str, language: str, course_id: Optional[str], k: int) -> list[str]:
        # Optional: rewrite the query first (step-back reasoning).
        # We bypass the rewriter's self-contained fast-path by NOT setting
        # ``rewritten_query`` in the state — that triggers the LLM path.
        retrieval_query = query
        if rewriter_agent is not None:
            try:
                state = {
                    "language": language or "fr",
                    "event_payload": {"text": query},
                    "history": [],
                    "last_slide_content": "",
                    "rewritten_query": "",   # force LLM rewrite (no fast-path)
                }
                out = rewriter_agent(state)
                rewritten = (out.get("rewritten_query") or "").strip()
                if rewritten:
                    retrieval_query = rewritten
                    log.debug("rewrite: %r → %r", query[:60], rewritten[:80])
            except Exception as exc:
                log.debug("rewriter call failed: %s", exc)

        try:
            raw = rag.retrieve_chunks(
                retrieval_query,
                k=k,
                course_id=course_id,
            )
        except Exception as exc:
            log.warning("rag.retrieve_chunks raised: %s", exc)
            return []

        # ── Build the chunk-dict format used by kg_augment_chunks ────────
        chunks_dict: list[dict] = []
        for i, item in enumerate(raw or []):
            doc = item[0] if isinstance(item, tuple) else item
            score = item[1] if (isinstance(item, tuple) and len(item) > 1) else 0.5
            meta = getattr(doc, "metadata", {}) or {}
            cid = meta.get("idea_id") or f"c{i}"
            chunks_dict.append({
                "content":   getattr(doc, "page_content", "") or "",
                "score":     float(score) if score is not None else 0.5,
                "idea_id":   cid,
                "idea_label": meta.get("idea_label", ""),
                "chapter":   meta.get("chapter_idx"),
                "section_title": meta.get("section_title", ""),
            })

        # ── KG augmentation (matches agentic/qa/retriever.py) ────────────
        if use_kg_augment and chunks_dict:
            try:
                from agentic.qa.retriever import kg_augment_chunks
                augmented = kg_augment_chunks(chunks_dict, rag)
                chunks_dict.extend(augmented)
            except Exception as exc:
                log.debug("kg augmentation skipped during eval: %s", exc)

        # ── Optional concept-name mapping ────────────────────────────────
        kg = None
        if return_concepts:
            try:
                from pedagogy.knowledge_graph import get_or_build
                kg = get_or_build(rag)
            except Exception as exc:
                log.warning("KG unavailable for concept mapping: %s", exc)
                kg = None

        ids: list[str] = []
        seen: set[str] = set()
        for c in chunks_dict:
            cid = c.get("idea_id") or ""
            if not cid:
                continue
            if return_concepts and kg is not None:
                concepts = kg.concepts_of_idea(str(cid))
                if not concepts:
                    if cid not in seen:
                        seen.add(cid)
                        ids.append(str(cid))
                else:
                    for cinfo in concepts:
                        if cinfo.name not in seen:
                            seen.add(cinfo.name)
                            ids.append(cinfo.name)
            else:
                if cid not in seen:
                    seen.add(cid)
                    ids.append(str(cid))
        return ids
    return _retrieve
