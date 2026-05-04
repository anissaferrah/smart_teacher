"""Run the RAG evaluation framework against the live MultiModalRAG.

Usage :
    python -m scripts.run_rag_eval                         # default dataset, k=5
    python -m scripts.run_rag_eval --k 10                  # custom k
    python -m scripts.run_rag_eval --dataset path/to.jsonl
    python -m scripts.run_rag_eval --output-md report.md   # save markdown
    python -m scripts.run_rag_eval --output-json report.json

Exit code is 0 on a successful run (regardless of metric values), 1 on
fatal errors (RAG not ready, dataset missing, etc.).

The script loads the live RAG instance via ``rag.multimodal_rag.MultiModalRAG``
which reads its index from disk. If your RAG instance is held by an
already-running process (FastAPI server), prefer running this against a
copy of that index — concurrent reads are safe but writes during eval
would invalidate the run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python -m scripts.run_rag_eval` from project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from rag.evaluation import load_jsonl, run_evaluation, wrap_rag


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the live RAG against a JSONL dataset.")
    p.add_argument(
        "--dataset",
        default=str(_ROOT / "tests" / "data" / "rag_eval.jsonl"),
        help="Path to the JSONL eval dataset (default: tests/data/rag_eval.jsonl)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K cutoff for the metrics (default: 5)",
    )
    p.add_argument(
        "--output-md",
        default=None,
        help="Write the markdown report to this path (defaults to stdout only)",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Write the JSON report to this path",
    )
    p.add_argument(
        "--db-dir",
        default="data/rag_cache",
        help="RAG database dir to load from (default: data/rag_cache)",
    )
    p.add_argument(
        "--by-concept",
        action="store_true",
        help="Translate retrieved idea_ids → concept_names via KnowledgeGraph. "
             "Use with datasets where expected_chunks are concept names "
             "(ex: tests/data/rag_eval_supervised.jsonl).",
    )
    p.add_argument(
        "--no-kg",
        action="store_true",
        help="Disable KG augmentation (prereqs/examples) for ablation. "
             "Default: KG augmentation is ON (matches the live agentic Q&A "
             "pipeline). Use this flag to measure the bare RAG layer only.",
    )
    p.add_argument(
        "--with-rewriter",
        action="store_true",
        help="Pipe each query through the QueryRewriterAgent (step-back "
             "reasoning) before retrieval. Forces the LLM rewrite path "
             "(bypasses the self-contained fast-path). Useful to measure "
             "the rewriter's contribution on under-represented concepts.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("scripts.run_rag_eval")

    # 1) Load dataset
    try:
        queries = load_jsonl(args.dataset)
    except FileNotFoundError as exc:
        log.error("dataset not found: %s", exc)
        return 1
    if not queries:
        log.error("no queries loaded from %s", args.dataset)
        return 1
    log.info("loaded %d queries from %s", len(queries), args.dataset)

    # 2) Build RAG and check readiness
    try:
        from rag.multimodal_rag import MultiModalRAG
    except Exception as exc:                                              # noqa: BLE001
        log.error("could not import MultiModalRAG: %s", exc)
        return 1

    rag = MultiModalRAG(db_dir=args.db_dir)
    if not getattr(rag, "is_ready", False):
        log.warning(
            "RAG instance is not 'ready' (no index loaded at %s). "
            "Eval will run but all queries will return empty results.",
            args.db_dir,
        )

    # 3) Concept-mode prep : register rag in deps + attach concepts to KG
    if args.by_concept:
        import deps
        deps.register_services(rag=rag)
        from pedagogy.knowledge_graph.concepts_loader import ensure_concepts_loaded
        course_ids = {q.course_id for q in queries if q.course_id}
        for cid in course_ids:
            log.info("attaching concepts for course=%s …", cid[:16])
            concepts = ensure_concepts_loaded(rag, cid, max_concepts=50, enrich=False)
            log.info("  → %d concepts attached", len(concepts))

    # 4) Run evaluation — by default we evaluate the FULL agentic pipeline
    # (RAG + KG augmentation), which is what the live /ask handler runs.
    # Flip --no-kg for an ablation that measures bare RAG only.
    # Flip --with-rewriter to add the step-back QueryRewriterAgent.
    use_kg = not args.no_kg
    retrieve_fn = wrap_rag(
        rag,
        return_concepts=args.by_concept,
        use_kg_augment=use_kg,
        use_rewriter=args.with_rewriter,
    )
    pipeline_parts = []
    if args.with_rewriter:
        pipeline_parts.append("Rewriter (step-back)")
    pipeline_parts.append("RAG + KG augmentation (agentic)" if use_kg else "bare RAG (no KG)")
    if args.by_concept:
        pipeline_parts.append("by-concept")
    log.info("pipeline = %s", " → ".join(pipeline_parts))
    report = run_evaluation(queries, retrieve_fn, k=args.k, dataset_path=args.dataset)

    # 4) Render
    md = report.to_markdown()
    print(md)

    if args.output_md:
        Path(args.output_md).write_text(md, encoding="utf-8")
        log.info("markdown report written → %s", args.output_md)
    if args.output_json:
        Path(args.output_json).write_text(report.to_json(), encoding="utf-8")
        log.info("json report written → %s", args.output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
