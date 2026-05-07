"""Profile the agentic Q&A graph latency over an eval dataset.

Usage :
    python -m scripts.profile_agentic                              # default dataset
    python -m scripts.profile_agentic --dataset path/to.jsonl
    python -m scripts.profile_agentic --output-md profile.md
    python -m scripts.profile_agentic --output-json profile.json

The script :
  1. Loads queries from the eval dataset (default: tests/data/rag_eval.jsonl).
  2. Builds the live agentic Q&A graph (intent → rewriter → retriever → responder).
  3. Runs each query through the graph, harvesting state["timings"].
  4. Aggregates per-node latency stats (mean / median / P95 / P99 / max).
  5. Renders the report (markdown + optional JSON).

The "bottleneck" reported is the node with the largest mean latency.
On Ollama CPU you should expect the LLM nodes (intent, rewriter,
responder) to dominate; cheap nodes (retriever, retrieval_decision)
should sit in the milliseconds range.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agentic.profiling import LatencyCollector
from rag.evaluation import load_jsonl


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Profile the agentic Q&A graph latency.")
    p.add_argument(
        "--dataset",
        default=str(_ROOT / "tests" / "data" / "rag_eval.jsonl"),
        help="JSONL eval dataset (default: tests/data/rag_eval.jsonl)",
    )
    p.add_argument(
        "--output-md",
        default=None,
        help="Write the markdown report to this path",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Write the JSON report to this path",
    )
    p.add_argument(
        "--db-dir",
        default="data/rag_cache",
        help="RAG database dir (default: data/rag_cache)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N queries (smoke / fast iterations)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return p


async def _run_one(graph, query_text: str, lang: str) -> dict:
    """Drive one query through the graph and return the final state."""
    state = {
        "event_type":     "student_speech",
        "event_payload":  {"text": query_text},
        "language":       lang,
        "history":        [],
        "last_slide_content": "",
        "timings":        {},
    }
    final_state = await graph.ainvoke(state)
    return final_state


async def _amain(args) -> int:
    log = logging.getLogger("scripts.profile_agentic")

    # 1) Load dataset
    try:
        queries = load_jsonl(args.dataset)
    except FileNotFoundError as exc:
        log.error("dataset not found: %s", exc)
        return 1
    if not queries:
        log.error("no queries loaded from %s", args.dataset)
        return 1
    if args.limit:
        queries = queries[: args.limit]
    log.info("profiling %d queries", len(queries))

    # 2) Build the agentic graph
    try:
        from ai.llm import Brain
        from rag.multimodal_rag import MultiModalRAG
        from agentic.qa.graph import build_qa_graph
    except Exception as exc:                                              # noqa: BLE001
        log.error("import failed: %s", exc)
        return 1

    rag = MultiModalRAG(db_dir=args.db_dir)
    brain = Brain()
    graph = build_qa_graph(brain, rag=rag)

    # 3) Run + collect
    collector = LatencyCollector()
    for q in queries:
        try:
            final_state = await _run_one(graph, q.query, q.language)
            collector.record(final_state.get("timings", {}))
            log.info(
                "%s | total=%.2fs | nodes=%s",
                q.id,
                sum(final_state.get("timings", {}).values()),
                ",".join(final_state.get("timings", {})),
            )
        except Exception as exc:                                          # noqa: BLE001
            log.warning("query %s failed: %s", q.id, exc)
            collector.record({})  # still count it as a run

    # 4) Render
    report = collector.build_report()
    md = report.to_markdown()
    print(md)

    if args.output_md:
        Path(args.output_md).write_text(md, encoding="utf-8")
        log.info("markdown report written → %s", args.output_md)
    if args.output_json:
        Path(args.output_json).write_text(report.to_json(), encoding="utf-8")
        log.info("json report written → %s", args.output_json)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
