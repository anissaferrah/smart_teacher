"""RAG evaluation framework — Recall@K, Precision@K, MRR, nDCG@K, MAP.

Quick start :

    from rag.evaluation import load_jsonl, run_evaluation, wrap_rag

    queries = load_jsonl("tests/data/rag_eval.jsonl")
    retrieve_fn = wrap_rag(rag_instance)        # MultiModalRAG
    report = run_evaluation(queries, retrieve_fn, k=5,
                            dataset_path="tests/data/rag_eval.jsonl")
    print(report.to_markdown())
    # or  print(report.to_json())

Pure metric functions are also re-exported for ad-hoc use :

    from rag.evaluation import recall_at_k, ndcg_at_k
    recall_at_k(["a", "b", "c"], {"b"}, k=2)        # → 1.0
"""
from rag.evaluation.dataset import EvalQuery, load_jsonl, parse_query, stream_jsonl
from rag.evaluation.metrics import (
    average_precision,
    dcg_at_k,
    f1_at_k,
    hit_at_k,
    mean_metric,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.evaluation.runner import (
    EvalReport,
    QueryResult,
    RetrieveFn,
    evaluate_query,
    run_evaluation,
    wrap_rag,
)

__all__ = [
    # dataset
    "EvalQuery", "load_jsonl", "parse_query", "stream_jsonl",
    # metrics
    "average_precision", "dcg_at_k", "f1_at_k", "hit_at_k",
    "mean_metric", "ndcg_at_k", "precision_at_k", "recall_at_k",
    "reciprocal_rank",
    # runner
    "EvalReport", "QueryResult", "RetrieveFn",
    "evaluate_query", "run_evaluation", "wrap_rag",
]
