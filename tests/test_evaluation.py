"""Tests for the RAG evaluation framework."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from rag.evaluation import (
    EvalQuery,
    EvalReport,
    average_precision,
    dcg_at_k,
    evaluate_query,
    f1_at_k,
    hit_at_k,
    load_jsonl,
    ndcg_at_k,
    parse_query,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_evaluation,
)


# ════════════════════════════════════════════════════════════════════
# Pure metric functions
# ════════════════════════════════════════════════════════════════════

class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_partial_precision(self):
        # Top-3 = ["a", "b", "x"], 2 hits → 2/3
        assert precision_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(2 / 3)

    def test_k_larger_than_retrieved(self):
        # P@5 on a 3-item list with 2 hits → still divides by k=5
        assert precision_at_k(["a", "b", "x"], {"a", "b"}, k=5) == pytest.approx(2 / 5)

    def test_empty_inputs(self):
        assert precision_at_k([], {"a"}, k=3) == 0.0
        assert precision_at_k(["a"], set(), k=3) == 0.0
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_partial_recall(self):
        # Top-2 = ["a", "x"], 1 hit out of 2 relevant → 1/2
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5

    def test_k_smaller_than_retrieved_does_not_help(self):
        # Even though "b" is in retrieved at index 2, k=2 cuts it off
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=3) == 1.0

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["a"], set(), k=3) == 0.0


class TestF1AtK:
    def test_f1_zero_when_both_zero(self):
        assert f1_at_k(["x"], {"a"}, k=1) == 0.0

    def test_f1_perfect_when_both_perfect(self):
        assert f1_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_f1_harmonic_mean(self):
        # P=1.0 (1/1 hit), R=0.5 (1/2 relevant) → F1 = 2·1·0.5 / 1.5 = 0.667
        # k=1, retrieved=["a"], relevant={"a", "b"}
        f1 = f1_at_k(["a"], {"a", "b"}, k=1)
        expected = 2 * 1.0 * 0.5 / (1.0 + 0.5)
        assert f1 == pytest.approx(expected)


class TestHitAtK:
    def test_hit_returns_one_on_any_match(self):
        assert hit_at_k(["x", "y", "a"], {"a"}, k=3) == 1.0

    def test_hit_returns_zero_on_no_match(self):
        assert hit_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_hit_respects_k(self):
        # "a" is at index 3, k=2 cuts it off
        assert hit_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0
        assert hit_at_k(["x", "y", "a"], {"a"}, k=3) == 1.0


class TestReciprocalRank:
    def test_first_item_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_item_relevant(self):
        assert reciprocal_rank(["x", "a", "c"], {"a"}) == 0.5

    def test_no_relevant_returns_zero(self):
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_only_first_relevant_counts(self):
        # MRR uses 1/rank of FIRST relevant — second hit ignored
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5


class TestDCGandNDCG:
    def test_dcg_known_formula(self):
        # retrieved = ["a", "b"], grades a=3, b=2 → DCG@2 = 3/log2(2) + 2/log2(3)
        dcg = dcg_at_k(["a", "b"], {"a": 3, "b": 2}, k=2)
        expected = 3.0 / math.log2(2) + 2.0 / math.log2(3)
        assert dcg == pytest.approx(expected)

    def test_ndcg_perfect_when_optimal_order(self):
        # Best-graded item first → nDCG@K = 1.0
        assert ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, k=2) == 1.0

    def test_ndcg_less_than_one_when_swapped(self):
        # Worse-graded item first → nDCG@K < 1.0
        ndcg = ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, k=2)
        assert 0.0 < ndcg < 1.0

    def test_ndcg_zero_when_no_relevant(self):
        assert ndcg_at_k(["x", "y"], {"a": 3}, k=2) == 0.0


class TestAveragePrecision:
    def test_ap_single_relevant_at_top(self):
        # AP = (1/1) * (1/1) = 1.0
        assert average_precision(["a", "x", "y"], {"a"}) == 1.0

    def test_ap_single_relevant_at_rank_2(self):
        # AP = (1/1) * (1/2) = 0.5
        assert average_precision(["x", "a"], {"a"}) == 0.5

    def test_ap_two_relevant(self):
        # retrieved=[a, x, b], relevant={a, b}
        # P@1=1/1, P@3=2/3 (when b is found at rank 3)
        # AP = (1/2) * (1/1 + 2/3) = (1/2) * 5/3 = 5/6
        ap = average_precision(["a", "x", "b"], {"a", "b"})
        assert ap == pytest.approx(5 / 6)


# ════════════════════════════════════════════════════════════════════
# Dataset loading
# ════════════════════════════════════════════════════════════════════

class TestDatasetLoading:
    def test_parse_minimal_query(self):
        rec = {"id": "q1", "query": "test", "expected_chunks": ["c1"]}
        q = parse_query(rec)
        assert q.id == "q1"
        assert q.query == "test"
        assert q.expected_chunks == ["c1"]
        assert q.language == "fr"  # default
        assert q.expected_grades == {}

    def test_parse_with_grades(self):
        rec = {
            "id": "q1", "query": "test",
            "expected_chunks": ["c1", "c2"],
            "expected_grades": {"c1": 3, "c2": 2},
        }
        q = parse_query(rec)
        assert q.expected_grades == {"c1": 3, "c2": 2}

    def test_relevance_property_falls_back_to_binary(self):
        q = EvalQuery(id="q1", query="t", expected_chunks=["c1", "c2"])
        assert q.relevance == {"c1": 1, "c2": 1}

    def test_relevance_property_uses_grades_when_present(self):
        q = EvalQuery(
            id="q1", query="t",
            expected_chunks=["c1"],
            expected_grades={"c1": 3, "c2": 2},
        )
        # When grades are set, they take precedence
        assert q.relevance == {"c1": 3, "c2": 2}

    def test_parse_rejects_missing_id(self):
        with pytest.raises(ValueError):
            parse_query({"query": "test", "expected_chunks": []})

    def test_parse_rejects_missing_query(self):
        with pytest.raises(ValueError):
            parse_query({"id": "q1", "expected_chunks": []})

    def test_load_jsonl_bootstrap_dataset(self):
        path = Path(__file__).parent / "data" / "rag_eval.jsonl"
        queries = load_jsonl(path)
        # Extended dataset: 50 queries spanning 10 sub-domains of the
        # "informatique" course (algorithms, data structures, OOP, FP,
        # databases, OS, networks, ML/data mining, software engineering,
        # cybersecurity) + cross-domain synthesis + edge cases.
        assert len(queries) == 50
        ids = [q.id for q in queries]
        assert "q001" in ids
        assert "q050" in ids

    def test_dataset_id_uniqueness(self):
        """All query IDs must be unique — duplicates would skew metrics."""
        path = Path(__file__).parent / "data" / "rag_eval.jsonl"
        queries = load_jsonl(path)
        ids = [q.id for q in queries]
        assert len(ids) == len(set(ids)), "duplicate query IDs found"

    def test_dataset_language_balance(self):
        """Both FR and EN queries are present so the eval covers both
        spoken languages of the tutor."""
        path = Path(__file__).parent / "data" / "rag_eval.jsonl"
        queries = load_jsonl(path)
        langs = {q.language for q in queries}
        assert "fr" in langs
        assert "en" in langs
        # No unexpected language codes
        assert langs <= {"fr", "en"}

    def test_dataset_chapter_coverage(self):
        """The dataset spans all 10 sub-domains of the informatique
        course (with explicit cross-chapter queries having
        chapter=None)."""
        path = Path(__file__).parent / "data" / "rag_eval.jsonl"
        queries = load_jsonl(path)
        chapters = {q.chapter for q in queries if q.chapter is not None}
        # Ten sub-domains expected
        assert chapters == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    def test_dataset_course_id_consistency(self):
        """All non-empty course_ids point at the informatique domain."""
        path = Path(__file__).parent / "data" / "rag_eval.jsonl"
        queries = load_jsonl(path)
        course_ids = {q.course_id for q in queries if q.course_id}
        assert course_ids == {"informatique"}

    def test_dataset_includes_edge_cases(self):
        """The dataset includes queries with empty expected_chunks
        (out-of-scope / ambiguous) — used to verify the system's
        ability to return low-confidence / no-result responses."""
        path = Path(__file__).parent / "data" / "rag_eval.jsonl"
        queries = load_jsonl(path)
        empty_expected = [q for q in queries if not q.expected_chunks]
        # At least one out-of-scope query (e.g. quantum ML)
        assert len(empty_expected) >= 1

    def test_dataset_graded_relevance_coverage(self):
        """A meaningful share of queries carry graded relevance for nDCG.
        Without grades, nDCG degenerates to a binary metric."""
        path = Path(__file__).parent / "data" / "rag_eval.jsonl"
        queries = load_jsonl(path)
        graded = [q for q in queries if q.expected_grades]
        # At least half the queries should carry grades
        assert len(graded) >= len(queries) // 2

    def test_load_jsonl_dedups_ids(self, tmp_path):
        path = tmp_path / "dups.jsonl"
        path.write_text(
            '{"id": "q1", "query": "a", "expected_chunks": ["c1"]}\n'
            '{"id": "q1", "query": "b", "expected_chunks": ["c2"]}\n'
            '{"id": "q2", "query": "c", "expected_chunks": ["c3"]}\n',
            encoding="utf-8",
        )
        queries = load_jsonl(path)
        assert len(queries) == 2  # q1 deduplicated
        assert {q.id for q in queries} == {"q1", "q2"}


# ════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════

class TestRunner:
    def test_evaluate_query_perfect_retrieval(self):
        q = EvalQuery(id="q1", query="t", expected_chunks=["a", "b"])
        # Fake retriever that returns expected items in order
        retrieve_fn = lambda query, lang, course, k: ["a", "b", "x", "y", "z"]
        result = evaluate_query(q, retrieve_fn, k=5)
        assert result.r_at_k == 1.0
        assert result.mrr == 1.0
        assert result.hit_at_k == 1.0

    def test_evaluate_query_no_results(self):
        q = EvalQuery(id="q1", query="t", expected_chunks=["a"])
        retrieve_fn = lambda *args, **kwargs: []
        result = evaluate_query(q, retrieve_fn, k=5)
        assert result.r_at_k == 0.0
        assert result.mrr == 0.0
        assert result.hit_at_k == 0.0

    def test_evaluate_query_handles_retriever_exception(self):
        q = EvalQuery(id="q1", query="t", expected_chunks=["a"])

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        result = evaluate_query(q, boom, k=5)
        # Should NOT raise; degraded scores
        assert result.r_at_k == 0.0
        assert result.retrieved == []

    def test_run_evaluation_aggregates(self):
        queries = [
            EvalQuery(id="q1", query="t1", expected_chunks=["a"]),
            EvalQuery(id="q2", query="t2", expected_chunks=["b"]),
        ]
        # Retriever returns the right thing for q1, wrong for q2
        def retrieve_fn(query, lang, course, k):
            if "1" in query:
                return ["a", "x", "y"]
            return ["x", "y", "z"]
        report = run_evaluation(queries, retrieve_fn, k=3, dataset_path="test.jsonl")
        assert report.n_queries == 2
        # Mean R@K: (1.0 + 0.0) / 2 = 0.5
        assert report.mean_r_at_k == 0.5
        # MRR: (1.0 + 0.0) / 2 = 0.5
        assert report.mrr == 0.5

    def test_report_to_markdown_contains_metrics(self):
        queries = [EvalQuery(id="q1", query="t", expected_chunks=["a"])]
        retrieve_fn = lambda *args, **kwargs: ["a", "b"]
        report = run_evaluation(queries, retrieve_fn, k=2)
        md = report.to_markdown()
        assert "Recall@2" in md
        assert "MRR" in md
        assert "nDCG@2" in md
        assert "q1" in md

    def test_report_to_json_is_valid(self):
        import json
        queries = [EvalQuery(id="q1", query="t", expected_chunks=["a"])]
        retrieve_fn = lambda *args, **kwargs: ["a", "b"]
        report = run_evaluation(queries, retrieve_fn, k=2)
        data = json.loads(report.to_json())
        assert data["n_queries"] == 1
        assert "metrics" in data
        assert data["metrics"]["MRR"] == 1.0
