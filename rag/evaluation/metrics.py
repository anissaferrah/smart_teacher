"""IR evaluation metrics — pure functions, citation-traceable.

All functions take :
  - ``retrieved`` : ordered list of result IDs (rank 1 = best, rank len = worst)
  - ``relevant``  : either a set of relevant IDs (binary judgment) or a dict
                    mapping ID → graded relevance ∈ {0, 1, 2, 3, …} (for nDCG)

Conventions :
  - Ranks are 1-indexed in formulas, 0-indexed in Python loops. The
    docstrings use the formula form (1-indexed) for clarity.
  - All metrics return floats in [0, 1].
  - Empty inputs are handled defensively (return 0.0, never raise).

# References

  - Manning, Raghavan, Schütze (2008). *Introduction to Information
    Retrieval.* Cambridge University Press. Chapter 8 covers all the
    metrics here.
  - Voorhees, E. M. (1999). *The TREC-8 Question Answering Track
    Report.* TREC. Original formulation of MRR.
  - Järvelin, K., & Kekäläinen, J. (2002). *Cumulated gain-based
    evaluation of IR techniques.* ACM TOIS 20(4). nDCG.
  - Sanderson, M. (2010). *Test Collection Based Evaluation of
    Information Retrieval Systems.* Foundations and Trends in IR.
    Recommended for evaluation methodology.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def _to_relevance_set(relevant: Mapping[str, int] | set[str] | Sequence[str]) -> set[str]:
    """Coerce ``relevant`` into a set of IDs with non-zero relevance."""
    if isinstance(relevant, set):
        return relevant
    if isinstance(relevant, Mapping):
        return {str(k) for k, v in relevant.items() if v and v > 0}
    return {str(x) for x in relevant}


def _to_graded_relevance(relevant: Mapping[str, int] | set[str] | Sequence[str]) -> dict[str, int]:
    """Coerce ``relevant`` into a dict ``{id: relevance_grade}`` (binary if not given)."""
    if isinstance(relevant, Mapping):
        return {str(k): int(v) for k, v in relevant.items()}
    if isinstance(relevant, set):
        return {str(k): 1 for k in relevant}
    return {str(k): 1 for k in relevant}


def precision_at_k(retrieved: Sequence[str], relevant: Mapping[str, int] | set[str] | Sequence[str], k: int) -> float:
    """P@K = |relevant ∩ retrieved[:k]| / k.

    Fraction of the top-K results that are relevant. Manning et al.
    Section 8.4. Returns 0.0 when ``k <= 0`` or ``retrieved`` is empty.
    """
    if k <= 0 or not retrieved:
        return 0.0
    rel_set = _to_relevance_set(relevant)
    if not rel_set:
        return 0.0
    top_k = list(retrieved[:k])
    hits = sum(1 for r in top_k if r in rel_set)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Mapping[str, int] | set[str] | Sequence[str], k: int) -> float:
    """R@K = |relevant ∩ retrieved[:k]| / |relevant|.

    Fraction of the relevant items that the system surfaced in its top-K.
    Manning et al. Section 8.4. Returns 0.0 when there are no relevant
    items (avoiding 0/0).
    """
    if k <= 0 or not retrieved:
        return 0.0
    rel_set = _to_relevance_set(relevant)
    if not rel_set:
        return 0.0
    top_k = list(retrieved[:k])
    hits = sum(1 for r in top_k if r in rel_set)
    return hits / len(rel_set)


def f1_at_k(retrieved: Sequence[str], relevant: Mapping[str, int] | set[str] | Sequence[str], k: int) -> float:
    """F1@K — harmonic mean of P@K and R@K.

    F1 = 2·P·R / (P+R), with the conventional 0/0 handling: returns 0
    when both P and R are zero.
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_at_k(retrieved: Sequence[str], relevant: Mapping[str, int] | set[str] | Sequence[str], k: int) -> float:
    """Hit@K — 1.0 if any relevant item appears in the top-K, else 0.0.

    Useful as a coarse "did we surface anything useful?" signal.
    """
    if k <= 0 or not retrieved:
        return 0.0
    rel_set = _to_relevance_set(relevant)
    if not rel_set:
        return 0.0
    return 1.0 if any(r in rel_set for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Mapping[str, int] | set[str] | Sequence[str]) -> float:
    """Reciprocal rank — 1/rank of the first relevant item, or 0.0 if none.

    Voorhees (1999), TREC-8 QA track. Mean Reciprocal Rank (MRR) is the
    average of this metric over a set of queries (see ``mean_metric``).
    """
    rel_set = _to_relevance_set(relevant)
    if not rel_set or not retrieved:
        return 0.0
    for i, r in enumerate(retrieved, start=1):
        if r in rel_set:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: Sequence[str], relevant: Mapping[str, int] | set[str] | Sequence[str], k: int) -> float:
    """DCG@K — Discounted Cumulative Gain, log₂ discount.

    DCG@K = Σᵢ rel(i) / log₂(i+1) for i in 1..K

    The log₂(i+1) discount is the standard formulation from Järvelin &
    Kekäläinen (2002). Graded relevance is read directly from the
    ``relevant`` mapping; absent IDs contribute 0.
    """
    if k <= 0 or not retrieved:
        return 0.0
    grades = _to_graded_relevance(relevant)
    if not grades:
        return 0.0
    total = 0.0
    for i, rid in enumerate(retrieved[:k], start=1):
        rel = grades.get(rid, 0)
        if rel:
            total += rel / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], relevant: Mapping[str, int] | set[str] | Sequence[str], k: int) -> float:
    """nDCG@K — DCG@K normalized by the ideal DCG@K.

    nDCG@K = DCG@K / IDCG@K, in [0, 1]. The ideal DCG is the DCG of the
    optimal ranking: relevant items sorted by descending grade. When
    there are no relevant items, returns 0.0 (instead of NaN).
    """
    if k <= 0 or not retrieved:
        return 0.0
    grades = _to_graded_relevance(relevant)
    if not grades:
        return 0.0
    actual = dcg_at_k(retrieved, relevant, k)
    ideal_order = sorted(grades.keys(), key=lambda x: -grades[x])
    ideal = dcg_at_k(ideal_order, relevant, k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def average_precision(retrieved: Sequence[str], relevant: Mapping[str, int] | set[str] | Sequence[str]) -> float:
    """AP — Average Precision over the full retrieved list.

    AP = (1/|relevant|) Σ P@i × 1[item_i is relevant]

    Manning et al. Section 8.4. Mean Average Precision (MAP) is the
    arithmetic mean of AP across queries.
    """
    rel_set = _to_relevance_set(relevant)
    if not rel_set or not retrieved:
        return 0.0
    hits = 0
    score_sum = 0.0
    for i, r in enumerate(retrieved, start=1):
        if r in rel_set:
            hits += 1
            score_sum += hits / i
    return score_sum / len(rel_set)


def mean_metric(per_query_scores: Sequence[float]) -> float:
    """Arithmetic mean of per-query metric values. Returns 0.0 on empty."""
    if not per_query_scores:
        return 0.0
    return sum(per_query_scores) / len(per_query_scores)
