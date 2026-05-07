"""Tests for the latency profiling framework."""
from __future__ import annotations

import json

import pytest

from agentic.profiling import LatencyCollector, LatencyReport, NodeStats, _percentile


# ════════════════════════════════════════════════════════════════════
# Pure functions
# ════════════════════════════════════════════════════════════════════

class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 50.0) == 0.0

    def test_single_sample_returns_itself(self):
        assert _percentile([1.5], 50.0) == 1.5
        assert _percentile([1.5], 99.0) == 1.5

    def test_known_percentile(self):
        # samples = [1, 2, 3, 4, 5], P50 = 3 (median)
        assert _percentile([1, 2, 3, 4, 5], 50.0) == 3.0
        # P0 = min, P100 = max
        assert _percentile([1, 2, 3, 4, 5], 0.0) == 1.0
        assert _percentile([1, 2, 3, 4, 5], 100.0) == 5.0

    def test_unsorted_input_handled(self):
        # Out-of-order input must not change the result
        assert _percentile([3, 1, 5, 2, 4], 50.0) == 3.0

    def test_linear_interpolation(self):
        # P25 of [1, 2, 3, 4]: rank = 0.25 × 3 = 0.75
        # → between sorted[0]=1 and sorted[1]=2, fraction 0.75
        # → 1 + 0.75 × (2 - 1) = 1.75
        assert _percentile([1, 2, 3, 4], 25.0) == pytest.approx(1.75)


# ════════════════════════════════════════════════════════════════════
# LatencyCollector
# ════════════════════════════════════════════════════════════════════

class TestLatencyCollector:
    def test_empty_collector_produces_empty_report(self):
        c = LatencyCollector()
        report = c.build_report()
        assert report.n_runs == 0
        assert report.nodes == []
        assert report.bottleneck() is None

    def test_single_run_aggregation(self):
        c = LatencyCollector()
        c.record({"intent": 1.0, "responder": 2.0})
        report = c.build_report()
        assert report.n_runs == 1
        assert len(report.nodes) == 2
        node_names = {n.name for n in report.nodes}
        assert node_names == {"intent", "responder"}

    def test_bottleneck_identifies_slowest_node(self):
        c = LatencyCollector()
        c.record({"intent": 0.5, "responder": 5.0, "retriever": 0.1})
        report = c.build_report()
        assert report.bottleneck() == "responder"

    def test_quantiles_known_distribution(self):
        c = LatencyCollector()
        # 100 runs of "responder" with values 0.01, 0.02, ..., 1.00
        for i in range(100):
            c.record({"responder": (i + 1) / 100.0})
        report = c.build_report()
        responder = next(n for n in report.nodes if n.name == "responder")
        # P95 should be around 0.95
        assert responder.p95_s == pytest.approx(0.96, abs=0.02)
        # Mean should be ~0.505
        assert responder.mean_s == pytest.approx(0.505, abs=0.005)
        # Max = 1.0
        assert responder.max_s == 1.0
        # Min = 0.01
        assert responder.min_s == 0.01

    def test_record_handles_none_timings(self):
        c = LatencyCollector()
        c.record(None)
        c.record({})
        report = c.build_report()
        assert report.n_runs == 2
        # Two zero-total runs
        assert report.total_per_run_s == [0.0, 0.0]

    def test_record_skips_non_numeric_values(self):
        c = LatencyCollector()
        c.record({"intent": 1.0, "broken": "not_a_number"})
        report = c.build_report()
        node_names = {n.name for n in report.nodes}
        # Non-numeric values are silently skipped, not recorded as 0
        assert "broken" not in node_names
        assert "intent" in node_names

    def test_record_many(self):
        c = LatencyCollector()
        c.record_many([
            {"intent": 1.0},
            {"intent": 2.0},
            {"intent": 3.0},
        ])
        report = c.build_report()
        intent = next(n for n in report.nodes if n.name == "intent")
        assert intent.n_samples == 3
        assert intent.mean_s == 2.0
        assert intent.median_s == 2.0
        assert intent.min_s == 1.0
        assert intent.max_s == 3.0


# ════════════════════════════════════════════════════════════════════
# LatencyReport rendering
# ════════════════════════════════════════════════════════════════════

class TestLatencyReportRendering:
    def _make_report(self) -> LatencyReport:
        c = LatencyCollector()
        for _ in range(10):
            c.record({"intent": 1.5, "responder": 8.0, "retriever": 0.05})
        return c.build_report()

    def test_to_markdown_contains_node_table(self):
        report = self._make_report()
        md = report.to_markdown()
        assert "Per-node latency" in md
        assert "intent" in md
        assert "responder" in md
        assert "retriever" in md

    def test_to_markdown_orders_by_mean_descending(self):
        report = self._make_report()
        md = report.to_markdown()
        # Bottleneck (responder, 8s mean) must appear before retriever (0.05s)
        assert md.index("responder") < md.index("retriever")

    def test_to_markdown_includes_bottleneck(self):
        report = self._make_report()
        md = report.to_markdown()
        # Bottleneck = responder (highest mean)
        assert "responder" in md
        # End-to-end mean should be ~9.55s (1.5 + 8.0 + 0.05)
        assert "9.55" in md or "9.5" in md

    def test_to_json_is_valid(self):
        report = self._make_report()
        data = json.loads(report.to_json())
        assert data["n_runs"] == 10
        assert data["bottleneck"] == "responder"
        assert "nodes" in data
        assert any(n["name"] == "responder" for n in data["nodes"])

    def test_total_per_run_aggregation(self):
        report = self._make_report()
        # 10 runs of (1.5 + 8.0 + 0.05) = 9.55s each
        assert report.total_mean_s == pytest.approx(9.55)
        # All runs identical → P95 == P99 == mean
        assert report.total_p95_s == pytest.approx(9.55)
        assert report.total_p99_s == pytest.approx(9.55)
