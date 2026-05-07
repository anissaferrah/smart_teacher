"""Latency profiling for the agentic graphs.

# Why this module

Every node in the Q&A and Teaching graphs writes its own duration to
``state["timings"]`` (e.g. ``{"intent": 1.234, "rewriter": 0.512}``).
Combined with OpenTelemetry spans (``agentic/observability.py``), this
gives two complementary signals :

  - ``state["timings"]`` is the embedded record kept in the graph's
    own state — useful for offline analysis and CI baselines, no extra
    infra required.
  - OTel spans are the live observability signal — useful when running
    against a backend (Jaeger / Honeycomb / Datadog).

This module focuses on the first : it collects per-node durations
across N graph executions and renders quantile statistics (P50, P95,
P99, mean, max). The output highlights bottlenecks so the engineering
effort can target the slowest nodes first.

# Why quantiles, not just mean

Mean alone hides tail latency. A node that averages 2s but spikes to
30s on 5% of calls is what students remember. P95 / P99 surface those
tails. Convention follows site-reliability literature (see Treynor et
al. 2017, *The Calculus of Service Availability*).

# Output format

``LatencyReport.to_markdown()`` produces a table ready to paste into a
thesis / soutenance deck. ``to_json()`` is the machine-readable form
for CI ingestion.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class NodeStats:
    """Per-node latency aggregates."""

    name:        str
    n_samples:   int
    mean_s:      float
    median_s:    float
    p95_s:       float
    p99_s:       float
    max_s:       float
    min_s:       float
    total_s:     float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(samples: list[float], p: float) -> float:
    """Linear-interpolation percentile, ``p`` in [0, 100].

    Equivalent to numpy.percentile with ``method='linear'`` (the default).
    Implemented directly to avoid the numpy dependency at runtime — this
    module is meant to be importable from any context.
    """
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    sorted_s = sorted(samples)
    rank = (p / 100.0) * (len(sorted_s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_s[lo]
    frac = rank - lo
    return sorted_s[lo] + frac * (sorted_s[hi] - sorted_s[lo])


@dataclass
class LatencyReport:
    """Aggregated latency stats across N graph executions."""

    n_runs:       int
    timestamp:    float = field(default_factory=time.time)
    nodes:        list[NodeStats] = field(default_factory=list)
    total_per_run_s: list[float] = field(default_factory=list)

    @property
    def total_mean_s(self) -> float:
        if not self.total_per_run_s:
            return 0.0
        return sum(self.total_per_run_s) / len(self.total_per_run_s)

    @property
    def total_p95_s(self) -> float:
        return _percentile(self.total_per_run_s, 95.0)

    @property
    def total_p99_s(self) -> float:
        return _percentile(self.total_per_run_s, 99.0)

    def bottleneck(self) -> str | None:
        """Return the name of the node with the largest mean latency."""
        if not self.nodes:
            return None
        return max(self.nodes, key=lambda n: n.mean_s).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_runs":          self.n_runs,
            "timestamp":       self.timestamp,
            "total_mean_s":    round(self.total_mean_s, 3),
            "total_p95_s":     round(self.total_p95_s, 3),
            "total_p99_s":     round(self.total_p99_s, 3),
            "bottleneck":      self.bottleneck(),
            "nodes":           [n.to_dict() for n in self.nodes],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Pretty-printed report for thesis / soutenance use."""
        bn = self.bottleneck() or "(none)"
        lines = [
            "# Agentic graph latency profile",
            "",
            f"- **Total runs** : {self.n_runs}",
            f"- **End-to-end mean** : {self.total_mean_s:.2f}s",
            f"- **End-to-end P95** : {self.total_p95_s:.2f}s",
            f"- **End-to-end P99** : {self.total_p99_s:.2f}s",
            f"- **Bottleneck** : `{bn}`",
            "",
            "## Per-node latency (seconds)",
            "",
            "| Node | n | mean | median | P95 | P99 | max | total |",
            "|---|---|---|---|---|---|---|---|",
        ]
        # Sort nodes by mean latency descending — bottlenecks at the top
        for stats in sorted(self.nodes, key=lambda n: -n.mean_s):
            lines.append(
                f"| `{stats.name}` | {stats.n_samples} | "
                f"{stats.mean_s:.3f} | {stats.median_s:.3f} | "
                f"{stats.p95_s:.3f} | {stats.p99_s:.3f} | "
                f"{stats.max_s:.3f} | {stats.total_s:.2f} |"
            )
        return "\n".join(lines)


class LatencyCollector:
    """Accumulate per-node timings across graph executions.

    Usage :

        collector = LatencyCollector()
        for query in queries:
            state = run_graph(query)
            collector.record(state.get("timings", {}))
        report = collector.build_report()
        print(report.to_markdown())

    The collector is thread-safe by virtue of being append-only (each
    ``record`` call appends to a list; the build phase is read-only).
    """

    def __init__(self) -> None:
        self._per_node: dict[str, list[float]] = {}
        self._totals: list[float] = []
        self._n_runs = 0

    def record(self, timings: Mapping[str, float] | None) -> None:
        """Add one run's timings to the accumulator."""
        if not timings:
            self._n_runs += 1
            self._totals.append(0.0)
            return
        run_total = 0.0
        for name, value in timings.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            self._per_node.setdefault(name, []).append(v)
            run_total += v
        self._totals.append(run_total)
        self._n_runs += 1

    def record_many(self, runs: Iterable[Mapping[str, float] | None]) -> None:
        for r in runs:
            self.record(r)

    def build_report(self) -> LatencyReport:
        nodes: list[NodeStats] = []
        for name, samples in self._per_node.items():
            if not samples:
                continue
            nodes.append(NodeStats(
                name=name,
                n_samples=len(samples),
                mean_s=round(sum(samples) / len(samples), 6),
                median_s=round(statistics.median(samples), 6),
                p95_s=round(_percentile(samples, 95.0), 6),
                p99_s=round(_percentile(samples, 99.0), 6),
                max_s=round(max(samples), 6),
                min_s=round(min(samples), 6),
                total_s=round(sum(samples), 6),
            ))
        return LatencyReport(
            n_runs=self._n_runs,
            nodes=nodes,
            total_per_run_s=list(self._totals),
        )
