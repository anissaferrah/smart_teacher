"""Compose breaker + fallback into a drop-in resilient node.

Usage in graph builder:

    from agentic.resilience import build_resilient_node

    graph.add_node("planner", build_resilient_node(planner, "planner"))

The wrapped callable preserves the original signature ``(state) -> dict`` and
runs the underlying agent under two layers:

  1. Circuit breaker — short-circuits to fallback if the node has been
     failing recently. No call is even attempted.
  2. Exception capture — any uncaught exception is replaced by the node's
     deterministic fallback so the graph state stays valid.

# Why no timeouts

Earlier iterations enforced a per-node ``asyncio.wait_for`` plus a pipeline-
global deadline. On Ollama CPU (Mistral 7B, 10-50 tok/s), the LLM calls
genuinely exceed any sane SLA budget — and the underlying calls were
synchronous, blocking the event loop and preventing ``wait_for`` from
firing reliably. Cancelling a sync HTTP call to Ollama isn't actually
possible with ``asyncio.wait_for`` (the thread keeps running). So we
removed the timeout enforcement: tutoring users prefer "slow but correct"
to "fast but degraded".

When the LLM client becomes async (httpx-based or GPU-fast), reintroduce
the timeout layer here.

The wrapper is async; sync nodes are auto-adapted via ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Awaitable, Callable

from agentic.observability import attach_node_attrs, node_span
from agentic.resilience.circuit_breaker import BreakerRegistry, BreakerState
from agentic.resilience.fallbacks import fallback_for

log = logging.getLogger("agentic.resilience.wrap")


# Optional KPI hook — best-effort import so the module remains usable in
# environments without observability wiring.
try:
    from observability.kpi_logger import KPITracker
    _kpi_available = True
except Exception:                                                       # noqa: BLE001
    _kpi_available = False


def _emit_event(node_name: str, kind: str, latency_s: float | None = None) -> None:
    """Log a structured event; KPI logger forwards via stats endpoints."""
    log.info("resilience event: node=%s kind=%s latency=%s",
             node_name, kind, f"{latency_s:.3f}" if latency_s else "n/a")
    if _kpi_available:
        try:
            tracker = KPITracker.get()
            sink = getattr(tracker, "_resilience_events", None)
            if sink is None:
                tracker._resilience_events = []                          # noqa: SLF001
                sink = tracker._resilience_events                        # noqa: SLF001
            sink.append({
                "ts":      time.time(),
                "node":    node_name,
                "kind":    kind,
                "latency": round(latency_s, 3) if latency_s is not None else None,
            })
            # Cap retention to last 500 events
            if len(sink) > 500:
                del sink[: len(sink) - 500]
        except Exception:
            pass


def build_resilient_node(node: Callable[..., Any], name: str) -> Callable[[dict], Awaitable[dict]]:
    """Wrap a graph node with breaker + exception-to-fallback.

    The original ``node(state)`` may be sync or async, returning a partial
    state-update dict. The returned wrapper is always async.
    """
    breaker = BreakerRegistry.get(name)

    # Detect once at build time: class instance with `async def __call__` is
    # the common LangGraph node shape. `iscoroutinefunction(node)` returns
    # False on instances even when `__call__` is async — we must inspect
    # `node.__call__` explicitly.
    _is_async = (
        inspect.iscoroutinefunction(node)
        or inspect.iscoroutinefunction(getattr(node, "__call__", None))
    )

    async def _invoke(state: dict) -> Any:
        if _is_async:
            return await node(state)
        # Sync nodes: run in default executor so we don't block the loop
        return await asyncio.to_thread(node, state)

    async def resilient(state: dict) -> dict:
        # Breaker check — short-circuits if recent failures dominate.
        if not breaker.allow():
            _emit_event(name, "circuit_open")
            with node_span(f"{name}.circuit_open") as span:
                if span is not None:
                    span.set_attribute("smartteacher.resilience.circuit_open", True)
            return fallback_for(name)(state)

        # Untimed call — let LLMs take as long as they need on this hardware.
        # Wrapped in an OTel span so each node execution is observable.
        # The span is a no-op when OpenTelemetry isn't installed.
        started = time.monotonic()
        with node_span(name) as span:
            try:
                result = await _invoke(state)
                breaker.record_success()
                latency = time.monotonic() - started
                _emit_event(name, "ok", latency_s=latency)
                if not isinstance(result, dict):
                    log.warning("node '%s' returned non-dict (%s) — coerced to {}",
                                name, type(result).__name__)
                    if span is not None:
                        span.set_attribute("smartteacher.node.returned_non_dict", True)
                    return {}
                attach_node_attrs(span, name, result)
                return result
            except Exception as exc:
                breaker.record_failure()
                latency = time.monotonic() - started
                _emit_event(name, "error", latency_s=latency)
                log.exception("node '%s' raised: %s", name, exc)
                if span is not None:
                    span.set_attribute("smartteacher.node.exception", type(exc).__name__)
                return fallback_for(name)(state)

    # Preserve introspection for LangGraph (some versions read __name__)
    resilient.__name__ = f"resilient_{name}"
    return resilient
