"""Optional OpenTelemetry instrumentation for the agentic graph.

# Design : zero-dependency by default

OpenTelemetry is intentionally NOT a hard requirement. The module
lazy-imports the OTel API, and falls back to a no-op tracer when the
package is missing. Every public function in this module remains
callable; instrumentation simply becomes a no-op until the user installs
``opentelemetry-api`` + ``opentelemetry-sdk``.

To enable tracing, run :

    pip install opentelemetry-api opentelemetry-sdk

and configure an exporter (Jaeger / OTLP / Console) at process startup.
``setup_tracing()`` provides a minimal Console exporter for local
debugging — production deployments should override this with a proper
exporter pointing at Jaeger / Honeycomb / Datadog / etc.

# Per-node attributes

``attach_node_attrs(span, node_name, result)`` extracts well-known
fields from a node's output dict and attaches them as span attributes,
keyed under the ``smartteacher.<node>.<field>`` namespace. The mapping
is hand-crafted per node type so each span surfaces the right signal :

  - intent     : type, confidence, needs_retrieval, feedback_polarity
  - rewriter   : rewritten_length, anchored_concept
  - retriever  : chunks_count, kg_augmented_count
  - responder  : answer_length, citations_count, grounded, fallback
  - reviewer   : grounded, score
  - fallback   : kind
  - planner    : ideas_count
  - narrator   : answer_length, retries

Unknown nodes get a generic ``output_keys`` attribute (the keys
returned). No silent loss of signal.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator

log = logging.getLogger("agentic.observability")


# ── Lazy OTel import ──────────────────────────────────────────────────────
# Done at module load. If OTel is available we wire ``_tracer`` to a real
# tracer; otherwise we keep ``None`` and the spans become no-ops.

_tracer: Any = None
_otel_available: bool = False

try:
    from opentelemetry import trace as _otel_trace
    _tracer = _otel_trace.get_tracer("smartteacher.agentic")
    _otel_available = True
    log.debug("OpenTelemetry available — agentic graph instrumentation active")
except Exception as _exc:                                                # noqa: BLE001
    log.debug("OpenTelemetry not installed — agentic instrumentation is a no-op (%s)", _exc)


@contextlib.contextmanager
def node_span(node_name: str) -> Iterator[Any]:
    """Context manager that wraps a node call in an OTel span.

    Yields the active span (or ``None`` if OTel isn't installed).
    Callers can attach attributes via ``span.set_attribute(...)`` if the
    span is not None.
    """
    if not _otel_available or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(f"agentic.{node_name}") as span:
        try:
            yield span
        except Exception:
            # Re-raise — the resilient wrapper handles fallback. We only
            # mark the span as errored before letting it propagate.
            if span is not None:
                try:
                    from opentelemetry.trace import StatusCode
                    span.set_status(StatusCode.ERROR)
                except Exception:                                         # noqa: BLE001
                    pass
            raise


def attach_node_attrs(span: Any, node_name: str, result: dict | None) -> None:
    """Attach node-specific signal to the span as semantic attributes.

    No-op when ``span`` is None (OTel not installed) or ``result`` is
    not a dict. Failures are swallowed — instrumentation must never
    affect graph execution.
    """
    if span is None or not isinstance(result, dict):
        return
    try:
        ns = "smartteacher"
        if node_name in ("intent", "intent_question"):
            intent = result.get("intent")
            if intent is not None:
                span.set_attribute(f"{ns}.intent.type", str(getattr(intent, "type", "?")))
                span.set_attribute(f"{ns}.intent.confidence", float(getattr(intent, "confidence", 0.0)))
                span.set_attribute(f"{ns}.intent.needs_retrieval", bool(getattr(intent, "needs_retrieval", True)))
                payload = getattr(intent, "payload", None) or {}
                if isinstance(payload, dict):
                    pol = payload.get("feedback_polarity")
                    if pol:
                        span.set_attribute(f"{ns}.intent.feedback_polarity", str(pol))
                    src = payload.get("source")
                    if src:
                        span.set_attribute(f"{ns}.intent.source", str(src))
        elif node_name == "rewriter":
            rewritten = result.get("rewritten_query") or ""
            span.set_attribute(f"{ns}.rewriter.length", len(rewritten))
            anchored = result.get("anchored_concept") or ""
            if anchored:
                span.set_attribute(f"{ns}.rewriter.anchored_concept", anchored[:120])
        elif node_name == "retriever":
            chunks = result.get("retrieved_chunks") or []
            span.set_attribute(f"{ns}.retriever.chunks_count", len(chunks))
            kg_count = sum(1 for c in chunks if isinstance(c, dict) and c.get("_via_kg"))
            span.set_attribute(f"{ns}.retriever.kg_augmented_count", kg_count)
        elif node_name == "responder":
            answer = result.get("answer") or ""
            citations = result.get("citations") or []
            span.set_attribute(f"{ns}.responder.answer_length", len(answer))
            span.set_attribute(f"{ns}.responder.citations_count", len(citations))
            span.set_attribute(f"{ns}.responder.grounded", bool(citations))
            confidence = result.get("confidence")
            if confidence is not None:
                span.set_attribute(f"{ns}.responder.confidence", float(confidence))
            actions = result.get("actions") or []
            for a in actions:
                payload = getattr(a, "payload", None) or {}
                if isinstance(payload, dict) and payload.get("fallback"):
                    span.set_attribute(f"{ns}.responder.fallback", True)
                    break
        elif node_name == "review":
            review = result.get("review")
            if review is not None:
                span.set_attribute(f"{ns}.review.grounded", bool(getattr(review, "grounded", False)))
                span.set_attribute(f"{ns}.review.score", float(getattr(review, "score", 0.0)))
        elif node_name == "fallback":
            actions = result.get("actions") or []
            for a in actions:
                payload = getattr(a, "payload", None) or {}
                if isinstance(payload, dict) and payload.get("fallback"):
                    span.set_attribute(f"{ns}.fallback.kind", str(payload.get("kind", "?")))
                    break
        elif node_name == "planner":
            plan = result.get("plan")
            if plan is not None:
                ideas = getattr(plan, "ideas", None) or []
                span.set_attribute(f"{ns}.planner.ideas_count", len(ideas))
        elif node_name == "narrator":
            answer = result.get("answer") or ""
            span.set_attribute(f"{ns}.narrator.answer_length", len(answer))
            retries = result.get("narrator_retries")
            if retries is not None:
                span.set_attribute(f"{ns}.narrator.retries", int(retries))
        else:
            # Generic — surface the keys produced so unknown nodes are
            # still observable.
            span.set_attribute(f"{ns}.output_keys", ",".join(sorted(result.keys()))[:240])
    except Exception as exc:                                              # noqa: BLE001
        log.debug("attach_node_attrs failed for %s: %s", node_name, exc)


def setup_tracing(service_name: str = "smartteacher", console: bool = True) -> bool:
    """Minimal local-debug tracing setup with a Console exporter.

    Returns True if OTel was wired, False if the package is missing.
    Production code should bypass this and configure its own exporter
    (Jaeger / OTLP / Honeycomb / Datadog).
    """
    if not _otel_available:
        log.warning("setup_tracing: OpenTelemetry not installed — tracing disabled")
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if console:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        log.info("setup_tracing: OpenTelemetry configured with %s exporter",
                 "console" if console else "user-defined")
        return True
    except Exception as exc:                                              # noqa: BLE001
        log.warning("setup_tracing failed: %s", exc)
        return False


def is_available() -> bool:
    """True if OpenTelemetry is importable and a tracer is wired."""
    return _otel_available
