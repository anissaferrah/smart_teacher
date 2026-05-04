"""Resilience layer for LangGraph nodes.

Two primitives:
  - fallbacks.NODE_FALLBACKS  : deterministic state-update producers used
                                  when a node raises or is short-circuited
                                  by the breaker.
  - circuit_breaker.Breaker   : per-node failure-rate tracker (Nygard 2007).
                                  Marks a node "open" when failures dominate
                                  and routes subsequent calls straight to the
                                  fallback for a cool-down period.

The stack is composable: build_resilient_node(node, name)
chains both so the graph builder simply substitutes node→resilient(node).
"""
from agentic.resilience.fallbacks import NODE_FALLBACKS, fallback_for
from agentic.resilience.circuit_breaker import Breaker, BreakerRegistry
from agentic.resilience.wrap import build_resilient_node

__all__ = [
    "NODE_FALLBACKS",
    "fallback_for",
    "Breaker",
    "BreakerRegistry",
    "build_resilient_node",
]
