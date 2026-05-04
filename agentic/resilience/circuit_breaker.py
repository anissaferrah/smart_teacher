"""Per-node circuit breaker (Nygard 2007 pattern).

# Problem

When an LLM endpoint degrades (Ollama queue saturated, model swap, GPU OOM
fallback to CPU), every call times out. Without a breaker, every user turn
pays the full timeout latency before the fallback kicks in. With 4 nodes
that's potentially 12–15s of pure waiting per turn, killing UX.

# Solution

A circuit breaker tracks the recent failure rate of each node. If the rate
crosses a threshold, the breaker **opens**: subsequent calls bypass the node
entirely and go straight to the fallback. After a cool-down, the breaker
moves to **half-open**, allowing a single probe call. Probe success → closed
(normal); probe failure → re-open with the same cool-down.

States:
    closed     → normal operation
    open       → all calls short-circuit to fallback
    half_open  → next call is a probe; success closes, failure re-opens

# Calibration

  - window      : last 20 calls per node
  - threshold   : 50% failure rate over the window
  - min_samples : 5 (avoid premature opening on early errors)
  - cool_down   : 60s open → half_open

These are conservative defaults. Production tuning is per-node-per-node.

# References
  - Nygard, M. T. (2007). *Release It!* — Chapter on Stability Patterns.
  - Fowler, M. CircuitBreaker (martinfowler.com/bliki/CircuitBreaker.html).
  - Netflix Hystrix design doc — adopted the same three-state machine.

This breaker is **per-process**. For multi-replica deployments, replace the
in-memory state with Redis-backed counters (same algorithm).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger("agentic.resilience.circuit_breaker")


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class Breaker:
    """Single-node circuit breaker. Thread-safe."""

    name: str
    window: int = 20                # last N calls considered
    threshold: float = 0.50         # failure rate that triggers open
    min_samples: int = 5            # don't open before this many calls
    cool_down_s: float = 60.0       # open → half_open after this delay

    _state: BreakerState = field(default=BreakerState.CLOSED, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _outcomes: deque = field(default_factory=deque, init=False)  # bool: True=fail
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # ── Public API ─────────────────────────────────────────────────

    def allow(self) -> bool:
        """Return True if the node call should proceed, False if short-circuit."""
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == BreakerState.OPEN:
                return False
            # CLOSED or HALF_OPEN both proceed; HALF_OPEN gets a single probe.
            return True

    def record_success(self) -> None:
        with self._lock:
            self._outcomes.append(False)
            self._trim()
            if self._state == BreakerState.HALF_OPEN:
                log.info("breaker[%s]: HALF_OPEN probe succeeded → CLOSED", self.name)
                self._state = BreakerState.CLOSED
                self._opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._outcomes.append(True)
            self._trim()
            if self._state == BreakerState.HALF_OPEN:
                log.warning("breaker[%s]: HALF_OPEN probe failed → OPEN (cool-down %.0fs)",
                            self.name, self.cool_down_s)
                self._state = BreakerState.OPEN
                self._opened_at = time.monotonic()
                return
            # Closed → check threshold
            if self._state == BreakerState.CLOSED and self._should_open():
                fr = self._failure_rate()
                log.warning(
                    "breaker[%s]: failure_rate=%.2f over %d samples → OPEN (cool-down %.0fs)",
                    self.name, fr, len(self._outcomes), self.cool_down_s,
                )
                self._state = BreakerState.OPEN
                self._opened_at = time.monotonic()

    def state(self) -> BreakerState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def stats(self) -> dict:
        with self._lock:
            return {
                "name":          self.name,
                "state":         self._state.value,
                "samples":       len(self._outcomes),
                "failure_rate":  round(self._failure_rate(), 3),
                "opened_at":     self._opened_at,
                "cool_down_s":   self.cool_down_s,
            }

    # ── Internal ───────────────────────────────────────────────────

    def _trim(self) -> None:
        while len(self._outcomes) > self.window:
            self._outcomes.popleft()

    def _failure_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(1 for x in self._outcomes if x) / len(self._outcomes)

    def _should_open(self) -> bool:
        return (
            len(self._outcomes) >= self.min_samples
            and self._failure_rate() >= self.threshold
        )

    def _maybe_transition_to_half_open(self) -> None:
        if self._state == BreakerState.OPEN and (time.monotonic() - self._opened_at) >= self.cool_down_s:
            log.info("breaker[%s]: cool-down elapsed → HALF_OPEN", self.name)
            self._state = BreakerState.HALF_OPEN


# ── Registry ──────────────────────────────────────────────────────────

class BreakerRegistry:
    """Process-wide registry of node breakers. Lazily creates on first ask."""

    _breakers: dict[str, Breaker] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get(cls, node_name: str) -> Breaker:
        with cls._lock:
            br = cls._breakers.get(node_name)
            if br is None:
                br = Breaker(name=node_name)
                cls._breakers[node_name] = br
            return br

    @classmethod
    def all_stats(cls) -> list[dict]:
        with cls._lock:
            return [b.stats() for b in cls._breakers.values()]

    @classmethod
    def reset(cls, node_name: str | None = None) -> None:
        """Reset a node's breaker (or all). For tests / admin endpoint."""
        with cls._lock:
            if node_name:
                cls._breakers.pop(node_name, None)
            else:
                cls._breakers.clear()
