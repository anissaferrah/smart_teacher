"""AgenticOrchestrator — bridge entre VoiceStateMachine et les LangGraph
agentic (Q&A + Teaching).

Resout :
  1. Cancellation propre des LLM calls quand user interrompt (FSM → INTERRUPTED)
  2. Per-session task tracking (1 active task max par session WS)
  3. Lifecycle transitions automatiques : THINKING → SPEAKING ou LISTENING
  4. Watchdog-friendly (cancel callback re-utilise par le FSM watchdog)

Usage minimal (depuis le WS handler) :

    orchestrator = AgenticOrchestrator(qa_graph, teaching_graph)
    vfsm.add_listener(orchestrator.on_state_change)   # auto-cancel on INTERRUPTED

    # Process a turn
    await vfsm.transition(VoiceState.THINKING, reason="user_question")
    result = await orchestrator.run("qa", session_id, state)
    if result is None:
        await vfsm.transition(VoiceState.LISTENING, reason="agentic_cancelled")
    else:
        await vfsm.transition(VoiceState.SPEAKING, reason="agentic_done")
        # ... stream TTS ...
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from audio.voice.state_machine import VoiceEvent, VoiceState

log = logging.getLogger("agentic.orchestrator")


GraphKind = Literal["qa", "teaching"]


class AgenticOrchestrator:
    """Coordonne les invocations LangGraph + cancel signal cross-session.

    Key responsibilities :
      - Track 1 active asyncio.Task par session
      - Cancel ce task quand FSM transitionne vers INTERRUPTED
      - Wrap les exceptions / timeouts proprement (return None si interrompu)
      - Logger lifecycle (start, success, cancel, error, duration)
    """

    def __init__(self, qa_graph=None, teaching_graph=None) -> None:
        self.qa_graph = qa_graph
        self.teaching_graph = teaching_graph
        # Per-session : 1 active task max
        self._active_tasks: dict[str, asyncio.Task] = {}
        # Stats
        self._stats: dict[str, int] = {"started": 0, "completed": 0, "cancelled": 0, "errored": 0}

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(
        self,
        graph_kind: GraphKind,
        session_id: str,
        state: dict[str, Any],
        timeout_s: float | None = None,
    ) -> dict[str, Any] | None:
        """Execute le graph (qa | teaching) pour cette session.

        Returns :
          - dict (graph output) si succes
          - None si cancelled, timeout, ou erreur (caller gere FSM transition)

        Pre-requis : FSM doit deja etre en THINKING avant l'appel.
        """
        graph = self.qa_graph if graph_kind == "qa" else self.teaching_graph
        if graph is None:
            log.warning(f"orchestrator.run: graph '{graph_kind}' not registered")
            return None

        # Cancel toute task precedente pour cette session (1 active par session)
        await self.cancel(session_id, reason="new_run_started")

        async def _exec() -> dict[str, Any]:
            return await graph.ainvoke(state)

        coro = _exec() if timeout_s is None else asyncio.wait_for(_exec(), timeout=timeout_s)
        task = asyncio.create_task(coro, name=f"agentic_{graph_kind}_{session_id[:8]}")
        self._active_tasks[session_id] = task
        self._stats["started"] += 1
        start_ts = time.time()

        log.info(
            f"agentic[{graph_kind}] STARTED session={session_id[:8]} "
            f"timeout={timeout_s or 'none'}"
        )

        try:
            result = await task
            elapsed = time.time() - start_ts
            self._stats["completed"] += 1
            log.info(
                f"agentic[{graph_kind}] DONE session={session_id[:8]} "
                f"elapsed={elapsed:.1f}s"
            )
            return result
        except asyncio.CancelledError:
            elapsed = time.time() - start_ts
            self._stats["cancelled"] += 1
            log.info(
                f"agentic[{graph_kind}] CANCELLED session={session_id[:8]} "
                f"elapsed={elapsed:.1f}s"
            )
            return None
        except asyncio.TimeoutError:
            elapsed = time.time() - start_ts
            self._stats["errored"] += 1
            log.warning(
                f"agentic[{graph_kind}] TIMEOUT session={session_id[:8]} "
                f"after {elapsed:.1f}s (limit={timeout_s}s)"
            )
            return None
        except Exception as exc:
            elapsed = time.time() - start_ts
            self._stats["errored"] += 1
            log.warning(
                f"agentic[{graph_kind}] ERROR session={session_id[:8]} "
                f"elapsed={elapsed:.1f}s exc={exc}"
            )
            return None
        finally:
            # Toujours retirer du registre (cleanup)
            self._active_tasks.pop(session_id, None)

    async def cancel(self, session_id: str, reason: str = "manual") -> bool:
        """Cancel l'active task pour cette session (si une existe).

        Returns True si une task a ete cancellee, False sinon.
        Best-effort : swallow toutes les exceptions du task cancelle.
        """
        task = self._active_tasks.get(session_id)
        if task is None or task.done():
            return False
        log.info(f"agentic CANCEL session={session_id[:8]} reason={reason}")
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        self._active_tasks.pop(session_id, None)
        return True

    def has_active(self, session_id: str) -> bool:
        """True si une task tourne pour cette session."""
        task = self._active_tasks.get(session_id)
        return task is not None and not task.done()

    def stats(self) -> dict[str, Any]:
        """Diagnostic stats pour debug endpoint."""
        return {
            **self._stats,
            "active_sessions": len(self._active_tasks),
            "active_session_ids": list(self._active_tasks.keys()),
        }

    # ── FSM listener (a brancher via vfsm.add_listener) ─────────────────────

    async def on_state_change(self, event: VoiceEvent) -> None:
        """VFSM listener : auto-cancel l'agentic task quand on entre en INTERRUPTED.

        A enregistrer via : vfsm.add_listener(orchestrator.on_state_change)
        """
        if event.type != "state_change":
            return
        target_state = event.payload.get("to")
        if target_state == VoiceState.INTERRUPTED.value:
            cancelled = await self.cancel(event.session_id, reason="fsm_interrupted")
            if cancelled:
                log.info(f"agentic auto-cancel triggered by FSM → INTERRUPTED [{event.session_id[:8]}]")

    # ── Watchdog cancel callback (a passer a watchdog_loop) ─────────────────

    async def watchdog_cancel_for(self, session_id: str) -> None:
        """Wrapper compatible avec le watchdog cancel_callback signature."""
        await self.cancel(session_id, reason="watchdog_timeout")
