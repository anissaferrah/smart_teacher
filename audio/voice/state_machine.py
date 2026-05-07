"""VoiceStateMachine — Single source of truth per session for voice state.

Resout 4 problemes architecturaux du voice runtime :
  1. Race conditions audio/state (transitions atomiques sous lock asyncio)
  2. State eparpille (RESPONDING + presentation_start_time + interrupt_audio + ctx.state) → centralisé ici
  3. Stuck states (THINKING infini si LLM hang) → watchdog timeout
  4. Coordination async (event bus broadcast aux listeners)

Usage :
    vfsm = VoiceStateMachine(session_id)
    await vfsm.transition(VoiceState.LISTENING, reason="user started speaking")
    if vfsm.state == VoiceState.SPEAKING and vfsm.elapsed_in_state() > 30:
        # stuck → recover
        await vfsm.transition(VoiceState.LISTENING, reason="timeout")
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

log = logging.getLogger("SmartTeacher.VoiceFSM")


class VoiceState(str, Enum):
    """5 etats possibles du voice runtime (mutuellement exclusifs)."""

    IDLE = "idle"                  # pas de session active
    LISTENING = "listening"        # mic ouvert, on attend/recoit l'audio user
    THINKING = "thinking"          # STT done, LLM/RAG en cours
    SPEAKING = "speaking"          # TTS streame vers le client
    INTERRUPTED = "interrupted"    # user a coupe TTS, transition vers LISTENING


# Transitions autorisees : un Set de targets valides depuis chaque state
_ALLOWED_TRANSITIONS: dict[VoiceState, set[VoiceState]] = {
    VoiceState.IDLE:        {VoiceState.LISTENING, VoiceState.SPEAKING},  # SPEAKING = narration auto-play
    VoiceState.LISTENING:   {VoiceState.THINKING, VoiceState.IDLE, VoiceState.SPEAKING},
    VoiceState.THINKING:    {VoiceState.SPEAKING, VoiceState.LISTENING, VoiceState.IDLE, VoiceState.INTERRUPTED},
    VoiceState.SPEAKING:    {VoiceState.LISTENING, VoiceState.INTERRUPTED, VoiceState.IDLE, VoiceState.THINKING},
    VoiceState.INTERRUPTED: {VoiceState.LISTENING, VoiceState.IDLE},
}

# Timeouts de safety (auto-recover) — secondes
_STATE_TIMEOUTS: dict[VoiceState, float] = {
    VoiceState.THINKING:    180.0,   # 3 min max (LLM Ollama CPU peut etre lent)
    VoiceState.SPEAKING:    240.0,   # 4 min max (TTS streaming + audio playback)
    VoiceState.INTERRUPTED:  10.0,   # 10s pour cleaner et repasser en LISTENING
    VoiceState.LISTENING:   600.0,   # 10 min user inactif → IDLE
}


@dataclass
class VoiceEvent:
    """Event emis par le FSM aux listeners (event bus)."""
    type: str                            # state_change | tts_start | tts_end | user_interrupt | ...
    session_id: str
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)


class VoiceStateMachine:
    """FSM centrale par session WS. Transitions atomiques + watchdog + listeners."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._state = VoiceState.IDLE
        self._entered_at = time.time()
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[VoiceEvent], Awaitable[None] | None]] = []
        # Voice runtime metadata co-localisee (1 source de verite)
        self.tts_started_at: float = 0.0
        self.last_user_activity: float = time.time()
        self.transition_count: int = 0
        self.last_reason: str = "init"

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def entered_at(self) -> float:
        return self._entered_at

    def elapsed_in_state(self) -> float:
        """Secondes depuis l'entree dans l'etat courant."""
        return time.time() - self._entered_at

    def is_stuck(self) -> bool:
        """True si l'etat actuel a depasse son timeout."""
        timeout = _STATE_TIMEOUTS.get(self._state)
        if timeout is None:
            return False
        return self.elapsed_in_state() > timeout

    def can_transition_to(self, target: VoiceState) -> bool:
        return target in _ALLOWED_TRANSITIONS.get(self._state, set())

    async def transition(self, target: VoiceState, reason: str = "") -> bool:
        """Transition atomique. Returns True si applique, False si illegal/no-op.

        Side effects :
          - Logs la transition
          - Emit event aux listeners (broadcast)
          - Reset entered_at + last_reason
          - Update tts_started_at si target = SPEAKING (premier passage)
        """
        async with self._lock:
            if target == self._state:
                return False  # no-op (silencieux)

            if target not in _ALLOWED_TRANSITIONS.get(self._state, set()):
                log.warning(
                    f"[{self.session_id[:8]}] VFSM ❌ illegal transition "
                    f"{self._state.value} → {target.value} (reason={reason})"
                )
                return False

            old_state = self._state
            self._state = target
            self._entered_at = time.time()
            self._last_reason = reason
            self.transition_count += 1

            # Hooks specifiques par target
            if target == VoiceState.SPEAKING and self.tts_started_at == 0.0:
                self.tts_started_at = time.time()
            elif target == VoiceState.LISTENING:
                # Reset TTS marker quand on revient en listening
                self.tts_started_at = 0.0
                self.last_user_activity = time.time()

            log.info(
                f"[{self.session_id[:8]}] VFSM : {old_state.value} → {target.value} "
                f"(reason: {reason})"
            )

            event = VoiceEvent(
                type="state_change",
                session_id=self.session_id,
                payload={
                    "from": old_state.value,
                    "to": target.value,
                    "reason": reason,
                    "transition_count": self.transition_count,
                },
            )
            # Broadcast (best-effort, errors logged)
            for listener in list(self._listeners):
                try:
                    result = listener(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    log.debug(f"VFSM listener error : {exc}")

            return True

    def add_listener(
        self, listener: Callable[[VoiceEvent], Awaitable[None] | None]
    ) -> None:
        """Subscribe a fn(VoiceEvent) (sync ou async) — appelee a chaque transition."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def force_recover(self, reason: str = "watchdog_recovery") -> bool:
        """Force un retour vers LISTENING (safety net depuis stuck states).

        Utilise par le watchdog quand un state depasse son timeout.
        """
        log.debug(
            f"[{self.session_id[:8]}] VFSM 🚨 force_recover : "
            f"state={self._state.value} elapsed={self.elapsed_in_state():.1f}s reason={reason}"
        )
        # Force a LISTENING (clear out)
        async with self._lock:
            old = self._state
            self._state = VoiceState.LISTENING
            self._entered_at = time.time()
            self.tts_started_at = 0.0
            self.transition_count += 1
            event = VoiceEvent(
                type="state_change",
                session_id=self.session_id,
                payload={"from": old.value, "to": "listening", "reason": reason, "forced": True},
            )
            for listener in list(self._listeners):
                try:
                    result = listener(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass
            return True

    def snapshot(self) -> dict[str, Any]:
        """Diagnostic dict (pour /debug or websocket state messages)."""
        return {
            "state": self._state.value,
            "elapsed_s": round(self.elapsed_in_state(), 2),
            "tts_started_at": self.tts_started_at,
            "last_user_activity": self.last_user_activity,
            "transitions": self.transition_count,
            "last_reason": getattr(self, "_last_reason", ""),
            "stuck": self.is_stuck(),
        }


async def watchdog_loop(
    vfsm: VoiceStateMachine,
    poll_interval_s: float = 5.0,
    cancel_callback: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Background task : check is_stuck() periodiquement et force_recover si oui.

    Si un cancel_callback est fourni, il est invoke avant le force_recover
    (pour ex. cancel l'audio stream / LLM task en cours).

    Usage :
        watchdog_task = asyncio.create_task(watchdog_loop(vfsm))
        # ... session ...
        watchdog_task.cancel()
    """
    log.debug(f"[{vfsm.session_id[:8]}] VFSM watchdog started")
    try:
        while True:
            await asyncio.sleep(poll_interval_s)
            if vfsm.state == VoiceState.IDLE:
                continue
            if vfsm.is_stuck():
                log.warning(
                    f"[{vfsm.session_id[:8]}] VFSM watchdog detected stuck "
                    f"state={vfsm.state.value} elapsed={vfsm.elapsed_in_state():.1f}s"
                )
                if cancel_callback is not None:
                    try:
                        await cancel_callback()
                    except Exception as exc:
                        log.debug(f"watchdog cancel_callback failed: {exc}")
                await vfsm.force_recover(reason="watchdog_timeout")
    except asyncio.CancelledError:
        log.debug(f"[{vfsm.session_id[:8]}] VFSM watchdog stopped")
        raise
    except Exception as exc:
        log.warning(f"[{vfsm.session_id[:8]}] VFSM watchdog error: {exc}")
