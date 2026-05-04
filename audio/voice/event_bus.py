"""EventBus — minimal async pub/sub pour le voice runtime.

Resout : coordination entre composants async (TTS, STT, FSM, Q&A graph)
sans coupler les modules entre eux.

Pattern :
    bus = EventBus()
    @bus.on("user_speaks")
    async def on_user_speaks(ev): ...
    await bus.emit(VoiceEvent(type="user_speaks", session_id=..., payload={"text": "..."}))
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Callable, Awaitable

from audio.voice.state_machine import VoiceEvent

log = logging.getLogger("SmartTeacher.EventBus")

# Event types standardises (pour eviter les typos cross-modules)
EVENT_USER_SPEAKS       = "user_speaks"        # mic activity detectee
EVENT_USER_INTERRUPTS   = "user_interrupts"    # interruption confirmee (3 gates passes)
EVENT_TTS_STARTED       = "tts_started"        # 1er chunk envoye au client
EVENT_TTS_FINISHED      = "tts_finished"       # streaming complete
EVENT_TTS_CANCELLED     = "tts_cancelled"      # cancel_audio_stream call
EVENT_STT_DONE          = "stt_done"           # whisper output ready
EVENT_LLM_STARTED       = "llm_started"        # llm.invoke begin
EVENT_LLM_FINISHED      = "llm_finished"       # llm reply ready
EVENT_STATE_CHANGE      = "state_change"       # VFSM transition (auto)
EVENT_TURN_DISCARDED    = "turn_discarded"     # RMS or echo xcorr filtered out


HandlerType = Callable[[VoiceEvent], Awaitable[None] | None]


class EventBus:
    """Async pub/sub par type. Best-effort delivery (errors logged, ne bloquent pas)."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[HandlerType]] = defaultdict(list)
        # Wildcard handlers reçoivent TOUS les events
        self._wildcard: list[HandlerType] = []

    def on(self, event_type: str):
        """Decorator : @bus.on("user_speaks") → register handler."""
        def decorator(fn: HandlerType) -> HandlerType:
            self.subscribe(event_type, fn)
            return fn
        return decorator

    def subscribe(self, event_type: str, handler: HandlerType) -> None:
        """Programmatic subscribe (use '*' pour wildcard)."""
        if event_type == "*":
            if handler not in self._wildcard:
                self._wildcard.append(handler)
        else:
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: HandlerType) -> None:
        if event_type == "*":
            if handler in self._wildcard:
                self._wildcard.remove(handler)
        else:
            if handler in self._handlers.get(event_type, []):
                self._handlers[event_type].remove(handler)

    async def emit(self, event: VoiceEvent) -> None:
        """Broadcast event aux handlers + wildcards. Best-effort, ne raise pas."""
        targets = list(self._handlers.get(event.type, [])) + list(self._wildcard)
        if not targets:
            return
        # Lance tout en parallele pour ne pas bloquer la chaine
        coros = []
        for handler in targets:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    coros.append(result)
            except Exception as exc:
                log.debug(f"EventBus sync handler error ({event.type}): {exc}")
        if coros:
            results = await asyncio.gather(*coros, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.debug(f"EventBus async handler error ({event.type}): {r}")

    def clear(self) -> None:
        """Reset tous les handlers (utile en teardown session)."""
        self._handlers.clear()
        self._wildcard.clear()
