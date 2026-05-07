"""Voice bypass handlers: execute navigation commands without calling LLM.

Provides simple functions that the WS handler can call when a voice
command is detected and should bypass the LLM (next, prev, goto, pause,
resume, repeat, quiz invocation wrapper).
"""
from __future__ import annotations

import logging

from pedagogy.dialogue import DialogueManager  # type: ignore

log = logging.getLogger("services.voice.bypass")


async def do_next(session_id: str, dialogue: DialogueManager) -> None:
    try:
        await dialogue.next_section(session_id)
    except Exception:
        log.exception("do_next failed")


async def do_prev(session_id: str, dialogue: DialogueManager) -> None:
    try:
        await dialogue.prev_section(session_id)
    except Exception:
        log.exception("do_prev failed")


async def do_goto(session_id: str, dialogue: DialogueManager, chapter_idx: int, section_idx: int) -> None:
    try:
        await dialogue.save_course_position(session_id, course_id=None, chapter_index=chapter_idx, section_index=section_idx, char_pos=0)
        await dialogue.transition(session_id, 'PRESENTING')
    except Exception:
        log.exception("do_goto failed")


async def do_pause(session_id: str, dialogue: DialogueManager) -> None:
    try:
        await dialogue.transition(session_id, 'LISTENING')
    except Exception:
        log.exception("do_pause failed")


async def do_resume(session_id: str, dialogue: DialogueManager) -> None:
    try:
        await dialogue.resume_from_clarification(session_id)
    except Exception:
        log.exception("do_resume failed")


async def flag_confusion(session_id: str, dialogue: DialogueManager, reason: str = "") -> None:
    try:
        await dialogue.mark_confusion_detected(session_id, reason=reason)
    except Exception:
        log.exception("flag_confusion failed")
