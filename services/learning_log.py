"""Persistance des "learning turns" (interactions) en PostgreSQL.

Une turn = un échange étudiant-prof (Q&A, pause, navigation, quiz, ...).
On stocke 3 choses par turn :
  - Interaction       (table simple : Q/R + timings + KPI)
  - LearningEvent     (table riche : event_type, payload, score confusion)
  - LearningSession   (mise à jour du curseur de présentation)

Tout est best-effort : si Postgres est indisponible, on log et on continue.
"""

import logging
import uuid
from typing import Optional

from core.config import Config
from database.init_db import AsyncSessionLocal
from database.crud import log_interaction, log_learning_event, update_session_state
from pedagogy.dialogue import DialogState

log = logging.getLogger("SmartTeacher.services.learning_log")


def safe_uuid(value: str | uuid.UUID | None) -> uuid.UUID | None:
    """Parse a UUID from str/UUID/None, returning None on any failure."""
    if value is None or value == "":
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except Exception:
        return None


async def persist_learning_turn(
    *,
    # Caller-bound context (was captured via closure in main.py)
    learning_session_db_id: uuid.UUID | None,
    session_id: str,
    ctx_course_id: str | uuid.UUID | None = None,
    # Per-turn data
    event_type: str,
    question_text: str = "",
    answer_text: str = "",
    language: str = "fr",
    subject: str = "",
    course_id_value: str | uuid.UUID | None = None,
    confusion_detected: bool = False,
    confusion_reason: str = "",
    action_taken: str = "answer",
    reward: float = 0.0,
    stt_time: float = 0.0,
    llm_time: float = 0.0,
    tts_time: float = 0.0,
    total_time: float = 0.0,
    profile_snapshot: dict | None = None,
    concept: str = "",
    chapter_index: int | None = None,
    section_index: int | None = None,
    char_position: int | None = None,
    extra_payload: dict | None = None,
    session_state: str | None = None,
) -> None:
    """Persist a turn in PostgreSQL so we can train later from real usage.

    Phase 2 enrichments are read from the agentic state via
    ``pedagogy.personalization.bandit.log_extractor.extra_payload_from_state``,
    which is merged into ``extra_payload`` at the ws.py call site. No
    new kwarg here — the extractor returns :
      - ``bandit_strategy`` / ``bandit_speech_rate`` / ``bandit_context``
        (bucket key) for offline RL training reproduction.
      - ``bandit_context_full`` (structured dict with student_state at
        decision time) for full reproducibility.
      - ``grounding_chunks`` (list of cited chunks) for transcript audit.
    """
    """Persist a turn in PostgreSQL so we can train later from real usage."""
    if learning_session_db_id is None:
        return
    if session_state is None:
        session_state = DialogState.LISTENING.value

    course_uuid = safe_uuid(course_id_value)
    if course_uuid is None and ctx_course_id:
        course_uuid = safe_uuid(ctx_course_id)

    try:
        async with AsyncSessionLocal() as db:
            await log_interaction(
                db=db,
                session_id=learning_session_db_id,
                student_id=session_id,
                course_id=course_uuid,
                interaction_type=event_type,
                question=question_text,
                answer=answer_text,
                language=language,
                stt_time=stt_time,
                llm_time=llm_time,
                tts_time=tts_time,
                total_time=total_time,
                kpi_ok=1 if total_time <= Config.MAX_RESPONSE_TIME else 0,
            )
            await log_learning_event(
                db=db,
                session_id=learning_session_db_id,
                student_id=session_id,
                course_id=course_uuid,
                event_type=event_type,
                input_text=question_text,
                output_text=answer_text,
                concept=concept or subject or None,
                action_taken=action_taken,
                confusion_score=1.0 if confusion_detected else 0.0,
                reward=reward,
                stt_time=stt_time,
                llm_time=llm_time,
                tts_time=tts_time,
                total_time=total_time,
                student_state=profile_snapshot or {},
                event_payload={
                    "subject": subject,
                    "language": language,
                    "confusion_reason": confusion_reason,
                    "chapter_index": chapter_index,
                    "section_index": section_index,
                    "char_position": char_position,
                    **(extra_payload or {}),
                },
            )
            await update_session_state(
                db,
                learning_session_db_id,
                session_state,
                chapter_index=chapter_index,
                section_index=section_index,
                char_position=char_position,
            )
    except Exception as exc:
        log.debug(f"[{session_id[:8]}] Learning event persistence skipped: {exc}")
