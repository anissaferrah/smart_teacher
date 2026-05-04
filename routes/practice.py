"""Practice / concept question endpoints (Sprint 1 — generate, fetch, submit)."""

import logging
from fastapi import APIRouter, Form, HTTPException

router = APIRouter()
log = logging.getLogger("SmartTeacher.routes.practice")


@router.post("/concept/{concept_id}/generate-questions")
async def generate_practice_questions(concept_id: str, language: str = "fr"):
    """Auto-generate 3 questions (easy/medium/hard) via LLM."""
    from pedagogy.practice_engine import PracticeEngine
    try:
        count = await PracticeEngine.generate_for_concept(concept_id, lang=language)
        return {"concept_id": concept_id, "questions_count": count, "language": language}
    except Exception as exc:
        log.exception("generate_practice_questions failed")
        raise HTTPException(status_code=500, detail=f"PracticeEngine error: {exc}")


@router.get("/concept/{concept_id}/practice")
async def get_practice_question(concept_id: str, difficulty: str | None = None):
    """Fetch une question pour ce concept (auto-generate si vide)."""
    from pedagogy.practice_engine import PracticeEngine
    try:
        q = await PracticeEngine.get_question(concept_id, difficulty=difficulty)
        if q is None:
            n = await PracticeEngine.generate_for_concept(concept_id)
            if n == 0:
                raise HTTPException(status_code=503, detail="LLM unavailable to generate questions")
            q = await PracticeEngine.get_question(concept_id, difficulty=difficulty)
        if q is None:
            raise HTTPException(status_code=404, detail="No question available")
        return q
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("get_practice_question failed")
        raise HTTPException(status_code=500, detail=f"PracticeEngine error: {exc}")


@router.post("/practice/{question_id}/submit")
async def submit_practice_answer(
    question_id: str,
    student_id: str = Form(...),
    answer_text: str = Form(...),
    hints_used: int = Form(0),
    time_taken_s: float = Form(0.0),
):
    """Submit student answer, grade, update mastery."""
    from pedagogy.practice_engine import PracticeEngine
    try:
        result = await PracticeEngine.submit_answer(
            question_id=question_id,
            student_id=student_id,
            answer_text=answer_text,
            hints_used=hints_used,
            time_taken_s=time_taken_s,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        # ✨ Bayes posterior — practice attempt = strong kinesthetic signal
        from pedagogy.personalization.learning_style.bayes import fire_signal
        fire_signal(student_id, "practice_attempt")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("submit_practice_answer failed")
        raise HTTPException(status_code=500, detail=f"PracticeEngine error: {exc}")
