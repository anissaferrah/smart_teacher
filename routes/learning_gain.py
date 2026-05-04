"""Pre-test / Post-test endpoints — measure pedagogical learning gain.

Why this matters
----------------
The standard pedagogy KPI for "did this teaching actually work ?" is
**Hake's normalized learning gain** (Hake 1998, *Am. J. Phys.*) :

    g = (post - pre) / (1 - pre)

Where ``pre`` and ``post`` are scores in [0, 1] on a comparable test
taken before and after the teaching session. ``g`` is in (-∞, 1] :

  - g > 0    : student learned something (the higher the better)
  - g ≈ 0    : no measurable progress
  - g < 0    : student got worse (rare — usually a flawed test)

Without this number, no defensible claim can be made for the thesis.

Pipeline
--------

  1. Student opens a course → ``POST /quiz/pretest/{course_id}/start``
     → server generates 5 MCQ questions from the KG, returns them with
     a ``test_id`` to track the submission.
  2. Student answers → ``POST /quiz/pretest/{course_id}/submit``
     → server scores, persists in ``learning_gain_tests`` (test_type='pretest').
  3. Student goes through the course (Smart Teacher tutoring).
  4. Student takes the post-test → same 2-step flow with
     ``test_type='posttest'``.
  5. Anyone can compute the gain via ``GET /quiz/{course_id}/gain``
     for one student or aggregated.

Question generation
-------------------
First version uses the **LLM** to draft 5 multiple-choice questions
from the course's KG concepts. We don't reuse ``PracticeEngine`` here
because that one's tuned for spaced-repetition (Bloom-level mix). For
gain measurement we want a *fixed difficulty* set — same questions
pre and post (or a parallel form).

For simplicity we serve the SAME questions pre/post in the first
iteration. A future v2 can implement parallel forms (different
questions, equivalent difficulty) once we have item-response data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid as _uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, desc

from handlers.auth import get_current_user

router = APIRouter()
log = logging.getLogger("SmartTeacher.routes.learning_gain")


# ── Operational caps ────────────────────────────────────────────────────
# Five MCQs is the standard short-form pre/post in education research
# (Hake's original studies used 30-item Force Concept Inventory ; 5 is
# the demo equivalent — enough to detect a 20% gain reliably with
# n ≥ 20 students). Configurable if the operator wants a longer form.
_DEFAULT_NUM_QUESTIONS = 5
_MIN_NUM_QUESTIONS = 3
_MAX_NUM_QUESTIONS = 15


# ── Pydantic input models ───────────────────────────────────────────────

class StartTestRequest(BaseModel):
    """Optional request body — the operator can override the question count."""
    num_questions: int | None = None


class SubmitTestRequest(BaseModel):
    """Per-question answers : a list of indices into the question's choice list."""
    test_id: str                         # the UUID returned by /start
    responses: list[int]                 # responses[i] = choice index for question i
    duration_s: int | None = None


# ── Question generation ─────────────────────────────────────────────────

_QUIZ_PROMPT_FR = """Tu es un examinateur de cours. Génère exactement {n} questions à choix multiples qui testent la maîtrise des concepts essentiels du cours suivant.

Concepts couverts : {concepts}

CONTRAINTES :
- 4 choix par question, exactement UNE bonne réponse.
- Niveau "compréhension" (pas de simple mémorisation, pas de calcul lourd).
- Langage clair, pas de jargon non défini par le cours.
- Réponse strictement en JSON valide, schéma exact :
{{"questions": [{{"q": "...", "choices": ["A", "B", "C", "D"], "correct_idx": 0}}, ...]}}

Pas de markdown, pas de commentaire en dehors du JSON."""


_QUIZ_PROMPT_EN = """You are a course examiner. Generate exactly {n} multiple-choice questions that test mastery of the essential concepts of the following course.

Concepts covered: {concepts}

CONSTRAINTS:
- 4 choices per question, exactly ONE correct answer.
- "Understanding" level (no rote memorization, no heavy calculation).
- Clear language, no jargon not defined by the course.
- Reply STRICT JSON only, exact schema :
{{"questions": [{{"q": "...", "choices": ["A", "B", "C", "D"], "correct_idx": 0}}, ...]}}

No markdown, no commentary outside the JSON."""


def _extract_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start: end + 1])
    except json.JSONDecodeError:
        return None


async def _generate_questions(
    course_id: str,
    language: str,
    n: int,
) -> list[dict[str, Any]]:
    """Build a list of MCQ questions for ``course_id`` using the LLM.

    Returns a list shaped ``[{"q": str, "choices": [str, ...], "correct_idx": int}, ...]``.
    Raises ``HTTPException(503)`` if the LLM is unreachable AND no fallback
    can be produced.
    """
    # Pull a short list of concept names from the KG.
    concepts: list[str] = []
    try:
        from deps import get_rag
        from pedagogy.knowledge_graph import get_or_build
        kg = get_or_build(get_rag())
        concept_infos = list(kg.list_concepts(course_id=course_id, limit=20))
        concepts = [c.canonical_name or c.display_name or c.name
                    for c in concept_infos]
    except Exception as exc:                                                # noqa: BLE001
        log.warning("learning_gain : KG unavailable (%s) — generic prompt", exc)

    if not concepts:
        # Without KG we still try — the LLM has the course material in
        # its training conversation history, but quality drops.
        concepts = ["the course's main concepts"]

    template = _QUIZ_PROMPT_FR if (language or "fr").startswith("fr") else _QUIZ_PROMPT_EN
    prompt = template.format(n=n, concepts=", ".join(concepts[:15]))

    # Use the project's LLMRouter — handles OpenAI/Ollama selection
    # and the DISABLE_OPENAI kill-switch for free.
    try:
        from ai.llm_router import get_default_router
        router_llm = get_default_router()
        # Run the (sync) router invoke in a thread so we don't block
        # the event loop on Mistral CPU calls.
        raw = await asyncio.to_thread(
            router_llm.invoke,
            prompt, prefer="openai", temperature=0.0, max_tokens=1500,
        )
    except Exception as exc:                                                # noqa: BLE001
        log.error("learning_gain : LLM call raised (%s)", exc)
        raise HTTPException(status_code=503, detail="LLM unavailable")

    if not raw:
        raise HTTPException(status_code=503, detail="LLM returned empty output")

    payload = _extract_json(raw)
    if not payload or not isinstance(payload.get("questions"), list):
        log.warning("learning_gain : LLM JSON parse failed (raw=%r)", raw[:200])
        raise HTTPException(status_code=503, detail="LLM output unparseable")

    questions = []
    for q in payload["questions"][:n]:
        if not isinstance(q, dict):
            continue
        q_text = str(q.get("q") or q.get("question") or "").strip()
        choices = q.get("choices") or []
        correct = q.get("correct_idx")
        if not q_text or not isinstance(choices, list) or len(choices) < 2:
            continue
        if not isinstance(correct, int) or not (0 <= correct < len(choices)):
            continue
        questions.append({
            "q": q_text,
            "choices": [str(c) for c in choices],
            "correct_idx": correct,
        })

    if len(questions) < _MIN_NUM_QUESTIONS:
        raise HTTPException(
            status_code=503,
            detail=f"LLM produced only {len(questions)} valid questions (min {_MIN_NUM_QUESTIONS})",
        )
    return questions


# ── Routes ──────────────────────────────────────────────────────────────

@router.post("/quiz/{test_type}/{course_id}/start")
async def start_test(
    test_type: str,
    course_id: str,
    body: StartTestRequest | None = None,
    user: dict = Depends(get_current_user),
):
    """Generate a fresh test and return its questions + a tracking id.

    ``test_type`` must be ``pretest`` or ``posttest``. The questions are
    stored in memory keyed by ``test_id`` until the student submits ;
    on submission, the row is persisted to ``learning_gain_tests``.
    """
    if test_type not in ("pretest", "posttest"):
        raise HTTPException(status_code=400, detail="test_type must be 'pretest' or 'posttest'")

    try:
        sid = _uuid.UUID(user["sub"])
        cid = _uuid.UUID(course_id)
    except (ValueError, TypeError, KeyError):
        raise HTTPException(status_code=400, detail="invalid student / course UUID")

    n = (body.num_questions if body else None) or _DEFAULT_NUM_QUESTIONS
    n = max(_MIN_NUM_QUESTIONS, min(_MAX_NUM_QUESTIONS, n))

    # Look up course language so the LLM produces the test in the right tongue.
    language = "fr"
    try:
        from database.init_db import AsyncSessionLocal
        from database.models import Course
        async with AsyncSessionLocal() as db:
            course = (await db.execute(
                select(Course).where(Course.id == cid)
            )).scalar_one_or_none()
            if course and course.language:
                language = course.language
            if course is None:
                raise HTTPException(status_code=404, detail="course not found")
    except HTTPException:
        raise
    except Exception as exc:                                                # noqa: BLE001
        log.warning("learning_gain : course lookup failed (%s)", exc)

    questions = await _generate_questions(str(cid), language, n)
    test_id = str(_uuid.uuid4())

    # Stash the answer key + question text in an in-process cache keyed
    # by test_id. The student sees questions WITHOUT correct_idx via the
    # "client view" returned below ; submission passes back the
    # ``test_id`` and we re-pair with the stored answers to score.
    _PENDING_TESTS[test_id] = {
        "student_id": str(sid),
        "course_id": str(cid),
        "test_type": test_type,
        "questions": questions,
        "language": language,
        "started_at": datetime.utcnow().isoformat(),
    }

    log.info(
        "📝 %s started : student=%s course=%s test_id=%s n_questions=%d",
        test_type, str(sid)[:8], str(cid)[:8], test_id[:8], len(questions),
    )

    # Return questions WITHOUT the correct answer (security : don't leak
    # the key to the client until they've submitted).
    return {
        "test_id": test_id,
        "test_type": test_type,
        "course_id": str(cid),
        "language": language,
        "questions": [
            {"q": q["q"], "choices": q["choices"]} for q in questions
        ],
    }


@router.post("/quiz/{test_type}/{course_id}/submit")
async def submit_test(
    test_type: str,
    course_id: str,
    body: SubmitTestRequest,
    user: dict = Depends(get_current_user),
):
    """Score the test, persist it, return the score + per-question feedback."""
    if test_type not in ("pretest", "posttest"):
        raise HTTPException(status_code=400, detail="test_type must be 'pretest' or 'posttest'")

    pending = _PENDING_TESTS.pop(body.test_id, None)
    if pending is None:
        raise HTTPException(status_code=404, detail="test_id not found or already submitted")
    if pending["test_type"] != test_type or pending["course_id"] != course_id:
        raise HTTPException(status_code=400, detail="test_type / course_id mismatch")
    if pending["student_id"] != user.get("sub"):
        raise HTTPException(status_code=403, detail="this test was started by another student")

    questions = pending["questions"]
    feedback: list[dict[str, Any]] = []
    correct_count = 0
    for i, q in enumerate(questions):
        chosen = body.responses[i] if i < len(body.responses) else -1
        is_correct = (chosen == q["correct_idx"])
        if is_correct:
            correct_count += 1
        feedback.append({
            "q_idx": i,
            "chosen_idx": chosen,
            "correct_idx": q["correct_idx"],
            "correct": is_correct,
        })
    score = correct_count / len(questions) if questions else 0.0

    # Persist the row.
    try:
        from database.init_db import AsyncSessionLocal
        from database.models import LearningGainTest
        async with AsyncSessionLocal() as db:
            row = LearningGainTest(
                student_id=_uuid.UUID(pending["student_id"]),
                course_id=_uuid.UUID(pending["course_id"]),
                test_type=test_type,
                questions=questions,
                responses=feedback,
                score=score,
                duration_s=body.duration_s,
            )
            db.add(row)
            await db.commit()
            persisted_id = str(row.id)
    except Exception as exc:                                                # noqa: BLE001
        log.error("learning_gain : DB persist failed (%s)", exc)
        raise HTTPException(status_code=500, detail="DB persist failed")

    log.info(
        "📝 %s submitted : student=%s course=%s score=%.2f (%d/%d) row=%s",
        test_type, pending["student_id"][:8], pending["course_id"][:8],
        score, correct_count, len(questions), persisted_id[:8],
    )
    return {
        "test_id": persisted_id,
        "test_type": test_type,
        "course_id": course_id,
        "score": round(score, 3),
        "correct_count": correct_count,
        "total": len(questions),
        "feedback": feedback,
    }


@router.get("/quiz/{course_id}/gain")
async def compute_gain(
    course_id: str,
    user: dict = Depends(get_current_user),
):
    """Return Hake's normalized learning gain for the calling student.

    Picks the most recent pretest and the most recent posttest for the
    (student, course) pair. Returns ``gain=None`` when one of them is
    missing — caller decides how to display.

    Formula : g = (post - pre) / (1 - pre).
    Edge cases :
      - pre == 1.0 → student already mastered ; gain undefined → None
      - missing one of the two tests → None
    """
    try:
        sid = _uuid.UUID(user["sub"])
        cid = _uuid.UUID(course_id)
    except (ValueError, TypeError, KeyError):
        raise HTTPException(status_code=400, detail="invalid student / course UUID")

    try:
        from database.init_db import AsyncSessionLocal
        from database.models import LearningGainTest
        async with AsyncSessionLocal() as db:
            pretest = (await db.execute(
                select(LearningGainTest).where(
                    LearningGainTest.student_id == sid,
                    LearningGainTest.course_id == cid,
                    LearningGainTest.test_type == "pretest",
                ).order_by(desc(LearningGainTest.taken_at)).limit(1)
            )).scalar_one_or_none()
            posttest = (await db.execute(
                select(LearningGainTest).where(
                    LearningGainTest.student_id == sid,
                    LearningGainTest.course_id == cid,
                    LearningGainTest.test_type == "posttest",
                ).order_by(desc(LearningGainTest.taken_at)).limit(1)
            )).scalar_one_or_none()
    except Exception as exc:                                                # noqa: BLE001
        log.error("learning_gain : DB read failed (%s)", exc)
        raise HTTPException(status_code=500, detail="DB read failed")

    pre_score = float(pretest.score) if pretest else None
    post_score = float(posttest.score) if posttest else None
    gain: float | None = None
    if pre_score is not None and post_score is not None:
        if pre_score < 1.0:
            gain = round((post_score - pre_score) / (1.0 - pre_score), 3)
        # else : pre = 1 → already perfect, gain undefined

    return {
        "student_id": str(sid),
        "course_id": str(cid),
        "pretest": {
            "score": pre_score,
            "taken_at": pretest.taken_at.isoformat() if pretest else None,
        } if pretest else None,
        "posttest": {
            "score": post_score,
            "taken_at": posttest.taken_at.isoformat() if posttest else None,
        } if posttest else None,
        "learning_gain": gain,
        "interpretation": _interpret_gain(gain),
    }


def _interpret_gain(g: float | None) -> str:
    """Hake's qualitative buckets — used in physics-education research."""
    if g is None:
        return "incomplete (missing one or both tests)"
    if g < 0:
        return "negative (student regressed — investigate test design)"
    if g < 0.30:
        return "low gain"
    if g < 0.70:
        return "medium gain"
    return "high gain"


# ── In-process cache for unsubmitted tests ─────────────────────────────
# Keyed by test_id. Only holds tests that were generated but not yet
# submitted. Cleared by the submission handler on success. Lost on
# server restart — that's intentional ; an unsubmitted test is throwaway
# state that doesn't deserve disk persistence.
_PENDING_TESTS: dict[str, dict[str, Any]] = {}
