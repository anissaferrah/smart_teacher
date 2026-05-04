"""Practice Engine — generate/grade practice questions par concept.

Stage 3 : concepts identifies par `concept_name` (string, = ConceptInfo.name
dans le KnowledgeGraph). Avant : `concept_id` UUID FK vers la table
concept_kg (retiree).

Tables (apres migration Stage 3) :
  - practice_question (concept_name, question, answer, difficulty, hints[])
  - practice_attempt  (student_id, question_id, answer_text, is_correct, hints_used)

Mastery loop :
  attempt.is_correct=True → record_clean (Beta posterior +1 success)
  attempt.is_correct=False → record_confusion (Beta posterior +1 failure)
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Optional, Any

from sqlalchemy import select

from database.init_db import AsyncSessionLocal
from database.models import PracticeQuestion, PracticeAttempt
from pedagogy.mastery_repo import MasteryRepo

log = logging.getLogger("SmartTeacher.PracticeEngine")


def _coerce_uuid(value) -> Optional[uuid.UUID]:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


def _strip_json_response(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
        raw = raw.rstrip("`").strip()
    s = raw.find("[")
    e = raw.rfind("]")
    if s != -1 and e > s:
        return raw[s:e + 1]
    s = raw.find("{")
    e = raw.rfind("}")
    if s != -1 and e > s:
        return raw[s:e + 1]
    return raw


def _call_llm(prompt: str, max_tokens: int = 1200) -> str | None:
    """Delegue au LLMRouter unifie (Ollama-prio anti-quota pour la generation
    de questions, qui peut etre massive sur un cours entier)."""
    from ai.llm_router import get_default_router
    return get_default_router().invoke(
        prompt, prefer="ollama", temperature=0.2, max_tokens=max_tokens,
    )


def _get_kg():
    """Recupere le KnowledgeGraph singleton (lazy)."""
    from deps import get_rag
    from pedagogy.knowledge_graph import get_or_build
    return get_or_build(get_rag())


def _ensure_concept_in_kg(concept_name: str) -> Optional[Any]:
    """Lookup un concept dans le KG, avec lazy extraction si pas trouve.

    Si on connait un course_id (via une idee deja indexee), on declenche
    l'extraction. Sinon on retourne None — le caller decide quoi faire.
    """
    from deps import get_rag
    from pedagogy.knowledge_graph import ensure_concepts_loaded
    kg = _get_kg()
    ci = kg.get_concept(concept_name)
    if ci is not None:
        return ci
    # Tenter d'extraire pour TOUS les cours connus du RAG (fast path).
    # Heuristique : on essaye d'abord les cours qui ont le plus de docs.
    rag = get_rag()
    course_doc_counts: dict[str, int] = {}
    for d in (getattr(rag, "all_docs", None) or []):
        cid = (getattr(d, "metadata", None) or {}).get("course")
        if cid:
            course_doc_counts[cid] = course_doc_counts.get(cid, 0) + 1
    for cid, _n in sorted(course_doc_counts.items(), key=lambda kv: -kv[1])[:5]:
        ensure_concepts_loaded(rag, cid, enrich=False)
        ci = kg.get_concept(concept_name)
        if ci is not None:
            return ci
    return None


class PracticeEngine:
    """Generate, fetch and grade practice questions per concept."""

    @staticmethod
    async def generate_for_concept(concept_name: str, lang: str = "fr") -> int:
        """Generate 3 questions (easy/medium/hard) for a concept and persist them.

        Idempotent : si questions existent deja, return count sans regenerer.
        Returns nb questions persisted.
        """
        if not concept_name or not isinstance(concept_name, str):
            return 0
        concept_name = concept_name.strip()
        if not concept_name:
            return 0

        async with AsyncSessionLocal() as db:
            existing = (await db.execute(
                select(PracticeQuestion.id).where(
                    PracticeQuestion.concept_name == concept_name
                )
            )).all()
            if existing:
                return len(existing)

            # Lookup ConceptInfo via KnowledgeGraph (in-memory, plus de DB).
            # Bootstrap auto si pas encore extrait pour le cours.
            try:
                concept = _ensure_concept_in_kg(concept_name)
            except Exception as exc:
                log.warning(f"generate_for_concept: KG unavailable ({exc})")
                return 0
            if not concept:
                log.warning(f"generate_for_concept: concept '{concept_name}' not in KG (after bootstrap)")
                return 0

            display = concept.canonical_name or concept.display_name or concept.name
            description = concept.description or "(no description)"

            if lang == "fr":
                prompt = (
                    f"Tu es un tuteur pedagogique. Génère 3 questions d'entrainement sur le concept :\n"
                    f"  Concept : {display}\n"
                    f"  Description : {description}\n\n"
                    "Trois niveaux : easy (memorisation), medium (comprehension/application), "
                    "hard (analyse/synthese).\n\n"
                    "Pour chaque question, fournis aussi 3 hints progressifs :\n"
                    "  hint1 = rappel conceptuel\n"
                    "  hint2 = methode/piste\n"
                    "  hint3 = solution partielle (sans la reponse complete)\n\n"
                    "Réponds en JSON STRICT (liste de 3 items, sans markdown) :\n"
                    "[\n"
                    '  {"difficulty": "easy", "question": "...", "answer": "...", '
                    '"hints": ["hint1", "hint2", "hint3"]},\n'
                    "  {...medium...},\n  {...hard...}\n]"
                )
            else:
                prompt = (
                    f"You are a pedagogical tutor. Generate 3 practice questions on the concept:\n"
                    f"  Concept: {display}\n"
                    f"  Description: {description}\n\n"
                    "Three levels: easy (memorization), medium (understanding/application), "
                    "hard (analysis/synthesis).\n\n"
                    "For each question, also provide 3 progressive hints:\n"
                    "  hint1 = conceptual reminder\n  hint2 = method/clue\n"
                    "  hint3 = partial solution (without giving the full answer)\n\n"
                    "Reply STRICT JSON (list of 3 items, no markdown):\n"
                    "[\n"
                    '  {"difficulty": "easy", "question": "...", "answer": "...", '
                    '"hints": ["hint1", "hint2", "hint3"]},\n'
                    "  {...medium...},\n  {...hard...}\n]"
                )

            raw = _call_llm(prompt, max_tokens=1500)
            if not raw:
                log.warning(f"LLM unavailable to generate questions for '{concept_name}'")
                return 0

            try:
                items = json.loads(_strip_json_response(raw))
                if not isinstance(items, list):
                    return 0
            except json.JSONDecodeError as exc:
                log.warning(f"generate_for_concept JSON parse error: {exc}")
                return 0

            count = 0
            for item in items[:3]:
                if not isinstance(item, dict):
                    continue
                question = (item.get("question") or "").strip()
                answer = (item.get("answer") or "").strip()
                if not question or not answer:
                    continue
                difficulty = (item.get("difficulty") or "medium").strip().lower()
                if difficulty not in {"easy", "medium", "hard"}:
                    difficulty = "medium"
                hints = item.get("hints") or []
                if not isinstance(hints, list):
                    hints = []
                hints = [str(h)[:300] for h in hints[:3]]

                db.add(PracticeQuestion(
                    concept_name=concept_name,
                    question=question[:2000],
                    answer=answer[:2000],
                    difficulty=difficulty,
                    hints=hints,
                    language=lang[:5],
                ))
                count += 1
            await db.commit()
            log.info(f"PracticeEngine: generated {count} questions for concept '{concept_name}'")
            return count

    @staticmethod
    async def get_question(concept_name: str, difficulty: str | None = None) -> dict[str, Any] | None:
        """Pick a question for the concept (random within difficulty if specified)."""
        if not concept_name:
            return None
        async with AsyncSessionLocal() as db:
            stmt = select(PracticeQuestion).where(PracticeQuestion.concept_name == concept_name)
            if difficulty in {"easy", "medium", "hard"}:
                stmt = stmt.where(PracticeQuestion.difficulty == difficulty)
            rows = (await db.execute(stmt)).scalars().all()
            if not rows:
                return None
            import random
            q = random.choice(rows)
            return {
                "question_id":  str(q.id),
                "concept_name": q.concept_name,
                "question":     q.question,
                "difficulty":   q.difficulty,
                "hints":        q.hints or [],
                # answer NOT included (server-side only)
            }

    @staticmethod
    def _llm_grade(
        question: str, expected: str, user: str, lang: str = "fr"
    ) -> dict[str, Any] | None:
        """LLM-based semantic grading. Returns {"correct": bool, "score": 0-1, "feedback": str, "missing": [...]}."""
        if not user or not expected:
            return None
        if lang == "fr":
            prompt = (
                "Tu es un examinateur indulgent mais juste. Compare la réponse de l'élève "
                "à la réponse attendue. Évalue si l'élève a compris le concept (acceptes "
                "des paraphrases, synonymes, formulations différentes).\n\n"
                f"QUESTION : {question}\n"
                f"RÉPONSE ATTENDUE : {expected}\n"
                f"RÉPONSE ÉLÈVE : {user}\n\n"
                "Réponds UNIQUEMENT en JSON STRICT (sans markdown) :\n"
                '{"correct": true|false, "score": 0.0-1.0, "feedback": "<1-2 phrases bienveillantes>", '
                '"missing_points": ["<point manquant>"]}'
            )
        else:
            prompt = (
                "You are a fair but lenient examiner. Compare the student's answer "
                "to the expected one. Judge if they grasped the concept (accept paraphrases).\n\n"
                f"QUESTION: {question}\n"
                f"EXPECTED: {expected}\n"
                f"STUDENT: {user}\n\n"
                "Reply STRICT JSON only:\n"
                '{"correct": true|false, "score": 0.0-1.0, "feedback": "<1-2 kind sentences>", '
                '"missing_points": ["<missing point>"]}'
            )

        raw = _call_llm(prompt, max_tokens=400)
        if not raw:
            return None
        try:
            data = json.loads(_strip_json_response(raw))
            if not isinstance(data, dict):
                return None
            return {
                "correct":         bool(data.get("correct", False)),
                "score":           float(data.get("score", 0.0)),
                "feedback":        str(data.get("feedback", ""))[:500],
                "missing_points":  [str(m)[:200] for m in (data.get("missing_points") or [])][:5],
            }
        except Exception as exc:
            log.debug(f"_llm_grade parse failed: {exc}")
            return None

    @staticmethod
    async def submit_answer(
        question_id,
        student_id,
        answer_text: str,
        hints_used: int = 0,
        time_taken_s: float = 0.0,
    ) -> dict[str, Any]:
        """Grade an answer (LLM semantic + token fallback) + persist attempt +
        update mastery + schedule FSRS review."""
        qid = _coerce_uuid(question_id)
        sid = _coerce_uuid(student_id)
        if not qid or not sid:
            return {"error": "invalid id"}

        async with AsyncSessionLocal() as db:
            q = (await db.execute(
                select(PracticeQuestion).where(PracticeQuestion.id == qid)
            )).scalar_one_or_none()
            if not q:
                return {"error": "question not found"}

            # ── LLM grading sémantique (priorité) ─────────────────────────
            llm_result = PracticeEngine._llm_grade(
                question=q.question, expected=q.answer, user=answer_text or "",
                lang=q.language or "fr",
            )

            is_correct = False
            score = 0.0
            feedback = ""
            missing_points: list[str] = []
            grading_method = "token"

            if llm_result:
                is_correct = llm_result["correct"]
                score = max(0.0, min(1.0, llm_result["score"]))
                feedback = llm_result["feedback"]
                missing_points = llm_result["missing_points"]
                grading_method = "llm"
            else:
                # ── Fallback : token overlap (LLM down) ────────────────────
                user_norm = re.sub(r"\s+", " ", answer_text.lower().strip())
                expected_norm = re.sub(r"\s+", " ", q.answer.lower().strip())
                if user_norm and expected_norm:
                    if expected_norm in user_norm or user_norm in expected_norm:
                        is_correct = True
                        score = 1.0
                    else:
                        user_tokens = set(re.findall(r"\w{3,}", user_norm))
                        expected_tokens = set(re.findall(r"\w{3,}", expected_norm))
                        if expected_tokens:
                            overlap = len(user_tokens & expected_tokens) / len(expected_tokens)
                            is_correct = overlap >= 0.7
                            score = overlap

            # ── Persist attempt ─────────────────────────────────────────────
            db.add(PracticeAttempt(
                student_id=sid,
                question_id=qid,
                answer_text=(answer_text or "")[:2000],
                is_correct=is_correct,
                hints_used=int(hints_used),
                time_taken_s=float(time_taken_s),
            ))
            await db.commit()

            # ── Mastery update via KG (concept lookup par name) ──────────────
            try:
                kg = _get_kg()
                concept = kg.get_concept(q.concept_name)
            except Exception:
                concept = None

            new_mastery = None
            if concept:
                # Mastery update is a Bayesian Beta posterior — each practice
                # attempt is a single Bernoulli observation. course_id depuis
                # le ConceptInfo (champ ajoute au KG en Stage 1).
                course_id = concept.course_id or None
                if is_correct:
                    new_mastery = await MasteryRepo.record_clean(
                        sid, course_id, concept.name,
                    )
                else:
                    new_mastery = await MasteryRepo.record_confusion(
                        sid, course_id, concept.name,
                    )

            # ── Spaced repetition (FSRS) hook ───────────────────────────────
            next_review = None
            if concept:
                if not is_correct:
                    rating = "again"
                elif hints_used >= 3 or score < 0.7:
                    rating = "hard"
                elif hints_used == 0 and score >= 0.95:
                    rating = "easy"
                else:
                    rating = "good"
                try:
                    from pedagogy.review_scheduler import ReviewScheduler
                    next_review = await ReviewScheduler.update_after_practice(
                        sid, concept.name, rating
                    )
                except Exception as exc:
                    log.debug(f"review schedule failed: {exc}")

            return {
                "question_id":       str(qid),
                "is_correct":        is_correct,
                "score":             round(score, 2),
                "feedback":          feedback,
                "missing_points":    missing_points,
                "grading_method":    grading_method,
                "expected_answer":   q.answer if not is_correct else None,
                "hints_used":        int(hints_used),
                "new_mastery_score": new_mastery,
                "concept_name":      concept.name if concept else q.concept_name,
                "next_review_at":    next_review.isoformat() if next_review else None,
            }
