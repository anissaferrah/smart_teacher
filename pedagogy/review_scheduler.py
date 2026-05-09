"""Spaced Repetition — FSRS scheduler integration.

Stage 3 : `concept_id` (UUID FK -> concept_kg) → `concept_name` (string,
= ConceptInfo.name dans le KnowledgeGraph). ReviewQueue stocke
concept_name; les concepts sont resolus via le KG in-memory.

Pour chaque (student, concept_name), on tient un Card FSRS dans la table
review_queue. A chaque pratique / Q&A interaction, on appelle
FSRS.review_card(card, rating) qui retourne le prochain due_date.

Ratings (FSRS) :
  Again : reponse fausse (re-show vite)
  Hard  : reponse correcte avec hints
  Good  : reponse correcte, normale
  Easy  : reponse parfaite, instant
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional, Any

from sqlalchemy import select

from database.init_db import AsyncSessionLocal
from database.models import ReviewQueue

log = logging.getLogger("SmartTeacher.ReviewScheduler")


def _coerce_uuid(value) -> Optional[uuid.UUID]:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


def _get_scheduler():
    """Returns a singleton FSRS Scheduler (fresh = default params)."""
    from fsrs import Scheduler
    return Scheduler()


def _serialize_card(card) -> dict[str, Any]:
    """fsrs Card → JSON dict via to_dict() if available, sinon manual."""
    try:
        return card.to_dict()
    except Exception:
        return {
            "due":        card.due.isoformat() if hasattr(card, "due") and card.due else None,
            "stability":  getattr(card, "stability", 0.0),
            "difficulty": getattr(card, "difficulty", 0.5),
            "state":      str(getattr(card, "state", "learning")),
            "step":       getattr(card, "step", 0),
        }


def _deserialize_card(data: dict | None):
    """Reconstruit un Card FSRS depuis JSON, ou crée un nouveau si data invalide."""
    from fsrs import Card
    if not data or not isinstance(data, dict):
        return Card()
    try:
        return Card.from_dict(data)
    except Exception:
        return Card()


class ReviewScheduler:
    """Wrapper async sur la table review_queue + FSRS engine.

    Tous les concepts sont identifies par leur `name` (string snake_case).
    """

    @staticmethod
    async def update_after_practice(
        student_id,
        concept_name: str,
        rating_str: str,   # "again" | "hard" | "good" | "easy"
        session_id: str | None = None,
    ) -> Optional[datetime]:
        """Hook a appeler apres chaque practice/Q&A pour scheduler la prochaine review.

        Returns le datetime de prochaine review, ou None.
        """
        from fsrs import Rating
        sid = _coerce_uuid(student_id)
        if not sid or not concept_name:
            return None

        rating_map = {
            "again": Rating.Again,
            "hard":  Rating.Hard,
            "good":  Rating.Good,
            "easy":  Rating.Easy,
        }
        rating = rating_map.get(rating_str.lower(), Rating.Good)

        try:
            async with AsyncSessionLocal() as db:
                row = (await db.execute(
                    select(ReviewQueue).where(
                        ReviewQueue.student_id == sid,
                        ReviewQueue.concept_name == concept_name,
                    )
                )).scalar_one_or_none()

                scheduler = _get_scheduler()
                card = _deserialize_card(row.fsrs_state if row else None)

                # FSRS review : retourne (new_card, review_log)
                new_card, _log = scheduler.review_card(card, rating)
                next_due = getattr(new_card, "due", None)

                serialized = _serialize_card(new_card)

                if row is None:
                    db.add(ReviewQueue(
                        student_id=sid,
                        concept_name=concept_name,
                        fsrs_state=serialized,
                        due=next_due or datetime.utcnow(),
                        last_review=datetime.utcnow(),
                        review_count=1,
                        lapse_count=1 if rating == Rating.Again else 0,
                        state=str(getattr(new_card, "state", "learning")),
                        stability=float(getattr(new_card, "stability", 0.0)),
                        difficulty=float(getattr(new_card, "difficulty", 0.5)),
                    ))
                else:
                    row.fsrs_state = serialized
                    row.due = next_due or datetime.utcnow()
                    row.last_review = datetime.utcnow()
                    row.review_count = (row.review_count or 0) + 1
                    if rating == Rating.Again:
                        row.lapse_count = (row.lapse_count or 0) + 1
                    row.state = str(getattr(new_card, "state", row.state))
                    row.stability = float(getattr(new_card, "stability", row.stability))
                    row.difficulty = float(getattr(new_card, "difficulty", row.difficulty))

                await db.commit()
                # Detail log : show the FSRS state evolution so the
                # operator understands WHY the next_due is what it is.
                _interval_days = (
                    (next_due - datetime.utcnow()).total_seconds() / 86400.0
                    if next_due else 0.0
                )
                log.info(
                    "🔍 FSRS UPDATE | concept='%s' rating=%s | "
                    "stability=%.2f difficulty=%.2f state=%s | "
                    "review_count=%d lapses=%d | next_due=%s (in %.1f days)",
                    concept_name[:40], rating_str,
                    float(getattr(new_card, "stability", 0.0)),
                    float(getattr(new_card, "difficulty", 0.5)),
                    str(getattr(new_card, "state", "?")),
                    int(getattr(new_card, "review_count", 0) or 0)
                        if hasattr(new_card, "review_count")
                        else (1 if row is None else (row.review_count or 0)),
                    row.lapse_count if row else (1 if rating == Rating.Again else 0),
                    next_due.isoformat() if next_due else "none",
                    _interval_days,
                )
                log.info(
                    f"📅 Review scheduled: student={sid} concept='{concept_name}' "
                    f"rating={rating_str} next_due={next_due}"
                )

                # Notify bandit of FSRS outcome for delayed reward
                if session_id:
                    try:
                        from pedagogy.personalization.bandit.controller import BanditController
                        stability_before = float(getattr(card, "stability", 0.0))
                        stability_after  = float(getattr(new_card, "stability", 0.0))
                        _bandit_ctrl = BanditController()
                        await _bandit_ctrl.on_concept_reviewed(
                            session_id=session_id,
                            concept_name=concept_name,
                            stability_before=stability_before,
                            stability_after=stability_after,
                            student_id=str(sid) if sid else None,
                        )
                    except Exception as exc:                              # noqa: BLE001
                        log.debug("bandit delayed reward hook failed (non-fatal): %s", exc)

                return next_due
        except Exception as exc:
            log.warning(f"update_after_practice failed: {exc}")
            return None

    @staticmethod
    async def list_due(student_id, course_id=None, limit: int = 20) -> list[dict[str, Any]]:
        """Returns concepts dont le due_date est <= now (i.e. to review).

        Avant : SQL JOIN sur ConceptKG. Maintenant : ReviewQueue par concept_name,
        et on resout le display name via le KnowledgeGraph (in-memory).
        course_id filter applique en post-process via KG.get_concept(name).course_id.
        """
        sid = _coerce_uuid(student_id)
        if not sid:
            return []
        try:
            async with AsyncSessionLocal() as db:
                rows = (await db.execute(
                    select(ReviewQueue)
                    .where(
                        ReviewQueue.student_id == sid,
                        ReviewQueue.due <= datetime.utcnow(),
                    )
                    .order_by(ReviewQueue.due.asc())
                    .limit(limit * 3)   # over-fetch, on filtre par course apres
                )).scalars().all()

            # Resoudre concept names + filtrer par course via KG
            try:
                from deps import get_rag
                from pedagogy.knowledge_graph import get_or_build
                kg = get_or_build(get_rag())
            except Exception:
                kg = None

            course_id_str = str(course_id) if course_id else None
            out = []
            for rq in rows:
                concept_label = rq.concept_name
                concept_display = rq.concept_name
                concept_course = None
                if kg is not None:
                    ci = kg.get_concept(rq.concept_name)
                    if ci:
                        concept_label = ci.name
                        concept_display = ci.canonical_name or ci.display_name or ci.name
                        concept_course = ci.course_id
                # Course filter
                if course_id_str and concept_course and concept_course != course_id_str:
                    continue
                out.append({
                    "concept_name":  concept_label,
                    "concept_label": concept_label,    # alias retro-compat
                    "concept_display": concept_display,
                    "due":           rq.due.isoformat() if rq.due else None,
                    "state":         rq.state,
                    "stability":     round(rq.stability or 0.0, 2),
                    "difficulty":    round(rq.difficulty or 0.5, 2),
                    "review_count":  rq.review_count,
                    "lapse_count":   rq.lapse_count,
                })
                if len(out) >= limit:
                    break
            return out
        except Exception as exc:
            log.warning(f"list_due failed: {exc}")
            return []
