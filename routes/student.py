"""Student profile, recommendations, reviews-due, skill-tree, profile reset."""

import logging
import uuid as _uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

import deps
from handlers.auth import get_current_user

router = APIRouter()
log = logging.getLogger("SmartTeacher.routes.student")


# ── Endpoints "me" (authenticated student looking at own data) ────────

@router.get("/student/me/profile")
async def get_my_profile(user: dict = Depends(get_current_user)):
    """L'étudiant authentifié voit son profil complet."""
    return await get_student_profile_full(user["sub"])


@router.get("/student/me/learning-style")
async def get_my_learning_style(user: dict = Depends(get_current_user)):
    """Calcule le profil d'apprentissage à la volée + adaptation prompt."""
    from pedagogy.personalization.learning_style.heuristic import compute_learning_style, style_to_prompt_hint
    sid = _uuid.UUID(user["sub"])
    scores = await compute_learning_style(sid)
    return {
        "scores": scores.to_dict(),
        "prompt_hint": style_to_prompt_hint(scores.dominant()),
    }


@router.post("/student/me/learning-style/recompute")
async def recompute_my_learning_style(user: dict = Depends(get_current_user)):
    """Force la recomputation + persistance dans StudentProfile."""
    from pedagogy.personalization.learning_style.heuristic import update_student_learning_style
    sid = _uuid.UUID(user["sub"])
    scores = await update_student_learning_style(sid)
    if scores is None:
        return {"status": "insufficient_data", "message": "Pas encore assez d'interactions pour calculer"}
    return {"status": "updated", "scores": scores.to_dict()}


# ── Bayesian learning style endpoints (M2 grade) ──────────────────────

@router.get("/student/me/learning-style/bayes")
async def get_my_bayes_posterior(user: dict = Depends(get_current_user)):
    """Posterior bayésien complet : means + alpha + HDI 95% + confidence."""
    from pedagogy.personalization.learning_style.bayes import load_posterior
    sid = _uuid.UUID(user["sub"])
    posterior = await load_posterior(sid)
    return posterior.to_dict()


@router.post("/student/me/learning-style/bayes/rebuild")
async def rebuild_my_bayes_posterior(
    user: dict = Depends(get_current_user), days: int = 90,
):
    """Reconstruit la posterior depuis l'historique DB (audit / reset)."""
    from pedagogy.personalization.learning_style.bayes import rebuild_posterior_from_history
    sid = _uuid.UUID(user["sub"])
    posterior = await rebuild_posterior_from_history(sid, days=days)
    return posterior.to_dict()


@router.get("/student/me/learning-style/cross-validate")
async def my_cross_validation(user: dict = Depends(get_current_user)):
    """Compare comportement observé vs VARK self-report — concordance."""
    from pedagogy.personalization.learning_style.bayes import cross_validate
    sid = _uuid.UUID(user["sub"])
    result = await cross_validate(sid)
    if result is None:
        return {"status": "no_data", "message": "Self-report VARK manquant ou < 5 interactions"}
    return {
        "status":               "ok",
        "behavioral_dominant":  result.behavioral_dominant,
        "self_report_dominant": result.self_report_dominant,
        "agreement":            result.agreement,
        "cosine_similarity":    result.cosine_similarity,
        "behavioral_scores":    result.behavioral_scores,
        "self_report_scores":   result.self_report_scores,
    }


# ── VARK questionnaire endpoints ──────────────────────────────────────

@router.get("/vark/questionnaire")
async def get_vark_questionnaire(lang: str = "fr"):
    """Liste les 8 questions VARK pour affichage frontend (pas d'auth requis)."""
    from pedagogy.personalization.learning_style.vark import serialize_for_frontend
    return {"questions": serialize_for_frontend(lang=lang), "lang": lang}


@router.post("/student/me/vark/submit")
async def submit_vark_responses(
    responses: dict,        # {q_id: [style_code, ...]}
    user: dict = Depends(get_current_user),
):
    """Reçoit les réponses VARK → seed le prior Bayes + persist dans profile."""
    from pedagogy.personalization.learning_style.vark import score_responses
    from pedagogy.personalization.learning_style.bayes import (
        DEFAULT_PRIOR_ALPHA, PosteriorEstimate, load_posterior, save_posterior, STYLES,
    )

    sid = _uuid.UUID(user["sub"])
    counts = score_responses(responses or {})

    # Capture previous VARK self-report (if re-taking the test) so we can
    # subtract its contribution from alpha — otherwise repeated submissions
    # would inflate the seed unboundedly.
    from sqlalchemy import select
    from database.init_db import AsyncSessionLocal
    from database.models import StudentProfile
    prev_vark: dict[str, int] = {}
    async with AsyncSessionLocal() as db:
        profile = (await db.execute(
            select(StudentProfile).where(StudentProfile.student_id == sid)
        )).scalar_one_or_none()
        if profile is None:
            from datetime import datetime
            profile = StudentProfile(student_id=sid, created_at=datetime.utcnow())
            db.add(profile)
        if isinstance(profile.preferences, dict):
            prev_vark = dict(profile.preferences.get("vark_self_report") or {})
        prefs = dict(profile.preferences or {})
        prefs["vark_self_report"] = counts
        profile.preferences = prefs
        await db.commit()

    # Update prior: existing alpha − 0.5 × prev_vark + 0.5 × counts.
    # Preserve n_observations so re-taking VARK doesn't wipe behavioural data.
    existing = await load_posterior(sid)
    new_alpha = list(existing.alpha)
    for i, dim in enumerate(STYLES):
        new_alpha[i] -= 0.5 * prev_vark.get(dim, 0)
        new_alpha[i] += 0.5 * counts.get(dim, 0)
        # Floor at the default prior so we never go below the uninformative baseline
        new_alpha[i] = max(DEFAULT_PRIOR_ALPHA[i], new_alpha[i])
    posterior = PosteriorEstimate(
        alpha=tuple(new_alpha),
        n_observations=existing.n_observations,
    )
    await save_posterior(sid, posterior)

    return {
        "status":         "ok",
        "vark_counts":    counts,
        "seeded_alpha":   list(posterior.alpha),
        "dominant":       posterior.dominant(),
        "message":        "Prior bayésien initialisé. Continue à utiliser Smart Teacher pour affiner.",
    }


@router.post("/student/{student_id}/profile/reset")
async def reset_student_profile(student_id: str):
    """Remet à zéro le profil d'un étudiant."""
    from pedagogy.personalization.profile import StudentProfile
    profile = StudentProfile(student_id=student_id)
    await deps.get_profile_mgr().save(profile)
    return {"status": "reset", "student_id": student_id}


@router.get("/student/{student_id}/recommend-next")
async def recommend_next_concept(
    student_id: str,
    course_id: str,
    chapter_idx: int | None = None,
    top_k: int = 3,
):
    """Recommande les top_k prochains concepts à apprendre."""
    from pedagogy.path_recommender import PathRecommender
    try:
        recs = await PathRecommender.recommend_next(
            student_id=student_id,
            course_id=course_id,
            current_chapter_idx=chapter_idx,
            top_k=top_k,
        )
        return {
            "student_id": student_id,
            "course_id": course_id,
            "recommendations": [r.to_dict() for r in recs],
            "count": len(recs),
        }
    except Exception as exc:
        log.exception("recommend_next failed")
        raise HTTPException(status_code=500, detail=f"Recommender error: {exc}")


@router.get("/student/{student_id}/profile")
async def get_student_profile_full(student_id: str):
    """Profile aggregé : profile + stats mastery + reviews + activité récente."""
    from sqlalchemy import select, func, desc
    from database.init_db import AsyncSessionLocal
    from database.models import (
        Student, StudentProfile, StudentMastery,
        PracticeAttempt, PracticeQuestion, ReviewQueue, LearningSession,
    )
    from pedagogy.personalization.engine import PersonalizationEngine

    try:
        sid = _uuid.UUID(student_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="invalid student_id (UUID required)")

    try:
        async with AsyncSessionLocal() as db:
            profile = (await db.execute(
                select(StudentProfile).where(StudentProfile.student_id == sid)
            )).scalar_one_or_none()
            student = (await db.execute(
                select(Student).where(Student.id == sid)
            )).scalar_one_or_none()

            mastery_rows = (await db.execute(
                select(StudentMastery).where(StudentMastery.student_id == sid)
            )).scalars().all()

            mastered_count = sum(1 for m in mastery_rows if (m.score or 0) >= 0.85)
            in_progress_count = sum(1 for m in mastery_rows if 0 < (m.score or 0) < 0.85)
            total_attempts_in_mastery = sum((m.attempts or 0) for m in mastery_rows)
            total_confusions = sum((m.confusions or 0) for m in mastery_rows)
            avg_mastery = (
                sum(m.score or 0 for m in mastery_rows) / len(mastery_rows)
                if mastery_rows else 0.0
            )

            course_stats: dict[str, dict] = {}
            for m in mastery_rows:
                cid = str(m.course_id) if m.course_id else "unknown"
                if cid not in course_stats:
                    course_stats[cid] = {"course_id": cid, "mastered": 0, "in_progress": 0, "total": 0, "avg_score": 0.0, "score_sum": 0.0}
                s = course_stats[cid]
                s["total"] += 1
                s["score_sum"] += float(m.score or 0)
                if (m.score or 0) >= 0.85:
                    s["mastered"] += 1
                elif (m.score or 0) > 0:
                    s["in_progress"] += 1
            for s in course_stats.values():
                s["avg_score"] = round(s["score_sum"] / max(1, s["total"]), 3)
                del s["score_sum"]

            reviews_due_count = (await db.execute(
                select(func.count(ReviewQueue.id)).where(
                    ReviewQueue.student_id == sid,
                    ReviewQueue.due <= datetime.utcnow(),
                )
            )).scalar() or 0

            # Stage 3 : ConceptKG retire — concepts identifies par concept_name
            # (string), display via KnowledgeGraph in-memory.
            recent_attempts = (await db.execute(
                select(PracticeAttempt, PracticeQuestion)
                .join(PracticeQuestion, PracticeAttempt.question_id == PracticeQuestion.id)
                .where(PracticeAttempt.student_id == sid)
                .order_by(desc(PracticeAttempt.created_at))
                .limit(10)
            )).all()

            try:
                from deps import get_rag
                from pedagogy.knowledge_graph import get_or_build
                kg = get_or_build(get_rag())
            except Exception:
                kg = None

            def _concept_display(name: str) -> str:
                if kg is None:
                    return name
                ci = kg.get_concept(name)
                return (ci.canonical_name or ci.display_name or ci.name) if ci else name

            recent_activity = [{
                "concept_label": q.concept_name,
                "concept_name":  _concept_display(q.concept_name),
                "is_correct":    bool(a.is_correct),
                "hints_used":    int(a.hints_used or 0),
                "time_taken_s":  round(float(a.time_taken_s or 0), 1),
                "answered_at":   a.created_at.isoformat() if a.created_at else None,
                "question_preview": (q.question[:80] + "…") if len(q.question) > 80 else q.question,
            } for a, q in recent_attempts]

            pers_ctx = await PersonalizationEngine.get_context(sid)

            session_count = (await db.execute(
                select(func.count(LearningSession.id)).where(LearningSession.student_id == str(sid))
            )).scalar() or 0

            return {
                "student_id": student_id,
                "name": getattr(student, "name", None) or getattr(student, "username", None) or "Étudiant",
                "profile": {
                    "learning_style": (profile.learning_style if profile else "visual"),
                    "preferred_difficulty": (profile.preferred_difficulty if profile else "intermediate"),
                    "pace": (profile.pace if profile else "normal"),
                    "preferred_explanation_depth": (profile.preferred_explanation_depth if profile else "balanced"),
                    "avg_response_time_s": round(float(profile.avg_response_time_s or 0), 1) if profile else 0.0,
                    "confusion_rate": round(float(profile.confusion_rate or 0), 3) if profile else 0.0,
                    "topics_of_interest": (profile.topics_of_interest if profile else []),
                },
                "personalization_runtime": pers_ctx.to_dict(),
                "gamification": {
                    "total_xp": int(profile.total_xp or 0) if profile else 0,
                    "streak_days": int(profile.streak_days or 0) if profile else 0,
                    "level": min(100, int((profile.total_xp or 0) / 100) + 1) if profile else 1,
                    "last_activity": (
                        profile.last_activity.isoformat()
                        if profile and profile.last_activity else None
                    ),
                },
                "stats": {
                    "concepts_mastered": mastered_count,
                    "concepts_in_progress": in_progress_count,
                    "concepts_tracked": len(mastery_rows),
                    "avg_mastery_score": round(avg_mastery, 3),
                    "total_attempts": total_attempts_in_mastery,
                    "total_confusions": total_confusions,
                    "reviews_due_now": int(reviews_due_count),
                    "session_count": int(session_count),
                },
                "courses_progression": list(course_stats.values()),
                "recent_activity": recent_activity,
            }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("get_student_profile failed")
        raise HTTPException(status_code=500, detail=f"Profile error: {exc}")


@router.get("/student/{student_id}/reviews-due")
async def get_reviews_due(student_id: str, course_id: str | None = None, limit: int = 20):
    """Spaced Repetition — concepts à réviser maintenant (FSRS due_date <= now)."""
    from pedagogy.review_scheduler import ReviewScheduler
    try:
        items = await ReviewScheduler.list_due(student_id, course_id=course_id, limit=limit)
        return {"student_id": student_id, "count": len(items), "reviews": items}
    except Exception as exc:
        log.exception("reviews-due failed")
        raise HTTPException(status_code=500, detail=f"ReviewScheduler error: {exc}")


@router.get("/student/{student_id}/skill-tree")
async def get_skill_tree(student_id: str, course_id: str):
    """Skill tree coloré par mastery state pour visualisation Cytoscape."""
    from pedagogy.skill_tree import SkillTreeBuilder
    try:
        tree = await SkillTreeBuilder.build(student_id=student_id, course_id=course_id)
        if not tree.get("nodes"):
            raise HTTPException(
                status_code=404,
                detail="No concept graph in DB for this course — call /course/{id}/concept-graph first",
            )
        return tree
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("skill_tree failed")
        raise HTTPException(status_code=500, detail=f"SkillTree error: {exc}")


# ── Course report — rapport de fin de cours pour l'eleve ───────────────

@router.get("/student/me/course-report")
async def my_course_report(course_id: str, user: dict = Depends(get_current_user)):
    """Rapport synthetique pour la fin de cours (vue eleve).

    Agrege :
      - mastery distribution (mastered/in_progress/not_started)
      - completion_pct + avg_score
      - top_strengths : 3 concepts les mieux maitrises
      - top_weaknesses : 3 concepts a reviser en priorite
      - upcoming_reviews : prochain reviews FSRS dus
      - time_spent_min : temps total estime (depuis learning_events)
      - badges : achievements mineurs (motivationnels)

    Tous calcules avec la propagation backward (Feature #1) — un concept
    boost-e via ses dependants compte comme maitrise.
    """
    return await build_course_report(user["sub"], course_id)


@router.get("/student/{student_id}/course-report")
async def get_course_report(student_id: str, course_id: str):
    """Variante admin/teacher : rapport pour un autre eleve."""
    return await build_course_report(student_id, course_id)


async def build_course_report(student_id: str, course_id: str) -> dict:
    """Construit le rapport agrege. Logique partagee /me et /{student_id}."""
    from sqlalchemy import select, func, desc
    from database.init_db import AsyncSessionLocal
    from database.models import (
        StudentMastery, ReviewQueue, LearningEvent, Course,
    )
    from pedagogy.mastery_repo import MasteryRepo, MASTERY_THRESHOLD

    try:
        sid = _uuid.UUID(student_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "invalid student_id")
    try:
        cid = _uuid.UUID(course_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "invalid course_id")

    # KG pour propagation + nom des concepts
    try:
        from pedagogy.knowledge_graph import get_or_build, ensure_concepts_loaded
        rag = deps.get_rag()
        kg = get_or_build(rag)
        concepts = kg.list_concepts(course_id)
        if not concepts:
            concepts = ensure_concepts_loaded(rag, course_id, enrich=False)
    except Exception as exc:
        log.warning(f"course-report: KG unavailable: {exc}")
        kg = None
        concepts = []

    async with AsyncSessionLocal() as db:
        # Course metadata
        course = (await db.execute(
            select(Course).where(Course.id == cid)
        )).scalar_one_or_none()

        # Mastery rows pour ce cours
        mastery_rows = (await db.execute(
            select(StudentMastery).where(
                StudentMastery.student_id == sid,
                StudentMastery.course_id == cid,
            )
        )).scalars().all()

        # Reviews dues
        reviews_due_count = (await db.execute(
            select(func.count(ReviewQueue.id)).where(
                ReviewQueue.student_id == sid,
                ReviewQueue.due <= datetime.utcnow(),
            )
        )).scalar() or 0

        # Time spent : somme des total_time depuis learning_events
        time_spent_s = (await db.execute(
            select(func.coalesce(func.sum(LearningEvent.total_time), 0.0)).where(
                LearningEvent.student_id == sid,
                LearningEvent.course_id == cid,
            )
        )).scalar() or 0.0

        # Confusion rate sur ce cours
        n_events = (await db.execute(
            select(func.count(LearningEvent.id)).where(
                LearningEvent.student_id == sid,
                LearningEvent.course_id == cid,
            )
        )).scalar() or 0
        n_confused = (await db.execute(
            select(func.count(LearningEvent.id)).where(
                LearningEvent.student_id == sid,
                LearningEvent.course_id == cid,
                LearningEvent.confusion_score >= 0.5,
            )
        )).scalar() or 0

    # Compute boosted scores per concept (via propagation backward)
    concept_scores: dict[str, float] = {}
    if concepts and kg is not None:
        all_idea_ids = set()
        for c in concepts:
            all_idea_ids.update(c.idea_ids)
        if all_idea_ids:
            try:
                idea_scores = await MasteryRepo.get_scores_bulk_with_propagation(
                    sid, course_id, list(all_idea_ids), kg=kg,
                )
                # Score concept = moyenne des ideas qu'il couvre
                for c in concepts:
                    if not c.idea_ids:
                        concept_scores[c.name] = 0.0
                        continue
                    cs = [idea_scores.get(iid, 0.0) for iid in c.idea_ids]
                    concept_scores[c.name] = sum(cs) / len(cs) if cs else 0.0
            except Exception as exc:
                log.debug(f"concept_scores compute failed: {exc}")

    # Distribution
    mastered = sum(1 for s in concept_scores.values() if s >= MASTERY_THRESHOLD)
    in_progress = sum(1 for s in concept_scores.values() if 0 < s < MASTERY_THRESHOLD)
    not_started = sum(1 for s in concept_scores.values() if s <= 0)
    total_concepts = len(concept_scores) or 1

    completion_pct = round(100 * mastered / total_concepts, 1)
    avg_score = (
        round(sum(concept_scores.values()) / total_concepts, 3)
        if concept_scores else 0.0
    )

    # Top strengths / weaknesses
    sorted_concepts = sorted(concept_scores.items(), key=lambda kv: -kv[1])
    strengths_raw = sorted_concepts[:3]
    weaknesses_raw = sorted(
        [(n, s) for n, s in concept_scores.items() if s < MASTERY_THRESHOLD],
        key=lambda kv: kv[1],
    )[:3]

    def _display(name: str) -> str:
        if kg is None:
            return name
        ci = kg.get_concept(name)
        return (ci.canonical_name or ci.display_name or ci.name) if ci else name

    top_strengths = [
        {"concept": n, "display": _display(n), "score": round(s, 3)}
        for n, s in strengths_raw
    ]
    top_weaknesses = [
        {"concept": n, "display": _display(n), "score": round(s, 3)}
        for n, s in weaknesses_raw
    ]

    # Badges (motivationnels — purement cosmetiques)
    badges: list[dict] = []
    if completion_pct >= 100:
        badges.append({"id": "graduate", "label": "🎓 Cours complete"})
    elif completion_pct >= 80:
        badges.append({"id": "near_done", "label": "🌟 Presque fini (80%+)"})
    if mastered >= 5:
        badges.append({"id": "5_concepts", "label": f"⭐ {mastered} concepts maitrises"})
    if (n_events > 0) and (n_confused / max(1, n_events)) < 0.1:
        badges.append({"id": "smooth", "label": "🎯 Excellent (<10% confusion)"})

    # Status pedagogique
    if completion_pct >= 100:
        status = "completed"
    elif completion_pct >= 70:
        status = "near_completion"
    elif completion_pct >= 30:
        status = "in_progress"
    else:
        status = "started"

    return {
        "student_id": str(sid),
        "course_id": str(cid),
        "course_title": course.title if course else "(unknown)",
        "status": status,
        "completion_pct": completion_pct,
        "avg_score": avg_score,
        "mastery": {
            "total":       total_concepts,
            "mastered":    mastered,
            "in_progress": in_progress,
            "not_started": not_started,
        },
        "top_strengths":  top_strengths,
        "top_weaknesses": top_weaknesses,
        "reviews_due":    int(reviews_due_count),
        "time_spent_min": round(float(time_spent_s) / 60, 1),
        "confusion_rate": round(n_confused / max(1, n_events), 3),
        "n_interactions": int(n_events),
        "badges":         badges,
        "next_action":    _suggest_next_action(status, completion_pct, top_weaknesses),
    }


def _suggest_next_action(status: str, completion_pct: float, weaknesses: list[dict]) -> dict:
    """Recommande l'action suivante pour l'eleve, base sur l'etat global."""
    if status == "completed":
        return {
            "type": "celebrate",
            "title": "🎉 Cours terminé !",
            "body": "Tous les concepts sont maîtrisés. Faites quelques révisions FSRS pour ancrer.",
            "cta_label": "Voir mes révisions dues",
            "cta_url": "/static/profile.html",
        }
    if weaknesses:
        weakest = weaknesses[0]
        return {
            "type": "review_weak",
            "title": f"Concentre-toi sur \"{weakest['display']}\"",
            "body": f"C'est ton point le plus faible (score {weakest['score']:.2f}). "
                    f"Un exercice ciblé fera la différence.",
            "cta_label": "Pratiquer ce concept",
            "cta_url":   f"/static/quiz.html?concept={weakest['concept']}",
        }
    return {
        "type": "continue",
        "title": "Continue ton apprentissage",
        "body": "Bonne progression. Continue les leçons pour progresser.",
        "cta_label": "Reprendre le cours",
        "cta_url":   "/static/index.html",
    }
