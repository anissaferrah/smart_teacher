"""Admin/teacher dashboard endpoints — view all students, learning data, timeline."""

import logging
import uuid as _uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, func, select

from database.init_db import AsyncSessionLocal
from database.models import (
    Interaction, LearningEvent, LearningSession,
    PracticeAttempt, PracticeQuestion, ReviewQueue,
    Student, StudentMastery, StudentProfile,
)
from handlers.auth import require_admin

router = APIRouter(prefix="/admin")
log = logging.getLogger("SmartTeacher.routes.admin")


# ── GET /admin/students — liste compacte de tous les étudiants ────────

@router.get("/students")
async def list_students(
    user: dict = Depends(require_admin),
    limit: int = 100,
    offset: int = 0,
    search: str = "",
):
    """Liste tous les étudiants avec stats agrégées (pour tableau admin)."""
    async with AsyncSessionLocal() as db:
        # Base query — tous les étudiants actifs
        stmt = select(Student).order_by(desc(Student.created_at))
        if search:
            like = f"%{search.lower()}%"
            stmt = stmt.where(
                (func.lower(Student.email).like(like)) |
                (func.lower(Student.first_name).like(like)) |
                (func.lower(Student.last_name).like(like))
            )
        stmt = stmt.limit(limit).offset(offset)
        students = (await db.execute(stmt)).scalars().all()

        # Stats par étudiant — un round-trip groupé
        items = []
        for s in students:
            mastery_rows = (await db.execute(
                select(StudentMastery).where(StudentMastery.student_id == s.id)
            )).scalars().all()
            mastered = sum(1 for m in mastery_rows if (m.score or 0) >= 0.85)
            in_progress = sum(1 for m in mastery_rows if 0 < (m.score or 0) < 0.85)
            avg = (sum(m.score or 0 for m in mastery_rows) / len(mastery_rows)
                   if mastery_rows else 0.0)

            session_count = (await db.execute(
                select(func.count(LearningSession.id))
                .where(LearningSession.student_id == str(s.id))
            )).scalar() or 0

            interaction_count = (await db.execute(
                select(func.count(Interaction.id))
                .where(Interaction.student_id == str(s.id))
            )).scalar() or 0

            profile = (await db.execute(
                select(StudentProfile).where(StudentProfile.student_id == s.id)
            )).scalar_one_or_none()

            items.append({
                "student_id":         str(s.id),
                "email":              s.email,
                "name":               f"{s.first_name} {s.last_name or ''}".strip(),
                "account_level":      s.account_level,
                "preferred_language": s.preferred_language,
                "is_active":          bool(s.is_active),
                "created_at":         s.created_at.isoformat() if s.created_at else None,
                "stats": {
                    "concepts_mastered":   mastered,
                    "concepts_in_progress": in_progress,
                    "concepts_tracked":    len(mastery_rows),
                    "avg_mastery":         round(avg, 3),
                    "sessions":            int(session_count),
                    "interactions":        int(interaction_count),
                },
                "profile_brief": {
                    "learning_style":    profile.learning_style if profile else "unknown",
                    "pace":              profile.pace if profile else "normal",
                    "level":             s.account_level,
                    "total_xp":          int(profile.total_xp or 0) if profile else 0,
                    "streak_days":       int(profile.streak_days or 0) if profile else 0,
                    "confusion_rate":    round(float(profile.confusion_rate or 0), 3) if profile else 0.0,
                } if profile else None,
            })

        total = (await db.execute(
            select(func.count(Student.id))
        )).scalar() or 0

    return {
        "total":   int(total),
        "limit":   limit,
        "offset":  offset,
        "search":  search,
        "items":   items,
    }


# ── GET /admin/students/{id}/timeline — historique complet d'un étudiant ──

@router.get("/students/{student_id}/timeline")
async def student_timeline(
    student_id: str,
    user: dict = Depends(require_admin),
    days: int = 30,
    limit: int = 200,
):
    """Timeline complète : sessions + interactions + practice + reviews + mastery."""
    try:
        sid = _uuid.UUID(student_id)
    except (ValueError, TypeError):
        raise HTTPException(400, "invalid student_id")

    since = datetime.utcnow() - timedelta(days=days)

    async with AsyncSessionLocal() as db:
        # Compte
        student = (await db.execute(
            select(Student).where(Student.id == sid)
        )).scalar_one_or_none()
        if not student:
            raise HTTPException(404, "Student not found")

        # Sessions
        sessions = (await db.execute(
            select(LearningSession)
            .where(LearningSession.student_id == str(sid))
            .where(LearningSession.created_at >= since)
            .order_by(desc(LearningSession.created_at))
            .limit(limit)
        )).scalars().all()

        # Interactions (Q&A)
        interactions = (await db.execute(
            select(Interaction)
            .where(Interaction.student_id == str(sid))
            .where(Interaction.created_at >= since)
            .order_by(desc(Interaction.created_at))
            .limit(limit)
        )).scalars().all()

        # Practice attempts (concept name lookup via KnowledgeGraph)
        # Avant : join SQL sur ConceptKG (table en cours de retrait Stage 3).
        # Maintenant : on resout les concept names via le KG en post-process.
        practice_attempts = (await db.execute(
            select(PracticeAttempt, PracticeQuestion)
            .join(PracticeQuestion, PracticeAttempt.question_id == PracticeQuestion.id)
            .where(PracticeAttempt.student_id == sid)
            .where(PracticeAttempt.created_at >= since)
            .order_by(desc(PracticeAttempt.created_at))
            .limit(limit)
        )).all()

        # Mastery courant — joint sur idea_id (et non concept_id, qui n'existait
        # pas sur StudentMastery — bug pre-existant de l'ancienne version).
        # Le concept name est resolu via kg.concepts_of_idea(idea_id) en post-process.
        mastery_rows = (await db.execute(
            select(StudentMastery)
            .where(StudentMastery.student_id == sid)
            .order_by(desc(StudentMastery.score))
        )).scalars().all()

        # Resoudre concept names via le KG (in-memory, pas de DB)
        try:
            from deps import get_rag
            from pedagogy.knowledge_graph import get_or_build
            kg = get_or_build(get_rag())
        except Exception:
            kg = None

        # Stage 3 : PracticeQuestion.concept_name (string) — display via KG.
        practice_rows = []
        for (a, q) in practice_attempts:
            concept_label = q.concept_name or "(concept inconnu)"
            if kg is not None:
                ci = kg.get_concept(q.concept_name)
                if ci:
                    concept_label = ci.canonical_name or ci.display_name or ci.name
            practice_rows.append((a, q, concept_label))

        # Pour mastery : idea_id -> concepts qui le contiennent via kg.concepts_of_idea
        mastery = []
        for m in mastery_rows:
            concept_label = "(idea_id: " + str(m.idea_id)[:12] + ")"
            if kg is not None:
                cis = kg.concepts_of_idea(str(m.idea_id))
                if cis:
                    # Premier concept : suffisant pour l'affichage timeline
                    ci = cis[0]
                    concept_label = ci.canonical_name or ci.display_name or ci.name
            mastery.append((m, concept_label))

        # Learning events (riches : confusion, prosody, etc.)
        events = (await db.execute(
            select(LearningEvent)
            .where(LearningEvent.student_id == str(sid))
            .where(LearningEvent.created_at >= since)
            .order_by(desc(LearningEvent.created_at))
            .limit(limit)
        )).scalars().all()

        # Reviews dues
        reviews_due = (await db.execute(
            select(func.count(ReviewQueue.id)).where(
                ReviewQueue.student_id == sid,
                ReviewQueue.due <= datetime.utcnow(),
            )
        )).scalar() or 0

    return {
        "student": {
            "id":         str(student.id),
            "email":      student.email,
            "name":       f"{student.first_name} {student.last_name or ''}".strip(),
            "account_level": student.account_level,
            "language":   student.preferred_language,
            "created_at": student.created_at.isoformat() if student.created_at else None,
        },
        "window_days": days,
        "summary": {
            "sessions":        len(sessions),
            "interactions":    len(interactions),
            "practice":        len(practice_rows),
            "events":          len(events),
            "concepts_tracked": len(mastery),
            "reviews_due":     int(reviews_due),
        },
        "sessions": [{
            "id":             str(s.id),
            "course_id":      str(s.course_id) if s.course_id else None,
            "language":       s.language,
            "level":          s.level,
            "state":          s.state,
            "chapter_index":  s.chapter_index,
            "section_index":  s.section_index,
            "created_at":     s.created_at.isoformat() if s.created_at else None,
        } for s in sessions],
        "interactions": [{
            "type":         i.type,
            "question":     i.question[:200] if i.question else "",
            "answer":       i.answer[:300] if i.answer else "",
            "language":     i.language,
            "stt_time":     float(i.stt_time or 0),
            "llm_time":     float(i.llm_time or 0),
            "tts_time":     float(i.tts_time or 0),
            "total_time":   float(i.total_time or 0),
            "kpi_ok":       int(i.kpi_ok or 0),
            "created_at":   i.created_at.isoformat() if i.created_at else None,
        } for i in interactions],
        "practice": [{
            "concept":       label,
            "is_correct":    bool(a.is_correct),
            "hints_used":    int(a.hints_used or 0),
            "time_taken_s":  round(float(a.time_taken_s or 0), 1),
            "answered_at":   a.created_at.isoformat() if a.created_at else None,
        } for (a, q, label) in practice_rows],
        "mastery": [{
            "concept":       label,
            "course_id":     str(m.course_id) if m.course_id else None,
            "score":         round(float(m.score or 0), 3),
            "attempts":      int(m.attempts or 0),
            "confusions":    int(m.confusions or 0),
            "last_seen":     m.updated_at.isoformat() if m.updated_at else None,
            "state":         "mastered" if (m.score or 0) >= 0.85
                              else "in_progress" if (m.score or 0) > 0
                              else "not_started",
        } for (m, label) in mastery],
        "events_recent": [{
            "event_type":    e.event_type,
            "concept":       e.concept,
            "action_taken":  e.action_taken,
            "confusion_score": float(e.confusion_score or 0),
            "reward":        float(e.reward or 0),
            "created_at":    e.created_at.isoformat() if e.created_at else None,
        } for e in events[:50]],
    }


# ── GET /admin/students/{id}/full — tout en un appel ──────────────────

@router.get("/students/{student_id}/full")
async def student_full_profile(
    student_id: str,
    user: dict = Depends(require_admin),
):
    """Profile complet : student card + profile + mastery + timeline last 7d.

    Réutilise l'endpoint /student/{id}/profile public + /admin/timeline.
    """
    from routes.student import get_student_profile_full
    profile = await get_student_profile_full(student_id)
    timeline = await student_timeline(student_id, user=user, days=7, limit=50)
    return {"profile": profile, "timeline": timeline}


# ── GET /admin/learning-styles — distribution ─────────────────────────

@router.get("/learning-styles")
async def learning_styles_distribution(user: dict = Depends(require_admin)):
    """Distribution des learning_style sur tous les profils — utile pour heatmap admin."""
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(StudentProfile.learning_style, func.count(StudentProfile.id))
            .group_by(StudentProfile.learning_style)
        )).all()
    return {
        "distribution": [{"style": r[0] or "unknown", "count": int(r[1])} for r in rows],
        "total": sum(int(r[1]) for r in rows),
    }


# ── GET /admin/resilience — circuit breakers + recent timeouts ────────

@router.get("/resilience")
async def resilience_status(user: dict = Depends(require_admin)):
    """Live state of node circuit breakers + recent timeout events.

    Used by ops to detect a degrading LLM endpoint. A node in `open` state
    means the breaker is bypassing it; in `half_open` we're probing recovery.
    """
    from agentic.resilience import BreakerRegistry
    breakers = BreakerRegistry.all_stats()

    # Recent events sink is attached to the KPI tracker by the wrap layer.
    events: list[dict] = []
    try:
        from observability.kpi_logger import KPITracker
        tracker = KPITracker.get()
        events = list(getattr(tracker, "_resilience_events", []) or [])
    except Exception:
        pass
    # Keep last 100 events most-recent-first
    events = list(reversed(events))[:100]

    # Roll-up: per-node count of {ok, timeout, error, circuit_open, deadline_expired}
    rollup: dict[str, dict[str, int]] = {}
    for ev in events:
        node = ev.get("node") or "?"
        kind = ev.get("kind") or "?"
        rollup.setdefault(node, {})
        rollup[node][kind] = rollup[node].get(kind, 0) + 1

    return {
        "breakers": breakers,
        "rollup":   rollup,
        "events":   events,
    }


# ── GET /admin/cache/stats — learning-style cache hit/miss counters ───

@router.get("/cache/stats")
async def cache_stats(user: dict = Depends(require_admin)):
    """Hit / miss / single-flight counters for the 2-level learning-style cache.

    Useful to confirm the cache is doing its job. After warm-up:
      hit_ratio = hits_l1 / (hits_l1 + hits_l2 + misses) should be > 0.9
    """
    from services.learning_style_cache import get_stats
    s = get_stats()
    total = s["hits_l1"] + s["hits_l2"] + s["misses"]
    hit_ratio = (s["hits_l1"] + s["hits_l2"]) / total if total > 0 else 0.0
    l1_ratio  = s["hits_l1"] / total if total > 0 else 0.0
    return {
        "counters":  s,
        "total":     total,
        "hit_ratio": round(hit_ratio, 3),
        "l1_ratio":  round(l1_ratio, 3),
    }


# ── KG concept-cache admin (P4 persistence) ──────────────────────────

@router.delete("/kg-cache/{course_id}")
async def kg_cache_invalidate(course_id: str, user: dict = Depends(require_admin)):
    """Drop the persisted concept cache for one course.

    Use after re-ingesting that course so the next concept query
    re-extracts from scratch instead of serving stale concepts.
    """
    from pedagogy.knowledge_graph import persistence
    persistence.invalidate_course(course_id)
    return {"status": "ok", "course_id": course_id, "action": "invalidated"}


@router.delete("/kg-cache")
async def kg_cache_clear_all(user: dict = Depends(require_admin)):
    """Drop the entire persisted concept cache (all courses)."""
    from pedagogy.knowledge_graph import persistence
    persistence.invalidate_all()
    return {"status": "ok", "action": "all_invalidated"}


# ── GET /admin/bandit/stats — personalization bandit observability ────

@router.get("/bandit/stats")
async def bandit_stats(user: dict = Depends(require_admin)):
    """Per-(context_bucket, arm) statistics for the Phase-1 personalization bandit.

    Returns :
      - aggregate counters (n_buckets, total_pulls)
      - per-bucket breakdown : top arm by posterior mean + n_pulls per arm
      - per-arm aggregate (across all buckets) : how often each strategy
        / speech-rate is selected globally

    Useful for :
      - confirming the bandit has accumulated enough samples (per-bucket
        n_pulls > ~30 before drawing conclusions)
      - identifying which strategies dominate per student profile
      - detecting starvation : if some arms are never picked, exploration
        may be insufficient
    """
    from collections import defaultdict
    from pedagogy.personalization.bandit import all_actions
    from pedagogy.personalization.bandit.repo import get_bandit
    from pedagogy.personalization.bandit.strategies import StrategyAction
    from pedagogy.personalization.bandit.thompson import ContextBucket

    bandit = await get_bandit()

    # Per-bucket aggregation
    buckets: dict[str, dict] = defaultdict(lambda: {
        "n_pulls": 0,
        "n_arms_explored": 0,
        "arms": {},
    })
    # Per-arm global aggregation
    arms_global: dict[str, dict] = defaultdict(lambda: {
        "n_pulls": 0,
        "alpha_total": 0.0,
        "beta_total": 0.0,
    })

    for key, post in bandit.posteriors.items():
        # key format: "<style>|<pace>|<mastery>|<arm_id>" (3 pipes)
        parts = key.split("|")
        if len(parts) < 4:
            continue
        bucket_key = "|".join(parts[:3])
        arm_id = parts[3]

        b = buckets[bucket_key]
        b["arms"][arm_id] = {
            "alpha":   round(post.alpha, 3),
            "beta":    round(post.beta, 3),
            "mean":    round(post.mean, 3),
            "n_pulls": post.n_pulls,
        }
        b["n_pulls"] += post.n_pulls
        if post.n_pulls > 0:
            b["n_arms_explored"] += 1

        ag = arms_global[arm_id]
        ag["n_pulls"] += post.n_pulls
        ag["alpha_total"] += post.alpha
        ag["beta_total"] += post.beta

    # Compute per-bucket best arm by mean
    for bucket_key, data in buckets.items():
        if data["arms"]:
            best_arm = max(data["arms"].items(), key=lambda kv: kv[1]["mean"])
            data["best_arm"] = {"arm_id": best_arm[0], "mean": best_arm[1]["mean"]}

    # Compute per-arm aggregate mean
    for arm_id, ag in arms_global.items():
        denom = ag["alpha_total"] + ag["beta_total"]
        ag["mean_global"] = round(ag["alpha_total"] / denom, 3) if denom else 0.5

    # Coverage : ratio of (bucket, arm) pairs that have any observation
    total_action_count = sum(1 for _ in all_actions())
    pairs_with_data = sum(
        1 for post in bandit.posteriors.values() if post.n_pulls > 0
    )
    pairs_total = max(1, total_action_count * len(buckets) if buckets else 1)
    coverage = round(pairs_with_data / pairs_total, 3) if pairs_total else 0.0

    return {
        "summary": {
            "n_buckets":     len(buckets),
            "total_pulls":   bandit.total_pulls(),
            "n_action_arms": total_action_count,
            "coverage":      coverage,
        },
        "per_bucket":    dict(buckets),
        "per_arm_global": dict(arms_global),
    }


# ── GET /admin/llm/stats — observability sur les fallbacks OpenAI ↔ Ollama ──
# Avant cet endpoint, un OpenAI down depuis 3 jours se manifestait par une
# facture etrangement basse. Maintenant on voit en clair :
#   - combien de calls par backend
#   - le ratio d'echecs
#   - combien de fallbacks (le backend prefere a echoue, on a bascule)
#   - "both_failed" : les deux ont echoue, le caller a recu None

@router.get("/llm/stats")
async def llm_stats(user: dict = Depends(require_admin)):
    """Compteurs threadsafe du LLMRouter (singleton partage par concept_extractor +
    multimodal_rag). Reset = restart du process.

    Healthy state apres warm-up :
      - openai_errors / openai_calls   < 0.05
      - ollama_errors / ollama_calls   < 0.10
      - fallback_events / total_calls  < 0.10  (si > 0.5 : backend prefere flaky)
      - both_failed                    == 0    (si > 0 : alerte serieuse)
    """
    from ai.llm_router import get_default_router

    router_default = get_default_router()
    snap = router_default.stats.snapshot()

    # Stats du router prive de MultiModalRAG (si initialise)
    rag_snap = None
    try:
        from deps import get_rag
        rag = get_rag()
        if rag is not None and hasattr(rag, "_llm"):
            rag_snap = rag._llm.stats.snapshot()
            rag_snap["openai_disabled_reason"] = rag._llm.openai_disabled_reason
    except Exception as exc:
        rag_snap = {"error": str(exc)}

    total = snap["openai_calls"] + snap["ollama_calls"]
    openai_err_rate = round(snap["openai_errors"] / max(1, snap["openai_calls"]), 3)
    ollama_err_rate = round(snap["ollama_errors"] / max(1, snap["ollama_calls"]), 3)
    fallback_rate = round(snap["fallback_events"] / max(1, total), 3)

    return {
        "default_router": {
            "counters":        snap,
            "total_calls":     total,
            "openai_err_rate": openai_err_rate,
            "ollama_err_rate": ollama_err_rate,
            "fallback_rate":   fallback_rate,
            "openai_disabled_reason": router_default.openai_disabled_reason,
        },
        "rag_router": rag_snap,
    }


# ── GET /admin/metrics — KPIs pedagogiques agreges ─────────────────────
# Endpoint unifie pour piloter le projet : mastery, engagement, confusion,
# performance bandit, sante du KG. Avant : ces signaux etaient dispersés
# dans 5 endpoints differents (cache, bandit, sessions, mastery...).

@router.get("/metrics")
async def admin_metrics(
    course_id: str | None = None,
    days: int = 7,
    user: dict = Depends(require_admin),
):
    """KPIs agreges sur une fenetre temporelle.

    Args:
        course_id: filtrer sur 1 cours (None = tous).
        days:      fenetre temporelle (default 7 jours).

    Returns: dashboard JSON avec :
        - global :     n_students, n_sessions, n_events, mean_reward,
                       confusion_rate, mean_session_duration_s
        - mastery :    distribution mastered/in_progress/not_started
        - top_confusions :     top 10 concepts les plus confus
        - bandit_arms :        performance des arms (mean reward)
        - kg_health :          stats du KnowledgeGraph
    """
    from datetime import datetime, timedelta
    from sqlalchemy import func

    since = datetime.utcnow() - timedelta(days=max(1, min(days, 90)))
    cid = None
    if course_id:
        try:
            cid = _uuid.UUID(course_id)
        except Exception:
            raise HTTPException(400, "invalid course_id (UUID required)")

    out: dict = {"window_days": days, "course_id": course_id, "since": since.isoformat()}

    async with AsyncSessionLocal() as db:
        # ── 1. Global counters depuis learning_events ───────────────────
        stmt = select(LearningEvent).where(LearningEvent.created_at >= since)
        if cid is not None:
            stmt = stmt.where(LearningEvent.course_id == cid)
        events = (await db.execute(stmt)).scalars().all()

        n_events = len(events)
        n_students = len({e.student_id for e in events if e.student_id})
        n_sessions = len({e.session_id for e in events if e.session_id})
        mean_reward = sum(e.reward or 0 for e in events) / n_events if n_events else 0.0
        n_confused = sum(1 for e in events if (e.confusion_score or 0) >= 0.5)
        confusion_rate = n_confused / n_events if n_events else 0.0
        mean_total_time = sum(e.total_time or 0 for e in events) / n_events if n_events else 0.0

        out["global"] = {
            "n_students":           n_students,
            "n_sessions":           n_sessions,
            "n_events":             n_events,
            "mean_reward":          round(mean_reward, 3),
            "confusion_rate":       round(confusion_rate, 3),
            "mean_total_time_s":    round(mean_total_time, 2),
        }

        # ── 2. Mastery distribution sur la fenetre ──────────────────────
        ms_stmt = select(StudentMastery)
        if cid is not None:
            ms_stmt = ms_stmt.where(StudentMastery.course_id == cid)
        masteries = (await db.execute(ms_stmt)).scalars().all()
        mastered = sum(1 for m in masteries if (m.score or 0) >= 0.85)
        in_progress = sum(1 for m in masteries if 0 < (m.score or 0) < 0.85)
        not_started = sum(1 for m in masteries if (m.score or 0) <= 0)
        out["mastery"] = {
            "total":       len(masteries),
            "mastered":    mastered,
            "in_progress": in_progress,
            "not_started": not_started,
            "mastered_ratio": round(mastered / len(masteries), 3) if masteries else 0.0,
        }

        # ── 3. Top confusions par concept ───────────────────────────────
        from collections import Counter, defaultdict
        confusion_per_concept: dict[str, list[float]] = defaultdict(list)
        for e in events:
            if e.concept:
                confusion_per_concept[e.concept].append(float(e.confusion_score or 0))
        confusion_ranking = []
        for concept, scores in confusion_per_concept.items():
            if len(scores) < 3:   # min 3 obs to be meaningful
                continue
            avg = sum(scores) / len(scores)
            confusion_ranking.append({
                "concept": concept,
                "rate":    round(avg, 3),
                "n":       len(scores),
            })
        confusion_ranking.sort(key=lambda x: -x["rate"])
        out["top_confusions"] = confusion_ranking[:10]

        # ── 4. Bandit arm performance par action_taken ──────────────────
        arm_rewards: dict[str, list[float]] = defaultdict(list)
        for e in events:
            if e.action_taken:
                arm_rewards[e.action_taken].append(float(e.reward or 0))
        arm_perf = []
        for arm, rewards in arm_rewards.items():
            arm_perf.append({
                "arm":         arm,
                "n_pulls":     len(rewards),
                "mean_reward": round(sum(rewards) / len(rewards), 3) if rewards else 0.0,
            })
        arm_perf.sort(key=lambda x: -x["mean_reward"])
        out["bandit_arms"] = arm_perf

    # ── 5. KG health ────────────────────────────────────────────────────
    try:
        from deps import get_rag
        from pedagogy.knowledge_graph import get_or_build
        kg = get_or_build(get_rag())
        out["kg_health"] = kg.stats()
    except Exception as exc:                                              # noqa: BLE001
        out["kg_health"] = {"error": str(exc)}

    return out
