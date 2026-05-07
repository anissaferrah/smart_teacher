"""Mastery repository — persistent CRUD on StudentMastery.

Centralises operations on the ``student_mastery`` table so agents
(Retriever / Responder / Reviewer / Planner) don't touch SQLAlchemy
directly.

# Mastery scoring — Beta posterior (Laplace rule of succession)

Each (student, course, idea) is modelled as a Bernoulli outcome:
correct or confused. The mastery probability ``θ`` is the posterior
mean of a Beta(α, β) where α counts successes (correct answers) and
β counts failures (confusions), both initialised to 1 (uniform prior).

    θ_hat = α / (α + β)
          = (correct + 1) / (attempts + 2)

This is the **Laplace rule of succession** (Laplace 1814; Jaynes 2003,
*Probability Theory: The Logic of Science*, ch. 6). Properties:

  - Initial estimate (no data): 0.5 — the uniform prior
  - One success: 2/3 ≈ 0.667
  - One failure: 1/3 ≈ 0.333
  - Many successes (n=10, k=10): 11/12 ≈ 0.917
  - Asymmetry built in: a single failure pulls the estimate down more
    than a single success pulls it up, *if* the count is small. This
    is the right Bayesian behaviour without arbitrary tuning weights.

The previous version used hand-picked deltas (``+0.10`` clean, ``-0.15``
confusion) which had no justification. They've been removed.

# Mastery threshold

``MASTERY_THRESHOLD = 0.85`` is the canonical "mastered" cutoff used
across the codebase. The choice of 0.85 is the conventional cutoff
in mastery learning (Bloom 1968, *Learning for Mastery*) and is the
single source of truth — re-import here, don't redefine elsewhere.

``MIN_ATTEMPTS_FOR_MASTERY = 3`` is operational: the posterior is too
uncertain after fewer than 3 attempts even if the score crosses the
threshold by chance.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from database.init_db import AsyncSessionLocal
from database.models import StudentMastery

log = logging.getLogger("SmartTeacher.MasteryRepo")

# MASTERY_THRESHOLD : seuil de score Bayesien pour declarer un concept maitrise.
#   - Reference : Bloom (1968) Mastery Learning. La litterature ITS converge
#     sur 0.80-0.90, avec 0.85 comme valeur canonique (ALEKS, Knewton, BKT).
#   - Au-dessous de 0.85 : "in_progress" (jaune dans skill_tree).
#   - 1 seul threshold pour le projet (avant : 0.70 pour prereqs / 0.85 pour
#     concept courant — choix arbitraire abandonne, voir path_recommender:23).
#   - Si vous baissez (0.75) : recommandations plus liberales, eleves passent
#     plus vite a la suite. Si vous montez (0.95) : plus strict, plus de
#     repetitions.
MASTERY_THRESHOLD = 0.85

# MIN_ATTEMPTS_FOR_MASTERY : un score moyen >= 0.85 sur < 3 tentatives n'est
# pas statistiquement fiable (Beta(2,1) → mean 0.67, mais l'IC est tres large).
# 3 = floor minimal pour avoir une posterior raisonnablement informee.
# Cf. mastery_repo.record_clean / record_confusion qui implementent un
# Beta posterior update (alpha succes, beta echecs).
MIN_ATTEMPTS_FOR_MASTERY = 3


def _coerce_uuid(value) -> Optional[uuid.UUID]:
    """Accept str ou UUID, retourne UUID ou None."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


def _laplace_score(attempts: int, confusions: int) -> float:
    """Posterior mean under Beta(1,1) prior — Laplace rule of succession.

        θ_hat = (correct + 1) / (attempts + 2)

    Returns 0.5 (the uniform prior) when ``attempts == 0``.
    """
    correct = max(0, attempts - confusions)
    return (correct + 1) / (attempts + 2)


class MasteryRepo:
    """Repository sans state interne. Methodes async, ouvre une session SQLAlchemy a chaque appel."""

    @staticmethod
    async def get_score(student_id, course_id, idea_id: str) -> float:
        """Posterior mastery probability for this (student, course, idea).

        Computed live from ``attempts`` and ``confusions`` so the score
        always reflects the current Bayesian estimate without storing a
        tunable scalar that can drift from the underlying counts.
        """
        sid = _coerce_uuid(student_id)
        cid = _coerce_uuid(course_id)
        if not sid or not idea_id:
            return 0.5
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(StudentMastery.attempts, StudentMastery.confusions).where(
                    StudentMastery.student_id == sid,
                    StudentMastery.course_id == cid,
                    StudentMastery.idea_id == idea_id,
                )
                row = (await db.execute(stmt)).first()
                if row is None:
                    return 0.5
                attempts, confusions = int(row[0] or 0), int(row[1] or 0)
                return _laplace_score(attempts, confusions)
        except Exception as exc:
            log.debug(f"get_score failed: {exc}")
            return 0.5

    @staticmethod
    async def upsert_score(
        student_id,
        course_id,
        idea_id: str,
        is_confusion: bool = False,
    ) -> Optional[float]:
        """Record an observation (clean / confusion) and return new posterior mean.

        Increments ``attempts`` (and ``confusions`` if applicable),
        recomputes the Beta posterior, and persists ``score = θ_hat`` for
        cheap reads. Sets ``mastered_at`` when ``θ_hat >= MASTERY_THRESHOLD``
        AND ``attempts >= MIN_ATTEMPTS_FOR_MASTERY``.
        """
        sid = _coerce_uuid(student_id)
        cid = _coerce_uuid(course_id)
        if not sid or not idea_id:
            return None
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(StudentMastery).where(
                    StudentMastery.student_id == sid,
                    StudentMastery.course_id == cid,
                    StudentMastery.idea_id == idea_id,
                )
                existing = (await db.execute(stmt)).scalar_one_or_none()

                if existing is None:
                    new_attempts = 1
                    new_confusions = 1 if is_confusion else 0
                    new_score = _laplace_score(new_attempts, new_confusions)
                    db.add(StudentMastery(
                        student_id=sid,
                        course_id=cid,
                        idea_id=idea_id,
                        score=new_score,
                        attempts=new_attempts,
                        confusions=new_confusions,
                        last_seen_at=datetime.utcnow(),
                    ))
                    await db.commit()
                    return new_score

                existing.attempts = (existing.attempts or 0) + 1
                if is_confusion:
                    existing.confusions = (existing.confusions or 0) + 1
                new_score = _laplace_score(existing.attempts, existing.confusions or 0)
                existing.score = new_score
                existing.last_seen_at = datetime.utcnow()
                if (
                    new_score >= MASTERY_THRESHOLD
                    and existing.attempts >= MIN_ATTEMPTS_FOR_MASTERY
                    and existing.mastered_at is None
                ):
                    existing.mastered_at = datetime.utcnow()
                    log.info(
                        f"🎓 Mastery achieved : student={sid} course={cid} idea={idea_id} "
                        f"score={new_score:.2f} attempts={existing.attempts}"
                    )
                await db.commit()
                return new_score
        except Exception as exc:
            log.warning(f"upsert_score failed: {exc}")
            return None

    @staticmethod
    async def list_mastered(student_id, course_id, threshold: float = MASTERY_THRESHOLD) -> set[str]:
        """Set of idea_ids whose posterior mastery is ≥ ``threshold``."""
        sid = _coerce_uuid(student_id)
        cid = _coerce_uuid(course_id)
        if not sid:
            return set()
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(StudentMastery.idea_id).where(
                    StudentMastery.student_id == sid,
                    StudentMastery.course_id == cid,
                    StudentMastery.score >= threshold,
                )
                rows = await db.execute(stmt)
                return {r[0] for r in rows.all()}
        except Exception as exc:
            log.debug(f"list_mastered failed: {exc}")
            return set()

    @staticmethod
    async def get_scores_bulk(student_id, course_id, idea_ids: list[str]) -> dict[str, float]:
        """Bulk lookup : returns {idea_id: posterior_mean}.

        Computed from stored ``attempts``/``confusions`` so the scores stay
        consistent with the underlying observation counts.
        """
        sid = _coerce_uuid(student_id)
        cid = _coerce_uuid(course_id)
        if not sid or not idea_ids:
            return {}
        try:
            async with AsyncSessionLocal() as db:
                stmt = select(
                    StudentMastery.idea_id,
                    StudentMastery.attempts,
                    StudentMastery.confusions,
                ).where(
                    StudentMastery.student_id == sid,
                    StudentMastery.course_id == cid,
                    StudentMastery.idea_id.in_(idea_ids),
                )
                rows = await db.execute(stmt)
                return {
                    r[0]: _laplace_score(int(r[1] or 0), int(r[2] or 0))
                    for r in rows.all()
                }
        except Exception as exc:
            log.debug(f"get_scores_bulk failed: {exc}")
            return {}

    @staticmethod
    async def record_confusion(student_id, course_id, idea_id: str) -> Optional[float]:
        """Record a confusion observation; returns new posterior mean."""
        return await MasteryRepo.upsert_score(student_id, course_id, idea_id, is_confusion=True)

    @staticmethod
    async def record_clean(student_id, course_id, idea_id: str) -> Optional[float]:
        """Record a clean (correct) observation; returns new posterior mean."""
        return await MasteryRepo.upsert_score(student_id, course_id, idea_id, is_confusion=False)

    # ── KG-augmented mastery (backward propagation) ──────────────────────
    #
    # Pourquoi backward (et pas forward) :
    #
    #   - FORWARD : maitriser un prereq P ne PROUVE pas que l'on maitrise
    #     ses dependants D. Le signal est faible (savoir l'addition ne dit
    #     rien sur la connaissance de la multiplication).
    #
    #   - BACKWARD : maitriser un dependant D PROUVE qu'on maitrise ses
    #     prereqs P (sinon impossible de faire D). Signal STRONG.
    #
    # Application : si l'eleve a 3 dependants de X tous mastered, on peut
    # raisonnablement creditter X meme s'il n'a pas ete teste directement.
    # Cela accelere le path_recommender (skip un concept X qui sera
    # implicitement valide par ses dependants) et corrige le skill_tree
    # (X colore vert au lieu de rouge).
    #
    # Implementation : score boost calcule a la lecture, jamais persiste.
    # Les counts "vrais" (attempts, confusions) restent purs.

    # PROPAGATION_DECAY : poids cumulatif max du signal de propagation.
    # 0.20 = au mieux 20% de boost depuis les voisins. Reste 80% pour les
    # observations directes. Borne pour eviter qu'un idea jamais teste
    # ne devienne mastered uniquement par propagation (probleme epistemique).
    PROPAGATION_DECAY = 0.20

    @staticmethod
    async def get_score_with_propagation(
        student_id,
        course_id,
        idea_id: str,
        kg=None,
    ) -> float:
        """Score Beta posterior + bonus retro-propagation depuis les dependants.

        Si `kg` est None, equivalent a `get_score`. Sinon, ajoute jusqu'a
        PROPAGATION_DECAY * mean_dependents_score au score direct.

        Returns: score borne dans [0, 1].
        """
        base = await MasteryRepo.get_score(student_id, course_id, idea_id)
        if kg is None or not idea_id:
            return base

        # Trouver les dependants directs (concepts qui requirent cette idee)
        try:
            dependents = kg.dependents_of(idea_id)
        except Exception:
            return base
        if not dependents:
            return base

        dep_ids = [d.idea_id for d in dependents]
        dep_scores = await MasteryRepo.get_scores_bulk(student_id, course_id, dep_ids)
        if not dep_scores:
            return base

        # Moyenne des dependants connus (default 0.5 = uninformative)
        mean_dep = sum(dep_scores.values()) / len(dep_scores)

        # Bonus = decay * (mean_dep - 0.5), capping a [0, decay]
        # Logique : si mean_dep == 0.5 (no info), pas de bonus.
        # Si mean_dep == 1.0 (perfect mastery aval), bonus max.
        # Si mean_dep == 0.0 (toujours faux aval), bonus negatif minore a 0
        # (on ne PUNIT pas X parce que ses dependants sont rates — peut-etre
        # que les dependants sont juste plus durs).
        bonus = max(0.0, MasteryRepo.PROPAGATION_DECAY * (mean_dep - 0.5) * 2)
        return min(1.0, base + bonus)

    @staticmethod
    async def get_scores_bulk_with_propagation(
        student_id,
        course_id,
        idea_ids: list[str],
        kg=None,
    ) -> dict[str, float]:
        """Bulk version : score boost backward sur N idea_ids.

        Optimisation : 1 seul SELECT bulk pour tous les scores (direct +
        dependants), au lieu de N requetes.
        """
        if not idea_ids:
            return {}
        if kg is None:
            return await MasteryRepo.get_scores_bulk(student_id, course_id, idea_ids)

        # Collecter tous les idea_ids necessaires : les cibles + leurs dependants
        all_needed: set[str] = set(idea_ids)
        target_dependents: dict[str, list[str]] = {}
        for iid in idea_ids:
            try:
                deps = kg.dependents_of(iid)
            except Exception:
                deps = []
            dep_ids = [d.idea_id for d in deps]
            target_dependents[iid] = dep_ids
            all_needed.update(dep_ids)

        # Un seul fetch pour tout
        all_scores = await MasteryRepo.get_scores_bulk(
            student_id, course_id, list(all_needed),
        )

        result: dict[str, float] = {}
        for iid in idea_ids:
            base = all_scores.get(iid, 0.5)
            deps = target_dependents.get(iid, [])
            if not deps:
                result[iid] = base
                continue
            dep_scores = [all_scores[d] for d in deps if d in all_scores]
            if not dep_scores:
                result[iid] = base
                continue
            mean_dep = sum(dep_scores) / len(dep_scores)
            bonus = max(0.0, MasteryRepo.PROPAGATION_DECAY * (mean_dep - 0.5) * 2)
            result[iid] = min(1.0, base + bonus)
        return result
