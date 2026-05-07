"""Personalization Engine — Cognitive style + pace adaptation.

Lit StudentProfile + StudentMistake + PracticeAttempt pour deriver :
  - learning_style    : visual | auditory | kinesthetic | reading
  - pace              : slow | normal | fast (auto-detecte)
  - explanation_depth : concise | balanced | detailed
  - tone              : encouraging | neutral | challenging

Et compose des fragments de prompt à injecter dans Responder/Planner.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, func

from database.init_db import AsyncSessionLocal
from database.models import StudentProfile, PracticeAttempt

log = logging.getLogger("SmartTeacher.Personalization")


# ── Honest defaults (NOT empirically calibrated) ─────────────────────────
# These thresholds bin a continuous signal (response time, confusion rate)
# into 3 buckets. The cutoffs were picked by intuition, not measured on
# real student data. They live as named constants so we can:
#   (a) document that they're unproven defaults,
#   (b) override them via Config without code changes once we have data,
#   (c) point a future calibration job at a single location.
# The right long-term fix is z-score-relative-to-cohort or quantile-based
# bucketing on real PracticeAttempt logs.

# Practice-attempt aggregation window. 30 days = "current term-ish",
# long enough to smooth noise from a bad day, short enough that an
# evolving student isn't held back by stale signal.
_PACE_WINDOW_DAYS = 30
# Minimum attempts before we trust the per-student average. Below this,
# we keep the stored profile.pace (or the default "normal") because
# the sample is too small to overrule it.
_PACE_MIN_ATTEMPTS = 5

# Pace cutoffs (seconds per practice attempt). 60s/15s are honest defaults.
_PACE_SLOW_S = 60.0
_PACE_FAST_S = 15.0

# Confusion-rate cutoffs (ratio in [0, 1]). 0.3 / 0.05 are honest defaults.
_TONE_ENCOURAGING_AT = 0.30
_TONE_CHALLENGING_AT = 0.05


def _coerce_uuid(value) -> Optional[uuid.UUID]:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


@dataclass
class PersonalizationContext:
    """Snapshot des preferences cognitives pour un eleve."""
    learning_style: str = "visual"          # visual | auditory | kinesthetic | reading
    pace: str = "normal"                     # slow | normal | fast
    explanation_depth: str = "balanced"      # concise | balanced | detailed
    tone: str = "encouraging"                # encouraging | neutral | challenging
    avg_response_time_s: float = 0.0
    confusion_rate: float = 0.0
    preferred_difficulty: str = "intermediate"

    def to_dict(self) -> dict:
        return {
            "learning_style": self.learning_style,
            "pace": self.pace,
            "explanation_depth": self.explanation_depth,
            "tone": self.tone,
            "avg_response_time_s": round(self.avg_response_time_s, 2),
            "confusion_rate": round(self.confusion_rate, 3),
            "preferred_difficulty": self.preferred_difficulty,
        }


# ── Fragments de prompt par style + pace ────────────────────────────────
_STYLE_PROMPTS = {
    "fr": {
        "visual":      "Privilégie des analogies visuelles, des schémas mentaux, des comparaisons spatiales (ex: 'imagine un graphe', 'visualise un cube'). ",
        "auditory":    "Privilégie un ton conversationnel, des rythmes oraux, des explications qu'on peut 'entendre' (ex: 'écoute bien...', 'la formule sonne ainsi...'). ",
        "kinesthetic": "Privilégie des exemples concrets, manipulables, des analogies physiques avec mouvement (ex: 'comme quand tu pousses une balle', 'simule l'opération étape par étape'). ",
        "reading":     "Privilégie un langage technique précis, structuré en points, références explicites au contenu écrit. ",
    },
    "en": {
        "visual":      "Use visual analogies, mental schemas, spatial comparisons (ex: 'picture a graph', 'visualize a cube'). ",
        "auditory":    "Use a conversational tone, oral rhythms, explanations that can be 'heard'. ",
        "kinesthetic": "Use concrete, manipulable examples, physical analogies with motion. ",
        "reading":     "Use precise technical language, bullet-point structure, explicit references to written content. ",
    },
}

_PACE_PROMPTS = {
    "fr": {
        "slow":   "L'élève apprend lentement : décompose en TRÈS petites étapes, répète les points clés, vérifie souvent la compréhension. ",
        "normal": "",
        "fast":   "L'élève apprend vite : sois concis, va à l'essentiel, propose des extensions ou défis avancés. ",
    },
    "en": {
        "slow":   "Student learns slowly: break into VERY small steps, repeat key points, check understanding often. ",
        "normal": "",
        "fast":   "Student learns fast: be concise, get to the point, offer extensions or advanced challenges. ",
    },
}

_DEPTH_PROMPTS = {
    "fr": {
        "concise":  "Format CONCIS : 1-2 phrases max. ",
        "balanced": "Format ÉQUILIBRÉ : 3-4 phrases. ",
        "detailed": "Format DÉTAILLÉ : 5-7 phrases avec exemple. ",
    },
    "en": {
        "concise":  "CONCISE format: 1-2 sentences max. ",
        "balanced": "BALANCED format: 3-4 sentences. ",
        "detailed": "DETAILED format: 5-7 sentences with example. ",
    },
}

_TONE_PROMPTS = {
    "fr": {
        "encouraging": "Ton bienveillant et encourageant. ",
        "neutral":     "Ton neutre et factuel. ",
        "challenging": "Ton stimulant : pose des sous-questions, pousse l'élève à réfléchir avant de donner la réponse. ",
    },
    "en": {
        "encouraging": "Kind and encouraging tone. ",
        "neutral":     "Neutral and factual tone. ",
        "challenging": "Challenging tone: ask sub-questions, push the student to think before giving answers. ",
    },
}


class PersonalizationEngine:
    """Lecture du profil + derivation de style/pace + composition de prompt fragments."""

    @staticmethod
    async def get_context(
        student_id,
        course_id=None,
    ) -> PersonalizationContext:
        """Read StudentProfile + derive pace/depth from recent history.

        Profile is scoped per ``(student, course)``. The lookup tries the
        course-specific row first, then falls back to the per-student
        ``course_id IS NULL`` row, so a student opening a brand-new course
        still inherits their cross-course preferences (VARK, broad pace).
        Practice-attempt aggregation is also filtered by course when
        provided — the same student can be fast on a familiar subject and
        slow on a new one, and we want the bandit / prompt to react to the
        *current* course rather than a blended average.
        """
        sid = _coerce_uuid(student_id)
        if not sid:
            return PersonalizationContext()

        cid = _coerce_uuid(course_id)

        try:
            async with AsyncSessionLocal() as db:
                # Profile : try the (student, course) row first, then the
                # cross-course default (course_id IS NULL).
                profile = None
                if cid is not None:
                    profile = (await db.execute(
                        select(StudentProfile).where(
                            StudentProfile.student_id == sid,
                            StudentProfile.course_id == cid,
                        )
                    )).scalar_one_or_none()
                if profile is None:
                    profile = (await db.execute(
                        select(StudentProfile).where(
                            StudentProfile.student_id == sid,
                            StudentProfile.course_id.is_(None),
                        )
                    )).scalar_one_or_none()

                ctx = PersonalizationContext()
                if profile:
                    ctx.learning_style = profile.learning_style or "visual"
                    ctx.preferred_difficulty = profile.preferred_difficulty or "intermediate"
                    ctx.pace = profile.pace or "normal"
                    ctx.avg_response_time_s = float(profile.avg_response_time_s or 0.0)
                    ctx.confusion_rate = float(profile.confusion_rate or 0.0)
                    ctx.explanation_depth = profile.preferred_explanation_depth or "balanced"

                # Pace + confusion are derived from the recent practice
                # attempts ONLY if we have ≥ _PACE_MIN_ATTEMPTS samples.
                # Below that, we trust the stored profile values (or the
                # defaults) — the sample is too small to overrule them.
                # This makes PracticeAttempt the single source of truth
                # whenever data is available, and the stored profile a
                # legitimate fallback otherwise.
                cutoff = datetime.utcnow() - timedelta(days=_PACE_WINDOW_DAYS)
                attempt_filters = [
                    PracticeAttempt.student_id == sid,
                    PracticeAttempt.created_at >= cutoff,
                ]
                if cid is not None and hasattr(PracticeAttempt, "course_id"):
                    attempt_filters.append(PracticeAttempt.course_id == cid)
                recent_attempts = (await db.execute(
                    select(
                        func.avg(PracticeAttempt.time_taken_s).label("avg_time"),
                        func.count(PracticeAttempt.id).label("total"),
                        func.sum(
                            # boolean cast as int : counts True
                            func.cast(PracticeAttempt.is_correct, type_=type(PracticeAttempt.hints_used.type))
                        ).label("correct"),
                    ).where(*attempt_filters)
                )).first()

                total_attempts = int(recent_attempts.total or 0) if recent_attempts else 0
                if total_attempts >= _PACE_MIN_ATTEMPTS:
                    avg_time = float(recent_attempts.avg_time or 0.0)
                    ctx.avg_response_time_s = avg_time
                    if avg_time > _PACE_SLOW_S:
                        ctx.pace = "slow"
                    elif avg_time < _PACE_FAST_S:
                        ctx.pace = "fast"
                    else:
                        ctx.pace = "normal"

                    # Issue 1.3 fix : confusion_rate was previously read
                    # from the StudentProfile column but never written.
                    # Compute it here as the empirical incorrect-attempt
                    # ratio over the same window. ``correct`` is summed
                    # over a boolean cast, so (total - correct) is the
                    # number of incorrect attempts.
                    correct = int(recent_attempts.correct or 0)
                    incorrect = max(0, total_attempts - correct)
                    ctx.confusion_rate = round(incorrect / total_attempts, 3)

                # Tone derived from confusion rate (honest-default cutoffs).
                if ctx.confusion_rate > _TONE_ENCOURAGING_AT:
                    ctx.tone = "encouraging"  # struggling student → kindness
                elif ctx.confusion_rate < _TONE_CHALLENGING_AT:
                    ctx.tone = "challenging"  # confident student → push
                else:
                    ctx.tone = "neutral"

                # Depth follows pace: slower student → more detail.
                if ctx.pace == "slow":
                    ctx.explanation_depth = "detailed"
                elif ctx.pace == "fast":
                    ctx.explanation_depth = "concise"

                return ctx
        except Exception as exc:
            log.debug(f"get_context failed: {exc}")
            return PersonalizationContext()

    @staticmethod
    def build_prompt_prefix(ctx: PersonalizationContext, lang: str = "fr") -> str:
        """Compose un fragment a injecter en debut de prompt LLM."""
        lang = lang[:2] if lang else "fr"
        if lang not in _STYLE_PROMPTS:
            lang = "fr"

        style_frag = _STYLE_PROMPTS[lang].get(ctx.learning_style, "")
        pace_frag = _PACE_PROMPTS[lang].get(ctx.pace, "")
        depth_frag = _DEPTH_PROMPTS[lang].get(ctx.explanation_depth, "")
        tone_frag = _TONE_PROMPTS[lang].get(ctx.tone, "")

        composed = (style_frag + pace_frag + depth_frag + tone_frag).strip()
        if not composed:
            return ""
        if lang == "fr":
            return f"PERSONNALISATION COGNITIVE (à respecter) : {composed}\n\n"
        return f"COGNITIVE PERSONALIZATION (apply): {composed}\n\n"
