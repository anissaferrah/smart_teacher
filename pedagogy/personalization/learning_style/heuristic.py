"""
Auto-détection du style d'apprentissage depuis le comportement de l'étudiant
(version heuristique v1, kept as fallback for cold-start when the Bayesian
posterior has insufficient signal — see learning_style/bayes.py for M2 model).

Au lieu d'un field statique `learning_style="visual"` rempli manuellement,
on l'infère depuis les interactions réelles :

  - VISUAL    : aime les schémas/images, regarde longtemps les slides,
                pose des questions "montre-moi", "quel graphique"
  - AUDITORY  : pose beaucoup de questions vocales, écoute les TTS jusqu'au bout,
                interrompt peu, demande des explications orales
  - KINESTHETIC: aime les exercices pratiques, fait beaucoup de quiz,
                préfère apprendre en faisant
  - READING   : lit les transcriptions, prend des notes (browser blur events),
                pose des questions textuelles plus que vocales

Le score est calculé sur la fenêtre des 30 derniers jours d'interactions.
Mis à jour async toutes les N interactions (idempotent).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, select

from database.init_db import AsyncSessionLocal
from database.models import (
    Interaction, LearningEvent, PracticeAttempt, StudentProfile,
)

log = logging.getLogger("SmartTeacher.LearningStyle")


@dataclass
class StyleScores:
    visual: float       # 0..1
    auditory: float
    kinesthetic: float
    reading: float

    def dominant(self) -> str:
        scores = {
            "visual": self.visual, "auditory": self.auditory,
            "kinesthetic": self.kinesthetic, "reading": self.reading,
        }
        # Si toutes proches (< 0.1 d'écart), retourne "mixed"
        max_score = max(scores.values())
        if max_score == 0:
            return "mixed"
        top_2 = sorted(scores.values(), reverse=True)[:2]
        if (top_2[0] - top_2[1]) < 0.1:
            return "mixed"
        return max(scores, key=scores.get)

    def to_dict(self) -> dict:
        return {
            "visual":      round(self.visual, 3),
            "auditory":    round(self.auditory, 3),
            "kinesthetic": round(self.kinesthetic, 3),
            "reading":     round(self.reading, 3),
            "dominant":    self.dominant(),
        }


# ── Heuristic signal weights ─────────────────────────────────────────
# Ces poids sont calibrés à la main, à raffiner avec données réelles.

_VISUAL_KEYWORDS = {
    "fr": ["montre", "schéma", "graphique", "image", "diagramme", "visualise", "voir"],
    "en": ["show", "diagram", "graph", "image", "picture", "visualize", "see", "chart"],
}
_KINESTHETIC_KEYWORDS = {
    "fr": ["exemple", "exercice", "pratique", "essaie", "essaye", "fais", "manipuler"],
    "en": ["example", "exercise", "practice", "try", "do", "hands-on", "manipulate"],
}
_READING_KEYWORDS = {
    "fr": ["définition", "écris", "résumé", "texte", "lis", "rédige"],
    "en": ["definition", "write", "summary", "text", "read", "draft"],
}


async def compute_learning_style(
    student_id: uuid.UUID, days: int = 30,
) -> StyleScores:
    """Compute the 4 style scores from the last `days` of behavior."""
    since = datetime.utcnow() - timedelta(days=days)

    async with AsyncSessionLocal() as db:
        audio_count = (await db.execute(
            select(func.count(Interaction.id))
            .where(Interaction.student_id == str(student_id))
            .where(Interaction.type == "qa")
            .where(Interaction.created_at >= since)
            .where(Interaction.stt_time > 0)
        )).scalar() or 0

        text_count = (await db.execute(
            select(func.count(Interaction.id))
            .where(Interaction.student_id == str(student_id))
            .where(Interaction.type == "qa")
            .where(Interaction.created_at >= since)
            .where(Interaction.stt_time == 0)
        )).scalar() or 0

        practice_count = (await db.execute(
            select(func.count(PracticeAttempt.id))
            .where(PracticeAttempt.student_id == student_id)
            .where(PracticeAttempt.created_at >= since)
        )).scalar() or 0

        interrupt_count = (await db.execute(
            select(func.count(LearningEvent.id))
            .where(LearningEvent.student_id == str(student_id))
            .where(LearningEvent.event_type == "interrupt")
            .where(LearningEvent.created_at >= since)
        )).scalar() or 0

        recent_questions = (await db.execute(
            select(Interaction.question, Interaction.language)
            .where(Interaction.student_id == str(student_id))
            .where(Interaction.created_at >= since)
            .where(Interaction.question.isnot(None))
            .limit(100)
        )).all()

    visual_hits = 0
    kinesthetic_hits = 0
    reading_hits = 0
    for q_text, lang in recent_questions:
        if not q_text:
            continue
        q_low = q_text.lower()
        keywords_lang = (lang or "fr")[:2]
        for kw in _VISUAL_KEYWORDS.get(keywords_lang, []):
            if kw in q_low:
                visual_hits += 1
                break
        for kw in _KINESTHETIC_KEYWORDS.get(keywords_lang, []):
            if kw in q_low:
                kinesthetic_hits += 1
                break
        for kw in _READING_KEYWORDS.get(keywords_lang, []):
            if kw in q_low:
                reading_hits += 1
                break

    total_signals = audio_count + text_count + practice_count + interrupt_count + 1

    visual = (interrupt_count * 0.6 + visual_hits * 1.0) / total_signals
    auditory = (audio_count * 1.0) / total_signals
    kinesthetic = (practice_count * 1.0 + kinesthetic_hits * 0.5) / total_signals
    reading = (text_count * 0.8 + reading_hits * 1.0) / total_signals

    s = max(0.0001, visual + auditory + kinesthetic + reading)
    return StyleScores(
        visual=visual / s,
        auditory=auditory / s,
        kinesthetic=kinesthetic / s,
        reading=reading / s,
    )


async def update_student_learning_style(
    student_id: uuid.UUID, days: int = 30, min_signals: int = 5,
) -> StyleScores | None:
    """Recompute + persist learning_style on the StudentProfile.

    Returns None if not enough data yet (< min_signals total interactions).
    """
    scores = await compute_learning_style(student_id, days=days)

    total_signal = scores.visual + scores.auditory + scores.kinesthetic + scores.reading
    if total_signal < 0.1:
        return None

    async with AsyncSessionLocal() as db:
        profile = (await db.execute(
            select(StudentProfile).where(StudentProfile.student_id == student_id)
        )).scalar_one_or_none()
        if profile is None:
            return scores
        profile.learning_style = scores.dominant()
        profile.updated_at = datetime.utcnow()
        await db.commit()

    log.info(
        f"[{str(student_id)[:8]}] learning_style updated → "
        f"{scores.dominant()} (V={scores.visual:.2f} A={scores.auditory:.2f} "
        f"K={scores.kinesthetic:.2f} R={scores.reading:.2f})"
    )
    return scores


def style_to_prompt_hint(style: str, lang: str = "fr") -> str:
    """Generate a system-prompt addition tailored to the dominant learning style.

    Used by Responder/Narrator agents to adapt their explanations.
    """
    hints = {
        "fr": {
            "visual":      "Privilégie les analogies visuelles, schémas mentaux et descriptions imagées.",
            "auditory":    "Explique avec un ton conversationnel, utilise des rythmes et répétitions clés.",
            "kinesthetic": "Donne des exemples pratiques et propose une mise en application immédiate.",
            "reading":     "Structure ta réponse comme un texte écrit : titres, points-clés, définitions précises.",
            "mixed":       "Combine analogie visuelle + exemple pratique + définition claire.",
        },
        "en": {
            "visual":      "Use visual analogies, mental imagery, and descriptive metaphors.",
            "auditory":    "Use conversational tone, rhythm, and key repetitions.",
            "kinesthetic": "Give hands-on examples and propose immediate application.",
            "reading":     "Structure as written text: headers, bullet points, precise definitions.",
            "mixed":       "Combine visual analogy + practical example + clear definition.",
        },
    }
    return hints.get(lang[:2], hints["fr"]).get(style, hints["fr"]["mixed"])

# NOTE: style_to_params() lives in `params.py` — it's a structured-output
# concern (narrator directives), distinct from the heuristic v1 logic.
