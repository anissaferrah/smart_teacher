"""
VARK questionnaire — instrument psychométrique pour seed du prior bayésien.

Basé sur : Fleming, N. D., & Mills, C. (1992). "Not another inventory, rather
a catalyst for reflection." To Improve the Academy, 11(1), 137–155.

Version courte (8 items) adaptée pour onboarding rapide. Chaque item propose
4 options (V, A, R, K) — le questionnaire permet le choix multiple
(score ≠ scale unidimensionnelle, c'est un profil).

Les pseudo-counts générés par ce questionnaire alimentent le prior Dirichlet
du modèle Bayes (services/learning_style_bayes.py).

Validation :
  - Test-retest reliability (Fleming, 1995) : r ≈ 0.79 sur 4 dimensions
  - Convergent validity avec Honey-Mumford : r = 0.61
  - On utilise une version courte donc on perd un peu mais reste indicatif.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Style codes utilisés dans le système
StyleCode = Literal["visual", "auditory", "reading", "kinesthetic"]


@dataclass(frozen=True)
class VARKQuestion:
    id: str
    text_fr: str
    text_en: str
    options: dict[StyleCode, dict[str, str]]    # {style: {fr, en}}


# ── 8 items courts + équilibrés (2 par dimension dominante) ──────────

VARK_QUESTIONS: list[VARKQuestion] = [
    VARKQuestion(
        id="q1",
        text_fr="Quand tu apprends un concept difficile, tu préfères :",
        text_en="When learning a hard concept, you prefer:",
        options={
            "visual":      {"fr": "Voir un schéma ou diagramme",
                            "en": "See a diagram or chart"},
            "auditory":    {"fr": "Écouter quelqu'un l'expliquer à voix haute",
                            "en": "Listen to someone explain it aloud"},
            "reading":     {"fr": "Lire la définition écrite",
                            "en": "Read the written definition"},
            "kinesthetic": {"fr": "Faire un exercice pratique tout de suite",
                            "en": "Do a hands-on exercise right away"},
        },
    ),
    VARKQuestion(
        id="q2",
        text_fr="Pour mémoriser une formule mathématique, tu :",
        text_en="To memorize a math formula, you:",
        options={
            "visual":      {"fr": "La visualises mentalement comme une image",
                            "en": "Visualize it mentally as an image"},
            "auditory":    {"fr": "La récites à voix haute plusieurs fois",
                            "en": "Recite it aloud several times"},
            "reading":     {"fr": "L'écris plusieurs fois",
                            "en": "Write it down several times"},
            "kinesthetic": {"fr": "L'utilises dans plusieurs exercices",
                            "en": "Use it in several exercises"},
        },
    ),
    VARKQuestion(
        id="q3",
        text_fr="Une présentation idéale, pour toi, c'est :",
        text_en="An ideal lecture, for you, would be:",
        options={
            "visual":      {"fr": "Beaucoup de schémas et illustrations",
                            "en": "Lots of diagrams and illustrations"},
            "auditory":    {"fr": "Un orateur engageant qui parle clairement",
                            "en": "An engaging speaker with clear voice"},
            "reading":     {"fr": "Des notes détaillées et un livre",
                            "en": "Detailed notes and a textbook"},
            "kinesthetic": {"fr": "Des démonstrations et expériences",
                            "en": "Demonstrations and experiments"},
        },
    ),
    VARKQuestion(
        id="q4",
        text_fr="Quand tu lis un texte technique, tu :",
        text_en="When reading a technical text, you:",
        options={
            "visual":      {"fr": "Regardes d'abord les figures et diagrammes",
                            "en": "Look at figures and diagrams first"},
            "auditory":    {"fr": "Le lis à voix haute (ou ta voix intérieure)",
                            "en": "Read it aloud (or in your head)"},
            "reading":     {"fr": "Prends des notes au fur et à mesure",
                            "en": "Take notes as you go"},
            "kinesthetic": {"fr": "Tries d'appliquer immédiatement à un exemple",
                            "en": "Try to apply it to an example right away"},
        },
    ),
    VARKQuestion(
        id="q5",
        text_fr="Pour t'orienter dans un nouvel endroit, tu :",
        text_en="To find your way in a new place, you:",
        options={
            "visual":      {"fr": "Regardes une carte ou un plan",
                            "en": "Look at a map"},
            "auditory":    {"fr": "Demandes oralement des indications",
                            "en": "Ask for verbal directions"},
            "reading":     {"fr": "Lis les panneaux et instructions écrites",
                            "en": "Read signs and written instructions"},
            "kinesthetic": {"fr": "Te promènes pour explorer",
                            "en": "Walk around to explore"},
        },
    ),
    VARKQuestion(
        id="q6",
        text_fr="Pour expliquer un sujet à un ami, tu :",
        text_en="To explain a topic to a friend, you:",
        options={
            "visual":      {"fr": "Lui dessines un schéma",
                            "en": "Draw them a diagram"},
            "auditory":    {"fr": "Lui parles à voix haute",
                            "en": "Talk it through aloud"},
            "reading":     {"fr": "Lui écris un résumé",
                            "en": "Write them a summary"},
            "kinesthetic": {"fr": "Lui montres comment faire avec un exemple",
                            "en": "Show them by doing an example"},
        },
    ),
    VARKQuestion(
        id="q7",
        text_fr="Pour réviser avant un examen, tu :",
        text_en="When reviewing before a test, you:",
        options={
            "visual":      {"fr": "Refais des cartes mentales colorées",
                            "en": "Redraw colourful mind maps"},
            "auditory":    {"fr": "Discutes avec quelqu'un du sujet",
                            "en": "Discuss the topic with someone"},
            "reading":     {"fr": "Relis tes notes plusieurs fois",
                            "en": "Re-read your notes several times"},
            "kinesthetic": {"fr": "Refais d'anciens exercices",
                            "en": "Redo past exercises"},
        },
    ),
    VARKQuestion(
        id="q8",
        text_fr="Tu te souviens mieux d'un cours quand :",
        text_en="You remember a lecture best when:",
        options={
            "visual":      {"fr": "Le prof utilisait des illustrations",
                            "en": "The teacher used illustrations"},
            "auditory":    {"fr": "Le prof racontait des histoires",
                            "en": "The teacher told stories"},
            "reading":     {"fr": "Tu avais un polycopié",
                            "en": "You had handouts to read"},
            "kinesthetic": {"fr": "Vous faisiez des manipulations en groupe",
                            "en": "You did group hands-on activities"},
        },
    ),
]


def serialize_for_frontend(lang: str = "fr") -> list[dict]:
    """Format les questions pour affichage frontend (langue courante)."""
    out = []
    for q in VARK_QUESTIONS:
        out.append({
            "id": q.id,
            "text": q.text_fr if lang == "fr" else q.text_en,
            "options": [
                {
                    "style": s,
                    "text":  q.options[s]["fr" if lang == "fr" else "en"],
                }
                for s in ("visual", "auditory", "reading", "kinesthetic")
            ],
        })
    return out


def score_responses(responses: dict[str, list[str]]) -> dict[str, int]:
    """Convertit les réponses utilisateur en pseudo-counts par dimension.

    Args:
        responses: {q_id: [style_code, ...]}  (multi-choice autorisé)

    Returns:
        {style_code: count}  utilisable comme pseudo-counts pour Dirichlet prior.
    """
    counts = {"visual": 0, "auditory": 0, "kinesthetic": 0, "reading": 0}
    valid_qids = {q.id for q in VARK_QUESTIONS}
    for q_id, styles in responses.items():
        if q_id not in valid_qids:
            continue
        for s in (styles or []):
            if s in counts:
                counts[s] += 1
    return counts
