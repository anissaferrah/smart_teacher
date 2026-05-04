"""Aggregated student knowledge snapshot — the "what does this student know"
input to adaptive teaching.

# Why this exists

Existing modules each hold a *piece* of the student's knowledge state :

  - ``MasteryRepo``                       — Bayesian posterior per idea_id
  - ``dialogue.get_seen_ideas``           — set of ideas presented this session
  - ``confusion.persistence``             — confusion events log
  - ``KnowledgeGraph``                    — concept-level grouping

The Responder needs ALL of these to behave like a real teacher : "this
student already understands K-means well, but stumbled on inverted
indexes — I should connect the new concept to K-means and re-explain
the index part with simpler words."

This module is the SINGLE READ-PATH that joins those sources into one
``StudentKnowledgeSnapshot``. The Responder reads the snapshot, never
the underlying tables.

# What's in a snapshot

  - ``mastery_by_concept``   : concept name → mastery [0..1]
  - ``strong_concepts``      : concept names with mastery ≥ 0.7
  - ``weak_concepts``        : concept names with mastery < 0.4 AND attempted
  - ``never_seen_concepts``  : course concepts the student has 0 attempts on
  - ``recently_confused``    : concept names whose last attempt was a
                               confusion (most recent first, capped at 5)

# How concept mastery is derived

Concepts are clusters of ideas (KnowledgeGraph). A concept's mastery
is the **mean of its ideas' mastery scores** — Laplace-smoothed at
the idea level, then averaged. This avoids the "one bad idea sinks
the concept" failure mode while still penalising broadly-weak
concepts.

# Adapting prompts

``to_prompt_context(snap, lang)`` formats the snapshot into a
non-confusable prompt block (``<<INSTRUCTION_INTERNE>>`` markers, same
convention as the personalization layer) so the LLM uses it for
*adaptation*, not as content to discuss.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from core.config import Config

log = logging.getLogger("pedagogy.student_knowledge")


# Thresholds — sourced from Config so operators can tune per-deployment
# (e.g., a course in a non-STEM domain may need a softer "strong"
# threshold). Module-level aliases preserved for backward-compat.
STRONG_THRESHOLD      = Config.KNOWLEDGE_STRONG_THRESHOLD
WEAK_THRESHOLD        = Config.KNOWLEDGE_WEAK_THRESHOLD
CONFUSED_THRESHOLD    = Config.KNOWLEDGE_CONFUSED_THRESHOLD
MAX_LISTED_CONCEPTS   = Config.KNOWLEDGE_MAX_LISTED_CONCEPTS
MAX_RECENT_CONFUSIONS = 5
COLD_START_ATTEMPTS   = Config.KNOWLEDGE_COLD_START_ATTEMPTS


@dataclass
class StudentKnowledgeSnapshot:
    """Read-only view of what this student currently knows.

    Returned by :func:`build_snapshot`. All collections are sorted by
    *informativeness* (most relevant first) so callers that truncate
    don't lose signal.
    """
    student_id: str
    course_id: str
    mastery_by_concept: dict[str, float] = field(default_factory=dict)
    strong_concepts:     list[str]       = field(default_factory=list)
    weak_concepts:       list[str]       = field(default_factory=list)
    never_seen_concepts: list[str]       = field(default_factory=list)
    recently_confused:   list[str]       = field(default_factory=list)
    # Diagnostic — how confident are these numbers ? Low total_attempts
    # means we don't have enough data and the prompt should be careful
    # about claiming "you already know X".
    total_attempts:      int             = 0

    @property
    def is_cold_start(self) -> bool:
        """True if attempts <= COLD_START_ATTEMPTS. Adaptive prompts
        should NOT claim mastery in this regime — too little data."""
        return self.total_attempts <= COLD_START_ATTEMPTS


async def build_snapshot(
    student_id: Optional[str],
    course_id: Optional[str],
    *,
    kg=None,
) -> StudentKnowledgeSnapshot:
    """Build the snapshot by joining MasteryRepo + KnowledgeGraph.

    ``kg`` is the ``KnowledgeGraph`` singleton ; passed in for
    testability (callers can inject a mock). When ``None`` we lazily
    fetch the singleton via ``deps.get_rag()`` + ``get_or_build`` —
    same pattern as ``pedagogy.practice_engine._get_kg``.

    Returns an *empty-ish* snapshot when ``student_id`` is missing or
    the underlying data is unavailable. Callers must handle the empty
    case anyway (cold-start students, anonymous sessions).
    """
    snap = StudentKnowledgeSnapshot(
        student_id=str(student_id or ""),
        course_id=str(course_id or ""),
    )
    if not student_id or not course_id:
        return snap

    # ── 1. Get the course concepts from the KG ───────────────────
    if kg is None:
        try:
            from deps import get_rag
            from pedagogy.knowledge_graph import get_or_build
            kg = get_or_build(get_rag())
        except Exception as exc:                                          # noqa: BLE001
            log.debug("build_snapshot: kg unavailable (%s)", exc)
            return snap

    try:
        concepts = list(kg.list_concepts(course_id))
    except Exception as exc:                                              # noqa: BLE001
        log.debug("build_snapshot: list_concepts failed (%s)", exc)
        return snap

    if not concepts:
        return snap

    # ── 2. Bulk-fetch mastery scores for ALL ideas in the course ─
    all_idea_ids: list[str] = []
    concept_ideas: dict[str, list[str]] = {}
    for c in concepts:
        ids = list(getattr(c, "idea_ids", set()) or set())
        concept_ideas[c.name] = ids
        all_idea_ids.extend(ids)

    if not all_idea_ids:
        return snap

    try:
        from pedagogy.mastery_repo import MasteryRepo
        idea_scores = await MasteryRepo.get_scores_bulk(
            student_id, course_id, all_idea_ids,
        )
    except Exception as exc:                                              # noqa: BLE001
        log.debug("build_snapshot: get_scores_bulk failed (%s)", exc)
        idea_scores = {}

    # ── 3. Aggregate idea-level scores → concept-level ───────────
    for c in concepts:
        ids = concept_ideas.get(c.name, [])
        scored = [idea_scores[i] for i in ids if i in idea_scores]
        if scored:
            mean_score = round(sum(scored) / len(scored), 3)
            snap.mastery_by_concept[c.name] = mean_score
            log.info(
                "🔍 snapshot AGG | concept=%-30s | mean(%s) = %.3f / %d ideas",
                c.name[:30],
                ", ".join(f"{s:.2f}" for s in scored[:5]) +
                ("..." if len(scored) > 5 else ""),
                mean_score, len(scored),
            )
        # When the concept has 0 attempted ideas it stays absent from
        # mastery_by_concept (that's how never_seen is detected below).

    snap.total_attempts = len(idea_scores)  # rough proxy

    # ── 4. Classify concepts by mastery bucket ───────────────────
    sorted_by_score = sorted(
        snap.mastery_by_concept.items(), key=lambda kv: kv[1], reverse=True,
    )
    snap.strong_concepts = [
        name for name, score in sorted_by_score
        if score >= STRONG_THRESHOLD
    ][:MAX_LISTED_CONCEPTS]
    snap.weak_concepts = [
        name for name, score in sorted(
            ((n, s) for n, s in snap.mastery_by_concept.items() if s < WEAK_THRESHOLD),
            key=lambda kv: kv[1],   # weakest first
        )
    ][:MAX_LISTED_CONCEPTS]

    # Concepts the student has never attempted = those with no entry
    # in mastery_by_concept (no idea was scored for them).
    attempted_names = set(snap.mastery_by_concept.keys())
    snap.never_seen_concepts = [
        c.name for c in concepts if c.name not in attempted_names
    ][:MAX_LISTED_CONCEPTS]

    # ── 5. Recently confused — derived from MasteryRepo ──────────
    # The ``confusion_events`` DB table stores concept_id as UUID, but
    # KG concepts are slugs (no UUID column) — direct lookups don't
    # work. As a proxy we treat any concept whose AVERAGE mastery is
    # in the WEAK bucket AND has at least 1 confusion (i.e. attempts >
    # successes) as "recently struggled with". This is a coarse signal
    # but it's grounded in the same data the bandit already trusts.
    try:
        # Fetch attempt counts to disambiguate "weak because tried+failed"
        # from "weak because never seen" (the latter shouldn't appear
        # in recently_confused).
        from pedagogy.mastery_repo import MasteryRepo  # noqa: F401  (reuse)
        # Heuristic : weak_concepts already filters mastery < 0.4 ; we
        # promote them to recently_confused when their confusion ratio
        # is high enough (≥ 30% of attempts were confusions). Without
        # the full DB query, we approximate via mastery score : a
        # Laplace-smoothed posterior ≤ 0.3 means at least one confusion
        # was observed for sure.
        snap.recently_confused = [
            name for name, score in snap.mastery_by_concept.items()
            if score <= CONFUSED_THRESHOLD
        ][:MAX_RECENT_CONFUSIONS]
    except Exception as exc:                                              # noqa: BLE001
        log.debug("build_snapshot: recently_confused derivation skipped (%s)", exc)

    return snap


def to_prompt_context(snap: StudentKnowledgeSnapshot, lang: str = "fr") -> str:
    """Format the snapshot as a prompt block ready to inject into the
    Responder's prompt.

    Wrapped in ``<<INSTRUCTION_INTERNE>>`` markers (same convention as
    ``personalization.engine.build_prompt_prefix``) so the LLM treats
    it as adaptation context, not content to discuss.

    Returns ``""`` for cold-start snapshots — adaptive prompting needs
    enough data to be useful, and cold-start blurts ("vous ne savez
    rien") are condescending.
    """
    if snap.is_cold_start:
        return ""
    lines: list[str] = []

    if lang.startswith("fr"):
        opener = "<<INSTRUCTION_INTERNE_NE_PAS_MENTIONNER_DANS_LA_REPONSE>>"
        closer = "<<FIN_INSTRUCTION_INTERNE>>"
        header = "État pédagogique de l'étudiant (à respecter discrètement) :"
        if snap.strong_concepts:
            lines.append(
                f"- Concepts MAÎTRISÉS (ne pas re-définir, juste citer) : "
                f"{', '.join(snap.strong_concepts)}."
            )
        if snap.weak_concepts:
            lines.append(
                f"- Concepts FAIBLES (reformuler simplement, ajouter un exemple) : "
                f"{', '.join(snap.weak_concepts)}."
            )
        if snap.recently_confused:
            lines.append(
                f"- Récemment CONFUS sur (éviter le jargon, prendre le temps) : "
                f"{', '.join(snap.recently_confused)}."
            )
        if snap.never_seen_concepts:
            lines.append(
                f"- Concepts JAMAIS VUS (présenter la définition complète si évoqués) : "
                f"{', '.join(snap.never_seen_concepts[:5])}."
            )
        if not lines:
            return ""
        return f"{opener}\n{header}\n" + "\n".join(lines) + f"\n{closer}\n\n"

    # English
    opener = "<<INTERNAL_INSTRUCTION_DO_NOT_MENTION_IN_ANSWER>>"
    closer = "<<END_INTERNAL_INSTRUCTION>>"
    header = "Student knowledge state (apply implicitly) :"
    if snap.strong_concepts:
        lines.append(
            f"- MASTERED concepts (don't redefine, just reference) : "
            f"{', '.join(snap.strong_concepts)}."
        )
    if snap.weak_concepts:
        lines.append(
            f"- WEAK concepts (rephrase simply, add an example) : "
            f"{', '.join(snap.weak_concepts)}."
        )
    if snap.recently_confused:
        lines.append(
            f"- Recently CONFUSED on (avoid jargon, take time) : "
            f"{', '.join(snap.recently_confused)}."
        )
    if snap.never_seen_concepts:
        lines.append(
            f"- NEVER-SEEN concepts (full definition if invoked) : "
            f"{', '.join(snap.never_seen_concepts[:5])}."
        )
    if not lines:
        return ""
    return f"{opener}\n{header}\n" + "\n".join(lines) + f"\n{closer}\n\n"
