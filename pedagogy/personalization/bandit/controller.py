"""High-level orchestrator for the bandit lifecycle.

# Why a controller layer

The responder needs to do four bandit-related things in a specific order :

  1. **Resolve any pending reward** : if the previous turn left a pending
     decision, compute its reward now (we know the new outcome) and
     update the bandit.
  2. **Build the current context** : translate the student profile +
     mastery into a discretized ContextBucket.
  3. **Pick an action** for the current turn via Thompson sampling.
  4. **Stash the decision** as pending, so the next turn can compute
     the reward.

Doing this inline inside the responder would scatter the logic. The
controller centralizes it behind a clean two-method API :

    controller = BanditController()

    # at the start of an answer-generating turn
    decision = await controller.start_turn(
        session_id=..., student_id=..., language="fr",
        learning_style=..., avg_response_time_s=..., mastery_score=...,
        primary_concept=...,
    )
    # decision.action.strategy → inject prompt fragment
    # decision.action.speech_rate → set TTS rate
    # ... LLM call happens ...

    # at the end of the turn (or beginning of next)
    await controller.end_turn(
        session_id=...,
        confusion_detected_next_turn=...,
        mastery_after=...,
        engaged=...,
    )

# Backwards-compat fallback

If the bandit module fails to load (Redis down, import error), the
controller transparently returns a no-op decision (the default
strategy = ANALOGY at NORMAL pace) and silently skips updates. The
responder keeps working without the bandit.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from pedagogy.personalization.bandit.repo import (
    PendingDecision,
    consume_pending,
    get_bandit,
    mark_dirty,
    maybe_save_periodically,
    record_pending,
)
from pedagogy.personalization.bandit.reward import TurnOutcome, compute_reward
from pedagogy.personalization.bandit.strategies import (
    SpeechRate,
    Strategy,
    StrategyAction,
    strategy_prompt,
)
from pedagogy.personalization.bandit.thompson import (
    ContextBucket,
    context_from_profile,
)

log = logging.getLogger("personalization.bandit.controller")


_DEFAULT_ACTION = StrategyAction(Strategy.ANALOGY, SpeechRate.NORMAL)


# ── KG-aware context helper ────────────────────────────────────────────

async def _compute_prereqs_mastered_ratio(
    student_id: str | None,
    course_id: str | None,
    primary_concept: str,
) -> float | None:
    """Calcule le ratio de prereqs maitrises pour `primary_concept`.

    Returns:
        - None si KG indispo, concept pas dans le KG, ou args invalides
          (→ bucket "isolated" downstream)
        - float dans [0, 1] = (prereqs mastered) / (total prereqs)

    Threshold mastered = MASTERY_THRESHOLD (0.85). Pour eviter tout
    couplage circulaire, on utilise le score boost (with_propagation),
    pas juste le score brut.
    """
    if not student_id or not course_id or not primary_concept:
        return None
    try:
        from deps import get_rag
        from pedagogy.knowledge_graph import get_or_build
        from pedagogy.mastery_repo import MasteryRepo, MASTERY_THRESHOLD

        rag = get_rag()
        kg = get_or_build(rag)
        if kg is None:
            return None

        # primary_concept peut etre un concept_name OU un idea_id.
        # On essaie d'abord comme concept (preferer concept-level).
        concept_info = kg.get_concept(primary_concept)
        if concept_info is not None:
            # Concept : agreger les prereqs au niveau concept
            prereq_concepts = kg.prereq_concepts(primary_concept)
            if not prereq_concepts:
                return None
            # Tous les idea_ids des prereqs
            prereq_idea_ids: set[str] = set()
            for pc in prereq_concepts:
                prereq_idea_ids.update(pc.idea_ids)
            if not prereq_idea_ids:
                return None
        else:
            # Sinon on tente comme idea_id direct
            prereq_nodes = kg.prerequisites_of(primary_concept)
            if not prereq_nodes:
                return None
            prereq_idea_ids = {p.idea_id for p in prereq_nodes}

        scores = await MasteryRepo.get_scores_bulk_with_propagation(
            student_id, course_id, list(prereq_idea_ids), kg=kg,
        )
        if not scores:
            return 0.0  # prereqs jamais testes → ratio = 0 (blocked)

        n_mastered = sum(1 for s in scores.values() if s >= MASTERY_THRESHOLD)
        return n_mastered / len(prereq_idea_ids)
    except Exception as exc:                                              # noqa: BLE001
        log.debug(f"_compute_prereqs_mastered_ratio failed: {exc}")
        return None


@dataclass
class TurnDecision:
    """Output of ``BanditController.start_turn``.

    Carries the action the responder should apply plus the prompt
    fragment that injects the chosen strategy. ``ok=False`` means the
    bandit was unavailable and a default action was returned ; the
    caller can still apply it but should not expect personalization.
    """

    context:        ContextBucket
    action:         StrategyAction
    prompt_fragment: str
    ok:             bool

    @property
    def strategy(self) -> Strategy:
        return self.action.strategy

    @property
    def speech_rate(self) -> SpeechRate:
        return self.action.speech_rate


class BanditController:
    """Orchestrates the start_turn / end_turn lifecycle around the bandit."""

    async def start_turn(
        self,
        *,
        session_id: str,
        learning_style: str,
        avg_response_time_s: float,
        mastery_score: float,
        primary_concept: str = "",
        language: str = "fr",
        student_id: str | None = None,
        course_id: str | None = None,
    ) -> TurnDecision:
        """Pick an action for the current turn and stash it as pending.

        Returns a TurnDecision (always — never raises). When the bandit
        infrastructure is unavailable, the decision falls back to a
        sensible default and ``ok=False``.

        `student_id` + `course_id` (optionnels) servent au calcul du
        kg_position bucket (Phase 2 — KG-aware bandit). Si fournis, on
        regarde combien de prereqs du `primary_concept` sont mastered.
        """
        try:
            bandit = await get_bandit()
        except Exception as exc:                                          # noqa: BLE001
            log.warning("bandit unavailable, using default action: %s", exc)
            return TurnDecision(
                context=ContextBucket("mixed", "normal", "medium"),
                action=_DEFAULT_ACTION,
                prompt_fragment=strategy_prompt(_DEFAULT_ACTION.strategy, language),
                ok=False,
            )

        # KG-aware : calculer le ratio de prereqs maitrises pour le concept courant
        prereqs_ratio = await _compute_prereqs_mastered_ratio(
            student_id=student_id,
            course_id=course_id,
            primary_concept=primary_concept,
        )

        context = context_from_profile(
            learning_style=learning_style,
            avg_response_time_s=avg_response_time_s,
            mastery_score=mastery_score,
            prereqs_mastered_ratio=prereqs_ratio,
        )
        action = bandit.select(context)
        decision = TurnDecision(
            context=context,
            action=action,
            prompt_fragment=strategy_prompt(action.strategy, language),
            ok=True,
        )

        # Stash the pending decision for the next-turn reward computation.
        # We capture mastery_before NOW so the next turn can compute the
        # delta against the *actual* mastery before this turn's LLM reply.
        try:
            await record_pending(session_id, PendingDecision(
                context=context,
                action=action,
                mastery_before=float(mastery_score),
                primary_concept=primary_concept or "",
                timestamp=time.time(),
            ))
        except Exception as exc:                                          # noqa: BLE001
            log.warning("bandit record_pending failed: %s", exc)

        log.info(
            "bandit.start_turn session=%s ctx=%s arm=%s",
            session_id[:8] if session_id else "?",
            context.bucket_key, action.arm_id,
        )
        return decision

    async def end_turn(
        self,
        *,
        session_id: str,
        confusion_detected: bool,
        mastery_after: float,
        engaged: bool = True,
    ) -> bool:
        """Resolve any pending decision for ``session_id``.

        Returns True if a pending decision was found and the bandit was
        updated, False otherwise (no pending = first turn of session,
        or pending expired, or bandit unavailable).
        """
        try:
            pending = await consume_pending(session_id)
        except Exception as exc:                                          # noqa: BLE001
            log.warning("bandit consume_pending failed: %s", exc)
            return False
        if pending is None:
            return False

        outcome = TurnOutcome(
            confusion_detected=bool(confusion_detected),
            mastery_before=float(pending.mastery_before),
            mastery_after=float(mastery_after),
            engaged=bool(engaged),
        )
        reward = compute_reward(outcome)

        try:
            bandit = await get_bandit()
            bandit.update(pending.context, pending.action, reward)
            mark_dirty()
            await maybe_save_periodically()
        except Exception as exc:                                          # noqa: BLE001
            log.warning("bandit update failed: %s", exc)
            return False

        log.info(
            "bandit.end_turn session=%s arm=%s reward=%.3f (Δm=%.3f confused=%s engaged=%s)",
            session_id[:8] if session_id else "?",
            pending.action.arm_id, reward,
            mastery_after - pending.mastery_before, confusion_detected, engaged,
        )
        return True
