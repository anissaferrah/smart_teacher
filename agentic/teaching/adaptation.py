"""AdaptationAgent — adjusts the plan based on student profile and rules.

Pure rule-based + profile lookup, no LLM. Tweaks the depth of each idea
according to:
    - student_level (collège/lycée/université)
    - learning_style (visual/auditory/mixed) — adds example slots if visual
    - confusion_history — concepts the student previously struggled with
      get bumped to depth=deep so the Narrator elaborates them.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from agentic.schemas import Idea, PresentationPlan
from agentic.state import TutorState

log = logging.getLogger("agentic.teaching.adaptation")


def _bump_depth(current: str, direction: str) -> str:
    """Move depth up (deepen) or down (simplify)."""
    order = ["shallow", "normal", "deep"]
    try:
        i = order.index(current)
    except ValueError:
        i = 1
    if direction == "deepen":
        i = min(len(order) - 1, i + 1)
    elif direction == "simplify":
        i = max(0, i - 1)
    return order[i]


class AdaptationAgent:
    """Tweaks the plan ideas based on student level and history."""

    def __init__(self, profile_mgr=None) -> None:
        self.profile_mgr = profile_mgr

    async def __call__(self, state: TutorState) -> dict[str, Any]:
        start = time.time()
        plan: PresentationPlan | None = state.get("plan")
        if not plan or not plan.ideas:
            return {
                "timings": {**state.get("timings", {}), "adaptation": 0.0},
            }

        level = state.get("student_level") or "lycée"
        confused_topics: set[str] = set()

        # Try to enrich with profile data (best effort — don't fail the graph)
        if self.profile_mgr and state.get("session_id"):
            try:
                profile = await self.profile_mgr.get_or_create(
                    state["session_id"],
                    language=state.get("language", "fr"),
                    level=level,
                )
                # confused_topics may be on the profile under various names
                topics = (
                    getattr(profile, "confused_topics", None)
                    or getattr(profile, "confusion_topics", None)
                    or []
                )
                confused_topics = {str(t).lower() for t in topics if t}
            except Exception as exc:
                log.debug("adaptation: profile fetch skipped: %s", exc)

        # Build new ideas list (don't mutate originals)
        new_ideas: list[Idea] = []
        bumps_for_confusion = 0
        for idea in plan.ideas:
            new_depth = idea.depth

            # Rule 1: simplify for collège, deepen for université
            if level == "collège":
                new_depth = _bump_depth(new_depth, "simplify")
            elif level == "université" and idea.type == "concept":
                new_depth = _bump_depth(new_depth, "deepen")

            # Rule 2: deepen ideas that match a previously confused topic
            brief_lower = (idea.content_brief or "").lower()
            if any(t in brief_lower for t in confused_topics):
                new_depth = _bump_depth(new_depth, "deepen")
                bumps_for_confusion += 1

            new_ideas.append(replace(idea, depth=new_depth))

        adapted_plan = replace(plan, ideas=new_ideas)
        log.info(
            "🔍 adaptation: level=%s confused_topics=%d bumps=%d",
            level,
            len(confused_topics),
            bumps_for_confusion,
        )
        if confused_topics:
            log.info("🔍   adaptation confused_topics : %s", list(confused_topics)[:5])
        for i, idea in enumerate(adapted_plan.ideas):
            log.info(
                "🔍   adapted_plan[%d] type=%s depth=%s",
                i, getattr(idea, "type", "?"), getattr(idea, "depth", "?"),
            )
        return {
            "plan": adapted_plan,
            "timings": {**state.get("timings", {}), "adaptation": round(time.time() - start, 3)},
        }
