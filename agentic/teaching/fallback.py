"""FallbackNarrator — safe-by-construction narration when grounding fails.

Triggered by the Teaching Graph when the reviewer judges the narrator's
output ungrounded *after* the retry budget is exhausted (see
``review_router`` in ``reviewer.py``). Without this node the graph would
ship the ungrounded narration silently — the student would hear
potentially hallucinated content with no signal of failure.

# Design: grounded by construction

The fallback prompt receives ONLY the source slide. No RAG chunks, no
dialogue history, no plan, no personalization prefix. Anything the LLM
writes can only come from one of two places:

  - the slide text (legitimate)
  - the model's pretrained parameters (illegitimate, but can't reference
    course content the model wasn't trained on)

Combined with a strict instruction ("rewrite, don't add"), this drives
the output toward a faithful textual paraphrase of the slide. It's not
literally a copy-paste — that would make the TTS output stilted — but
the search space is small enough that the reviewer's grounding criterion
is met by construction.

The fallback annotates ``state.actions`` with ``fallback=True`` and
``reason="ungrounded_after_retry"`` so downstream consumers (UI, logs,
analytics) can flag it as a degraded answer.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from agentic.schemas import Action, ReviewResult
from agentic.state import TutorState

log = logging.getLogger("agentic.teaching.fallback")


_FALLBACK_PROMPT_FR = """Tu dois reformuler le contenu d'une slide de cours en 2-3 phrases naturelles, comme un prof qui parle.

CONTRAINTES STRICTES :
- Utilise UNIQUEMENT les informations de la slide ci-dessous.
- N'invente RIEN qui n'y figure pas.
- Pas de markdown, pas de [1], pas de "selon la source".
- Style oral, pédagogique, court.

SLIDE :
{slide_content}

Réponds UNIQUEMENT en JSON strict, sans markdown :
{{"answer": "..."}}
"""


_FALLBACK_PROMPT_EN = """Reword the content of a course slide in 2-3 natural spoken sentences, like a teacher talking.

STRICT CONSTRAINTS:
- Use ONLY the information from the slide below.
- Do NOT invent anything not in it.
- No markdown, no [1], no "according to the source".
- Spoken style, pedagogical, short.

SLIDE:
{slide_content}

Reply STRICT JSON ONLY, no markdown:
{{"answer": "..."}}
"""


# Operational cap: how much slide text to feed the fallback LLM. Same
# 2000-char budget the main narrator uses, kept here for symmetry.
_SLIDE_CAP = 2000


def _extract_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _verbatim_fallback(slide: str, lang: str) -> str:
    """Last-resort fallback when even the LLM-based fallback fails.

    Returns the slide text verbatim with a one-line preamble. Grounded
    by literal copy — impossible to hallucinate from a copy.
    """
    if lang == "fr":
        preamble = "Voici ce que dit la slide :"
    else:
        preamble = "Here is what the slide says:"
    return f"{preamble} {slide.strip()[:1500]}"


class FallbackNarratorAgent:
    """Generates a grounded-by-construction reply when the reviewer rejects
    the main narration after the retry budget is exhausted."""

    def __init__(self, brain) -> None:
        self.brain = brain

    def __call__(self, state: TutorState) -> dict[str, Any]:
        start = time.time()
        slide = (state.get("last_slide_content") or "").strip()
        lang = (state.get("language") or "fr")[:2]
        prior_review = state.get("review")
        prior_feedback = ""
        if isinstance(prior_review, ReviewResult):
            prior_feedback = (prior_review.feedback or "")[:120]

        # If the slide itself is empty, the LLM has no source to anchor on.
        # Return an honest "no source" reply rather than letting the model
        # fill the void from its pretraining.
        if not slide:
            log.warning("fallback: no slide content available, returning honest 'no source' reply")
            answer = (
                "Je n'ai pas assez d'éléments pour répondre précisément ici."
                if lang == "fr"
                else "I don't have enough material to answer precisely here."
            )
            return _build_state_update(answer, prior_feedback, fallback_kind="no_source", start=start, state=state)

        template = _FALLBACK_PROMPT_FR if lang == "fr" else _FALLBACK_PROMPT_EN
        prompt = template.format(slide_content=slide[:_SLIDE_CAP])

        try:
            raw, _ = self.brain.ask(prompt, reply_language=lang, session_id=state.get("session_id"))
        except Exception as exc:
            log.warning("fallback LLM call failed: %s — using verbatim fallback", exc)
            answer = _verbatim_fallback(slide, lang)
            return _build_state_update(answer, prior_feedback, fallback_kind="verbatim", start=start, state=state)

        data = _extract_json(raw)
        answer = ""
        if data:
            answer = str(data.get("answer", "")).strip()

        if not answer:
            log.warning("fallback: LLM JSON parse failed → verbatim fallback")
            answer = _verbatim_fallback(slide, lang)
            kind = "verbatim"
        else:
            kind = "llm_paraphrase"

        log.info(
            "fallback: ungrounded narration replaced (kind=%s, %d chars)",
            kind, len(answer),
        )
        return _build_state_update(answer, prior_feedback, fallback_kind=kind, start=start, state=state)


def _build_state_update(
    answer: str,
    prior_feedback: str,
    fallback_kind: str,
    start: float,
    state: TutorState,
) -> dict[str, Any]:
    """Common state update for all fallback exit paths."""
    actions = list(state.get("actions") or [])
    actions.append(Action(
        type="answer",
        payload={
            "fallback": True,
            "reason": "ungrounded_after_retry",
            "kind": fallback_kind,
            "prior_review_feedback": prior_feedback,
        },
    ))
    return {
        "answer": answer,
        "actions": actions,
        # Confidence is conservative: the fallback is grounded by
        # construction but it's still a degraded answer (the model
        # couldn't deliver the planned narration). Surface it as a
        # mid-confidence grounded reply.
        "confidence": 0.5,
        # The reply is grounded in the slide by construction. Mark the
        # review accordingly so downstream consumers don't see a stale
        # "not grounded" verdict from the rejected narration.
        "review": ReviewResult(
            grounded=True,
            score=1.0,
            feedback=f"fallback narration ({fallback_kind})",
        ),
        "timings": {**state.get("timings", {}), "fallback": round(time.time() - start, 3)},
    }
