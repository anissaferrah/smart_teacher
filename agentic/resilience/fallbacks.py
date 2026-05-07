"""Deterministic fallbacks for each LangGraph node.

Each fallback returns a partial-state dict with the *same keys* the original
node would have written, so downstream nodes can run unchanged. The goal is
**graceful degradation** — never an empty turn from the user's perspective,
even if every LLM call timed out.

# Design rules

1. **No I/O, no LLM** — fallbacks must be pure-Python and run in <10ms.
2. **Honest output** — never fabricate confident answers. When we can't
   reason, we say so (e.g. "Désolé, je n'arrive pas à répondre maintenant").
3. **Schema-faithful** — match the original node's return shape exactly so
   the graph state stays valid (e.g. `intent` returns a `VoiceIntent`).
4. **Locale-aware** — every user-facing string keys off ``state.language``.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from agentic.schemas import (
    Action,
    Idea,
    PresentationPlan,
    ReviewResult,
    VoiceIntent,
)

log = logging.getLogger("agentic.resilience.fallbacks")


def _lang(state: dict) -> str:
    return (state.get("language") or "fr")[:2]


def _timing(state: dict, node: str, start: float, fallback: bool = True) -> dict:
    """Append node timing + a `__fallback_used` marker for telemetry."""
    timings = {**state.get("timings", {}), node: round(time.time() - start, 3)}
    used = list(state.get("__fallback_used", []))
    if fallback and node not in used:
        used.append(node)
    return {"timings": timings, "__fallback_used": used}


# ── Teaching graph fallbacks ──────────────────────────────────────────

def planner_fallback(state: dict) -> dict[str, Any]:
    """Build a minimal 1-idea plan from the slide so the narrator can still run."""
    start = time.time()
    slide = (state.get("last_slide_content") or "").strip()
    section_title = (state.get("section_title") or "").strip() or "Concept"
    chapter_title = (state.get("chapter_title") or "").strip() or "Cours"
    brief = slide[:300] if slide else (
        "Pas de source disponible." if _lang(state) == "fr" else "No source available."
    )
    plan = PresentationPlan(
        section_title=section_title[:120],
        chapter_title=chapter_title[:120],
        ideas=[Idea(id="idea-fallback", type="explication", content_brief=brief, depth="normal")],
    )
    return {"plan": plan, **_timing(state, "planner", start)}


def context_fallback(state: dict) -> dict[str, Any]:
    """No RAG → empty chunks. Narrator handles slide-only generation."""
    start = time.time()
    return {"retrieved_chunks": [], **_timing(state, "context", start)}


def adaptation_fallback(state: dict) -> dict[str, Any]:
    """Identity: keep the plan as-is, no profile-based tweaks."""
    start = time.time()
    return _timing(state, "adaptation", start)


def narrator_fallback(state: dict) -> dict[str, Any]:
    """Last-resort: echo a truncated slide as narration so the turn is not empty."""
    start = time.time()
    lang = _lang(state)
    slide = (state.get("last_slide_content") or "").strip()
    if slide:
        # Emit an honest fallback notice so the student knows it's degraded
        prefix = (
            "Voici l'essentiel de la slide (j'ai eu un souci pour la reformuler) : "
            if lang == "fr"
            else "Here is the slide content (I had trouble rephrasing it): "
        )
        narration = prefix + slide[:600]
    else:
        narration = (
            "Je n'arrive pas à présenter cette section pour le moment."
            if lang == "fr"
            else "I can't present this section right now."
        )
    return {
        "answer": narration,
        "last_narrated_idea": None,
        **_timing(state, "narrator", start),
    }


def review_fallback(state: dict) -> dict[str, Any]:
    """Without LLM, trust the narration (better than gating a degraded turn)."""
    start = time.time()
    return {
        "review": ReviewResult(grounded=True, score=0.7, feedback="reviewer fallback"),
        **_timing(state, "review", start),
    }


# ── Q&A graph fallbacks ───────────────────────────────────────────────

def intent_fallback(state: dict) -> dict[str, Any]:
    """Default to 'question' — the safest interpretation when we don't know."""
    start = time.time()
    payload = state.get("event_payload") or {}
    raw_text = ""
    if isinstance(payload, dict):
        raw_text = payload.get("text") or payload.get("transcript") or ""
    return {
        "intent": VoiceIntent(type="question", confidence=0.3, payload={"raw_text": raw_text}),
        **_timing(state, "intent", start),
    }


def rewriter_fallback(state: dict) -> dict[str, Any]:
    """Use the raw question — no pronoun resolution but better than nothing."""
    start = time.time()
    intent = state.get("intent")
    raw = ""
    if intent and isinstance(getattr(intent, "payload", None), dict):
        raw = intent.payload.get("raw_text", "") or ""
    if not raw:
        payload = state.get("event_payload") or {}
        if isinstance(payload, dict):
            raw = payload.get("text") or payload.get("transcript") or ""
    return {"rewritten_query": (raw or "").strip()[:512], **_timing(state, "rewriter", start)}


def retriever_fallback(state: dict) -> dict[str, Any]:
    """Empty chunks — Responder will rely on the slide context only."""
    start = time.time()
    return {"retrieved_chunks": [], **_timing(state, "retriever", start)}


def responder_fallback(state: dict) -> dict[str, Any]:
    """Honest 'try again' message; never fake an answer we can't ground."""
    start = time.time()
    lang = _lang(state)
    answer = (
        "Désolé, je n'arrive pas à répondre tout de suite — peux-tu reformuler ?"
        if lang == "fr"
        else "Sorry, I can't answer right now — could you rephrase?"
    )
    return {
        "answer": answer,
        "actions": [Action(type="answer", payload={"intent": "fallback"})],
        "confidence": 0.2,
        **_timing(state, "responder", start),
    }


# ── Registry (graph builders read from this) ──────────────────────────

NODE_FALLBACKS: dict[str, Callable[[dict], dict[str, Any]]] = {
    "planner":    planner_fallback,
    "context":    context_fallback,
    "adaptation": adaptation_fallback,
    "narrator":   narrator_fallback,
    "review":     review_fallback,
    "intent":     intent_fallback,
    "rewriter":   rewriter_fallback,
    "retriever":  retriever_fallback,
    "responder":  responder_fallback,
}


def fallback_for(node_name: str) -> Callable[[dict], dict[str, Any]]:
    """Lookup with a no-op default so unknown nodes don't break the graph."""
    fn = NODE_FALLBACKS.get(node_name)
    if fn is None:
        log.warning("no fallback registered for node '%s' — returning identity", node_name)
        return lambda state: {}
    return fn
