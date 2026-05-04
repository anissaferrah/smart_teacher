"""Voice-navigation dispatcher.

When the IntentAgent classifies a student utterance as ``navigation`` and
the ResponderAgent emits ``Action(type="navigate", payload={nav_action,
nav_target})``, the WS handler calls :func:`dispatch_nav_action` to
actually perform the navigation. Without this, the tutor would only
*say* "OK, je passe à la suivante" without ever moving.

Why a dedicated module rather than inline in ``handlers/ws.py`` :

  - ws.py is already ~3200 lines and the per-action branches would add
    another 200+. Keeping them here lets the WS code stay readable and
    the dispatch logic become unit-testable in isolation (pure
    dependency injection — every external is a parameter).
  - The same dispatcher is reusable from non-WS callers (REST, tests,
    a future MCP tool, etc.) — anything that has a ``dialogue``, a
    ``send`` callable and a session ``ctx``.

# Sub-actions

The 7 ``nav_action`` values are exhaustive (see ``agentic.qa.intent.
_VALID_NAV_ACTIONS``) and each one maps to a deterministic side-effect :

    next           → dialogue.next_section + UI "next_section" event
    previous       → dialogue.prev_section + UI "prev_section" event
    repeat         → re-emit the current slide for re-presentation
    skip           → alias of ``next`` (semantic only — same effect)
    go_to_concept  → KG lookup → save_course_position → UI slide_update
    explain_more   → ask the UI to re-trigger presentation with depth=deep
    slow_down      → adjust the per-session speech_rate down × 0.85,
                     send the new rate so the next TTS uses it

Unknown actions fall back to ``next`` — same defensive default the
IntentAgent already applies, so the contract is consistent end-to-end.

# Return value

Returns a small dict ``{"action": str, "ok": bool, "detail": str}``
mostly for logging + tests. The ``send`` callable is responsible for
delivering UI events ; the dispatcher itself doesn't await any user
acknowledgement.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger("services.nav_dispatcher")

# Same set as agentic.qa.intent._VALID_NAV_ACTIONS — duplicated here on
# purpose so this module has no dependency on the agentic layer (which
# pulls in LLM clients). Keep the two in sync if either grows.
_VALID_NAV_ACTIONS = frozenset({
    "next", "previous", "repeat", "skip",
    "go_to_concept", "explain_more", "slow_down",
})

# Sourced from Config (SPEECH_RATE_SLOW_DOWN_FACTOR / SPEECH_RATE_FLOOR).
# 0.85 = ~15 % slower, perceptible but still understandable. Floor at
# 0.5 so we never end up with TTS so slow it sounds broken — Edge-TTS
# gets unstable below ~0.5×.
from core.config import Config as _NavConfig
_SLOW_DOWN_FACTOR = _NavConfig.SPEECH_RATE_SLOW_DOWN_FACTOR
_SPEECH_RATE_FLOOR = _NavConfig.SPEECH_RATE_FLOOR

# Send type strings — kept as constants so test code and the frontend
# share the same vocabulary without typos drifting.
_EVT_NEXT_SECTION = "next_section"
_EVT_PREV_SECTION = "prev_section"
_EVT_SLIDE_UPDATE = "slide_update"
_EVT_EXPLAIN_MORE = "explain_more_request"
_EVT_SPEECH_RATE  = "speech_rate_changed"


def _slide_payload_from_ctx(slide_ctx: dict | None,
                            fallback_section_idx: int = 0) -> dict[str, Any]:
    """Build the WS ``slide_update`` payload from a slide_ctx dict.

    Mirrors the shape used by the voice-command bypass path
    (``handlers/ws.py:2299``) so the frontend doesn't need a separate
    handler — it sees the same event regardless of which path produced
    it.
    """
    sc = slide_ctx or {}
    return {
        "type": _EVT_SLIDE_UPDATE,
        "presentation_request_id": "",
        "slide_type":     sc.get("slide_type", "section"),
        "slide_index":    sc.get("slide_index", fallback_section_idx),
        "slide_title":    sc.get("section_title") or sc.get("chapter_title", ""),
        "slide_content":  sc.get("content", ""),
        "content_original": sc.get("content", ""),
        "image_url":      sc.get("slide_path", ""),
        "slide_path":     sc.get("slide_path", ""),
        "keywords":       sc.get("keywords", []),
        "chapter":        sc.get("chapter_title", ""),
        "chapter_index":  sc.get("chapter_order", 0),
        "section_index":  sc.get("slide_index", fallback_section_idx),
        "section_title":  sc.get("section_title", ""),
        "course_id":      sc.get("course_id", ""),
        "course_title":   sc.get("course_title", ""),
        "course_domain":  sc.get("course_domain", "general"),
        "progress_pct":   sc.get("progress_pct", 0),
    }


async def _resolve_concept_target(
    nav_target: str,
    course_id: str | None,
    *,
    kg_getter: Callable[[], Any] | None = None,
) -> tuple[Optional[int], Optional[int], Optional[Any]]:
    """Look ``nav_target`` up in the KG → return (chapter_idx, section_idx, concept).

    Resolution order :
      1. Exact ``get_concept(name)`` (the IntentAgent has already capped
         the target name at 80 chars).
      2. Fuzzy : iterate ``list_concepts(course_id)`` and accept the first
         concept whose ``display_name`` / ``canonical_name`` substring-matches
         (case-insensitive). Concepts in the KG often live as snake_case
         (k_means) while the student says "K-means" — the substring match
         bridges both.

    Returns ``(None, None, None)`` if not found ; the caller falls back to
    the safe-default ``next`` behaviour rather than crashing.
    """
    if not nav_target or kg_getter is None:
        return None, None, None
    try:
        kg = kg_getter()
    except Exception as exc:                                          # noqa: BLE001
        log.debug("nav_dispatcher: kg unavailable (%s)", exc)
        return None, None, None
    if kg is None:
        return None, None, None

    needle = nav_target.strip().lower()

    # Exact match first
    concept = None
    try:
        concept = kg.get_concept(needle) or kg.get_concept(nav_target)
    except Exception:                                                  # noqa: BLE001
        concept = None

    # Fuzzy substring match across the course's concepts
    if concept is None:
        try:
            for ci in kg.list_concepts(course_id):
                names = (
                    str(getattr(ci, "name", "") or "").lower(),
                    str(getattr(ci, "display_name", "") or "").lower(),
                    str(getattr(ci, "canonical_name", "") or "").lower(),
                )
                if any(needle in n or n in needle for n in names if n):
                    concept = ci
                    break
        except Exception as exc:                                       # noqa: BLE001
            log.debug("nav_dispatcher: kg.list_concepts failed (%s)", exc)
            return None, None, None

    if concept is None:
        return None, None, None

    # The chapter is the smallest one in chapter_idxs (concept may span
    # multiple chapters ; jumping to the earliest is the most pedagogical
    # — that's where the concept is *introduced*).
    chap_idxs = getattr(concept, "chapter_idxs", None) or set()
    chap_idx = min(chap_idxs) if chap_idxs else None
    # Section : pick the smallest section_idx among the ideas in this
    # concept that live in chap_idx.
    section_idx: Optional[int] = None
    try:
        ideas = kg.ideas_in_concept(getattr(concept, "name", ""))
        relevant = [i for i in ideas
                    if getattr(i, "chapter_idx", None) == chap_idx]
        if relevant:
            section_idx = min(int(getattr(i, "section_idx", 0) or 0)
                              for i in relevant)
    except Exception:                                                  # noqa: BLE001
        section_idx = None
    if chap_idx is None and section_idx is None:
        return None, None, concept
    return chap_idx, section_idx, concept


async def dispatch_nav_action(
    *,
    nav_action: str,
    nav_target: str,
    ctx,
    dialogue,
    send: Callable[[dict], Awaitable[None]],
    turn_id: int,
    slide_loader: Callable[..., Awaitable[dict | None]] | None = None,
    kg_getter: Callable[[], Any] | None = None,
    profile_updater: Callable[..., Awaitable[Any]] | None = None,
    current_speech_rate: float = 1.0,
) -> dict[str, Any]:
    """Execute one of the 7 navigation sub-actions.

    Args :
        nav_action          : one of _VALID_NAV_ACTIONS (others → "next")
        nav_target          : free text, only used by ``go_to_concept``
        ctx                 : SessionContext (course_id, chapter_index,
                              section_index, session_id)
        dialogue            : DialogueManager (next_section / prev_section
                              / save_course_position)
        send                : async callable shipping a dict to the WS
        turn_id             : the active turn — included in every UI event
                              so the client can de-dupe stale events
        slide_loader        : ``load_course_slide_context`` ; injected
                              for testability (None → repeat / go_to_concept
                              fall back to a minimal slide_update)
        kg_getter           : returns the KnowledgeGraph singleton ;
                              injected so tests don't have to bootstrap
                              the whole RAG
        profile_updater     : ``update_profile`` ; injected so the
                              ``slow_down`` path can persist the new rate
        current_speech_rate : starting rate (read from the profile by
                              the caller) ; the new rate is computed as
                              max(_SPEECH_RATE_FLOOR, current × 0.85)

    Returns :
        ``{"action": effective_action, "ok": bool, "detail": str}``
        — useful for the WS log line + the dispatcher tests.
    """
    if not ctx:
        return {"action": nav_action, "ok": False, "detail": "no ctx"}

    action = (nav_action or "").strip().lower()
    log.info(
        "🧭 nav_dispatcher: action=%r target=%r ctx={course=%s, ch=%s, sec=%s}",
        action, (nav_target or "")[:40],
        getattr(ctx, "course_id", "?"),
        getattr(ctx, "chapter_index", "?"),
        getattr(ctx, "section_index", "?"),
    )
    if action not in _VALID_NAV_ACTIONS:
        log.info("nav_dispatcher: unknown action %r → defaulting to 'next'", action)
        action = "next"

    # ── next / skip ──────────────────────────────────────────────────
    if action in ("next", "skip"):
        try:
            await dialogue.next_section(ctx.session_id)
        except Exception as exc:                                        # noqa: BLE001
            log.debug("nav_dispatcher: next_section failed (%s)", exc)
        await send({"type": _EVT_NEXT_SECTION, "turn_id": turn_id})
        return {"action": action, "ok": True, "detail": "advanced one section"}

    # ── previous ─────────────────────────────────────────────────────
    if action == "previous":
        try:
            await dialogue.prev_section(ctx.session_id)
        except Exception as exc:                                        # noqa: BLE001
            log.debug("nav_dispatcher: prev_section failed (%s)", exc)
        await send({"type": _EVT_PREV_SECTION, "turn_id": turn_id})
        return {"action": "previous", "ok": True, "detail": "moved back one section"}

    # ── repeat ───────────────────────────────────────────────────────
    if action == "repeat":
        slide_ctx = None
        if slide_loader is not None and ctx.course_id is not None:
            try:
                slide_ctx = await slide_loader(
                    ctx.course_id, ctx.chapter_index, ctx.section_index,
                )
            except Exception as exc:                                    # noqa: BLE001
                log.debug("nav_dispatcher: slide_loader failed (%s)", exc)
        payload = _slide_payload_from_ctx(slide_ctx, ctx.section_index)
        payload["turn_id"] = turn_id
        await send(payload)
        return {"action": "repeat", "ok": True, "detail": "re-presenting current slide"}

    # ── go_to_concept ────────────────────────────────────────────────
    if action == "go_to_concept":
        target = (nav_target or "").strip()
        if not target:
            log.info("nav_dispatcher: go_to_concept with empty target → fallback to next")
            try:
                await dialogue.next_section(ctx.session_id)
            except Exception:                                           # noqa: BLE001
                pass
            await send({"type": _EVT_NEXT_SECTION, "turn_id": turn_id})
            return {"action": "go_to_concept", "ok": False,
                    "detail": "empty target, defaulted to next"}

        chap_idx, sec_idx, concept = await _resolve_concept_target(
            target, ctx.course_id, kg_getter=kg_getter,
        )
        if concept is None or chap_idx is None:
            log.info("nav_dispatcher: concept %r not found in KG → fallback to next", target[:40])
            try:
                await dialogue.next_section(ctx.session_id)
            except Exception:                                           # noqa: BLE001
                pass
            await send({"type": _EVT_NEXT_SECTION, "turn_id": turn_id})
            return {"action": "go_to_concept", "ok": False,
                    "detail": f"concept '{target[:40]}' not found"}

        section_idx = sec_idx if sec_idx is not None else 0
        try:
            await dialogue.save_course_position(
                ctx.session_id,
                course_id=ctx.course_id,
                chapter_index=chap_idx,
                section_index=section_idx,
                char_pos=0,
            )
        except Exception as exc:                                        # noqa: BLE001
            log.debug("nav_dispatcher: save_course_position failed (%s)", exc)

        slide_ctx = None
        if slide_loader is not None and ctx.course_id is not None:
            try:
                slide_ctx = await slide_loader(ctx.course_id, chap_idx, section_idx)
            except Exception as exc:                                    # noqa: BLE001
                log.debug("nav_dispatcher: slide_loader failed for go_to_concept (%s)", exc)

        payload = _slide_payload_from_ctx(slide_ctx, section_idx)
        payload["chapter_index"] = chap_idx
        payload["section_index"] = section_idx
        payload["turn_id"] = turn_id
        # Annotate so the client can show "→ K-means" in the navigation toast
        payload["concept_target"] = getattr(concept, "display_name", "") or getattr(concept, "name", "")
        await send(payload)
        return {"action": "go_to_concept", "ok": True,
                "detail": f"jumped to concept '{getattr(concept, 'name', '')}' "
                          f"at chapter={chap_idx} section={section_idx}"}

    # ── explain_more ─────────────────────────────────────────────────
    if action == "explain_more":
        # The actual deeper re-explanation is a teaching-graph job (the
        # narrator can re-plan with depth='deep'). Here we just signal
        # the request — the WS handler / frontend re-trigger the
        # presentation flow. Keeping the deep-presentation pipeline
        # responsibility out of this module keeps the dispatcher lean.
        await send({
            "type": _EVT_EXPLAIN_MORE,
            "turn_id": turn_id,
            "depth": "deep",
            "section_index": ctx.section_index,
            "chapter_index": ctx.chapter_index,
        })
        return {"action": "explain_more", "ok": True,
                "detail": "explain_more event emitted (frontend re-triggers presentation)"}

    # ── slow_down ────────────────────────────────────────────────────
    if action == "slow_down":
        new_rate = max(_SPEECH_RATE_FLOOR,
                       float(current_speech_rate or 1.0) * _SLOW_DOWN_FACTOR)
        if profile_updater is not None:
            try:
                await profile_updater(
                    ctx.session_id,
                    {"speech_rate": new_rate},
                    course_id=ctx.course_id,
                )
            except Exception as exc:                                    # noqa: BLE001
                log.debug("nav_dispatcher: profile_updater failed (%s)", exc)
        await send({
            "type": _EVT_SPEECH_RATE,
            "turn_id": turn_id,
            "speech_rate": round(new_rate, 3),
            "previous_rate": round(float(current_speech_rate or 1.0), 3),
        })
        return {"action": "slow_down", "ok": True,
                "detail": f"speech_rate {current_speech_rate:.2f} → {new_rate:.2f}"}

    # Unreachable (every valid action is handled above), but keeps the
    # type checker happy.
    return {"action": action, "ok": False, "detail": "no handler matched"}
