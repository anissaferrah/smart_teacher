"""IntentAgent — classify the student's utterance.

Output: VoiceIntent with type ∈ {"question", "navigation", "feedback",
"confusion_signal"} plus a Self-RAG ``needs_retrieval`` flag for
"question" intents (Asai et al. 2023, arXiv:2310.11511).

# Architecture: learned-only (no rules)

Two tiers, both learned:

  1. SIGHT model (xlm-roberta fine-tuned, ~50-100ms) — semantic confusion
     classifier. Trusts the model's binary verdict; no extra threshold.
  2. LLM (Ollama / OpenAI, ~1-15s)                   — multi-class intent +
     Self-RAG retrieval signal in a single JSON output.

# What used to be here

A previous version had three regex-based pre-filters (``_NAV_PATTERNS``,
``_FEEDBACK_PATTERNS``, ``_CONFUSION_PATTERNS``) that classified
utterances on keyword matches before reaching the LLM. They were fast
(0ms) and worked for canonical phrasings ("oui", "next slide", "je
comprends pas"), but they have three structural problems that make them
unfit for a research-grade tutor:

  - **Non-generalisation**: misspellings, paraphrases, code-switching,
    or unusual phrasing fall through. "I do not understand" matched but
    "ça m'échappe complètement" didn't.
  - **No calibration**: the confidence floats (0.95 / 0.9 / 0.85) were
    hand-picked to make downstream gates work, not measured.
  - **Locked taxonomy**: adding a new intent class meant editing regex,
    not retraining a classifier.

The cost of removing them is latency: every utterance now pays the SIGHT
inference (~50-100ms), and every non-confusion utterance pays the LLM
classification (~1-15s on Ollama CPU). This is the deliberate trade-off
for a learned-only architecture. Mitigations live elsewhere: a small
fine-tuned classifier or embedding-similarity router can be added as a
fast learned tier later (still no rules).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from agentic.schemas import VoiceIntent
from agentic.state import TutorState

log = logging.getLogger("agentic.qa.intent")

# SIGHT confusion classifier (lazy-loaded). Learned model, not rules.
try:
    from pedagogy.confusion.detector import predict_confusion as _sight_predict
except Exception as _sight_import_exc:  # pragma: no cover
    _sight_predict = None
    log.debug("SIGHT import failed (intent agent): %s", _sight_import_exc)


_INTENT_PROMPT_FR = """Tu aides un système de tutorat à comprendre une phrase d'étudiant.

Historique récent (dialogue):
{history}

Slide en cours :
{slide_excerpt}

Phrase brute de l'étudiant : "{utterance}"

ÉTAPE 1 — Classe la phrase en UNE seule catégorie :
- "question"          : demande d'information ou explication
- "navigation"        : demande de revenir/passer à une slide/chapitre
- "feedback"          : oui/non/c'est clair/continue
- "confusion_signal"  : exprime explicitement la confusion

ÉTAPE 2 — Champs supplémentaires selon le type :

Si "question" :
  a) "needs_retrieval": false UNIQUEMENT pour :
       - méta-questions sur la conversation ("tu as dit quoi ?", "répète")
       - calculs purs ("combien fait 7 × 8")
       - bavardage ("merci", "d'accord")
     Sinon "needs_retrieval": true (par défaut sûr).

  b) "is_definition": true si c'est une demande de DÉFINITION d'un terme
     technique standard. Exemples : "qu'est-ce que K-NN ?" → true ;
     "explique cette slide" → false (pas un terme spécifique).

  c) "definition_term": SI is_definition=true, donne le terme exact à
     définir (le nom du concept, sans le verbe interrogatif).
     SI is_definition=false, mets une chaîne vide "".

  d) "anchored_concept": le concept principal sur lequel porte la question
     (en t'appuyant sur l'historique et la slide). Court, < 60 caractères.

  e) "needed_rewrite": true SI la question brute est ambiguë sans
     contexte (pronoms, ellipses, références au dialogue). Sinon false.

  f) "rewritten": SI needed_rewrite=true, réécris la question en français
     pour qu'elle soit auto-suffisante : résous les pronoms, expanse les
     ellipses, garde l'intention. SI needed_rewrite=false, recopie la
     question brute exactement.

  REMARQUE : même si is_definition=true, GARDE needs_retrieval=true. Le
  système va d'abord chercher dans le cours.

Si "feedback" : indique la polarité.
  - "feedback_polarity": "positive" si l'étudiant valide / veut continuer
  - "feedback_polarity": "negative" s'il rejette / demande de réexpliquer

Si "navigation" : précise l'action exacte.
  "nav_action" doit être UNE de ces 7 valeurs :
    - "next"           : passer à la suivante (slide / chapitre / section)
    - "previous"       : revenir en arrière
    - "repeat"         : refaire/réécouter la même slide
    - "skip"           : sauter ce contenu / passer au suivant en ignorant
    - "go_to_concept"  : aller à un concept précis (l'étudiant a NOMMÉ un concept)
    - "explain_more"   : approfondir, donner plus de détails sur le même contenu
    - "slow_down"      : trop vite, ralentir le débit / la vitesse de parole
  Si l'étudiant n'a pas été clair, choisis "next" par défaut sûr.
  Si nav_action == "go_to_concept", remplis aussi "nav_target" avec le nom
  du concept demandé (max 80 chars). Sinon mets "nav_target": "".

Si "confusion_signal" : pas de champs supplémentaires.

Réponds UNIQUEMENT en JSON strict, sans markdown :
{{"intent": "...", "confidence": 0.0-1.0, "needs_retrieval": true|false, "is_definition": true|false, "definition_term": "...", "feedback_polarity": "positive|negative", "nav_action": "...", "nav_target": "...", "anchored_concept": "...", "needed_rewrite": true|false, "rewritten": "..."}}
"""

_INTENT_PROMPT_EN = """You help a tutoring system understand a student's utterance.

Recent dialogue history:
{history}

Current slide:
{slide_excerpt}

Raw student utterance: "{utterance}"

STEP 1 — Classify the utterance into ONE category:
- "question"          : asks for information or explanation
- "navigation"        : asks to go back/forward to a slide/chapter
- "feedback"          : yes/no/clear/continue
- "confusion_signal"  : explicitly expresses confusion

STEP 2 — Extra fields depending on type:

If "question":
  a) "needs_retrieval": false ONLY for:
       - meta-questions about the conversation ("what did you say?", "repeat")
       - pure arithmetic ("how much is 7 × 8")
       - small talk ("thanks", "ok")
     Otherwise "needs_retrieval": true (safe default).

  b) "is_definition": true if it's a request for a DEFINITION of a
     standard technical term. Example : "what is K-NN?" → true ;
     "explain this slide" → false.

  c) "definition_term": IF is_definition=true, the exact term to define
     (concept name, no question verb). IF false, empty string "".

  d) "anchored_concept": the main concept the question is about (using
     history and slide). Short, < 60 chars.

  e) "needed_rewrite": true IF the raw question is ambiguous without
     context (pronouns, ellipses, dialogue references). Else false.

  f) "rewritten": IF needed_rewrite=true, rewrite the question in English
     so it stands alone: resolve pronouns, expand ellipses, keep intent.
     IF needed_rewrite=false, copy the raw question exactly.

  NOTE: even when is_definition=true, KEEP needs_retrieval=true. The
  system will first look in the course material.

If "feedback": indicate polarity.
  - "feedback_polarity": "positive" if approves / wants to continue
  - "feedback_polarity": "negative" if rejects / asks to re-explain

If "navigation": specify the exact action.
  "nav_action" must be ONE of these 7 values:
    - "next"           : move to the next (slide / chapter / section)
    - "previous"       : go back
    - "repeat"         : redo / re-listen the same slide
    - "skip"           : skip this content / move past without listening
    - "go_to_concept"  : jump to a specific concept (student NAMED one)
    - "explain_more"   : go deeper, give more detail on the same content
    - "slow_down"      : too fast, slow down speech rate
  If unclear, default to "next" (safe).
  If nav_action == "go_to_concept", fill "nav_target" with the requested
  concept name (max 80 chars). Otherwise set "nav_target": "".

If "confusion_signal": no extra fields needed.

Reply STRICT JSON ONLY, no markdown:
{{"intent": "...", "confidence": 0.0-1.0, "needs_retrieval": true|false, "is_definition": true|false, "definition_term": "...", "feedback_polarity": "positive|negative", "nav_action": "...", "nav_target": "...", "anchored_concept": "...", "needed_rewrite": true|false, "rewritten": "..."}}
"""

_VALID_INTENTS = {"question", "navigation", "feedback", "confusion_signal"}
_VALID_POLARITIES = {"positive", "negative"}

# Granular navigation actions. The legacy ``navigation`` intent collapsed
# all of these into one bucket — fine for "go to next slide" / "go back",
# but useless to distinguish "explain more" from "skip this" or
# "slow down". The 7-way split below matches what a real classroom
# voice tutor needs to dispatch on. ``next`` is the safe default when
# the LLM can't decide.
_VALID_NAV_ACTIONS = {
    "next",            # forward by one unit (slide/section/chapter)
    "previous",        # back by one unit
    "repeat",          # re-explain the same content from scratch
    "skip",            # jump past without listening
    "go_to_concept",   # named-concept jump (uses ``nav_target``)
    "explain_more",    # deepen the current explanation
    "slow_down",       # student is overwhelmed, reduce TTS rate
}
_NAV_TARGET_CAP = 80   # protective char cap on the named-concept payload

# Operational caps (not behavioural thresholds): keep the merged
# Intent+Rewriter prompt within Ollama's 4k context.
# SLIDE_EXCERPT_CAP is shared with rewriter (same budget). HISTORY_TURNS
# / HISTORY_MSG_CAP are tighter than the responder's because the merged
# prompt must also carry the rewrite block.
from agentic.qa._shared import SLIDE_EXCERPT_CAP as _SLIDE_CAP, format_history as _format_history_shared

_HISTORY_TURNS = 4          # 2 round-trips, raw roles (no FR/EN labels)
_HISTORY_MSG_CAP = 160       # per-message cap inside this prompt's history block
_REWRITTEN_CAP = 800


def _format_history(history: list) -> str:
    if not history:
        return "(empty)"
    out = _format_history_shared(
        history,
        pairs=_HISTORY_TURNS // 2,
        msg_cap=_HISTORY_MSG_CAP,
        labelled=False,
    )
    # The shared helper returns the "start of conversation" marker for
    # tails that contain no usable rows; this prompt expects "(empty)".
    if out.startswith("(début") or out.startswith("(start"):
        return "(empty)"
    return out


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


class IntentAgent:
    """Classify the student's utterance into a VoiceIntent."""

    def __init__(self, brain) -> None:
        self.brain = brain

    def __call__(self, state: TutorState) -> dict[str, Any]:
        start = time.time()
        text = ""
        payload = state.get("event_payload") or {}
        if isinstance(payload, dict):
            text = (payload.get("text") or payload.get("transcript") or "").strip()
        text = text[:512]

        if not text:
            log.warning("intent: empty utterance → defaulting to 'question'")
            return {
                "intent": VoiceIntent(type="question", confidence=0.0, payload={"source": "empty"}),
                "timings": {**state.get("timings", {}), "intent": 0.0},
            }

        # Tier 1: SIGHT — learned semantic confusion classifier. Fast (~50-100ms).
        if _sight_predict is not None:
            try:
                pred = _sight_predict(text)
                if pred is not None and bool(getattr(pred, "confused", False)):
                    prob = float(getattr(pred, "prob", 0.0))
                    intent = VoiceIntent(
                        type="confusion_signal",
                        confidence=prob,
                        payload={
                            "source": "sight",
                            "model": getattr(pred, "model_name", "sight"),
                            "raw_text": text,
                        },
                        # confusion never needs retrieval — answer comes from
                        # reformulation strategy, not document lookup.
                        needs_retrieval=False,
                    )
                    log.info("intent: SIGHT-classified as confusion_signal (prob=%.2f)", prob)
                    return {
                        "intent": intent,
                        "timings": {**state.get("timings", {}), "intent": round(time.time() - start, 3)},
                    }
            except Exception as exc:
                log.debug("SIGHT inference failed: %s — falling back to LLM", exc)

        # Tier 2: merged Intent + Rewriter LLM call.
        # The prompt includes slide context + dialogue history so the
        # SAME call produces classification, Self-RAG signal, definition
        # extraction AND a context-resolved rewrite when applicable.
        # This collapses two sequential LLM calls (~50s each on Ollama
        # CPU) into one — the rewriter node will short-circuit when it
        # sees ``rewritten_query`` already populated in state.
        lang = (state.get("language") or "fr")[:2]
        template = _INTENT_PROMPT_FR if lang == "fr" else _INTENT_PROMPT_EN
        slide_excerpt = (state.get("last_slide_content") or "")[:_SLIDE_CAP].replace("\n", " ") or "(none)"
        history_str = _format_history(state.get("history") or [])
        prompt = template.format(
            utterance=text,
            slide_excerpt=slide_excerpt,
            history=history_str,
        )

        log.info("🔍 intent USER text : %r", text[:200])
        try:
            raw, _ = self.brain.ask(prompt, reply_language=lang)
        except Exception as exc:
            log.warning("intent: LLM call failed (%s) → defaulting to 'question'", exc)
            return {
                "intent": VoiceIntent(type="question", confidence=0.4, payload={"source": "llm_fail"}),
                "timings": {**state.get("timings", {}), "intent": round(time.time() - start, 3)},
            }
        log.info("🔍 intent LLM raw : %r", (raw or "")[:300])

        data = _extract_json(raw) or {}
        raw_type = str(data.get("intent", "question")).strip().lower()
        intent_type = raw_type if raw_type in _VALID_INTENTS else "question"
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        # Self-RAG flag from LLM. True is the safe default for any
        # uninterpretable output. Non-question intents force False since
        # they don't trigger retrieval anyway.
        if intent_type == "question":
            needs_retrieval = bool(
                data.get("needs_retrieval", data.get("needsretrieval", data.get("needsRetrieval", True)))
            )
        else:
            needs_retrieval = False

        # Feedback polarity (only meaningful for intent_type == "feedback").
        # Carried in payload so the responder can decide between "continue"
        # and "replay" without doing its own substring matching. Default
        # "positive" is the safer interpretation: don't replay unless the
        # LLM explicitly says the feedback is negative.
        payload_dict: dict[str, Any] = {"source": "llm", "raw_text": text}
        if intent_type == "feedback":
            raw_polarity = str(
                data.get("feedback_polarity") or data.get("feedbackpolarity") or data.get("feedbackPolarity") or ""
            ).strip().lower()
            payload_dict["feedback_polarity"] = (
                raw_polarity if raw_polarity in _VALID_POLARITIES else "positive"
            )

        # Navigation sub-action — granular dispatch for navigation intents.
        # The downstream handler (handlers/ws.py) routes on this field to
        # call the right service : ``next/previous/repeat/skip`` go through
        # ``DialogueManager``; ``go_to_concept`` resolves the target via
        # the KG; ``explain_more`` re-runs the planner with deeper depth ;
        # ``slow_down`` adjusts the TTS rate via ``tts_adapter``.
        # An unknown value collapses to "next" (safe default — moving
        # forward is harder to misinterpret than e.g. silently skipping).
        if intent_type == "navigation":
            raw_action = str(
                data.get("nav_action") or data.get("navaction") or data.get("navAction") or ""
            ).strip().lower()
            payload_dict["nav_action"] = (
                raw_action if raw_action in _VALID_NAV_ACTIONS else "next"
            )
            # nav_target is only meaningful for go_to_concept ; capped to
            # avoid pathological inputs polluting downstream logs.
            if payload_dict["nav_action"] == "go_to_concept":
                payload_dict["nav_target"] = str(
                    data.get("nav_target") or data.get("navtarget") or data.get("navTarget") or ""
                ).strip()[:_NAV_TARGET_CAP]
            else:
                payload_dict["nav_target"] = ""
        # is_definition flag + definition_term : "is the student asking
        # for a definition of a standard technical term, and what is the
        # term ?" When is_definition=true, the responder will FIRST try to
        # answer from the course material via RAG, and ONLY fall back to
        # general LLM knowledge if the term isn't found in the retrieved
        # chunks. This preserves course-specific definitions when they
        # exist (e.g. the lecturer's own notation) while still answering
        # for standard terms the course doesn't cover.
        # We do NOT force needs_retrieval=False — retrieval ALWAYS runs,
        # and the responder decides afterwards whether to use chunks or
        # the LLM-direct definition fallback.
        if intent_type == "question":
            is_definition = bool(
                data.get("is_definition", data.get("isdefinition", data.get("isDefinition", False)))
            )
            payload_dict["is_definition"] = is_definition
            if is_definition:
                term = str(
                    data.get("definition_term") or data.get("definitionterm") or data.get("definitionTerm") or ""
                ).strip()
                payload_dict["definition_term"] = term
        intent = VoiceIntent(
            type=intent_type,
            confidence=confidence,
            payload=payload_dict,
            needs_retrieval=needs_retrieval,
        )

        # ── Merged-rewriter output ──
        # When intent_type == "question", the LLM also produced
        # ``anchored_concept``, ``needed_rewrite``, ``rewritten``. We pre-
        # populate the rewriter's output keys so the rewriter node can
        # short-circuit (it checks state.get("rewritten_query")). For
        # non-question intents the rewriter doesn't run anyway (graph
        # routes to the responder directly).
        result: dict[str, Any] = {
            "intent": intent,
            "timings": {**state.get("timings", {}), "intent": round(time.time() - start, 3)},
        }
        if intent_type == "question":
            anchored = str(
                data.get("anchored_concept") or data.get("anchoredconcept") or data.get("anchoredConcept") or ""
            ).strip()
            rewritten = str(data.get("rewritten", "") or "").strip().strip('"').strip("«»").strip()
            needed_rewrite = bool(
                data.get("needed_rewrite", data.get("neededrewrite", data.get("neededRewrite", False)))
            )
            # Operational guard: empty/oversized rewrites fall back to raw.
            if not rewritten or len(rewritten) > _REWRITTEN_CAP:
                rewritten = text
            result["rewritten_query"] = rewritten
            result["anchored_concept"] = anchored
            log.info(
                "intent: %s (conf=%.2f, retrieve=%s) | rewriter merged: %s (%d→%d, '%s')",
                intent.type, intent.confidence, intent.needs_retrieval,
                "rewritten" if needed_rewrite else "no-op",
                len(text), len(rewritten), anchored[:60],
            )
        else:
            # Surface the navigation sub-action when present, so a quick
            # eyeball of the logs tells the operator whether the LLM
            # heard "next" or "explain more" or "slow down".
            extras = ""
            if intent_type == "navigation":
                extras = (f" | nav_action={payload_dict.get('nav_action')!r}"
                          + (f" target={payload_dict.get('nav_target')!r}"
                             if payload_dict.get("nav_target") else ""))
            elif intent_type == "feedback":
                extras = f" | polarity={payload_dict.get('feedback_polarity')!r}"
            log.info(
                "intent: LLM-classified as %s (conf=%.2f, retrieve=%s)%s",
                intent.type, intent.confidence, intent.needs_retrieval, extras,
            )
        return result


def intent_router(state: TutorState) -> str:
    """Conditional edge: route to 'question' (full pipeline) or 'short' (responder only)."""
    intent = state.get("intent")
    if intent and getattr(intent, "type", None) == "question":
        return "question"
    return "short"
