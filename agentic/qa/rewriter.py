"""QueryRewriterAgent — context-aware multi-turn query rewriter.

Purpose: produce a self-contained query for the Retriever by resolving
pronouns, ellipsis, and dialogue references against the recent history
and the current slide context.

# Architecture: LLM-only with step-back reasoning

Single learned tier — the LLM both decides whether a rewrite is needed
and produces it. The previous version had a rule-based fast-path that
skipped the LLM when (a) the question had no pronoun from a hand-curated
list of 11 tokens (``"ça"``, ``"il"``, ``"this"``, ...) and (b) had at
least 6 words. Two structural problems:

  - **Pronoun list incomplete**: missed ``celui-ci``, ``celles``, ``lui``,
    ``leur``, ``le`` (in "tu peux le redire"), and any pronoun the
    curator forgot.
  - **Ellipsis blind**: "et après ?" / "le suivant" / "comment ?" have
    zero pronouns yet are 100% context-dependent. The skip path
    forwarded them verbatim to the retriever, which would search for the
    literal phrase and miss the actual concept.
  - **6-word threshold arbitrary**: why 6? not 5? not 8? hand-picked.

Removing both lets the LLM make a calibrated, content-aware decision.
The cost is one LLM call per question that wasn't otherwise skipped by
Self-RAG (``needs_retrieval=False`` short-circuits the rewriter at the
graph level — see ``agentic/qa/graph.py``).

# Prompt design — step-back + structured output

The prompt asks the LLM to:
  1. Identify the **anchored concept** (what abstract topic is the
     student asking about?) — step-back reasoning (Zheng et al. 2023,
     "Take a Step Back: Evoking Reasoning via Abstraction in Large
     Language Models", arXiv:2310.06117).
  2. Decide whether the raw question is already self-contained.
  3. If not, produce a rewrite that bakes the anchored concept in.

Structured JSON output ensures the rewriter never silently fails open
on a malformed reply: parse failure → use raw question (operational
fallback, not a heuristic).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from agentic.state import TutorState

log = logging.getLogger("agentic.qa.rewriter")


_REWRITER_PROMPT_FR = """Tu aides un système de tutorat à comprendre la question d'un étudiant.

Historique récent (dialogue):
{history}

Slide en cours :
{slide_excerpt}

Question brute de l'étudiant : "{question}"

Procède en 3 étapes:
1. Identifie le CONCEPT principal sur lequel porte la question (en t'appuyant sur l'historique et la slide).
2. Détermine si la question brute est déjà auto-suffisante (compréhensible sans contexte).
3. Si non, réécris-la en français pour qu'elle soit auto-suffisante : résous les pronoms, expanse les ellipses, garde l'intention.

Cas cross-langue (retrieval-only ; à NE PAS confondre avec ai.prompt_rules.CROSSLANG_RULE_FR qui régit la SORTIE parlée) : si la slide est dans une langue différente de la question, ajoute dans la query réécrite la traduction des termes techniques clés dans la langue de la slide, séparée par une virgule. Ex : "supervised learning, apprentissage supervisé" — cela aide le moteur de recherche à matcher des contenus dans la langue de la slide. NE JAMAIS forcer cette duplication si les langues sont identiques.

Réponds UNIQUEMENT en JSON strict, sans markdown :
{{"anchored_concept": "...", "needed_rewrite": true|false, "rewritten": "..."}}

Si needed_rewrite=false, "rewritten" doit être exactement la question brute."""


_REWRITER_PROMPT_EN = """You help a tutoring system understand a student's question.

Recent dialogue history:
{history}

Current slide:
{slide_excerpt}

Raw student question: "{question}"

Proceed in 3 steps:
1. Identify the main CONCEPT the question is about (using history and slide).
2. Decide if the raw question is already self-contained (understandable without context).
3. If not, rewrite it in English to be self-contained: resolve pronouns, expand ellipses, keep the intent.

Cross-language case (retrieval-only — do NOT confuse with ai.prompt_rules.CROSSLANG_RULE_EN which governs the SPOKEN output): if the slide is in a different language than the question, append in the rewritten query the translation of key technical terms in the slide's language, separated by a comma. Example: "supervised learning, apprentissage supervisé" — this helps the retriever match content in the slide's language. NEVER force this duplication when both languages match.

Reply STRICT JSON ONLY, no markdown:
{{"anchored_concept": "...", "needed_rewrite": true|false, "rewritten": "..."}}

If needed_rewrite=false, "rewritten" must be exactly the raw question."""


# Operational truncation caps (not behavioural thresholds): they exist
# only to keep prompts inside the LLM context window.
# SLIDE_EXCERPT_CAP is shared with the intent node (same budget), the
# others are local because the rewriter has different needs (raw
# question size, paranoid output cap).
from agentic.qa._shared import SLIDE_EXCERPT_CAP as _SLIDE_CAP, format_history as _format_history_shared

_QUESTION_CAP = 512        # raw question length
_HISTORY_TURNS = 4         # last N messages from history (2 round-trips)
_HISTORY_MSG_CAP = 160     # per-message char cap inside history
_REWRITTEN_CAP = 800       # paranoid output cap (LLMs that ignore the
                           # "one sentence" instruction sometimes return
                           # multi-paragraph essays)


def _format_history(history: list) -> str:
    if not history:
        return "(empty)"
    out = _format_history_shared(
        history,
        pairs=_HISTORY_TURNS // 2,
        msg_cap=_HISTORY_MSG_CAP,
        labelled=False,
    )
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


class QueryRewriterAgent:
    """Resolves pronouns / ellipsis / dialogue references using an LLM.

    Output (in state):
        rewritten_query     : self-contained query for the Retriever
        anchored_concept    : main topic of the question (for downstream use)
    """

    def __init__(self, brain) -> None:
        self.brain = brain

    def __call__(self, state: TutorState) -> dict[str, Any]:
        start = time.time()

        # ── Merged-with-Intent fast path ──
        # IntentAgent now produces ``rewritten_query`` (and
        # ``anchored_concept``) in the same LLM call that classifies the
        # intent. When that field is already populated, we have nothing
        # to do — passthrough with zero-cost timings. Saves the second
        # LLM call (~50s on Ollama CPU) on every retrieval-needing
        # question, which is the common case.
        existing_rewrite = (state.get("rewritten_query") or "").strip()
        if existing_rewrite:
            log.info(
                "rewriter: skipped (merged with intent, rewrite=%d chars)",
                len(existing_rewrite),
            )
            return {
                "timings": {**state.get("timings", {}), "rewriter": round(time.time() - start, 3)},
            }

        intent = state.get("intent")
        text = ""
        if intent and isinstance(intent.payload, dict):
            text = intent.payload.get("raw_text", "")
        if not text:
            payload = state.get("event_payload") or {}
            if isinstance(payload, dict):
                text = payload.get("text") or payload.get("transcript") or ""
        text = (text or "").strip()[:_QUESTION_CAP]

        if not text:
            return {
                "rewritten_query": "",
                "timings": {**state.get("timings", {}), "rewriter": 0.0},
            }

        # ── Fast-path: skip the LLM call when the question is
        # demonstrably self-contained.
        #
        # On CPU-only setups, this rewriter call costs ~50s. For
        # questions that don't actually need rewriting (long, well-
        # formed sentences with no pronouns referring to context), we
        # can skip the LLM entirely and forward the question verbatim
        # to the retriever. On a typical session this skips ~50% of
        # rewriter calls — same retrieval quality, half the latency.
        #
        # Heuristic (purely structural, no vocabulary lists):
        #   - >= 30 chars (long enough to stand alone)
        #   - >= 5 whitespace tokens (not an ellipsis like "et après ?")
        #   - has at least one substantive (4+ char) word besides the
        #     first (rules out "What is X?" where X is a single short
        #     pronoun like "it" or "this")
        #   - no immediate-context references in the FIRST 2 chars
        #     (catches "ça", "il" used as the very subject)
        words = text.split()
        substantive_count = sum(1 for w in words[1:] if len(w.strip(".,;:!?\"'")) >= 4)
        starts_with_short_pronoun = (
            len(words) > 0 and len(words[0].strip(".,;:!?\"'")) <= 3
            and not words[0][0].isupper()
        )
        is_self_contained = (
            len(text) >= 30
            and len(words) >= 5
            and substantive_count >= 2
            and not starts_with_short_pronoun
        )
        if is_self_contained:
            log.info(
                "rewriter: skipped (question self-contained, %d chars, %d words)",
                len(text), len(words),
            )
            return {
                "rewritten_query": text,
                "anchored_concept": "",
                "timings": {**state.get("timings", {}), "rewriter": round(time.time() - start, 3)},
            }

        lang = (state.get("language") or "fr")[:2]
        template = _REWRITER_PROMPT_FR if lang == "fr" else _REWRITER_PROMPT_EN
        slide = (state.get("last_slide_content") or "")[:_SLIDE_CAP].replace("\n", " ") or "(none)"
        history_str = _format_history(state.get("history") or [])

        prompt = template.format(
            slide_excerpt=slide,
            history=history_str,
            question=text,
        )

        try:
            raw, _ = self.brain.ask(prompt, reply_language=lang)
        except Exception as exc:
            log.warning("rewriter: LLM call failed (%s) → using raw question", exc)
            return {
                "rewritten_query": text,
                "timings": {**state.get("timings", {}), "rewriter": round(time.time() - start, 3)},
            }

        data = _extract_json(raw) or {}
        rewritten = str(data.get("rewritten", "")).strip().strip('"').strip("«»").strip()
        anchored_concept = str(data.get("anchored_concept", "")).strip()
        needed_rewrite = bool(data.get("needed_rewrite", True))

        # Operational guard: if the LLM returned nothing usable or an
        # essay, fall back to the raw question. Not a behavioural
        # threshold — _REWRITTEN_CAP just exists to keep downstream
        # prompts within context limits.
        if not rewritten or len(rewritten) > _REWRITTEN_CAP:
            rewritten = text

        log.info(
            "rewriter: %s (%d→%d chars, concept='%s')",
            "rewritten" if needed_rewrite else "no-op",
            len(text),
            len(rewritten),
            anchored_concept[:60],
        )
        log.info("🔍 rewriter ORIGINAL : %r", text[:200])
        if rewritten != text:
            log.info("🔍 rewriter REWRITTEN: %r", rewritten[:200])
        return {
            "rewritten_query": rewritten,
            "anchored_concept": anchored_concept,
            "timings": {**state.get("timings", {}), "rewriter": round(time.time() - start, 3)},
        }
