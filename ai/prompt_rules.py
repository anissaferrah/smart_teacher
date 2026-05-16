"""Shared prompt rules used in multiple LLM call sites.

Three places used to carry their own copy of the cross-language rule
(``ai/llm.py:get_system_prompt``, ``ai/llm.py:get_presentation_prompt``,
``agentic/qa/rewriter.py``). Pulling the canonical phrasing here means a
single edit propagates everywhere — and the rule's intent doesn't
silently drift across surfaces.

The rules are returned as plain strings; callers wrap them in their own
section headers / line breaks. They are deliberately concise (one block
of guidance, no examples) — the goal is to be a contract the LLM
respects, not a tutorial.
"""
from __future__ import annotations


# ── Cross-language behaviour ────────────────────────────────────────────
# When the source material (slide / chunk / quoted snippet) and the
# response language differ, the LLM must:
#   1. parse the source language naturally — technical content is
#      universal across languages,
#   2. answer in the requested language,
#   3. preserve the original wording of key technical terms in
#      parentheses on first mention so the student can map what they
#      hear to what they see on the slide.
# Proper nouns, acronyms, and formula symbols are NEVER translated.

CROSSLANG_RULE_EN = (
    "CROSS-LANGUAGE RULE: If the source material is in a different "
    "language than the language you are answering in, parse the source "
    "naturally (technical content is universal) and respond ONLY in the "
    "requested language. Do NOT insert source-language equivalents in "
    "parentheses by default — translate terms inline and move on. "
    "Only provide the source-language wording in parentheses if the "
    "student explicitly asks for it (e.g. 'what's the French term?', "
    "'how is it written on the slide?'). Proper nouns, acronyms, and "
    "formula symbols are never translated."
)

CROSSLANG_RULE_FR = (
    "RÈGLE CROSS-LANGUE : Si le matériel source est dans une langue "
    "différente de celle dans laquelle tu réponds, comprends la source "
    "naturellement (le contenu technique est universel) et réponds "
    "UNIQUEMENT dans la langue demandée. N'insère PAS l'équivalent en "
    "langue source entre parenthèses par défaut — traduis les termes "
    "directement et continue. Ne fournis le terme en langue source "
    "entre parenthèses que si l'étudiant le demande explicitement "
    "(ex : « quel est le terme anglais ? », « comment c'est écrit sur "
    "la slide ? »). Les noms propres, acronymes et symboles de formule "
    "ne se traduisent jamais."
)


def crosslang_rule(language: str = "en") -> str:
    """Return the cross-language rule for the requested response language."""
    return CROSSLANG_RULE_FR if (language or "")[:2].lower() == "fr" else CROSSLANG_RULE_EN
