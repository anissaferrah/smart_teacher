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
    "naturally (technical content is universal) but respond in the "
    "requested language. For each key technical term, keep the ORIGINAL "
    "spelling from the source in parentheses on first mention so the "
    "student can map your spoken word to the source. Example: "
    "'supervised learning (apprentissage supervisé)' or 'embedding "
    "(représentation vectorielle)'. Do not translate proper nouns, "
    "acronyms, or formula symbols."
)

CROSSLANG_RULE_FR = (
    "RÈGLE CROSS-LANGUE : Si le matériel source est dans une langue "
    "différente de celle dans laquelle tu réponds, comprends la source "
    "naturellement (le contenu technique est universel) puis réponds "
    "dans la langue demandée. Pour chaque terme technique clé, conserve "
    "l'ÉCRITURE ORIGINALE de la source entre parenthèses à la première "
    "mention pour que l'étudiant fasse le lien entre ce qu'il entend et "
    "ce qu'il voit. Exemple : 'apprentissage supervisé (supervised "
    "learning)' ou 'embedding (représentation vectorielle)'. Ne traduis "
    "pas les noms propres, acronymes ou symboles de formules."
)


def crosslang_rule(language: str = "en") -> str:
    """Return the cross-language rule for the requested response language."""
    return CROSSLANG_RULE_FR if (language or "")[:2].lower() == "fr" else CROSSLANG_RULE_EN
