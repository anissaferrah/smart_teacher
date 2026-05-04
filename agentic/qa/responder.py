"""ResponderAgent — produces the final answer for the student with grounded citations.

Branches on intent:
  - question         : answer using retrieved chunks + Brain.ask, with grounded
                       citations to the chunks the answer is built on
  - navigation       : short acknowledgement
  - feedback         : short acknowledgement (continue vs replay)
  - confusion_signal : empathic clarification using slide context

Sets state.answer (narration text), state.citations (list of grounded chunk
references), and state.actions.

# Citations design

The responder asks the LLM to output JSON ``{"answer": str,
"supporting_chunks": list[chunk_id]}``. The chunk IDs come from a stable
tag injected next to each chunk in the prompt (``[id:concept_knn]``).
After parsing:

  - ``state.answer`` carries the natural language reply (TTS-friendly,
    no inline markers leaking).
  - ``state.citations`` carries the structured grounding evidence: each
    entry is the chunk metadata the LLM marked as supporting the answer.

Empty ``supporting_chunks`` means the LLM didn't ground its answer in any
retrieved chunk — useful signal for downstream consumers (UI badge
"unsupported answer", reviewer hallucination check, telemetry).

The previous version told the LLM "NEVER mention 'source', 'reference',
'[1]', '[2]'" because it was designed for spoken output where such
artifacts would leak into TTS. The new design separates the two concerns:
the JSON ``answer`` field stays clean for TTS, the ``supporting_chunks``
field carries the grounding evidence as structured data.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from agentic.schemas import Action
from agentic.state import TutorState

log = logging.getLogger("agentic.qa.responder")


# ── Static templates (no LLM) for navigation / feedback ───────────────────

# Per-nav_action acknowledgements. Each one is what the tutor SAYS to
# the student WHILE the actual dispatch (next/prev/repeat/...) happens
# in handlers/ws.py. They're short by design — the spoken
# acknowledgement should be < 1 second of TTS so the navigation feels
# instant.
#
# Why per-action and not a single "OK je navigue" :
# the legacy code spoke the same line ("D'accord, je reviens sur ce
# point") for every nav request — confusing when the student said
# "ralentis" and got back "I'll go back to that". Each action now has
# a phrase that matches what's about to happen.
_NAV_REPLIES_BY_ACTION = {
    "fr": {
        "next":           "D'accord, je passe à la suivante.",
        "previous":       "D'accord, on revient en arrière.",
        "repeat":         "OK, je réexplique cette slide.",
        "skip":           "OK, on saute cette partie.",
        "go_to_concept":  "D'accord, je vais directement à ce concept.",
        "explain_more":   "Je vais détailler davantage.",
        "slow_down":      "OK, je ralentis le débit.",
        # Fallback when nav_action is missing or unknown — collapses to
        # the legacy line so existing behaviour is preserved verbatim.
        "_default":       "D'accord, je reviens sur ce point.",
    },
    "en": {
        "next":           "Sure, moving to the next one.",
        "previous":       "OK, going back.",
        "repeat":         "OK, I'll re-explain this slide.",
        "skip":           "OK, skipping this part.",
        "go_to_concept":  "Sure, jumping to that concept.",
        "explain_more":   "I'll go into more detail.",
        "slow_down":      "OK, slowing down.",
        "_default":       "Sure, let me go back to that.",
    },
}

# Backwards-compat : old call sites that imported _NAV_REPLIES read
# the default line. Keeps the module API stable for any consumer that
# may still depend on the simple shape.
_NAV_REPLIES = {
    "fr": _NAV_REPLIES_BY_ACTION["fr"]["_default"],
    "en": _NAV_REPLIES_BY_ACTION["en"]["_default"],
}

_FEEDBACK_CONTINUE = {
    "fr": "Parfait, je continue.",
    "en": "Great, let's continue.",
}

_FEEDBACK_REPEAT = {
    "fr": "D'accord, je vais réexpliquer.",
    "en": "OK, I'll re-explain.",
}


# Operational caps come from agentic.qa._shared so the prompt-budget
# story lives in one place (rewriter and intent share these too).
from agentic.qa._shared import (
    CHUNK_TEXT_CAP as _CHUNK_TEXT_CAP,
    MAX_CITED_CHUNKS as _MAX_CITED_CHUNKS,
    SLIDE_CONTENT_CAP as _SLIDE_CONTENT_CAP,
    HISTORY_PAIRS as _HISTORY_PAIRS,
    HISTORY_MSG_MAX as _HISTORY_MSG_MAX,
    format_history as _format_history,
)


def _stable_chunk_id(chunk: dict, index: int) -> str:
    """Return a short, prompt-friendly identifier for a chunk.

    Prefers the chunk's ``idea_id`` (semantic, stable across retrievals).
    Falls back to a positional id ``c{index}`` when ``idea_id`` is missing
    so the LLM always has something to cite.
    """
    raw = chunk.get("idea_id") or ""
    raw = str(raw).strip()
    if raw:
        # Trim to 40 chars to keep prompt compact; the full id stays in
        # state.citations for downstream consumers.
        return raw[:40]
    return f"c{index}"


# Labels affiches au LLM pour les chunks ajoutes via KG augmentation.
# Permet au modele d'adapter sa formulation : "voici un exemple : ..." vs
# "avant tout, il faut comprendre que ..." pour un prerequis.
# Sans ce label, l'augmentation `_kg_relation` etait calculee mais inutile.
_KG_RELATION_LABELS = {
    "fr": {
        "prereq":      "prerequis",
        "example":     "exemple concret",
        "illustrated": "concept illustre",
    },
    "en": {
        "prereq":      "prerequisite",
        "example":     "concrete example",
        "illustrated": "illustrated concept",
    },
}


def _format_chunks_with_ids(
    chunks: list,
    max_chunks: int = _MAX_CITED_CHUNKS,
    lang: str = "fr",
) -> tuple[str, dict[str, dict]]:
    """Format chunks for the prompt with explicit citation tags.

    KG-augmented chunks (``_via_kg=True``) get an extra label in their header
    based on ``_kg_relation`` (prereq / example / illustrated) so the LLM
    can adapt its tone — e.g., "voici un exemple :" vs "avant cela, il faut
    comprendre que :".

    Returns:
        block       : multi-line text "[id:tag] content\\n---\\n[id:tag2 | label] content"
        id_to_chunk : mapping from citation tag back to the original chunk dict,
                      used when post-processing the LLM output to populate
                      state.citations.
    """
    if not chunks:
        return "", {}
    labels = _KG_RELATION_LABELS.get(lang, _KG_RELATION_LABELS["fr"])
    parts: list[str] = []
    id_to_chunk: dict[str, dict] = {}
    for i, ch in enumerate(chunks[:max_chunks]):
        if not isinstance(ch, dict):
            continue
        cid = _stable_chunk_id(ch, i)
        # Disambiguate if two chunks share the same idea_id (rare but possible)
        suffix_n = 1
        unique_cid = cid
        while unique_cid in id_to_chunk:
            suffix_n += 1
            unique_cid = f"{cid}#{suffix_n}"
        content = (ch.get("content") or "")[:_CHUNK_TEXT_CAP]
        if not content:
            continue
        # KG label si ce chunk vient de l'augmentation graphe
        kg_relation = ch.get("_kg_relation") if ch.get("_via_kg") else None
        if kg_relation and kg_relation in labels:
            header = f"[id:{unique_cid} | {labels[kg_relation]}]"
        else:
            header = f"[id:{unique_cid}]"
        parts.append(f"{header} {content}")
        id_to_chunk[unique_cid] = ch
    return "\n---\n".join(parts), id_to_chunk


# ── Conversational memory ────────────────────────────────────────────
# `_format_history`, `_HISTORY_PAIRS`, `_HISTORY_MSG_MAX` are imported
# from agentic.qa._shared (single source of truth shared with rewriter
# and intent nodes).


def _classify_memory_mode(chunks: list[dict]) -> str:
    """Determine memory mode based on chunk seen state.

    Returns:
        'first_contact' : aucun chunk deja vu  → explication complete
        'partial'       : melange seen + new   → focus sur new
        'revision'      : tous deja vus        → revision rapide
    """
    if not chunks:
        return "first_contact"
    seen_count = sum(1 for c in chunks if c.get("seen"))
    if seen_count == 0:
        return "first_contact"
    if seen_count == len(chunks):
        return "revision"
    return "partial"


def _raw_question_from_intent(intent: Any) -> str:
    """Extract the original question text from the intent payload."""
    if intent is None:
        return ""
    payload = getattr(intent, "payload", None)
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("raw_text", "") or "")


def _term_in_chunks(term: str, chunks: list[dict]) -> bool:
    """True if the technical term appears in at least one retrieved chunk.

    Substring match, case-insensitive, accent-insensitive on the chunk
    side (the lecturer might write "régression" while the LLM extracted
    "regression"). Bounds the term to a sensible length to avoid pathological
    short matches.
    """
    if not term or not chunks:
        return False
    needle = term.strip().lower()
    if len(needle) < 2:
        # Single-character "terms" would match too liberally.
        return False
    # Cheap accent fold : NFD decomposition strips diacritics. Standard
    # technique, see Unicode TR15.
    import unicodedata
    def _fold(s: str) -> str:
        nfd = unicodedata.normalize("NFD", s)
        return "".join(c for c in nfd if not unicodedata.combining(c)).lower()
    needle_folded = _fold(needle)
    for ch in chunks:
        content = ch.get("content") if isinstance(ch, dict) else ""
        if not isinstance(content, str) or not content:
            continue
        if needle in content.lower() or needle_folded in _fold(content):
            return True
    return False


# ── Off-topic / leak guardrail ────────────────────────────────────────
#
# Why we need this : even with the strict course_bound_rule + the
# <<INSTRUCTION_INTERNE>> markers, weak LLMs (Ollama 7-8B) sometimes :
#   (a) leak the meta-instruction into the answer ("Voici comment
#       expliquer la stratégie pédagogique que tu as demandée...")
#   (b) silently pull from general knowledge when retrieval came back
#       empty ("ce qoui ri?" → answer about supervised learning + house
#       prices, totally unrelated to the IR course slide).
#
# The guardrail is a cheap post-check on the LLM output. It rejects an
# answer ONLY when BOTH conditions hold :
#   1. The answer is not grounded — no citations AND low lexical overlap
#      with the slide content.
#   2. Either it contains a known leak phrase, OR it shares < threshold
#      content-word overlap with slide+chunks.
# When rejected, we substitute the honest fallback ("Ce point n'est pas
# abordé...") rather than retrying the LLM — retries are expensive and
# would just produce the same answer with the same broken context.
#
# Importantly, a slide-grounded answer (no citations but recovers slide
# vocabulary) is ACCEPTED : the user asked for "il faut répondre surtout
# dans le cas il y a une relation avec le sujet" — we don't want this
# guardrail to suppress legitimate on-topic answers.

# Phrases that almost always indicate a meta-instruction leak. These are
# specific enough that legitimate answers in pedagogical texts won't hit
# them — they only fire when the LLM repeated the prompt-internal label.
_LEAK_PHRASES_FR = (
    "voici comment expliquer la stratégie",
    "voici comment expliquer la strategie",
    "stratégie pédagogique que tu",
    "strategie pedagogique que tu",
    "personnalisation cognitive",
    "instruction interne",
    "<<instruction_interne",
    "fin_instruction_interne",
)
_LEAK_PHRASES_EN = (
    "here's how to explain the strategy",
    "here is how to explain the strategy",
    "pedagogical strategy you asked",
    "pedagogical strategy that you",
    "cognitive personalization",
    "internal instruction",
    "<<internal_instruction",
    "end_internal_instruction",
)

# Phrases that signal the model is ALREADY using the honest fallback.
# When detected, the guardrail leaves the answer alone — no point
# rejecting "this point is not covered" by replacing it with another
# "this point is not covered".
_HONEST_FALLBACK_MARKERS_FR = (
    "n'est pas abordé dans ce cours",
    "n'est pas abordée dans ce cours",
    "pas couvert par ce cours",
    "m'écarter du programme",
)
_HONEST_FALLBACK_MARKERS_EN = (
    "not covered in this course",
    "not covered by this course",
    "outside the syllabus",
)

# Sourced from Config (env var ``GROUNDING_OVERLAP_THRESHOLD``).
# Default 0.08 = 8 %. The threshold catches egregious mismatches
# (IR course answering with "supervised learning + house prices",
# overlap ≈ 0) without rejecting legitimate cross-slide paraphrases
# (a chunk-grounded answer typically retains 15-30% slide overlap).
# Tune up for stricter grounding ; tune down if false-positives in
# domains with high vocabulary diversity (literature, philosophy).
from core.config import Config as _RespConfig
_GROUNDING_OVERLAP_THRESHOLD = _RespConfig.GROUNDING_OVERLAP_THRESHOLD

# Content words (length >= 4) — short words are almost all stop-words
# in FR / EN and would dilute the overlap signal.
_MIN_CONTENT_WORD_LEN = 4

# The honest fallback substituted in when the guardrail rejects.
_HONEST_FALLBACK = {
    "fr": ("Ce point n'est pas abordé dans cette partie du cours — "
           "je préfère ne pas m'écarter du programme."),
    "en": ("This point is not covered in this part of the course — "
           "I'd rather not go outside the syllabus."),
}


def _content_words(text: str) -> set[str]:
    """Tokenise + lowercase + accent-fold + drop short tokens.

    Returns a set of "content words" (>= 4 chars) suitable for cheap
    Jaccard-style overlap. Implementation is intentionally simple : no
    NLTK, no language model, no domain dictionary — this guardrail must
    stay a few-microseconds cost on the answer path.
    """
    if not text:
        return set()
    import re
    import unicodedata
    nfd = unicodedata.normalize("NFD", text)
    folded = "".join(c for c in nfd if not unicodedata.combining(c)).lower()
    # Word characters incl. underscore. \W splits on punctuation/space.
    tokens = re.split(r"\W+", folded)
    return {t for t in tokens if len(t) >= _MIN_CONTENT_WORD_LEN}


def _detect_leak_or_offtopic(
    answer: str,
    slide: str,
    chunks: list[dict],
    lang: str,
    has_citations: bool,
) -> tuple[bool, str]:
    """Return (should_reject, reason).

    Args :
        answer         : the LLM's parsed answer text
        slide          : the current slide content (always available)
        chunks         : retrieved chunks (may be empty)
        lang           : "fr" or "en"
        has_citations  : True if the LLM cited at least one chunk

    Decision tree :
      1. Empty answer → False (handled elsewhere as a hard error)
      2. Answer already starts with the honest fallback → False
      3. Answer contains a known leak phrase → True (always rejects)
      4. Has citations → False (trust grounding)
      5. Compute overlap with slide ∪ chunks. If below threshold → True.
      6. Otherwise → False (slide-grounded paraphrase, on-topic).
    """
    if not answer:
        return False, ""

    answer_lower = answer.lower()

    # Already an honest fallback → nothing to reject
    fallback_markers = (
        _HONEST_FALLBACK_MARKERS_FR if lang == "fr" else _HONEST_FALLBACK_MARKERS_EN
    )
    for marker in fallback_markers:
        if marker in answer_lower:
            return False, ""

    # Hard reject : known leak phrases
    leak_phrases = _LEAK_PHRASES_FR if lang == "fr" else _LEAK_PHRASES_EN
    for phrase in leak_phrases:
        if phrase in answer_lower:
            return True, f"leak phrase {phrase!r}"

    # Trust LLM citations
    if has_citations:
        return False, ""

    # No citations : measure lexical overlap with the only context the
    # LLM saw (slide + chunks). Low overlap = the answer was generated
    # from outside the course material.
    answer_words = _content_words(answer)
    if not answer_words:
        # Pathological tiny answer ; don't reject on overlap, let it go.
        return False, ""

    source_text_parts = [slide or ""]
    for ch in chunks or []:
        if isinstance(ch, dict):
            source_text_parts.append(str(ch.get("content", "") or ""))
    source_words = _content_words(" ".join(source_text_parts))
    if not source_words:
        # No source at all → can't measure ; don't reject (the LLM was
        # asked a question with zero context, the prompt-level rule must
        # do the work). Avoids over-eager rejection on cold-start cases.
        return False, ""

    overlap = len(answer_words & source_words) / max(1, len(answer_words))
    # Detail log — even when accepted, show the overlap so the
    # operator can tune the threshold and spot near-misses.
    log.info(
        "🔍 guardrail | overlap=%.2f (threshold=%.2f) | answer_words=%d source_words=%d | "
        "common=%d | has_citations=%s",
        overlap, _GROUNDING_OVERLAP_THRESHOLD,
        len(answer_words), len(source_words),
        len(answer_words & source_words),
        has_citations,
    )
    if overlap < _GROUNDING_OVERLAP_THRESHOLD:
        return True, (
            f"low grounding overlap {overlap:.2f} < {_GROUNDING_OVERLAP_THRESHOLD:.2f} "
            f"(no citations, answer not anchored on slide/chunks)"
        )

    return False, ""


def _make_definition_prompt(
    question: str,
    lang: str,
    history_text: str = "",
) -> str:
    """Definition-style answer — STRICTLY grounded in the course material.

    Smart Teacher is a course-bound tutor, not a generic chatbot. Even
    on definition-style questions, the answer must come from what was
    actually taught in the course. If the term is not covered, the
    teacher honestly says so instead of paraphrasing Wikipedia.

    The previous version of this prompt allowed "answer from your general
    academic knowledge (textbook/Wikipedia style)" which is exactly the
    behaviour we now forbid: it makes the system feel like a chatbot,
    not a teacher who sticks to the syllabus.
    """
    if lang == "fr":
        return (
            "Tu es Smart Teacher, un tuteur IA limité au contenu du cours.\n"
            "RÈGLE STRICTE : tu ne réponds QUE à partir du matériel de cours fourni "
            "(slides, contexte interne). Si la question porte sur un terme NON couvert "
            "par le cours, tu DOIS le dire honnêtement et ne PAS paraphraser une "
            "définition générale comme un chatbot. Tu n'es pas Wikipédia.\n\n"
            "Quand le terme EST couvert par le cours, formule la réponse en 3 éléments :\n"
            "  1. La DÉFINITION telle que présentée dans le cours (1-2 phrases).\n"
            "  2. L'INTUITION ou le cas d'usage en 1 phrase.\n"
            "  3. Un EXEMPLE COURT du cours si pertinent.\n\n"
            "Quand le terme N'EST PAS couvert, réponds UNIQUEMENT par une phrase "
            "honnête, ex : \"Ce point n'est pas abordé dans ce cours — je ne peux "
            "pas l'expliquer ici sans m'écarter du programme.\"\n\n"
            "⚠️ INSTRUCTIONS INTERNES — Les blocs encadrés par "
            "<<INSTRUCTION_INTERNE_...>> ... <<FIN_INSTRUCTION_INTERNE>> sont "
            "des directives de STYLE pour TOI — ne les mentionne JAMAIS dans "
            "ta réponse, ne les traite PAS comme la question. La question "
            "réelle est uniquement celle qui suit \"TERME À DÉFINIR :\".\n\n"
            "Ton oral, pas de markdown, pas de LaTeX. Formules en clair "
            "(\"x au carré\", pas $x^2$).\n\n"
            f"═══ ÉCHANGES RÉCENTS ═══\n{history_text}\n\n"
            f"═══ TERME À DÉFINIR ═══\n{question}\n\n"
            "Réponds UNIQUEMENT en JSON strict, sans markdown :\n"
            "{\"answer\": \"...\", \"supporting_chunks\": []}"
        )
    # English
    return (
        "You are Smart Teacher, an AI tutor strictly bound to the course material.\n"
        "STRICT RULE: you only answer from the provided course material (slides, "
        "internal context). If the question concerns a term NOT covered by the "
        "course, you MUST say so honestly and NOT paraphrase a generic definition "
        "like a chatbot. You are not Wikipedia.\n\n"
        "When the term IS covered by the course, formulate the answer in 3 parts:\n"
        "  1. The DEFINITION as presented in the course (1-2 sentences).\n"
        "  2. The INTUITION or use case in 1 sentence.\n"
        "  3. A SHORT EXAMPLE from the course if relevant.\n\n"
        "When the term is NOT covered, reply ONLY with a single honest sentence, "
        "e.g.: \"This point is not covered in this course — I cannot explain it "
        "here without going outside the syllabus.\"\n\n"
        "⚠️ INTERNAL INSTRUCTIONS — Blocks wrapped in "
        "<<INTERNAL_INSTRUCTION_...>> ... <<END_INTERNAL_INSTRUCTION>> are "
        "STYLE directives FOR YOU — NEVER mention them in your answer, do "
        "NOT treat them as the question. The real question is only what "
        "follows \"TERM TO DEFINE:\".\n\n"
        "Spoken tone, no markdown, no LaTeX. Formulas in plain words "
        "(\"x squared\", not $x^2$).\n\n"
        f"═══ RECENT EXCHANGE ═══\n{history_text}\n\n"
        f"═══ TERM TO DEFINE ═══\n{question}\n\n"
        "Reply STRICT JSON ONLY, no markdown:\n"
        "{\"answer\": \"...\", \"supporting_chunks\": []}"
    )


def _make_qa_prompt(
    question: str,
    chunks_block: str,
    slide: str,
    lang: str,
    memory_mode: str = "first_contact",
    history_text: str = "",
) -> str:
    """Build the QA prompt asking for JSON {answer, supporting_chunks}."""
    if lang == "fr":
        if memory_mode == "revision":
            tone = (
                "L'étudiant a déjà vu ces concepts. Fais une RÉVISION RAPIDE en 3-4 phrases :\n"
                "  1. Rappelle la DÉFINITION clé en une phrase.\n"
                "  2. Reformule l'IDÉE PRINCIPALE avec d'autres mots.\n"
                "  3. Ajoute un PETIT EXEMPLE OU APPLICATION si pertinent.\n"
                "Pas une simple liste d'exemples — il faut la définition + la reformulation."
            )
        elif memory_mode == "partial":
            tone = (
                "L'étudiant connaît déjà certains concepts. Structure ta réponse en 4-6 phrases :\n"
                "  1. RAPPELLE BRIÈVEMENT ce qui est déjà connu (1 phrase).\n"
                "  2. DÉFINIS ou clarifie le NOUVEAU élément (1-2 phrases, vocabulaire précis).\n"
                "  3. REFORMULE en mots simples pour confirmer la compréhension.\n"
                "  4. Donne 1 EXEMPLE CONCRET tiré du cours pour ancrer.\n"
                "Évite la simple énumération d'exemples — l'étudiant a besoin de la définition aussi."
            )
        else:
            tone = (
                "Réponds comme un VRAI ENSEIGNANT qui explique en 5-7 phrases naturelles, "
                "pas comme un chatbot qui balance un seul exemple. Structure obligatoire :\n"
                "  1. DÉFINITION précise du concept (1-2 phrases, vocabulaire technique correct).\n"
                "  2. REFORMULATION en mots plus simples pour confirmer la compréhension.\n"
                "  3. INTUITION : à quoi ça sert, dans quel contexte on l'utilise (1 phrase).\n"
                "  4. UN EXEMPLE CONCRET tiré du cours, pas de Wikipédia (1-2 phrases).\n"
                "  5. (Optionnel) Lien avec un concept proche ou prérequis si présent dans le cours.\n"
                "Une réponse qui donne UNIQUEMENT un exemple sans définition ni reformulation "
                "est INCOMPLÈTE — c'est une erreur. Toujours définir AVANT d'illustrer."
            )

        course_bound_rule = (
            "⚠️ RÈGLE — TUTEUR LIMITÉ AU MATÉRIEL DU COURS : tu réponds à partir de "
            "TOUT le matériel du cours fourni ci-dessous (la SLIDE EN COURS et le "
            "CONTEXTE INTERNE — qui peut contenir des extraits de N'IMPORTE QUELLE "
            "partie du cours, pas seulement de la slide en cours). L'étudiant peut "
            "poser des questions qui dépassent la slide affichée — c'est légitime. "
            "Tu n'es PAS un chatbot encyclopédique : tu n'utilises PAS ta connaissance "
            "générale, Wikipédia, ni d'exemples extérieurs au cours. Si la question "
            "porte sur un sujet qui n'est ni dans la slide ni dans le contexte interne, "
            "tu DOIS répondre honnêtement par UNE SEULE phrase, ex : \"Ce point n'est "
            "pas abordé dans ce cours — je ne peux pas l'expliquer ici sans m'écarter "
            "du programme.\" (et laisse \"supporting_chunks\" vide). N'invente RIEN, "
            "ne paraphrase RIEN depuis l'extérieur du cours.\n"
            "⚠️ INSTRUCTIONS INTERNES — Les blocs encadrés par "
            "<<INSTRUCTION_INTERNE_...>> ... <<FIN_INSTRUCTION_INTERNE>> "
            "sont des directives de STYLE pour TOI — tu ne dois JAMAIS les "
            "mentionner, les paraphraser, ni les considérer comme la question "
            "de l'étudiant. Ne commence JAMAIS ta réponse par \"Voici comment "
            "expliquer la stratégie...\" ou \"Bien sûr, je vais utiliser une "
            "analogie...\" — la question réelle est uniquement celle qui suit "
            "\"QUESTION ACTUELLE :\" en bas du prompt."
        )

        grounding_rule = (
            "ANCRAGE : la SLIDE EN COURS est ta source de vérité prioritaire. "
            "Si elle contient des exemples, des tableaux, des définitions — utilise-les. "
            "Le CONTEXTE INTERNE est un complément ; chaque morceau y est étiqueté par "
            "[id:xxx]. Certains morceaux portent un label supplémentaire issu du graphe "
            "pédagogique : [id:xxx | prerequis] (à connaître AVANT pour comprendre la "
            "réponse), [id:xxx | exemple concret] (cas concret du concept), "
            "[id:xxx | concept illustre] (le concept général dont la question est un "
            "exemple). Sers-toi en pour structurer : 'pour comprendre cela, il faut "
            "d'abord savoir que ...' (prerequis) ou 'concrètement : ...' (exemple). "
            "Si tu utilises un morceau, ajoute son id dans \"supporting_chunks\". "
            "Si la réponse vient uniquement de la slide, laisse \"supporting_chunks\" vide."
        )

        return (
            f"Tu es Smart Teacher, un tuteur IA qui répond à la question d'un étudiant en cours.\n\n"
            f"{course_bound_rule}\n\n"
            f"INSTRUCTION PÉDAGOGIQUE (uniquement si la question est couverte par le cours) : {tone}\n\n"
            f"{grounding_rule}\n\n"
            f"Le champ \"answer\" est lu à voix haute : pas de markdown, pas de [id:...] "
            f"dans le texte parlé — les ids vont uniquement dans \"supporting_chunks\".\n\n"
            f"═══ SLIDE EN COURS ═══\n{slide[:_SLIDE_CONTENT_CAP]}\n\n"
            f"═══ CONTEXTE INTERNE (chaque morceau a un [id:xxx]) ═══\n{chunks_block or '(aucun)'}\n\n"
            f"═══ ÉCHANGES RÉCENTS (continuité — ne les répète pas) ═══\n{history_text}\n\n"
            f"═══ QUESTION ACTUELLE ═══\n{question}\n\n"
            f"Réponds UNIQUEMENT en JSON strict, sans markdown :\n"
            f"{{\"answer\": \"...\", \"supporting_chunks\": [\"id1\", \"id2\"]}}"
        )

    # English
    if memory_mode == "revision":
        tone = (
            "Student already saw these concepts. Give a QUICK REFRESH in 3-4 sentences :\n"
            "  1. Recall the key DEFINITION in one sentence.\n"
            "  2. Rephrase the MAIN IDEA in different words.\n"
            "  3. Add a SHORT EXAMPLE OR APPLICATION if relevant.\n"
            "Don't reply with just a list of examples — definition + rephrasing required."
        )
    elif memory_mode == "partial":
        tone = (
            "Student already knows some of these concepts. Structure your answer in 4-6 sentences :\n"
            "  1. BRIEFLY REMIND what is already known (1 sentence).\n"
            "  2. DEFINE or clarify the NEW element (1-2 sentences, precise vocabulary).\n"
            "  3. REPHRASE in simple words to confirm understanding.\n"
            "  4. Give 1 CONCRETE EXAMPLE from the course to anchor it.\n"
            "Avoid just listing examples — the student needs the definition too."
        )
    else:
        tone = (
            "Answer like a REAL TEACHER explaining, in 5-7 natural spoken sentences, "
            "not like a chatbot tossing a single example. Required structure :\n"
            "  1. PRECISE DEFINITION of the concept (1-2 sentences, correct technical terms).\n"
            "  2. REPHRASING in simpler words to confirm understanding.\n"
            "  3. INTUITION : what it's for, when it's used (1 sentence).\n"
            "  4. ONE CONCRETE EXAMPLE from the course, not from Wikipedia (1-2 sentences).\n"
            "  5. (Optional) Link to a related concept or prerequisite if present in the course.\n"
            "An answer giving ONLY an example without definition or rephrasing is "
            "INCOMPLETE — that's an error. Always define BEFORE illustrating."
        )

    course_bound_rule = (
        "⚠️ RULE — COURSE-MATERIAL-BOUND TUTOR: you answer from ALL the course "
        "material provided below (the CURRENT SLIDE and the INTERNAL CONTEXT — "
        "which may contain excerpts from ANY part of the course, not only the "
        "current slide). The student may ask questions that go beyond the "
        "displayed slide — that's legitimate. You are NOT an encyclopedic "
        "chatbot: you do NOT use general knowledge, Wikipedia, or examples "
        "outside this course. If the question concerns a topic that is neither "
        "on the slide nor in the internal context, you MUST honestly reply with "
        "ONE sentence, e.g.: \"This point is not covered in this course — I "
        "cannot explain it here without going outside the syllabus.\" (and "
        "leave \"supporting_chunks\" empty). Do NOT invent, do NOT paraphrase "
        "anything from outside the course.\n"
        "⚠️ INTERNAL INSTRUCTIONS — Blocks wrapped in "
        "<<INTERNAL_INSTRUCTION_...>> ... <<END_INTERNAL_INSTRUCTION>> are "
        "STYLE directives FOR YOU — you must NEVER mention them, paraphrase "
        "them, nor treat them as the student's question. NEVER begin your "
        "answer with \"Here's how to explain the strategy...\" or \"Sure, "
        "I'll use an analogy...\" — the actual question is only what "
        "follows \"CURRENT QUESTION:\" at the bottom of the prompt."
    )

    grounding_rule = (
        "GROUNDING: the CURRENT SLIDE is your priority source of truth. "
        "If it contains examples, tables, or definitions — use those. "
        "The INTERNAL CONTEXT is a supplement; each piece is tagged with [id:xxx]. "
        "Some pieces carry an extra label from the pedagogical graph : "
        "[id:xxx | prerequisite] (must be understood BEFORE the answer), "
        "[id:xxx | concrete example] (concrete case of the concept), "
        "[id:xxx | illustrated concept] (the general concept the question is an "
        "example of). Use them to structure: 'to understand this, you first need "
        "to know that ...' (prerequisite) or 'concretely: ...' (example). "
        "If you use a piece, add its id to \"supporting_chunks\". "
        "If the answer comes only from the slide, leave \"supporting_chunks\" empty."
    )

    return (
        f"You are Smart Teacher, an AI tutor answering a student's question during a lecture.\n\n"
        f"{course_bound_rule}\n\n"
        f"PEDAGOGICAL INSTRUCTION (only if the question is covered by the course): {tone}\n\n"
        f"{grounding_rule}\n\n"
        f"The \"answer\" field is read out loud: no markdown, no [id:...] markers in the "
        f"spoken text — ids go only in \"supporting_chunks\".\n\n"
        f"═══ CURRENT SLIDE ═══\n{slide[:_SLIDE_CONTENT_CAP]}\n\n"
        f"═══ INTERNAL CONTEXT (each piece has an [id:xxx]) ═══\n{chunks_block or '(none)'}\n\n"
        f"═══ RECENT EXCHANGE (continuity — do not repeat it) ═══\n{history_text}\n\n"
        f"═══ CURRENT QUESTION ═══\n{question}\n\n"
        f"Reply STRICT JSON ONLY, no markdown:\n"
        f"{{\"answer\": \"...\", \"supporting_chunks\": [\"id1\", \"id2\"]}}"
    )


def _make_confusion_prompt(
    slide: str,
    lang: str,
    history_text: str = "",
    raw_question: str = "",
) -> str:
    """Reformulation prompt — delegates to ``compose_reformulation_prompt``."""
    from pedagogy.dialogue import compose_reformulation_prompt
    return compose_reformulation_prompt(
        original_question=raw_question,
        language=lang,
        last_slide_content=slide or "",
        history_text=history_text,
    )


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


def _parse_qa_response(
    raw: str,
    id_to_chunk: dict[str, dict],
) -> tuple[str, list[dict[str, Any]]]:
    """Parse the LLM JSON output into (answer_text, citations).

    Falls back to (raw_text, []) when the JSON is malformed — the
    structured citations are nice-to-have, not critical: the spoken
    answer must always be deliverable.
    """
    data = _extract_json(raw)
    if not data:
        log.debug("responder: JSON parse failed, returning raw text without citations")
        return (raw or "").strip(), []

    answer = str(data.get("answer", "") or "").strip()
    raw_ids = data.get("supporting_chunks", [])
    if not isinstance(raw_ids, list):
        raw_ids = []

    citations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for cid in raw_ids:
        cid_s = str(cid).strip()
        if not cid_s or cid_s in seen_ids:
            continue
        seen_ids.add(cid_s)
        chunk = id_to_chunk.get(cid_s)
        if chunk is None:
            # LLM hallucinated an id; skip silently (not a failure mode worth raising)
            continue
        citations.append({
            "chunk_id":   cid_s,
            "idea_label": chunk.get("idea_label", ""),
            "source":     chunk.get("source", ""),
            "score":      float(chunk.get("score", 0.0)),
        })

    # Defensive fallback: if the LLM omitted "answer" entirely, use the
    # raw response so we never return empty.
    if not answer:
        answer = (raw or "").strip()

    return answer, citations


# Static-reply confidence values are deterministic (the response doesn't
# depend on the model output), so 1.0 is the honest signal — these
# replies are never wrong in the model-output sense.
_STATIC_CONFIDENCE = 1.0


class ResponderAgent:
    """Generates the final spoken answer based on intent + retrieved chunks."""

    def __init__(self, brain) -> None:
        self.brain = brain

    async def __call__(self, state: TutorState) -> dict[str, Any]:
        start = time.time()
        intent = state.get("intent")
        intent_type = getattr(intent, "type", "question") if intent else "question"
        lang = (state.get("language") or "fr")[:2]
        slide = state.get("last_slide_content") or ""

        # ── Static fast paths ────────────────────────────────────────
        if intent_type == "navigation":
            # Read the granular sub-action set by the IntentAgent
            # (next | previous | repeat | skip | go_to_concept |
            # explain_more | slow_down). Falls back to the safe default
            # when the field is missing — same "D'accord, je reviens
            # sur ce point" behaviour as before, so legacy intent
            # classifiers without ``nav_action`` keep working.
            nav_action = ""
            nav_target = ""
            if intent and isinstance(intent.payload, dict):
                nav_action = str(intent.payload.get("nav_action", "") or "").strip().lower()
                nav_target = str(intent.payload.get("nav_target", "") or "").strip()

            replies = _NAV_REPLIES_BY_ACTION.get(lang, _NAV_REPLIES_BY_ACTION["fr"])
            answer = replies.get(nav_action) or replies["_default"]

            # Emit a structured "navigate" action carrying the sub-action
            # + target. The WS handler reads ``actions[0].payload`` after
            # the Q&A graph returns and dispatches to the right service
            # (dialogue.next_section, replay_current_slide, jump_to_concept,
            # speech_rate adjustment, etc.). Without this payload, the
            # handler can't tell "next" from "slow_down" and falls back
            # to a generic replay.
            actions = [Action(type="navigate", payload={
                "nav_action": nav_action or "next",   # safe default
                "nav_target": nav_target,
            })]
            log.info(
                "responder: navigation reply | action=%r%s",
                nav_action or "(default)",
                f" target={nav_target!r}" if nav_target else "",
            )
            return {
                "answer": answer,
                "actions": actions,
                "citations": [],
                "confidence": _STATIC_CONFIDENCE,
                "timings": {**state.get("timings", {}), "responder": round(time.time() - start, 3)},
            }

        if intent_type == "feedback":
            # Polarity (continue vs repeat) comes from the upstream LLM
            # intent classifier via intent.payload["feedback_polarity"].
            # The previous version did substring matching on the raw text
            # ("non", "no", "répète", ...) — rule-based and language-coupled.
            # When the field is missing, default to "continue" (the safer
            # interpretation: don't replay if unclear).
            polarity = ""
            if intent and isinstance(intent.payload, dict):
                polarity = str(intent.payload.get("feedback_polarity", "")).strip().lower()
            if polarity == "negative":
                answer = _FEEDBACK_REPEAT.get(lang, _FEEDBACK_REPEAT["fr"])
                actions = [Action(type="replay_concept", payload={"reason": "feedback_repeat"})]
            else:
                answer = _FEEDBACK_CONTINUE.get(lang, _FEEDBACK_CONTINUE["fr"])
                actions = [Action(type="continue", payload={})]
            log.info("responder: feedback static reply (polarity=%s)", polarity or "default")
            return {
                "answer": answer,
                "actions": actions,
                "citations": [],
                "confidence": _STATIC_CONFIDENCE,
                "timings": {**state.get("timings", {}), "responder": round(time.time() - start, 3)},
            }

        # ── LLM-backed paths (question / confusion_signal) ───────────
        chunks = state.get("retrieved_chunks") or []
        session_id = state.get("session_id") or ""
        student_id = state.get("student_id")
        course_id = state.get("course_id")
        confusion = state.get("confusion")
        is_confused_now = bool(confusion and getattr(confusion, "detected", False))

        # ── Primary concept + current mastery (for the bandit) ────────
        # The "primary concept" is the most-relevant retrieved chunk's
        # idea_id. We use the first chunk after RAG ranking — the
        # retriever already sorted by score so chunks[0] is the top hit.
        primary_concept = ""
        idea_ids_for_mastery: list[str] = []
        for ch in chunks:
            cid = ch.get("idea_id")
            if cid:
                if not primary_concept:
                    primary_concept = str(cid)
                idea_ids_for_mastery.append(str(cid))

        # Fetch current mastery scores (best-effort). Used as :
        #   - mastery_after for the previous turn's bandit reward,
        #   - mastery_score input for the new turn's context bucket.
        mastery_score_now = 0.5
        if student_id and idea_ids_for_mastery:
            try:
                from pedagogy.mastery_repo import MasteryRepo
                mastery_map = await MasteryRepo.get_scores_bulk(
                    student_id, course_id, idea_ids_for_mastery
                )
                if mastery_map:
                    mastery_score_now = sum(mastery_map.values()) / len(mastery_map)
            except Exception as exc:                                      # noqa: BLE001
                log.debug("responder: mastery fetch for bandit failed: %s", exc)

        # ── Bandit lifecycle : resolve previous + pick new ────────────
        bandit_decision = None
        try:
            from pedagogy.personalization.bandit import BanditController
            controller = BanditController()

            # Step 1 : resolve any pending decision from the previous turn.
            # The previous turn's action's reward is computed FROM THE
            # CURRENT outcomes — confusion detected on the new student
            # utterance signals that the previous answer was confusing.
            await controller.end_turn(
                session_id=session_id,
                confusion_detected=is_confused_now,
                mastery_after=mastery_score_now,
                engaged=True,  # the student came back, so engaged
            )

            # Step 2 : pick a strategy for THIS turn. Profile features
            # come from the per-(student, course) profile so the same
            # student can have a different pace / style adaptation per
            # course (fast on Python, slow on linguistics, etc.).
            learning_style = "mixed"
            avg_rt = 0.0
            try:
                from pedagogy.personalization.profile import get_or_create_profile
                profile = await get_or_create_profile(session_id, course_id=course_id)
                learning_style = str(profile.get("learning_style", "mixed"))
                # avg_response_time_s is the canonical field; legacy blobs
                # exposed `avg_response_time` so we accept both as input.
                avg_rt = float(
                    profile.get("avg_response_time_s",
                                profile.get("avg_response_time", 0.0))
                )
            except Exception as exc:                                      # noqa: BLE001
                log.debug("responder: profile fetch for bandit failed: %s", exc)

            bandit_decision = await controller.start_turn(
                session_id=session_id,
                learning_style=learning_style,
                avg_response_time_s=avg_rt,
                mastery_score=mastery_score_now,
                primary_concept=primary_concept,
                language=lang,
                student_id=student_id,
                course_id=course_id,
            )
        except Exception as exc:                                          # noqa: BLE001
            log.debug("responder: bandit lifecycle skipped: %s", exc)

        # ── Personalization prefix ────────────────────────────────────
        # Two layers : (a) generic style/pace/tone from PersonalizationEngine,
        # (b) bandit-chosen strategy fragment. The strategy fragment is
        # appended AFTER the generic personalization so it overrides the
        # default tone-shaping with the explicit pedagogical move.
        # Both layers are scoped per (student, course) so the same student
        # gets different adaptation across subjects.
        personalization_prefix = ""
        try:
            from pedagogy.personalization.engine import PersonalizationEngine
            if student_id:
                pers_ctx = await PersonalizationEngine.get_context(student_id, course_id=course_id)
                personalization_prefix = PersonalizationEngine.build_prompt_prefix(pers_ctx, lang=lang)
        except Exception as exc:
            log.debug(f"personalization fetch failed: {exc}")

        # ── Student knowledge snapshot ───────────────────────────
        # Aggregated "what does this student already know vs not"
        # built upstream by the WS handler (pedagogy.student_knowledge.
        # build_snapshot). When present, format it as a non-confusable
        # prompt block (<<INSTRUCTION_INTERNE>>) so the LLM adapts
        # tone/depth without ever speaking *about* the snapshot.
        # When absent (anon student, cold start, KG miss), we just
        # skip — the answer stays correct, just not adaptive.
        try:
            _snap = state.get("student_snapshot")
            if _snap is not None:
                from pedagogy.student_knowledge import to_prompt_context
                _snap_block = to_prompt_context(_snap, lang=lang)
                if _snap_block:
                    personalization_prefix = personalization_prefix + _snap_block
                    log.info(
                        "responder: student snapshot injected | "
                        "strong=%d weak=%d never_seen=%d confused=%d",
                        len(getattr(_snap, "strong_concepts", [])),
                        len(getattr(_snap, "weak_concepts", [])),
                        len(getattr(_snap, "never_seen_concepts", [])),
                        len(getattr(_snap, "recently_confused", [])),
                    )
        except Exception as exc:                                          # noqa: BLE001
            log.debug(f"student snapshot injection skipped: {exc}")

        # ── Engagement-aware tone modulation ─────────────────────
        # When the engagement scorer says the student is "disengaged"
        # we add a directive to RE-CAPTURE attention : shorter answer,
        # 1 concrete example, end with a question to prompt them back.
        # When "engaged" we can afford to push harder. "neutral" keeps
        # default. The engagement dict is set upstream in the WS
        # handler from ``compute_engagement(...)``.
        try:
            _eng = state.get("engagement") or {}
            _label = str(_eng.get("label", "")).lower()
            if _label == "disengaged":
                if lang == "fr":
                    eng_block = (
                        "<<INSTRUCTION_INTERNE_NE_PAS_MENTIONNER_DANS_LA_REPONSE>>\n"
                        "ALERTE : étudiant peu engagé en ce moment. Sois CONCIS "
                        "(3-4 phrases max), donne UN exemple concret, et termine "
                        "par UNE question courte qui invite à réagir (\"tu vois "
                        "l'idée ?\", \"essaie de me dire avec tes mots\").\n"
                        "<<FIN_INSTRUCTION_INTERNE>>\n\n"
                    )
                else:
                    eng_block = (
                        "<<INTERNAL_INSTRUCTION_DO_NOT_MENTION_IN_ANSWER>>\n"
                        "ALERT: student low-engagement. Keep your answer SHORT "
                        "(3-4 sentences max), give ONE concrete example, end "
                        "with a short prompt question (\"does that click?\", "
                        "\"can you put it in your own words?\").\n"
                        "<<END_INTERNAL_INSTRUCTION>>\n\n"
                    )
                personalization_prefix = personalization_prefix + eng_block
                log.info("responder: engagement=disengaged → concise+prompt mode")
            elif _label == "engaged":
                if lang == "fr":
                    eng_block = (
                        "<<INSTRUCTION_INTERNE_NE_PAS_MENTIONNER_DANS_LA_REPONSE>>\n"
                        "Étudiant très engagé : tu peux pousser un peu plus loin "
                        "(détails techniques, lien avec un autre concept du cours, "
                        "petit défi à la fin).\n"
                        "<<FIN_INSTRUCTION_INTERNE>>\n\n"
                    )
                else:
                    eng_block = (
                        "<<INTERNAL_INSTRUCTION_DO_NOT_MENTION_IN_ANSWER>>\n"
                        "Highly engaged student : you can push deeper (technical "
                        "detail, link to another course concept, mini-challenge "
                        "at the end).\n"
                        "<<END_INTERNAL_INSTRUCTION>>\n\n"
                    )
                personalization_prefix = personalization_prefix + eng_block
                log.debug("responder: engagement=engaged → push-deeper mode")
        except Exception as exc:                                          # noqa: BLE001
            log.debug(f"engagement modulation skipped: {exc}")

        # Append the bandit-chosen strategy fragment, if any.
        #
        # The fragment is wrapped in ``<<INSTRUCTION_INTERNE...>>`` markers
        # for the same reason as build_prompt_prefix above : without the
        # wrapping, weak LLMs (Ollama 7-8B) confuse the meta-directive
        # with the user's question. We saw responses like "Voici comment
        # expliquer la stratégie pédagogique que tu as demandée, avec un
        # exemple concret..." — the model literally answered the
        # personalization label instead of the student's question. The
        # course_bound_rule below tells the LLM to never mention these
        # blocks ; the markers themselves make the directive clearly
        # *not* part of the user-visible content.
        if bandit_decision is not None and bandit_decision.prompt_fragment:
            if lang == "fr":
                personalization_prefix = (
                    f"{personalization_prefix}"
                    "<<INSTRUCTION_INTERNE_NE_PAS_MENTIONNER_DANS_LA_REPONSE>>\n"
                    f"Stratégie pédagogique à appliquer : {bandit_decision.prompt_fragment}\n"
                    "<<FIN_INSTRUCTION_INTERNE>>\n\n"
                )
            else:
                personalization_prefix = (
                    f"{personalization_prefix}"
                    "<<INTERNAL_INSTRUCTION_DO_NOT_MENTION_IN_ANSWER>>\n"
                    f"Pedagogical strategy to apply: {bandit_decision.prompt_fragment}\n"
                    "<<END_INTERNAL_INSTRUCTION>>\n\n"
                )

        history_text = _format_history(state.get("history") or [], lang=lang)
        id_to_chunk: dict[str, dict] = {}

        if intent_type == "confusion_signal":
            # Confusion → reformulation. No citation extraction (the
            # reformulation is a tonal rewrite of the previous explanation,
            # not a fresh grounded answer).
            _payload = getattr(intent, "payload", None) if intent else None
            _raw_q = (_payload or {}).get("raw_text", "") if isinstance(_payload, dict) else ""
            prompt = personalization_prefix + _make_confusion_prompt(
                slide, lang,
                history_text=history_text,
                raw_question=_raw_q,
            )
            memory_mode = "first_contact"
            expects_json = False
        else:
            query = state.get("rewritten_query") or ""
            if not query and intent and isinstance(intent.payload, dict):
                query = intent.payload.get("raw_text", "") or ""

            # Definition-question routing : the IntentAgent flagged this as
            # a request for a standard technical term's definition AND
            # extracted the term itself. The course is the ONLY source ;
            # when the term doesn't appear in any retrieved chunk we route
            # to the strict definition prompt — which forces the LLM to
            # honestly say "this is not covered in the course" rather than
            # answering from general knowledge (Smart Teacher is a tutor,
            # not a chatbot).
            #
            # Decision tree :
            #   is_definition == True
            #     ├── term found in chunks → standard QA path (grounded
            #     │                          on the course's own definition)
            #     └── term NOT found       → strict definition prompt
            #                                (refuses if not in course,
            #                                citations=[])
            #   is_definition == False    → standard QA path
            is_definition = False
            definition_term = ""
            if intent is not None and isinstance(intent.payload, dict):
                is_definition = bool(intent.payload.get("is_definition", False))
                definition_term = str(intent.payload.get("definition_term", "") or "")

            term_in_course = (
                is_definition
                and definition_term
                and _term_in_chunks(definition_term, chunks)
            )
            use_definition_fallback = is_definition and not term_in_course

            if use_definition_fallback:
                prompt = personalization_prefix + _make_definition_prompt(
                    query or _raw_question_from_intent(intent),
                    lang,
                    history_text=history_text,
                )
                memory_mode = "definition_strict"
                expects_json = True
                log.info(
                    "responder: strict definition prompt (term '%s' not in %d chunks — "
                    "LLM will refuse if not in course)",
                    definition_term[:40], len(chunks),
                )
            else:
                chunks_block, id_to_chunk = _format_chunks_with_ids(chunks, lang=lang)
                memory_mode = _classify_memory_mode(chunks)
                prompt = personalization_prefix + _make_qa_prompt(
                    query, chunks_block, slide, lang,
                    memory_mode=memory_mode, history_text=history_text,
                )
                expects_json = True
                if is_definition and term_in_course:
                    log.info(
                        "responder: definition grounded on course (term '%s' found in chunks)",
                        definition_term[:40],
                    )

        # ── Retry-with-feedback path ─────────────────────────────────
        # When the QA reviewer rejected the previous attempt, it left
        # ``state["review"]`` with ``grounded=False`` and a short feedback.
        # We append that feedback to the prompt so the LLM can fix the
        # specific issue (ungrounded claim, encyclopedic leak, off-topic
        # answer) on this second pass. Same pattern as the teaching
        # narrator (`narrator.py:395` increments narrator_retries when it
        # sees retry_feedback).
        prior_review = state.get("review")
        retry_feedback = ""
        if prior_review is not None and not getattr(prior_review, "grounded", True):
            retry_feedback = (getattr(prior_review, "feedback", "") or "").strip()
            if retry_feedback:
                if lang == "fr":
                    prompt += (
                        "\n\n⚠️ TENTATIVE PRÉCÉDENTE REJETÉE par l'examinateur — "
                        f"raison : {retry_feedback}\n"
                        "Corrige ce point précis dans ta nouvelle réponse. "
                        "Si tu ne peux pas répondre depuis le matériel du cours, dis-le honnêtement."
                    )
                else:
                    prompt += (
                        "\n\n⚠️ PREVIOUS ATTEMPT REJECTED by the reviewer — "
                        f"reason: {retry_feedback}\n"
                        "Fix this specific issue in your new answer. "
                        "If you cannot answer from the course material, say so honestly."
                    )
                log.info("responder: retry with reviewer feedback (%r)", retry_feedback[:80])

        # Detailed log of the FULL prompt sent to the LLM. Truncated
        # to 4000 chars so a 50KB knowledge_graph block doesn't dump
        # to the operator console — but still long enough to see the
        # course_bound_rule, the snapshot block, the chunks, and the
        # actual question. Grep with `🔍 responder PROMPT` to pull all
        # prompt traces.
        log.info(
            "🔍 responder PROMPT (intent=%s, history=%d turns, chunks=%d) ===\n%s\n=== END PROMPT",
            intent_type,
            len(state.get("history") or []),
            len(chunks),
            prompt[:4000] + ("…[truncated]" if len(prompt) > 4000 else ""),
        )
        # Detail of the history that was actually formatted for the LLM
        if history_text:
            log.info("🔍 responder HISTORY block (%d chars):\n%s",
                     len(history_text), history_text[:1500])

        try:
            raw, _duration = self.brain.ask(
                prompt,
                reply_language=lang,
                session_id=state.get("session_id"),
            )
            raw_text = (raw or "").strip()
        except Exception as exc:
            log.error("responder LLM call failed: %s", exc)
            raw_text = (
                "Je suis désolé, je n'arrive pas à répondre tout de suite."
                if lang == "fr"
                else "Sorry, I can't answer right now."
            )
        log.info("🔍 responder LLM RAW (%d chars):\n%s",
                 len(raw_text), raw_text[:2000])

        # Parse output: JSON for QA path, raw text for confusion path.
        if expects_json:
            answer, citations = _parse_qa_response(raw_text, id_to_chunk)
        else:
            answer, citations = raw_text, []
        log.info(
            "🔍 responder PARSED answer (%d chars), citations=%d : %r",
            len(answer), len(citations), answer[:200],
        )

        # ── Off-topic / leak guardrail ───────────────────────────────
        # Only on the QA paths (skipped for confusion_signal which is a
        # tonal rewrite of an already-known explanation).
        # Permissive by design : an answer that recovers slide/chunk
        # vocabulary OR cites at least one chunk passes through. Only
        # answers that BOTH have no citations AND no lexical anchor on
        # the course material — or contain a known meta-instruction
        # leak — are replaced with the honest fallback. This way we
        # never silence a legitimate cross-slide answer (the student
        # may ask about anything in the course, not just the current
        # slide).
        if expects_json and answer:
            should_reject, reject_reason = _detect_leak_or_offtopic(
                answer=answer,
                slide=slide,
                chunks=chunks,
                lang=lang,
                has_citations=bool(citations),
            )
            if should_reject:
                log.warning(
                    "responder: GUARDRAIL rejected answer (%s) | original=%r",
                    reject_reason, answer[:120],
                )
                answer = _HONEST_FALLBACK.get(lang, _HONEST_FALLBACK["fr"])
                citations = []

        # Mark chunk ideas as seen + update mastery scores (best-effort, async).
        # session_id / student_id / course_id / is_confused_now are already
        # bound at the top of the LLM-backed branch (used by the bandit lifecycle).
        idea_ids = [c.get("idea_id") for c in chunks if c.get("idea_id")]
        is_confused = is_confused_now

        if session_id and idea_ids:
            try:
                from pedagogy.dialogue import mark_ideas_seen
                await mark_ideas_seen(session_id, idea_ids)
            except Exception as exc:
                log.debug(f"mark_ideas_seen failed: {exc}")

        # Mastery update via Bayesian Beta posterior (Laplace's rule of
        # succession in mastery_repo). The previous "+0.10 / -0.15"
        # delta-based scoring was retired; record_clean / record_confusion
        # update attempt counters instead.
        if student_id and idea_ids:
            try:
                from pedagogy.mastery_repo import MasteryRepo
                if is_confused:
                    log.info("responder: recording confusion attempt on %d ideas", len(idea_ids))
                    for iid in idea_ids:
                        await MasteryRepo.record_confusion(student_id, course_id, iid)
                else:
                    log.info("responder: recording clean attempt on %d ideas", len(idea_ids))
                    for iid in idea_ids:
                        await MasteryRepo.record_clean(student_id, course_id, iid)
            except Exception as exc:
                log.debug(f"mastery update failed: {exc}")

        # ── FSRS spaced-repetition queue update ──────────────────
        # On every Q&A turn, push the involved CONCEPTS into the
        # ReviewQueue with an FSRS rating. The mapping idea_id →
        # concept_name uses the KG (concepts_of_idea). A confusion
        # rates "again" (re-review soon) ; a clean answer rates
        # "good" (longer interval). Without this hook, the
        # ReviewQueue table only fills via the practice_engine flow
        # — and most students never trigger that, so the spaced-
        # repetition queue stays empty for them.
        if student_id and idea_ids:
            try:
                from deps import get_rag
                from pedagogy.knowledge_graph import get_or_build
                from pedagogy.review_scheduler import ReviewScheduler
                _kg = get_or_build(get_rag())
                # Aggregate idea_ids → concept_names (one update per concept)
                _concept_names: set[str] = set()
                for iid in idea_ids:
                    for ci in (_kg.concepts_of_idea(iid) or []):
                        _concept_names.add(ci.name)
                _rating = "again" if is_confused else "good"
                for cname in _concept_names:
                    await ReviewScheduler.update_after_practice(
                        student_id=student_id,
                        concept_name=cname,
                        rating_str=_rating,
                    )
                if _concept_names:
                    log.info(
                        "responder: %d concept(s) → ReviewQueue (rating=%s) [%s]",
                        len(_concept_names), _rating,
                        ", ".join(sorted(_concept_names))[:80],
                    )
            except Exception as exc:                                        # noqa: BLE001
                log.debug(f"review-queue update skipped: {exc}")

        # Detailed confusion event log — best-effort row in
        # ``confusion_events``. Lets retrospective analyses answer
        # questions like "which concept causes most confusions" or
        # "does the SIGHT detector agree with the prosody one ?". The
        # call swallows its own exceptions ; never blocks the user.
        if is_confused and student_id:
            try:
                from pedagogy.confusion.persistence import record_confusion_event
                # ``confusion`` is a ConfusionResult dataclass with
                # ``score`` and ``source`` fields ; we read defensively
                # in case it's None or shaped differently.
                score_val = float(getattr(confusion, "score", 0.0) or 0.0)
                source_val = str(getattr(confusion, "source", "unknown") or "unknown")
                question_text = (intent.payload.get("raw_text", "")
                                 if intent and isinstance(intent.payload, dict) else "")
                strategy_val = (bandit_decision.action.strategy.value
                                if bandit_decision is not None else None)
                await record_confusion_event(
                    student_id=student_id,
                    course_id=course_id,
                    session_id=session_id,
                    # primary_concept may be a concept slug rather than a
                    # UUID ; the helper coerces and stores ``None`` on
                    # bad input rather than rejecting the whole row.
                    concept_id=primary_concept,
                    slide_idx=state.get("section_idx"),
                    trigger_text=question_text,
                    source=source_val,
                    score=score_val,
                    resolved=False,
                    resolution_strategy=strategy_val,
                )
            except Exception as exc:                                        # noqa: BLE001
                log.debug(f"confusion event log failed: {exc}")

        # Confidence is derived from grounding evidence:
        #   - average citation chunk score when at least one chunk was cited
        #   - 0.0 when nothing was cited (ungrounded answer — caller can
        #     surface this as a hallucination warning)
        # Confusion-signal path skips citation parsing, so we report a
        # neutral 0.5 (we know the model produced something, we can't
        # measure its grounding from the text alone).
        if expects_json:
            confidence = (
                round(sum(c["score"] for c in citations) / len(citations), 3)
                if citations
                else 0.0
            )
        else:
            confidence = 0.5

        action_payload: dict[str, Any] = {
            "intent": intent_type,
            "memory_mode": memory_mode,
            "grounded": bool(citations),
        }
        # Surface the bandit's choice for downstream consumers (TTS layer
        # reads ``bandit_speech_rate`` to set the audio rate ; UI/logs can
        # display ``bandit_strategy`` to explain why the answer is shaped
        # the way it is).
        if bandit_decision is not None:
            action_payload["bandit_strategy"] = bandit_decision.action.strategy.value
            action_payload["bandit_speech_rate"] = bandit_decision.action.speech_rate.value
            action_payload["bandit_context"] = bandit_decision.context.bucket_key
        actions = [Action(type="answer", payload=action_payload)]
        log.info(
            "responder: %s reply (%d chars, mem=%s, chunks=%d, citations=%d, conf=%.2f, strategy=%s)",
            intent_type, len(answer), memory_mode, len(chunks), len(citations), confidence,
            bandit_decision.action.strategy.value if bandit_decision else "none",
        )
        # Bump the retry counter only when this run was a retry (reviewer
        # had previously rejected the answer). On a fresh first pass we
        # leave the counter at whatever value the state already has — the
        # teaching narrator uses the same convention.
        prior_retries = int(state.get("responder_retries", 0) or 0)
        next_retries = prior_retries + 1 if retry_feedback else prior_retries
        return {
            "answer": answer,
            "actions": actions,
            "citations": citations,
            "confidence": confidence,
            "responder_retries": next_retries,
            # Clear the prior review on the way out so the reviewer's next
            # call starts from a clean slate (otherwise a stale verdict
            # could leak into the routing decision).
            "review": None,
            "timings": {**state.get("timings", {}), "responder": round(time.time() - start, 3)},
        }
