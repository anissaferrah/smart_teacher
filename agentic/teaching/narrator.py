"""NarratorAgent — turns a PresentationPlan into spoken narration text.

Input  (from TutorState): plan, last_slide_content, language, chapter_title,
                          main_concept_hint, previous_concept,
                          previous_narration_summary
Output (into TutorState): answer (narration string ready for TTS),
                          narration_summary (short recap for next slide)
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from agentic.state import TutorState

log = logging.getLogger("agentic.teaching.narrator")


# Cliffhanger phrases that often appear when the LLM runs out of tokens
# mid-explanation ("…which we will define next") or punts on the concept.
# When a narration ENDS with one of these, we know it was truncated and
# the student gets a teaser without payoff. Surfaces a warning so the
# operator can bump max_tokens or split the slide.
_CLIFFHANGER_FR = re.compile(
    r"(?:nous (?:allons|verrons) (?:le |la |les |l['’])?(?:défin|voir|examin|étudi|découvr|abord|expliqu)|"
    r"sera (?:défini|vu|examin|étudi|abord|expliqu)|"
    r"(?:à|a) (?:venir|suivre)|"
    r"que nous (?:allons |)d[ée]finir|"
    r"dans (?:la |les )?prochain)\b[^.]*[.!?…]?\s*$",
    flags=re.IGNORECASE,
)
_CLIFFHANGER_EN = re.compile(
    r"(?:we (?:will|'ll|shall) (?:define|see|examine|study|discuss|explore|cover|address|introduce)|"
    r"to be (?:defined|seen|examined|discussed|covered)|"
    r"(?:will be|to be) (?:explained|introduced|covered|discussed) (?:next|later|soon|shortly)|"
    r"in the next)\b[^.]*[.!?…]?\s*$",
    flags=re.IGNORECASE,
)


def _is_cliffhanger(text: str, lang: str) -> bool:
    """True if the narration ends with a teaser that never pays off."""
    if not text:
        return False
    # Look only at the last ~140 chars — the cliffhanger is always at
    # the tail. Avoids false positives in mid-text discussion.
    tail = text.strip()[-140:]
    pattern = _CLIFFHANGER_FR if lang == "fr" else _CLIFFHANGER_EN
    return bool(pattern.search(tail))


# ── Greeting opener stripper ─────────────────────────────────────────
#
# Why this exists : weak LLMs (Ollama 7-8B) ignore the explicit "no
# greetings" instruction in the prompt and still open every slide with
# "Bonjour", "Bienvenue", "Bienvenue dans la partie III", "Dans le
# cadre de cette introduction", "Allez, let's go!", etc. — making the
# lecture feel like 10 separate intro sessions instead of a continuous
# flow. Prompt-only enforcement is unreliable for these models, so we
# add a deterministic post-processor that strips the offending opener
# from the *first* sentence(s) of the narration.
#
# We only strip the very beginning — once the narration is past the
# opener, "bonjour" inside a quote ("le serveur dit 'bonjour'") is
# legitimate content and must be preserved.

# Single-clause regex (no outer ``+``) — we apply it iteratively in
# _strip_greeting_opener. Greedy multi-clause matching turned out to
# consume too much : a narration like "Bienvenue dans ce cours...
# Nous allons commencer par...  Dans ce chapitre, nous verrons..." has
# THREE consecutive greeting-shaped sentences, and the greedy version
# would strip ALL of them → empty narration → guard reverts everything,
# stripping nothing. The iterative approach strips one clause at a
# time and stops when the remainder is below the safety threshold.
# Clause terminator : either standard punctuation [.!?]+\s* OR a comma
# followed by a CAPITAL LETTER (the "intro phrase, Real Content..." case
# we saw in production : "Dans le cadre de cette introduction à la RI,
# Le modèle vectoriel..."). The lookahead doesn't consume the capital
# letter so the body of the narration starts with that word intact.
#
# (?-i:...) disables IGNORECASE locally — without it, the surrounding
# re.IGNORECASE flag would make [A-Z] match BOTH cases, and the comma
# variant would fire on "Dans ce chapitre, nous..." (lowercase n),
# stripping only the "Dans ce chapitre," fragment.
_CLAUSE_END = r"(?:[.!?]+\s*|,\s+(?=(?-i:[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ])))"

_GREETING_OPENER_FR = re.compile(
    rf"""^\s*
    (?:
        (?:bonjour|bonsoir|salut)[^.!?]*?{_CLAUSE_END}                   # "Bonjour à tous."
        |
        bienvenue(?:\s+dans|\s+à|\s*[!,])[^.!?]*?{_CLAUSE_END}           # "Bienvenue dans la partie III..."
        |
        dans\s+(?:le\s+cadre\s+de\s+)?(?:cette|ce|ces|notre)\s+(?:introduction|chapitre|cours|partie|section|leçon|presentation)[^.!?]*?{_CLAUSE_END}
        |
        (?:nous\s+allons\s+commencer|commençons|on\s+commence|on\s+va\s+commencer)[^.!?]*?{_CLAUSE_END}
        |
        (?:allez|alors|bon)[\s,!]*(?:c'est\s+parti|let'?s\s+go|on\s+y\s+va|commençons)[^.!?]*?{_CLAUSE_END}
        |
        (?:dans|pour)\s+ce\s+chapitre,?\s+(?:nous|on)\s+(?:verrons|allons\s+voir|aborderons|découvrirons)[^.!?]*?{_CLAUSE_END}
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_GREETING_OPENER_EN = re.compile(
    r"""^\s*
    (?:
        (?:hello|hi|good\s+(?:morning|afternoon|evening))[^.!?]*?[.!?]+\s*
        |
        welcome(?:\s+to|\s+back|\s*[!,])[^.!?]*?[.!?]+\s*
        |
        (?:in|for)\s+this\s+(?:introduction|chapter|course|part|section|lecture|presentation),?[^.!?]*?[.!?]+\s*
        |
        (?:let'?s\s+(?:start|begin|get\s+started|dive\s+in|kick\s+off))[^.!?]*?[.!?]+\s*
        |
        (?:today|in\s+this\s+lecture),?\s+we(?:'ll|\s+will)\s+(?:see|cover|explore|look\s+at|start\s+with)[^.!?]*?[.!?]+\s*
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# Hard cap — refuse to strip more than 400 chars of "opener". If the
# narration's first 400 chars are all greeting-shaped, something else
# is wrong (likely the LLM ignored the slide entirely) ; better to
# keep the bad output than truncate to nothing and confuse downstream
# logic that expects non-empty narrations.
_MAX_OPENER_STRIP = 400


# Minimum length the narration must keep after stripping. If a strip
# would leave less than this many chars of body, we abort — protects
# against eating the entire narration when every sentence happens to
# match a greeting pattern (worst-case symptom : the LLM ignored the
# slide content and produced only meta-talk). 15 chars ≈ "The vector
# model..." — clearly substantive content.
_MIN_REMAINING_AFTER_STRIP = 15


# ── Parenthetical-duplicate stripper ────────────────────────────────
#
# Why this exists : weak LLMs (Ollama 7-8B, sometimes Llama 3.1 8B)
# tend to "explain" technical terms by appending a parenthetical that
# is either :
#   (a) a verbatim duplicate of the term itself, e.g.
#       "Notation de Bohlen-Muller (Notation de Bohlen-Muller)"
#   (b) a near-duplicate translation of the same term, e.g.
#       "L'analyse de termes-clés (Keyword Analysis)" then again the
#       same in the next sentence
# The prompt forbids this pattern but small LLMs often ignore it.
# This post-processor is the deterministic safety net.
#
# We collapse the pattern only when the parenthesised content is :
#   - identical (modulo case + whitespace) to the term right before it
#   - or contains the same noun roots (e.g., "Aspects pratiques de
#     l'IR (Aspects Pratiques de l'IR)")
# A legitimate parenthetical (e.g., "RI (Recherche d'Information)" with
# a different word inside the parens) is left intact.

_PAREN_DUPLICATE_RE = __import__("re").compile(
    # capture : (term)( (paren_content) )
    # term  = up-to-80 chars before the paren, broken on punctuation
    # paren = matching parenthesised group
    r"([A-Za-zÀ-ÿ0-9'’\-]+(?:\s+[A-Za-zÀ-ÿ0-9'’\-]+){0,7})"
    r"\s*\(([^()]{2,80})\)",
    flags=__import__("re").UNICODE,
)


def _normalise_for_compare(s: str) -> str:
    """Lowercase + accent-fold + drop short tokens, for duplicate
    detection. Same approach as the off-topic guardrail."""
    if not s:
        return ""
    import re as _re
    import unicodedata
    nfd = unicodedata.normalize("NFD", s)
    folded = "".join(c for c in nfd if not unicodedata.combining(c)).lower()
    tokens = [t for t in _re.split(r"\W+", folded) if len(t) >= 3]
    return " ".join(tokens)


def _strip_parenthetical_duplicates(text: str) -> tuple[str, int]:
    """Remove ``X (X)`` patterns where X and the parens content are
    semantically identical. Returns ``(cleaned, n_removed)``.

    Heuristic : a parenthetical is a duplicate when, after lowercase +
    accent-fold + 3-char-min token filter, every token in the parens
    appears in the term right before. That catches :
      - exact dups  : "Aspects Pratiques (Aspects Pratiques)"
      - case dups   : "modèle vectoriel (Modèle Vectoriel)"
      - language dups : "L'analyse de termes-clés (Keyword Analysis)"
        — fails this check (different tokens), kept intact ; that's
        actually OK because the FR + EN duplicate is informative.
    Safe : never removes a parenthetical that adds genuine information
    (acronym expansion, synonym, etc.).
    """
    if not text or "(" not in text:
        return text, 0

    n_removed = 0

    def _replace(match):
        nonlocal n_removed
        term = match.group(1)
        paren = match.group(2)
        norm_term = _normalise_for_compare(term)
        norm_paren = _normalise_for_compare(paren)
        if not norm_term or not norm_paren:
            return match.group(0)
        # All paren tokens must appear in the term -> duplicate
        paren_tokens = norm_paren.split()
        term_tokens_set = set(norm_term.split())
        if all(t in term_tokens_set for t in paren_tokens):
            n_removed += 1
            return term  # drop the parenthetical
        return match.group(0)

    cleaned = _PAREN_DUPLICATE_RE.sub(_replace, text)
    return cleaned, n_removed


def _strip_greeting_opener(text: str, lang: str) -> tuple[str, str]:
    """Strip greeting opener(s) from the start of ``text``.

    Iterative : strips ONE clause at a time, then checks if the
    remainder still has substantial content (>= _MIN_REMAINING_AFTER_STRIP
    chars). Stops when either no opener matches or the next strip would
    eat too much. This avoids two failure modes :

      - greedy multi-clause stripping consuming the whole narration
        when the LLM stacks 3-4 meta-sentences ("Bienvenue dans...
        Nous allons commencer... Dans ce chapitre, nous verrons...")
        followed by the actual content
      - leaving the narration with just one trailing word

    Returns ``(stripped_text, removed_openers)``. ``removed_openers``
    concatenates everything stripped (newline-joined) for log diagnosis.
    Safe on empty / short / opener-free inputs (returns them unchanged).
    Idempotent : applying twice yields the same result.
    """
    if not text:
        return text, ""
    pattern = _GREETING_OPENER_FR if lang == "fr" else _GREETING_OPENER_EN

    current = text
    removed_parts: list[str] = []
    total_stripped = 0

    while True:
        match = pattern.match(current)
        if not match:
            break
        end = match.end()
        # Lookahead : would the remainder still have enough content ?
        remainder = current[end:].lstrip()
        if len(remainder) < _MIN_REMAINING_AFTER_STRIP:
            # Stripping this clause would leave too little — keep it.
            break
        if total_stripped + end > _MAX_OPENER_STRIP:
            # Cumulative cap — refuse to strip more than _MAX_OPENER_STRIP
            # chars of "opener". An LLM that produces 400+ chars of
            # greetings before the actual content has likely ignored
            # the slide entirely ; better to surface the bad output.
            break
        removed_parts.append(current[:end].strip())
        total_stripped += end
        current = remainder

    if not removed_parts:
        return text, ""

    # Capitalise the first letter so the narration doesn't start mid-sentence.
    if current and current[0].islower():
        current = current[0].upper() + current[1:]

    return current, "\n".join(removed_parts)


def _summarise_for_next(plan, narration: str) -> str:
    """Build a 1-2 sentence recap of what was just narrated.

    Used as ``previous_narration_summary`` for the next slide so the
    narrator there can open with a continuity bridge. Cheap derivation:
    take the plan's summary brief if it exists, otherwise the last full
    sentence of the narration. No extra LLM call.
    """
    if plan and getattr(plan, "ideas", None):
        for idea in reversed(plan.ideas):
            if getattr(idea, "type", "") == "summary" and idea.content_brief:
                return idea.content_brief.strip()[:240]
    if not narration:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", narration.strip())
    if not sentences:
        return ""
    # Last 1-2 non-trivial sentences, truncated.
    tail = " ".join(s for s in sentences[-2:] if len(s) > 10)
    return tail.strip()[:240]


def _slide_similarity(a: str, b: str) -> float:
    """Quick token-Jaccard similarity between two slide contents.

    Used to detect near-duplicate consecutive slides (the same content
    with one tiny variation — a single bolded word, one bullet changed,
    a step revealed in a builds-style deck). Cheap (no embeddings) and
    monotone enough for a threshold gate.

    Returns 0.0 when either side is empty so first-slide-of-session
    never triggers the dedup path.
    """
    if not a or not b:
        return 0.0
    import re as _re
    tok_a = set(_re.findall(r"\w+", a.lower()))
    tok_b = set(_re.findall(r"\w+", b.lower()))
    if not tok_a or not tok_b:
        return 0.0
    inter = len(tok_a & tok_b)
    union = len(tok_a | tok_b)
    return inter / union if union else 0.0


# Threshold above which two consecutive slides are considered "near
# duplicates" — the narrator switches to diff-mode instead of repeating
# the full explanation. 0.80 = >=80% token overlap, calibrated on
# typical builds decks (PowerPoint "appear" animations split a single
# slide into N pages where the only delta is which bullet is highlighted).
_DUPLICATE_SLIDE_THRESHOLD = 0.80


def _build_augmented_content(
    plan,
    slide_content: str,
    lang: str,
    chunks: list | None = None,
    retry_feedback: str | None = None,
    style_hint: str | None = None,    # adaptation prose au learning_style étudiant
    main_concept: str | None = None,
    previous_concept: str | None = None,
    previous_summary: str | None = None,
    previous_slide_content: str | None = None,
) -> str:
    """Combine the plan brief with the original slide content + retrieved context.

    The plan acts as a roadmap that Brain.present() follows when generating
    the narration. Retrieved chunks supply authoritative grounding (definitions,
    prerequisites). retry_feedback is included when the reviewer has rejected
    a previous attempt and asks for corrections.
    """
    intro_fr = (
        "Voici un plan de présentation à suivre, suivi du contenu source de la slide. "
        "Présente les idées DANS L'ORDRE du plan, en respectant le type et la profondeur indiqués. "
        "Reste fidèle au contenu source — ne hallucine pas. "
        "IMPORTANT : NE MENTIONNE JAMAIS les mots 'source', 'référence', 'document', '[1]', '[2]'. "
        "Le contexte ci-dessous est INTERNE — parle naturellement comme un prof, sans citer.\n"
        "🚫 INTERDICTION ABSOLUE DES SALUTATIONS : tu es au MILIEU d'un cours en cours, "
        "PAS au début. Ne commence JAMAIS la narration par \"Bonjour\", \"Bienvenue\", "
        "\"Bienvenue dans la partie X\", \"Dans le cadre de cette introduction\", "
        "\"Bonjour à tous\", ou toute autre formule de salutation/d'introduction de chapitre. "
        "Le cours coule en continu — chaque slide enchaîne avec la précédente comme un "
        "professeur qui parle sans interruption. Commence DIRECTEMENT par le contenu "
        "(une transition courte si applicable, sinon directement le concept). "
        "Exemple INTERDIT : \"Bienvenue dans la partie III du cours...\". "
        "Exemple CORRECT : \"Voyons maintenant les applications de la RI...\" ou "
        "directement \"Le moteur de recherche Web est l'application la plus connue...\".\n"
        "🚫 NE PAS DUPLIQUER LES TERMES : ne mets JAMAIS un terme suivi de sa "
        "traduction ou de lui-même entre parenthèses. "
        "Exemples INTERDITS : \"Notation de Bohlen-Muller (Notation de Bohlen-Muller)\", "
        "\"L'analyse de termes-clés (Keyword Analysis)\", \"Le modèle vectoriel "
        "(Vector Model)\", \"Aspects pratiques (Aspects Pratiques)\". "
        "Si tu mentionnes un terme technique, dis-le UNE SEULE FOIS dans la "
        "langue du cours. La traduction n'est UTILE que pour un acronyme "
        "important à la première occurrence (ex : \"la RI, ou Information "
        "Retrieval, ...\"). N'inverse pas l'ordre, ne réplique pas le terme.\n"
        "🚫 NE PAS INVENTER DE NOMS PROPRES : ne mentionne JAMAIS un nom de "
        "personne, théorème, méthode ou outil qui ne figure PAS explicitement "
        "dans la slide ou le contexte. Si tu hésites sur un nom, omets-le "
        "plutôt que d'inventer (ex : ne dis pas \"Notation de Bohlen-Muller\" "
        "si \"Bohlen-Muller\" n'est pas dans la slide)."
    )
    intro_en = (
        "Below is a presentation plan, followed by the source slide content. "
        "Present the ideas IN THE PLAN ORDER, respecting their type and depth. "
        "Stay faithful to the source content — do not hallucinate. "
        "IMPORTANT: NEVER mention the words 'source', 'reference', 'document', '[1]', '[2]'. "
        "The context below is INTERNAL — speak naturally like a teacher, without citing.\n"
        "🚫 NO GREETINGS ALLOWED : you are in the MIDDLE of an ongoing lecture, "
        "NOT at the start. NEVER begin the narration with \"Hello\", \"Welcome\", "
        "\"Welcome to part X\", \"In this introduction\", or any greeting / "
        "chapter-opening phrase. The lecture flows continuously — each slide "
        "carries on from the previous one like a professor speaking without "
        "interruption. Start DIRECTLY with the content (a short transition if "
        "relevant, otherwise straight into the concept). "
        "FORBIDDEN example: \"Welcome to part III of the course...\". "
        "CORRECT example: \"Let's now look at IR applications...\" or "
        "directly \"The Web search engine is the best-known application...\".\n"
        "🚫 DO NOT DUPLICATE TERMS : NEVER write a term followed by its "
        "translation or itself in parentheses. FORBIDDEN examples : "
        "\"Boolean Notation (Notation de Bohlen-Muller)\", \"Vector Model "
        "(Vecteur de Représentation Model)\", \"Practical Aspects (Aspects "
        "Pratiques)\". Mention a technical term ONCE, in the course language. "
        "A translation is only useful for an acronym at first occurrence "
        "(e.g., \"IR, or Information Retrieval, ...\"). Do not invert the "
        "order or repeat the term.\n"
        "🚫 DO NOT INVENT PROPER NAMES : never mention a person's name, "
        "theorem, method or tool that is NOT explicitly in the slide or "
        "the context. If you're unsure of a name, omit it rather than invent "
        "(e.g., do not say \"Bohlen-Muller Notation\" if \"Bohlen-Muller\" "
        "is not in the slide)."
    )
    intro = intro_fr if lang == "fr" else intro_en

    # Concept anchor — keep the narrator on the *real* concept, not a
    # footnote or generic section title.
    concept_block = ""
    if main_concept:
        if lang == "fr":
            concept_block = (
                f"\n\n🎯 CONCEPT PRINCIPAL DE CETTE SLIDE : {main_concept}\n"
                f"Centre toute la narration sur ce concept. Ne traite pas une légende "
                f"de notation (ex. \"n est la taille de l'échantillon\") comme le sujet principal."
            )
        else:
            concept_block = (
                f"\n\n🎯 MAIN CONCEPT OF THIS SLIDE: {main_concept}\n"
                f"Anchor the entire narration on this concept. Do not treat a notation "
                f"legend (e.g. \"n is the sample size\") as the main subject."
            )

    # Continuity bridge — open with a short link to the previous slide so
    # the lesson flows instead of restarting cold on every slide.
    bridge_block = ""
    if previous_concept or previous_summary:
        if lang == "fr":
            bridge_block = (
                "\n\n🔗 CONTINUITÉ AVEC LA SLIDE PRÉCÉDENTE :\n"
            )
            if previous_concept:
                bridge_block += f"- Concept précédent : {previous_concept}\n"
            if previous_summary:
                bridge_block += f"- Résumé : {previous_summary}\n"
            bridge_block += (
                "Ouvre la narration par UNE phrase de transition qui relie le concept "
                "précédent au nouveau (ex. \"Nous venons de voir X — voyons maintenant Y\"). "
                "Ne re-définis PAS le concept précédent, contente-toi de le mentionner."
            )
        else:
            bridge_block = (
                "\n\n🔗 CONTINUITY WITH PREVIOUS SLIDE:\n"
            )
            if previous_concept:
                bridge_block += f"- Previous concept: {previous_concept}\n"
            if previous_summary:
                bridge_block += f"- Recap: {previous_summary}\n"
            bridge_block += (
                "Open the narration with ONE bridging sentence that links the previous "
                "concept to the new one (e.g. \"We just saw X — now let's look at Y\"). "
                "Do NOT re-define the previous concept, just acknowledge it."
            )

    # Near-duplicate detection — consecutive slides with very similar text
    # (typical of "build" decks where each slide reveals one more bullet
    # of the same list). Without this hint the LLM re-explains the whole
    # slide every time, which is repetitive and frustrating.
    duplicate_block = ""
    sim = _slide_similarity(previous_slide_content or "", slide_content)
    if sim >= _DUPLICATE_SLIDE_THRESHOLD:
        if lang == "fr":
            duplicate_block = (
                f"\n\n♻️ SLIDE QUASI-IDENTIQUE À LA PRÉCÉDENTE "
                f"(similarité {sim:.0%}) :\n"
                "Cette slide est presque la même que la précédente — typiquement une "
                "diapositive 'build' qui révèle un point supplémentaire ou met en "
                "évidence un mot. NE RÉPÈTE PAS l'explication complète. À la place, "
                "POINTE UNIQUEMENT la différence en 1-2 phrases : ce qui apparaît, "
                "ce qui est mis en avant, ou ce qui est ajouté. "
                "Ex. : \"Sur cette slide, le prof met l'accent sur X — c'est le point "
                "à retenir parmi les éléments qu'on a déjà vus.\""
            )
        else:
            duplicate_block = (
                f"\n\n♻️ NEAR-DUPLICATE SLIDE (similarity {sim:.0%}):\n"
                "This slide is almost identical to the previous one — typically a "
                "'build' slide that reveals one more bullet or highlights a different "
                "word. DO NOT repeat the full explanation. Instead, POINT OUT ONLY "
                "the difference in 1-2 sentences: what just appeared, what is now "
                "emphasised, or what was added. "
                "E.g.: \"On this slide, the focus shifts to X — that's the takeaway "
                "from the items we just saw.\""
            )

    plan_brief = plan.to_brief() if plan else "(no plan available — present the slide naturally)"

    # Retrieved chunks (RAG context for grounding) — labels KG injectes pour
    # structurer la narration. Avant : ContextAgent bypass-ait l'augmentation
    # KG → la narration ignorait la structure pedagogique. Maintenant les
    # chunks `_via_kg=True` portent un tag [rappel] / [exemple] / [concept
    # general] que le narrator peut utiliser pour structurer son discours
    # (ouvrir par un rappel, illustrer par un exemple), sans citer le tag
    # dans la voix lue.
    _KG_NARRATOR_LABELS = {
        "fr": {"prereq": "rappel", "example": "exemple",
               "illustrated": "concept general"},
        "en": {"prereq": "reminder", "example": "example",
               "illustrated": "general concept"},
    }
    chunks_block = ""
    if chunks:
        labels = _KG_NARRATOR_LABELS.get(lang, _KG_NARRATOR_LABELS["fr"])
        # Prendre jusqu'a 5 chunks (plus que les 3 d'avant) pour profiter des
        # KG-augmented qui sont append-es a la fin par ContextAgent.
        formatted = []
        kg_relations_seen: set[str] = set()
        for ch in chunks[:5]:
            if not isinstance(ch, dict):
                content = str(ch)[:400]
                if content:
                    formatted.append(content)
                continue
            content = (ch.get("content") or "")[:400]
            if not content:
                continue
            kg_relation = ch.get("_kg_relation") if ch.get("_via_kg") else None
            if kg_relation and kg_relation in labels:
                tag = labels[kg_relation]
                kg_relations_seen.add(kg_relation)
                formatted.append(f"[{tag}] {content}")
            else:
                formatted.append(content)
        if formatted:
            label = ("CONTEXTE INTERNE (ne pas mentionner dans la narration) :"
                     if lang == "fr"
                     else "INTERNAL CONTEXT (do not mention in narration):")
            # Hint au LLM : utiliser les tags pour structurer la narration
            usage_hint = ""
            if kg_relations_seen:
                if lang == "fr":
                    parts = []
                    if "prereq" in kg_relations_seen:
                        parts.append("[rappel] = prerequis a evoquer brievement en ouverture")
                    if "example" in kg_relations_seen:
                        parts.append("[exemple] = cas concret a integrer pour illustrer")
                    if "illustrated" in kg_relations_seen:
                        parts.append("[concept general] = concept dont la slide est un exemple — situer brievement")
                    usage_hint = " " + " | ".join(parts) + "."
                else:
                    parts = []
                    if "prereq" in kg_relations_seen:
                        parts.append("[reminder] = prerequisite to briefly recall at the opening")
                    if "example" in kg_relations_seen:
                        parts.append("[example] = concrete case to weave in for illustration")
                    if "illustrated" in kg_relations_seen:
                        parts.append("[general concept] = concept the slide illustrates — situate briefly")
                    usage_hint = " " + " | ".join(parts) + "."
            chunks_block = f"\n\n{label}{usage_hint}\n" + "\n---\n".join(formatted)

    # Retry feedback
    feedback_block = ""
    if retry_feedback:
        label = "CORRECTION DEMANDÉE par le réviseur :" if lang == "fr" else "REVIEWER FEEDBACK to address:"
        feedback_block = f"\n\n⚠️ {label} {retry_feedback}"

    # Style hint (prose) — soft cognitive guidance, not a hard directive.
    # The earlier numerical "directives" block (max_sentences=5, etc.) was
    # removed because those magic numbers had no empirical justification.
    style_block = ""
    if style_hint:
        label = "STYLE D'APPRENTISSAGE de l'étudiant :" if lang == "fr" else "STUDENT LEARNING STYLE:"
        style_block = f"\n\n🎯 {label} {style_hint}"

    return (
        f"{intro}"
        f"{concept_block}"
        f"{bridge_block}"
        f"{duplicate_block}\n\n"
        f"{plan_brief}"
        f"{chunks_block}"
        f"{feedback_block}"
        f"{style_block}\n\n"
        f"---\n"
        f"SOURCE SLIDE CONTENT:\n{slide_content}"
    )


class NarratorAgent:
    """Calls Brain.present() with an augmented brief built from the plan."""

    def __init__(self, brain) -> None:
        self.brain = brain

    def __call__(self, state: TutorState) -> dict[str, Any]:
        start = time.time()
        plan = state.get("plan")
        slide = state.get("last_slide_content") or ""
        lang = (state.get("language") or "fr")[:2]
        chunks = state.get("retrieved_chunks") or []
        review = state.get("review")
        # Reviewer is binary (grounded / not grounded). Pass feedback only on
        # rejection so the narrator can correct.
        retry_feedback = (
            review.feedback if (review and not getattr(review, "grounded", True)) else None
        )
        retries = int(state.get("narrator_retries", 0))

        if not slide:
            log.warning("narrator: empty slide_content → returning empty narration")
            return {
                "answer": "",
                "timings": {**state.get("timings", {}), "narrator": 0.0},
            }

        style_hint = state.get("learning_style_hint") or None
        main_concept = (state.get("main_concept_hint") or "").strip() or None
        previous_concept = (state.get("previous_concept") or "").strip() or None
        previous_summary = (state.get("previous_narration_summary") or "").strip() or None
        previous_slide_content = (state.get("previous_slide_content") or "").strip() or None
        augmented = _build_augmented_content(
            plan, slide, lang, chunks, retry_feedback,
            style_hint=style_hint,
            main_concept=main_concept,
            previous_concept=previous_concept,
            previous_summary=previous_summary,
            previous_slide_content=previous_slide_content,
        )

        # Brain.present is sync; LangGraph is fine calling sync funcs
        log.info(
            "🔍 narrator AUGMENTED PROMPT (%d chars, lang=%s, plan=%d ideas) ===\n%s\n=== END",
            len(augmented), lang, len((plan and plan.ideas) or []),
            augmented[:3000] + ("…[truncated]" if len(augmented) > 3000 else ""),
        )
        narration, _duration = self.brain.present(
            section_content=augmented,
            language=lang,
            chapter_idx=state.get("chapter_idx"),
            chapter_title=state.get("chapter_title", ""),
            section_title=state.get("section_title", ""),
            domain=state.get("domain"),
            session_id=state.get("session_id"),
        )
        log.info("🔍 narrator LLM RAW (%d chars):\n%s",
                 len(narration or ""), (narration or "")[:1500])

        # Strip greeting openers ("Bonjour", "Bienvenue", "Dans le cadre
        # de cette introduction…") that weak LLMs inject despite the
        # prompt's "no greetings" rule. The student is in the MIDDLE of
        # a course — every slide opening with a fresh greeting breaks
        # the flow. Deterministic post-processing because prompt-only
        # enforcement isn't reliable on Ollama 7-8B.
        if narration:
            stripped, removed_opener = _strip_greeting_opener(narration, lang)
            if removed_opener:
                log.info(
                    "🚫 narrator: stripped greeting opener (%d chars): %r",
                    len(removed_opener), removed_opener[:120],
                )
                narration = stripped
            else:
                log.debug("narrator: no greeting opener detected")

            # Strip parenthetical duplicates ("X (X)" or "X (x case-fold)").
            # The prompt forbids them but Ollama 7-8B reliably ignores
            # the rule. Deterministic safety net.
            narration_after_paren, n_dup_removed = _strip_parenthetical_duplicates(narration)
            if n_dup_removed:
                log.info(
                    "🚫 narrator: stripped %d parenthetical duplicate(s)",
                    n_dup_removed,
                )
                narration = narration_after_paren

        # Surface truncation/cliffhanger so the operator can act on it.
        # Doesn't block the response — the cliffhanger narration is still
        # better than nothing — but tags it in logs and in the returned
        # state so callers can decide whether to retry or extend.
        truncated = _is_cliffhanger(narration or "", lang)
        if truncated:
            log.warning(
                "narrator: narration ends in a cliffhanger (likely token-truncated). "
                "tail=%r",
                (narration or "")[-120:],
            )

        log.info(
            "narrator: %d chars (ch=%s lang=%s retry=%d chunks=%d concept=%r truncated=%s)",
            len(narration or ""),
            state.get("chapter_idx"),
            lang,
            retries,
            len(chunks),
            (main_concept or "")[:40],
            truncated,
        )

        last_idea_id = plan.ideas[-1].id if (plan and plan.ideas) else None
        narration_summary = _summarise_for_next(plan, narration or "")
        return {
            "answer": narration or "",
            "last_narrated_idea": last_idea_id,
            "narration_summary": narration_summary,
            "narrator_retries": retries + 1 if retry_feedback else retries,
            "timings": {**state.get("timings", {}), "narrator": round(time.time() - start, 3)},
        }
