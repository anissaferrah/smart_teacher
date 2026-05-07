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
        "Le contexte ci-dessous est INTERNE — parle naturellement comme un prof, sans citer."
    )
    intro_en = (
        "Below is a presentation plan, followed by the source slide content. "
        "Present the ideas IN THE PLAN ORDER, respecting their type and depth. "
        "Stay faithful to the source content — do not hallucinate. "
        "IMPORTANT: NEVER mention the words 'source', 'reference', 'document', '[1]', '[2]'. "
        "The context below is INTERNAL — speak naturally like a teacher, without citing."
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
        narration, _duration = self.brain.present(
            section_content=augmented,
            language=lang,
            chapter_idx=state.get("chapter_idx"),
            chapter_title=state.get("chapter_title", ""),
            section_title=state.get("section_title", ""),
            domain=state.get("domain"),
            session_id=state.get("session_id"),
        )

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
