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

_NAV_REPLIES = {
    "fr": "D'accord, je reviens sur ce point.",
    "en": "Sure, let me go back to that.",
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
                "L'étudiant a déjà vu ces concepts. Fais une RÉVISION RAPIDE en 2-3 phrases "
                "max — un rappel-clé, pas une explication complète."
            )
        elif memory_mode == "partial":
            tone = (
                "L'étudiant connait déjà certains concepts. "
                "Concentre-toi sur les NOUVEAUX éléments et ne ré-explique pas ce qu'il sait déjà."
            )
        else:
            tone = "Réponds en 3-4 phrases naturelles, comme un prof qui parle."

        course_bound_rule = (
            "⚠️ RÈGLE STRICTE — TUTEUR LIMITÉ AU COURS : tu ne réponds QUE à partir de la "
            "SLIDE EN COURS et du CONTEXTE INTERNE fournis ci-dessous. Tu n'es PAS un "
            "chatbot encyclopédique : tu n'utilises PAS ta connaissance générale, ni "
            "Wikipédia, ni d'exemples extérieurs au cours. Si la question porte sur un "
            "sujet qui n'est ni dans la slide ni dans le contexte interne, tu DOIS répondre "
            "honnêtement par UNE SEULE phrase, ex : \"Ce point n'est pas abordé dans ce "
            "cours — je ne peux pas l'expliquer ici sans m'écarter du programme.\" "
            "(et laisse \"supporting_chunks\" vide). N'invente RIEN, ne paraphrase RIEN "
            "depuis l'extérieur du cours."
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
            "Student already saw these concepts. Give a QUICK REFRESH in 2-3 sentences max — "
            "a key reminder, not a full explanation."
        )
    elif memory_mode == "partial":
        tone = (
            "Student already knows some of these concepts. "
            "Focus on the NEW elements and don't re-explain what they already know."
        )
    else:
        tone = "Answer in 3-4 natural spoken sentences."

    course_bound_rule = (
        "⚠️ STRICT RULE — COURSE-BOUND TUTOR: you only answer from the CURRENT SLIDE "
        "and the INTERNAL CONTEXT provided below. You are NOT an encyclopedic chatbot: "
        "you do NOT use your general knowledge, Wikipedia, or examples outside this "
        "course. If the question concerns a topic that is neither on the slide nor in "
        "the internal context, you MUST honestly reply with ONE sentence, e.g.: \"This "
        "point is not covered in this course — I cannot explain it here without going "
        "outside the syllabus.\" (and leave \"supporting_chunks\" empty). Do NOT invent, "
        "do NOT paraphrase anything from outside the course."
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
            answer = _NAV_REPLIES.get(lang, _NAV_REPLIES["fr"])
            actions = [Action(type="replay_concept", payload={"hint": "raw"})]
            log.info("responder: navigation static reply")
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

        # Append the bandit-chosen strategy fragment, if any.
        if bandit_decision is not None and bandit_decision.prompt_fragment:
            label = "STRATÉGIE PÉDAGOGIQUE" if lang == "fr" else "PEDAGOGICAL STRATEGY"
            personalization_prefix = (
                f"{personalization_prefix}{label} : {bandit_decision.prompt_fragment}\n\n"
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

        # Parse output: JSON for QA path, raw text for confusion path.
        if expects_json:
            answer, citations = _parse_qa_response(raw_text, id_to_chunk)
        else:
            answer, citations = raw_text, []

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
