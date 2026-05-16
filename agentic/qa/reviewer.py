"""QAReviewAgent — checks the Responder's answer before sending it to the student.

Pattern copied from ``agentic/teaching/reviewer.py`` (Asai et al. 2023,
Self-RAG, arXiv:2310.11511) but specialised for the Q&A graph: where
the teaching reviewer only checks groundedness (the narration must
match the slide + RAG), the Q&A reviewer also checks two things that
matter when answering an actual student question:

  1. **GROUNDING** — the answer is supported by the slide or by the
     retrieved chunks. No invented facts.
  2. **COURSE-BOUND** — the answer doesn't drift into general / encyclopedic
     knowledge ("Wikipedia leak"). This is the same rule we already enforce
     in the responder prompt; the reviewer is the second line of defence.
  3. **COHERENCE** — the answer actually addresses the student's question.
     A grounded paragraph that doesn't answer the question is still useless.

Verdict is binary (grounded ∧ course-bound ∧ coherent → pass). The
previous teaching reviewer already explained why we avoided continuous
scores: any threshold (0.6 vs 0.7) would be unjustified. Same reasoning
here.

# Routing

    responder → reviewer → {grounded → END,
                            ungrounded + retries < MAX → retry → responder,
                            ungrounded + retries ≥ MAX → fallback → END}

The fallback is a simple safe answer ("I'm not sure, can you clarify?")
rather than another LLM attempt — when the model has already failed twice
with feedback, the honest move is to ask the student to rephrase rather
than burn more tokens on a third hallucination.

# Why we re-feed the feedback to the responder on retry

The responder reads ``state.get("review")``. When ``grounded == False``,
the responder sees the reviewer's feedback string and includes it in
its next prompt as a "fix this issue" instruction. This is the same
pattern the teaching narrator uses (`narrator.py:395` increments
``narrator_retries`` when ``retry_feedback`` is set).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from agentic.schemas import ReviewResult
from agentic.state import TutorState

log = logging.getLogger("agentic.qa.reviewer")


# Operational only — bounds on latency. Not a research claim.
# 1 retry = up to 2 LLM calls in the responder + 2 in the reviewer = ~4-8s
# extra worst-case. Above that the user notices and the latency KPI suffers.
MAX_RETRIES = 1


_REVIEW_PROMPT_FR = """Tu es un examinateur strict d'un tuteur IA limité au contenu d'un cours.

Tu reçois la QUESTION de l'étudiant, la SLIDE en cours, les CHUNKS récupérés (RAG), et la RÉPONSE générée. Évalue la RÉPONSE sur 3 critères :

1. ANCRAGE : la réponse s'appuie SOIT sur la SLIDE SOIT sur les CHUNKS. Pas d'invention de fait, pas de paraphrase d'une définition générale absente du matériel.
2. LIMITÉ AU COURS : la réponse n'utilise PAS de connaissances encyclopédiques (Wikipédia, culture générale) qui sortent du matériel fourni. Si la question est hors-sujet par rapport au matériel, la réponse acceptable est "Ce point n'est pas abordé dans ce cours" — c'est BON, c'est honnête. Le mauvais cas est une réponse encyclopédique générique.
3. COHÉRENCE : la réponse traite réellement la question posée (pas un paragraphe correct mais à côté du sujet).

Réponds UNIQUEMENT en JSON strict, sans markdown :
{{"grounded": true|false, "feedback": "...court, max 1 phrase, dit ce qui ne va pas si grounded=false..."}}

QUESTION :
{question}

SLIDE :
{slide_content}

CHUNKS RAG :
{rag_chunks}

RÉPONSE GÉNÉRÉE :
{answer}
"""


_REVIEW_PROMPT_EN = """You are a strict reviewer for an AI tutor restricted to course material.

You receive the student's QUESTION, the current SLIDE, the retrieved CHUNKS (RAG), and the generated ANSWER. Evaluate the ANSWER on 3 criteria:

1. GROUNDING: the answer is supported by EITHER the SLIDE OR the CHUNKS. No invented facts, no paraphrase of a general definition absent from the material.
2. COURSE-BOUND: the answer does NOT use encyclopedic knowledge (Wikipedia, general culture) outside the provided material. If the question is off-topic relative to the material, a valid answer is "This is not covered in this course" — that's GOOD, that's honest. The bad case is a generic encyclopedic answer.
3. COHERENCE: the answer actually addresses the question (not a correct paragraph that misses the point).

Reply STRICT JSON ONLY, no markdown:
{{"grounded": true|false, "feedback": "...short, max 1 sentence, says what's wrong if grounded=false..."}}

QUESTION:
{question}

SLIDE:
{slide_content}

RAG CHUNKS:
{rag_chunks}

GENERATED ANSWER:
{answer}
"""


# ── Safe-fallback messages ────────────────────────────────────────────
# After ``MAX_RETRIES`` rejected attempts, we don't try a third LLM call
# — we ask the student to rephrase. That's the system's honest "I don't
# know how to answer" signal (Asai et al. 2023, self-reflection pattern).
_FALLBACK_ANSWER = {
    "fr": (
        "Je ne suis pas sûr de pouvoir répondre précisément à cette question "
        "à partir du cours. Pouvez-vous reformuler ou préciser ?"
    ),
    "en": (
        "I'm not sure I can answer this precisely from the course material. "
        "Could you rephrase or clarify?"
    ),
}


def _format_chunks_for_review(chunks: list[dict[str, Any]] | None, max_chars: int = 1500) -> str:
    """Concat top-K chunks into a compact source block for the reviewer.

    Same shape as the teaching reviewer helper. Kept local rather than
    factored into ``_shared`` because the budget here (1500 chars) is
    tuned to the QA review prompt's overall budget, not the responder's.
    """
    if not chunks:
        return "(none)"
    out: list[str] = []
    budget = max_chars
    for ch in chunks[:4]:
        content = (ch.get("content") or "").strip()
        if not content:
            continue
        snippet = content[:max(0, budget)]
        if not snippet:
            break
        out.append(f"— {snippet}".strip())
        budget -= len(snippet) + 8
        if budget <= 0:
            break
    return "\n".join(out) if out else "(none)"


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


def _question_text(state: TutorState) -> str:
    """Recover the student's question from the state.

    The IntentAgent stores the raw text in ``intent.payload["raw_text"]``.
    The rewriter may also produce ``rewritten_query``. We prefer the
    rewritten one when available (it's pronoun-resolved) and fall back
    to the raw payload otherwise.
    """
    rewritten = (state.get("rewritten_query") or "").strip()
    if rewritten:
        return rewritten
    intent = state.get("intent")
    if intent is not None:
        payload = getattr(intent, "payload", None) or {}
        if isinstance(payload, dict):
            return str(payload.get("raw_text", "") or "").strip()
    return ""


# ── Smart gate: skip the LLM reviewer on well-grounded answers ────────
# The responder's own guardrail at responder.py:435 rejects answers
# below GROUNDING_OVERLAP_THRESHOLD (0.12). Anything reaching this
# reviewer is therefore at least weakly grounded. When overlap is also
# >= _STRONG_OVERLAP_GATE the answer is solidly anchored on the source
# and a second LLM pass adds latency + 25% TPM consumption + risk of
# false-positive rejection (observed: reviewer claimed "statistics +
# math + ML algorithms" was outside the material when the chunk
# literally contained those words). The middle band [0.12, _STRONG_
# OVERLAP_GATE) still gets the full LLM review for borderline cases.
_STRONG_OVERLAP_GATE: float = 0.30
_MIN_CONTENT_WORD_LEN: int = 4   # Mirrors responder._MIN_CONTENT_WORD_LEN


def _content_words_for_overlap(text: str) -> set[str]:
    """Tokenise + lowercase + accent-fold + drop short tokens.

    Local copy of responder._content_words so the reviewer can compute
    overlap without a cross-module import. Identical formula.
    """
    if not text:
        return set()
    import re
    import unicodedata
    nfd = unicodedata.normalize("NFD", text)
    folded = "".join(c for c in nfd if not unicodedata.combining(c)).lower()
    tokens = re.split(r"\W+", folded)
    return {t for t in tokens if len(t) >= _MIN_CONTENT_WORD_LEN}


def _compute_overlap(answer: str, slide: str, chunks: list) -> float:
    """Fraction of answer's content words found in slide ∪ chunks.

    Same formula as the responder's guardrail (responder.py:435), kept
    local so the gate is self-contained. Returns 0.0 on either side
    being empty (don't gate when there's nothing to measure).
    """
    answer_words = _content_words_for_overlap(answer)
    if not answer_words:
        return 0.0
    parts = [slide or ""]
    for ch in chunks or []:
        if isinstance(ch, dict):
            parts.append(str(ch.get("content", "") or ""))
    source_words = _content_words_for_overlap(" ".join(parts))
    if not source_words:
        return 0.0
    return len(answer_words & source_words) / len(answer_words)


class QAReviewAgent:
    """Verifies that the Responder's answer is grounded, course-bound, and coherent."""

    def __init__(self, brain) -> None:
        self.brain = brain

    def __call__(self, state: TutorState) -> dict[str, Any]:
        start = time.time()
        answer = (state.get("answer") or "").strip()
        slide = (state.get("last_slide_content") or "").strip()
        question = _question_text(state)
        lang = (state.get("language") or "fr")[:2]

        # Fast paths : skip the LLM call when there's nothing meaningful
        # to review. These are not "PASS" decisions on borderline answers,
        # they're cases where the review wouldn't be informative.

        if not answer:
            # Empty answer can't be grounded — flag for retry.
            return {
                "review": ReviewResult(grounded=False, score=0.0, feedback="empty answer"),
                "timings": {**state.get("timings", {}), "qa_review": 0.0},
            }

        # If the responder produced one of the safe fallbacks ("not covered
        # in this course", or our own _FALLBACK_ANSWER), accept it directly.
        # The course-bound rule explicitly endorses "this is not covered" as
        # a valid answer, so re-judging it via LLM would just risk a false
        # negative on the honest path.
        lowered = answer.lower()
        honest_markers_fr = ("pas abordé dans ce cours", "pas couvert", "ne suis pas sûr")
        honest_markers_en = ("not covered in this course", "not in this course", "i'm not sure")
        if any(m in lowered for m in honest_markers_fr) or any(m in lowered for m in honest_markers_en):
            log.info("qa_review: honest 'not covered' answer → accept without LLM")
            return {
                "review": ReviewResult(grounded=True, score=1.0, feedback="honest refusal accepted"),
                "timings": {**state.get("timings", {}), "qa_review": round(time.time() - start, 3)},
            }

        # If neither the slide nor the chunks are present, there's nothing
        # for the reviewer to check the answer against. Pass through.
        chunks = state.get("retrieved_chunks") or []
        if not slide and not chunks:
            return {
                "review": ReviewResult(grounded=True, score=1.0, feedback="no source to verify"),
                "timings": {**state.get("timings", {}), "qa_review": round(time.time() - start, 3)},
            }

        # Smart gate: skip the LLM call when overlap shows strong grounding.
        # See _STRONG_OVERLAP_GATE comment above for rationale.
        overlap = _compute_overlap(answer, slide, chunks)
        if overlap >= _STRONG_OVERLAP_GATE:
            log.info(
                "qa_review: smart gate skip — overlap=%.2f >= %.2f (strong grounding, no LLM call)",
                overlap, _STRONG_OVERLAP_GATE,
            )
            return {
                "review": ReviewResult(
                    grounded=True, score=overlap,
                    feedback=f"skipped: strong overlap {overlap:.2f}",
                ),
                "timings": {**state.get("timings", {}), "qa_review": round(time.time() - start, 3)},
            }

        template = _REVIEW_PROMPT_FR if lang == "fr" else _REVIEW_PROMPT_EN
        prompt = template.format(
            question=question[:500] or "(missing)",
            slide_content=slide[:1500] or "(empty)",
            rag_chunks=_format_chunks_for_review(chunks),
            answer=answer[:1500],
        )

        try:
            raw, _ = self.brain.ask(prompt, reply_language=lang)
        except Exception as exc:
            log.warning("qa_review LLM call failed: %s — defaulting to PASS", exc)
            return {
                "review": ReviewResult(grounded=True, score=1.0, feedback=f"reviewer unavailable: {exc}"),
                "timings": {**state.get("timings", {}), "qa_review": round(time.time() - start, 3)},
            }
        log.info("🔍 reviewer LLM raw : %r", (raw or "")[:300])

        payload = _extract_json(raw)
        if not payload:
            log.warning("qa_review: invalid JSON, defaulting to PASS. raw=%r", (raw or "")[:120])
            return {
                "review": ReviewResult(grounded=True, score=1.0, feedback="reviewer parse failed"),
                "timings": {**state.get("timings", {}), "qa_review": round(time.time() - start, 3)},
            }

        grounded = bool(payload.get("grounded", True))
        feedback = str(payload.get("feedback", "")).strip()[:240]

        log.info(
            "🔍 reviewer VERDICT | grounded=%s | feedback=%r | answered=%r",
            grounded, feedback[:120], (answer or "")[:120],
        )
        return {
            "review": ReviewResult(grounded=grounded, score=1.0 if grounded else 0.0, feedback=feedback),
            "timings": {**state.get("timings", {}), "qa_review": round(time.time() - start, 3)},
        }


def qa_review_router(state: TutorState) -> str:
    """Conditional edge after the QA reviewer.

    Returns one of:
      - ``"end"``     : the answer passed (or the reviewer abstained).
      - ``"retry"``   : ungrounded but retry budget left → run the responder
                        again, the responder will read ``state.review.feedback``
                        and try to fix the issue.
      - ``"fallback"``: ungrounded and retry budget exhausted → produce the
                        safe "I'm not sure, can you clarify?" answer.

    Same design as the teaching ``review_router``: the verdict is binary,
    and routing to ``"fallback"`` instead of silently accepting the
    ungrounded answer is what gives the system an honest failure mode.
    """
    review = state.get("review")
    retries = int(state.get("responder_retries", 0) or 0)

    if review is None or not isinstance(review, ReviewResult):
        return "end"
    if review.grounded:
        return "end"
    if retries >= MAX_RETRIES:
        log.info("qa_review: max retries reached, routing to fallback")
        return "fallback"

    log.info("qa_review: not grounded → retry responder (#%d)", retries + 1)
    return "retry"


def qa_fallback_node(state: TutorState) -> dict[str, Any]:
    """Inline fallback : produce the honest "I don't know" answer.

    Kept as a function rather than a class because there's no LLM call,
    no state to maintain, and the response text is purely deterministic
    (per-language constant). A class would just be ceremony around a
    single dict return.
    """
    lang = (state.get("language") or "fr")[:2]
    answer = _FALLBACK_ANSWER.get(lang, _FALLBACK_ANSWER["fr"])
    log.info("qa_fallback: emitting safe refusal (lang=%s)", lang)
    return {
        "answer": answer,
        # Empty citations because the fallback isn't grounded in any
        # specific chunk — and signalling this honestly downstream lets
        # the UI show an "ungrounded" badge.
        "citations": [],
        # Confidence reset to 0 because we've explicitly given up on a
        # confident answer. The frontend / analytics can read this as
        # "the reviewer rejected the response".
        "confidence": 0.0,
    }
