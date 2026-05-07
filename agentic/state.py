"""Smart Teacher — Shared TutorState for all LangGraph pipelines.

A single TypedDict shared between Teaching, Q&A and Assessment graphs.
Most fields are optional (`total=False`) because each graph only reads/writes
a subset. The Global Orchestrator decides which fields are populated before
dispatching to a subgraph.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from agentic.schemas import (
    Action,
    ConfusionEvent,
    ConfusionResult,
    PresentationPlan,
    ReviewResult,
    VoiceIntent,
)


EventType = Literal["present_section", "student_speech", "feedback", "quiz_request"]
StudentLevel = Literal["collège", "lycée", "université"]
LearningStyle = Literal["visual", "auditory", "mixed"]


class TutorState(TypedDict, total=False):
    """Shared state across Teaching, Q&A and Assessment graphs."""

    # ── Identification ──────────────────────────────────────────────
    session_id: str
    student_id: str | None  # UUID, nullable pour sessions anonymes
    course_id: str
    language: Literal["fr", "en"]
    domain: str | None

    # ── Position pédagogique (sémantique, pas slide) ────────────────
    chapter_idx: int
    chapter_title: str
    section_idx: int
    section_title: str
    current_idea_id: str | None
    slide_idx_hint: int | None

    # ── Student model (long-term, lu via ProfileManager) ────────────
    student_level: StudentLevel
    learning_style: LearningStyle | None
    confusion_history: list[ConfusionEvent]
    mastery_map: dict[str, float]  # concept_id -> mastery in [0, 1]

    # ── Working memory ──────────────────────────────────────────────
    history: list[dict[str, Any]]  # list of {"role", "content"}
    last_slide_content: str
    last_narrated_idea: str | None

    # ── Slide-level pedagogical anchors (Teaching Graph) ────────────
    # slide_image_path    : on-disk path (or /media/ URL) to the rendered
    #                       slide PNG. When set, the planner asks a vision
    #                       LLM to identify the slide's main concept by
    #                       *looking* at it — far more robust across
    #                       arbitrary slide layouts than text heuristics.
    # main_concept_hint   : the concept name for the current slide
    #                       (e.g. "Trimmed mean", "Median"). Resolved
    #                       primarily by the vision call; falls back to a
    #                       text-structural heuristic when no image is
    #                       available or vision providers are down.
    # previous_concept    : the concept narrated on the prior slide.
    # previous_narration_summary
    #                     : 1-2 sentences summarising what was just taught,
    #                       so the narrator can open with a bridge instead
    #                       of restarting cold on every slide.
    # previous_slide_content
    #                     : raw text of the prior slide. Used to detect
    #                       near-duplicate consecutive slides ("build"
    #                       decks) so the narrator points out only the
    #                       diff instead of repeating the full explanation.
    # narration_summary   : produced by the narrator for the *next* slide
    #                       to consume — closes the continuity loop.
    slide_image_path: str
    main_concept_hint: str
    previous_concept: str
    previous_narration_summary: str
    previous_slide_content: str
    narration_summary: str

    # ── Personalization (resolved by services/presentation.py) ──────
    learning_style_hint: str           # prose hint for the narrator

    # ── Inputs (déclencheur du graph) ───────────────────────────────
    event_type: EventType
    event_payload: dict[str, Any]

    # ── Outputs remplis par les agents ──────────────────────────────
    plan: PresentationPlan | None  # Teaching Graph
    retrieved_chunks: list[dict[str, Any]]  # Q&A Graph
    intent: VoiceIntent | None  # Q&A Graph
    confusion: ConfusionResult | None  # Q&A Graph
    rewritten_query: str  # Q&A Graph: query after pronoun resolution
    anchored_concept: str  # Q&A Graph: main topic identified by step-back rewriter
                           # (Zheng et al. 2023, arXiv:2310.06117). Available
                           # downstream for query expansion / retrieval anchoring.
    answer: str | None  # narration text or Q&A response
    actions: list[Action]  # decisions taken
    confidence: float
    citations: list[dict[str, Any]]  # Q&A Graph: chunks the responder grounded the
                                     # answer on. Each entry: {chunk_id, idea_label,
                                     # source, score}. Empty list = ungrounded answer
                                     # (LLM didn't cite any retrieved chunk).
    review: ReviewResult | None
    review_mode: bool  # NEW (Option C) : true si l'eleve maitrise deja bcp d'ideas → mode revision
    is_transition: bool  # True for header/divider slides (very thin body or
                         # main_concept ≈ section_title). Triggers a short
                         # bridge narration instead of a full explanation.

    # ── Diagnostics ─────────────────────────────────────────────────
    errors: list[str]
    timings: dict[str, float]  # node_name -> duration_s
    narrator_retries: int  # incremented when reviewer asks for a re-narration
    responder_retries: int  # Q&A graph counterpart: bumped when the Q&A
                            # reviewer asks the responder to try again with
                            # its grounding/coherence feedback. Kept separate
                            # from narrator_retries so a state object that
                            # transits both graphs doesn't crosstalk.
