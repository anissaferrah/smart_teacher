"""Smart Teacher — Agentic schemas (Plan, Idea, Action, Intent, Confusion).

Used as inter-agent contracts inside the LangGraph pipelines (Teaching, Q&A,
Assessment). Lightweight dataclasses, not Pydantic — keeps things fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


IdeaType = Literal["intro", "concept", "example", "summary"]
IdeaDepth = Literal["shallow", "normal", "deep"]


@dataclass
class Idea:
    """A single pedagogical unit inside a presentation plan."""

    id: str  # short ASCII id, e.g. "intro", "concept_backprop", "example_xor"
    type: IdeaType
    content_brief: str  # one-line hint of what the professor should say
    depth: IdeaDepth = "normal"


@dataclass
class PresentationPlan:
    """Output of the PlannerAgent — sequence of Ideas to narrate."""

    section_title: str
    chapter_title: str
    ideas: list[Idea] = field(default_factory=list)
    estimated_duration_s: int = 60

    def to_brief(self) -> str:
        """Render the plan as a compact text brief for the NarratorAgent."""
        lines = [
            f"Plan: chapter='{self.chapter_title}' section='{self.section_title}'",
            "Ideas to present (in order):",
        ]
        for i, idea in enumerate(self.ideas, 1):
            lines.append(
                f"  {i}. [{idea.type}/{idea.depth}] ({idea.id}) {idea.content_brief}"
            )
        return "\n".join(lines)


# ── Q&A graph (placeholders for later sprints) ─────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
#  CONVENTIONS DE NAMING POUR LA CONFUSION (a respecter cross-modules)
# ─────────────────────────────────────────────────────────────────────────
#  Niveau semantique     │ Nom canonique           │ Type        │ Ou
#  ──────────────────────┼─────────────────────────┼─────────────┼──────────
#  Intent (Q&A graph)    │ "confusion_signal"      │ str literal │ VoiceIntent
#  Modele (data class)   │ ConfusionResult.confused│ bool field  │ ce module
#  Variable locale       │ is_confused             │ bool var    │ python code
#  Event type            │ "confusion_detected"    │ str event   │ EventBus
#  Dashboard JSON key    │ "confusion"             │ dict key    │ API response
#
#  Regle : pour CHECK utiliser is_confused = bool(result.confused)
# ─────────────────────────────────────────────────────────────────────────

VoiceIntentType = Literal["question", "navigation", "feedback", "confusion_signal"]


@dataclass
class VoiceIntent:
    """Classified student utterance.

    ``needs_retrieval`` follows Self-RAG (Asai et al. 2023, "Self-RAG:
    Learning to Retrieve, Generate, and Critique through Self-Reflection",
    arXiv:2310.11511): the agent decides at runtime whether a question
    needs the document store at all. Meta-questions ("can you repeat?",
    "what did you just say?") and pure conversation ("thanks") don't.
    Skipping retrieval on those queries cuts ~1-2s of latency without
    losing grounding (the answer doesn't need to be grounded in source
    material because the question isn't about source material).

    Default ``True`` is the safe fallback: when in doubt, retrieve.
    """

    type: VoiceIntentType
    confidence: float
    payload: dict[str, Any] = field(default_factory=dict)
    needs_retrieval: bool = True


ConfusionType = Literal["lexical", "conceptuel", "contextuel", "none"]


@dataclass
class ConfusionResult:
    """Resultat de detection de confusion (cross-modules canonical type).

    Attributes:
        confused : bool authoritative (utilise is_confused = bool(result.confused) en local)
        score    : probability [0..1] depuis SIGHT ou LLM ou keyword
        type     : nature de la confusion si identifiee
        trigger_concept : concept_id qui a déclenche, si determinable
        source   : "model" (SIGHT) | "keyword" (regex) | "prosody" | "rule" | "llm"
    """
    confused: bool
    score: float
    type: ConfusionType = "none"
    trigger_concept: str | None = None
    source: str = "model"  # "model", "keyword", "prosody", "rule", "llm"

    @property
    def detected(self) -> bool:
        """Alias public pour `confused` (lecture user-facing). API compatible."""
        return bool(self.confused)


@dataclass
class ConfusionEvent:
    timestamp: float
    concept_id: str
    score: float
    resolved: bool = False


# ── Action types (used by Pedagogy Policy + Q&A response) ──────────────────

ActionType = Literal[
    "continue",
    "replay_concept",
    "jump_to",
    "clarify",
    "answer",
    "quiz",
    "slow_down",
]


@dataclass
class Action:
    type: ActionType
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewResult:
    grounded: bool
    score: float
    feedback: str = ""
