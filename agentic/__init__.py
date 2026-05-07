"""Smart Teacher — Agentic core (LangGraph-based Tutor OS).

Top-level exports for convenient imports from main.py:
    from agentic import build_teaching_graph, TutorState, PresentationPlan
"""
from agentic.schemas import (
    Action,
    ConfusionEvent,
    ConfusionResult,
    Idea,
    PresentationPlan,
    ReviewResult,
    VoiceIntent,
)
from agentic.qa import build_qa_graph
from agentic.state import TutorState
from agentic.teaching import build_teaching_graph

__all__ = [
    # graph builders
    "build_teaching_graph",
    "build_qa_graph",
    # state + schemas
    "TutorState",
    "PresentationPlan",
    "Idea",
    "Action",
    "VoiceIntent",
    "ConfusionResult",
    "ConfusionEvent",
    "ReviewResult",
]
