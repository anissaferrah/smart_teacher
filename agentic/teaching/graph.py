"""Teaching Graph — full pipeline with self-reflection fallback.

    Planner → Context → Adaptation → Narrator → Review → router
                                          ↑       │
                                          │       ├── grounded               → END
                                          │       ├── ungrounded + retries<max → retry → Narrator
                                          │       └── ungrounded + retries≥max → Fallback → END
                                          └─ retry ─┘

- Planner          : LLM, decomposes the slide into 3-5 ideas
- Context          : RAG, retrieves grounding chunks (no LLM)
- Adaptation       : profile lookup, adjusts depth per idea (no LLM)
- Narrator         : LLM, builds the spoken narration using plan + context
- Review           : LLM (light), checks groundedness; binary verdict
- FallbackNarrator : LLM (strict, slide-only) when retries are exhausted.
                     Grounded by construction — the prompt receives ONLY
                     the slide so the model has no other source to draw
                     from. Failure to land here is the system's "I don't
                     know" signal, not a silent accept.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agentic.resilience import build_resilient_node
from agentic.state import TutorState
from agentic.teaching.adaptation import AdaptationAgent
from agentic.teaching.context import ContextAgent
from agentic.teaching.fallback import FallbackNarratorAgent
from agentic.teaching.narrator import NarratorAgent
from agentic.teaching.planner import PlannerAgent
from agentic.teaching.reviewer import ReviewAgent, review_router

log = logging.getLogger("agentic.teaching.graph")


def build_teaching_graph(brain, rag=None, profile_mgr=None):
    """Compile the Teaching Graph.

    Parameters
    ----------
    brain : ai.llm.Brain
        LLM brain (used by Planner, Narrator, Reviewer).
    rag : rag.multimodal_rag.MultiModalRAG | None
        Optional RAG. If None, the Context node is skipped (chunks=[]).
    profile_mgr : pedagogy.personalization.student_profile.ProfileManager | None
        Optional profile manager for the AdaptationAgent.
    """
    planner = PlannerAgent(brain)
    narrator = NarratorAgent(brain)
    reviewer = ReviewAgent(brain)
    fallback = FallbackNarratorAgent(brain)
    context = ContextAgent(rag) if rag is not None else None
    adapter = AdaptationAgent(profile_mgr=profile_mgr)

    graph = StateGraph(TutorState)
    graph.add_node("planner", build_resilient_node(planner, "planner"))
    if context is not None:
        graph.add_node("context", build_resilient_node(context, "context"))
    graph.add_node("adaptation", build_resilient_node(adapter, "adaptation"))
    graph.add_node("narrator", build_resilient_node(narrator, "narrator"))
    graph.add_node("review", build_resilient_node(reviewer, "review"))
    graph.add_node("fallback", build_resilient_node(fallback, "fallback"))

    graph.add_edge(START, "planner")
    if context is not None:
        graph.add_edge("planner", "context")
        graph.add_edge("context", "adaptation")
    else:
        graph.add_edge("planner", "adaptation")
    graph.add_edge("adaptation", "narrator")
    graph.add_edge("narrator", "review")
    graph.add_conditional_edges(
        "review",
        review_router,
        {
            "retry": "narrator",
            "fallback": "fallback",
            "end": END,
        },
    )
    graph.add_edge("fallback", END)

    compiled = graph.compile()
    nodes_count = 6 if context is not None else 5
    log.info(
        "teaching graph compiled (%d nodes: planner→%scontext→adaptation→narrator→review→[fallback])",
        nodes_count,
        "" if context is not None else "[no-context]→",
    )
    return compiled
