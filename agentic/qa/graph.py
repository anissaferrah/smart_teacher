"""Q&A Graph with Self-RAG router + self-correction (Asai et al. 2023).

    START → IntentAgent → [intent_router]
                            │
                            ├── question → [retrieval_router]
                            │                  │
                            │                  ├── retrieve → QueryRewriter → Retriever → Responder → Reviewer → router
                            │                  │                                                          ↑       │
                            │                  │                                                          │       ├── grounded            → END
                            │                  │                                                          │       ├── ungrounded + retries<max → retry → Responder
                            │                  │                                                          │       └── ungrounded + retries≥max → Fallback → END
                            │                  │                                                          └─ retry ─┘
                            │                  └── skip     → Responder → Reviewer → router (same logic)
                            │
                            └── short    → Responder → END  (navigation / feedback / confusion_signal)

Three routing decisions, all driven by LLM signals:
  1. ``intent_router``      on intent.type            → full QA vs short circuit
  2. ``retrieval_router``   on intent.needs_retrieval → retrieve vs skip retrieval (Self-RAG)
  3. ``qa_review_router``   on review.grounded        → end vs retry vs fallback (self-correction)

Skipping retrieval on meta-questions ("repeat that", "what did you just
say?") removes one rewriter call and one vector search per turn,
cutting ~1-2s of latency on conversational follow-ups.

The reviewer + retry layer is the symmetric Q&A counterpart of the
teaching graph's narrator + reviewer + fallback. It catches three
failure modes the responder can hit even with the strict course-bound
prompt :
  - ungrounded claim (the LLM made something up despite the rule),
  - encyclopedic leak (Wikipedia-style content slipping in),
  - off-topic answer (correct text but doesn't address the question).

The "short" intent path (navigation / feedback) skips the reviewer
because the responder produces a static reply there — there's nothing
for an LLM to verify.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agentic.qa.intent import IntentAgent, intent_router
from agentic.qa.responder import ResponderAgent
from agentic.qa.retriever import RetrieverAgent
from agentic.qa.reviewer import QAReviewAgent, qa_review_router, qa_fallback_node
from agentic.qa.rewriter import QueryRewriterAgent
from agentic.resilience import build_resilient_node
from agentic.state import TutorState

log = logging.getLogger("agentic.qa.graph")


def retrieval_router(state: TutorState) -> str:
    """Self-RAG decision (Asai et al. 2023, arXiv:2310.11511).

    Reads ``intent.needs_retrieval`` populated by the IntentAgent's LLM.
    The flag defaults to ``True`` when the LLM output is missing or
    ambiguous — retrieving when in doubt is the safer failure mode (a
    grounded answer with unused context is better than an ungrounded
    answer on a question that did need context).
    """
    intent = state.get("intent")
    if intent is not None and getattr(intent, "needs_retrieval", True):
        return "retrieve"
    return "skip"


def post_responder_router(state: TutorState) -> str:
    """Decide whether the responder's output needs review.

    The "short" intent path (navigation, feedback, confusion_signal)
    produces a static, deterministic reply — there's nothing for an LLM
    reviewer to verify, and routing through it would just waste latency.
    Only the "question" intent path emits an LLM-generated answer, so
    only that path goes through the review.
    """
    intent = state.get("intent")
    if intent is None:
        # Defensive: if the intent classification is missing, treat the
        # output as a real answer that warrants review.
        return "review"
    intent_type = getattr(intent, "type", "question")
    if intent_type in {"navigation", "feedback"}:
        return "skip_review"
    return "review"


def build_qa_graph(brain, rag=None):
    """Compile the Q&A graph.

    Parameters
    ----------
    brain : ai.llm.Brain
        Active LLM brain (used by intent, rewriter, responder, reviewer).
    rag : rag.multimodal_rag.MultiModalRAG | None
        RAG instance for the Retriever node. If None, retriever returns no chunks
        and Responder answers from slide context only.
    """
    intent = IntentAgent(brain)
    rewriter = QueryRewriterAgent(brain)
    retriever = RetrieverAgent(rag)
    responder = ResponderAgent(brain)
    reviewer = QAReviewAgent(brain)

    graph = StateGraph(TutorState)
    graph.add_node("intent", build_resilient_node(intent, "intent"))
    graph.add_node("rewriter", build_resilient_node(rewriter, "rewriter"))
    graph.add_node("retriever", build_resilient_node(retriever, "retriever"))
    graph.add_node("responder", build_resilient_node(responder, "responder"))
    graph.add_node("qa_review", build_resilient_node(reviewer, "qa_review"))
    # Fallback is a deterministic function (no LLM call), so we still wrap
    # it through build_resilient_node for consistency — the wrapper is a
    # cheap pass-through when the inner node doesn't raise.
    graph.add_node("qa_fallback", qa_fallback_node)

    graph.add_edge(START, "intent")
    # First branch: intent type — full QA pipeline vs short-circuit responder.
    graph.add_conditional_edges(
        "intent",
        intent_router,
        {
            "question": "retrieval_decision",
            "short": "responder",
        },
    )
    # Pseudo-node: Self-RAG decision. We model it as a no-op pass-through
    # whose only job is to host the conditional edge. LangGraph requires
    # a node to attach conditional edges to.
    graph.add_node("retrieval_decision", lambda state: {})
    graph.add_conditional_edges(
        "retrieval_decision",
        retrieval_router,
        {
            "retrieve": "rewriter",
            "skip": "responder",
        },
    )
    graph.add_edge("rewriter", "retriever")
    graph.add_edge("retriever", "responder")

    # After the responder, decide whether to review (LLM-generated answer)
    # or skip review (static navigation/feedback reply).
    graph.add_conditional_edges(
        "responder",
        post_responder_router,
        {
            "review": "qa_review",
            "skip_review": END,
        },
    )

    # After the reviewer, three outcomes per ``qa_review_router``:
    #   grounded → END
    #   ungrounded + retries left → back to responder (with feedback in state)
    #   ungrounded + no retries → fallback safe answer → END
    graph.add_conditional_edges(
        "qa_review",
        qa_review_router,
        {
            "end": END,
            "retry": "responder",
            "fallback": "qa_fallback",
        },
    )
    graph.add_edge("qa_fallback", END)

    compiled = graph.compile()
    log.info(
        "Q&A graph compiled (7 nodes: intent → [retrieval_decision → "
        "rewriter → retriever →] responder → qa_review ⇄ {retry, "
        "qa_fallback}) — Self-RAG + self-correction active"
    )
    return compiled
