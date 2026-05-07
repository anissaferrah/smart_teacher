"""Knowledge Graph layer over indexed ideas.

Each idea (RAG chunk) declares its intra-document relations via metadata:
  - depends_on_ids : list[str]    pre-requisites
  - illustrates_id : str | None   concept illustrated by an example

This package consumes those edges to expose graph queries:
  - prerequisites_of(idea_id)
  - dependents_of(idea_id)
  - examples_of(idea_id)
  - learning_path(target_id, mastered_set)

The graph is currently INTRA-CHUNK only (LLM extracts relations within
each title-chunk). Cross-chunk edges are a TODO that needs a 2-pass
post-ingestion LLM step.
"""
from pedagogy.knowledge_graph.graph import (
    KnowledgeGraph,
    IdeaGraph,        # deprecated alias for KnowledgeGraph (Stage 1 retro-compat)
    IdeaNode,
    ConceptInfo,
)
from pedagogy.knowledge_graph.builder import build_from_rag, get_or_build, reset
from pedagogy.knowledge_graph.concepts_loader import ensure_concepts_loaded
from pedagogy.knowledge_graph import persistence

__all__ = [
    "KnowledgeGraph",
    "IdeaGraph",       # deprecated alias
    "IdeaNode",
    "ConceptInfo",
    "build_from_rag",
    "get_or_build",
    "reset",
    "ensure_concepts_loaded",
    "persistence",
]
