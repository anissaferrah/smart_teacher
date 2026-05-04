"""Unit tests for the IdeaGraph layer.

Coverage:
  - IdeaNode construction from Document metadata (with/without graph fields)
  - Forward + reverse edge indexing
  - prerequisites_of (direct + transitive)
  - dependents_of (direct + transitive)
  - examples_of
  - learning_path (topological order + mastered filtering + cycle safety)
  - Graceful handling of missing / malformed metadata
"""
import pytest

from pedagogy.knowledge_graph import IdeaGraph, IdeaNode


class _FakeDoc:
    """Minimal Document-like object for tests (no langchain dependency)."""
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def _make_doc(idea_id: str, label: str = "definition", text: str = "",
              depends: list[str] | None = None,
              illustrates: str | None = None,
              course: str = "course-a", chapter: int = 0, section: int = 0) -> _FakeDoc:
    return _FakeDoc(
        page_content=text or f"Content of {idea_id}",
        metadata={
            "idea_id":        idea_id,
            "idea_label":     label,
            "course":         course,
            "chapter_idx":    chapter,
            "section_idx":    section,
            "depends_on_ids": depends or [],
            "illustrates_id": illustrates,
        },
    )


# ════════════════════════════════════════════════════════════════════
# IdeaNode.from_document
# ════════════════════════════════════════════════════════════════════

class TestIdeaNodeFromDocument:
    def test_basic_extraction(self):
        doc = _make_doc("abc123", label="theorem", depends=["lemma1"], illustrates=None)
        node = IdeaNode.from_document(doc)
        assert node is not None
        assert node.idea_id == "abc123"
        assert node.label == "theorem"
        assert node.depends_on_ids == {"lemma1"}
        assert node.illustrates_id is None

    def test_missing_idea_id_returns_none(self):
        doc = _FakeDoc("text", {"idea_label": "definition"})
        assert IdeaNode.from_document(doc) is None

    def test_legacy_doc_without_graph_fields(self):
        # Pre-graph documents still work — depends/illustrates default to empty
        doc = _FakeDoc("text", {"idea_id": "old123", "idea_label": "note"})
        node = IdeaNode.from_document(doc)
        assert node is not None
        assert node.depends_on_ids == set()
        assert node.illustrates_id is None

    def test_malformed_depends_on_ignored(self):
        doc = _FakeDoc("t", {"idea_id": "x", "depends_on_ids": "not_a_list"})
        node = IdeaNode.from_document(doc)
        assert node is not None
        assert node.depends_on_ids == set()


# ════════════════════════════════════════════════════════════════════
# Graph build & basic queries
# ════════════════════════════════════════════════════════════════════

class TestGraphBuild:
    def test_add_and_get(self):
        g = IdeaGraph()
        g.add_node(IdeaNode(idea_id="A", label="definition", text="..."))
        assert "A" in g
        assert g.get("A").label == "definition"
        assert len(g) == 1

    def test_add_documents_skips_malformed(self):
        g = IdeaGraph()
        n = g.add_documents([
            _make_doc("A"),
            _FakeDoc("text", {}),                # missing idea_id
            _make_doc("B"),
        ])
        assert n == 2
        assert "A" in g and "B" in g

    def test_reverse_index_built(self):
        g = IdeaGraph()
        g.add_documents([
            _make_doc("def_grad"),
            _make_doc("thm_back", depends=["def_grad"]),
            _make_doc("ex_back",  depends=["thm_back"], illustrates="thm_back"),
        ])
        # def_grad has 1 dependent (thm_back)
        deps = g.dependents_of("def_grad")
        assert {n.idea_id for n in deps} == {"thm_back"}
        # thm_back has 1 example (ex_back)
        exs = g.examples_of("thm_back")
        assert {n.idea_id for n in exs} == {"ex_back"}


# ════════════════════════════════════════════════════════════════════
# Direct vs transitive walks
# ════════════════════════════════════════════════════════════════════

class TestTransitiveWalks:
    @pytest.fixture
    def linear_chain(self):
        """A → B → C → D (D depends_on C, C depends_on B, B depends_on A)."""
        g = IdeaGraph()
        g.add_documents([
            _make_doc("A"),
            _make_doc("B", depends=["A"]),
            _make_doc("C", depends=["B"]),
            _make_doc("D", depends=["C"]),
        ])
        return g

    def test_direct_prerequisites(self, linear_chain):
        prereqs = linear_chain.prerequisites_of("D")
        assert {n.idea_id for n in prereqs} == {"C"}

    def test_transitive_prerequisites(self, linear_chain):
        prereqs = linear_chain.prerequisites_of("D", transitive=True)
        assert {n.idea_id for n in prereqs} == {"A", "B", "C"}

    def test_transitive_dependents(self, linear_chain):
        deps = linear_chain.dependents_of("A", transitive=True)
        assert {n.idea_id for n in deps} == {"B", "C", "D"}

    def test_walk_handles_cycles(self):
        # Pathological: LLM extracted A→B→A (cycle). Must not infinite-loop.
        g = IdeaGraph()
        g.add_documents([
            _make_doc("A", depends=["B"]),
            _make_doc("B", depends=["A"]),
        ])
        prereqs = g.prerequisites_of("A", transitive=True)
        # Only B is returned (A itself excluded by visited set)
        assert {n.idea_id for n in prereqs} == {"B"}


# ════════════════════════════════════════════════════════════════════
# Learning path
# ════════════════════════════════════════════════════════════════════

class TestLearningPath:
    @pytest.fixture
    def diamond(self):
        r"""    A
              / \
             B   C
              \ /
               D
        """
        g = IdeaGraph()
        g.add_documents([
            _make_doc("A"),
            _make_doc("B", depends=["A"]),
            _make_doc("C", depends=["A"]),
            _make_doc("D", depends=["B", "C"]),
        ])
        return g

    def test_path_to_leaf_with_nothing_mastered(self, diamond):
        path = diamond.learning_path("D", mastered=set())
        ids = [n.idea_id for n in path]
        # A must come before B and C ; B and C before D
        assert ids[-1] == "D"
        assert ids.index("A") < ids.index("B")
        assert ids.index("A") < ids.index("C")
        assert ids.index("B") < ids.index("D")
        assert ids.index("C") < ids.index("D")

    def test_path_skips_mastered_prereqs(self, diamond):
        path = diamond.learning_path("D", mastered={"A", "B"})
        ids = [n.idea_id for n in path]
        # A and B already mastered → skipped
        assert ids == ["C", "D"]

    def test_path_when_target_already_mastered(self, diamond):
        path = diamond.learning_path("D", mastered={"A", "B", "C", "D"})
        assert path == []

    def test_unknown_target_returns_empty(self, diamond):
        assert diamond.learning_path("Z", mastered=set()) == []


# ════════════════════════════════════════════════════════════════════
# Stats
# ════════════════════════════════════════════════════════════════════

class TestBuilder:
    """The lazy singleton builder rebuilds when RAG document count changes."""

    def setup_method(self):
        from pedagogy.knowledge_graph import reset
        reset()

    def test_build_from_rag_extracts_all_docs(self):
        from pedagogy.knowledge_graph import build_from_rag

        class FakeRAG:
            all_docs = [
                _make_doc("A"),
                _make_doc("B", depends=["A"]),
            ]
        g = build_from_rag(FakeRAG())
        assert len(g) == 2
        assert "A" in g and "B" in g

    def test_get_or_build_caches(self):
        from pedagogy.knowledge_graph import get_or_build

        class FakeRAG:
            all_docs = [_make_doc("X")]
        rag = FakeRAG()
        g1 = get_or_build(rag)
        g2 = get_or_build(rag)
        assert g1 is g2  # same instance

    def test_get_or_build_rebuilds_when_corpus_grows(self):
        from pedagogy.knowledge_graph import get_or_build

        class FakeRAG:
            all_docs = [_make_doc("X")]
        rag = FakeRAG()
        g1 = get_or_build(rag)
        rag.all_docs.append(_make_doc("Y"))
        g2 = get_or_build(rag)
        assert g1 is not g2  # rebuilt
        assert len(g2) == 2

    def test_get_or_build_with_none_rag(self):
        from pedagogy.knowledge_graph import get_or_build
        g = get_or_build(None)
        assert len(g) == 0  # empty graph, no crash


class TestStats:
    def test_stats_on_diamond(self):
        g = IdeaGraph()
        g.add_documents([
            _make_doc("A"),
            _make_doc("B", depends=["A"]),
            _make_doc("C", depends=["A"]),
            _make_doc("D", depends=["B", "C"]),
            _make_doc("ex_D", depends=["D"], illustrates="D"),
        ])
        s = g.stats()
        assert s["nodes"] == 5
        assert s["depends_edges"] == 5     # B→A, C→A, D→B, D→C, ex_D→D
        assert s["illustrates_edges"] == 1
        assert s["roots"] == 1             # only A has no prereq
        assert s["leaves"] == 1            # only ex_D has no dependents
        assert s["courses"] == 1
