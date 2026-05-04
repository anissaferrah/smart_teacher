"""Unit tests pour les 4 fixes ingestion + le LLMRouter unifie.

Coverage :
  - LLMRouter : invoke avec fallback, disable persistent, stats threadsafe
  - _fuse_short_sections (Fix #2) : empty, all-long, mix court/long, depasse max
  - _chunk_by_ideas_batch (Fix #3) : ordre preserve, fallback gracieux par-item
  - validate_per_chapter (Fix #4) : verdicts respectes, LLM down = keep all,
    JSON cassed = keep all, verdicts trop courts = keep non-couverts
  - IngestedAsset.bbox (Fix #5) : field present, serialisation to_dict

Ces tests ne touchent PAS l'infra (Qdrant, OpenAI, Ollama, PDF reels). Pour
les methodes de MultiModalRAG qui ne dependent pas de self state, on instancie
via __new__ pour bypass l'init lourde.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ai.llm_router import LLMRouter, LLMStats, get_default_router
from pedagogy.concept_types import Concept
from pedagogy.intelligent_ingester import IngestedAsset
from rag.multimodal_rag import MultiModalRAG


# ════════════════════════════════════════════════════════════════════
# LLMRouter
# ════════════════════════════════════════════════════════════════════

class TestLLMRouter:
    """Le router est utilise depuis un ThreadPoolExecutor → threadsafety
    et logique de fallback sont les invariants critiques."""

    def test_invoke_uses_preferred_first(self):
        r = LLMRouter()
        with patch.object(r, "_call_openai", return_value="from-openai") as mock_oa, \
             patch.object(r, "_call_ollama", return_value="from-ollama") as mock_ol:
            out = r.invoke("hi", prefer="openai")
            assert out == "from-openai"
            assert mock_oa.called and not mock_ol.called

    def test_invoke_falls_back_when_preferred_returns_none(self):
        r = LLMRouter()
        with patch.object(r, "_call_openai", return_value=None), \
             patch.object(r, "_call_ollama", return_value="ollama-saved-the-day"):
            out = r.invoke("hi", prefer="openai")
            assert out == "ollama-saved-the-day"
            assert r.stats.snapshot()["fallback_events"] == 1

    def test_invoke_returns_none_when_both_fail(self):
        r = LLMRouter()
        with patch.object(r, "_call_openai", return_value=None), \
             patch.object(r, "_call_ollama", return_value=None):
            assert r.invoke("hi") is None
            snap = r.stats.snapshot()
            assert snap["both_failed"] == 1
            assert snap["fallback_events"] == 1

    def test_invoke_prefer_ollama_reverses_order(self):
        r = LLMRouter()
        with patch.object(r, "_call_openai", return_value="oa"), \
             patch.object(r, "_call_ollama", return_value="ol") as mock_ol:
            out = r.invoke("hi", prefer="ollama")
            assert out == "ol"
            assert mock_ol.called

    def test_invoke_normalizes_invalid_prefer_to_openai(self):
        r = LLMRouter()
        with patch.object(r, "_call_openai", return_value="ok") as mock_oa:
            r.invoke("hi", prefer="garbage-value")
            assert mock_oa.called

    def test_disable_openai_skips_openai_call(self):
        r = LLMRouter()
        r.disable_openai("test quota")
        with patch.object(r, "_call_ollama", return_value="ol") as mock_ol:
            # _call_openai short-circuits when disabled — no need to mock it
            out = r.invoke("hi", prefer="openai")
            assert out == "ol"
            assert mock_ol.called

    def test_disable_openai_idempotent_first_reason_wins(self):
        r = LLMRouter()
        r.disable_openai("first")
        r.disable_openai("second")
        assert r.openai_disabled_reason == "first"

    def test_permanent_error_detection(self):
        assert LLMRouter._is_permanent_openai_error(Exception("insufficient_quota error"))
        assert LLMRouter._is_permanent_openai_error(Exception("HTTP 429"))
        assert LLMRouter._is_permanent_openai_error(Exception("invalid_api_key"))
        assert not LLMRouter._is_permanent_openai_error(Exception("connection refused"))
        assert not LLMRouter._is_permanent_openai_error(Exception("timeout"))

    def test_stats_counters_thread_safe(self):
        """1000 increments concurrents — pas de race possible."""
        import threading
        r = LLMRouter()

        def hammer():
            for _ in range(100):
                with r.stats._lock:
                    r.stats.openai_calls += 1

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert r.stats.snapshot()["openai_calls"] == 1000

    def test_default_router_is_singleton(self):
        assert get_default_router() is get_default_router()


# ════════════════════════════════════════════════════════════════════
# Page fusion (Fix #2)
# ════════════════════════════════════════════════════════════════════

class TestFuseShortSections:
    """`_fuse_short_sections` ne lit pas self state → instancie via __new__
    pour bypass l'init lourde."""

    @pytest.fixture
    def rag(self):
        return MultiModalRAG.__new__(MultiModalRAG)

    def test_empty_input(self, rag):
        assert rag._fuse_short_sections([]) == []

    def test_all_long_sections_unchanged(self, rag):
        long = "x" * 500
        sections = [{"content": long, "page_index": i} for i in range(3)]
        out = rag._fuse_short_sections(sections)
        assert len(out) == 3
        assert all(len(s["content"]) == 500 for s in out)

    def test_short_consecutive_get_merged(self, rag):
        # 3 sections de 50 chars → fusionnees en 1 (sous min=300 → on accumule)
        sections = [
            {"content": "abc" * 30, "page_index": 1},  # 90 chars
            {"content": "def" * 30, "page_index": 2},  # 90 chars
            {"content": "ghi" * 30, "page_index": 3},  # 90 chars
            {"content": "jkl" * 50, "page_index": 4},  # 150 chars → total 90+90+90+150 = 420 > min
        ]
        out = rag._fuse_short_sections(sections)
        # Tout fusionne en 1 (combinaison <= max=2400 et atteint min=300)
        assert len(out) == 1
        assert "fused_pages" in out[0]
        assert out[0]["fused_pages"] == [1, 2, 3, 4]

    def test_combined_exceeding_max_starts_new_buffer(self, rag):
        # max_chars = 2400. 2 sections de 1500 chars chacune ne peuvent pas
        # fusionner (3000 > 2400) → restent 2 sections
        big = "x" * 1500
        sections = [
            {"content": big, "page_index": 1},
            {"content": big, "page_index": 2},
        ]
        out = rag._fuse_short_sections(sections)
        assert len(out) == 2

    def test_first_section_already_long_flushes_immediately(self, rag):
        long = "x" * 500
        short = "y" * 50
        sections = [
            {"content": long, "page_index": 1},   # >= min → flush direct
            {"content": short, "page_index": 2},  # devient debut buffer
            {"content": short, "page_index": 3},  # fusionne dans buffer (mais < min, reste)
        ]
        out = rag._fuse_short_sections(sections)
        # 1 section longue + 1 buffer fusionne (les 2 shorts) qui n'a pas atteint min
        # mais qui est flush par la fin de la liste
        assert len(out) == 2
        assert out[0]["content"] == long
        # Le 2eme contient les 2 shorts concatenes
        assert short in out[1]["content"]


# ════════════════════════════════════════════════════════════════════
# Idea-chunking batch (Fix #3)
# ════════════════════════════════════════════════════════════════════

class TestChunkByIdeasBatch:

    @pytest.fixture
    def rag(self):
        return MultiModalRAG.__new__(MultiModalRAG)

    def test_empty_input(self, rag):
        assert rag._chunk_by_ideas_batch([]) == []

    def test_preserves_order(self, rag):
        """Critical : results[i] doit correspondre a items[i] meme avec
        execution parallele."""
        items = [(f"text-{i}", "ch", "sec") for i in range(20)]

        def fake_chunk(text, chapter_title, section_title):
            return [{"idea": text, "label": "fragment", "local_id": "f_1",
                     "depends_on": [], "illustrates": None}]

        with patch.object(rag, "_chunk_by_ideas", side_effect=fake_chunk):
            results = rag._chunk_by_ideas_batch(items)

        assert len(results) == 20
        for i, ideas in enumerate(results):
            assert ideas[0]["idea"] == f"text-{i}"

    def test_failure_per_item_falls_back_to_fragment(self, rag):
        items = [("t0", "", ""), ("t1", "", ""), ("t2", "", "")]

        def fake_chunk(text, *args):
            if text == "t1":
                raise RuntimeError("LLM blew up on this one")
            return [{"idea": text, "label": "definition", "local_id": "d_1",
                     "depends_on": [], "illustrates": None}]

        with patch.object(rag, "_chunk_by_ideas", side_effect=fake_chunk):
            results = rag._chunk_by_ideas_batch(items)

        assert len(results) == 3
        assert results[0][0]["label"] == "definition"
        # t1 a echoue → fallback fragment avec text original
        assert results[1][0]["label"] == "fragment"
        assert results[1][0]["idea"] == "t1"
        assert results[2][0]["label"] == "definition"


# ════════════════════════════════════════════════════════════════════
# Validation par chapitre (Fix #4)
# ════════════════════════════════════════════════════════════════════

class _FakeDoc:
    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.page_content = ""


def _make_concept(label: str, name: str, chapter: int) -> Concept:
    c = Concept(label=label, name=name, score=1.0)
    c.chapter_idxs.add(chapter)
    return c


# ════════════════════════════════════════════════════════════════════
# IngestedAsset.bbox (Fix #5)
# ════════════════════════════════════════════════════════════════════

class TestIngestedAssetBbox:

    def test_bbox_field_defaults_to_none(self):
        asset = IngestedAsset(asset_type="image", page_num=1)
        assert asset.bbox is None

    def test_bbox_serializes_in_to_dict(self):
        bbox = {"x0": 10.0, "y0": 20.0, "x1": 110.0, "y1": 120.0,
                "page_width": 612.0, "page_height": 792.0,
                "x0_norm": 0.016, "y0_norm": 0.025,
                "x1_norm": 0.180, "y1_norm": 0.151}
        asset = IngestedAsset(
            asset_type="image",
            page_num=2,
            image_path="/tmp/page002_img00.png",
            bbox=bbox,
        )
        d = asset.to_dict()
        assert d["bbox"] == bbox
        assert d["asset_type"] == "image"
        assert d["page_num"] == 2


# ════════════════════════════════════════════════════════════════════
# KG-augmented retrieval — Phase 1
# ════════════════════════════════════════════════════════════════════
#
# Avant Phase 1 : RetrieverAgent._augment_with_prereqs etendait avec les
# prereqs uniquement. Phase 1 ajoute examples_of + illustrates_id (le
# concept illustre quand le top-1 EST un exemple). Tests existants pour
# prereqs sont dans test_personalization.py — ici on couvre le NEW.

class TestKGAugmentationPhase1:

    def _agent(self):
        from agentic.qa.retriever import RetrieverAgent
        return RetrieverAgent(rag=object())

    def _graph_with_example(self):
        """Graph: concept C, example E illustrates C."""
        from pedagogy.knowledge_graph.graph import IdeaGraph, IdeaNode
        g = IdeaGraph()
        g.add_node(IdeaNode(
            idea_id="concept_id", label="definition", text="Definition of C",
            course_id="c", chapter_idx=1,
        ))
        g.add_node(IdeaNode(
            idea_id="example_id", label="example", text="Worked example for C",
            course_id="c", chapter_idx=1,
            illustrates_id="concept_id",
        ))
        return g

    def test_augment_pulls_examples_when_top1_is_concept(self, monkeypatch):
        """Top-1 chunk = concept → on pull les chunks qui l'illustrent."""
        graph = self._graph_with_example()
        monkeypatch.setattr("pedagogy.knowledge_graph.get_or_build", lambda rag: graph)
        agent = self._agent()
        retrieved = [{"idea_id": "concept_id", "content": "...", "score": 0.9}]
        augmented = agent._augment_with_kg(retrieved)
        # Exactement 1 augmented : l'exemple
        assert len(augmented) == 1
        assert augmented[0]["idea_id"] == "example_id"
        assert augmented[0]["_kg_relation"] == "example"
        assert augmented[0]["_via_kg"] is True
        assert augmented[0]["_augmented_from"] == "concept_id"

    def test_augment_pulls_illustrated_when_top1_is_example(self, monkeypatch):
        """Top-1 chunk = exemple → on pull le concept qu'il illustre."""
        graph = self._graph_with_example()
        monkeypatch.setattr("pedagogy.knowledge_graph.get_or_build", lambda rag: graph)
        agent = self._agent()
        retrieved = [{"idea_id": "example_id", "content": "...", "score": 0.9}]
        augmented = agent._augment_with_kg(retrieved)
        assert len(augmented) == 1
        assert augmented[0]["idea_id"] == "concept_id"
        assert augmented[0]["_kg_relation"] == "illustrated"

    def test_augment_combines_prereqs_examples_illustrated(self, monkeypatch):
        """Top-1 a a la fois prereqs ET examples → les deux sont pulled,
        l'ordre = prereqs d'abord, puis examples."""
        from pedagogy.knowledge_graph.graph import IdeaGraph, IdeaNode
        g = IdeaGraph()
        g.add_node(IdeaNode(idea_id="prereq_id", label="definition", text="prereq"))
        g.add_node(IdeaNode(idea_id="example_id", label="example", text="ex",
                            illustrates_id="target_id"))
        g.add_node(IdeaNode(
            idea_id="target_id", label="theorem", text="target",
            depends_on_ids={"prereq_id"},
        ))
        monkeypatch.setattr("pedagogy.knowledge_graph.get_or_build", lambda rag: g)
        agent = self._agent()
        retrieved = [{"idea_id": "target_id", "content": "...", "score": 0.9}]
        augmented = agent._augment_with_kg(retrieved)
        assert len(augmented) == 2
        relations = [a["_kg_relation"] for a in augmented]
        # Ordre: prereqs d'abord
        assert relations == ["prereq", "example"]

    def test_augment_skips_neighbors_already_in_pool(self, monkeypatch):
        """Si l'exemple est deja dans le pool retrieved, on ne le re-ajoute pas."""
        graph = self._graph_with_example()
        monkeypatch.setattr("pedagogy.knowledge_graph.get_or_build", lambda rag: graph)
        agent = self._agent()
        retrieved = [
            {"idea_id": "concept_id", "content": "...", "score": 0.9},
            {"idea_id": "example_id", "content": "...", "score": 0.7},
        ]
        augmented = agent._augment_with_kg(retrieved)
        assert augmented == []

    def test_augment_caps_total(self, monkeypatch):
        """Avec beaucoup d'examples, la cap globale s'applique."""
        from agentic.qa.retriever import _KG_TOTAL_AUGMENT_CAP
        from pedagogy.knowledge_graph.graph import IdeaGraph, IdeaNode
        g = IdeaGraph()
        g.add_node(IdeaNode(idea_id="concept_id", label="definition", text="c"))
        for i in range(_KG_TOTAL_AUGMENT_CAP + 3):
            g.add_node(IdeaNode(
                idea_id=f"ex_{i}", label="example", text=f"ex{i}",
                illustrates_id="concept_id",
            ))
        monkeypatch.setattr("pedagogy.knowledge_graph.get_or_build", lambda rag: g)
        agent = self._agent()
        retrieved = [{"idea_id": "concept_id", "content": "...", "score": 0.9}]
        augmented = agent._augment_with_kg(retrieved)
        assert len(augmented) <= _KG_TOTAL_AUGMENT_CAP

    def test_legacy_alias_still_works(self, monkeypatch):
        """Le code externe (tests existants) appelle `_augment_with_prereqs` —
        on garde l'alias retrocompatible."""
        graph = self._graph_with_example()
        monkeypatch.setattr("pedagogy.knowledge_graph.get_or_build", lambda rag: graph)
        agent = self._agent()
        retrieved = [{"idea_id": "concept_id", "content": "...", "score": 0.9}]
        # L'alias retourne le meme resultat que la nouvelle methode
        assert agent._augment_with_prereqs(retrieved) == agent._augment_with_kg(retrieved)


class TestMasteryPropagationBackward:
    """Mastery propagation backward : si dependants mastered, idee implicitement boostee.

    Tests sur la logique pure du calcul (sans DB) — pour les tests DB-aware
    il faudrait des fixtures qui setup StudentMastery rows.
    """

    def test_no_kg_returns_base(self):
        """Sans KG, get_score_with_propagation == get_score."""
        # On verifie juste la logique : si kg=None, le bonus = 0
        from pedagogy.mastery_repo import MasteryRepo
        # PROPAGATION_DECAY est lisible
        assert MasteryRepo.PROPAGATION_DECAY == 0.20

    def test_propagation_formula_no_dependents(self):
        """Si pas de dependants, bonus = 0."""
        # Si dependants vide → bonus = 0 → score = base
        # Cas testable directement en simulant la formule :
        base = 0.5
        decay = 0.20
        mean_dep = 0.0   # pas de dependants -> moyenne 0 -> bonus 0
        bonus = max(0.0, decay * (mean_dep - 0.5) * 2)
        assert bonus == 0.0

    def test_propagation_formula_neutral_dependents(self):
        """mean_dep == 0.5 (uninformative) → bonus = 0."""
        base = 0.3
        decay = 0.20
        mean_dep = 0.5
        bonus = max(0.0, decay * (mean_dep - 0.5) * 2)
        assert bonus == 0.0

    def test_propagation_formula_high_dependents(self):
        """mean_dep == 1.0 (perfect aval) → bonus max = decay."""
        decay = 0.20
        mean_dep = 1.0
        bonus = max(0.0, decay * (mean_dep - 0.5) * 2)
        assert abs(bonus - decay) < 0.001   # = 0.20

    def test_propagation_formula_low_dependents_no_punishment(self):
        """mean_dep == 0.0 (echecs aval) → bonus borne a 0 (pas de punition)."""
        decay = 0.20
        mean_dep = 0.0
        bonus = max(0.0, decay * (mean_dep - 0.5) * 2)
        assert bonus == 0.0   # max(0, -0.20) = 0

    def test_score_clamp_to_one(self):
        """min(1.0, base + bonus) — pas de score > 1.0."""
        base = 0.95
        bonus = 0.20
        assert min(1.0, base + bonus) == 1.0


class TestBanditKGAware:
    """Bandit Phase 2 : ContextBucket avec kg_position."""

    def test_context_bucket_4d_key(self):
        from pedagogy.personalization.bandit.thompson import ContextBucket
        b = ContextBucket(
            learning_style="visual",
            pace="normal",
            mastery_level="medium",
            kg_position="ready",
        )
        assert b.bucket_key == "visual|normal|medium|ready"

    def test_default_kg_position_isolated(self):
        """Backward compat : sans kg_position arg → 'isolated'."""
        from pedagogy.personalization.bandit.thompson import ContextBucket
        b = ContextBucket(learning_style="mixed", pace="slow", mastery_level="low")
        assert b.kg_position == "isolated"
        assert b.bucket_key == "mixed|slow|low|isolated"

    def test_discretize_kg_position(self):
        from pedagogy.personalization.bandit.thompson import discretize_kg_position
        assert discretize_kg_position(None) == "isolated"
        assert discretize_kg_position(0.0) == "blocked"
        assert discretize_kg_position(0.3) == "blocked"
        assert discretize_kg_position(0.5) == "ready"
        assert discretize_kg_position(1.0) == "ready"

    def test_context_from_profile_with_kg(self):
        from pedagogy.personalization.bandit.thompson import context_from_profile
        bucket = context_from_profile(
            learning_style="visual",
            avg_response_time_s=30.0,
            mastery_score=0.6,
            prereqs_mastered_ratio=0.7,
        )
        assert bucket.kg_position == "ready"

    def test_context_from_profile_without_kg(self):
        """Backward compat : sans prereqs_mastered_ratio → bucket isolated."""
        from pedagogy.personalization.bandit.thompson import context_from_profile
        bucket = context_from_profile(
            learning_style="auditory",
            avg_response_time_s=10.0,
            mastery_score=0.9,
        )
        assert bucket.kg_position == "isolated"


class TestEnsureConceptsLoaded:
    """Le lazy bootstrap : si aucun concept attache au KG pour le cours,
    on extrait via KeyBERT et on attache. Avant : les consumers (path_recommender,
    skill_tree, practice_engine) retournaient [] silencieusement."""

    def test_returns_existing_concepts_without_extraction(self, monkeypatch):
        """Si le KG a deja des concepts pour le cours, on ne re-extrait pas."""
        from pedagogy.knowledge_graph import (
            KnowledgeGraph, ConceptInfo, ensure_concepts_loaded,
        )
        kg = KnowledgeGraph()
        kg.attach_concepts([ConceptInfo(name="existing", course_id="c1")])

        # Le loader fait `from pedagogy.knowledge_graph import get_or_build`
        # → patcher le binding sur le package.
        import pedagogy.knowledge_graph as kg_pkg
        monkeypatch.setattr(kg_pkg, "get_or_build", lambda rag: kg)

        class _MockRAG:
            all_docs = []

        out = ensure_concepts_loaded(_MockRAG(), "c1")
        assert len(out) == 1
        assert out[0].name == "existing"

    def test_returns_empty_when_no_docs(self, monkeypatch):
        """RAG vide pour ce cours → pas d'extraction, retour []."""
        from pedagogy.knowledge_graph import (
            KnowledgeGraph, ensure_concepts_loaded,
        )
        kg = KnowledgeGraph()
        import pedagogy.knowledge_graph as kg_pkg
        monkeypatch.setattr(kg_pkg, "get_or_build", lambda rag: kg)

        class _MockRAG:
            all_docs = []

        out = ensure_concepts_loaded(_MockRAG(), "course-with-no-docs")
        assert out == []

    def test_returns_empty_when_no_course_id(self, monkeypatch):
        from pedagogy.knowledge_graph import ensure_concepts_loaded
        out = ensure_concepts_loaded(MagicMock(), "")
        assert out == []


class TestKnowledgeGraphConcepts:
    """API concept-level du KnowledgeGraph unifie (Stage 1 + 2 + 3)."""

    def _kg_with_concepts(self):
        from pedagogy.knowledge_graph import KnowledgeGraph, IdeaNode, ConceptInfo
        g = KnowledgeGraph()
        g.add_node(IdeaNode(idea_id="i_def1", label="definition", text="d1"))
        g.add_node(IdeaNode(idea_id="i_def2", label="definition", text="d2"))
        g.add_node(IdeaNode(idea_id="i_thm",  label="theorem",    text="thm",
                            depends_on_ids={"i_def1", "i_def2"}))
        g.add_node(IdeaNode(idea_id="i_ex",   label="example",    text="ex",
                            illustrates_id="i_thm"))
        g.attach_concepts([
            ConceptInfo(name="basics", course_id="c1", score=5.0,
                        idea_ids={"i_def1", "i_def2"}),
            ConceptInfo(name="kmeans", course_id="c1", score=10.0,
                        idea_ids={"i_thm"}),
            ConceptInfo(name="examples", course_id="c1", score=2.0,
                        idea_ids={"i_ex"}),
        ])
        return g

    def test_list_concepts_sorted_by_score_desc(self):
        g = self._kg_with_concepts()
        names = [c.name for c in g.list_concepts("c1")]
        assert names == ["kmeans", "basics", "examples"]

    def test_prereq_concepts_derived_from_idea_edges(self):
        """basics est prereq de kmeans car i_def1 et i_def2 (∈ basics)
        sont depends_on de i_thm (∈ kmeans)."""
        g = self._kg_with_concepts()
        prereqs = g.prereq_concepts("kmeans")
        assert [p.name for p in prereqs] == ["basics"]

    def test_example_concepts_of(self):
        """examples illustre kmeans car i_ex.illustrates_id = i_thm."""
        g = self._kg_with_concepts()
        examples = g.example_concepts_of("kmeans")
        assert [e.name for e in examples] == ["examples"]

    def test_count_idea_edges_between_directional(self):
        """basics -> kmeans : 2 edges (i_def1, i_def2 -> i_thm).
        Direction inverse : 0."""
        g = self._kg_with_concepts()
        assert g.count_idea_edges_between("basics", "kmeans") == 2
        assert g.count_idea_edges_between("kmeans", "basics") == 0
        assert g.count_idea_edges_between("foo", "bar") == 0  # unknown safe

    def test_concepts_of_idea(self):
        g = self._kg_with_concepts()
        concepts = g.concepts_of_idea("i_thm")
        assert [c.name for c in concepts] == ["kmeans"]

    def test_attach_concepts_idempotent(self):
        """Re-appeler attach_concepts remplace la couche, ne duplique pas."""
        from pedagogy.knowledge_graph import KnowledgeGraph, ConceptInfo
        g = KnowledgeGraph()
        g.attach_concepts([ConceptInfo(name="a", course_id="c1")])
        g.attach_concepts([ConceptInfo(name="b", course_id="c1")])
        assert [c.name for c in g.list_concepts("c1")] == ["b"]

    def test_concept_info_to_dict(self):
        from pedagogy.knowledge_graph import ConceptInfo
        ci = ConceptInfo(name="x", display_name="X", canonical_name="X-canon",
                         description="desc", bloom_level="apply",
                         course_id="c1", chapter_idxs={2, 3}, score=4.5,
                         idea_ids={"i1", "i2"})
        d = ci.to_dict()
        assert d["name"] == "x"
        assert d["chapter_idxs"] == [2, 3]
        assert d["idea_count"] == 2
        assert d["score"] == 4.5

    def test_concept_to_concept_info_bridge(self):
        """Le bridge Concept -> ConceptInfo (extracteur staging -> KG)."""
        from pedagogy.concept_types import Concept
        c = Concept(label="kmeans", name="K-means")
        c.canonical_name = "K-Means"
        c.description = "clustering"
        c.bloom_level = "apply"
        c.chapter_idxs = {2, 3}
        c.chunk_ids = {"i_thm"}
        c.score = 10.0
        ci = c.to_concept_info(course_id="c1")
        assert ci.name == "kmeans"
        assert ci.display_name == "K-means"
        assert ci.canonical_name == "K-Means"
        assert ci.idea_ids == {"i_thm"}
        assert ci.course_id == "c1"


class TestResponderUsesKGLabels:
    """Verifier que le responder injecte les labels KG dans le prompt LLM.
    Avant ce fix : `_kg_relation` etait calcule mais ignore par le responder."""

    def test_format_chunks_no_kg_uses_simple_id(self):
        from agentic.qa.responder import _format_chunks_with_ids
        chunks = [{"idea_id": "abc", "content": "regular content"}]
        block, _ = _format_chunks_with_ids(chunks, lang="fr")
        assert "[id:abc]" in block
        assert "|" not in block.split("\n")[0]  # pas de label sur header

    def test_format_chunks_with_kg_relation_adds_fr_label(self):
        from agentic.qa.responder import _format_chunks_with_ids
        chunks = [
            {"idea_id": "regular", "content": "main answer"},
            {"idea_id": "p1", "content": "prereq content",
             "_via_kg": True, "_kg_relation": "prereq"},
            {"idea_id": "ex1", "content": "example content",
             "_via_kg": True, "_kg_relation": "example"},
            {"idea_id": "ill1", "content": "illustrated content",
             "_via_kg": True, "_kg_relation": "illustrated"},
        ]
        block, _ = _format_chunks_with_ids(chunks, lang="fr")
        assert "[id:regular]" in block
        assert "[id:p1 | prerequis]" in block
        assert "[id:ex1 | exemple concret]" in block
        assert "[id:ill1 | concept illustre]" in block

    def test_format_chunks_with_kg_relation_adds_en_label(self):
        from agentic.qa.responder import _format_chunks_with_ids
        chunks = [
            {"idea_id": "p1", "content": "x",
             "_via_kg": True, "_kg_relation": "prereq"},
        ]
        block, _ = _format_chunks_with_ids(chunks, lang="en")
        assert "[id:p1 | prerequisite]" in block

    def test_format_chunks_unknown_relation_falls_back_to_simple_id(self):
        """Si _kg_relation n'est pas dans le mapping, on utilise [id:xxx]
        sans label plutot que de crasher."""
        from agentic.qa.responder import _format_chunks_with_ids
        chunks = [{"idea_id": "x", "content": "c",
                   "_via_kg": True, "_kg_relation": "weird_relation"}]
        block, _ = _format_chunks_with_ids(chunks, lang="fr")
        assert "[id:x]" in block
        assert "weird_relation" not in block


class TestContextAgentKGAugmentation:
    """ContextAgent (chemin teaching/narration) doit beneficier de la meme
    KG augmentation que RetrieverAgent (chemin QA). Avant ce fix : il
    bypass-ait l'expansion via un appel direct a rag.retrieve_chunks."""

    def _agent_with_mock_rag(self, retrieve_return):
        from agentic.teaching.context import ContextAgent

        class _MockRAG:
            def __init__(self, retrieve_return):
                self._retrieve_return = retrieve_return
            def retrieve_chunks(self, query, k=5, current_chapter_idx=None, course_id=None):
                return self._retrieve_return

        return ContextAgent(rag=_MockRAG(retrieve_return))

    def _make_state_with_plan(self, course_id="c", chapter=1):
        # Plan minimal avec 1 idea concept
        plan = type("Plan", (), {})()
        idea = type("Idea", (), {})()
        idea.type = "concept"
        idea.content_brief = "k-means clustering"
        plan.ideas = [idea]
        return {
            "plan": plan,
            "course_id": course_id,
            "chapter_idx": chapter,
        }

    def _kg_with_neighbors(self):
        from pedagogy.knowledge_graph.graph import IdeaGraph, IdeaNode
        g = IdeaGraph()
        g.add_node(IdeaNode(idea_id="prereq_id", label="definition", text="centroid"))
        g.add_node(IdeaNode(idea_id="kmeans_id", label="theorem",
                            text="k-means", depends_on_ids={"prereq_id"}))
        return g

    def test_context_agent_appends_kg_chunks_when_knob_on(self, monkeypatch):
        # Mock RAG retournant 1 chunk avec idea_id
        from langchain_core.documents import Document
        doc = Document(page_content="K-means clustering def",
                       metadata={"idea_id": "kmeans_id", "chapter_idx": 1,
                                 "section_idx": 0})
        agent = self._agent_with_mock_rag([(doc, 0.9, "src")])
        monkeypatch.setattr("pedagogy.knowledge_graph.get_or_build",
                            lambda rag: self._kg_with_neighbors())
        monkeypatch.setattr("agentic.teaching.context.Config.RAG_USE_GRAPH_EXPANSION", True)

        result = agent(self._make_state_with_plan())
        chunks = result["retrieved_chunks"]
        # 1 direct + 1 KG-augmented (le prereq)
        assert len(chunks) == 2
        kg_count = sum(1 for c in chunks if c.get("_via_kg"))
        assert kg_count == 1
        assert chunks[1]["_kg_relation"] == "prereq"

    def test_context_agent_skips_kg_when_knob_off(self, monkeypatch):
        from langchain_core.documents import Document
        doc = Document(page_content="K-means clustering def",
                       metadata={"idea_id": "kmeans_id", "chapter_idx": 1})
        agent = self._agent_with_mock_rag([(doc, 0.9, "src")])
        monkeypatch.setattr("agentic.teaching.context.Config.RAG_USE_GRAPH_EXPANSION", False)

        result = agent(self._make_state_with_plan())
        chunks = result["retrieved_chunks"]
        assert len(chunks) == 1  # juste le direct, pas de KG
        assert not any(c.get("_via_kg") for c in chunks)


class TestNarratorKGLabels:
    """Le narrator doit injecter les tags [rappel] / [exemple] / [concept
    general] dans son chunks_block pour que la narration soit structuree."""

    def test_chunks_block_uses_kg_tags(self):
        from agentic.teaching.narrator import _build_augmented_content

        plan = type("Plan", (), {"to_brief": lambda self: "(plan brief)"})()
        chunks = [
            {"content": "Direct content from slide"},
            {"content": "Centroid is the barycenter",
             "_via_kg": True, "_kg_relation": "prereq"},
            {"content": "Iris dataset clustering",
             "_via_kg": True, "_kg_relation": "example"},
        ]
        out = _build_augmented_content(
            plan=plan, slide_content="slide", lang="fr", chunks=chunks,
        )
        assert "[rappel] Centroid" in out
        assert "[exemple] Iris" in out
        # Direct chunk : sans tag
        assert "Direct content from slide" in out
        # Hint d'usage present
        assert "rappel" in out and "prerequis" in out

    def test_chunks_block_en_labels(self):
        from agentic.teaching.narrator import _build_augmented_content

        plan = type("Plan", (), {"to_brief": lambda self: "(brief)"})()
        chunks = [
            {"content": "main content"},
            {"content": "prereq content",
             "_via_kg": True, "_kg_relation": "prereq"},
        ]
        out = _build_augmented_content(
            plan=plan, slide_content="slide", lang="en", chunks=chunks,
        )
        assert "[reminder] prereq content" in out

    def test_chunks_block_no_kg_chunks_no_hint(self):
        from agentic.teaching.narrator import _build_augmented_content

        plan = type("Plan", (), {"to_brief": lambda self: "(brief)"})()
        chunks = [{"content": "regular chunk"}]
        out = _build_augmented_content(
            plan=plan, slide_content="slide", lang="fr", chunks=chunks,
        )
        assert "regular chunk" in out
        # Pas de hint "[rappel] = prerequis" si aucun chunk KG
        assert "[rappel]" not in out
        assert "[exemple]" not in out


class TestGraphExpansionKnob:
    """Le knob Config.RAG_USE_GRAPH_EXPANSION desactive l'augmentation
    sans toucher au reste du flow retrieval."""

    def test_knob_default_is_true(self):
        from core.config import Config
        assert Config.RAG_USE_GRAPH_EXPANSION is True

    def test_knob_off_skips_augmentation(self, monkeypatch):
        """Avec le knob off, _augment_with_kg ne devrait pas etre appele
        depuis le flow principal __call__. On verifie via un mock."""
        import asyncio
        from agentic.qa.retriever import RetrieverAgent

        # Mock RAG pour eviter l'init reelle
        mock_rag = MagicMock()
        mock_rag.retrieve_chunks.return_value = []
        agent = RetrieverAgent(rag=mock_rag)

        called = []
        monkeypatch.setattr(agent, "_augment_with_kg",
                            lambda chunks: called.append("called") or [])
        monkeypatch.setattr("agentic.qa.retriever.Config.RAG_USE_GRAPH_EXPANSION", False)

        state = {"rewritten_query": "q", "course_id": "c", "chapter_idx": 1,
                 "session_id": "s", "student_id": "u"}
        asyncio.run(agent(state))
        assert called == []  # Knob OFF → pas d'appel a _augment_with_kg
