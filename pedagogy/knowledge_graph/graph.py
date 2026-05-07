"""KnowledgeGraph — single source of truth for ideas + concepts of a course.

Avant : il y avait DEUX systemes paralleles :
  - IdeaGraph : graphe d'idees fines (def, theoreme), edges semantiques LLM
  - ConceptKG : index de concepts agreges (KeyBERT), edges cooccurrence
                heuristique stockees en Postgres

Probleme : edges concept = "apparait dans les memes chunks" — heuristique
faible. Et 2 sources d'edges = drift inevitable.

Maintenant : 1 graphe ``KnowledgeGraph`` avec 2 niveaux de granularite.
Les CONCEPTS sont des CLUSTERS d'IDEES (tags), et leurs edges sont DERIVEES
des edges idee. Une seule source de verite (le LLM a l'ingestion).

# Data model

## IdeaNode (niveau fin, sortie LLM idea-chunking)

    idea_id            : str    stable hash, primary key
    label              : str    definition | theorem | example | …
    text               : str    the actual content
    course_id          : str
    chapter_idx        : int
    section_idx        : int
    depends_on_ids     : set[str]   prerequisites (incoming edges)
    illustrates_id     : str|None   single concept illustrated (example→concept)

## ConceptInfo (niveau gros, agregation KeyBERT + LLM enrichment)

    name               : str    snake_case unique id ("k_means")
    display_name       : str    Human readable ("K-means")
    canonical_name     : str    LLM-canonicalised ("K-Means clustering")
    description        : str    1-phrase explanation
    bloom_level        : str    remember | understand | apply | analyze
    course_id          : str
    chapter_idxs       : set[int]
    score              : float  importance (somme KeyBERT scores)
    idea_ids           : set[str]   les idees qui appartiennent a ce concept

# Edges

    A --depends_on--> B   means "A requires understanding B"        (idea-level)
    A --illustrates--> B  means "A is an example of B"              (idea-level)
    Concept C1 --prereq--> C2 IFF some idea in C1 depends on some idea in C2
    Concept C1 --example--> C2 IFF some idea in C1 illustrates some idea in C2

# Provided queries

    # Idea-level (was IdeaGraph)
    get_idea(idea_id)
    prerequisites_of(idea_id, transitive=False)
    dependents_of(idea_id, transitive=False)
    examples_of(idea_id)
    learning_path(target, mastered)

    # Concept-level (was ConceptKG, now derived)
    list_concepts(course_id=None)
    get_concept(name)
    ideas_in_concept(name)
    prereq_concepts(name)
    example_concepts_of(name)
    concepts_of_idea(idea_id)

# References

  - Doignon & Falmagne (1985). *Knowledge Spaces.* Foundational theory of
    competence-based curriculum modeling.
  - Pirolli & Bielaczyc (1989). *Empirical analyses of self-explanation.*
    Intra-concept dependency tracking.
  - ALEKS / Knewton — modern industrial uses of dependency graphs in ITS.

# Limits known

  - Intra-chunk only (LLM doesn't see cross-chunk references)
  - Static once ingested (no online refinement from interaction logs)
  - No probabilistic edges (all dependencies are hard)
  - Concepts = KeyBERT clusters → bruit possible (mitige par Fix #4
    validate_per_chapter)
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

log = logging.getLogger("pedagogy.knowledge_graph")


@dataclass
class IdeaNode:
    """A single idea in the graph.

    Built from one ``Document.metadata`` record produced by the RAG.
    """
    idea_id:        str
    label:          str
    text:           str
    course_id:      str = ""
    chapter_idx:    int = 0
    section_idx:    int = 0
    depends_on_ids: set[str] = field(default_factory=set)
    illustrates_id: Optional[str] = None

    @classmethod
    def from_document(cls, doc) -> Optional["IdeaNode"]:
        """Build from a langchain Document or any object with ``metadata`` dict.

        Returns None if the document lacks an idea_id (legacy chunks pre-graph).
        """
        meta = getattr(doc, "metadata", None) or {}
        idea_id = meta.get("idea_id")
        if not idea_id:
            return None
        deps = meta.get("depends_on_ids") or []
        if not isinstance(deps, list):
            deps = []
        illus = meta.get("illustrates_id")
        if not isinstance(illus, str) or not illus:
            illus = None
        return cls(
            idea_id=idea_id,
            label=str(meta.get("idea_label", "note")),
            text=getattr(doc, "page_content", "") or meta.get("original_text", ""),
            course_id=str(meta.get("course", "")),
            chapter_idx=int(meta.get("chapter_idx", 0) or 0),
            section_idx=int(meta.get("section_idx", 0) or 0),
            depends_on_ids=set(deps),
            illustrates_id=illus,
        )


@dataclass
class ConceptInfo:
    """A concept = aggregated cluster of ideas (KeyBERT + dedup + LLM enrich).

    Replaces the old `Concept` from concept_extractor + the `concept_kg` DB
    table. Stored alongside ideas in a single KnowledgeGraph.
    """
    name:           str                          # snake_case primary key ("k_means")
    display_name:   str = ""                     # human readable ("K-means")
    canonical_name: str = ""                     # LLM-canonicalised ("K-Means clustering")
    description:    str = ""                     # 1-phrase explanation
    bloom_level:    str = ""                     # remember|understand|apply|analyze
    course_id:      str = ""
    chapter_idxs:   set[int] = field(default_factory=set)
    score:          float = 0.0                  # KeyBERT importance score
    idea_ids:       set[str] = field(default_factory=set)   # ideas in this concept

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "display_name":   self.display_name or self.name,
            "canonical_name": self.canonical_name,
            "description":    self.description,
            "bloom_level":    self.bloom_level,
            "course_id":      self.course_id,
            "chapter_idxs":   sorted(self.chapter_idxs),
            "score":          round(self.score, 3),
            "idea_count":     len(self.idea_ids),
            "idea_ids":       list(self.idea_ids),
        }


class KnowledgeGraph:
    """In-memory directed graph keyed by ``idea_id``, with concept-level
    aggregation derived from idea-level edges.

    Build once at startup from the RAG's indexed Documents (ideas), then
    optionally enrich with concept clusters via ``attach_concepts(...)``.
    All queries are dict / set / BFS — sub-millisecond on graphs of <100k
    nodes.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, IdeaNode] = {}
        # Reverse indices computed once at ingest time
        self._dependents:    dict[str, set[str]] = {}    # b → {a : a depends_on b}
        self._examples_of:   dict[str, set[str]] = {}    # b → {a : a illustrates b}
        # Course-scoped node lists (for cohort queries)
        self._by_course:     dict[str, set[str]] = {}
        # ── Concept layer (filled by attach_concepts) ───────────────────
        self._concepts:           dict[str, ConceptInfo] = {}    # name → info
        self._concept_for_idea:   dict[str, set[str]] = {}       # idea_id → {concept_name}
        self._concepts_by_course: dict[str, set[str]] = {}       # course_id → {concept_name}

    # ── Build API ─────────────────────────────────────────────────────

    def add_node(self, node: IdeaNode) -> None:
        """Insert (or replace) a node and rebuild reverse edges around it."""
        self._nodes[node.idea_id] = node
        if node.course_id:
            self._by_course.setdefault(node.course_id, set()).add(node.idea_id)
        # Forward edges → reverse index
        for dep in node.depends_on_ids:
            self._dependents.setdefault(dep, set()).add(node.idea_id)
        if node.illustrates_id:
            self._examples_of.setdefault(node.illustrates_id, set()).add(node.idea_id)

    def add_documents(self, documents: Iterable) -> int:
        """Bulk ingest from a list of langchain Documents.

        Returns the number of nodes successfully added (skips ones missing idea_id).
        """
        n = 0
        for doc in documents:
            node = IdeaNode.from_document(doc)
            if node is not None:
                self.add_node(node)
                n += 1
        log.info("IdeaGraph: ingested %d nodes (skipped %d malformed)",
                 n, sum(1 for _ in documents) - n if hasattr(documents, "__len__") else 0)
        return n

    # ── Read API ──────────────────────────────────────────────────────

    def get(self, idea_id: str) -> Optional[IdeaNode]:
        return self._nodes.get(idea_id)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, idea_id: str) -> bool:
        return idea_id in self._nodes

    def prerequisites_of(self, idea_id: str, transitive: bool = False) -> list[IdeaNode]:
        """Ideas that ``idea_id`` directly (or transitively) depends on.

        ``transitive=True`` walks the full prerequisite chain, BFS, with
        cycle protection (the LLM occasionally emits cycles).
        """
        node = self._nodes.get(idea_id)
        if node is None:
            return []
        if not transitive:
            return [self._nodes[d] for d in node.depends_on_ids if d in self._nodes]
        # Transitive : BFS with visited set to break cycles
        out: list[IdeaNode] = []
        seen: set[str] = {idea_id}
        queue = deque(node.depends_on_ids)
        while queue:
            curr = queue.popleft()
            if curr in seen or curr not in self._nodes:
                continue
            seen.add(curr)
            out.append(self._nodes[curr])
            queue.extend(self._nodes[curr].depends_on_ids)
        return out

    def dependents_of(self, idea_id: str, transitive: bool = False) -> list[IdeaNode]:
        """Ideas that directly (or transitively) depend on ``idea_id``."""
        if idea_id not in self._dependents:
            return []
        if not transitive:
            return [self._nodes[d] for d in self._dependents[idea_id] if d in self._nodes]
        out: list[IdeaNode] = []
        seen: set[str] = {idea_id}
        queue = deque(self._dependents.get(idea_id, set()))
        while queue:
            curr = queue.popleft()
            if curr in seen or curr not in self._nodes:
                continue
            seen.add(curr)
            out.append(self._nodes[curr])
            queue.extend(self._dependents.get(curr, set()))
        return out

    def examples_of(self, idea_id: str) -> list[IdeaNode]:
        """Ideas with ``illustrates_id == idea_id`` (i.e. examples of this concept)."""
        if idea_id not in self._examples_of:
            return []
        return [self._nodes[e] for e in self._examples_of[idea_id] if e in self._nodes]

    def learning_path(self, target_id: str, mastered: set[str]) -> list[IdeaNode]:
        """Ordered list of ideas to teach BEFORE reaching ``target_id``.

        Returns prerequisites the student has NOT yet mastered, in topological
        order (deepest first). The target itself is included at the end if
        not in ``mastered``. Cycle-safe.

        Use case : student wants to understand idea X. Compute the minimal
        path from where they are now to X.
        """
        if target_id not in self._nodes:
            return []

        # Topological order via post-order DFS
        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(nid: str) -> None:
            if nid in visited or nid in visiting:
                return                       # cycle guard
            visiting.add(nid)
            node = self._nodes.get(nid)
            if node is not None:
                for dep in node.depends_on_ids:
                    dfs(dep)
            visiting.discard(nid)
            visited.add(nid)
            order.append(nid)

        dfs(target_id)
        # Filter out already-mastered ones
        return [self._nodes[nid] for nid in order
                if nid in self._nodes and nid not in mastered]

    # ── Concept layer (was ConceptKG, now an aggregation view) ────────

    def attach_concepts(self, concepts: Iterable["ConceptInfo"]) -> int:
        """Attach KeyBERT concept clusters to the graph.

        Each `ConceptInfo` carries the set of `idea_ids` it covers. We build
        the reverse index (idea → concepts) so all queries are O(1).
        Idempotent : recalling with same concepts replaces the layer.

        Returns the number of concepts attached.
        """
        # Reset concept indices (idempotent rebuild)
        self._concepts.clear()
        self._concept_for_idea.clear()
        self._concepts_by_course.clear()

        n = 0
        for c in concepts:
            if not isinstance(c, ConceptInfo):
                continue
            self._concepts[c.name] = c
            if c.course_id:
                self._concepts_by_course.setdefault(c.course_id, set()).add(c.name)
            for idea_id in c.idea_ids:
                self._concept_for_idea.setdefault(idea_id, set()).add(c.name)
            n += 1
        log.info("KnowledgeGraph: attached %d concepts (%d idea-concept links)",
                 n, sum(len(v) for v in self._concept_for_idea.values()))
        return n

    def list_concepts(self, course_id: Optional[str] = None) -> list[ConceptInfo]:
        """All concepts, optionally scoped to a course. Sorted by descending score."""
        if course_id is None:
            items = list(self._concepts.values())
        else:
            names = self._concepts_by_course.get(course_id, set())
            items = [self._concepts[n] for n in names if n in self._concepts]
        items.sort(key=lambda c: -c.score)
        return items

    def get_concept(self, name: str) -> Optional[ConceptInfo]:
        return self._concepts.get(name)

    def ideas_in_concept(self, name: str) -> list[IdeaNode]:
        """All ideas tagged with this concept."""
        c = self._concepts.get(name)
        if c is None:
            return []
        return [self._nodes[iid] for iid in c.idea_ids if iid in self._nodes]

    def concepts_of_idea(self, idea_id: str) -> list[ConceptInfo]:
        """Which concepts a given idea belongs to (usually 0 or 1, sometimes more)."""
        names = self._concept_for_idea.get(idea_id, set())
        return [self._concepts[n] for n in names if n in self._concepts]

    def prereq_concepts(self, name: str) -> list[ConceptInfo]:
        """Concepts that must be understood BEFORE ``name``.

        Derived semantically: concept C2 is a prereq of C1 IFF there exists
        idea i1 ∈ C1 and idea i2 ∈ C2 such that i1 depends_on i2 (and C2 ≠ C1).

        Replaces the old `concept_cooccurrence` heuristic (which used
        "appears in same chunks" — much weaker signal).
        """
        c = self._concepts.get(name)
        if c is None:
            return []
        prereq_names: set[str] = set()
        for idea_id in c.idea_ids:
            node = self._nodes.get(idea_id)
            if node is None:
                continue
            for dep_id in node.depends_on_ids:
                for cn in self._concept_for_idea.get(dep_id, set()):
                    if cn != name:
                        prereq_names.add(cn)
        return [self._concepts[n] for n in prereq_names if n in self._concepts]

    def count_idea_edges_between(self, source_name: str, target_name: str) -> int:
        """Compte les edges idee-level depends_on entre 2 concepts.

        Usage : poids semantique pour les edges Cytoscape (epaisseur visuelle).
        Plus il y a d'idees de `target` qui dependent d'idees de `source`,
        plus la dependance pedagogique est forte.

        Avant unification : poids = "nb chunks ou les 2 cooccurrent" (cooccurrence
        symetrique heuristique). Maintenant : nb d'edges semantiques directionnelles.
        """
        source = self._concepts.get(source_name)
        target = self._concepts.get(target_name)
        if source is None or target is None:
            return 0
        count = 0
        for tid in target.idea_ids:
            node = self._nodes.get(tid)
            if node is None:
                continue
            count += sum(1 for d in node.depends_on_ids if d in source.idea_ids)
        return count

    def example_concepts_of(self, name: str) -> list[ConceptInfo]:
        """Concepts whose ideas illustrate ``name``.

        Concept C2 illustrates C1 IFF there exists idea i2 ∈ C2 with
        illustrates_id pointing to some idea in C1.
        """
        c = self._concepts.get(name)
        if c is None:
            return []
        # Reverse: trouver les idees qui illustrent une idee de ce concept
        example_names: set[str] = set()
        for idea_id in c.idea_ids:
            for ex_idea_id in self._examples_of.get(idea_id, set()):
                for cn in self._concept_for_idea.get(ex_idea_id, set()):
                    if cn != name:
                        example_names.add(cn)
        return [self._concepts[n] for n in example_names if n in self._concepts]

    # ── Stats / introspection ─────────────────────────────────────────

    def stats(self) -> dict:
        n_nodes = len(self._nodes)
        n_dep_edges = sum(len(n.depends_on_ids) for n in self._nodes.values())
        n_illus_edges = sum(1 for n in self._nodes.values() if n.illustrates_id)
        roots = sum(1 for n in self._nodes.values() if not n.depends_on_ids)
        leaves = sum(1 for nid in self._nodes if nid not in self._dependents)
        return {
            "nodes":             n_nodes,
            "depends_edges":     n_dep_edges,
            "illustrates_edges": n_illus_edges,
            "courses":           len(self._by_course),
            "roots":             roots,        # no prerequisites
            "leaves":            leaves,       # no dependent
            "avg_in_degree":     round(n_dep_edges / max(1, n_nodes), 2),
            "concepts":          len(self._concepts),
            "concept_links":     sum(len(v) for v in self._concept_for_idea.values()),
        }


# Backward compatibility : ancien nom IdeaGraph.
# Du code consumer (qa/retriever, teaching/context, knowledge_graph/builder,
# tests) reference IdeaGraph — on garde l'alias pour ne rien casser pendant
# la migration. A supprimer apres Stage 3.
IdeaGraph = KnowledgeGraph
