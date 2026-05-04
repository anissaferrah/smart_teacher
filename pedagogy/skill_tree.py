"""Skill Tree builder — Khan-like ITS sur le KnowledgeGraph unifie.

Avant : edges via concept_cooccurrence (heuristique = "apparaissent dans
les memes chunks") + ordering par chapter_idx.

Maintenant : edges semantiques DERIVEES via `kg.prereq_concepts(name)` —
basees sur les `depends_on` au niveau idee (LLM idea-chunking).

Etats des noeuds :
  🔴 not_started   (mastery == 0)
  🟡 in_progress   (0 < mastery < MASTERY_THRESHOLD)
  🟢 mastered      (mastery >= MASTERY_THRESHOLD)
  🔒 locked        (prereqs not satisfied — derive du graphe semantique)
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional, Any

from sqlalchemy import select

from database.init_db import AsyncSessionLocal
from database.models import StudentMastery

log = logging.getLogger("SmartTeacher.SkillTree")

# Single canonical mastery threshold across the codebase.
from pedagogy.mastery_repo import MASTERY_THRESHOLD  # noqa: E402


def _coerce_uuid(value) -> Optional[uuid.UUID]:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


class SkillTreeBuilder:
    """Compose un Cytoscape JSON enrichi avec mastery state per student."""

    @staticmethod
    async def build(student_id, course_id) -> dict[str, Any]:
        """Retourne {nodes, edges, stats} avec colors + locked flags."""
        sid = _coerce_uuid(student_id)
        course_id_str = str(course_id) if course_id else ""
        if not course_id_str:
            return {"nodes": [], "edges": [], "stats": {}}

        # 1. Recuperer le KnowledgeGraph
        try:
            from deps import get_rag
            from pedagogy.knowledge_graph import get_or_build, ensure_concepts_loaded
            rag = get_rag()
            kg = get_or_build(rag)
        except Exception as exc:
            log.warning(f"skill_tree: KG unavailable ({exc})")
            return {"nodes": [], "edges": [], "stats": {"reason": str(exc)}}

        # Lazy bootstrap si aucun concept attache pour ce cours
        concepts = kg.list_concepts(course_id_str)
        if not concepts:
            concepts = ensure_concepts_loaded(rag, course_id_str, enrich=False)
        if not concepts:
            return {"nodes": [], "edges": [], "stats": {"reason": "no concepts (extraction skipped or empty RAG)"}}

        # 2. Mastery scores avec propagation KG (backward)
        # Boost les ideas non testees mais dont les dependants sont mastered.
        mastery_map: dict[str, float] = {}
        if sid:
            try:
                from pedagogy.mastery_repo import MasteryRepo
                all_idea_ids = set()
                for c in concepts:
                    all_idea_ids.update(c.idea_ids)
                if all_idea_ids:
                    mastery_map = await MasteryRepo.get_scores_bulk_with_propagation(
                        sid, course_id_str, list(all_idea_ids), kg=kg,
                    )
            except Exception as exc:
                log.debug(f"mastery lookup failed: {exc}")

        def concept_mastery(c) -> float:
            """Moyenne des mastery des idea_ids du concept."""
            if not c.idea_ids:
                return 0.0
            scores = [mastery_map.get(iid, 0.0) for iid in c.idea_ids]
            return sum(scores) / len(scores) if scores else 0.0

        # Stats
        stats = {
            "total":       len(concepts),
            "mastered":    0,
            "in_progress": 0,
            "not_started": 0,
            "locked":      0,
        }

        # 3. Build nodes with state
        nodes = []
        for c in concepts:
            mastery = concept_mastery(c)

            if mastery >= MASTERY_THRESHOLD:
                state = "mastered"
                stats["mastered"] += 1
            elif mastery > 0:
                state = "in_progress"
                stats["in_progress"] += 1
            else:
                state = "not_started"
                stats["not_started"] += 1

            # Locked si prereqs pas tous OK (et concept pas encore commence)
            locked = False
            weak_prereqs: list[str] = []
            prereqs = kg.prereq_concepts(c.name)
            if state == "not_started":
                for p in prereqs:
                    p_mastery = concept_mastery(p)
                    if p_mastery < MASTERY_THRESHOLD:
                        locked = True
                        weak_prereqs.append(p.name)
                if locked:
                    stats["locked"] += 1

            nodes.append({
                "data": {
                    "id":            c.name,
                    "label":         c.canonical_name or c.display_name or c.name,
                    "score":         round(float(c.score), 3),
                    "mastery":       round(mastery, 3),
                    "state":         state,
                    "locked":        locked,
                    "weak_prereqs":  weak_prereqs[:5],
                    "bloom":         c.bloom_level or "",
                    "chapters":      sorted(c.chapter_idxs),
                    "description":   c.description or "",
                }
            })

        # 4. Build edges = depends_on (concept C1 → C2 si C1 prereq C2)
        # Edges direction : prereq → dependent (C2 a besoin de C1).
        # Avant : symetrique cooccurrence + ordering chapter. Maintenant
        # direction explicite via le graphe semantique. Poids = nb d'edges
        # idee-level entre les 2 concepts (force semantique de la dependance).
        edge_list = []
        seen_edges: set[tuple[str, str]] = set()
        for c in concepts:
            for p in kg.prereq_concepts(c.name):
                key = (p.name, c.name)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                weight = max(1, kg.count_idea_edges_between(p.name, c.name))
                edge_list.append({
                    "data": {
                        "id":     f"{p.name}__{c.name}",
                        "source": p.name,
                        "target": c.name,
                        "weight": weight,
                    }
                })

        return {
            "nodes":      nodes,
            "edges":      edge_list,
            "stats":      stats,
            "course_id":  course_id_str,
            "student_id": str(sid) if sid else None,
        }
