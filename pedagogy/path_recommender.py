"""Adaptive Path Recommender — Khan-like ITS sur le KnowledgeGraph unifie.

Avant : lisait Postgres (concept_kg + concept_cooccurrence) → edges
heuristiques basees sur la cooccurrence dans les chunks.

Maintenant : lit le `KnowledgeGraph` (in-memory, partage avec le retrieval) →
edges semantiques DERIVEES des `depends_on` au niveau idee. Plus de table
cooccurrence, plus de drift entre les 2 graphes.

Algorithme :
  1. Concepts du cours via `kg.list_concepts(course_id)`
  2. Pour chaque concept C : prereqs = `kg.prereq_concepts(C.name)`
  3. Mastery du concept = moyenne des mastery de ses idea_ids (nouveau)
  4. Concept "unlocked" si mastery prereqs >= MASTERY_THRESHOLD
  5. Tri : prereqs satisfaits, distance chapitre, bloom difficulty, score
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Any

from sqlalchemy import select

from database.init_db import AsyncSessionLocal
from database.models import StudentMastery

log = logging.getLogger("SmartTeacher.PathRecommender")

# Single canonical mastery threshold across the whole codebase.
from pedagogy.mastery_repo import MASTERY_THRESHOLD  # noqa: E402

# BLOOM_DIFFICULTY : ordering des verbes Bloom de la taxonomie revisee
# (Anderson & Krathwohl 2001).
#   - Usage : tri des candidats-recommandation (apprendre "remember" avant
#     "analyze" si tout le reste est egal).
#   - Echelle ordinale (1..6), pas une distance metrique. La difference
#     remember(1) -> understand(2) n'est PAS la meme que apply(3) -> analyze(4).
#   - Default si bloom_level absent ou inconnu : 3 (apply, milieu de l'echelle).
#   - L'enrichment LLM (Layer 3 dans concept_extractor) classe chaque concept
#     dans une de ces 6 categories.
BLOOM_DIFFICULTY = {
    "remember":   1,    # rappel / reconnaissance
    "understand": 2,    # explication / interpretation
    "apply":      3,    # execution / utilisation dans un contexte connu
    "analyze":    4,    # decomposition / inference de structure
    "evaluate":   5,    # jugement avec criteres
    "create":     6,    # production originale / synthese
}


def _coerce_uuid(value) -> Optional[uuid.UUID]:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


@dataclass
class Recommendation:
    concept_id: str            # = ConceptInfo.name (stable string ID)
    label: str                 # alias of concept_id (retro-compat)
    name: str                  # display name
    canonical_name: str
    description: str
    bloom_level: str
    chapter_idxs: list[int]
    score: float               # importance KeyBERT
    mastery_score: float       # student mastery [0..1]
    prereqs_met: bool
    prereq_labels: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PathRecommender:
    """Recommander le prochain concept a apprendre pour un eleve donne."""

    @staticmethod
    async def recommend_next(
        student_id,
        course_id,
        current_chapter_idx: int | None = None,
        top_k: int = 3,
    ) -> list[Recommendation]:
        """Retourne top_k concepts recommandes (ordonnés par priorité)."""
        sid = _coerce_uuid(student_id)
        course_id_str = str(course_id) if course_id else ""
        if not course_id_str:
            return []

        # 1. Recuperer le KnowledgeGraph (source unique de verite)
        try:
            from deps import get_rag
            from pedagogy.knowledge_graph import get_or_build, ensure_concepts_loaded
            rag = get_rag()
            kg = get_or_build(rag)
        except Exception as exc:
            log.warning(f"recommend_next: KG unavailable ({exc})")
            return []

        # Lazy bootstrap : si aucun concept attache, on extrait via KeyBERT
        # (fast path, sans enrichment LLM). Avant : retournait [] et l'admin
        # devait appeler /concept-graph manuellement.
        concepts = kg.list_concepts(course_id_str)
        if not concepts:
            concepts = ensure_concepts_loaded(rag, course_id_str, enrich=False)
        if not concepts:
            return []

        # 2. Charger mastery scores en bulk + propagation KG (backward)
        # Au lieu de juste les scores directs, on prend get_scores_bulk_with_propagation
        # qui boost les ideas qu'on n'a pas testees mais dont les dependants
        # sont mastered. → recommandations plus intelligentes : on skippe
        # des concepts implicitement valides par leurs dependants.
        mastery_map: dict[str, float] = {}
        if sid:
            try:
                from pedagogy.mastery_repo import MasteryRepo
                # Collecter tous les idea_ids des concepts du cours
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
            """Mastery d'un concept = moyenne des mastery de ses idea_ids
            (incluant la propagation KG backward via get_scores_bulk_with_propagation).
            Si aucune idea n'a de mastery enregistree (eleve neuf) → 0.0.
            """
            if not c.idea_ids:
                return 0.0
            scores = [mastery_map.get(iid, 0.0) for iid in c.idea_ids]
            return sum(scores) / len(scores) if scores else 0.0

        # 3. Build candidates
        candidates: list[Recommendation] = []
        for c in concepts:
            ms = concept_mastery(c)
            if ms >= MASTERY_THRESHOLD:
                continue  # déjà maîtrisé

            # Prereqs derives semantiquement du graphe d'idees
            prereqs = kg.prereq_concepts(c.name)
            prereq_labels = [p.name for p in prereqs]

            prereqs_met = True
            weak_prereqs: list[str] = []
            for p in prereqs:
                p_score = concept_mastery(p)
                if p_score < MASTERY_THRESHOLD:
                    prereqs_met = False
                    weak_prereqs.append(p.name)

            rationale_parts = []
            if not prereqs:
                rationale_parts.append("Pas de prerequis identifié")
            elif prereqs_met:
                rationale_parts.append(f"{len(prereqs)} prerequis satisfaits")
            else:
                rationale_parts.append(
                    f"⚠️ Prerequis manquants : {', '.join(weak_prereqs[:3])}"
                )

            if current_chapter_idx is not None and c.chapter_idxs:
                proximity = min(abs(int(ci) - current_chapter_idx) for ci in c.chapter_idxs)
                rationale_parts.append(f"Distance chapitre actuelle : {proximity}")

            candidates.append(Recommendation(
                concept_id=c.name,
                label=c.name,
                name=c.display_name or c.name,
                canonical_name=c.canonical_name or "",
                description=c.description or "",
                bloom_level=c.bloom_level or "",
                chapter_idxs=sorted(c.chapter_idxs),
                score=float(c.score),
                mastery_score=ms,
                prereqs_met=prereqs_met,
                prereq_labels=prereq_labels[:5],
                rationale=" • ".join(rationale_parts),
            ))

        # 4. Sort multi-critères :
        #    1) prereqs_met=True d'abord
        #    2) chapter proximity ascending
        #    3) bloom difficulty ascending
        #    4) importance score descending
        def sort_key(r: Recommendation):
            bloom_diff = BLOOM_DIFFICULTY.get(r.bloom_level, 3)
            chapter_dist = (
                min(abs(int(ci) - current_chapter_idx) for ci in r.chapter_idxs)
                if (current_chapter_idx is not None and r.chapter_idxs) else 0
            )
            return (
                not r.prereqs_met,   # False (= prereqs OK) en premier
                chapter_dist,
                bloom_diff,
                -r.score,
            )

        candidates.sort(key=sort_key)
        return candidates[:top_k]
