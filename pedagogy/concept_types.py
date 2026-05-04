"""Shared concept data types — used by all extraction methods.

Was previously in ``pedagogy/concept_extractor.py``, but that file mixed
the KeyBERT-specific extraction pipeline with the data class. After the
KeyBERT and clustering methods were removed (in favour of title-based
extraction), only ``Concept`` and ``_slugify`` remain shared.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any


def _slugify(text: str) -> str:
    """Convertit 'Vecteur Unitaire' → 'vecteur_unitaire'.

    The 64-char cap is operational, not aesthetic : downstream stores
    (Postgres VARCHAR, Redis keys, URLs) all benefit from a bounded
    length. Previously the cap was a raw ``s[:64]`` which sometimes
    cut a word in half and produced URL-unfriendly tails like
    ``..._recherche_d_`` (orphan ``_`` from ``d'information``). That
    ate ~5% of long titles and made the practice-engine lookup miss
    because the frontend / KG didn't agree on whether to keep the
    truncated half-word.

    Fix : when the cap falls inside a word, back up to the previous
    underscore and strip the trailing ``_``. That gives ``..._recherche``
    instead of ``..._recherche_d_``, which is a valid clean slug at
    the cost of dropping one (truncated) word.
    """
    s = re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s.strip())
    if len(s) <= 64:
        return s or "concept"
    truncated = s[:64]
    # If the cut landed inside a word, back up to the last word boundary.
    # We know the cut was inside a word when the next char in `s` exists
    # and isn't an underscore — i.e., we'd be removing the second half
    # of a token.
    if s[64:65] and s[64] != "_":
        last_underscore = truncated.rfind("_")
        if last_underscore > 0:
            truncated = truncated[:last_underscore]
    # Strip any trailing underscores left by the cut (the truncation
    # may have ended right after a separator: ``..._d_`` → ``..._d``).
    truncated = truncated.rstrip("_")
    return truncated or "concept"


@dataclass
class Concept:
    """Une idee atomique extraite. Layer 3 enrichi via LLM : description + bloom_level."""

    label: str               # snake_case unique id
    name: str                # Human readable ("Vecteur unitaire")
    keywords: list[str] = field(default_factory=list)  # phrases brutes extraites
    chunk_ids: set[str] = field(default_factory=set)   # idea_ids RAG ou indices
    chapter_idxs: set[int] = field(default_factory=set)
    score: float = 0.0       # importance — taille du cluster ou somme de scores
    embedding: Any | None = None
    # Layer 3 LLM enrichment (optionnel)
    canonical_name: str = ""     # nom canonique remplace par LLM (ex: "PCA")
    description: str = ""        # 1-phrase explication
    bloom_level: str = ""        # remember | understand | apply | analyze

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)
        d["chunk_ids"] = list(self.chunk_ids)
        d["chapter_idxs"] = sorted(self.chapter_idxs)
        return d

    def to_concept_info(self, course_id: str = ""):
        """Convertit ce Concept en ConceptInfo (couche publique du
        KnowledgeGraph). Le ``Concept`` reste interne au pipeline d'extraction ;
        ``ConceptInfo`` est ce que les consumers (path_recommender,
        skill_tree, frontend) voient.
        """
        from pedagogy.knowledge_graph.graph import ConceptInfo
        return ConceptInfo(
            name=self.label,
            display_name=self.name,
            canonical_name=self.canonical_name,
            description=self.description,
            bloom_level=self.bloom_level,
            course_id=course_id,
            chapter_idxs=set(self.chapter_idxs),
            score=self.score,
            idea_ids=set(self.chunk_ids),
        )
