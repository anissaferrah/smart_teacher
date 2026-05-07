"""Typed metadata pour les Documents RAG et les chunks agentic.

Pourquoi TypedDict (et pas Pydantic) :
  - Pydantic forcerait a migrer ~50 call sites `meta.get(...)` en assignations
    `model.field` (risque + temps).
  - TypedDict donne les memes garanties cote IDE / mypy / pyright (auto-complete,
    detection de fautes de frappe) avec **zero cout runtime** et aucune
    refactor breaking.
  - Pour reposer sur des validations runtime, ajouter ulterieurement un
    `validate_metadata()` qui leve sur structure invalide — non bloquant
    aujourd'hui (tout marche en dict).

Usage :
    from rag.metadata import RAGDocumentMetadata, RetrievedChunk

    def my_func(meta: RAGDocumentMetadata) -> str:
        return meta.get("idea_id") or ""        # autocomplete + typo detection

    # Dans une assignation explicite :
    meta: RAGDocumentMetadata = {
        "source_file": "x.pdf",
        "chunk_idx":   0,
        ...
    }
"""
from __future__ import annotations

from typing import TypedDict, NotRequired, Optional


class RAGDocumentMetadata(TypedDict, total=False):
    """Metadata typee pour ``langchain_core.documents.Document.metadata``.

    `total=False` : tous les champs sont optionnels au niveau type-checker
    (la realite : le pipeline d'ingestion garantit la presence des essentiels
    mais des Documents legacy peuvent en manquer). Les NotRequired explicites
    documentent ceux qu'on ajoute a posteriori (retrieval-time).
    """
    # ── Source / position ───────────────────────────────────────────────
    source_file: str                # chemin du PDF source
    chunk_idx: int                  # ordre dans la passe d'ingestion
    idea_index_in_chunk: int        # ordre de l'idee dans le title-chunk parent
    domain: str                     # ex: "informatique"
    course: str                     # course_id (UUID-stringified)
    language: str                   # "fr" | "en" | ...
    slide_idx: int                  # 1-based page number (PDF) ou index slide PPTX
    image_url: str                  # path PNG slide ou ""

    # ── Contenu et hash ─────────────────────────────────────────────────
    original_text: str              # texte brut avant idea-chunking (capped 500)
    content_hash: str               # md5[:8] du texte de l'idee

    # ── Structure pedagogique (TOC) ─────────────────────────────────────
    chapter_idx: int
    chapter_title: str
    section_idx: int
    section_title: str

    # ── Knowledge graph (idea-level) ────────────────────────────────────
    idea_id: str                    # md5[:12] global stable, primary key dans IdeaGraph
    idea_label: str                 # definition|theorem|procedure|example|warning|note|fragment
    depends_on_ids: list[str]       # prereqs (idea_ids)
    illustrates_id: Optional[str]   # concept illustre (idea_id|None)

    # ── Ajoutes a retrieval-time (pas a l'ingestion) ────────────────────
    _vector_score: NotRequired[float]    # cosine score Qdrant


class RetrievedChunk(TypedDict, total=False):
    """Chunk dict format produit par RetrieverAgent / ContextAgent et consomme
    par responder / narrator.

    Plus pauvre que `RAGDocumentMetadata` (deja deserialise) : copie d'un
    sous-ensemble + champs runtime (mastery, seen, kg augmentation flags).
    """
    # Contenu
    content: str
    source: str
    score: float                    # RAG raw score (cosine ou rerank ou 0.5 si KG-augmented)

    # Position dans le cours
    chapter: Optional[int]
    chapter_title: NotRequired[str]
    section_idx: Optional[int]
    section_title: str

    # KG metadata
    idea_id: Optional[str]
    idea_label: str

    # Memory layer (mastery + seen, ajoutes par RetrieverAgent)
    mastery_score: NotRequired[Optional[float]]
    seen: NotRequired[bool]

    # KG augmentation flags (presents iff le chunk vient du graph expansion)
    _via_kg: NotRequired[bool]
    _kg_relation: NotRequired[str]            # "prereq" | "example" | "illustrated"
    _augmented_from: NotRequired[str]         # idea_id du chunk source (top-1)


# ── Validation runtime optionnelle ──────────────────────────────────────
# Exposee pour le code qui veut une garantie forte (ex: API publique).
# A NE PAS appeler dans le hot path retrieval — TypedDict-only suffit la.

_REQUIRED_DOC_KEYS = ("source_file", "course", "chapter_idx", "section_idx",
                      "idea_id", "idea_label")


def validate_document_metadata(meta: dict) -> list[str]:
    """Verifie qu'un dict contient les champs essentiels d'un RAGDocumentMetadata.

    Returns la liste des cles manquantes. Liste vide = valide.
    Usage : seulement aux frontieres de confiance (API publique, persistance).
    """
    if not isinstance(meta, dict):
        return list(_REQUIRED_DOC_KEYS)
    return [k for k in _REQUIRED_DOC_KEYS if k not in meta]
