"""Lazy loader pour la couche concept du KnowledgeGraph.

Avant : la couche concept etait peuplee uniquement quand quelqu'un appelait
`/course/{id}/concept-graph`. Si un consumer (path_recommender, skill_tree,
practice_engine) etait sollicite avant cet appel, il retournait `[]` —
mauvaise UX.

Maintenant : les consumers appellent `ensure_concepts_loaded(rag, course_id)`
qui :
  1. Retourne immediatement si le KG a deja des concepts pour ce cours
  2. Sinon extrait via KeyBERT (rapide, ~1-3s) et attache au KG
  3. Optionnellement enrichit (lent, +5-15s si Ollama)

Le `/concept-graph` endpoint appelle avec enrich=True (full quality).
Les consumers internes appellent avec enrich=False (fast path).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("pedagogy.knowledge_graph.concepts_loader")


_concept_extractor_singleton: Any = None


def _get_extractor():
    """Lazy init du ``ConceptFromTitles`` (extraction par section_title +
    enrichissement LLM). Singleton pour éviter le coût d'init à chaque appel.
    """
    global _concept_extractor_singleton
    if _concept_extractor_singleton is not None:
        return _concept_extractor_singleton
    from deps import get_rag
    rag = get_rag()
    if rag is None or not getattr(rag, "embeddings", None):
        raise RuntimeError("RAG embeddings unavailable")
    from pedagogy.concept_from_titles import ConceptFromTitles
    _concept_extractor_singleton = ConceptFromTitles(rag.embeddings)
    return _concept_extractor_singleton


def ensure_concepts_loaded(
    rag: Any,
    course_id: str,
    max_concepts: int = 50,
    enrich: bool = False,
) -> list:
    """Garantit que les concepts du cours sont attaches au KG.

    Returns la liste des `ConceptInfo` du cours (apres extraction si necessaire).
    Liste vide si le RAG est vide ou si l'extraction echoue (graceful).

    Args:
        rag:           instance RAG
        course_id:     id du cours
        max_concepts:  cap KeyBERT (default 50)
        enrich:        si True, ajoute description + bloom_level via LLM
                       (lent, 5-15s sur Ollama). Default False = fast path.
    """
    from pedagogy.knowledge_graph import get_or_build
    from pedagogy.knowledge_graph import persistence

    if not course_id or rag is None:
        return []

    kg = get_or_build(rag)

    # Fast path #1 : déjà attaché en mémoire (process déjà chaud)
    existing = kg.list_concepts(course_id)
    if existing:
        return existing

    # Pas d'idees ingerees pour ce cours → rien a extraire
    course_docs = [
        d for d in (getattr(rag, "all_docs", None) or [])
        if (getattr(d, "metadata", None) or {}).get("course") == course_id
    ]
    if not course_docs:
        log.debug(f"ensure_concepts_loaded: no docs for course={course_id[:16]}")
        return []

    n_docs = len(course_docs)

    # Fast path #2 : cache disque (concepts persistés depuis un run précédent).
    # Évite de relancer 5-15 min d'extraction LLM au démarrage du serveur.
    cached = persistence.load_concepts(course_id, n_docs=n_docs)
    if cached:
        existing_others = [
            ci for ci in kg.list_concepts() if ci.course_id != course_id
        ]
        kg.attach_concepts(existing_others + cached)
        return kg.list_concepts(course_id)

    log.info(
        f"🧠 Lazy concept extraction for course={course_id[:16]} "
        f"(enrich={enrich}, n_docs={n_docs})"
    )

    try:
        extractor = _get_extractor()
    except Exception as exc:
        log.warning(f"ensure_concepts_loaded: extractor unavailable ({exc})")
        return []

    sample_lang = next(
        (d.metadata.get("language", "fr") for d in course_docs),
        "fr",
    )

    try:
        concepts = extractor.extract(
            documents=rag.all_docs,
            course_id=course_id,
            max_concepts=max_concepts,
            lang=sample_lang[:2],
        )
        if not concepts:
            return []

        # Convert + attach to KG (preserve concepts d'autres cours)
        concept_infos = [c.to_concept_info(course_id=course_id) for c in concepts]
        existing_others = [
            ci for ci in kg.list_concepts() if ci.course_id != course_id
        ]
        kg.attach_concepts(existing_others + concept_infos)

        # Persist for next process restart — skips LLM extraction next time.
        try:
            persistence.save_concepts(course_id, concept_infos, n_docs=n_docs)
        except Exception as exc:                                            # noqa: BLE001
            log.debug(f"persist concepts failed (non-fatal): {exc}")

        return kg.list_concepts(course_id)
    except Exception as exc:
        log.warning(f"ensure_concepts_loaded extraction failed: {exc}")
        return []
