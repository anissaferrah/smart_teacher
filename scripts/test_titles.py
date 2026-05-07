"""Quick test of CONCEPT_EXTRACTION_METHOD=titles.

Usage: python -m scripts.test_titles <course_id>
"""
from __future__ import annotations

import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")

from rag.multimodal_rag import MultiModalRAG
import deps


def main(course_id: str) -> None:
    print(f"\n{'='*70}")
    print(f"Loading RAG from data/multimodal_db ...")
    print(f"{'='*70}")
    rag = MultiModalRAG(db_dir="data/multimodal_db")
    deps.register_services(rag=rag)

    n_docs = len(rag.all_docs or [])
    course_docs = [
        d for d in (rag.all_docs or [])
        if (getattr(d, "metadata", None) or {}).get("course") == course_id
    ]
    print(f"Total docs: {n_docs}  |  course docs: {len(course_docs)}")

    if not course_docs:
        print("No docs for this course.")
        return

    print(f"\n{'='*70}")
    print(f"Test : CONCEPT_EXTRACTION_METHOD=titles")
    print(f"{'='*70}")

    from pedagogy.concept_from_titles import ConceptFromTitles
    extractor = ConceptFromTitles(rag.embeddings)

    t0 = time.time()
    concepts = extractor.extract(
        documents=rag.all_docs,
        course_id=course_id,
        max_concepts=50,
        lang="en",
    )
    elapsed = time.time() - t0

    print(f"\n--> {len(concepts)} concepts extracted in {elapsed:.1f}s\n")
    print(f"{'idx':>3}  {'slug':<35}  {'name':<40}  {'#chunks':>7}  {'bloom':<11}")
    print("-" * 105)
    for i, c in enumerate(concepts):
        slug = (c.label or "")[:33]
        name = (c.name or "")[:38]
        n_chunks = len(c.chunk_ids)
        bloom = (c.bloom_level or "")[:11]
        print(f"{i:>3}  {slug:<35}  {name:<40}  {n_chunks:>7}  {bloom:<11}")

    if concepts:
        print(f"\nDescriptions:")
        for c in concepts[:10]:
            desc = (c.description or "")
            if desc:
                print(f"  - {c.label}:")
                print(f"      {desc}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_titles <course_id>")
        sys.exit(1)
    main(sys.argv[1])
