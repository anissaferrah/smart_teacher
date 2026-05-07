"""E2E test pour les 4 fixes ingestion + KG augmentation.

Lance sur un PDF reel : courses/informatique/data_mining/Chapter 1.pdf.

Mesure chaque fix individuellement avec des donnees reelles. Pour les fixes
qui dependent d'un LLM, utilise un stub avec sleep simule pour mesurer la
parallelisation isolement (sans subir la lenteur de Mistral CPU).

Output : tableau de metrics + verdict par fix.

Usage : python scripts/e2e_ingestion_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

# Force UTF-8 stdout/stderr (Windows cp1252 by default casse les box-drawing)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Project root sur sys.path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))


PDF_PATH = _ROOT / "courses" / "informatique" / "data_mining" / "Chapter 1.pdf"


def _hr(title: str) -> None:
    print()
    print("═" * 70)
    print(f"  {title}")
    print("═" * 70)


def _row(name: str, value: str) -> None:
    print(f"  {name:.<45s} {value}")


# ════════════════════════════════════════════════════════════════════
# Fix #5 — bbox images
# ════════════════════════════════════════════════════════════════════

def test_bbox_extraction() -> dict:
    _hr("Fix #5 — bbox images via PyMuPDF")
    import tempfile
    from pedagogy.intelligent_ingester import IntelligentIngester

    ingester = IntelligentIngester(enable_vision_llm=False)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "images"
        out_dir.mkdir()
        t0 = time.time()
        n_total, assets = ingester._extract_and_ocr_images(str(PDF_PATH), out_dir)
        dt = time.time() - t0

    n_with_bbox = sum(1 for a in assets if a.bbox is not None)
    extractor_used = assets[0].metadata.get("extractor", "unknown") if assets else "n/a"

    _row("PDF", PDF_PATH.name)
    _row("Total images extracted", str(n_total))
    _row("Assets with bbox", f"{n_with_bbox}/{len(assets)}")
    _row("Extractor used", extractor_used)
    _row("Time", f"{dt:.2f}s")

    if assets and n_with_bbox > 0:
        sample = next(a for a in assets if a.bbox)
        bbox = sample.bbox
        _row("  sample bbox (page %d)" % sample.page_num,
             f"x0={bbox['x0']:.0f} y0={bbox['y0']:.0f} "
             f"x1={bbox['x1']:.0f} y1={bbox['y1']:.0f}")

    verdict = "✅ PASS" if (n_with_bbox == len(assets) and extractor_used == "pymupdf") \
              else ("⚠️ DEGRADED (pypdf fallback)" if extractor_used == "pypdf" else "❌ FAIL")
    _row("VERDICT", verdict)
    return {"n_total": n_total, "n_bbox": n_with_bbox, "extractor": extractor_used,
            "time_s": dt, "verdict": verdict}


# ════════════════════════════════════════════════════════════════════
# Fix #2 — page fusion (sections courtes mergees)
# ════════════════════════════════════════════════════════════════════

def test_page_fusion() -> dict:
    _hr("Fix #2 — page fusion sur sections reelles du PDF")
    from rag.multimodal_rag import MultiModalRAG

    rag = MultiModalRAG.__new__(MultiModalRAG)

    # Extract pages reelles puis split en pseudo-sections (1 par page)
    # comme le course_builder produit en gros — pour exercer la fusion
    import pypdf, warnings
    warnings.filterwarnings("ignore")
    with open(PDF_PATH, "rb") as f:
        reader = pypdf.PdfReader(f, strict=False)
        pages = [(p.extract_text() or "").strip() for p in reader.pages]

    # 1 section par page
    sections = [{"content": p, "page_index": i + 1} for i, p in enumerate(pages) if p]

    n_before = len(sections)
    short_before = sum(1 for s in sections if len(s["content"]) < 300)

    fused = rag._fuse_short_sections(sections)
    n_after = len(fused)
    short_after = sum(1 for s in fused if len(s["content"]) < 300)
    fused_count = sum(1 for s in fused if "fused_pages" in s)

    _row("PDF pages", str(len(pages)))
    _row("Sections avant fusion", str(n_before))
    _row("  dont < 300 chars (cibles)", str(short_before))
    _row("Sections apres fusion", str(n_after))
    _row("  dont < 300 chars restantes", str(short_after))
    _row("Sections marquees fused_pages", str(fused_count))

    # Critere : doit avoir reduit le nombre de sections courtes ET total
    verdict = ("✅ PASS — fusion active"
               if (n_after < n_before or fused_count > 0)
               else "⚠️ NO-OP — toutes sections deja >= 300 chars")
    _row("VERDICT", verdict)
    return {"n_before": n_before, "n_after": n_after, "fused": fused_count,
            "verdict": verdict}


# ════════════════════════════════════════════════════════════════════
# Fix #3 — parallelisation idea-chunking (mesure isolee LLM stub)
# ════════════════════════════════════════════════════════════════════

def test_parallel_speedup() -> dict:
    _hr("Fix #3 — speedup parallelisation _chunk_by_ideas_batch (stub LLM)")
    from rag.multimodal_rag import MultiModalRAG

    rag = MultiModalRAG.__new__(MultiModalRAG)

    # 12 sections "fake" pour mesurer — chaque appel LLM simule = 0.5s
    items = [(f"section text {i} " * 50, f"chapter {i//4}", f"section {i}")
             for i in range(12)]
    SIMULATED_LATENCY = 0.5

    def stub_chunk(text, chapter_title, section_title):
        time.sleep(SIMULATED_LATENCY)  # simule appel LLM
        return [{"idea": text, "label": "fragment", "local_id": "f_1",
                 "depends_on": [], "illustrates": None}]

    # Sequential (parallelism=1)
    with patch.object(rag, "_chunk_by_ideas", side_effect=stub_chunk), \
         patch("rag.multimodal_rag.Config.RAG_LLM_PARALLELISM", 1):
        t0 = time.time()
        results_seq = rag._chunk_by_ideas_batch(items)
        t_seq = time.time() - t0

    # Parallel (parallelism=6)
    with patch.object(rag, "_chunk_by_ideas", side_effect=stub_chunk), \
         patch("rag.multimodal_rag.Config.RAG_LLM_PARALLELISM", 6):
        t0 = time.time()
        results_par = rag._chunk_by_ideas_batch(items)
        t_par = time.time() - t0

    speedup = t_seq / t_par if t_par > 0 else float("inf")

    _row("Items (sections)", str(len(items)))
    _row("Latency simulee/call", f"{SIMULATED_LATENCY}s")
    _row("Sequential (P=1)", f"{t_seq:.2f}s")
    _row("Parallel   (P=6)", f"{t_par:.2f}s")
    _row("Speedup", f"{speedup:.2f}×")
    _row("Order preserved", str(all(
        results_seq[i][0]["idea"] == results_par[i][0]["idea"]
        for i in range(len(items))
    )))

    verdict = "✅ PASS" if speedup >= 3.0 else "⚠️ WEAK (<3×)"
    _row("VERDICT", verdict)
    return {"t_seq": t_seq, "t_par": t_par, "speedup": speedup, "verdict": verdict}


# ════════════════════════════════════════════════════════════════════
# KG augmentation Phase 1 — sur graphe synthetique mais realiste
# ════════════════════════════════════════════════════════════════════

def test_kg_augmentation() -> dict:
    _hr("Phase 1 — KG augmentation (prereqs + examples + illustrated)")
    from agentic.qa.retriever import RetrieverAgent
    from pedagogy.knowledge_graph.graph import IdeaGraph, IdeaNode

    # Graphe : 1 concept central (k-means), 2 prereqs (centroid, distance),
    # 1 exemple (clustering iris dataset), 1 inverse-illustration test
    g = IdeaGraph()
    g.add_node(IdeaNode(idea_id="centroid_id", label="definition",
                       text="Centroid = barycenter of cluster",
                       course_id="dm", chapter_idx=2))
    g.add_node(IdeaNode(idea_id="distance_id", label="definition",
                       text="Euclidean distance metric",
                       course_id="dm", chapter_idx=2))
    g.add_node(IdeaNode(idea_id="kmeans_id", label="theorem",
                       text="K-means convergence theorem",
                       course_id="dm", chapter_idx=2,
                       depends_on_ids={"centroid_id", "distance_id"}))
    g.add_node(IdeaNode(idea_id="iris_example_id", label="example",
                       text="Iris dataset clustering with k=3",
                       course_id="dm", chapter_idx=2,
                       illustrates_id="kmeans_id"))

    # Cas 1 : top-1 = concept (k-means) → on attend prereqs + example
    agent = RetrieverAgent(rag=object())
    with patch("pedagogy.knowledge_graph.get_or_build", return_value=g):
        chunks = [{"idea_id": "kmeans_id", "content": "...", "score": 0.9}]
        aug1 = agent._augment_with_kg(chunks)

    # Cas 2 : top-1 = exemple (iris) → on attend illustrated (kmeans)
    with patch("pedagogy.knowledge_graph.get_or_build", return_value=g):
        chunks = [{"idea_id": "iris_example_id", "content": "...", "score": 0.9}]
        aug2 = agent._augment_with_kg(chunks)

    relations1 = sorted(a["_kg_relation"] for a in aug1)
    relations2 = [a["_kg_relation"] for a in aug2]

    _row("Cas 1: top-1 = k-means concept", "")
    _row("  augmented count", str(len(aug1)))
    _row("  relations", str(relations1))
    _row("  ids", str([a["idea_id"] for a in aug1]))
    _row("Cas 2: top-1 = iris example", "")
    _row("  augmented count", str(len(aug2)))
    _row("  relations", str(relations2))
    _row("  ids", str([a["idea_id"] for a in aug2]))

    ok1 = (sorted(relations1) == sorted(["prereq", "prereq", "example"]))
    ok2 = (relations2 == ["illustrated"]
           and aug2[0]["idea_id"] == "kmeans_id")

    verdict = "✅ PASS" if (ok1 and ok2) else "❌ FAIL"
    _row("VERDICT", verdict)
    return {"case1_relations": relations1, "case2_relations": relations2,
            "verdict": verdict}


# ════════════════════════════════════════════════════════════════════
# Bonus : sanity sur un VRAI appel Ollama (1 section, courte) pour
# verifier que toute l'integration LLMRouter fonctionne en bout-en-bout
# ════════════════════════════════════════════════════════════════════

def test_real_ollama_smoke() -> dict:
    _hr("Bonus — vrai appel Ollama via LLMRouter (sanity end-to-end)")
    from ai.llm_router import LLMRouter

    router = LLMRouter(ollama_model="mistral")
    prompt = "Reply with exactly the word OK and nothing else."

    t0 = time.time()
    out = router.invoke(prompt, prefer="ollama", temperature=0.0, max_tokens=10)
    dt = time.time() - t0

    snap = router.stats.snapshot()
    _row("Backend prefere", "ollama")
    _row("Response", str((out or "")[:80]))
    _row("Time", f"{dt:.2f}s")
    _row("Stats", str(snap))
    verdict = "✅ PASS" if out else "❌ FAIL"
    _row("VERDICT", verdict)
    return {"response": out, "time_s": dt, "stats": snap, "verdict": verdict}


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    if not PDF_PATH.exists():
        print(f"❌ PDF introuvable: {PDF_PATH}")
        sys.exit(1)

    results = {}
    results["bbox"]            = test_bbox_extraction()
    results["page_fusion"]     = test_page_fusion()
    results["parallel"]        = test_parallel_speedup()
    results["kg_augmentation"] = test_kg_augmentation()
    results["ollama_smoke"]    = test_real_ollama_smoke()

    _hr("RESUME GLOBAL")
    for name, r in results.items():
        _row(name, r.get("verdict", "?"))


if __name__ == "__main__":
    main()
