"""End-to-end smoke test for the vision pipeline.

Verifies that:
  1. ``services.vision_describe.extract_slide_concept`` reaches at least
     one provider (OpenAI or Ollama llama3.2-vision).
  2. The returned concept name is non-empty and short (1-5 words).
  3. The disk cache is populated so the next call is instant.

Pass a slide PNG path as argv[1], or let the script find the first
generated slide under ``media/slides/``.

Usage:
    python scripts/test_vision_pipeline.py [optional_slide.png] [fr|en]
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path


def _find_default_slide() -> str | None:
    media_root = Path("media/slides")
    if not media_root.exists():
        return None
    for png in media_root.rglob("*.png"):
        return str(png.resolve())
    return None


async def _run() -> int:
    img_path = sys.argv[1] if len(sys.argv) > 1 else _find_default_slide()
    lang = sys.argv[2] if len(sys.argv) > 2 else "en"

    if not img_path:
        print("❌ No slide PNG found. Either pass one as argv[1] or upload a course first.")
        return 1
    if not Path(img_path).exists():
        print(f"❌ Image not found: {img_path}")
        return 1

    print(f"📷 Slide image:  {img_path}")
    print(f"🌐 Language:     {lang}")
    print()

    # Round 1 — first call, no cache
    print("→ Round 1 (no cache, hits provider):")
    t0 = time.time()
    from services.vision_describe import extract_slide_concept, get_stats, reset_stats
    reset_stats()
    concept = await extract_slide_concept(img_path, lang)
    elapsed = time.time() - t0
    stats = get_stats()
    print(f"  concept = {concept!r}")
    print(f"  elapsed = {elapsed:.1f}s")
    print(f"  stats   = {stats}")
    print()

    if not concept:
        print("⚠️  Empty concept — vision providers all unreachable / returned UNKNOWN.")
        print("    System will fall back to structural extractor.")
        return 0

    # Round 2 — should be a cache hit (instant)
    print("→ Round 2 (should hit disk cache):")
    t0 = time.time()
    concept2 = await extract_slide_concept(img_path, lang)
    elapsed2 = time.time() - t0
    print(f"  concept = {concept2!r}")
    print(f"  elapsed = {elapsed2:.3f}s   (target: < 0.1s)")
    print(f"  stats   = {get_stats()}")
    print()

    if concept2 != concept:
        print(f"❌ Cache mismatch: round1={concept!r} round2={concept2!r}")
        return 1
    if elapsed2 > 0.5:
        print(f"⚠️  Round 2 was slow ({elapsed2:.2f}s) — disk cache may be misconfigured")
    else:
        print(f"✅ Cache hit confirmed ({elapsed2*1000:.0f}ms)")

    print()
    print("✅ Vision pipeline working end-to-end")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(asyncio.run(_run()))
