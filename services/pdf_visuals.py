"""Per-slide visual content detection from PDF source.

# Why

The text-based gate in [services/vision_describe.should_describe_slide]
inspects OCR-extracted text for math symbols, formula markers, visual
keywords. It misses slides whose visual content is rendered as embedded
images at PDF level — most notably math formulas (σ, Σ, √, …) which
slide-deck tools export as inline PNG glyphs rather than as font text.
The OCR text on such a slide reads only the surrounding labels
("Variance", "Standard deviation"), the gate sees no triggers, and the
slide is wrongly classified text-only — vision is never called and the
QA tutor literally cannot see the formula.

This module bypasses the OCR limitation by looking at the PDF directly:

  1. Embedded raster images (`page.get_images`) — formulas, diagrams,
     screenshots, photos.
  2. Vector drawings (`page.get_drawings`) — chart shapes, schemas,
     hand-drawn boxes.

Result is a per-slide ``has_visuals`` boolean cached to disk by image
MD5, alongside the existing vision_descriptions cache.

# Template-image filter

Slide decks repeat background imagery on every page (orange waves, page
number boxes, logos). Without filtering those out we'd flag every slide
as "has visuals" — defeating the gate's purpose. The template detector
counts how many pages each image XREF appears on; an XREF that appears
on ≥ ``_TEMPLATE_PAGE_FRACTION`` of pages is treated as template and
ignored when scoring per-page content.

# Performance

Runs once per ingested deck. ~50ms per page on a typical CPU. Output
cached forever (per image MD5), so subsequent reads are O(file open).
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from core.config import Config

log = logging.getLogger("services.pdf_visuals")


# ── Cache layout ──────────────────────────────────────────────────────
# Reuse the vision_descriptions cache directory so all per-image
# sidecars live in one place. Suffix ``_visuals.txt`` distinguishes them
# from descriptions (``_<lang>.txt``) and concepts (``_<lang>.concept.txt``).
_CACHE_DIR: Path = Path(getattr(Config, "LOGS_DIR", "logs")).parent / "cache" / "vision_descriptions"
try:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception as exc:                                                # noqa: BLE001
    log.warning("Could not create pdf_visuals cache dir %s: %s", _CACHE_DIR, exc)


# ── Tunables ──────────────────────────────────────────────────────────

# An image XREF that appears on ≥ this fraction of pages is template
# (background, page numbers, logos) — ignored when scoring content.
_TEMPLATE_PAGE_FRACTION: float = 0.5

# A page with strictly more vector-drawing paths than this is considered
# visually rich. Calibrated empirically on a templated deck (Chapter 1
# of the data-mining course):
#   - text-only slides (definitions, bullet lists): 7–12 paths (template only)
#   - slides with 1–3 colored boxes / simple flows:  19–29 paths
#   - slides with pyramids, multi-box diagrams:      33–52 paths
#   - timelines, dense schemas:                      170+ paths
# 15 separates real text-only from anything with content shapes — what
# the gate cares about. Raise if false positives appear on text-heavy
# decks (every page header decoration counts as 5–10 paths).
_DRAWING_THRESHOLD: int = 15


# ── MD5 helper ────────────────────────────────────────────────────────

def _md5_image(image_path: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Cache I/O ─────────────────────────────────────────────────────────

def _flag_path(md5: str) -> Path:
    return _CACHE_DIR / f"{md5}_visuals.txt"


def read_visuals_flag(md5: str) -> Optional[bool]:
    """Return cached has_visuals for an image MD5.

    ``True``  → PDF analysis flagged this slide as having visual content.
    ``False`` → PDF analysis explicitly classified text-only.
    ``None``  → no flag computed (legacy slide, ingestion before this
                module existed, or computation failed).
    """
    p = _flag_path(md5)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").strip() == "1"
    except Exception:                                                   # noqa: BLE001
        return None


def _write_flag(md5: str, has_visuals: bool) -> None:
    try:
        _flag_path(md5).write_text("1" if has_visuals else "0", encoding="utf-8")
    except Exception as exc:                                            # noqa: BLE001
        log.debug("write visuals flag failed for %s: %s", md5, exc)


# ── Detection ─────────────────────────────────────────────────────────

def _identify_template_xrefs(doc) -> set[int]:
    """Image XREFs that appear on ≥ _TEMPLATE_PAGE_FRACTION of pages."""
    xref_pages: Counter = Counter()
    n_pages = doc.page_count
    if n_pages == 0:
        return set()
    for page_idx in range(n_pages):
        try:
            for img_info in doc[page_idx].get_images(full=True):
                xref_pages[img_info[0]] += 1
        except Exception:                                               # noqa: BLE001
            continue
    threshold = max(2, int(n_pages * _TEMPLATE_PAGE_FRACTION))
    return {xref for xref, count in xref_pages.items() if count >= threshold}


def _page_has_content_visuals(page, template_xrefs: set[int]) -> tuple[bool, str]:
    """Return (has_visuals, reason) for a single page."""
    # 1. Non-template embedded raster images
    try:
        for img_info in page.get_images(full=True):
            if img_info[0] not in template_xrefs:
                return True, f"content_image(xref={img_info[0]})"
    except Exception:                                                   # noqa: BLE001
        pass
    # 2. Many vector drawings (charts, diagrams, hand-drawn schemas)
    try:
        n_drawings = len(page.get_drawings())
        if n_drawings > _DRAWING_THRESHOLD:
            return True, f"drawings({n_drawings})"
    except Exception:                                                   # noqa: BLE001
        pass
    return False, "text_only"


# ── Public entry point (called from the ingester) ─────────────────────

def precompute_slide_visuals(pdf_path: str, slide_pngs: list) -> dict:
    """Compute and cache the has_visuals flag for every slide in a deck.

    Args:
        pdf_path: source PDF path
        slide_pngs: ordered list of rendered slide PNG paths (1 per page).
                    The PNG MD5 is the cache key — must match what the
                    vision pipeline will use later.

    Returns:
        Mapping of 0-based page index → has_visuals bool. Side effect:
        writes one ``{md5}_visuals.txt`` per slide to the cache dir so
        ``services.vision_describe.describe_slide_image`` can read it
        without needing the PDF.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.warning("pymupdf not installed → cannot precompute slide visuals")
        return {}

    result: dict = {}
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:                                            # noqa: BLE001
        log.warning("fitz.open failed for %s: %s", pdf_path, exc)
        return {}

    try:
        template_xrefs = _identify_template_xrefs(doc)
        n_total = min(doc.page_count, len(slide_pngs))
        for page_idx in range(n_total):
            try:
                page = doc[page_idx]
                has_v, reason = _page_has_content_visuals(page, template_xrefs)
            except Exception as exc:                                    # noqa: BLE001
                log.debug("page %d analysis failed: %s", page_idx + 1, exc)
                has_v, reason = False, "analysis_error"
            result[page_idx] = has_v
            try:
                md5 = _md5_image(slide_pngs[page_idx])
                _write_flag(md5, has_v)
            except Exception as exc:                                    # noqa: BLE001
                log.debug("md5/write failed for page %d: %s", page_idx + 1, exc)
        n_visual = sum(1 for v in result.values() if v)
        log.info(
            "📐 pdf_visuals: %d/%d slides flagged as visual-content "
            "(template_xrefs=%d)", n_visual, n_total, len(template_xrefs),
        )
    finally:
        try:
            doc.close()
        except Exception:                                               # noqa: BLE001
            pass

    return result
