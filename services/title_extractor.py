"""LLM-based slide title extraction from raw OCR content.

# Why a dedicated module

The structural extractor in ``agentic.teaching.planner._extract_main_concept``
relies on regex/heuristics over flat-extracted PDF text. It works on
some decks but fails on academic slides where:

  - Roman-numeral section headers ("I. CONTEXTE ET MOTIVATION", "VI.1.
    PERTINENCE UTILISATEUR") appear at the top of each page but are
    interleaved with sub-headers and body text that also look like
    titles.
  - Long titles wrap across two lines ("VI. SYSTEME DE RECHERCHE\n
    D'INFORMATION -SRI") and the structural extractor picks just one.
  - PDF extraction inserts spurious whitespace ("RECHERCHE     D'INFORMATION").
  - Sub-section bullets ("Bibliothèques numériques", "RECHERCHE ADHOC")
    win over the actual page title.

A small LLM call ("here is the slide text — what is the title?") solves
all of these in one pass. The prompt is one-shot, deterministic
(temperature=0), and capped to a few tokens — so it's cheap on Ollama
(~3-5s on CPU mistral) and cached on disk by content hash to avoid
re-paying on re-ingestions.

# Routing

Calls go through :class:`ai.llm_router.LLMRouter` with ``prefer="openai"``.
When ``DISABLE_OPENAI=true`` (Config), the router skips OpenAI at
construction and falls straight to Ollama. So with the kill-switch on,
this module's only LLM cost is the local Mistral call.

# Cache

Disk-cached at ``cache/slide_titles/{md5_of_content}_{lang}.txt``.
The MD5 is over the *normalised* content (whitespace collapsed) so
re-ingestions of the same PDF hit the cache even when pypdf produces
slightly different whitespace between runs. Cache entries never expire
on their own — clear them via ``scripts/reset_for_reingest.py``.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

from core.config import Config

log = logging.getLogger("services.title_extractor")


# ── Cache directory : same parent as vision_describe so a single
# `cache/` cleanup wipes all LLM-derived disk artifacts at once.
_CACHE_DIR = Path(getattr(Config, "LOGS_DIR", "logs")).parent / "cache" / "slide_titles"
try:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception as exc:                                                    # noqa: BLE001
    log.warning("Could not create slide-titles cache dir %s: %s", _CACHE_DIR, exc)


# ── Operational caps ────────────────────────────────────────────────────
# Slide content fed to the LLM. 1500 chars is enough to surface the
# title (typically lines 1-3) without ballooning the prompt — Mistral
# CPU latency scales near-linearly with prompt length.
_CONTENT_CAP = 1500
# Output cap. Titles are 1-12 words ; 80 tokens leaves headroom for
# the model that occasionally inserts a leading "Title:" or quotes.
_MAX_OUTPUT_TOKENS = 80
# Minimum content length below which we skip the LLM and return "" —
# a 30-char slide doesn't carry a real title anyway, structural is
# faster and just as good.
_MIN_CONTENT_CHARS = 30


# ── Prompts (FR/EN, deterministic) ──────────────────────────────────────

_PROMPT_FR = """Tu reçois le texte OCR d'une slide de cours. Identifie son TITRE PRINCIPAL.

RÈGLES :
- Le titre est généralement la PREMIÈRE ou DEUXIÈME ligne en MAJUSCULES, ou commence par un numéro romain (I., II., VI.1., VII.2., etc.).
- Si le titre est sur 2 lignes (ex: "VI. SYSTEME DE RECHERCHE\\nD'INFORMATION -SRI"), CONCATÈNE-les avec un espace.
- IGNORE l'en-tête répété "INTRODUCTION À LA RI" (ou similaire) qui apparaît sur chaque slide — ce n'est PAS le titre.
- IGNORE les sous-titres / bullets (ex: "Bibliothèques numériques", "RECHERCHE ADHOC", "Web", "Entreprises") quand un titre principal existe.
- IGNORE le numéro de page en bas.
- Préserve la casse et la ponctuation du titre original. Collapse les espaces multiples en un seul.
- Réponds UNIQUEMENT par le titre, sans guillemets, sans préambule, sans explication. Maximum 12 mots.

CONTENU DE LA SLIDE :
{content}

TITRE :"""


_PROMPT_EN = """You receive the OCR text of a lecture slide. Identify its MAIN TITLE.

RULES:
- The title is usually the FIRST or SECOND line in CAPS, or starts with a Roman numeral (I., II., VI.1., VII.2., etc.).
- If the title spans 2 lines (e.g. "VI. SYSTEME DE RECHERCHE\\nD'INFORMATION -SRI"), CONCATENATE them with a space.
- IGNORE repeated chapter headers (e.g. "INTRODUCTION TO IR") that appear on every slide — they are NOT the title.
- IGNORE sub-titles / bullets (e.g. "Digital Libraries", "AD-HOC SEARCH", "Web", "Enterprises") when a main title exists.
- IGNORE page numbers at the bottom.
- Preserve the case and punctuation of the original title. Collapse multiple spaces into one.
- Reply ONLY with the title, no quotes, no preamble, no explanation. Max 12 words.

SLIDE CONTENT:
{content}

TITLE:"""


# ── Helpers ─────────────────────────────────────────────────────────────

# Patterns for stripping running headers/footers/markers from slide
# content before sending it to the LLM title extractor. Without this,
# the LLM picks the running chapter header ("2. Data, Dataset, Data
# Warehouse") as the slide title on transition / diagram / example
# slides where the actual heading isn't the largest text feature.
_LINE_PAGE_NUM = re.compile(r"^\d{1,3}$")
_LINE_CHAPTER_HEADER = re.compile(r"^\d+\.\s+[A-Z]")
_LINE_TODO_MARKER = re.compile(r"^TODO$", re.IGNORECASE)


def _strip_running_headers(content: str, chapter_title: str | None = None) -> str:
    """Remove lines that consistently appear on every slide and would
    otherwise dominate the LLM's title-pick heuristic.

    Stripped:
      - lone page-number lines
      - chapter-header pattern lines (e.g. "2. Data, Dataset, Data Warehouse")
      - literal ``TODO`` placeholders
      - the literal ``chapter_title`` when the caller passes it (for
        chapters whose header doesn't match the numbered pattern)
    """
    if not content:
        return content

    chapter_lit = (chapter_title or "").strip().casefold()
    kept = []
    for ln in content.splitlines():
        stripped = ln.strip()
        if not stripped:
            kept.append(ln)
            continue
        if _LINE_PAGE_NUM.match(stripped):
            continue
        if _LINE_CHAPTER_HEADER.match(stripped):
            continue
        if _LINE_TODO_MARKER.match(stripped):
            continue
        if chapter_lit and stripped.casefold() == chapter_lit:
            continue
        kept.append(ln)
    return "\n".join(kept)


def _normalise_for_hash(content: str) -> str:
    """Collapse whitespace + lowercase, used only to compute the cache key.

    Two pypdf runs can produce slightly different whitespace; we want
    them to share a cache entry. The actual prompt sees the original
    content (preserving case and structure for the LLM).
    """
    return re.sub(r"\s+", " ", content or "").strip().lower()


def _content_hash(content: str) -> str:
    return hashlib.md5(_normalise_for_hash(content).encode("utf-8")).hexdigest()


def _cache_path(md5: str, lang: str) -> Path:
    return _CACHE_DIR / f"{md5}_{lang}.txt"


def _cache_read(md5: str, lang: str) -> Optional[str]:
    p = _cache_path(md5, lang)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").rstrip("\n")
    except Exception:
        return None


def _cache_write(md5: str, lang: str, title: str) -> None:
    try:
        _cache_path(md5, lang).write_text(title, encoding="utf-8")
    except Exception as exc:                                                # noqa: BLE001
        log.debug("title cache write failed: %s", exc)


def _normalise_title(raw: str) -> str:
    """Trim quotes, leading "Titre:"/"Title:", whitespace runs, period.

    The model occasionally prefixes "Titre : XXX" or wraps the title in
    quotes. Strip those without rejecting otherwise valid output.
    """
    s = (raw or "").strip()
    # Strip leading "Title:" / "Titre:" / "TITRE:" / "TITLE:" prefixes
    s = re.sub(r"^(?:titre|title)\s*:\s*", "", s, flags=re.IGNORECASE)
    # Strip wrapping quotes (single, double, French)
    s = s.strip().strip("\"'`«»“”").strip()
    # Strip trailing period / ellipsis
    s = s.rstrip(".!?:; ")
    # Collapse internal whitespace runs to one space
    s = re.sub(r"\s+", " ", s).strip()
    # Hard cap to 100 chars : titles longer than that mean the model
    # ignored the "max 12 words" instruction, treat as junk.
    return s[:100]


def _looks_like_junk(title: str) -> bool:
    """Reject obviously bad outputs without re-running the LLM.

    Triggers when:
      - empty after normalisation,
      - the model echoed a refusal ("je ne peux pas", "i cannot"),
      - the model returned the prompt header ("le titre est :"),
      - one-letter or numbers-only output.
    """
    if not title:
        return True
    low = title.lower()
    refusals = (
        "je ne peux pas", "je ne sais pas", "i cannot", "i don't know",
        "n/a", "none", "(none)", "unknown", "inconnu",
    )
    if any(r in low for r in refusals):
        return True
    if len(title) < 2:
        return True
    if title.isdigit():
        return True
    return False


# ── Public API ──────────────────────────────────────────────────────────

def extract_title_via_llm(content: str, language: str = "fr", chapter_title: str | None = None) -> str:
    """Synchronous LLM-based title extraction.

    Returns the title string, or ``""`` when:
      - content is too short to bother (< _MIN_CONTENT_CHARS),
      - both LLM backends failed,
      - the LLM returned a junk / refusal output.

    Cached on disk by content MD5. The router's own
    ``DISABLE_OPENAI`` handling means this is one Ollama call when the
    OpenAI kill-switch is on.

    ``chapter_title`` is optional; when provided, lines matching it
    literally are stripped along with the numbered chapter-header
    pattern. This prevents the LLM from picking the running chapter
    header as the slide title on transition/diagram slides.
    """
    if not content or len(content.strip()) < _MIN_CONTENT_CHARS:
        return ""

    # Strip running headers BEFORE hashing, so the cache key is stable
    # against decorative differences and old miscached titles
    # (extracted from un-stripped content) don't get returned.
    cleaned = _strip_running_headers(content, chapter_title=chapter_title)
    if len(cleaned.strip()) < _MIN_CONTENT_CHARS:
        return ""

    lang = (language or "fr")[:2].lower()
    md5 = _content_hash(cleaned)

    # Cache fast-path : skip the LLM entirely on a hit.
    cached = _cache_read(md5, lang)
    if cached is not None:
        return cached  # may be empty (we cache misses too, see below)

    # Build prompt with truncated content
    template = _PROMPT_FR if lang == "fr" else _PROMPT_EN
    prompt = template.format(content=cleaned[:_CONTENT_CAP])

    # Call via LLMRouter — single source of truth for OpenAI/Ollama
    # routing, DISABLE_OPENAI handling, error fallback. We prefer
    # OpenAI when available (better instruction-following on this kind
    # of task) but the router transparently falls back to Ollama.
    try:
        from ai.llm_router import get_default_router
        router = get_default_router()
        raw = router.invoke(prompt, prefer="openai", temperature=0.0,
                            max_tokens=_MAX_OUTPUT_TOKENS)
    except Exception as exc:                                                # noqa: BLE001
        log.info("title_extractor : LLM call raised %s — fallback to structural", exc)
        return ""

    if raw is None:
        log.info("title_extractor : both LLM backends failed → fallback to structural")
        # Don't cache misses : the next call might land on a healthy backend.
        return ""

    title = _normalise_title(raw)
    if _looks_like_junk(title):
        log.info("title_extractor : junk output %r → fallback to structural", raw[:60])
        # Cache junk too : the same content will produce the same junk
        # next call, no point asking again.
        _cache_write(md5, lang, "")
        return ""

    _cache_write(md5, lang, title)
    log.info("title_extractor : %r → %r (cached)", cleaned[:40].replace("\n", " "), title)
    return title


async def extract_title_via_llm_async(content: str, language: str = "fr", chapter_title: str | None = None) -> str:
    """Async wrapper — runs the (blocking) LLM call in a thread.

    Used by the course builder which already runs slide-title
    resolution under a parallelism semaphore. Putting the LLM call in
    a thread keeps the asyncio loop free during the (potentially
    multi-second) Ollama HTTP wait.
    """
    return await asyncio.to_thread(extract_title_via_llm, content, language, chapter_title)
