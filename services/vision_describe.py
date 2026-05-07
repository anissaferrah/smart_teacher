"""Slide-level vision description with multi-provider fallback chain.

# Why

Slide PNGs contain visual elements (scatter plots, schemas, hand-drawn
formulas, K-NN neighborhood diagrams) that text extraction never captures.
Without describing them, the LLM tutor literally cannot explain "the star
in the middle of the plot" or "the distance d between the two clusters" —
it only sees the bullet text. This service produces a 2-4 sentence
visual description that is appended to the slide_content fed to the QA
graph, giving the tutor the context it needs to discuss the picture.

# Provider chain

Three layers tried in order; whichever returns first wins:

  1. OpenAI gpt-4o-mini      — fast (~1-2s), cheap (~$0.0002/slide), best
                                quality. Fails on quota exhaustion.
  2. Ollama LLaVA / vision   — local, free, slow on CPU (~30-60s/slide).
                                Configurable via OLLAMA_VISION_MODEL.
  3. Skip                    — return "" silently. Pipeline keeps working
                                without the visual layer.

# Caching

Each slide is described **at most once** across the application's
lifetime. The cache key is the MD5 of the image bytes (so re-rendering
or moving the file doesn't invalidate the cache as long as bytes are
identical). Empty results are also cached, to avoid retrying a known-
to-fail image (e.g. quota exhausted on every run, no Ollama available).

The cache lives on disk under ``Config.LOGS_DIR/../cache/vision_descriptions/``
(survives restarts, growable to thousands of entries before it matters).

# Single-flight

Concurrent calls for the same image (multiple users on same slide,
prefetch + live request) coalesce around a single ``asyncio.Future`` so
only one provider call happens. Standard pattern, see Bigtable read
coalescing or golang.org/x/sync/singleflight.

# Observability

Process-local counters in ``_stats``: hits, openai_ok, ollama_ok,
skipped, errors. Read via ``get_stats()``, exposed at
``/admin/cache/stats`` (alongside the learning-style cache stats).

# References

  - GPT-4V technical report (OpenAI 2023) — image understanding capabilities.
  - LLaVA (Liu et al., NeurIPS 2023) — open-source vision-language model.
  - Llama-3.2-Vision (Meta 2024) — alternative if LLaVA underperforms.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from core.config import Config

log = logging.getLogger("services.vision_describe")


# ── Configuration ─────────────────────────────────────────────────────

_CACHE_DIR: Path = Path(getattr(Config, "LOGS_DIR", "logs")).parent / "cache" / "vision_descriptions"
try:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception as exc:                                                # noqa: BLE001
    log.warning("Could not create vision cache dir %s: %s", _CACHE_DIR, exc)

_OLLAMA_TIMEOUT_S: Optional[int] = None    # No timeout — let llama3.2-vision
                                           # take whatever time it needs on
                                           # the user's hardware. CPU vision
                                           # can run 1-3 min for complex
                                           # slides; an upper cap was causing
                                           # spurious failures on slower
                                           # machines and forcing fallback to
                                           # the structural extractor.

# Sentinels meaning "the slide is text-only, no visual content worth describing"
_TEXT_ONLY_MARKERS = {"text_only", "texte_seul", "(text_only)", "(texte_seul)"}


# ── Stats ─────────────────────────────────────────────────────────────

_stats: dict[str, int] = {
    "hits":        0,
    "openai_ok":   0,
    "ollama_ok":   0,
    "skipped":     0,
    "errors":      0,
    "text_only":   0,
}


def get_stats() -> dict[str, int]:
    return dict(_stats)


def reset_stats() -> None:
    """For tests."""
    for k in _stats:
        _stats[k] = 0


# ── Disk cache ────────────────────────────────────────────────────────

def _md5_image(image_path: str) -> str:
    h = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(md5: str, lang: str) -> Path:
    return _CACHE_DIR / f"{md5}_{lang}.txt"


def _cache_read(md5: str, lang: str) -> Optional[str]:
    p = _cache_path(md5, lang)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").rstrip("\n")
    except Exception as exc:
        log.debug("vision cache read failed: %s", exc)
        return None


def _cache_write(md5: str, lang: str, description: str) -> None:
    try:
        _cache_path(md5, lang).write_text(description, encoding="utf-8")
    except Exception as exc:
        log.debug("vision cache write failed: %s", exc)


# ── Single-flight registry ────────────────────────────────────────────

_inflight: dict[str, asyncio.Future] = {}
_inflight_lock = asyncio.Lock()


# ── Prompt builder ────────────────────────────────────────────────────

def _build_prompt(lang: str) -> str:
    if lang == "fr":
        return (
            "Décris cette slide pédagogique en 2-4 phrases courtes. "
            "Concentre-toi UNIQUEMENT sur le contenu VISUEL non textuel : "
            "graphiques (axes, points, courbes, légendes, couleurs), "
            "schémas, diagrammes, formules manuscrites, exemples illustrés, "
            "tableaux dessinés. "
            "Si la slide ne contient que du texte (déjà capté par OCR), "
            "réponds exactement : TEXTE_SEUL. "
            "Ne paraphrase pas le texte — décris ce qu'un OCR ne peut pas capturer."
        )
    return (
        "Describe this educational slide in 2-4 short sentences. "
        "Focus ONLY on the NON-textual visual content: "
        "charts (axes, points, curves, legends, colors), "
        "schemas, diagrams, handwritten formulas, illustrated examples, "
        "drawn tables. "
        "If the slide is text-only (already captured by OCR), "
        "reply exactly: TEXT_ONLY. "
        "Do not paraphrase text — describe what OCR cannot capture."
    )


def _is_text_only_marker(s: str) -> bool:
    return s.strip().lower() in _TEXT_ONLY_MARKERS


def _read_image_b64(image_path: str) -> tuple[str, str]:
    """Return (b64, mime). Raises on read failure."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lstrip(".").lower() or "png"
    mime = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
    return img_b64, mime


# ── Provider 1: OpenAI gpt-4o-mini (or whatever VISION_LLM_MODEL says) ──

def _try_openai(image_path: str, lang: str) -> Optional[str]:
    """Returns the description, or None if quota/availability prevents the call."""
    # Honour the global DISABLE_OPENAI kill-switch (Config). Without this
    # check, the slide-description path still hits OpenAI even when the
    # operator turned the switch on, producing 429s and noisy logs.
    if getattr(Config, "DISABLE_OPENAI", False):
        log.debug("vision describe : DISABLE_OPENAI=true → skip OpenAI provider")
        return None
    api_key = os.getenv("OPENAI_API_KEY") or getattr(Config, "OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        log.debug("openai package not installed — skipping OpenAI vision")
        return None

    try:
        img_b64, mime = _read_image_b64(image_path)
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=Config.VISION_LLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_prompt(lang)},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                ],
            }],
            max_tokens=200,
            temperature=0.0,
        )
        description = (resp.choices[0].message.content or "").strip()
        if not description:
            return None
        if _is_text_only_marker(description):
            _stats["text_only"] += 1
            return ""           # treat as "no visual content" — empty cached
        _stats["openai_ok"] += 1
        log.info("🔭 OpenAI vision OK (%d chars)", len(description))
        return description
    except Exception as exc:                                            # noqa: BLE001
        log.debug("OpenAI vision failed: %s", exc)
        return None


# ── Provider 2: Ollama LLaVA / Llama-3.2-Vision ────────────────────────

def _try_ollama(image_path: str, lang: str) -> Optional[str]:
    """Returns the description, or None if Ollama unreachable / model missing."""
    try:
        import requests
    except ImportError:
        return None
    try:
        img_b64, _ = _read_image_b64(image_path)
        _opts = {"temperature": 0.0}
        _n_threads = int(getattr(Config, "OLLAMA_NUM_THREADS", 0) or 0)
        if _n_threads > 0:
            _opts["num_thread"] = _n_threads
        payload = {
            "model":     Config.OLLAMA_VISION_MODEL,
            "prompt":    _build_prompt(lang),
            "images":    [img_b64],
            "stream":    False,
            "options":   _opts,
        }
        resp = requests.post(
            f"{Config.OLLAMA_URL}/api/generate",
            json=payload,
            timeout=_OLLAMA_TIMEOUT_S,
        )
        if resp.status_code != 200:
            log.debug("Ollama vision HTTP %d: %s", resp.status_code, resp.text[:120])
            return None
        description = (resp.json().get("response") or "").strip()
        if not description:
            return None
        if _is_text_only_marker(description):
            _stats["text_only"] += 1
            return ""
        _stats["ollama_ok"] += 1
        log.info("🔭 Ollama vision OK (model=%s, %d chars)", Config.OLLAMA_VISION_MODEL, len(description))
        return description
    except Exception as exc:                                            # noqa: BLE001
        log.debug("Ollama vision failed: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────

async def describe_slide_image(image_path: str | None, lang: str = "en") -> str:
    """Resolve a 2-4 sentence visual description for a slide PNG.

    Returns ``""`` when no provider produced a description (or the slide is
    deemed text-only). Always non-blocking on the cache hit path; on a
    miss, runs blocking provider calls in a thread executor so the event
    loop stays free.
    """
    if not Config.VISION_DESCRIBE_ENABLED:
        _stats["skipped"] += 1
        return ""
    if not image_path or not isinstance(image_path, str):
        return ""

    # Resolve relative `/media/...` paths against project root if needed.
    candidate = image_path
    if candidate.startswith("/media/"):
        candidate = os.path.join(os.getcwd(), candidate.lstrip("/"))
    if not os.path.exists(candidate):
        log.debug("vision describe: image not found at %s", candidate)
        return ""

    lang2 = (lang or "en")[:2]

    try:
        md5 = await asyncio.to_thread(_md5_image, candidate)
    except Exception as exc:                                            # noqa: BLE001
        log.debug("md5 failed: %s", exc)
        return ""

    # Disk cache hit — fastest path
    cached = _cache_read(md5, lang2)
    if cached is not None:
        _stats["hits"] += 1
        return cached

    # Single-flight registration
    fut: Optional[asyncio.Future] = None
    follower: Optional[asyncio.Future] = None
    async with _inflight_lock:
        existing = _inflight.get(md5)
        if existing is not None:
            follower = existing
        else:
            # Late re-check inside the lock
            cached = _cache_read(md5, lang2)
            if cached is not None:
                _stats["hits"] += 1
                return cached
            fut = asyncio.get_running_loop().create_future()
            _inflight[md5] = fut

    if follower is not None:
        return await follower

    # Leader path — try providers in order
    description = ""
    try:
        description = await asyncio.to_thread(_try_openai, candidate, lang2) or ""
        if not description:
            description = await asyncio.to_thread(_try_ollama, candidate, lang2) or ""
        if not description:
            _stats["skipped"] += 1
        # Cache (even an empty result, to avoid retrying a known-fail image)
        _cache_write(md5, lang2, description)
        if fut and not fut.done():
            fut.set_result(description)
        return description
    except Exception as exc:                                            # noqa: BLE001
        log.warning("vision describe failed: %s", exc)
        _stats["errors"] += 1
        if fut and not fut.done():
            fut.set_result("")
        return ""
    finally:
        async with _inflight_lock:
            _inflight.pop(md5, None)


def merge_into_slide_content(slide_text: str, vision_desc: str, lang: str = "en") -> str:
    """Produce the merged slide content for the QA prompt.

    The vision description is appended with an explicit marker so the LLM
    knows it's complementary to the OCR-extracted text.
    """
    slide_text = (slide_text or "").rstrip()
    vision_desc = (vision_desc or "").strip()
    if not vision_desc:
        return slide_text
    label = "[Description visuelle de la slide]" if lang == "fr" else "[Visual description of the slide]"
    if not slide_text:
        return f"{label} {vision_desc}"
    return f"{slide_text}\n\n{label} {vision_desc}"


# ══════════════════════════════════════════════════════════════════════
#  Concept extraction from slide image (vision-first replacement for
#  the regex/structural heuristics in agentic.teaching.planner)
# ══════════════════════════════════════════════════════════════════════
#
# Why a separate function (rather than reusing describe_slide_image):
#   - Different prompt: ask the model to *identify* the slide's main
#     concept, not describe its visuals.
#   - Different cache key suffix: same image can answer both "describe
#     the chart" and "what's the concept" calls without conflict.
#   - Different output shape: short token (1-5 words), not 2-4 sentences.
#
# Why vision and not regex/structure:
#   Slide layouts vary too much across courses for any regex or rule set
#   to cover them all (yellow headings, white headings, no headings,
#   hand-drawn boxes, table-only slides, image-only slides). The model
#   actually *looks* at the slide the way a teacher does: it reads the
#   visual hierarchy, finds the largest/colored/positioned-as-heading
#   text, and returns that. No vocabulary, no language assumptions.
#
# Fallback: when no image is available (audio-only, text imports, or
# vision providers all down), the planner falls back to its structural
# heuristics. They're imperfect, but ship-able as a degraded mode.

def _build_concept_prompt(
    lang: str,
    section_title: str = "",
    chapter_title: str = "",
) -> str:
    """Build a vision prompt qui demande au modele d'identifier le SUJET
    SPECIFIQUE de la slide (pas les headers chapitre/section qui se
    repetent sur toutes les slides).

    Pourquoi c'est subtil : sur des slides templatees (deck pedagogique
    typique), il y a souvent DEUX niveaux de headers qui se repetent :
      - chapter header (ex: "6. Supervised Machine Learning")
      - section header (ex: "K-NN", "Decision Trees")

    Un small vision model (llava:7b) tend a saisir le texte le plus
    grand / le plus visible, qui est SOUVENT un de ces headers — pas
    le titre specifique de la slide.

    Fix : prompt generique qui dit au modele d'IGNORER tout texte qui
    se repete d'une slide a l'autre. Bilingue (fr/en) — pas de hardcoding
    de noms de cours, marche pour n'importe quel deck.
    """
    is_fr = (lang or "").lower().startswith("fr")

    # Garde-fou cours bilingue : la slide peut etre dans l'autre langue
    # que le cours. On invite explicitement le modele a repondre dans la
    # langue de la slide (peu importe celle des instructions). Vital pour
    # des decks mixtes FR/EN (ex: cours FR avec des termes techniques en
    # anglais, ou cours EN avec exemples francais).
    bilingual_output_hint_fr = (
        "Le contenu de la slide peut etre en anglais ou en francais — "
        "reponds dans la langue de la slide."
    )
    bilingual_output_hint_en = (
        "The slide content may be in English or French — "
        "reply in the slide's own language."
    )

    # ── FR ──────────────────────────────────────────────────────────
    if is_fr:
        universal_hints_fr = (
            "Les supports de cours affichent souvent 2 types de headers qui se "
            "repetent : un header de CHAPITRE (identique sur toutes les slides "
            "du chapitre) et parfois un header de SECTION (identique sur des "
            "slides consecutives). Aucun de ces deux n'est le sujet de CETTE "
            "slide — ignore-les. Cherche le titre principal ou le sujet central "
            "UNIQUE a cette slide (generalement un titre ou le plus grand texte "
            "qui ne se repete pas). Si la slide ne contient qu'un schema ou des "
            "exemples, deduis le sujet du contenu affiche."
        )
        if section_title:
            return (
                f"Lis cette slide de cours. {universal_hints_fr} "
                f'Le header de section qui se repete ici est "{section_title}" '
                f"— IGNORE-le. Quel est le sujet SPECIFIQUE de CETTE slide ? "
                f"{bilingual_output_hint_fr} "
                f"Reponds en 1 a 5 mots. Pas de phrase."
            )
        if chapter_title:
            return (
                f'Lis cette slide de cours du chapitre "{chapter_title}". '
                f"{universal_hints_fr} "
                f"Quel est le sujet SPECIFIQUE de CETTE slide ? "
                f"{bilingual_output_hint_fr} "
                f"Reponds en 1 a 5 mots. Pas de phrase."
            )
        return (
            f"Lis cette slide de cours. {universal_hints_fr} "
            f"Quel est le sujet SPECIFIQUE de CETTE slide ? "
            f"{bilingual_output_hint_fr} "
            f"Reponds en 1 a 5 mots. Pas de phrase."
        )

    # ── EN ──────────────────────────────────────────────────────────
    universal_hints_en = (
        "Slide decks typically show 2 kinds of repeating headers : "
        "a CHAPTER header (always the same on every slide of the chapter) "
        "and sometimes a SECTION header (the same across consecutive slides). "
        "Both are NOT the topic of this specific slide — ignore them. "
        "Look for the main title or central topic UNIQUE to this slide "
        "(usually a heading or the largest non-repeating text). "
        "If the slide only has a diagram or examples, infer the topic "
        "from the content shown."
    )
    if section_title:
        return (
            f"Read this lecture slide. {universal_hints_en} "
            f'The repeating section header here is "{section_title}" — IGNORE it. '
            f"What is the SPECIFIC topic of THIS slide? "
            f"{bilingual_output_hint_en} "
            f"Reply with just 1 to 5 words. No sentence."
        )
    if chapter_title:
        return (
            f'Read this lecture slide from chapter "{chapter_title}". '
            f"{universal_hints_en} "
            f"What is the SPECIFIC topic of THIS slide? "
            f"{bilingual_output_hint_en} "
            f"Reply with just 1 to 5 words. No sentence."
        )
    return (
        f"Read this lecture slide. {universal_hints_en} "
        f"What is the SPECIFIC topic of THIS slide? "
        f"{bilingual_output_hint_en} "
        f"Reply with just 1 to 5 words. No sentence."
    )


# Kept for backward compatibility — older callers pass no titles, get
# a non-contextual prompt.
_CONCEPT_PROMPT_FR = _build_concept_prompt("fr")
_CONCEPT_PROMPT_EN = _build_concept_prompt("en")


def _concept_cache_path(md5: str, lang: str) -> Path:
    # ``.concept.txt`` suffix keeps the concept cache distinct from the
    # description cache for the same image.
    return _CACHE_DIR / f"{md5}_{lang}.concept.txt"


def _concept_cache_read(md5: str, lang: str) -> Optional[str]:
    p = _concept_cache_path(md5, lang)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").rstrip("\n")
    except Exception as exc:                                                # noqa: BLE001
        log.debug("concept cache read failed: %s", exc)
        return None


def _concept_cache_write(md5: str, lang: str, concept: str) -> None:
    try:
        _concept_cache_path(md5, lang).write_text(concept, encoding="utf-8")
    except Exception as exc:                                                # noqa: BLE001
        log.debug("concept cache write failed: %s", exc)


def _normalise_concept(raw: str) -> str:
    """Trim quotes, periods, trailing punctuation; clamp to 60 chars.

    The model occasionally wraps the concept in quotes or appends a stop;
    strip those without rejecting otherwise valid output.
    """
    s = (raw or "").strip().strip('"').strip("'").strip("`").strip()
    s = s.rstrip(".!?:; ")
    return s[:60]


def _is_unknown_marker(s: str) -> bool:
    return s.strip().lower() in {"unknown", "inconnu", "(unknown)", "(inconnu)", "n/a", "none"}


def _try_openai_concept(
    image_path: str,
    lang: str,
    section_title: str = "",
    chapter_title: str = "",
) -> Optional[str]:
    # Honour the global DISABLE_OPENAI kill-switch (Config). When true,
    # we skip the OpenAI provider entirely so the chain falls through to
    # Ollama / structural fallback without spending a network round-trip
    # discovering the missing key.
    if getattr(Config, "DISABLE_OPENAI", False):
        log.info("🎯 OpenAI concept skip: DISABLE_OPENAI=true")
        return None
    api_key = os.getenv("OPENAI_API_KEY") or getattr(Config, "OPENAI_API_KEY", "")
    if not api_key:
        log.info("🎯 OpenAI concept skip: no API key configured")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        log.info("🎯 OpenAI concept skip: openai package not installed")
        return None
    try:
        img_b64, mime = _read_image_b64(image_path)
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=Config.VISION_LLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_concept_prompt(lang, section_title, chapter_title)},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                ],
            }],
            max_tokens=30,            # concept is at most a few words
            temperature=0.0,
        )
        out = _normalise_concept((resp.choices[0].message.content or ""))
        if not out or _is_unknown_marker(out):
            log.info("🎯 OpenAI returned UNKNOWN/empty for %s", Path(image_path).name)
            return ""                  # known-empty (cached to avoid retry)
        log.info("🎯 Concept (OpenAI vision): %r", out[:40])
        return out
    except Exception as exc:                                                # noqa: BLE001
        # Surface the failure at INFO so the operator sees *why* the
        # call dropped through to the next provider (was DEBUG and
        # therefore invisible at default log level).
        log.info("🎯 OpenAI concept failed: %s", str(exc)[:160])
        return None


def _try_ollama_concept(
    image_path: str,
    lang: str,
    section_title: str = "",
    chapter_title: str = "",
) -> Optional[str]:
    try:
        import requests
    except ImportError:
        log.info("🎯 Ollama concept skip: requests package not installed")
        return None
    try:
        img_b64, _ = _read_image_b64(image_path)
        prompt = _build_concept_prompt(lang, section_title, chapter_title)
        # Resolve model name. If the user configured a bare model name
        # without a tag (e.g. "llama3.2-vision"), Ollama's /api/generate
        # returns HTTP 404 "model not found" even when the tagged
        # version (llama3.2-vision:latest) is installed. Try the bare
        # name first; on 404, retry with :latest.
        model_candidates = [Config.OLLAMA_VISION_MODEL]
        if ":" not in Config.OLLAMA_VISION_MODEL:
            model_candidates.append(f"{Config.OLLAMA_VISION_MODEL}:latest")
        # Honour OLLAMA_NUM_THREADS (0 = let Ollama auto-pick).
        _options = {"temperature": 0.0, "num_predict": 30}
        _n_threads = int(getattr(Config, "OLLAMA_NUM_THREADS", 0) or 0)
        if _n_threads > 0:
            _options["num_thread"] = _n_threads
        last_resp = None
        for model_name in model_candidates:
            payload = {
                "model":   model_name,
                "prompt":  prompt,
                "images":  [img_b64],
                "stream":  False,
                "options": _options,
            }
            log.info("🎯 Ollama concept call → %s (model=%s)",
                     Path(image_path).name, model_name)
            resp = requests.post(
                f"{Config.OLLAMA_URL}/api/generate",
                json=payload,
                timeout=_OLLAMA_TIMEOUT_S,
            )
            last_resp = resp
            if resp.status_code == 200:
                break
            log.info("🎯 Ollama HTTP %d (model=%s): %s",
                     resp.status_code, model_name, resp.text[:200])
            # Only retry on 404 (model name issue). Other errors are
            # not retried because they're not name-related.
            if resp.status_code != 404:
                return None
        if last_resp is None or last_resp.status_code != 200:
            return None
        out = _normalise_concept((last_resp.json().get("response") or ""))
        if not out or _is_unknown_marker(out):
            log.info("🎯 Ollama returned UNKNOWN/empty for %s", Path(image_path).name)
            return ""
        log.info("🎯 Concept (Ollama vision): %r", out[:40])
        return out
    except Exception as exc:                                                # noqa: BLE001
        log.info("🎯 Ollama concept failed: %s", str(exc)[:160])
        return None


async def extract_slide_concept(
    image_path: str | None,
    lang: str = "en",
    section_title: str = "",
    chapter_title: str = "",
) -> str:
    """Ask a vision LLM to identify the slide's specific topic.

    Section/chapter titles are passed as context so the model can
    recognize and skip the headers that repeat across every slide of
    the section, focusing on what makes THIS slide different.

    Also auto-rejects results that are equal (case-insensitive) to the
    section or chapter title — small vision models tend to read those
    even when instructed to ignore them. On rejection, returns ``""``
    so the caller can fall back to the structural extractor (which
    handles section-title repetition correctly).

    Returns the concept as a short string (1-5 words) or ``""`` when:
      - vision is disabled by config,
      - no valid image is provided / accessible,
      - all providers returned UNKNOWN or failed,
      - the model returned the section/chapter title (already known).
    """
    if not Config.VISION_DESCRIBE_ENABLED:
        _stats["skipped"] += 1
        return ""
    # Operator-level kill-switch for vision titles. When the local Ollama
    # vision model is too small to read text reliably (llava 7B, etc.) and
    # produces hallucinated titles ("Introduction à la rive", "Introduction
    # à la recherche d'agrément"), set DISABLE_VISION_TITLES=true in the
    # env to force the deterministic structural extractor.
    if getattr(Config, "DISABLE_VISION_TITLES", False):
        _stats["skipped"] += 1
        log.info("🎯 vision concept skipped: DISABLE_VISION_TITLES=true → structural fallback")
        return ""
    if not image_path or not isinstance(image_path, str):
        return ""

    candidate = image_path
    if candidate.startswith("/media/"):
        candidate = os.path.join(os.getcwd(), candidate.lstrip("/"))
    if not os.path.exists(candidate):
        log.debug("concept extract: image not found at %s", candidate)
        return ""

    lang2 = (lang or "en")[:2]

    try:
        md5 = await asyncio.to_thread(_md5_image, candidate)
    except Exception as exc:                                                # noqa: BLE001
        log.debug("md5 failed: %s", exc)
        return ""

    cached = _concept_cache_read(md5, lang2)
    if cached is not None:
        _stats["hits"] += 1
        return cached

    # Single-flight on the concept namespace (separate from the description
    # namespace — same image can have both calls in parallel without
    # blocking each other).
    flight_key = f"concept:{md5}"
    fut: Optional[asyncio.Future] = None
    follower: Optional[asyncio.Future] = None
    async with _inflight_lock:
        existing = _inflight.get(flight_key)
        if existing is not None:
            follower = existing
        else:
            cached = _concept_cache_read(md5, lang2)
            if cached is not None:
                _stats["hits"] += 1
                return cached
            fut = asyncio.get_running_loop().create_future()
            _inflight[flight_key] = fut

    if follower is not None:
        return await follower

    concept = ""
    try:
        concept = await asyncio.to_thread(
            _try_openai_concept, candidate, lang2, section_title, chapter_title,
        ) or ""
        if not concept:
            concept = await asyncio.to_thread(
                _try_ollama_concept, candidate, lang2, section_title, chapter_title,
            ) or ""
        # Reject results that are just the section or chapter title —
        # small vision models often read those even when told to ignore.
        # Empty string lets the caller fall back to the structural
        # extractor, which handles section-title repetition correctly.
        #
        # Match tolerant : on normalise les deux cotes en stripant
        # - les prefixes numeriques courants ("6. ", "Chapter 6 :", "Ch. 6 -")
        # - la ponctuation et les espaces
        # puis on rejecte si l'un contient l'autre (substring). Ainsi :
        #   chapter_title = "6. Supervised Machine Learning"
        #   concept       = "Supervised machine learning"
        #   → match → reject (avant : pas matche a cause du "6. ").
        if concept:
            def _normalize(s: str) -> str:
                """Lower, strip numeric/chapter prefixes, collapse whitespace."""
                import re
                t = s.strip().lower()
                # Strip "chapter N", "ch N", "section N" prefixes
                t = re.sub(r"^(chapter|ch\.?|chapitre|section)\s+\d+\s*[:\-—]?\s*",
                           "", t)
                # Strip leading "N. " or "N) " or "N - " (numero seul)
                t = re.sub(r"^\d+\s*[.\)\-—:]\s*", "", t)
                # Collapse internal whitespace
                t = re.sub(r"\s+", " ", t).strip()
                return t

            n_concept = _normalize(concept)
            n_section = _normalize(section_title) if section_title else ""
            n_chapter = _normalize(chapter_title) if chapter_title else ""

            # Reject si concept ⊆ section/chapter OU section/chapter ⊆ concept
            # (les 2 sens parce que le modele peut renvoyer une partie du titre).
            def _matches(a: str, b: str) -> bool:
                if not a or not b:
                    return False
                return a == b or a in b or b in a

            if section_title and _matches(n_concept, n_section):
                log.info("🎯 vision returned section title-like %r → reject, fallback structurel",
                         concept[:40])
                concept = ""
            elif chapter_title and _matches(n_concept, n_chapter):
                log.info("🎯 vision returned chapter title-like %r → reject, fallback structurel",
                         concept[:40])
                concept = ""
        if not concept:
            _stats["skipped"] += 1
        _concept_cache_write(md5, lang2, concept)
        if fut and not fut.done():
            fut.set_result(concept)
        return concept
    except Exception as exc:                                                # noqa: BLE001
        log.warning("concept extraction failed: %s", exc)
        _stats["errors"] += 1
        if fut and not fut.done():
            fut.set_result("")
        return ""
    finally:
        async with _inflight_lock:
            _inflight.pop(flight_key, None)
