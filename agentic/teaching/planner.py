"""PlannerAgent — decomposes a slide into a sequence of pedagogical Ideas.

Input  (from TutorState): last_slide_content, chapter_title, section_title, language
Output (into TutorState): plan: PresentationPlan
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from agentic.schemas import Idea, PresentationPlan
from agentic.state import TutorState

log = logging.getLogger("agentic.teaching.planner")


_PLANNER_PROMPT_FR = """Tu es un planificateur pédagogique pour un professeur virtuel qui présente une slide de cours.

À partir du contenu de la slide ci-dessous, décompose la présentation en 3 à 5 IDÉES logiques que le professeur doit présenter dans l'ordre.

CONCEPT PRINCIPAL DE LA SLIDE : {main_concept}
La présentation DOIT être centrée sur ce concept. Ne le confonds pas avec une note de bas de page (par ex. "n est la taille de l'échantillon") — celle-ci n'est qu'une légende, pas le concept enseigné.

RÈGLES IMPORTANTES :
- Les "id" et "content_brief" DOIVENT être dérivés du vrai contenu de la slide. N'invente pas de noms de concepts.
- N'utilise JAMAIS de libellés génériques numérotés comme "Indice 1", "Méthode 2", "Concept 3", "Notion 1" — ce sont des placeholders d'un template, pas de vrais concepts.
- Si la slide ne contient pas assez de matière pour identifier un concept réel, renvoie une seule idée d'introduction qui présente le titre de la section.

Chaque idée doit contenir :
- "id"            : identifiant ASCII court dérivé du contenu réel (ex: "intro", "definition_eda", "exemple_demographie", "synthese")
- "type"          : "intro" | "concept" | "example" | "summary"
- "content_brief" : UNE LIGNE décrivant ce que le professeur doit dire (PAS la narration complète)
- "depth"         : "shallow" | "normal" | "deep"

Réponds UNIQUEMENT en JSON strict, sans markdown, sans explication :
{{"ideas": [{{"id":"...","type":"...","content_brief":"...","depth":"..."}}, ...]}}

Chapitre : {chapter}
Section  : {section}
Slide :
{slide_content}
"""


_PLANNER_PROMPT_EN = """You are a pedagogy planner for a virtual professor presenting a lecture slide.

Given the slide content below, decompose the presentation into 3 to 5 logical IDEAS the professor should present in order.

MAIN CONCEPT OF THIS SLIDE: {main_concept}
The plan MUST be anchored on this concept. Do not confuse it with a footnote
(e.g. "n is the sample size") — that's a notation legend, not the concept being taught.

IMPORTANT RULES:
- "id" and "content_brief" MUST be derived from the actual slide content. Do not invent concept names.
- NEVER use generic numbered labels like "Indice 1", "Method 2", "Concept 3", "Notion 1" — those are template placeholders, not real concepts.
- If the slide doesn't contain enough material to identify a real concept, return a single introduction idea that presents the section title.

Each idea must contain:
- "id"            : short ASCII id derived from real content (e.g. "intro", "definition_eda", "example_demographics", "summary")
- "type"          : "intro" | "concept" | "example" | "summary"
- "content_brief" : ONE LINE describing what the professor should say (NOT the full narration)
- "depth"         : "shallow" | "normal" | "deep"

Output STRICT JSON ONLY, no markdown, no commentary:
{{"ideas": [{{"id":"...","type":"...","content_brief":"...","depth":"..."}}, ...]}}

Chapter: {chapter}
Section: {section}
Slide:
{slide_content}
"""


# Lines that look like notation legends ("n: sample size", "x_i: value",
# "Q1: 25%", "wi : weight"). Course-agnostic: the rule is *structural* —
# a short identifier-shaped token (1-3 chars, optionally with a sub/
# superscript) introducing a definition. Carries no domain vocabulary,
# so it works for stats, physics, CS, languages, anything.
_NOTATION_LEGEND = re.compile(
    r"^[A-Za-zα-ωΑ-Ω](?:[A-Za-z0-9]|[_\^][A-Za-z0-9]+){0,2}\s*[:=]\s*\S",
    flags=re.UNICODE,
)


def _context_stopwords(section_title: str, chapter_title: str = "") -> set[str]:
    """Words that recur on every slide because they're in the section or
    chapter title.

    Pure dynamic derivation — no hardcoded subject vocabulary. Tokenize
    the titles passed in by the caller and treat any 4+ char alpha word
    as a context word that shouldn't be picked as the per-slide concept
    (because it's the *parent* concept, not the slide's specific one).
    Works for any course because the input is the course's own titles.
    """
    words: set[str] = set()
    for title in (section_title, chapter_title):
        if not title:
            continue
        for w in re.findall(r"[A-Za-zÀ-ÿ]+", title):
            if len(w) >= 4:
                words.add(w.lower())
    return words


def _looks_like_sentence(text: str) -> bool:
    """Structural sentence detector — no vocabulary. A piece of text is a
    sentence if any of the following hold:

      - it ends with a sentence terminator (``.``, ``!``, ``?``);
      - it has more than 4 whitespace-separated words (headings on
        these decks are at most ~4 words: "Pearson correlation",
        "Standard deviation", "Quantile–Quantile Plot (Q-Q)");
      - it contains a comma (headings rarely do).
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[-1] in ".!?":
        return True
    if "," in stripped:
        return True
    if len(stripped.split()) > 4:
        return True
    return False


def _looks_like_data_body(body: str) -> bool:
    """True if the post-colon body is data (numbers, list, array) rather
    than a definition. Distinguishes ``Data: [10, 12, 13]`` (a data
    label inside a worked example) from ``Trimmed mean: the trimmed
    mean is…`` (a concept definition).

    Pure structural: looks at the ratio of digits to letters and at
    list/array/value markers — no vocabulary.
    """
    body = body.strip()
    if not body:
        return False
    if body[0] in "[({":
        return True
    letters = sum(1 for c in body if c.isalpha())
    digits = sum(1 for c in body if c.isdigit())
    return digits >= 3 and digits >= letters


def _is_meaningful_concept_token(text: str) -> bool:
    """A heading must be at least 4 chars long, OR be an all-uppercase
    acronym of 2-3 chars (PCA, EDA, IQR). This rules out 2-letter math
    notation like ``xi``, ``wi``, ``Q1``, ``Q3``, ``di``, ``fi``.
    Pure structural rule — no vocabulary.
    """
    if len(text) >= 4:
        return True
    return text.isupper() and 2 <= len(text) <= 3


def _is_duplicated_label(text: str) -> bool:
    """Detect category labels like ``"Numerical – Numerical"`` or
    ``"Categorical -- Categorical"`` where the same word repeats.
    These appear in tables/sections as classifiers, not as the slide's
    actual concept.
    """
    words = [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]+", text)]
    return len(words) >= 2 and len(set(words)) < len(words)


def _heading_candidate(
    line: str,
    section_title: str,
    context_stops: set[str],
) -> tuple[int, str, str] | None:
    """Decide if ``line`` carries a concept heading. Returns
    ``(priority, candidate_text, source_line)`` or ``None``.

    Three shapes recognised, in decreasing priority:
      0. ``Heading:`` standalone — line is the heading and ends with `:`.
      1. ``Heading: body sentence on the same line`` — PDF extraction
         often concatenates a yellow sub-heading and the first body line.
      2. ``Heading`` — short capitalised standalone line, no colon.

    Pure structural test — no vocabulary lists. The duplicated-label
    check applies to the candidate text only, not the full line, so a
    concept heading whose definition repeats words in the same flat-
    extracted line ("Trimmed mean: the trimmed mean is the mean
    calculated…") isn't rejected for the body's repetition.
    """
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(("-", "•", "→", "➔", "◆", "●", "*", "○")):
        return None
    if not stripped[0].isalpha():
        return None
    if _NOTATION_LEGEND.match(stripped):
        return None
    if section_title and stripped.lower().strip(":").strip() == section_title.lower().strip():
        return None
    first_word = re.split(r"[\s:]", stripped, maxsplit=1)[0].lower()
    if first_word in context_stops:
        return None

    # Case 1/2: line contains a `:`. Use the prefix as the candidate.
    if ":" in stripped:
        prefix, _, _body = stripped.partition(":")
        prefix = prefix.strip()
        if _looks_like_data_body(_body):
            # ``Data: [10, 12, 13]`` — prefix is a data label, not a
            # concept heading. Drop the colon path and let Case 3 handle
            # the line as a possible standalone (it usually won't, since
            # the body has digits/symbols).
            pass
        elif (
            prefix
            and prefix[0].isalpha()
            and not _looks_like_sentence(prefix)
            and not _is_duplicated_label(prefix)
            and _is_meaningful_concept_token(prefix.split()[0])
            and _is_meaningful_concept_token(prefix)
        ):
            pf_first = re.split(r"\s", prefix, maxsplit=1)[0].lower()
            if pf_first not in context_stops:
                priority = 0 if not _body.strip() else 1
                return priority, prefix, line

    # Case 3: short capitalised standalone line, no colon.
    if (
        stripped[0].isupper()
        and not _looks_like_sentence(stripped)
        and not _is_duplicated_label(stripped)
        and _is_meaningful_concept_token(stripped)
    ):
        return 2, stripped, line
    return None


def _body_mention_score(candidate: str, slide_text_lower: str, source_line_lower: str) -> int:
    """Count how often the candidate's significant words appear in the
    slide body (excluding the candidate's own source line).

    Real concept headings get echoed in their own definition: a slide
    on "Median" repeats "median" through the explanation; on "Outliers"
    the body says "outlier", "outliers" both. Generic structural labels
    ("Useful for:", "Example:", "Note:", "Definition:") don't get echoed
    — the body talks about something else. Mention-count is a course-
    agnostic, structural way to tell them apart.

    Uses prefix-stem matching (first 5 chars) so plural / conjugated
    forms count: candidate "Outliers" stems to "outli" and matches both
    "outlier" and "outliers" in body without needing a real stemmer.
    Significant = words of 4+ chars (skips short connectives "of",
    "the", "for"). For all-short candidates ("PCA", "EDA", "IQR") falls
    back to counting the whole candidate as a token.
    """
    body = slide_text_lower.replace(source_line_lower, " ", 1)
    significant = [w for w in re.findall(r"\w+", candidate.lower()) if len(w) >= 4]
    if not significant:
        return body.count(candidate.lower())
    # Truncate to 5-char stems for plural/conjugation tolerance. 5 is
    # short enough to match "outlier"/"outliers" via "outli", "median"/
    # "medians" via "media", but long enough to avoid spurious matches
    # ("data" → "data", not truncated; "datasets" → "datas" still
    # matches "data" prefix).
    return sum(body.count(w[:5]) for w in significant)


def _extract_main_concept(
    slide_content: str,
    section_title: str = "",
    chapter_title: str = "",
) -> str:
    """Return the slide's main pedagogical concept, derived from structure.

    PDF extraction flattens the visual hierarchy — the yellow sub-heading
    that announces the concept ("Trimmed mean:", "Median", "Quartile Q1",
    "Pearson correlation", "Newton's second law") becomes a plain
    paragraph indistinguishable from body text or notation legends
    ("n: sample size"). This helper applies subject-agnostic heuristics
    to recover the heading:

      1. Skip notation legends (short identifier + colon + gloss).
      2. Skip bullets and lines matching the section title.
      3. Skip lines that look like sentences (terminator, comma, > 4 words).
      4. Skip words already in the section/chapter title (dynamic — works
         for any course because the input is the course's own titles).
      5. Demote candidates whose significant words don't appear elsewhere
         in the slide body. "Useful for:" / "Example:" / "Note:" lose to
         the actual concept because their words don't get echoed in the
         definition body.
      6. Prefer ``Heading:`` standalone (priority 0), then ``Heading:
         inline body`` (priority 1), then short capitalised standalones
         (priority 2). Within a (possibly demoted) priority, more body
         mentions win; within that, longer text wins.

    Returns "" when no clean heading is found; the planner falls back to
    the section title in that case.

    NOTE : on academic slides where each page carries a Roman-numeral
    section header at the top (``I. CONTEXTE``, ``VI.1. PERTINENCE``),
    this structural extractor frequently picks the wrong line — body
    bullets that happen to look like headings, or sub-section labels.
    The cleanest fix is the LLM-based title extractor in
    ``services.title_extractor`` ; the course_builder calls it before
    falling back here. This function remains the last-ditch fallback
    when no LLM is reachable.
    """
    context_stops = _context_stopwords(section_title, chapter_title)
    raw: list[tuple[int, str, str]] = []
    for ln in slide_content.splitlines():
        cand = _heading_candidate(ln, section_title, context_stops)
        if cand is not None:
            raw.append(cand)
    if not raw:
        return ""

    slide_lower = slide_content.lower()
    scored: list[tuple[int, int, int, str]] = []  # (eff_priority, -mentions, -length, text)
    for priority, text, source_line in raw:
        mentions = _body_mention_score(text, slide_lower, source_line.lower())
        # Words that don't appear elsewhere in the slide → likely a
        # generic structural label, not the concept. Demote sharply so
        # any candidate with at least one mention beats it regardless of
        # original priority.
        eff_priority = priority if mentions > 0 else priority + 100
        scored.append((eff_priority, -mentions, -len(text), text))
    scored.sort()
    return scored[0][3]


# Placeholder line: a slide bullet that's nothing more than a short word
# followed by a digit ("Indice 1", "Item 2", "Concept 3"). Universal
# template-skeleton shape — pure structural test, no vocabulary.
_PLACEHOLDER_LINE = re.compile(
    r"^[-•*\d.)\s]*"
    r"[A-Za-zÀ-ÿ]{3,15}"           # one short word
    r"\s*[#n°:\.\-]*\s*\d+\s*$",    # separator + digit at EOL
    flags=re.IGNORECASE,
)

# Same shape applied to a single content_brief returned by the planner
# LLM. A brief that's just "<word> <digit>" leaks into narration as a
# fake concept name when read aloud.
_PLACEHOLDER_BRIEF = re.compile(
    r"^\s*[A-Za-zÀ-ÿ]{3,15}"
    r"\s*[#n°:\.\-]*\s*\d+\s*$",
    flags=re.IGNORECASE,
)


def _is_slide_too_thin(slide: str) -> bool:
    """True when a slide has too little real content for the planner LLM.

    Triggers the fallback plan and skips the LLM call. Avoids the failure
    mode where the LLM, given a slide of mostly placeholder labels, echoes
    those labels back as content_briefs ("Indice 1", "Méthode 2") that
    the narrator then reads as if they were real concept names.

    Threshold: fewer than 15 word-characters tokens, OR more than 70 % of
    non-empty lines match the placeholder pattern.
    """
    words = len(re.findall(r"\w+", slide, flags=re.UNICODE))
    if words < 15:
        return True
    lines = [ln.strip() for ln in slide.splitlines() if ln.strip()]
    if not lines:
        return True
    real_lines = sum(1 for ln in lines if not _PLACEHOLDER_LINE.match(ln))
    return (real_lines / len(lines)) < 0.30


def _sanitize_brief(raw_brief: str) -> str:
    """Drop content_briefs that are nothing more than template labels.

    The narrator copies the brief into the prompt that generates the
    spoken text. A brief like ``"Indice 1"`` therefore leaks into the
    final narration as a fake concept name. When that happens, replace
    it with a neutral hint so the narrator falls back on the slide
    content and the retrieved chunks instead of the placeholder.
    """
    brief = (raw_brief or "").strip()
    if not brief:
        return ""
    if _PLACEHOLDER_BRIEF.match(brief):
        return "Explain the relevant concept grounded in the slide content"
    return brief[:240]


def _extract_json(raw: str) -> dict[str, Any] | None:
    """Strip markdown fences and locate the first {...} block."""
    if not raw:
        return None
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _coerce_idea(raw: dict[str, Any], idx: int) -> Idea:
    """Best-effort conversion of LLM dict → Idea, with safe defaults."""
    raw_type = str(raw.get("type", "concept")).strip().lower()
    if raw_type not in ("intro", "concept", "example", "summary"):
        raw_type = "concept"

    raw_depth = str(raw.get("depth", "normal")).strip().lower()
    if raw_depth not in ("shallow", "normal", "deep"):
        raw_depth = "normal"

    raw_id = str(raw.get("id", f"idea_{idx}")).strip()
    raw_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_id)[:48] or f"idea_{idx}"

    return Idea(
        id=raw_id,
        type=raw_type,  # type: ignore[arg-type]
        content_brief=_sanitize_brief(str(raw.get("content_brief", ""))),
        depth=raw_depth,  # type: ignore[arg-type]
    )


def _fallback_plan(state: TutorState) -> PresentationPlan:
    """If the LLM is unavailable or returns garbage, build a 3-step skeleton."""
    return PresentationPlan(
        section_title=state.get("section_title", ""),
        chapter_title=state.get("chapter_title", ""),
        ideas=[
            Idea(id="intro", type="intro", content_brief="Introduce the topic", depth="shallow"),
            Idea(id="concept_main", type="concept", content_brief="Explain the main concept", depth="normal"),
            Idea(id="summary", type="summary", content_brief="Wrap up and key takeaways", depth="shallow"),
        ],
    )


class PlannerAgent:
    """Builds a PresentationPlan from raw slide content using the LLM.

    Concept-aware:
      - When the student has mastered ALL retrieved-related ideas for the
        current chapter, the planner switches to review_mode (shorter plan).
        The decision is made per slide based on mastery of the local
        chunks, not on a global "≥ N mastered" count — that earlier
        threshold (5) was arbitrary and didn't generalize across course
        sizes.
      - Recent confusion history disables review_mode (learning mode wins).
    """

    def __init__(self, brain) -> None:
        self.brain = brain

    async def __call__(self, state: TutorState) -> dict[str, Any]:
        start = time.time()
        slide = (state.get("last_slide_content") or "").strip()

        # Local review-mode decision: do we already master the slide's chunks?
        # No global magic threshold — we look at the chunks the retriever
        # produced for this exact context. If they're all mastered AND
        # there's no recent confusion, this is review territory.
        review_mode = False
        student_id = state.get("student_id")
        confusion_history = state.get("confusion_history") or []
        recent_confusion = bool(confusion_history)
        chunks = state.get("retrieved_chunks") or []

        if student_id and not recent_confusion and chunks:
            mastery_scores = [
                c.get("mastery_score") for c in chunks
                if isinstance(c, dict) and c.get("mastery_score") is not None
            ]
            if mastery_scores and all(s >= 0.85 for s in mastery_scores):
                # Every chunk relevant to this slide is mastered → review.
                # Threshold 0.85 is the SAME definition of "mastered" used
                # across the codebase (MasteryRepo.list_mastered) — it's
                # the system-wide canonical mastery cutoff, not a local
                # decision. If we ever change "mastered", we change it
                # there once.
                review_mode = True
                log.info("planner: review_mode=True (all relevant chunks mastered)")

        section_title = state.get("section_title", "") or ""
        chapter_title = state.get("chapter_title", "") or ""
        lang_for_concept = (state.get("language") or "fr")[:2]

        # Resolve the slide's main concept. Three sources, in order:
        #
        #   1. Pre-populated ``main_concept_hint`` — caller already
        #      resolved it (cached, or supplied by a structured ingester).
        #   2. Vision LLM looking at the rendered slide image — handles
        #      arbitrary layouts (any heading colour/position, any
        #      language, hand-drawn boxes) without rules or vocabulary.
        #      Same multi-provider chain as the slide-description service
        #      (OpenAI gpt-4o-mini → Ollama LLaVA → skip), with disk
        #      cache and single-flight already in place.
        #   3. Structural heuristic over the flattened text — fallback
        #      for audio-only sessions, text-only imports, or when every
        #      vision provider is unreachable.
        main_concept = (state.get("main_concept_hint") or "").strip()
        if not main_concept:
            slide_image_path = (state.get("slide_image_path") or "").strip()
            if slide_image_path:
                log.info("planner: calling vision on %s", slide_image_path)
                try:
                    from services.vision_describe import extract_slide_concept
                    # Pass section + chapter titles as context so the
                    # vision model knows which headers repeat across
                    # slides and can focus on the slide-specific topic.
                    main_concept = (
                        await extract_slide_concept(
                            slide_image_path,
                            lang_for_concept,
                            section_title=section_title,
                            chapter_title=chapter_title,
                        )
                    ).strip()
                    if main_concept:
                        log.info("planner: ✅ vision concept = %r", main_concept[:40])
                    else:
                        log.info("planner: ⚠️ vision returned empty → fallback structurel")
                except Exception as exc:
                    log.info("planner: ❌ vision raised: %s → fallback structurel", str(exc)[:160])
            else:
                log.info("planner: ℹ️ no slide_image_path → fallback structurel")
        if not main_concept and slide:
            main_concept = _extract_main_concept(slide, section_title, chapter_title)
        if not main_concept:
            main_concept = section_title or "(à inférer du contenu)"

        if not slide:
            log.warning("planner: empty slide_content → fallback plan")
            return {
                "plan": _fallback_plan(state),
                "main_concept_hint": main_concept,
                "review_mode": review_mode,
                "timings": {**state.get("timings", {}), "planner": 0.0},
            }

        # Content-quality gate: if the slide is mostly placeholder labels
        # or extremely short, skip the planner LLM call entirely. Calling
        # it on thin content tends to produce ideas with placeholder
        # briefs ("Indice 1", "Méthode 2") that leak into the narration
        # as fake concept names. The neutral fallback plan lets Context
        # fetch chunks and Narrator ground on those + the slide directly.
        if _is_slide_too_thin(slide):
            log.warning(
                "planner: slide too thin (%d words) → fallback plan, skipping LLM call",
                len(re.findall(r"\w+", slide, flags=re.UNICODE)),
            )
            return {
                "plan": _fallback_plan(state),
                "main_concept_hint": main_concept,
                "review_mode": review_mode,
                "timings": {**state.get("timings", {}), "planner": round(time.time() - start, 3)},
            }

        lang = (state.get("language") or "fr")[:2]
        template = _PLANNER_PROMPT_FR if lang == "fr" else _PLANNER_PROMPT_EN
        prompt_body = template.format(
            chapter=state.get("chapter_title", "") or "(unspecified)",
            section=section_title or "(unspecified)",
            main_concept=main_concept,
            slide_content=slide[:2000],
        )

        # 🧠 Personalization fragment (style + pace + depth + tone) —
        # scoped per (student, course) so the planner adapts to the
        # current course's learning pattern, not a blended global one.
        personalization_prefix = ""
        try:
            from pedagogy.personalization.engine import PersonalizationEngine
            if student_id:
                pers_ctx = await PersonalizationEngine.get_context(
                    student_id,
                    course_id=state.get("course_id"),
                )
                personalization_prefix = PersonalizationEngine.build_prompt_prefix(pers_ctx, lang=lang)
        except Exception as exc:
            log.debug(f"planner personalization fetch failed: {exc}")

        prompt = personalization_prefix + prompt_body
        raw, _duration = self.brain.ask(prompt, reply_language=lang)
        payload = _extract_json(raw)

        if not payload or not isinstance(payload.get("ideas"), list):
            log.warning("planner: invalid LLM output, using fallback. raw=%r", raw[:160])
            plan = _fallback_plan(state)
        else:
            ideas = [_coerce_idea(it, i) for i, it in enumerate(payload["ideas"]) if isinstance(it, dict)]
            if not ideas:
                ideas = _fallback_plan(state).ideas
            # En review_mode : on garde 1-2 ideas seulement (rappel rapide)
            if review_mode and len(ideas) > 2:
                ideas = ideas[:2]
                log.info("planner: review_mode → narrowing plan to 2 ideas")
            plan = PresentationPlan(
                section_title=state.get("section_title", "") or "",
                chapter_title=state.get("chapter_title", "") or "",
                ideas=ideas,
            )

        log.info(
            "planner: %d ideas planned (lang=%s, review_mode=%s, concept=%r)",
            len(plan.ideas), lang, review_mode, main_concept[:40],
        )
        return {
            "plan": plan,
            "main_concept_hint": main_concept,
            "review_mode": review_mode,
            "timings": {**state.get("timings", {}), "planner": round(time.time() - start, 3)},
        }
