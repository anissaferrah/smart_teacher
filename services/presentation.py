import asyncio
import logging
from typing import Callable

from deps import get_brain, get_voice, get_dialogue

log = logging.getLogger("SmartTeacher.services.presentation")


def split_sentences_with_spans(text: str) -> list[tuple[str, tuple[int, int]]]:
    import re

    spans: list[tuple[str, tuple[int, int]]] = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|\Z)", text, flags=re.S):
        raw = match.group()
        sentence = raw.strip()
        if not sentence:
            continue
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        spans.append((sentence, (match.start() + left_trim, match.start() + right_trim)))

    if not spans and text.strip():
        stripped = text.strip()
        start = text.find(stripped)
        spans.append((stripped, (max(0, start), max(0, start) + len(stripped))))

    return spans


async def synthesize_cached_tts(
    text_to_speak: str,
    *,
    language_code: str,
    rate: str | None = None,
    cache_scope: str = "presentation",
    session_id: str = "",
    confusion_score: float = 0.0,
) -> tuple[bytes | None, float, str, str, str | None, bool]:
    """Generate TTS once and reuse cached audio for repeated phrases.

    If `rate` is None and `session_id` is provided, the speech rate is
    derived adaptively from the student profile + confusion_score
    (via personalization.tts_adapter.get_edge_tts_rate).
    Otherwise, falls back to the explicit `rate` param or '+0%'.

    Lookup order:
      1. dialogue.load_tts_phrase_cache (Redis-backed phrase cache, cross-session)
      2. voice.generate_audio_async (live synthesis)

    On miss, generated audio is cached for all known voice signatures.

    Returns (audio_bytes, duration_s, engine, voice_name, mime, was_cached).
    """
    if not text_to_speak or not text_to_speak.strip():
        return None, 0.0, "none", "none", None, False

    voice = get_voice()
    dialogue = get_dialogue()
    sid_tag = (session_id[:8] if session_id else "anon")

    # ✨ Adaptive TTS rate from personalization profile (when session_id available)
    if rate is None:
        if session_id:
            from pedagogy.personalization.tts_adapter import get_edge_tts_rate
            rate = await get_edge_tts_rate(session_id, confusion_score=confusion_score)
        else:
            rate = "+0%"

    cache_signatures = []
    try:
        cache_signatures = voice.get_cache_signatures(language_code)
    except Exception:
        cache_signatures = []

    if not cache_signatures:
        request_provider = getattr(voice, "provider", "edge") or "edge"
        request_voice_name = getattr(voice, "voice_name", None) or getattr(voice, "voice_id", "none") or "none"
        cache_signatures = [(request_provider, request_voice_name)]

    cached = None
    for request_provider, request_voice_name in cache_signatures:
        try:
            cached = await dialogue.load_tts_phrase_cache(
                text_to_speak,
                language=language_code,
                rate=rate,
                provider=request_provider,
                voice_name=request_voice_name,
            )
        except Exception as cache_exc:
            log.debug(f"[{sid_tag}] TTS cache lookup skipped: {cache_exc}")
            cached = None
        if cached and cached.get("audio_bytes"):
            break

    if cached and cached.get("audio_bytes"):
        log.info(
            f"[{sid_tag}] ♻️ TTS cache hit | scope={cache_scope} | "
            f"provider={cached.get('provider')} | voice={cached.get('voice_name')}"
        )
        return (
            cached.get("audio_bytes"),
            0.0,
            cached.get("provider") or request_provider,
            cached.get("voice_name") or request_voice_name,
            cached.get("mime"),
            True,
        )

    audio_bytes, duration_s, engine_name, voice_name, mime_type = await voice.generate_audio_async(
        text_to_speak,
        language_code=language_code,
        rate=rate,
    )

    if audio_bytes:
        try:
            cache_targets = list(cache_signatures)
            cache_targets.append((engine_name or cache_signatures[0][0], voice_name or cache_signatures[0][1]))
            seen_targets: set[tuple[str, str]] = set()
            for provider_name, voice_label in cache_targets:
                target = (provider_name, voice_label)
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                await dialogue.save_tts_phrase_cache(
                    text_to_speak,
                    audio_bytes,
                    language=language_code,
                    rate=rate,
                    provider=provider_name,
                    voice_name=voice_label,
                    mime=mime_type or "audio/mpeg",
                    metadata={
                        "scope": cache_scope,
                        "engine": engine_name,
                        "voice_name": voice_name,
                        "request_signatures": cache_signatures,
                    },
                )
        except Exception as cache_exc:
            log.debug(f"[{sid_tag}] TTS cache save skipped: {cache_exc}")

    return audio_bytes, duration_s, engine_name, voice_name, mime_type, False


def _last_complete_sentence_before(narration: str, char_offset: int) -> str:
    """Return the last fully-spoken sentence ending at or before ``char_offset``.

    The boundary regex matches "[.!?…] + whitespace". A complete sentence is
    everything up to a boundary inside [:char_offset]. Anything between the
    last boundary and char_offset is a PARTIAL sentence (the one the student
    was hearing when they paused) — we don't include it.
    """
    import re as _re
    if not narration or char_offset <= 0:
        return ""
    text_before = narration[:char_offset]
    sent_ends = [m.end() for m in _re.finditer(r"[.!?…](?:\s|\n)+", text_before)]
    if not sent_ends:
        return ""   # student hasn't heard a complete sentence yet
    last_end = sent_ends[-1]
    if len(sent_ends) == 1:
        # Only one complete sentence before the cursor: return [0..last_end].
        return text_before[:last_end].strip()
    # Two or more boundaries → return the last complete sentence between
    # the previous-to-last and last boundary.
    return text_before[sent_ends[-2]:last_end].strip()


def resume_at_clean_boundary(narration: str, char_offset: int, language: str = "fr") -> str:
    """Resume forward of ``char_offset``, advancing past the partial
    sentence the student was hearing.

    # Pedagogical principle (changed from rewind to forward-skip)

    Real teachers don't replay what the student already heard. They move
    on. So when resuming we :

      1. Find the FORWARD sentence boundary right after ``char_offset``
         (not backward — backward repeats content the student just heard).
      2. Return the narration starting at that boundary.

    The optional pedagogical recap ("We were just covering X — let's
    continue.") is a SEPARATE concern handled by
    :func:`build_resume_recap` so the cursor in the caller stays
    correctly aligned with positions in ``narration`` (a recap prepended
    in-line would break ``effective_resume + end`` arithmetic).

    Edge cases :
      - char_offset = 0          → first run, return narration unchanged.
      - char_offset >= len(...)  → slide already finished, return "".
      - mid-final-sentence       → no forward boundary; return raw slice
                                   from char_offset (a few words at most).
    """
    import re as _re
    if not narration:
        return ""
    if char_offset >= len(narration):
        return ""
    if char_offset <= 0:
        return narration

    text_after = narration[char_offset:]
    forward_match = _re.search(r"[.!?…](?:\s|\n)+", text_after)
    if forward_match:
        forward_start = char_offset + forward_match.end()
    else:
        # No more sentence boundaries → student paused inside the last
        # sentence; just give them the remaining tail.
        forward_start = char_offset

    if forward_start >= len(narration):
        return ""
    return narration[forward_start:].lstrip()


def build_resume_recap(narration: str, char_offset: int, language: str = "fr") -> str:
    """Build a one-sentence pedagogical recap of what the student last
    heard, to be spoken BEFORE the resumed narration.

    Returns "" when there's nothing meaningful to recap (e.g. very early
    pause, single short sentence). The caller plays this string through
    TTS as a separate chunk and does NOT advance the narration cursor
    while it plays — that way the cursor stays aligned with positions in
    the original narration.

    # Examples

      narration = "We will study X. Y is critical here. The next..."
      char_offset = 35  (inside "Y is critical here.")
        → "We were just covering: We will study X. Let's continue. "

      char_offset = 0
        → ""  (first time, no recap needed)
    """
    if not narration or char_offset <= 0:
        return ""
    last_sentence = _last_complete_sentence_before(narration, char_offset)
    if not last_sentence:
        return ""
    words = last_sentence.split()
    if len(words) < 5:
        # Recap on a 4-word sentence ("OK so we move on.") gives no value.
        return ""
    if len(words) > 25:
        last_sentence = " ".join(words[:25]) + "…"
    if language.lower().startswith("fr"):
        return f"On était en train de voir : {last_sentence} Continuons. "
    return f"We were just covering: {last_sentence} Let's continue. "


# Backwards-compat alias for the old name (keeps existing imports working).
resume_at_next_idea = resume_at_clean_boundary


async def explain_slide_focused(
    slide_content: str,
    chapter_idx: int = 0,
    chapter_title: str = "",
    section_title: str = "",
    language: str = "fr",
    student_level: str = "lycée",
    course_summary: str = "",
    is_resume: bool = False,
    session_id: str | None = None,
    course_id: str = "",
    section_idx: int = 0,
    on_plan_ready: Callable | None = None,
    max_sentences: int = 4,  # backward-compat (test uses it); the Teaching Graph self-regulates length
    teaching_graph=None,  # passed from main.py (LangGraph instance)
    student_id: str | None = None,  # ✨ Auth user — pour adapter au learning_style
    slide_image_path: str = "",  # ← NEW: rendered slide PNG → vision-based concept extraction
) -> str:
    """Expliquer un slide via Teaching Graph (Planner→Narrator) avec cross-session cache.

    Pipeline :
      1. Cache cross-session Redis (sauf en mode resume)
      2. Teaching Graph (LangGraph) avec callback on_plan_ready
      3. Fallback brain.ask
      4. Persistance cache cross-session
      5. Background : génération de 5 idées de présentation
    """
    from cache.slide_cache import get_narration as cache_get_narration
    from cache.slide_cache import set_narration as cache_set_narration

    brain = get_brain()
    sid_tag = (session_id or "na")[:8]

    # ── Cross-session narration cache (Redis) ─────────────────────────
    # NOTE: we ALSO check this on `is_resume=True`. Earlier the resume path
    # skipped the cache entirely and went straight to brain.ask, which
    # regenerated the whole narration from scratch — causing the system to
    # "re-explain everything" on every resume. The slicing at the next idea
    # boundary is then done by the caller (handlers/ws.py:run_presentation)
    # using the cached cursor.
    if course_id:
        try:
            cached_narration = await cache_get_narration(
                course_id=course_id,
                chapter_idx=chapter_idx,
                section_idx=section_idx,
                language=language,
                student_level=student_level,
            )
            if cached_narration:
                log.info(
                    f"[{sid_tag}] 💾 Cross-session HIT "
                    f"(course={course_id[:16]} ch={chapter_idx} sec={section_idx} "
                    f"lang={language} level={student_level} resume={is_resume}) → 0 LLM call"
                )
                return cached_narration
        except Exception as cache_exc:
            log.debug(f"Cross-session cache lookup failed: {cache_exc}")

    # ✨ Resolve learning style via 2-level cache (in-process + Redis, single-flight).
    # Returns the prose hint only — the structured-params mapping
    # (max_sentences=5, must_include=…) was removed because those numbers
    # were arbitrary heuristics with no empirical backing.
    learning_style_hint = ""
    if student_id:
        try:
            import uuid as _uuid_mod
            from pedagogy.personalization.learning_style.heuristic import style_to_prompt_hint
            from pedagogy.personalization.learning_style.bayes import load_posterior
            from services.learning_style_cache import get_or_compute

            async def _recompute() -> dict:
                sid_uuid = _uuid_mod.UUID(student_id)
                posterior = await load_posterior(sid_uuid)
                dominant = posterior.dominant()
                return {
                    "hint":     style_to_prompt_hint(dominant, lang=language),
                    "dominant": dominant,
                    "source":   "bayes",
                }

            cached = await get_or_compute(student_id, language, _recompute)
            if cached is not None:
                learning_style_hint = cached.get("hint", "") or ""
                log.debug(
                    f"[{sid_tag}] style={cached.get('dominant')} (source={cached.get('source')})"
                )
        except Exception as exc:
            log.debug(f"learning style lookup skipped: {exc}")

    # Continuity context: the previous slide's main concept and 1-2
    # sentence recap, so the narrator can open with a bridge instead of
    # restarting cold. Read from SessionContext (populated at the end of
    # the previous successful run, below). First-slide-of-session has
    # both empty → narrator simply skips the bridge block.
    previous_concept = ""
    previous_narration_summary = ""
    previous_slide_content = ""
    if session_id:
        try:
            from deps import get_dialogue
            _dialogue = get_dialogue()
            _ctx = await _dialogue.get_session(session_id)
            if _ctx is not None:
                previous_concept = _ctx.last_concept_explained or ""
                previous_narration_summary = _ctx.last_narration_summary or ""
                # Raw text of the previously narrated slide. Used by the
                # narrator to detect near-duplicate slides (consecutive
                # pages with the same content but a tiny variation —
                # e.g. one bullet bolded, a single word changed). When
                # high similarity is detected, the narrator focuses on
                # the diff instead of restating the whole explanation.
                previous_slide_content = _ctx.last_slide_explained or ""
        except Exception as exc:
            log.debug(f"[{sid_tag}] previous-slide lookup skipped: {exc}")

    try:
        # ── Teaching Graph (LangGraph: Planner → Narrator) ────────────
        if not is_resume and teaching_graph is not None:
            try:
                # Diagnostic: surface whether the planner will get an
                # image to call vision against. Empty path = no vision,
                # falls to structural extractor silently.
                if slide_image_path:
                    log.info(
                        f"[{sid_tag}] 🎯 vision input: slide_image_path={slide_image_path!r}"
                    )
                else:
                    log.info(
                        f"[{sid_tag}] ⚠️ no slide_image_path → planner will skip vision"
                    )

                initial_state = {
                    "session_id": session_id or "anonymous",
                    "course_id": course_id or "",
                    "language": language[:2] if language else "fr",
                    "chapter_idx": chapter_idx,
                    "chapter_title": chapter_title,
                    "section_idx": section_idx,
                    "section_title": section_title,
                    "last_slide_content": slide_content,
                    # Slide image for vision-based concept extraction.
                    # Empty string when caller has no rendered PNG (audio-
                    # only sessions, raw-text imports). Planner falls back
                    # to text-structural heuristic in that case.
                    "slide_image_path":            slide_image_path or "",
                    "event_type": "present_section",
                    "event_payload": {"is_resume": False},
                    "domain": None,
                    "student_level": student_level,
                    "learning_style_hint":   learning_style_hint,    # prose hint
                    # Continuity (empty on first slide of session).
                    "previous_concept":            previous_concept,
                    "previous_narration_summary":  previous_narration_summary,
                    "previous_slide_content":      previous_slide_content,
                }
                final_state = dict(initial_state)
                async for update_chunk in teaching_graph.astream(initial_state, stream_mode="updates"):
                    if not isinstance(update_chunk, dict):
                        continue
                    for node_name, updates in update_chunk.items():
                        if isinstance(updates, dict):
                            final_state.update(updates)
                        if node_name == "planner" and on_plan_ready and isinstance(updates, dict):
                            plan_obj = updates.get("plan")
                            if plan_obj is not None:
                                try:
                                    await on_plan_ready(plan_obj)
                                except Exception as cb_exc:
                                    log.debug(f"on_plan_ready callback failed: {cb_exc}")
                narration = (final_state.get("answer") or "").strip()
                if narration:
                    timings = final_state.get("timings") or {}
                    log.info(
                        "✅ Teaching Graph response | planner=%.1fs narrator=%.1fs | %d chars",
                        timings.get("planner", 0.0),
                        timings.get("narrator", 0.0),
                        len(narration),
                    )
                    # Persist continuity context for the *next* slide.
                    # Stored on SessionContext; read on entry (above) by
                    # the next call to explain_slide_focused for this
                    # session. Failure here is non-fatal — worst case the
                    # next slide opens without a bridge.
                    if session_id:
                        try:
                            from deps import get_dialogue
                            _dialogue = get_dialogue()
                            _ctx = await _dialogue.get_session(session_id)
                            if _ctx is not None:
                                new_concept = (final_state.get("main_concept_hint") or "").strip()
                                new_summary = (final_state.get("narration_summary") or "").strip()
                                if new_concept:
                                    _ctx.last_concept_explained = new_concept
                                if new_summary:
                                    _ctx.last_narration_summary = new_summary
                                _ctx.last_slide_explained = slide_content[:1000]
                                await _dialogue._save(_ctx)
                        except Exception as exc:
                            log.debug(f"[{sid_tag}] continuity save skipped: {exc}")
                    if course_id:
                        try:
                            await cache_set_narration(
                                course_id=course_id,
                                chapter_idx=chapter_idx,
                                section_idx=section_idx,
                                language=language,
                                student_level=student_level,
                                text=narration,
                                engine="teaching_graph",
                                meta={"timings": timings},
                            )
                        except Exception as cache_exc:
                            log.debug(f"Cross-session cache save skipped: {cache_exc}")
                    # Background : 5 idées de présentation
                    try:
                        asyncio.create_task(_bg_generate_slide_ideas(
                            slide_text=slide_content,
                            title=chapter_title,
                            language=language,
                            student_level=student_level,
                            session_id=session_id or "",
                            chapter_title=chapter_title,
                        ))
                    except Exception:
                        pass
                    return narration
                log.warning("⚠️ Teaching Graph returned empty narration → fallback brain.ask")
            except Exception as graph_exc:
                log.warning(
                    "⚠️ Teaching Graph failed (%s): %s → fallback brain.ask",
                    type(graph_exc).__name__,
                    str(graph_exc)[:120],
                )

        # ── Fallback : brain.ask direct ───────────────────────────────
        try:
            response, duration = await asyncio.to_thread(
                brain.ask,
                question=slide_content[:100],
                course_context=slide_content,
                reply_language=language,
                chapter_idx=chapter_idx,
                chapter_title=chapter_title,
                section_title=section_title,
                domain=None,
                session_id=session_id,
            )
        except Exception as exc:
            log.error(f"brain.ask failed: {exc}")
            return f"Erreur lors de l'explication : {exc}"

        log.info(f"✅ LLM focused response ({duration:.1f}s) | {len(response)} chars")
        if not is_resume and course_id and response:
            try:
                await cache_set_narration(
                    course_id=course_id,
                    chapter_idx=chapter_idx,
                    section_idx=section_idx,
                    language=language,
                    student_level=student_level,
                    text=response,
                    engine="brain.ask",
                )
            except Exception as cache_exc:
                log.debug(f"Cross-session cache save skipped: {cache_exc}")
        return response

    except Exception as e:
        log.error(f"❌ Error in explain_slide_focused: {e}")
        return f"Erreur lors de l'explication : {str(e)}"


async def _bg_generate_slide_ideas(
    slide_text: str,
    title: str | None,
    language: str,
    student_level: str,
    session_id: str,
    chapter_title: str,
) -> None:
    """Background task : génère 5 idées courtes pour la slide via Brain.present."""
    try:
        import re
        from observability.dashboard import record_checkpoint_event as _rec
        brain = get_brain()
        text, _ = await asyncio.to_thread(
            brain.present, slide_text,
            language=(language or "fr"),
            student_level=student_level,
            chapter_title=title or "",
        )
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        ideas = sentences[:5]
        _rec({
            "session_id": session_id,
            "slide_title": title or chapter_title or "",
            "ideas": ideas,
            "source": "auto_slide_ideas",
        })
    except Exception as exc:
        log.debug(f"Slide ideas bg task failed: {exc}")
