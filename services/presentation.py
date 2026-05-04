import asyncio
import logging
from typing import Callable

from deps import get_brain, get_voice, get_dialogue

log = logging.getLogger("SmartTeacher.services.presentation")


def compute_text_cursor_from_audio_progress(
    audio_progress: float | int | None,
    narration_text: str,
) -> int | None:
    """Convert a 0..1 audio playback fraction into a text char offset.

    The frontend reports ``currentAudio.currentTime / currentAudio.duration``
    when the student interrupts mid-playback. The backend uses this to
    set the saved cursor to the *actual audio position*, instead of the
    text-streaming position which is always ``len(narration)`` once
    streaming finished (the operator-visible bug : pause near end →
    cursor saved as ``len`` even though audio still had several seconds
    left to play).

    Linear approximation : assumes constant chars-per-second over the
    whole narration. TTS speech rate isn't perfectly uniform (longer
    words, technical terms, pauses around punctuation), but the error
    is small enough to land within the same sentence boundary, which
    is what the downstream forward-skip / rewind helpers operate on.

    Returns :
      - ``int`` cursor in [0, len(narration_text)] when ``audio_progress``
        is a usable float in (0, 1].
      - ``None`` when the input is missing, NaN, infinite, or out of
        range — caller should fall back to its existing cursor logic.
    """
    if audio_progress is None or not narration_text:
        log.info(
            "🔇 audio_progress→cursor SKIP | progress=%s narration_len=%d (no input)",
            audio_progress, len(narration_text or ""),
        )
        return None
    try:
        progress = float(audio_progress)
    except (TypeError, ValueError):
        log.info("🔇 audio_progress→cursor SKIP | invalid_float=%r", audio_progress)
        return None
    # NaN, infinity, negative, or impossible >1 → reject ; the caller
    # already has a sane fallback.
    if progress != progress:  # NaN check (NaN != NaN)
        log.info("🔇 audio_progress→cursor SKIP | NaN")
        return None
    if progress <= 0.0 or progress > 1.0:
        log.info(
            "🔇 audio_progress→cursor SKIP | out_of_range progress=%.4f (audio not yet started or invalid)",
            progress,
        )
        return None
    cursor = int(round(progress * len(narration_text)))
    cursor = max(0, min(cursor, len(narration_text)))
    log.info(
        "🎯 audio_progress→cursor | progress=%.4f × narration_len=%d = cursor=%d (%.1f%% played)",
        progress, len(narration_text), cursor, progress * 100.0,
    )
    return cursor


def current_sentence_span(narration: str, cursor: int) -> tuple[int, int, str]:
    """Return ``(start, end, text)`` of the sentence containing ``cursor``.

    ``start`` is the offset of the sentence's first character (= what
    ``rewind_to_current_sentence_start`` returns). ``end`` is the
    offset right AFTER the sentence's terminating punctuation + any
    trailing whitespace, so ``narration[end:]`` starts cleanly with
    the next sentence.

    Used by the ``REEXPLAIN_AND_CONTINUE`` resume strategy : the LLM
    re-explains the slice ``narration[start:end]`` in different words,
    and the WS handler then plays the remainder ``narration[end:]``
    verbatim from the cached narration.

    Edge cases :
      - empty narration  → (0, 0, "")
      - cursor <= 0      → (0, end_of_first, narration[0:end_of_first])
      - cursor >= len    → last sentence's span
      - no terminator    → whole narration is one "sentence"
    """
    if not narration:
        log.info("🎯 sentence_span | empty narration → (0,0,'')")
        return (0, 0, "")
    n = len(narration)
    # Clamp cursor into [0, n-1]
    pos = max(0, min(cursor, n - 1))

    # Sentence start (re-use the existing helper).
    start = rewind_to_current_sentence_start(narration, pos)

    # Sentence end : look forward from ``start`` for the next inter-
    # sentence boundary (terminator + whitespace), or end-of-narration.
    import re as _re
    forward = _re.search(
        r"(?:[.!?…؟](?:\s|\n)+|[。！？](?=[\s\S]))",
        narration[start:],
    )
    if forward:
        end = start + forward.end()
    else:
        # Fall through : no terminator after start → take everything
        # to end-of-narration.
        end = n
    end = max(start, min(end, n))
    sent = narration[start:end].strip()
    log.info(
        "🎯 sentence_span | cursor=%d → start=%d end=%d (len=%d) | preview='%s'",
        cursor, start, end, end - start,
        sent[:80] + ("..." if len(sent) > 80 else ""),
    )
    return (start, end, sent)


def rewind_to_current_sentence_start(narration: str, cursor: int) -> int:
    """Return the char offset of the start of the sentence containing ``cursor``.

    Used on resume after a short interruption : the student wants to hear
    the FULL idea/sentence they were on, from its beginning, rather than
    losing the partial sentence to a forward-skip.

    Search backward from ``cursor`` for the nearest preceding sentence
    terminator (``[.!?…]\\s+``). The char right after that whitespace is
    the start of the current sentence. If no terminator is found before
    ``cursor``, the current sentence is the FIRST one — return 0.

    Edge cases :
      - cursor <= 0           → return 0 (already at start)
      - cursor >= len(text)   → cursor was at end of text ; rewind to
                                start of last sentence (delegate to
                                rewind_to_last_sentence_start to keep
                                the two helpers consistent)
      - empty narration       → 0
    """
    if not narration:
        log.info("🎯 rewind_to_current_sentence_start | empty → 0")
        return 0
    if cursor <= 0:
        log.info("🎯 rewind_to_current_sentence_start | cursor<=0 → 0 (already at start)")
        return 0
    if cursor >= len(narration):
        last = rewind_to_last_sentence_start(narration)
        log.info(
            "🎯 rewind_to_current_sentence_start | cursor=%d >= len=%d → last_sentence_start=%d",
            cursor, len(narration), last,
        )
        return last

    import re as _re
    # Look at the slice [0:cursor] and find the LAST inter-sentence
    # boundary in it. The sentence we're in starts right after.
    head = narration[:cursor]
    terminators = list(_re.finditer(r"(?:[.!?…؟]\s+|[。！？](?=[\s\S]))", head))
    if terminators:
        start = terminators[-1].end()
        log.info(
            "🎯 rewind_to_current_sentence_start | cursor=%d → rewound to sentence start=%d (skipped back %d chars across %d boundaries)",
            cursor, start, cursor - start, len(terminators),
        )
        return start
    log.info(
        "🎯 rewind_to_current_sentence_start | cursor=%d but no prior terminator → 0 (in first sentence)",
        cursor,
    )
    return 0


def rewind_to_last_sentence_start(narration: str) -> int:
    """Return the char offset of the LAST sentence's start.

    Used when the saved cursor is at or past the end of the narration
    (text streaming finished, but TTS audio may still be playing). On
    resume we want to replay only the LAST sentence — not the whole
    slide (legacy "restart from 0" caused operator-visible 3x repeats),
    not nothing (TTS may not have actually finished playing yet).

    Implementation : ``re.finditer(r"[.!?…]\\s+", narration)`` returns
    the INTER-sentence boundaries (terminator + trailing whitespace).
    The trailing terminator on the very last sentence has no whitespace
    after it, so it doesn't appear in this list — meaning N inter-
    boundaries correspond to N+1 sentences. The LAST inter-boundary
    is the start of the last sentence.

    Returns :
        offset >= 0 within ``narration``. Empty / single-sentence input
        returns 0 (replay from start). All-no-punctuation input returns
        max(0, len - 100) so the student hears at least the tail.
    """
    if not narration:
        return 0
    import re as _re
    terminators = list(_re.finditer(r"(?:[.!?…؟]\s+|[。！？](?=[\s\S]))", narration))
    if terminators:
        return terminators[-1].end()
    # No sentence breaks at all → rewind ~100 chars from the end so the
    # student gets at least the tail of the narration.
    return max(0, len(narration) - 100)


def decide_narration_cache_reuse(
    *,
    requested_slide_key: tuple,
    current_presentation_key: tuple | None,
    current_presentation_text: str,
    cached_snapshot: dict | None,
    cached_pause_state: dict | None,
) -> tuple[bool, str, str, int]:
    """Pure decision : should we reuse cached narration for this slide ?

    Three independent cache sources, ANY of which is enough :

      (1) **In-memory** : we just generated this slide in the current session
          and ``current_presentation_text`` is still RAM-resident.
      (2) **Redis snapshot keyed by THIS slide** : ``cached_snapshot`` was
          loaded via ``dialogue.load_presentation_snapshot(session_id,
          requested_slide_id)``. Its text/cursor *belong to this slide* by
          construction, regardless of which slide is currently paused.
      (3) **Paused-state fallback** : the live ``ctx.paused_state`` happens
          to refer to the slide we're requesting. Only useful in cold-Redis
          edge cases where (2) didn't return.

    Why this function exists as a pure helper :
    the previous monolithic version inside ``handlers/ws.py`` gated path
    (2) on ``cached_pause_state.slide_id == requested_slide_id``, which is
    wrong — the paused_state holds the LAST PAUSED slide, not the current
    one. After a pause on slide 4 and a back-navigation to slide 0, the
    Redis snapshot for slide 0 was ignored and the LLM regenerated a
    60-300s narration the student had already heard. Extracting the
    decision here lets us unit-test that exact sequence (see
    ``tests/test_presentation_cache_reuse.py``).

    Returns:
        (reuse, source, text, cursor)

        - ``reuse``  : whether to skip the LLM call.
        - ``source`` : "memory" | "redis" | "paused_state" | "miss".
        - ``text``   : the narration text to use (may be empty on miss).
        - ``cursor`` : char position to resume from (clamped to len(text)).
    """
    snap = cached_snapshot or {}
    pause = cached_pause_state or {}
    snapshot_text = str(snap.get("presentation_text") or "")
    pause_text = str(pause.get("presentation_text") or "")
    pause_slide_id = str(
        pause.get("presentation_key") or pause.get("slide_id") or ""
    )
    requested_slide_id = ":".join(str(p) for p in requested_slide_key)
    log.info(
        "📍 cache_decide INPUT | requested=%s | mem_key=%s mem_text=%dch | "
        "snap_text=%dch snap_cursor=%s | pause_slide=%s pause_text=%dch pause_cursor=%s",
        requested_slide_id,
        ":".join(str(p) for p in (current_presentation_key or ())) or "none",
        len(current_presentation_text or ""),
        len(snapshot_text), snap.get("presentation_cursor"),
        pause_slide_id or "none", len(pause_text),
        pause.get("presentation_cursor") if pause.get("presentation_cursor") is not None
            else pause.get("char_offset"),
    )

    # Helper : extract the resume cursor from paused_state IF and only IF
    # the paused state's slide matches the slide we're about to present.
    # Without this guard, the in-memory branch would always return
    # ``len(text)`` (= "play the whole slide again"), discarding the
    # actual pause position the WS handler stored via the audio_progress
    # override. That was the operator-visible bug : "Interrupt cursor
    # override 719 → 366" then on resume "raw=719" — the 366 was lost
    # because the cache helper returned ``len(text)``.
    def _paused_cursor_if_same_slide() -> int | None:
        if pause_slide_id != requested_slide_id:
            return None
        raw = (
            pause.get("presentation_cursor")
            if pause.get("presentation_cursor") is not None
            else pause.get("char_offset")
        )
        if raw is None:
            return None
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return None

    # (1) In-memory hit. If paused_state has a cursor for THIS slide,
    # use it ; otherwise fall back to len(text) (legacy behaviour for
    # the first present_section after a fresh narration).
    if (current_presentation_key == requested_slide_key
            and current_presentation_text):
        text = current_presentation_text
        paused_cursor = _paused_cursor_if_same_slide()
        if paused_cursor is not None:
            final = min(paused_cursor, len(text))
            log.info(
                "📍 cache_decide HIT memory + paused | text_len=%d cursor=%d (%.1f%%) | resume from paused position",
                len(text), final, (final / len(text) * 100.0) if len(text) else 0.0,
            )
            return True, "memory", text, final
        log.info(
            "📍 cache_decide HIT memory (no paused match) | text_len=%d cursor=len (%d) | full replay",
            len(text), len(text),
        )
        return True, "memory", text, len(text)

    # (2) Redis snapshot for this exact slide. Snapshot cursor wins
    # because it's per-slide ; but if paused_state has a more recent
    # cursor for the same slide (e.g. user paused after the snapshot
    # save), prefer the paused one.
    if snapshot_text:
        snap_cursor = max(0, int(snap.get("presentation_cursor") or 0))
        paused_cursor = _paused_cursor_if_same_slide()
        if paused_cursor is not None and paused_cursor < snap_cursor:
            final = min(paused_cursor, len(snapshot_text))
            log.info(
                "📍 cache_decide HIT redis | text_len=%d snap_cursor=%d but paused=%d (more recent, before snap) → cursor=%d",
                len(snapshot_text), snap_cursor, paused_cursor, final,
            )
            return True, "redis", snapshot_text, final
        # Revisit-completed-slide guard. When the user navigates BACK to
        # a slide they previously finished (cursor saved at/near the end
        # of the narration) AND there is no active pause on this slide
        # (paused_cursor is None), they want to hear the slide again from
        # the start — not resume from the very end (which means silence,
        # since the TTS has nothing left to play).
        # Operator-visible bug : student clicks "previous slide" → snap
        # has cursor=len(text)=1918 → TTS resumes at index 1918 → no
        # audio plays → it looks like the slide was skipped.
        end_threshold = int(len(snapshot_text) * 0.95)
        revisit_completed = (
            paused_cursor is None
            and len(snapshot_text) > 0
            and snap_cursor >= end_threshold
        )
        if revisit_completed:
            log.info(
                "📍 cache_decide REVISIT | text_len=%d snap_cursor=%d (%.0f%%) ≥ 95%% AND no active pause → reset cursor=0 (replay slide from start)",
                len(snapshot_text), snap_cursor,
                (snap_cursor / max(1, len(snapshot_text))) * 100.0,
            )
            return True, "redis", snapshot_text, 0
        final = min(snap_cursor, len(snapshot_text))
        log.info(
            "📍 cache_decide HIT redis | text_len=%d cursor=%d (%.1f%%) | %s",
            len(snapshot_text), final,
            (final / len(snapshot_text) * 100.0) if len(snapshot_text) else 0.0,
            "snap wins (paused not before snap)" if paused_cursor is not None else "from snapshot",
        )
        return True, "redis", snapshot_text, final

    # (3) Paused-state happens to match (cold-cache fallback)
    if pause_text and pause_slide_id == requested_slide_id:
        pause_cursor = max(0, int(
            pause.get("presentation_cursor")
            or pause.get("char_offset")
            or 0
        ))
        final = min(pause_cursor, len(pause_text))
        log.info(
            "📍 cache_decide HIT paused_state (cold-cache fallback) | text_len=%d cursor=%d",
            len(pause_text), final,
        )
        return True, "paused_state", pause_text, final

    log.info(
        "📍 cache_decide MISS | requested=%s | will regenerate via LLM",
        requested_slide_id,
    )
    return False, "miss", "", 0


# Multi-script sentence terminators :
#   - Latin       : .  !  ?  …
#   - CJK         : 。 ！ ？  (full-width forms used in Chinese/Japanese)
#   - Arabic      : ؟  (Arabic question mark ; period uses Latin .)
# Spanish inverted ¿¡ are NOT terminators — they OPEN a sentence.
_SENTENCE_TERMINATORS = ".!?…。！？؟"


def split_sentences_with_spans(text: str) -> list[tuple[str, tuple[int, int]]]:
    import re

    spans: list[tuple[str, tuple[int, int]]] = []
    # Pattern : "non-terminator chars + (terminator+ | end-of-text)" — generalised
    # to multi-script terminators above.
    _pat = rf"[^{_SENTENCE_TERMINATORS}]+(?:[{_SENTENCE_TERMINATORS}]+|\Z)"
    for match in re.finditer(_pat, text, flags=re.S):
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
        cached_kb = len(cached.get("audio_bytes") or b"") / 1024.0
        log.info(
            f"[{sid_tag}] ♻️ TTS cache HIT | scope={cache_scope} | "
            f"provider={cached.get('provider')} voice={cached.get('voice_name')} | "
            f"chars={len(text_to_speak)} bytes={len(cached.get('audio_bytes') or b'')} ({cached_kb:.1f}KB) | rate={rate}"
        )
        return (
            cached.get("audio_bytes"),
            0.0,
            cached.get("provider") or request_provider,
            cached.get("voice_name") or request_voice_name,
            cached.get("mime"),
            True,
        )

    import time as _time
    _tts_t0 = _time.time()
    log.info(
        f"[{sid_tag}] 🔊 TTS GENERATE START | scope={cache_scope} | "
        f"chars={len(text_to_speak)} lang={language_code} rate={rate} | "
        f"text_preview='{text_to_speak[:60]}{'...' if len(text_to_speak) > 60 else ''}'"
    )
    audio_bytes, duration_s, engine_name, voice_name, mime_type = await voice.generate_audio_async(
        text_to_speak,
        language_code=language_code,
        rate=rate,
    )
    _tts_elapsed = _time.time() - _tts_t0
    _audio_kb = (len(audio_bytes) / 1024.0) if audio_bytes else 0.0
    _cps_synth = (len(text_to_speak) / _tts_elapsed) if _tts_elapsed > 0 else 0.0
    _cps_audio = (len(text_to_speak) / duration_s) if duration_s > 0 else 0.0
    log.info(
        f"[{sid_tag}] 🔊 TTS GENERATE DONE | took={_tts_elapsed:.2f}s | "
        f"audio_dur={duration_s:.2f}s engine={engine_name} voice={voice_name} | "
        f"bytes={len(audio_bytes or b'')} ({_audio_kb:.1f}KB) | "
        f"synth_speed={_cps_synth:.0f}ch/s | speech_speed={_cps_audio:.0f}ch/s"
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
    sent_ends = [m.end() for m in _re.finditer(r"(?:[.!?…؟](?:\s|\n)+|[。！？](?=[\s\S]))", text_before)]
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
    forward_match = _re.search(r"(?:[.!?…؟](?:\s|\n)+|[。！？](?=[\s\S]))", text_after)
    if forward_match:
        forward_start = char_offset + forward_match.end()
    else:
        # No more sentence boundaries → student paused inside the last
        # sentence; just give them the remaining tail.
        forward_start = char_offset

    if forward_start >= len(narration):
        return ""
    return narration[forward_start:].lstrip()


# ── Recap thresholds (operational, not pedagogical research)──────────
# Three buckets calibrated against typical classroom interruption lengths :
#   - 0-30s   : a quick clarification question. A long recap would
#               annoy the student ("yes I just heard that, get on with it").
#   - 30s-2min: a short tangent. A brief reminder of the topic helps
#               re-anchor without being patronising.
#   - 2min+   : full break (got coffee, took a phone call). The student
#               needs the full last-sentence recap to get back on track.
# These cutoffs are honest defaults — adjust once we have real
# observation data on resume-success rates per interruption duration.
_RECAP_SHORT_THRESHOLD_S = 30.0
_RECAP_LONG_THRESHOLD_S = 120.0


def build_resume_recap(
    narration: str,
    char_offset: int,
    language: str = "fr",
    *,
    interruption_duration_s: float | None = None,
) -> str:
    """Build a pedagogical resume prefix sized to how long the student paused.

    The recap is spoken BEFORE the resumed narration, as a SEPARATE TTS
    chunk so the narration cursor stays aligned with positions in
    ``narration`` (the caller doesn't advance the cursor while the
    recap plays).

    Three styles depending on ``interruption_duration_s`` :

      - **< 30s** ("quick"): just a continuity word. The student barely
        looked away — no recap needed, just a polite acknowledgement.
        Returns the spoken equivalent of "…je continue : " / "…I continue: ".

      - **30s - 2min** ("medium"): one-sentence reminder of the topic.
        Names the topic without re-reading the previous sentence.
        Useful after a tangent question, a quick break.

      - **>= 2min** OR ``None`` ("full"): full last-sentence recap.
        Re-reads the last fully-spoken sentence so the student can pick
        up the thread after a long break. This is the legacy behaviour ;
        ``None`` selects it as the safe default.

    Returns "" when ``char_offset == 0`` (first run) or when the only
    candidate sentence is too short to be useful (< 5 words).
    """
    if not narration or char_offset <= 0:
        return ""

    # Bucket selection ------------------------------------------------
    if interruption_duration_s is None:
        bucket = "full"
    elif interruption_duration_s < _RECAP_SHORT_THRESHOLD_S:
        bucket = "quick"
    elif interruption_duration_s < _RECAP_LONG_THRESHOLD_S:
        bucket = "medium"
    else:
        bucket = "full"

    is_fr = language.lower().startswith("fr")

    # Quick bucket : no recap content, just a continuity opener -------
    if bucket == "quick":
        return "Je continue. " if is_fr else "I continue. "

    # The other two buckets need a recap candidate.
    last_sentence = _last_complete_sentence_before(narration, char_offset)
    if not last_sentence:
        return "Je continue. " if is_fr else "I continue. "
    words = last_sentence.split()
    if len(words) < 5:
        # Recap on a 4-word sentence gives no value — fall back to the
        # quick continuity opener regardless of bucket.
        return "Je continue. " if is_fr else "I continue. "

    # Medium bucket : "we were on X" — extract a topic phrase --------
    # Heuristic : the first 6-8 words of the last sentence usually
    # carry the topic ("Le modèle vectoriel utilise des représentations…"
    # → "Le modèle vectoriel"). Truncate to ~6 words to keep the recap
    # short (a full re-read here would feel patronising for a 60s pause).
    if bucket == "medium":
        topic = " ".join(words[:6])
        if is_fr:
            return f"On était sur : {topic}… Continuons. "
        return f"We were on: {topic}… Let's continue. "

    # Full bucket : the legacy long recap (used after a real break) ---
    if len(words) > 25:
        last_sentence = " ".join(words[:25]) + "…"
    if is_fr:
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
