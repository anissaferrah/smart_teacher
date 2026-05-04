"""Session utilities and common helpers for WebSocket and REST handlers."""

import functools
import logging
import tempfile
import os

import numpy as np
from langdetect import detect, DetectorFactory

from core.config import Config

# Determinism for the langdetect fallback path.
DetectorFactory.seed = 0

log = logging.getLogger("SmartTeacher.SessionManager")


# lingua is a statistical n-gram detector built specifically to handle
# short queries reliably (langdetect classifies "can you give us more
# examples" as French 10/10 times). We build the EN/FR detector lazily
# to keep import time low.
_lingua_detector = None


def _get_lingua():
    global _lingua_detector
    if _lingua_detector is not None:
        return _lingua_detector
    try:
        from lingua import Language, LanguageDetectorBuilder
        _lingua_detector = (
            LanguageDetectorBuilder.from_languages(
                Language.ENGLISH, Language.FRENCH
            ).build()
        )
    except Exception as exc:
        log.warning(f"lingua unavailable, falling back to langdetect: {exc}")
        _lingua_detector = False
    return _lingua_detector


def detect_lang_text(text: str) -> str:
    """Detect language (fr/en) reliably, including on short queries."""
    if not text or not text.strip():
        return "en"
    detector = _get_lingua()
    if detector:
        try:
            result = detector.detect_language_of(text)
            if result is not None:
                code = result.iso_code_639_1.name.lower()
                return "fr" if code == "fr" else "en"
        except Exception as exc:
            log.debug(f"lingua detection failed: {exc}")
    try:
        code = detect(text)
        return "fr" if code.startswith("fr") else "en"
    except Exception:
        return "en"


# def audio_bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
#     """Convert audio bytes (WebM) to numpy float32 array."""
#     log.info(f"🔊 audio_bytes_to_numpy START: {len(audio_bytes)} bytes")
    
#     # DIAGNOSTIC: Hex dump of first 64 bytes
#     hex_header = " ".join(f"{b:02x}" for b in audio_bytes[:64])
#     log.info(f"   Bytes header (hex): {hex_header}")
    
#     # Check for different file signatures
#     signatures = {
#         "MP3": b"ID3" in audio_bytes[:12] or b"FF" in audio_bytes[:12].hex().encode(),
#         "WebM": audio_bytes[:4] == b"\x1a\x45\xdf\xa3",
#         "WAV": audio_bytes[:4] == b"RIFF",
#         "OGG": audio_bytes[:4] == b"OggS",
#         "FLAC": audio_bytes[:4] == b"fLaC",
#         "UNKNOWN": True
#     }
#     detected = [fmt for fmt, found in signatures.items() if found and fmt != "UNKNOWN"]
#     log.info(f"   Format detection: {', '.join(detected) if detected else 'UNKNOWN'}")
    
#     try:
#         log.debug("Trying soundfile...")
#         data, sr = sf.read(io.BytesIO(audio_bytes))
#         log.info(f"   ✅ Soundfile OK: sr={sr} shape={data.shape}")
        
#         if len(data.shape) > 1:
#             data = data.mean(axis=1)
#         if sr != Config.SAMPLE_RATE:
#             import librosa
#             data = librosa.resample(data, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
#             log.info(f"   Resampled to {Config.SAMPLE_RATE}")
        
#         result = data.astype(np.float32)
#         rms = np.sqrt(np.mean(result ** 2))
#         log.info(f"   Soundfile result: {len(result)} samples, RMS={rms:.6f}, range=[{result.min():.6f}, {result.max():.6f}]")
#         log.debug(f"   First 20 samples: {result[:20].tolist()}")
#         return result
#     except Exception as e:
#         log.warning(f"Soundfile failed: {e}, trying pydub...")
    
#     try:
#         log.debug("Trying pydub + ffmpeg...")
#         from pydub import AudioSegment
#         with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
#             tmp.write(audio_bytes)
#             path = tmp.name
        
#         # Detailed WebM analysis
#         if len(audio_bytes) > 4:
#             magic = audio_bytes[:4]
#             is_webm = (magic[0] == 0x1A and magic[1] == 0x45 and magic[2] == 0xDF and magic[3] == 0xA3)
#             log.info(f"   WebM magic bytes: {magic.hex()}, is_valid_webm={is_webm}")
            
#             # Check for multiple EBML headers (sign of concatenated chunks)
#             ebml_count = audio_bytes.count(b"\x1a\x45\xdf\xa3")
#             if ebml_count > 1:
#                 log.warning(f"   ⚠️  CRITICAL: Found {ebml_count} EBML headers in WebM (expected 1)")
#                 pos = 0
#                 for i in range(ebml_count):
#                     pos = audio_bytes.find(b"\x1a\x45\xdf\xa3", pos)
#                     log.warning(f"      EBML header #{i+1} at byte offset {pos}")
#                     pos += 4
        
#         try:
#             seg = AudioSegment.from_file(path, format="webm")
#             log.info(f"   Loaded WebM: {len(seg.raw_data)} bytes, channels={seg.channels}, sr={seg.frame_rate}")
            
#             seg = seg.set_frame_rate(Config.SAMPLE_RATE)
#             seg = seg.set_channels(1)
#             seg = seg.set_sample_width(2)
            
#             samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
#             result = (samples / 32768.0).astype(np.float32)
            
#             rms = np.sqrt(np.mean(result ** 2))
#             log.info(f"   ✅ Pydub OK: {len(result)} samples, RMS={rms:.6f}")
#             log.debug(f"   First 20 samples: {result[:20].tolist()}")
            
#             if rms < 0.0001:
#                 log.warning(f"⚠️  Audio decoded but ZERO RMS (all silence)")
            
#             return result
#         finally:
#             os.unlink(path)
#     except Exception as exc:
#         log.error(f"❌ Pydub/ffmpeg failed: {exc}, trying librosa...")
    
#     # Last resort: librosa
#     try:
#         log.debug("Trying librosa...")
#         import librosa
#         with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
#             tmp.write(audio_bytes)
#             path = tmp.name
#         try:
#             data, sr = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
#             result = data.astype(np.float32)
            
#             rms = np.sqrt(np.mean(result ** 2))
#             log.info(f"   ✅ Librosa OK: {len(result)} samples, RMS={rms:.6f}")
            
#             return result
#         finally:
#             os.unlink(path)
#     except Exception as exc:
#         log.error(f"❌ All converters failed: {exc}")
#         raise RuntimeError(f"Audio conversion failed: {exc}")


def audio_bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    """
    FIXED VERSION:
    - safer WebM decoding
    - better RMS detection
    - early silence detection
    - prevents fake VAD failures
    """

    log.info(f"🔊 audio_bytes_to_numpy START: {len(audio_bytes)} bytes")

    if len(audio_bytes) < 100:
        log.warning("⚠️ Audio too small → likely empty chunk")
        return np.array([], dtype=np.float32)

    # Detect WebM
    is_webm = audio_bytes[:4] == b"\x1a\x45\xdf\xa3"
    log.info(f"   Format: {'WebM' if is_webm else 'UNKNOWN'}")

    # =========================
    # TRY PYDUB FIRST (KEEP IT)
    # =========================
    try:
        from pydub import AudioSegment

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name

        try:
            seg = AudioSegment.from_file(path, format="webm")

            # DEBUG: Check WebM file details
            log.info(f"   WebM file size: {len(audio_bytes)} bytes")
            log.info(f"   Pydub segment: {len(seg.raw_data)} bytes, {seg.channels}ch, {seg.frame_rate}Hz, {seg.sample_width} bytes/sample")
            
            # normalize
            seg = seg.set_channels(1)
            seg = seg.set_frame_rate(Config.SAMPLE_RATE)
            seg = seg.set_sample_width(2)

            samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
            audio = samples.astype(np.float32) / 32768.0

            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
            max_val = float(np.max(np.abs(audio))) if len(audio) else 0.0
            min_val = float(np.min(audio)) if len(audio) else 0.0

            log.info(f"   ✅ decoded samples={len(audio)} RMS={rms:.6f} max={max_val:.6f} min={min_val:.6f}")

            # 🚨 EARLY SILENCE DETECTION
            if rms < 1e-4:
                log.warning("⚠️ SILENCE DETECTED (VERY LOW ENERGY)")
                # DEBUG: Check first few samples
                log.info(f"   First 10 samples: {audio[:10].tolist()}")
                log.info(f"   All samples same value? {np.all(audio == audio[0]) if len(audio) else 'N/A'}")

            return audio

        finally:
            try:
                os.unlink(path)
            except:
                pass

    except Exception as e:
        log.error(f"❌ pydub failed: {e}")
        
        # FALLBACK: Try other formats
        log.info("   Trying fallback formats...")
        
        # Try as WAV
        try:
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            log.info(f"   ✅ WAV fallback: {len(audio)} samples")
            return audio
        except:
            pass
            
        # Try as raw PCM
        try:
            audio = np.frombuffer(audio_bytes, dtype=np.float32)
            log.info(f"   ✅ PCM fallback: {len(audio)} samples")
            return audio
        except:
            pass

    # =========================
    # FALLBACK: LIBROSA
    # =========================
    try:
        import librosa
        # NOTE: do NOT re-import `tempfile` / `os` here — they are imported at
        # module level. A local `import` would shadow them and trigger
        # `UnboundLocalError: cannot access local variable 'tempfile'` on the
        # earlier pydub branch (Python scopes the name as local for the
        # entire function once it sees any local binding, even after use).

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name

        try:
            audio, sr = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)

            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

            log.info(f"   ✅ librosa samples={len(audio)} RMS={rms:.6f}")

            return audio.astype(np.float32)

        finally:
            try:
                os.unlink(path)
            except:
                pass

    except Exception as e:
        log.error(f"❌ librosa failed: {e}")

    # =========================
    # HARD FAIL SAFE
    # =========================
    log.error("❌ ALL AUDIO DECODERS FAILED")
    return np.array([], dtype=np.float32)
def _normalise_subject_response(raw: str) -> str | None:
    """Sanitise the LLM's subject classification.

    Accepts a free-form reply (the LLM might add quotes, dots, prefixes
    like 'Subject:' or '1.') and returns a lowercase single-word slug
    suitable for analytics tagging (``math``, ``physics``, ``cs``,
    ``medicine``, ``law``, ``art``, ``language``, etc.). Returns ``None``
    if the reply doesn't look like a subject label.
    """
    if not raw:
        return None
    import re as _re
    cleaned = raw.strip().lower()
    # Drop any "subject:" / "1." / "answer:" prefix the LLM sometimes emits
    cleaned = _re.sub(r"^(subject|category|answer|réponse|sujet)\s*[:\-]\s*", "", cleaned)
    cleaned = _re.sub(r"^\d+[.)]\s*", "", cleaned)
    # Take the first whitespace-separated token, then strip everything
    # that isn't a letter (handles quotes, brackets, dots, commas, etc.)
    parts = cleaned.split()
    if not parts:
        return None
    first = _re.sub(r"[^a-zA-Zà-ÿ_-]", "", parts[0])
    if not first or not (2 <= len(first) <= 24):
        return None
    return first


@functools.lru_cache(maxsize=2048)
def _detect_subject_cached(text_key: str, lang: str) -> str | None:
    """LLM-backed subject classifier with process-wide LRU cache.

    Cache key is the lowercased / whitespace-normalised question (so
    minor variations of the same question hit the same entry). The
    classifier is purely advisory — the result is consumed by analytics
    only — so we use the cheap fallback path (Ollama prefer) and don't
    block on errors.
    """
    if not text_key or len(text_key) < 5:
        return None
    try:
        from ai.llm_router import get_default_router
    except Exception:
        return None
    if lang == "fr":
        prompt = (
            "Classifie cette question dans UN domaine académique (ex: math, "
            "physique, biologie, chimie, informatique, histoire, géographie, "
            "économie, médecine, droit, philosophie, littérature, langue, art). "
            "Réponds par UN SEUL mot, en minuscules, sans ponctuation.\n\n"
            f"Question : {text_key[:400]}"
        )
    else:
        prompt = (
            "Classify this question into ONE academic subject (e.g. math, "
            "physics, biology, chemistry, cs, history, geography, economics, "
            "medicine, law, philosophy, literature, language, art). "
            "Reply with ONE word only, lowercase, no punctuation.\n\n"
            f"Question: {text_key[:400]}"
        )
    try:
        router = get_default_router()
        # Ollama prefer — analytics tag, latency-tolerant. ~100 tokens
        # roundtrip is enough; cap small to keep Mistral fast.
        response = router.invoke(prompt, prefer="ollama", temperature=0.0, max_tokens=10)
    except Exception:
        return None
    return _normalise_subject_response(response or "")


def detect_subject(text: str, language: str = "en") -> str | None:
    """Detect the academic subject of a question (analytics tag).

    Uses an LLM classifier (cached) instead of a hardcoded keyword
    table — the previous version had 8 hand-curated FR keywords per
    subject and missed any domain not in the list (medicine, law, art,
    philosophy, languages...). The LLM classifier handles arbitrary
    domains in any language.

    Returns ``None`` on classification failure; downstream analytics
    treat ``None`` as 'unknown' which is the honest default.
    """
    if not text or len(text.strip()) < 5:
        return None
    # Normalise for cache hits (case + whitespace + trim)
    import re as _re
    key = _re.sub(r"\s+", " ", text.strip().lower())[:400]
    return _detect_subject_cached(key, (language or "en")[:2])


# ════════════════════════════════════════════════════════════════════════
# Session storage (Redis-backed, persistent across restarts)
#
# Two domains:
#   1. Auth tokens (one-time use for WebSocket start_session validation)
#      - Key: auth_token:{sid}     TTL: Config.SESSION_TOKEN_TTL (5 min)
#   2. HTTP /ask conversation history
#      - Key: http_history:{sid}   TTL: Config.HTTP_HISTORY_TTL (1 h)
#      - Active set: http_sessions:active (for /health count)
# ════════════════════════════════════════════════════════════════════════

import json as _json
import uuid as _uuid
from typing import Optional
import redis.asyncio as _aioredis

_redis_pool: Optional[_aioredis.ConnectionPool] = None


def _get_redis() -> _aioredis.Redis:
    """Lazy-initialise a shared Redis connection pool."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = _aioredis.ConnectionPool(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True,
            max_connections=20,
        )
    return _aioredis.Redis(connection_pool=_redis_pool)


_AUTH_TOKEN_KEY = "auth_token:{sid}"
# Conversation history is now scoped per (session, course). Keeping all
# courses under the same key would let a Q&A turn from course A leak as
# "history" when the same student opens course B in the same session —
# which is exactly the kind of cross-course pollution we want to avoid
# for both UX (irrelevant context) and pedagogy (per-course continuity).
# The legacy single-key bucket is preserved for callers that don't pass
# a course (REST /ask without X-Course-ID, scripts, tests).
_HTTP_HISTORY_KEY = "http_history:{sid}:{course}"
_NO_COURSE = "_global"
_ACTIVE_SESSIONS_SET = "http_sessions:active"


def _course_key(course_id: str | None) -> str:
    """Normalise the course component of the Redis key.

    Empty / None / whitespace → ``_NO_COURSE`` bucket so old callers that
    never pass a course still get a deterministic key.
    """
    cid = (course_id or "").strip()
    return cid if cid else _NO_COURSE


# ── Auth tokens (one-time use) ─────────────────────────────────────────

async def set_session_token(sid: str, token: str) -> None:
    """Store an auth token for `sid`, expires after Config.SESSION_TOKEN_TTL."""
    r = _get_redis()
    await r.setex(_AUTH_TOKEN_KEY.format(sid=sid), Config.SESSION_TOKEN_TTL, token)


async def get_session_token(sid: str) -> Optional[str]:
    r = _get_redis()
    return await r.get(_AUTH_TOKEN_KEY.format(sid=sid))


async def consume_session_token(sid: str) -> Optional[str]:
    """Atomically read+delete the token (one-time use)."""
    r = _get_redis()
    # GETDEL is atomic; supported in Redis >= 6.2
    try:
        return await r.getdel(_AUTH_TOKEN_KEY.format(sid=sid))
    except Exception:
        # Fallback for older Redis: pipeline GET+DEL
        async with r.pipeline(transaction=True) as pipe:
            pipe.get(_AUTH_TOKEN_KEY.format(sid=sid))
            pipe.delete(_AUTH_TOKEN_KEY.format(sid=sid))
            value, _ = await pipe.execute()
        return value


async def has_session_token(sid: str) -> bool:
    r = _get_redis()
    return bool(await r.exists(_AUTH_TOKEN_KEY.format(sid=sid)))


# ── HTTP /ask conversation history ─────────────────────────────────────

async def _get_history_raw(sid: str, course_id: str | None = None) -> list[dict]:
    """Return the raw history list for ``(sid, course_id)``.

    When ``course_id`` is omitted, the legacy ``_NO_COURSE`` bucket is
    returned so callers that don't yet thread the course through still
    work (their history just lives in a single shared bucket).
    """
    r = _get_redis()
    raw = await r.get(_HTTP_HISTORY_KEY.format(sid=sid, course=_course_key(course_id)))
    if not raw:
        return []
    try:
        return _json.loads(raw)
    except Exception:
        return []


async def get_or_create_http_session(
    request,
    course_id: str | None = None,
) -> tuple[str, list]:
    """Get/create HTTP session — returns (session_id, history list).

    History is loaded for the (session, course) pair. ``course_id`` falls
    back to the ``X-Course-ID`` header when not passed explicitly so the
    front-end only needs to set it once.
    """
    sid = request.headers.get("X-Session-ID") or str(_uuid.uuid4())
    cid = course_id or request.headers.get("X-Course-ID")
    history = await _get_history_raw(sid, cid)
    r = _get_redis()
    await r.sadd(_ACTIVE_SESSIONS_SET, sid)
    return sid, history


async def replace_http_history(
    sid: str,
    history: list[dict],
    course_id: str | None = None,
) -> None:
    """Overwrite the full history for ``(sid, course_id)``, capped to MAX_HISTORY_TURNS*2 turns."""
    cap = Config.MAX_HISTORY_TURNS * 2
    trimmed = list(history)[-cap:] if len(history) > cap else list(history)
    r = _get_redis()
    await r.setex(
        _HTTP_HISTORY_KEY.format(sid=sid, course=_course_key(course_id)),
        Config.HTTP_HISTORY_TTL,
        _json.dumps(trimmed, ensure_ascii=False),
    )
    await r.sadd(_ACTIVE_SESSIONS_SET, sid)


async def clear_http_history(sid: str, course_id: str | None = None) -> None:
    """Drop the history for ``(sid, course_id)``.

    When ``course_id`` is None, ALL per-course history buckets for this
    session are cleared (the user explicitly asked to forget). The
    pattern scan is bounded to one session so the cost stays O(courses).
    """
    r = _get_redis()
    if course_id is not None:
        await r.delete(_HTTP_HISTORY_KEY.format(sid=sid, course=_course_key(course_id)))
        return
    pattern = _HTTP_HISTORY_KEY.format(sid=sid, course="*")
    keys: list[str] = []
    async for k in r.scan_iter(match=pattern, count=64):
        keys.append(k)
    if keys:
        await r.delete(*keys)
    await r.srem(_ACTIVE_SESSIONS_SET, sid)


async def count_http_sessions() -> int:
    """Approximate count of active sessions (set may contain stale TTL'd entries)."""
    r = _get_redis()
    return int(await r.scard(_ACTIVE_SESSIONS_SET) or 0)


# ── Backward-compat shim ──────────────────────────────────────────────
# Old sync `get_http_session(request)` is replaced by an async equivalent.
# Callers must `await get_http_session(request)`.
get_http_session = get_or_create_http_session
