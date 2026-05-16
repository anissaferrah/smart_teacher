"""Smart Teacher — Speech-to-Text (faster-whisper or WhisperLiveKit)"""

import asyncio
import time
import logging
import unicodedata

import numpy as np
from faster_whisper import WhisperModel
from langdetect import DetectorFactory, detect_langs

from core.config import Config

log = logging.getLogger("SmartTeacher.STT")
DetectorFactory.seed = 0


class Transcriber:
    """Convertit l'audio en texte avec le backend STT configuré."""

    def __init__(self):
        self.backend = (Config.STT_BACKEND or "faster-whisper").strip().lower()
        self.model = None
        self._wlk_engine = None
        self._wlk_audio_processor_cls = None
        self._wlk_test_state_cls = None

        if self.backend == "whisperlivekit":
            try:
                from whisperlivekit import AudioProcessor, TranscriptionEngine
                from whisperlivekit.test_harness import TestState

                self._wlk_audio_processor_cls = AudioProcessor
                self._wlk_test_state_cls = TestState
                self._wlk_engine = TranscriptionEngine(
                    model_size=Config.WHISPER_MODEL_SIZE,
                    lan="auto",
                    backend="faster-whisper",
                    backend_policy="localagreement",
                    transcription=True,
                    diarization=False,
                    target_language="",
                    pcm_input=True,
                    vac=False,
                    vad=True,
                    min_chunk_size=0.1,
                    buffer_trimming="segment",
                    confidence_validation=False,
                )
                log.info("✅ WhisperLiveKit backend loaded")
            except Exception as exc:
                log.warning(
                    "⚠️ WhisperLiveKit unavailable (%s) — fallback faster-whisper", exc
                )
                self.backend = "faster-whisper"

        if self.backend != "whisperlivekit":
            import time as _wt
            _wload_t0 = _wt.time()
            log.info(
                "🎤 WHISPER LOAD START | model_size=%s device=%s compute_type=%s",
                Config.WHISPER_MODEL_SIZE,
                getattr(Config, "WHISPER_DEVICE", "cpu"),
                getattr(Config, "WHISPER_COMPUTE_TYPE", "int8"),
            )
            self.model = WhisperModel(
                Config.WHISPER_MODEL_SIZE,
                device=Config.WHISPER_DEVICE,
                compute_type=Config.WHISPER_COMPUTE,
                cpu_threads=Config.WHISPER_THREADS,
            )
            log.info(
                "🎤 WHISPER LOAD DONE | model=%s | took=%.2fs",
                Config.WHISPER_MODEL_SIZE, _wt.time() - _wload_t0,
            )

    def validate_audio_quality(self, audio: np.ndarray) -> tuple[bool, str]:
        """Check if audio is completely empty/corrupt.
        ⚠️  VERY RELAXED: Only reject if audio is 100% zeros (not even soft speech)
        Let Whisper handle detection - even whispers should pass through.
        Returns: (is_valid_audio, reason_if_invalid)
        """
        if len(audio) == 0:
            log.warning("❌ Audio validation: empty array")
            return False, "Audio vide"
        
        # Check if audio is ALL zeros (corrupted decode, not soft speech)
        # Use max absolute value instead of RMS to catch completely silent audio
        max_val = np.max(np.abs(audio))
        min_val = np.min(audio)
        mean_val = np.mean(np.abs(audio))
        
        log.info(f"   Audio quality: len={len(audio)} samples, max_abs={max_val:.8f}, min={min_val:.8f}, mean={mean_val:.8f}")
        log.debug(f"   First 30 samples: {audio[:30].tolist()}")
        log.debug(f"   Last 30 samples: {audio[-30:].tolist()}")
        
        # Only reject if max_abs is near-zero = truly no audio
        if max_val < 0.00001:
            log.warning(f"❌ Audio rejected: max_abs={max_val:.8f} (all zeros) - likely WebM container corruption")
            return False, "Audio all zeros"
        
        return True, ""

    def trim_silence(self, audio: np.ndarray, threshold: float = 0.001) -> np.ndarray:
        """Remove silence from beginning/end of audio
        
        ✅ Threshold TRÈS réduit de 0.02 → 0.001 (1000x moins agressif!)
        pour laisser passer même la parole très faible
        """
        above = np.abs(audio) > threshold
        
        # Diagnostic
        above_count = np.sum(above)
        above_pct = (above_count / len(audio)) * 100 if len(audio) > 0 else 0
        log.info(f"   trim_silence: threshold={threshold}, {above_count}/{len(audio)} samples above threshold ({above_pct:.1f}%)")
        
        if not np.any(above):
            log.warning(f"   ⚠️  trim_silence: NO samples above threshold={threshold}! Returning original audio")
            return audio

        first = int(np.argmax(above))
        last  = len(above) - int(np.argmax(above[::-1]))

        pad_before = int(0.05 * Config.SAMPLE_RATE)   # 50 ms
        pad_after  = int(0.15 * Config.SAMPLE_RATE)   # 150 ms

        start   = max(0, first - pad_before)
        end     = min(len(audio), last + pad_after)
        trimmed = audio[start:end]

        orig_dur    = len(audio)   / Config.SAMPLE_RATE
        trimmed_dur = len(trimmed) / Config.SAMPLE_RATE
        if orig_dur - trimmed_dur > 0.5:
            log.debug(f"✂️  Silence trimmed: {orig_dur:.1f}s → {trimmed_dur:.1f}s (threshold={threshold})")

        return trimmed

    def extract_prosody(self, text: str, audio_duration: float) -> dict:
        """Extract prosodic deviation markers from a transcribed utterance.

        Returns three observable measurements plus a list of triggered
        markers. Downstream consumers (e.g. ``dialogue.detect_confusion``)
        decide how to interpret the markers. No composite "confidence"
        score is computed here — that would require empirically calibrated
        weights we don't have.

        Threshold rationale:
          - ``slow_speech_rate`` : speech_rate < 60 wpm.
            Normal conversational speech is 120-180 wpm (Yuan, Liberman,
            Cieri 2006; Goldman-Eisler 1968). 60 wpm is well below the
            slow tail of the natural range, marking unambiguous slowdown.
          - ``frequent_hesitations`` : hesitation_rate ≥ 5 fillers/min.
            Conversational disfluency rate is ~3-6 per minute (Bortfeld
            et al. 2001). We use the upper end of the normal band as the
            "above baseline" signal.
          - ``high_silence_ratio`` : silence_ratio > 0.5 (i.e. more than
            half the audio duration is non-speech). Heuristic cutoff —
            kept as a soft signal pending empirical validation.
        """
        if not text or audio_duration <= 0:
            return {"speech_rate": 0.0, "hesitation_count": 0,
                    "silence_ratio": 0.0, "markers": []}

        word_count = len(text.split())
        speech_rate = (word_count / audio_duration) * 60   # words / min
        duration_min = audio_duration / 60.0

        hesitation_words = {
            "fr": ["euh", "uh", "um", "hum", "heu", "bah", "ben", "disons"],
            "en": ["um", "uh", "like", "you know", "i mean"],
        }
        text_lower = text.lower()
        hesitation_count = sum(
            text_lower.count(hes)
            for lang_hes in hesitation_words.values()
            for hes in lang_hes
        )
        hesitation_rate = hesitation_count / max(duration_min, 1e-6)  # per minute

        # Silence ratio approximation : 1 - words / (duration * normal_rate)
        # where 100 wpm is used as the lower-end normal rate for the inverse.
        silence_ratio = max(0.0, min(1.0, 1.0 - word_count / max(audio_duration * 100, 1)))

        markers: list[str] = []
        if speech_rate < 60:
            markers.append("slow_speech_rate")
        if hesitation_rate >= 5:
            markers.append("frequent_hesitations")
        if silence_ratio > 0.5:
            markers.append("high_silence_ratio")

        return {
            "speech_rate":      round(speech_rate, 1),
            "hesitation_count": hesitation_count,
            "hesitation_rate":  round(hesitation_rate, 2),
            "silence_ratio":    round(silence_ratio, 2),
            "markers":          markers,
        }

    def _detect_language_from_text(self, text: str, fallback_language: str = "unknown") -> tuple[str, float]:
        if not text:
            return fallback_language, 0.0

        try:
            detected = detect_langs(text)
            if detected:
                best = detected[0]
                return best.lang, float(best.prob)
        except Exception:
            pass

        return fallback_language, 0.0

    async def _transcribe_with_whisperlivekit(
        self,
        audio: np.ndarray,
        force_language: str | None = None,
    ) -> tuple[str, str, float]:
        if self._wlk_engine is None or self._wlk_audio_processor_cls is None or self._wlk_test_state_cls is None:
            raise RuntimeError("WhisperLiveKit backend is not initialized")

        processor = self._wlk_audio_processor_cls(
            transcription_engine=self._wlk_engine,
            language=None if not force_language or force_language == "auto" else force_language,
        )
        processor.is_pcm_input = True

        results_gen = await processor.create_tasks()
        final_front_data = None

        async def collect_results():
            nonlocal final_front_data
            async for front_data in results_gen:
                final_front_data = front_data

        collect_task = asyncio.create_task(collect_results())

        pcm = np.clip(audio, -1.0, 1.0)
        pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()
        chunk_bytes = Config.SAMPLE_RATE * 2

        try:
            for index in range(0, len(pcm_bytes), chunk_bytes):
                await processor.process_audio(pcm_bytes[index:index + chunk_bytes])

            await processor.process_audio(b"")
            await asyncio.wait_for(collect_task, timeout=120.0)
        finally:
            await processor.cleanup()

        if final_front_data is None:
            return "", "unknown", 0.0

        final_state = self._wlk_test_state_cls.from_front_data(final_front_data)
        text = getattr(final_state, "text", "") or ""
        language, language_prob = self._detect_language_from_text(text, fallback_language="unknown")
        return text.strip(), language, language_prob

    def transcribe(
        self,
        audio: np.ndarray,
        force_language: str = None,
        slide_context: str = None,
    ) -> tuple[str, float, str, float, float]:
        """Transcribe audio array to text. Returns: (text, stt_time, language, lang_prob, audio_duration)

        Args:
            audio: Audio samples
            force_language: If set (e.g. 'en', 'fr'), use this instead of auto-detection
            slide_context: Optional text passed to Whisper as ``initial_prompt`` to
                bias decoding toward the lecture's vocabulary (chapter + section
                title + first 200 chars of the live narration). Capped upstream
                to ~300 chars to stay safely under Whisper's 224-token prompt
                limit. ``None`` or empty → Whisper runs unprompted.
        """
        start = time.time()
        log.info(f"🎤 STT START: len={len(audio)} samples, force_lang={force_language}")

        try:
            # 1. VALIDATION: Basic sanity check on audio (very relaxed)
            is_valid, reason = self.validate_audio_quality(audio)
            if not is_valid:
                log.warning(f"❌ Audio validation failed: {reason}")
                return "", 0.0, "silence", 0.0, len(audio) / Config.SAMPLE_RATE

            # 2. Suppression silences
            orig_len = len(audio)
            audio = self.trim_silence(audio)
            audio_duration = len(audio) / Config.SAMPLE_RATE
            log.info(f"   After trim_silence: {orig_len} → {len(audio)} samples ({audio_duration:.3f}s)")
            
            # Extra diagnostic: After trim_silence, verify we still have audio
            if len(audio) > 0:
                max_val_after = np.max(np.abs(audio))
                log.info(f"   After trim: max_abs={max_val_after:.8f}, min={np.min(audio):.8f}, mean={np.mean(np.abs(audio)):.8f}")

            # 3. Audio trop court → ignorer
            if audio_duration < Config.STT_MIN_AUDIO_SEC:
                log.warning(f"❌ Audio too short: {audio_duration:.3f}s < {Config.STT_MIN_AUDIO_SEC}s min")
                return "", 0.0, "unknown", 0.0, audio_duration
            
            log.info(f"   ✅ Audio OK: {len(audio)} samples ({audio_duration:.3f}s) ready for STT backend")
            log.debug(f"   Audio samples: min={audio.min():.8f}, max={audio.max():.8f}, mean={audio.mean():.8f}")

            # 4. Transcription
            if self.backend == "whisperlivekit":
                log.info(f"   → Calling WhisperLiveKit pipeline (language={force_language})")
                text, lang, lang_prob = asyncio.run(
                    self._transcribe_with_whisperlivekit(audio, force_language=force_language)
                )
            else:
                log.info(
                    f"   → Calling Whisper.transcribe(vad_filter=False, language={force_language}, "
                    f"initial_prompt={'<set>' if slide_context else '<none>'})"
                )
                segments, info = self.model.transcribe(
                    audio,
                    language=force_language,          # ← Peut être force 'en', 'fr' si besoin
                    beam_size=Config.STT_BEAM_SIZE,
                    best_of=1,
                    temperature=0.0,                  # déterministe
                    vad_filter=False,                 # ✅ DÉSACTIVER VAD agressif (filtre toute la parole!)
                    # ✅ ALTERNATIVEMENT: VAD avec paramètres moins agressifs:
                    # vad_filter=True,
                    # vad_parameters=dict(
                    #     min_speech_duration_ms=100,   # Au minimum 100ms de parole
                    #     min_silence_duration_ms=1500, # Besoin de 1.5s de silence (par défaut 2s, très strict)
                    #     speech_pad_ms=300,            # Padding 300ms (par défaut 400ms)
                    # ),
                    condition_on_previous_text=False, # pas de mémoire → évite hallucinations
                    without_timestamps=True,
                    word_timestamps=False,
                    initial_prompt=slide_context if slide_context else None,
                )

                # ⚠️  CRITICAL: segments is a GENERATOR, convert to list immediately!
                segments = list(segments)
                log.info(f"   Whisper returned {len(segments)} segments")
                
                # 5. Fusion des segments
                text = " ".join(s.text.strip() for s in segments).strip()
                log.info(f"   Whisper raw output: '{text[:100]}'... ({len(segments)} segments)")

                lang      = getattr(info, "language",             "unknown")
                lang_prob = getattr(info, "language_probability", 0.0)

            stt_time  = time.time() - start

            # 6. REJECTION: Only detect clear Unicode corruption.
            rtl_languages = {"ar", "fa", "he", "ur", "ps", "yi"}
            is_rtl_language = lang in rtl_languages or any(lang.startswith(prefix + "-") for prefix in rtl_languages)
            if len(text) > 5:
                # Check for Unicode control/format characters (actual corruption)
                # These are actual invalid characters, not just non-ASCII
                invalid_categories = {'Cc', 'Cf', 'Cn', 'Co'}  # Control, Format, Not assigned, Private use
                invalid_char_count = sum(
                    1 for c in text
                    if unicodedata.category(c) in invalid_categories
                )
                
                # Only reject if > 30% of text is actual control/format characters
                if len(text) > 0 and invalid_char_count / len(text) > 0.3:
                    log.warning(f"🚨 STT Corruption: {invalid_char_count}/{len(text)} invalid Unicode chars, rejecting")
                    return "", stt_time, lang, lang_prob, audio_duration
            if len(text) <= 5 and is_rtl_language:
                log.info(f"✅ RTL language detected ({lang}) — skipping corruption check")
            
            rtf = stt_time / audio_duration if audio_duration > 0 else 0
            log.info(
                f"STT | '{text[:60]}…' | lang={lang}({lang_prob:.0%}) "
                f"| dur={audio_duration:.2f}s | stt={stt_time:.2f}s | RTF={rtf:.2f}x"
            )
            return text, stt_time, lang, lang_prob, audio_duration

        except Exception as exc:
            log.error(f"❌ Transcription error: {exc}")
            return "", time.time() - start, "error", 0.0, 0.0