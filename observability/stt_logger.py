"""
╔══════════════════════════════════════════════════════════════════════╗
║           SMART TEACHER — STT Metrics Logger                       ║
║                                                                      ║
║  Enregistre TOUS les paramètres et métriques STT pour chaque        ║
║  transcription : paramètres Whisper, timing, WER (mode test), etc.  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import csv
import logging
import os
from datetime import datetime
from typing import Optional

from core.config import Config

log = logging.getLogger("SmartTeacher.STTLogger")


class STTLogger:
    """
    Logger détaillé pour le module STT.
    Utile pour l'analyse WER, RTF, et l'optimisation des paramètres Whisper.
    """

    HEADERS = [
        # ── Contexte ─────────────────────────────────────────────────
        "timestamp",
        "session_id",
        "utt_id",

        # ── Paramètres audio (depuis Config) ─────────────────────────
        "sample_rate",
        "chunk_size",
        "silence_duration",
        "speech_threshold",
        "max_audio_duration",

        # ── Paramètres modèle Whisper ─────────────────────────────────
        "whisper_model_size",
        "whisper_device",
        "whisper_compute_type",
        "whisper_cpu_threads",

        # ── Paramètres de transcription ───────────────────────────────
        "language_detected",
        "language_prob",
        "explicit_language_param",
        "beam_size",
        "temperature",
        "vad_filter",
        "condition_on_prev_text",
        "word_timestamps",

        # ── Métriques temporelles ─────────────────────────────────────
        "audio_duration_sec",
        "stt_time",
        "rtf",

        # ── Résultat ─────────────────────────────────────────────────
        "transcription_text",
        "transcription_length_chars",
        "transcription_length_words",
        "stt_confidence",

        # ── Évaluation (mode test) ────────────────────────────────────
        "ref_text",
        "wer",
        "cer",
    ]

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or Config.STT_LOG_FILE
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADERS)
            log.info(f"✅ STTLogger initialisé : {self.filepath}")

    def log(
        self,
        *,
        session_id:              str,
        utt_id:                  str,
        audio_duration_sec:      float,
        language_detected:       str,
        language_prob:           Optional[float],
        stt_time:                float,
        transcription_text:      str,
        explicit_language_param: Optional[str]  = None,
        beam_size:               int            = 1,
        temperature:             float          = 0.0,
        vad_filter:              bool           = True,
        condition_on_prev_text:  bool           = False,
        word_timestamps:         bool           = False,
        stt_confidence:          Optional[float] = None,
        ref_text:                Optional[str]  = None,
        wer:                     Optional[float] = None,
        cer:                     Optional[float] = None,
    ) -> None:
        """Enregistre une transcription avec tous ses paramètres."""
        try:
            rtf           = stt_time / audio_duration_sec if audio_duration_sec > 0 else 0.0
            length_chars  = len(transcription_text)
            length_words  = len(transcription_text.split()) if transcription_text else 0

            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.utcnow().isoformat(),
                    session_id,
                    utt_id,

                    Config.SAMPLE_RATE,
                    Config.CHUNK_SIZE,
                    Config.SILENCE_DURATION,
                    Config.SPEECH_THRESHOLD,
                    getattr(Config, "MAX_AUDIO_DURATION", None),

                    Config.WHISPER_MODEL_SIZE,
                    Config.WHISPER_DEVICE,
                    Config.WHISPER_COMPUTE,
                    Config.WHISPER_THREADS,

                    language_detected,
                    round(language_prob, 3) if language_prob is not None else None,
                    explicit_language_param,
                    beam_size,
                    temperature,
                    vad_filter,
                    condition_on_prev_text,
                    word_timestamps,

                    round(audio_duration_sec, 3),
                    round(stt_time,           3),
                    round(rtf,                3),

                    transcription_text,
                    length_chars,
                    length_words,
                    round(stt_confidence, 3) if stt_confidence is not None else None,

                    ref_text,
                    round(wer, 3) if wer is not None else None,
                    round(cer, 3) if cer is not None else None,
                ])
        except Exception as exc:
            log.warning(f"⚠️  STT logging error: {exc}")