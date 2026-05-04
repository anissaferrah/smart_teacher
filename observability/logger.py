"""
╔══════════════════════════════════════════════════════════════════════╗
║           SMART TEACHER — Logger CSV (Métriques globales)          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import csv
import os
import logging
from datetime import datetime
from core.config import Config

log = logging.getLogger("SmartTeacher.Logger")


class CsvLogger:
    """
    Enregistre les métriques de chaque tour d'interaction dans un fichier CSV.
    Utilisé pour analyser les performances : STT / LLM / TTS / total.
    """

    HEADERS = [
        "timestamp",
        "session_id",
        "audio_duration_sec",
        "stt_time",
        "llm_time",
        "tts_time",
        "total_time",
        "meets_kpi",          # NOUVEAU : 1 si total < MAX_RESPONSE_TIME
        "language",
        "model_used",
        "tts_engine_used",
        "tts_model_used",
        "transcription_preview",   # NOUVEAU : 40 premiers chars
    ]

    def __init__(self, filepath: str | None = None):
        self.filepath = filepath or Config.CSV_LOG_FILE
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADERS)
            log.info(f"✅ CsvLogger initialisé : {self.filepath}")

    def log_turn(
        self,
        audio_duration_sec: float,
        stt_time:           float,
        llm_time:           float,
        tts_time:           float,
        total_time:         float,
        language:           str,
        model_used:         str,
        tts_engine_used:    str  = "edge",
        tts_model_used:     str  = "",
        session_id:         str  = "",
        transcription:      str  = "",
    ) -> None:
        """Enregistre un tour d'interaction."""
        meets_kpi = 1 if total_time <= Config.MAX_RESPONSE_TIME else 0
        try:
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.utcnow().isoformat(),
                    session_id,
                    round(audio_duration_sec, 3),
                    round(stt_time,           3),
                    round(llm_time,           3),
                    round(tts_time,           3),
                    round(total_time,         3),
                    meets_kpi,
                    language,
                    model_used,
                    tts_engine_used,
                    tts_model_used,
                    transcription[:40].replace("\n", " "),
                ])
        except Exception as exc:
            log.warning(f"⚠️  CSV logging error: {exc}")