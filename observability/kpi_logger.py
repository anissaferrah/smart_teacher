"""
╔══════════════════════════════════════════════════════════════════════╗
║          SMART TEACHER — KPI Logger (Cahier des charges)            ║
║                                                                      ║
║  Mesure les 4 KPIs explicitement demandés par l'entreprise :        ║
║                                                                      ║
║    1. Latence interruption  < 500ms                                 ║
║         (VAD detect → TTS stop)                                     ║
║    2. Latence réponse       < 5s                                    ║
║         (audio_end → first response audio chunk back)               ║
║    3. WER (Word Error Rate) < 5%                                    ║
║         (calculé seulement si ground truth fournie)                 ║
║    4. MOS (Naturalité TTS)  > 4.0                                   ║
║         (collecté via feedback utilisateur — endpoint dédié)        ║
║                                                                      ║
║  Usage typique dans le WS handler :                                  ║
║                                                                      ║
║    kpi = KPITracker.get()                                           ║
║    kpi.mark_interrupt_detected(session_id, turn_id)                 ║
║    ... cancel TTS ...                                                ║
║    kpi.mark_tts_stopped(session_id, turn_id)                        ║
║                                                                      ║
║  Lecture côté API :                                                  ║
║                                                                      ║
║    GET /analytics/kpi → p50/p95/p99 + compliance rate               ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import csv
import logging
import os
import statistics
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.config import Config

log = logging.getLogger("SmartTeacher.KPI")


# ── KPI thresholds (cahier des charges) ────────────────────────────────
KPI_INTERRUPT_LATENCY_MAX_S: float = 0.500   # 500 ms
KPI_RESPONSE_LATENCY_MAX_S:  float = 5.000   # 5 s
KPI_WER_MAX:                 float = 0.050   # 5%
KPI_MOS_MIN:                 float = 4.0     # /5


def _percentile(values: list[float], p: float) -> float:
    """Approximate p-th percentile (0..100) without numpy."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


class KPITracker:
    """In-memory rolling tracker of the 4 entreprise KPIs.

    Thread-safe (locks around list append/read). Persists each turn to a CSV
    line for offline analysis. Aggregates served via /analytics/kpi.
    """

    _instance: Optional["KPITracker"] = None

    def __init__(self, filepath: Optional[str] = None, window: int = 200) -> None:
        self.filepath = filepath or os.path.join(Config.LOGS_DIR, "kpi_metrics.csv")
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._window = window

        # Rolling windows of latencies (seconds)
        self._interrupt_latencies: deque[float] = deque(maxlen=window)
        self._response_latencies:  deque[float] = deque(maxlen=window)
        self._wer_samples:         deque[float] = deque(maxlen=window)
        self._mos_samples:         deque[float] = deque(maxlen=window)

        # In-flight timestamps keyed by (session_id, turn_id)
        # — paired up when the matching mark_*_end event arrives.
        self._t_interrupt_start: dict[tuple[str, int], float] = {}
        self._t_response_start:  dict[tuple[str, int], float] = {}

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "session_id", "turn_id",
                    "kpi_kind",           # interrupt | response | wer | mos
                    "value",              # latency seconds OR ratio OR score
                    "meets_kpi",          # 1 if within target, 0 otherwise
                    "extra",              # optional JSON-ish notes
                ])
            log.info(f"✅ KPILogger initialisé : {self.filepath}")

    # ── Singleton accessor ─────────────────────────────────────────────

    @classmethod
    def get(cls) -> "KPITracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Helpers ────────────────────────────────────────────────────────

    def _record_csv(
        self, session_id: str, turn_id: int, kind: str,
        value: float, meets_kpi: bool, extra: str = "",
    ) -> None:
        try:
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.utcnow().isoformat(), session_id, turn_id,
                    kind, round(value, 4), 1 if meets_kpi else 0, extra,
                ])
        except Exception as exc:
            log.debug(f"KPI CSV write skipped: {exc}")

    # ── KPI #1 : interruption latency ─────────────────────────────────

    def mark_interrupt_detected(self, session_id: str, turn_id: int = 0) -> None:
        """Called when VAD detects student starting to speak during TTS."""
        self._t_interrupt_start[(session_id, turn_id)] = time.time()

    def mark_tts_stopped(self, session_id: str, turn_id: int = 0) -> Optional[float]:
        """Called when TTS streaming is actually cancelled. Returns latency in seconds."""
        key = (session_id, turn_id)
        start = self._t_interrupt_start.pop(key, None)
        if start is None:
            return None
        latency = time.time() - start
        meets = latency <= KPI_INTERRUPT_LATENCY_MAX_S
        with self._lock:
            self._interrupt_latencies.append(latency)
        self._record_csv(session_id, turn_id, "interrupt", latency, meets)
        if not meets:
            log.warning(
                f"⚠️ KPI interrupt: {latency*1000:.0f}ms > "
                f"{KPI_INTERRUPT_LATENCY_MAX_S*1000:.0f}ms target"
            )
        return latency

    # ── KPI #2 : response latency ─────────────────────────────────────

    def mark_question_received(self, session_id: str, turn_id: int = 0) -> None:
        """Called at audio_end (student finished speaking)."""
        self._t_response_start[(session_id, turn_id)] = time.time()

    def mark_first_response_chunk(self, session_id: str, turn_id: int = 0) -> Optional[float]:
        """Called at first answer audio chunk sent back. Returns latency in seconds."""
        key = (session_id, turn_id)
        start = self._t_response_start.pop(key, None)
        if start is None:
            return None
        latency = time.time() - start
        meets = latency <= KPI_RESPONSE_LATENCY_MAX_S
        with self._lock:
            self._response_latencies.append(latency)
        self._record_csv(session_id, turn_id, "response", latency, meets)
        if not meets:
            log.warning(
                f"⚠️ KPI response: {latency:.2f}s > "
                f"{KPI_RESPONSE_LATENCY_MAX_S}s target"
            )
        return latency

    # ── KPI #3 : WER (when ground truth available) ───────────────────

    def record_wer(self, session_id: str, turn_id: int, wer: float, ref_len: int = 0) -> None:
        """Record a WER sample (0.0 = perfect, 1.0 = full error)."""
        meets = wer <= KPI_WER_MAX
        with self._lock:
            self._wer_samples.append(wer)
        self._record_csv(session_id, turn_id, "wer", wer, meets, extra=f"ref_len={ref_len}")

    # ── KPI #4 : MOS (user feedback) ──────────────────────────────────

    def record_mos(self, session_id: str, turn_id: int, mos: float) -> None:
        """Record user feedback on TTS quality (1.0..5.0)."""
        mos = max(1.0, min(5.0, float(mos)))
        meets = mos >= KPI_MOS_MIN
        with self._lock:
            self._mos_samples.append(mos)
        self._record_csv(session_id, turn_id, "mos", mos, meets)

    # ── Aggregation ────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return p50/p95/p99 + compliance rate for each KPI."""
        with self._lock:
            interrupts = list(self._interrupt_latencies)
            responses  = list(self._response_latencies)
            wers       = list(self._wer_samples)
            moss       = list(self._mos_samples)

        def stats(vals: list[float], threshold: float, lower_is_better: bool = True):
            if not vals:
                return {"count": 0, "p50": None, "p95": None, "p99": None,
                        "mean": None, "compliance_rate": None}
            if lower_is_better:
                ok = sum(1 for v in vals if v <= threshold)
            else:
                ok = sum(1 for v in vals if v >= threshold)
            return {
                "count":           len(vals),
                "p50":             round(_percentile(vals, 50), 4),
                "p95":             round(_percentile(vals, 95), 4),
                "p99":             round(_percentile(vals, 99), 4),
                "mean":            round(statistics.mean(vals), 4),
                "compliance_rate": round(ok / len(vals), 3),
            }

        return {
            "thresholds": {
                "interrupt_latency_max_s": KPI_INTERRUPT_LATENCY_MAX_S,
                "response_latency_max_s":  KPI_RESPONSE_LATENCY_MAX_S,
                "wer_max":                 KPI_WER_MAX,
                "mos_min":                 KPI_MOS_MIN,
            },
            "interrupt_latency_s": stats(interrupts, KPI_INTERRUPT_LATENCY_MAX_S),
            "response_latency_s":  stats(responses,  KPI_RESPONSE_LATENCY_MAX_S),
            "wer":                 stats(wers,       KPI_WER_MAX),
            "mos":                 stats(moss,       KPI_MOS_MIN, lower_is_better=False),
            "window":              self._window,
        }

    def reset(self) -> None:
        """Clear all rolling buffers (useful for tests/benchmarks)."""
        with self._lock:
            self._interrupt_latencies.clear()
            self._response_latencies.clear()
            self._wer_samples.clear()
            self._mos_samples.clear()
            self._t_interrupt_start.clear()
            self._t_response_start.clear()
