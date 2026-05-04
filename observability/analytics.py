"""
╔══════════════════════════════════════════════════════════════════════╗
║        SMART TEACHER — Analytics (ClickHouse / CSV fallback)       ║
║                                                                      ║
║  Métriques d'apprentissage détaillées :                             ║
║    - Progression par étudiant / cours / chapitre                   ║
║    - KPIs temps réel (latence STT/LLM/TTS)                        ║
║    - Taux de compréhension, nombre d'interruptions                  ║
║    - Heatmaps d'activité                                            ║
║    - Suivi long terme sur plusieurs sessions                        ║
║                                                                      ║
║  Si ClickHouse non configuré → CSV + mémoire (compatible existant) ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import csv
import time
import logging
import base64
from datetime import datetime, date
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.config import Config

log = logging.getLogger("SmartTeacher.Analytics")

CSV_DIR = Path(Config.ANALYTICS_CSV_DIR)


@dataclass
class LearningEvent:
    """Un événement d'apprentissage (interaction complète)."""
    event_time:    str   = ""     # ISO timestamp
    session_id:    str   = ""
    student_id:    str   = ""
    course_id:     str   = ""
    language:      str   = "fr"
    level:         str   = "lycée"
    event_type:    str   = "qa"   # qa | section_start | section_end | interrupt | quiz
    question:      str   = ""
    answer_length: int   = 0
    stt_time:      float = 0.0
    llm_time:      float = 0.0
    tts_time:      float = 0.0
    total_time:    float = 0.0
    kpi_ok:        bool  = False
    subject:       str   = ""
    chapter_idx:   int   = 0
    section_idx:   int   = 0
    confidence:    float = 0.0    # STT confidence
    interruptions: int   = 0


class AnalyticsEngine:
    """
    Moteur analytics : ClickHouse si disponible, sinon CSV + cache mémoire.
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS learning_events (
        event_time    DateTime,
        session_id    String,
        student_id    String,
        course_id     String,
        language      LowCardinality(String),
        level         LowCardinality(String),
        event_type    LowCardinality(String),
        question      String,
        answer_length UInt32,
        stt_time      Float32,
        llm_time      Float32,
        tts_time      Float32,
        total_time    Float32,
        kpi_ok        UInt8,
        subject       LowCardinality(String),
        chapter_idx   UInt8,
        section_idx   UInt8,
        confidence    Float32,
        interruptions UInt8
    ) ENGINE = MergeTree()
    ORDER BY (event_time, session_id)
    PARTITION BY toYYYYMM(event_time)
    """

    def __init__(self):
        self._ch     = None
        self._use_ch = False
        self._ch_initialized = False  # Track if initialization was attempted
        self._cache: list[LearningEvent] = []
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        self._csv_path = CSV_DIR / f"events_{date.today().isoformat()}.csv"
        # Lazy initialization of ClickHouse - don't block startup

    def _init_ch(self):
        # Skip if already tried to initialize
        if self._ch_initialized:
            return
        
        self._ch_initialized = True
        
        if not Config.CLICKHOUSE_HOST:
            log.info("📊 ClickHouse non configuré → analytics CSV + mémoire")
            return

        try:
            probe_url = f"http://{Config.CLICKHOUSE_HOST}:{Config.CLICKHOUSE_PORT}/?query=SELECT%201"
            request = Request(probe_url)
            if Config.CLICKHOUSE_PASSWORD:
                credentials = f"{Config.CLICKHOUSE_USER}:{Config.CLICKHOUSE_PASSWORD}".encode("utf-8")
                request.add_header("Authorization", f"Basic {base64.b64encode(credentials).decode('ascii')}")

            with urlopen(request, timeout=1.5) as response:
                body = response.read().decode("utf-8", errors="ignore").strip().lower()
                if response.status != 200 or body not in {"1", "1\n"}:
                    raise RuntimeError(f"unexpected ping response: {response.status} {body!r}")
        except HTTPError as exc:
            log.info("📊 ClickHouse indisponible sur %s:%s (%s) → analytics CSV + mémoire", Config.CLICKHOUSE_HOST, Config.CLICKHOUSE_PORT, exc)
            return
        except URLError as exc:
            log.info("📊 ClickHouse indisponible sur %s:%s (%s) → analytics CSV + mémoire", Config.CLICKHOUSE_HOST, Config.CLICKHOUSE_PORT, exc.reason)
            return
        except Exception as exc:
            log.info("📊 ClickHouse indisponible sur %s:%s (%s) → analytics CSV + mémoire", Config.CLICKHOUSE_HOST, Config.CLICKHOUSE_PORT, exc)
            return

        try:
            import clickhouse_connect
            self._ch = clickhouse_connect.get_client(
                host=Config.CLICKHOUSE_HOST, port=Config.CLICKHOUSE_PORT,
                database=Config.CLICKHOUSE_DB, username=Config.CLICKHOUSE_USER, password=Config.CLICKHOUSE_PASSWORD,
                connect_timeout=2,  # Short timeout to avoid blocking
                send_receive_timeout=2,
            )
            self._ch.command(self.CREATE_TABLE_SQL)
            self._use_ch = True
            log.info("✅ ClickHouse connecté : %s:%s/%s", Config.CLICKHOUSE_HOST, Config.CLICKHOUSE_PORT, Config.CLICKHOUSE_DB)
        except ImportError:
            log.info("📊 clickhouse-connect non installé → analytics CSV + mémoire")
        except Exception as e:
            log.info("📊 ClickHouse non dispo (%s) → analytics CSV + mémoire", e)

    # ── Enregistrement ────────────────────────────────────────────────────

    def record(self, evt: LearningEvent) -> None:
        """Enregistre un événement d'apprentissage."""
        if not evt.event_time:
            evt.event_time = datetime.utcnow().isoformat()

        # Cache mémoire (max 2000)
        self._cache.append(evt)
        if len(self._cache) > 2000:
            self._cache.pop(0)

        # CSV
        self._write_csv(evt)

        # ClickHouse (lazy init)
        self._init_ch()
        if self._use_ch:
            self._ch_insert(evt)

    def record_interaction(self, session_id: str, question: str, answer: str,
                           stt_time: float, llm_time: float, tts_time: float,
                           language: str = "fr", course_id: str = "",
                           subject: str = "", level: str = "lycée",
                           chapter_idx: int = 0, section_idx: int = 0,
                           confidence: float = 0.0) -> LearningEvent:
        """Raccourci pour enregistrer une interaction Q&A."""
        total = stt_time + llm_time + tts_time
        evt = LearningEvent(
            session_id=session_id, course_id=course_id,
            language=language, level=level,
            event_type="qa",
            question=question[:200],
            answer_length=len(answer),
            stt_time=round(stt_time, 3),
            llm_time=round(llm_time, 3),
            tts_time=round(tts_time, 3),
            total_time=round(total, 3),
            kpi_ok=total < 5.0,
            subject=subject,
            chapter_idx=chapter_idx,
            section_idx=section_idx,
            confidence=round(confidence, 3),
        )
        self.record(evt)
        return evt

    def record_section(self, session_id: str, course_id: str, chapter_idx: int,
                       section_idx: int, event_type: str = "section_start",
                       language: str = "fr") -> None:
        self.record(LearningEvent(
            session_id=session_id, course_id=course_id,
            language=language, event_type=event_type,
            chapter_idx=chapter_idx, section_idx=section_idx,
        ))

    # ── Requêtes analytics ────────────────────────────────────────────────

    def kpi_summary(self, hours: int = 24) -> dict:
        """Résumé des KPIs sur les N dernières heures."""
        self._init_ch()
        if self._use_ch:
            return self._ch_kpi_summary(hours)
        return self._mem_kpi_summary(hours)

    def progression_by_course(self, course_id: str) -> list[dict]:
        """Progression des sections vues pour un cours."""
        self._init_ch()
        if self._use_ch:
            return self._ch_progression(course_id)
        return self._mem_progression(course_id)

    def latency_distribution(self) -> dict:
        """Distribution des latences (min/avg/max/p95)."""
        times = [e.total_time for e in self._cache if e.total_time > 0]
        if not times:
            return {}
        times.sort()
        n = len(times)
        return {
            "count": n,
            "min":   round(min(times), 3),
            "avg":   round(sum(times) / n, 3),
            "max":   round(max(times), 3),
            "p50":   round(times[int(n * .50)], 3),
            "p95":   round(times[int(n * .95)], 3),
            "kpi_rate_pct": round(sum(1 for t in times if t < 5.0) / n * 100, 1),
        }

    def by_language(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._cache:
            counts[e.language] = counts.get(e.language, 0) + 1
        return counts

    def by_subject(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._cache:
            if e.subject:
                counts[e.subject] = counts.get(e.subject, 0) + 1
        return counts

    def full_report(self) -> dict:
        """Rapport complet pour le dashboard."""
        return {
            "total_events":   len(self._cache),
            "kpi":            self.kpi_summary(),
            "latency":        self.latency_distribution(),
            "by_language":    self.by_language(),
            "by_subject":     self.by_subject(),
            "backend":        "clickhouse" if self._use_ch else "csv+memory",
        }

    # ── Internals ─────────────────────────────────────────────────────────

    def _write_csv(self, evt: LearningEvent) -> None:
        write_header = not self._csv_path.exists()
        field_names  = [f.name for f in fields(evt)]
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=field_names)
                if write_header:
                    w.writeheader()
                w.writerow(asdict(evt))
        except Exception as e:
            log.debug("CSV write error: %s", e)

    def _ch_insert(self, evt: LearningEvent) -> None:
        import time as _cht
        try:
            row = asdict(evt)
            row["kpi_ok"]     = 1 if row["kpi_ok"] else 0
            row["event_time"] = datetime.fromisoformat(row["event_time"])
            _t0 = _cht.time()
            self._ch.insert("learning_events", [list(row.values())],
                            column_names=list(row.keys()))
            log.info(
                "🟧 CLICKHOUSE INSERT | table=learning_events lang=%s subj=%s | "
                "stt=%.2fs llm=%.2fs total=%.2fs | took=%.0fms",
                row.get("language", "?"), row.get("subject", "?"),
                float(row.get("stt_time", 0) or 0), float(row.get("llm_time", 0) or 0),
                float(row.get("total_time", 0) or 0),
                (_cht.time() - _t0) * 1000,
            )
        except Exception as e:
            log.error(f"🟧 CLICKHOUSE INSERT FAIL | {e}")

    def _ch_kpi_summary(self, hours: int) -> dict:
        try:
            r = self._ch.query(f"""
                SELECT
                    count()              AS total,
                    avg(total_time)      AS avg_time,
                    avg(stt_time)        AS avg_stt,
                    avg(llm_time)        AS avg_llm,
                    avg(tts_time)        AS avg_tts,
                    sum(kpi_ok) / count() * 100 AS kpi_rate
                FROM learning_events
                WHERE event_time >= now() - INTERVAL {hours} HOUR
                  AND event_type = 'qa'
            """)
            row = r.first_row
            return {
                "total": int(row[0] or 0), "avg_time": round(float(row[1] or 0), 2),
                "avg_stt": round(float(row[2] or 0), 2),
                "avg_llm": round(float(row[3] or 0), 2),
                "avg_tts": round(float(row[4] or 0), 2),
                "kpi_rate_pct": round(float(row[5] or 0), 1),
            }
        except Exception as e:
            log.error("CH kpi_summary: %s", e)
            return {}

    def _ch_progression(self, course_id: str) -> list[dict]:
        try:
            r = self._ch.query(f"""
                SELECT chapter_idx, section_idx, count() AS views
                FROM learning_events
                WHERE course_id = '{course_id}' AND event_type = 'section_start'
                GROUP BY chapter_idx, section_idx
                ORDER BY chapter_idx, section_idx
            """)
            return [{"chapter": row[0], "section": row[1], "views": row[2]}
                    for row in r.result_rows]
        except Exception:
            return []

    def _mem_kpi_summary(self, hours: int) -> dict:
        cutoff = time.time() - hours * 3600
        evts   = [e for e in self._cache if e.event_type == "qa"]
        if not evts:
            return {"total": 0}
        times = [e.total_time for e in evts if e.total_time > 0]
        return {
            "total":        len(evts),
            "avg_time":     round(sum(times) / len(times), 2) if times else 0,
            "avg_stt":      round(sum(e.stt_time for e in evts) / len(evts), 2),
            "avg_llm":      round(sum(e.llm_time for e in evts) / len(evts), 2),
            "avg_tts":      round(sum(e.tts_time for e in evts) / len(evts), 2),
            "kpi_rate_pct": round(sum(1 for e in evts if e.kpi_ok) / len(evts) * 100, 1),
        }

    def _mem_progression(self, course_id: str) -> list[dict]:
        seen: dict[tuple, int] = {}
        for e in self._cache:
            if e.course_id == course_id and e.event_type == "section_start":
                k = (e.chapter_idx, e.section_idx)
                seen[k] = seen.get(k, 0) + 1
        return [{"chapter": k[0], "section": k[1], "views": v}
                for k, v in sorted(seen.items())]


# Singleton
_analytics: Optional[AnalyticsEngine] = None

def get_analytics() -> AnalyticsEngine:
    global _analytics
    if _analytics is None:
        _analytics = AnalyticsEngine()
    return _analytics