"""Analytics endpoints — KPIs entreprise (cahier des charges) + analytics historiques."""

from fastapi import APIRouter, HTTPException

from deps import get_analytics_engine
from observability.kpi_logger import KPITracker

router = APIRouter(prefix="/analytics")


@router.get("/report")
async def analytics_report():
    """Rapport analytics complet."""
    return get_analytics_engine().full_report()


@router.get("/kpi")
async def analytics_kpi():
    """KPIs entreprise temps réel — rolling window des 200 dernières mesures.

    Retourne les 4 KPIs explicitement demandés au cahier des charges :
      - interrupt_latency_s : VAD detect → TTS stop  (target < 500ms)
      - response_latency_s  : audio_end → first audio chunk back  (target < 5s)
      - wer                 : Word Error Rate STT  (target < 5%)
      - mos                 : Naturalité TTS via feedback user  (target > 4.0)

    Pour chaque KPI : count, p50, p95, p99, mean, compliance_rate.
    """
    return KPITracker.get().summary()


@router.post("/kpi/feedback/mos")
async def submit_mos_feedback(session_id: str, turn_id: int, mos: float):
    """L'étudiant note la qualité de la voix TTS (1.0 → 5.0)."""
    if not (1.0 <= mos <= 5.0):
        raise HTTPException(status_code=400, detail="mos must be in [1.0, 5.0]")
    KPITracker.get().record_mos(session_id, turn_id, mos)
    return {"status": "ok", "session_id": session_id, "turn_id": turn_id, "mos": mos}


@router.get("/kpi/transcripts/recent")
async def list_recent_transcripts(limit: int = 50, language: str = ""):
    """Liste les N dernières transcriptions STT pour review humaine.

    Lit le CSV stt_metrics.csv (tail). Utilisé par l'admin UI pour identifier
    les transcriptions douteuses à corriger manuellement.
    """
    import csv
    from pathlib import Path
    from core.config import Config

    path = Path(Config.STT_LOG_FILE)
    if not path.exists():
        return {"items": [], "total": 0}

    # Lire les N dernières lignes (simple tail)
    lines: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
        for row in all_rows[-limit * 2 :]:        # over-fetch pour le filtre
            if language and row.get("language_detected") != language:
                continue
            lines.append({
                "timestamp":          row.get("timestamp"),
                "session_id":         row.get("session_id"),
                "utt_id":             row.get("utt_id"),
                "language_detected":  row.get("language_detected"),
                "language_prob":      row.get("language_prob"),
                "audio_duration_sec": row.get("audio_duration_sec"),
                "transcription_text": row.get("transcription_text"),
                "stt_confidence":     row.get("stt_confidence"),
                "wer_already_set":    bool(row.get("wer")),
            })
        return {"items": lines[-limit:], "total": len(all_rows)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcripts read error: {exc}")


@router.post("/kpi/transcripts/{utt_id}/correct")
async def correct_transcript(
    utt_id: str,
    session_id: str,
    turn_id: int,
    hypothesis: str,        # ce que Whisper a transcrit (mauvais)
    reference: str,         # la version corrigée (vérité terrain)
):
    """Soumet une correction → calcule WER + alimente le KPITracker.

    Le frontend admin envoie :
      - hypothesis : transcription Whisper (récupérée via /kpi/transcripts/recent)
      - reference  : version corrigée par le humain
    On calcule le WER, on record dans le KPITracker, on retourne les détails.
    """
    from observability.wer import compute_wer
    if not reference.strip():
        raise HTTPException(status_code=400, detail="reference cannot be empty")

    result = compute_wer(reference=reference, hypothesis=hypothesis)
    KPITracker.get().record_wer(
        session_id, turn_id,
        wer=result.wer,
        ref_len=result.ref_words,
    )
    return {
        "status":    "ok",
        "utt_id":    utt_id,
        "wer":       result.to_dict(),
        "kpi_target": 0.05,
    }


@router.get("/kpi/historical")
async def analytics_kpi_historical(hours: int = 24):
    """KPIs agrégés sur les N dernières heures (depuis ClickHouse / CSV)."""
    return get_analytics_engine().kpi_summary(hours=hours)


@router.get("/progression/{course_id}")
async def analytics_progression(course_id: str):
    """Progression des sections pour un cours."""
    return {
        "course_id":   course_id,
        "progression": get_analytics_engine().progression_by_course(course_id),
    }


@router.get("/latency")
async def analytics_latency():
    """Distribution des latences (min/avg/max/p95)."""
    return get_analytics_engine().latency_distribution()
