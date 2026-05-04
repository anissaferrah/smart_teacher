from fastapi import APIRouter
from handlers import rest_routes

# Note: GET /session/{session_id} is intentionally NOT in this router —
# main.py's version returns dialogue.get_stats() (Redis dialogue state),
# which is the WebSocket session, distinct from rest_routes.handle_session_get
# (the REST /ask conversation history, also Redis-backed).
router = APIRouter(prefix="/session")


@router.get("/{session_id}/profile")
async def get_profile(session_id: str):
    return await rest_routes.handle_session_profile(session_id)


@router.post("/{session_id}/profile")
async def update_profile(session_id: str, payload: dict):
    return await rest_routes.handle_session_profile_update(session_id, payload)


@router.get("/{session_id}/tts_params")
async def tts_params(session_id: str, confusion_score: float = 0.0):
    return await rest_routes.handle_session_tts_params(session_id, confusion_score)


@router.post("/{session_id}/speech_rate")
async def set_speech_rate(session_id: str, payload: dict):
    """Student-driven speech rate override.

    UI sends ``{"rate": 0.85, "manual_override": true}`` (rate as multiplier
    in [0.5, 1.5]). When ``manual_override`` is true, both the bandit and
    confusion-driven adaptation are bypassed for this student — the rate
    they picked is the rate they get. Sending ``manual_override=false``
    re-enables the automatic adaptation.

    Convenience wrapper around POST /session/{id}/profile that targets only
    the speech_rate fields. Returns the updated TTS params.
    """
    try:
        rate = float(payload.get("rate", 1.0))
        rate = max(0.5, min(1.5, rate))     # clamp to safe Edge-TTS range
    except (TypeError, ValueError):
        rate = 1.0
    manual = bool(payload.get("manual_override", True))

    from pedagogy.personalization.profile import get_or_create_profile, update_profile
    profile = await get_or_create_profile(session_id)
    prefs = dict(profile.get("preferences") or {})
    prefs["speech_rate"]          = round(rate, 2)
    prefs["manual_rate_override"] = manual
    await update_profile(session_id, {"preferences": prefs})

    from pedagogy.personalization.tts_adapter import compute_tts_params, rate_float_to_edge_str
    params = await compute_tts_params(session_id)
    return {
        "status": "ok",
        "rate":            params.get("rate"),
        "edge_tts_rate":   rate_float_to_edge_str(params.get("rate", 1.0)),
        "manual_override": params.get("manual_override"),
        "rate_source":     params.get("rate_source"),
    }
