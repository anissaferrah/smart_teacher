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
