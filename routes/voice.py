"""Voice + agentic diagnostics endpoints."""

from fastapi import APIRouter, HTTPException

import deps

router = APIRouter()


@router.get("/voice/state/{session_id}")
async def get_voice_state(session_id: str):
    """Diagnostic : état live du VoiceStateMachine d'une session."""
    vfsm = deps.active_vfsms.get(session_id)
    if vfsm is None:
        raise HTTPException(status_code=404, detail="No active FSM for this session")
    return vfsm.snapshot()


@router.get("/agentic/stats")
async def get_agentic_stats():
    """Diagnostic : stats globales de l'orchestrator agentic."""
    return deps.get_agentic_orchestrator().stats()


@router.post("/agentic/cancel/{session_id}")
async def cancel_agentic_session(session_id: str):
    """Force-cancel l'agentic task d'une session (debug / admin)."""
    cancelled = await deps.get_agentic_orchestrator().cancel(
        session_id, reason="manual_api_cancel",
    )
    return {"session_id": session_id, "cancelled": cancelled}
