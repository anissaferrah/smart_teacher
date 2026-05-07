"""Search endpoints — full-text search over transcript history."""

from fastapi import APIRouter, HTTPException

from deps import get_transcript_searcher

router = APIRouter(prefix="/search")


@router.get("/transcripts")
async def search_transcripts(
    q: str,
    language: str = "",
    course_id: str = "",
    role: str = "",
    limit: int = 20,
):
    """Recherche full-text dans l'historique des transcriptions."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Paramètre 'q' requis")
    results = get_transcript_searcher().search(
        q, language=language, course_id=course_id, role=role, limit=limit,
    )
    return {"query": q, "count": len(results), "results": results}


@router.get("/session/{session_id}")
async def get_session_transcript(session_id: str):
    """Retourne l'historique complet d'une session."""
    history = get_transcript_searcher().get_session_history(session_id)
    return {"session_id": session_id, "count": len(history), "history": history}


@router.get("/stats")
async def search_stats():
    """Statistiques sur l'index de recherche."""
    return get_transcript_searcher().get_stats()
