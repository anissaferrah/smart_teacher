"""Health, RAG status, ingestion status, debug, embedding cache stats."""

from fastapi import APIRouter

from core.config import Config
from deps import get_rag, get_ingestion_manager
from handlers.session_manager import count_http_sessions

router = APIRouter()


@router.get("/health")
async def health():
    rag = get_rag()
    return {
        "server":   "ok",
        "rag":      {"ready": rag.is_ready, **rag.get_stats()},
        "whisper":  Config.WHISPER_MODEL_SIZE,
        "llm":      Config.GPT_MODEL,
        "tts":      Config.TTS_PROVIDER,
        "sessions": await count_http_sessions(),
    }


@router.get("/rag/stats")
async def rag_stats():
    return get_rag().get_stats()


@router.get("/ingestion/status")
async def get_ingestion_status():
    """Endpoint pour tracker l'état d'ingestion en cours."""
    return await get_ingestion_manager().get_status()


@router.get("/debug/rag_test")
async def debug_rag_test(q: str = "explain this topic", k: int = 5):
    """DEBUG: teste le RAG et affiche ce qu'il retriève."""
    rag = get_rag()
    if not rag.is_ready:
        return {"error": "RAG not ready", "status": rag.get_status()}
    try:
        result = rag.debug_retrieve(q, k=k)
        return {"status": "ok", "debug_data": result, "rag_status": rag.get_status()}
    except Exception as exc:
        return {"error": str(exc), "status": rag.get_status()}


@router.get("/cache/stats")
async def cache_stats():
    """Stats du cache d'embeddings (Redis)."""
    from rag.embedding_cache import embedding_cache

    stats = embedding_cache.stats()
    rag_stats = get_rag().get_stats()

    return {
        "status": "ok",
        "embedding_cache": stats,
        "rag_status": {
            "total_docs": rag_stats.get("total_docs"),
            "collection": rag_stats.get("collection"),
            "bm25_ready": rag_stats.get("bm25_ready"),
        },
        "recommendations": {
            "note": "hit_rate > 0.6 = très bon. < 0.3 = queries très variées.",
            "redis_tip": "Assurez-vous que Redis est running: `redis-cli ping`",
        },
    }
