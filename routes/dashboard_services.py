"""Dashboard services probe — RAG / Redis / Postgres / Ollama / ES / MinIO."""

import asyncio
import logging
import time
from fastapi import APIRouter

from core.config import Config
import deps
from pedagogy.dialogue import get_redis

router = APIRouter()
log = logging.getLogger("SmartTeacher.routes.dashboard_services")


@router.get("/dashboard/services")
async def dashboard_services():
    from sqlalchemy import text
    from database.init_db import AsyncSessionLocal
    from storage.transcript_search import get_searcher

    rag = deps.get_rag()
    brain = deps.get_brain()
    media_storage_obj = None
    try:
        from storage.media_storage import MediaStorage
        # Single instance — reuse the one created at startup if available
        # MediaStorage is a singleton-style class
        media_storage_obj = MediaStorage()
    except Exception:
        media_storage_obj = None

    services: dict[str, dict] = {}

    # RAG
    rag_status = rag.get_status()
    rag_stats = rag.get_stats()
    storage_status = media_storage_obj.get_status() if media_storage_obj else {"active": False}
    services["rag"] = {
        "healthy": bool(rag_status.get("rag_ready")),
        "ready": bool(rag_status.get("rag_ready")),
        "embedding_source": rag_status.get("embedding_source"),
        "embedding_model": rag_status.get("embedding_model"),
        "docs_loaded": rag_status.get("docs_loaded"),
        "bm25_ready": rag_status.get("bm25_available"),
        "qdrant_connected": rag_status.get("qdrant_connected"),
        "vectorstore_available": rag_status.get("vectorstore_available"),
        "collection": rag_stats.get("collection"),
        "backend": rag_stats.get("backend"),
        "role": "Orchestre la recherche hybride et la génération de réponses",
        "retrieves": "Chunks vectoriels Qdrant, scores BM25 et cache d'embeddings",
        "used_in": "rag/multimodal_rag.py, routes/rest.py /ask, routes/course.py /course/build, /rag/stats, /debug/rag_test",
    }

    # Redis
    redis_status = {
        "connected": False,
        "endpoint": f"{Config.REDIS_HOST}:{Config.REDIS_PORT}/{Config.REDIS_DB}",
        "latency_ms": None,
    }
    try:
        redis_client = await get_redis()
        redis_kwargs = getattr(redis_client.connection_pool, "connection_kwargs", {}) or {}
        redis_host = redis_kwargs.get("host", Config.REDIS_HOST)
        redis_port = redis_kwargs.get("port", Config.REDIS_PORT)
        redis_db = redis_kwargs.get("db", Config.REDIS_DB)
        start = time.time()
        await asyncio.wait_for(redis_client.ping(), timeout=2.0)
        redis_status.update({
            "connected": True,
            "endpoint": f"{redis_host}:{redis_port}/{redis_db}",
            "latency_ms": round((time.time() - start) * 1000, 1),
        })
    except Exception as exc:
        redis_status["error"] = str(exc)
    redis_status.update({
        "role": "Cache et état temps réel",
        "retrieves": "Sessions WebSocket, état temporaire et latence de traitement",
        "used_in": "handlers/session_manager.py, pedagogy/dialogue.py, routes/dashboard_services.py",
    })
    services["redis"] = redis_status

    # Postgres
    postgres_status = {
        "connected": False,
        "endpoint": f"{Config.POSTGRES_HOST}:{Config.POSTGRES_PORT}/{Config.POSTGRES_DB}",
        "latency_ms": None,
    }
    try:
        start = time.time()
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        postgres_status.update({
            "connected": True,
            "latency_ms": round((time.time() - start) * 1000, 1),
        })
    except Exception as exc:
        postgres_status["error"] = str(exc)
    postgres_status.update({
        "role": "Persistance transactionnelle",
        "retrieves": "Sessions, interactions, profils étudiants et événements d'apprentissage",
        "used_in": "database/models.py, database/init_db.py, routes/course.py /course/build, routes/rest.py /session",
    })
    services["postgres"] = postgres_status

    # Ollama probe (sync, run in thread)
    def _probe_ollama() -> dict:
        import requests
        fallback = brain.fallback
        base_url = getattr(fallback, "base_url", "http://localhost:11434")
        model_name = getattr(fallback, "model", "mistral")
        endpoint = f"{base_url}/api/tags"
        try:
            response = requests.get(endpoint, timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [str(m.get("name", "")).split(":")[0] for m in models]
                available = any(model_name in name for name in model_names)
                return {
                    "available": available,
                    "model": model_name,
                    "endpoint": endpoint,
                    "status": "✅ Ready" if available else "⚠️ Model missing",
                    "models": model_names,
                }
            return {
                "available": False, "model": model_name, "endpoint": endpoint,
                "status": f"❌ HTTP {response.status_code}",
            }
        except Exception as exc:
            return {
                "available": False, "model": model_name, "endpoint": endpoint,
                "status": "❌ Unavailable", "error": str(exc),
            }

    try:
        ollama_status = await asyncio.to_thread(_probe_ollama)
    except Exception as exc:
        ollama_status = {
            "available": False,
            "model": getattr(brain.fallback, "model", "mistral") if brain.fallback else "mistral",
            "endpoint": (getattr(brain.fallback, "base_url", "http://localhost:11434") + "/api/tags") if brain.fallback else "http://localhost:11434/api/tags",
            "status": "❌ Unavailable",
            "error": str(exc),
        }
    ollama_status.update({
        "role": "LLM local de secours",
        "retrieves": "Modèles disponibles via /api/tags et réponses locales via Ollama",
        "used_in": "ai/llm.py, routes/rest.py /ask, quiz fallback",
    })
    services["ollama"] = ollama_status

    # Elasticsearch (transcript search)
    try:
        search_stats = get_searcher().get_stats()
    except Exception as exc:
        search_stats = {"backend": "memory", "total": 0, "error": str(exc)}
    services["elasticsearch"] = {
        "available": search_stats.get("backend") == "elasticsearch",
        "backend": search_stats.get("backend"),
        "total": search_stats.get("total"),
        "host": search_stats.get("host"),
        "index": search_stats.get("index"),
        "note": search_stats.get("note"),
        "role": "Recherche full-text historique",
        "retrieves": "Questions, réponses et index texte des transcriptions",
        "used_in": "storage/transcript_search.py, routes/dashboard_services.py",
    }

    # MinIO
    services["minio"] = {
        "configured": storage_status.get("configured"),
        "active": storage_status.get("active"),
        "provider": storage_status.get("provider"),
        "endpoint": storage_status.get("endpoint"),
        "bucket": storage_status.get("bucket"),
        "secure": storage_status.get("secure"),
        "local_root": storage_status.get("local_root"),
        "role": "Stockage objet des médias",
        "retrieves": "PDF, slides, audio et objets listés via /media-list",
        "used_in": "storage/media_storage.py, routes/course.py /course/build",
    }

    return {"updated_at": time.time(), "services": services}
