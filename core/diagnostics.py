"""Startup diagnostics — log connectivity to ES / MinIO / ClickHouse / Redis."""

import asyncio
import logging

from core.config import Config
from deps import (
    get_transcript_searcher,
    get_media_storage,
    get_analytics_engine,
)
from pedagogy.dialogue import get_redis

log = logging.getLogger("SmartTeacher.Diagnostics")


async def log_backend_diagnostics() -> None:
    """Log backend connectivity at startup so the active fallback path is visible in the terminal."""
    log.info("🔎 Diagnostic démarrage des services de données:")

    try:
        search_stats = get_transcript_searcher().get_stats()
        if search_stats.get("backend") == "elasticsearch":
            log.info("   • Elasticsearch: ✅ connecté (%s)", search_stats.get("host", "n/a"))
        else:
            log.info("   • Elasticsearch: ℹ️ recherche en mémoire (fallback actif)")
    except Exception as exc:
        log.info("   • Elasticsearch: ℹ️ recherche en mémoire (%s)", exc)

    try:
        storage_status = get_media_storage().get_status()
        if storage_status.get("provider") == "minio":
            log.info("   • MinIO: ✅ connecté (%s)", storage_status.get("endpoint", "n/a"))
        else:
            log.info("   • MinIO: ℹ️ stockage local (%s)", storage_status.get("local_root", "media"))
    except Exception as exc:
        log.info("   • MinIO: ℹ️ stockage local (%s)", exc)

    try:
        analytics_engine = get_analytics_engine()
        analytics_engine._init_ch()
        report = analytics_engine.full_report()
        if report.get("backend") == "clickhouse":
            log.info("   • ClickHouse: ✅ connecté")
        else:
            log.info("   • ClickHouse: ℹ️ analytics CSV + mémoire")
    except Exception as exc:
        log.info("   • ClickHouse: ℹ️ analytics CSV + mémoire (%s)", exc)

    try:
        redis_client = await get_redis()
        await asyncio.wait_for(redis_client.ping(), timeout=2.0)
        redis_kwargs = getattr(redis_client.connection_pool, "connection_kwargs", {}) or {}
        redis_host = redis_kwargs.get("host", Config.REDIS_HOST)
        redis_port = redis_kwargs.get("port", Config.REDIS_PORT)
        log.info("   • Redis: ✅ connecté (%s:%s)", redis_host, redis_port)
    except Exception as exc:
        log.info("   • Redis: ⚠️ indisponible (%s)", exc)
