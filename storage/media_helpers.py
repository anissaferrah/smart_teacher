"""Media persistence helpers — save bytes/JSON to MinIO/local storage."""

import asyncio
import json
import logging

from deps import get_media_storage

log = logging.getLogger("SmartTeacher.storage.media_helpers")


async def save_media_bytes(object_name: str, data: bytes, content_type: str) -> None:
    """Persist arbitrary bytes to media storage (best-effort)."""
    try:
        ms = get_media_storage()
        await asyncio.to_thread(ms.upload_bytes, data, object_name, content_type)
    except Exception as exc:
        log.debug("media save skipped (%s): %s", object_name, exc)


async def save_media_json(object_name: str, payload: dict) -> None:
    """Persist a dict as pretty-printed JSON to media storage."""
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    await save_media_bytes(object_name, data, "application/json")
