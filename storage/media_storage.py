"""
Smart Teacher — Media Storage Module.

Unified abstraction layer for media file storage:
    - Local filesystem storage (default fallback)
    - MinIO S3-compatible object storage (if configured)
    - Automatic provider detection and fallback
    
Supported media types:
    - PDF documents (user uploads)
    - Slide images (PNG/JPG from PDF conversion)
    - Audio files (MP3 from TTS generation)
    - Course resources and metadata
    
Usage:
    storage = MediaStorage()
    
    # Upload
    url = storage.upload_bytes(b"...", "slides/ch1/1.png", "image/png")
    
    # Download
    audio_bytes = storage.get_bytes("audio/response_123.mp3")
    
    # Check existence
    exists = storage.exists("pdfs/chapter_1.pdf")
    
    # List contents
    files = storage.list_objects("slides/ch1/")
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

from core.config import Config

log = logging.getLogger("SmartTeacher.MediaStorage")

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION (read from core.config.Config)
# ════════════════════════════════════════════════════════════════════════

LOCAL_MEDIA_DIR: Path = Path(Config.LOCAL_MEDIA_DIR)


# ════════════════════════════════════════════════════════════════════════
# MEDIA STORAGE
# ════════════════════════════════════════════════════════════════════════


class MediaStorage:
    """
    Unified media storage interface with automatic provider fallback.
    
    Attempts to use MinIO if configured, otherwise falls back to
    local filesystem storage. All operations are transparent to caller.
    
    Attributes
    ----------
    _minio : object or None
        MinIO client instance (if initialized)
    _use_minio : bool
        Flag indicating active provider (False = local, True = MinIO)
    """

    def __init__(self) -> None:
        """Initialize storage backend automatically."""
        self._minio: Optional[object] = None
        self._use_minio: bool = False
        self._initialized = False  # Lazy initialization flag
        self._init_local_dirs()

    def _init_local_dirs(self) -> None:
        """Initialize local directories only (non-blocking)."""
        LOCAL_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        (LOCAL_MEDIA_DIR / "slides").mkdir(exist_ok=True)
        (LOCAL_MEDIA_DIR / "pdfs").mkdir(exist_ok=True)
        (LOCAL_MEDIA_DIR / "audio").mkdir(exist_ok=True)
        log.info(f"📁 Local storage directories ready: {LOCAL_MEDIA_DIR}")

    def _init_storage(self) -> None:
        """
        Initialize storage backend with automatic fallback (lazy).
        
        Attempts MinIO connection; falls back to local filesystem if
        MinIO is not configured or unavailable.
        """
        if self._initialized:
            return
        
        self._initialized = True

        if not Config.MINIO_ENDPOINT:
            log.info("📁 Using local storage (MinIO not configured)")
            return

        try:
            from minio import Minio
            import urllib3

            http_client = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=2.0, read=5.0),
            )

            self._minio = Minio(
                Config.MINIO_ENDPOINT,
                access_key=Config.MINIO_ACCESS_KEY,
                secret_key=Config.MINIO_SECRET_KEY,
                secure=Config.MINIO_SECURE,
                http_client=http_client,
                region="us-east-1",
            )

            # Ensure bucket exists
            if not self._minio.bucket_exists(Config.MINIO_BUCKET):
                self._minio.make_bucket(Config.MINIO_BUCKET)
                log.info(f"🪣 MinIO bucket created: {Config.MINIO_BUCKET}")
            else:
                log.info(f"✅ MinIO connected: {Config.MINIO_ENDPOINT}/{Config.MINIO_BUCKET}")

            self._use_minio = True

        except ImportError:
            log.warning("⚠️ minio package not installed → local storage fallback")
        except Exception as exc:
            log.warning(f"⚠️ MinIO unavailable ({exc}) → local storage fallback")

    # ════════════════════════════════════════════════════════════════════════
    # UPLOAD OPERATIONS
    # ════════════════════════════════════════════════════════════════════════

    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload bytes to storage and return accessible URL/path.
        
        Parameters
        ----------
        data : bytes
            File content
        object_name : str
            Storage path (e.g., "slides/ch1/1.png", "audio/response.mp3")
        content_type : str, optional
            MIME type (default: "application/octet-stream")
        
        Returns
        -------
        str
            Public URL (MinIO) or relative path (local)
        """
        self._init_storage()
        if self._use_minio:
            return self._minio_upload_bytes(data, object_name, content_type)
        return self._local_save_bytes(data, object_name)

    def upload_file(self, file_path: str, object_name: str) -> str:
        """
        Upload file from disk to storage.
        
        Parameters
        ----------
        file_path : str
            Local file pathname
        object_name : str
            Storage destination path
        
        Returns
        -------
        str
            Public URL or relative path
        """
        if self._use_minio:
            return self._minio_upload_file(file_path, object_name)
        return self._local_copy_file(file_path, object_name)

    # ════════════════════════════════════════════════════════════════════════
    # DOWNLOAD OPERATIONS
    # ════════════════════════════════════════════════════════════════════════

    def get_url(self, object_name: str, expires_hours: int = 24) -> str:
        """
        Generate presigned URL for file access.
        
        Parameters
        ----------
        object_name : str
            Storage path
        expires_hours : int, optional
            URL expiration time in hours (default: 24)
        
        Returns
        -------
        str
            Presigned URL (MinIO) or relative path (local)
        """
        if self._use_minio:
            try:
                from datetime import timedelta
                return self._minio.presigned_get_object(
                    Config.MINIO_BUCKET,
                    object_name,
                    expires=timedelta(hours=expires_hours)
                )
            except Exception as exc:
                log.error(f"MinIO URL generation failed: {exc}")
                return ""

        # Local: return relative URL
        return f"/media/{object_name}"

    def get_bytes(self, object_name: str) -> Optional[bytes]:
        """
        Download file content as bytes.
        
        Parameters
        ----------
        object_name : str
            Storage path
        
        Returns
        -------
        bytes or None
            File content if found, None otherwise
        """
        if self._use_minio:
            try:
                response = self._minio.get_object(Config.MINIO_BUCKET, object_name)
                data = response.read()
                response.close()
                return data
            except Exception as exc:
                log.error(f"MinIO download failed: {exc}")
                return None

        # Local: read from filesystem
        path = LOCAL_MEDIA_DIR / object_name
        return path.read_bytes() if path.exists() else None

    # ════════════════════════════════════════════════════════════════════════
    # METADATA OPERATIONS
    # ════════════════════════════════════════════════════════════════════════

    def exists(self, object_name: str) -> bool:
        """
        Check if file exists in storage.
        
        Parameters
        ----------
        object_name : str
            Storage path
        
        Returns
        -------
        bool
            True if file exists, False otherwise
        """
        if self._use_minio:
            try:
                self._minio.stat_object(Config.MINIO_BUCKET, object_name)
                return True
            except Exception:
                return False

        return (LOCAL_MEDIA_DIR / object_name).exists()

    def delete(self, object_name: str) -> bool:
        """
        Delete file from storage.
        
        Parameters
        ----------
        object_name : str
            Storage path
        
        Returns
        -------
        bool
            True if deleted successfully, False otherwise
        """
        if self._use_minio:
            try:
                self._minio.remove_object(Config.MINIO_BUCKET, object_name)
                return True
            except Exception:
                return False

        path = LOCAL_MEDIA_DIR / object_name
        if path.exists():
            path.unlink()
            return True
        return False

    def list_objects(self, prefix: str = "") -> List[str]:
        """
        List all objects under prefix.
        
        Parameters
        ----------
        prefix : str, optional
            Storage path prefix (e.g., "slides/ch1/")
        
        Returns
        -------
        list[str]
            List of object names matching prefix
        """
        if self._use_minio:
            try:
                objects = self._minio.list_objects(Config.MINIO_BUCKET, prefix=prefix, recursive=True)
                return [o.object_name for o in objects]
            except Exception:
                return []
        base = LOCAL_MEDIA_DIR / prefix if prefix else LOCAL_MEDIA_DIR
        if not base.exists():
            return []
        return [str(p.relative_to(LOCAL_MEDIA_DIR)) for p in base.rglob("*") if p.is_file()]

    def get_status(self) -> dict:
        """Return a compact status snapshot for dashboards and diagnostics."""
        self._init_storage()
        provider = "minio" if self._use_minio else "local"
        endpoint = f"{Config.MINIO_ENDPOINT}/{Config.MINIO_BUCKET}" if Config.MINIO_ENDPOINT else "local"
        return {
            "configured": bool(Config.MINIO_ENDPOINT),
            "active": self._use_minio,
            "provider": provider,
            "endpoint": endpoint,
            "bucket": Config.MINIO_BUCKET,
            "secure": Config.MINIO_SECURE,
            "local_root": str(LOCAL_MEDIA_DIR),
        }

    # ── MinIO internals ──────────────────────────────────────────────────

    def _minio_upload_bytes(self, data: bytes, object_name: str, content_type: str) -> str:
        import io
        try:
            self._minio.put_object(
                Config.MINIO_BUCKET, object_name,
                io.BytesIO(data), len(data),
                content_type=content_type,
            )
            return self.get_url(object_name)
        except Exception as e:
            log.error("MinIO upload_bytes error: %s", e)
            return self._local_save_bytes(data, object_name)

    def _minio_upload_file(self, file_path: str, object_name: str) -> str:
        try:
            self._minio.fput_object(Config.MINIO_BUCKET, object_name, file_path)
            return self.get_url(object_name)
        except Exception as e:
            log.error("MinIO upload_file error: %s", e)
            return self._local_copy_file(file_path, object_name)

    # ── Local internals ──────────────────────────────────────────────────

    def _local_save_bytes(self, data: bytes, object_name: str) -> str:
        path = LOCAL_MEDIA_DIR / object_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"/media/{object_name}"

    def _local_copy_file(self, file_path: str, object_name: str) -> str:
        import shutil
        dest = LOCAL_MEDIA_DIR / object_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        return f"/media/{object_name}"

    # ── Helpers métier ───────────────────────────────────────────────────

    def save_course_pdf(self, course_id: str, filename: str, data: bytes) -> str:
        """Sauvegarde le PDF d'un cours."""
        obj = f"pdfs/{course_id}/{filename}"
        return self.upload_bytes(data, obj, "application/pdf")

    def save_audio(self, session_id: str, data: bytes, mime: str = "audio/mpeg") -> str:
        """Sauvegarde un fichier audio TTS."""
        ts  = int(time.time())
        ext = "mp3" if "mpeg" in mime else "webm"
        obj = f"audio/{session_id}/{ts}.{ext}"
        return self.upload_bytes(data, obj, mime)

    def save_slide_image(
        self,
        course_id: str,
        slide_idx: int,
        data: bytes,
        *,
        domain: str | None = None,
        course: str | None = None,
        chapter: str | None = None,
    ) -> str:
        """Sauvegarde une image de slide."""
        if domain and course and chapter:
            obj = f"slides/{domain}/{course}/{chapter}/page_{slide_idx:03d}.png"
        else:
            obj = f"slides/{course_id}/slide_{slide_idx:04d}.png"
        return self.upload_bytes(data, obj, "image/png")


# Singleton
_storage: Optional[MediaStorage] = None

def get_storage() -> MediaStorage:
    global _storage
    if _storage is None:
        _storage = MediaStorage()
    return _storage
