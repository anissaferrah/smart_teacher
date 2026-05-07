"""Smart Teacher — Ingestion Manager with async state tracking"""

import asyncio
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("SmartTeacher.IngestionManager")


class IngestionState(str, Enum):
    IDLE = "idle"  # Pas d'ingestion en cours
    PROCESSING = "processing"  # En cours
    READY = "ready"  # Successful completion
    ERROR = "error"  # Failed


@dataclass
class IngestionStatus:
    """Statut d'une ingestion en cours"""
    state: IngestionState = IngestionState.IDLE
    progress: int = 0  # 0-100
    total_chunks: int = 0
    processed_files: int = 0
    total_files: int = 0
    error_message: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "progress": self.progress,
            "total_chunks": self.total_chunks,
            "processed_files": self.processed_files,
            "total_files": self.total_files,
            "error_message": self.error_message,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


class IngestionManager:
    """
    Gère l'état d'ingestion asynchrone.
    
    Permet de:
    - Tracer l'ingestion en arrière-plan
    - Retourner le statut (progress, errors)
    - Bloquer les opérations RAG pendant l'ingestion
    """
    
    def __init__(self):
        self.status = IngestionStatus()
        self._lock = asyncio.Lock()
        self._current_task: Optional[asyncio.Task] = None
    
    async def start_ingestion(self, total_files: int) -> None:
        """Démarre un nouvelle ingestion"""
        async with self._lock:
            self.status = IngestionStatus(
                state=IngestionState.PROCESSING,
                total_files=total_files,
                start_time=time.time()
            )
            log.info(f"🚀 Ingestion démarrée ({total_files} fichiers)")
    
    async def update_progress(
        self,
        processed_files: int,
        chunks_count: int,
        progress_percent: int
    ) -> None:
        """Met à jour la progression"""
        async with self._lock:
            self.status.processed_files = processed_files
            self.status.total_chunks = chunks_count
            self.status.progress = min(100, progress_percent)
            log.debug(f"📊 Progress: {progress_percent}% ({processed_files}/{self.status.total_files} files, {chunks_count} chunks)")
    
    async def complete_ingestion(self, total_chunks: int) -> None:
        """Marque l'ingestion comme complétée"""
        async with self._lock:
            self.status.state = IngestionState.READY
            self.status.progress = 100
            self.status.total_chunks = total_chunks
            self.status.end_time = time.time()
            log.info(f"✅ Ingestion complétée | {total_chunks} chunks | {self.status.elapsed_seconds:.1f}s")
    
    async def fail_ingestion(self, error_message: str) -> None:
        """Marque l'ingestion comme échouée"""
        async with self._lock:
            self.status.state = IngestionState.ERROR
            self.status.error_message = error_message
            self.status.end_time = time.time()
            log.error(f"❌ Ingestion échouée: {error_message}")
    
    async def is_busy(self) -> bool:
        """Retourne True si ingestion en cours"""
        async with self._lock:
            return self.status.state == IngestionState.PROCESSING
    
    async def get_status(self) -> dict:
        """Retourne le statut actuel"""
        async with self._lock:
            return self.status.to_dict()
    
    async def reset(self) -> None:
        """Réinitialise le statut"""
        async with self._lock:
            self.status = IngestionStatus()
            log.info("🔄 Ingestion status reset")

# Singleton instance
ingestion_manager = IngestionManager()
