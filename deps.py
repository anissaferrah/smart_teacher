"""
Dependency registry — single source of truth for shared services.

main.py instantiates services at startup and calls register_services().
Other modules (routes/, services/, handlers/) call the get_*() accessors
to retrieve them without circular imports back to main.py.
"""

from typing import Any

_services: dict[str, Any] = {}


def register_services(**kwargs: Any) -> None:
    """Called once from main.py at startup."""
    _services.update(kwargs)


def _require(name: str) -> Any:
    if name not in _services:
        raise RuntimeError(
            f"Service '{name}' not registered. main.py must call "
            f"deps.register_services({name}=..., ...) at startup."
        )
    return _services[name]


# Active VoiceStateMachines registry — populated by the WS handler in main.py,
# read by routes/voice.py for diagnostics.
active_vfsms: dict[str, Any] = {}


def get_transcriber() -> Any:    return _require("transcriber")
def get_brain() -> Any:          return _require("brain")
def get_voice() -> Any:          return _require("voice")
def get_rag() -> Any:            return _require("rag")
def get_dialogue() -> Any:       return _require("dialogue")
def get_profile_mgr() -> Any:    return _require("profile_mgr")
def get_csv_logger() -> Any:     return _require("csv_logger")
def get_stt_logger() -> Any:     return _require("stt_logger")
def get_transcript_searcher() -> Any: return _require("transcript_searcher")
def get_analytics_engine() -> Any:    return _require("analytics_engine")
def get_agentic_orchestrator() -> Any: return _require("agentic_orchestrator")
def get_ingestion_manager() -> Any:   return _require("ingestion_manager")
def get_slide_sync() -> Any:     return _require("slide_sync")
def get_media_storage() -> Any:  return _require("media_storage")
def get_teaching_graph() -> Any: return _require("teaching_graph")
def get_qa_graph() -> Any:       return _require("qa_graph")
def get_audio_input() -> Any:    return _require("audio_input")
