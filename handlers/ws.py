"""
Smart Teacher WebSocket handler — pipeline vocal temps réel.

Ce module héberge le handler unique `/ws/{session_id}`. Il dépend de tous
les services partagés via deps.get_*() (transcriber, brain, voice, rag,
dialogue, etc.) — ces services sont initialisés et enregistrés dans main.py
au démarrage. Le handler est monté via `app.include_router(ws_router)`.

Le handler reste une fonction async monolithique (~3200 lignes) car son
état est intriqué : FSM voix, anti-echo (xcorr FFT), orchestrator agentic
cancellable, presentation cursor, learning session DB, slide cache cross-
session. Une décomposition propre nécessiterait une classe WSSessionHandler
— refactor à part entière.
"""

# ── stdlib ─────────────────────────────────────────────────────────────
import asyncio
import base64
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime

# ── third-party ────────────────────────────────────────────────────────
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# ── core / config ──────────────────────────────────────────────────────
from core.config import Config


class _SkipLLMPipeline(Exception):
    """Sentinel raised when a Q&A cache hit lets us bypass the LLM
    pipeline. Caught locally by the surrounding try/except so the
    cached answer flows straight to TTS without 3-5 LLM calls.
    Internal control-flow only — not a real error."""
    pass


# ── audio / voice ──────────────────────────────────────────────────────
from audio.echo_helpers import (
    is_echo_of_recent_tts as _is_echo_of_recent_tts,
    tts_bytes_to_pcm as _tts_bytes_to_pcm,
)
from audio.voice.event_bus import EventBus
from audio.voice.state_machine import VoiceStateMachine, VoiceState

# ── pedagogy / dialogue ────────────────────────────────────────────────
from pedagogy.dialogue import DialogState, SessionContext
from pedagogy.personalization.tts_adapter import compute_tts_params

# ── services (extracted helpers) ───────────────────────────────────────
from services.course_slides import load_course_slide_context
from services.learning_log import (
    persist_learning_turn as _persist_learning_turn_external,
    safe_uuid as _safe_uuid_external,
)
from services.presentation import (
    compute_text_cursor_from_audio_progress as _cursor_from_audio_progress,
    decide_narration_cache_reuse as _decide_narration_cache_reuse,
    explain_slide_focused as _explain_slide_external,
    rewind_to_last_sentence_start as _rewind_to_last_sentence_start,
    synthesize_cached_tts as _synth_cached_tts_external,
)
from pedagogy.resume_intelligence import (
    ResumeContext as _ResumeCtx,
    ResumeStrategy as _ResumeStrategy,
    compose_resume_action as _compose_resume_action,
)
from services.nav_dispatcher import dispatch_nav_action as _dispatch_nav_action

# ── observability ──────────────────────────────────────────────────────
from observability.dashboard import (
    record_checkpoint_event, record_session_event, record_trace_event,
)

# ── handlers (audio pipeline) ──────────────────────────────────────────
from handlers.audio_pipeline import run_pipeline_streaming
from handlers.session_manager import (
    audio_bytes_to_numpy, consume_session_token, detect_lang_text,
    detect_subject, get_session_token,
)

# ── storage / cache ────────────────────────────────────────────────────
from cache.slide_cache import (
    get_narration as cache_get_narration,
    set_narration as cache_set_narration,
)
from storage.media_helpers import save_media_bytes, save_media_json

# ── database ───────────────────────────────────────────────────────────
from database.crud import (
    create_learning_session, log_interaction, log_learning_event,
    update_session_state,
)
from database.init_db import AsyncSessionLocal

# ── shared deps registry (services + active VFSMs) ─────────────────────
import deps as _deps

log = logging.getLogger("SmartTeacher.WS")
router = APIRouter()


# ══════════════════════════════════════════════════════════════════════

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint principal.
    Gère le cycle complet : présentation → écoute → réponse → reprise.
    """

    # Resolve shared services (registered by main.py at startup).
    transcriber          = _deps.get_transcriber()
    brain                = _deps.get_brain()
    voice                = _deps.get_voice()
    rag                  = _deps.get_rag()
    dialogue             = _deps.get_dialogue()
    profile_mgr          = _deps.get_profile_mgr()
    csv_logger           = _deps.get_csv_logger()
    stt_logger           = _deps.get_stt_logger()
    slide_sync           = _deps.get_slide_sync()
    audio_input          = _deps.get_audio_input()
    agentic_orchestrator = _deps.get_agentic_orchestrator()
    teaching_graph       = _deps.get_teaching_graph()
    qa_graph             = _deps.get_qa_graph()

    await websocket.accept()
    log.info(f"🔌 WebSocket connecté : {session_id[:8]}")
    # NB : le watchdog FSM est demarre PLUS BAS, apres init de vfsm + _vfsm_cancel_callback

    ctx: SessionContext | None = None
    audio_buffer: list[bytes] = []
    history: list[dict]       = []
    # Tracks the last course_id we saw on this connection. When the
    # FE switches course mid-session, this lets us reset the
    # conversational history so the LLM doesn't carry over Q&A
    # context from a different syllabus (the operator-visible bug
    # was: "tu peux répéter la slide?" on an IR slide getting
    # answered with content about supervised learning because
    # history still held a previous course's exchange).
    last_active_course_id: str | None = None
    # Passive-reader detection : counts slides where the student
    # didn't interact at all (no pause, no question, no manual nav).
    # Reset to 0 when an interaction happens ; incremented when a new
    # slide is presented while the previous one received zero events.
    # Surfaced to the engagement scorer to detect "lost the student".
    interactions_on_current_slide: int = 0
    consecutive_passive_slides:    int = 0
    last_presented_slide_key: tuple | None = None
    # Session-level engagement signals
    questions_in_session:         int = 0
    last_interrupt_latency_ms:    float | None = None
    session_started_at:           float = time.time()
    session_lang:  str         = "fr"
    session_level: str         = "lycée"
    in_course:     bool        = False  # étudiant en cours de présentation
    send_lock:     asyncio.Lock = asyncio.Lock()
    audio_stream_task: asyncio.Task | None = None
    current_stream_id: int = 0
    presentation_task: asyncio.Task | None = None
    next_slide_prefetch_task: asyncio.Task | None = None
    current_presentation_key: tuple[str, int, int] | None = None
    current_presentation_text: str = ""
    current_presentation_cursor: int = 0
    current_chapter_title: str = ""
    current_section_title: str = ""
    websocket_closed = False
    student_profile: dict = {}  # ✅ Profil de l'étudiant pour timing adaptatif
    interrupt_audio: bool = False  # 🚨 Flag pour interrompre le TTS en temps réel
    presentation_start_time: float = 0.0  # ✅ NOUVEAU: Timestamp quand présentation commence
    # ⏳ Pause-progress ticker : background task qui logge toutes les 15s
    # combien de temps on attend depuis la pause. Annulé au resume.
    pause_progress_task: asyncio.Task | None = None
    pause_started_at: float = 0.0
    # 🛡️ Anti-echo gates (Option C) :
    #   - tts_started_at : timestamp du 1er chunk TTS envoye au client (reset a la fin du TTS)
    #   - voice_pending_since : pour debouncer les interruptions (require continuous voice)
    #   - tts_pcm_history : ring buffer PCM 16kHz du TTS recent (5s) pour cross-correlation echo
    tts_started_at: float = 0.0
    voice_pending_since: float | None = None
    tts_pcm_history: deque = deque(maxlen=int(Config.ECHO_XCORR_BUFFER_S * Config.SAMPLE_RATE))
    learning_session_db_id: uuid.UUID | None = None
    text_question_task: asyncio.Task | None = None
    active_text_turn_id: int = 0
    text_turn_seq: int = 0

    # 🛡️ VoiceStateMachine — single source of truth voice state (imports au module-level)
    vfsm = VoiceStateMachine(session_id)
    voice_bus = EventBus()
    voice_watchdog_task: asyncio.Task | None = None

    async def _vfsm_cancel_callback() -> None:
        """Callback du watchdog : appelle les cancel functions + orchestrator agentic."""
        try:
            await agentic_orchestrator.cancel(session_id, reason="watchdog")
        except Exception:
            pass
        try:
            await cancel_presentation_task(notify_client=False)
        except Exception:
            pass
        try:
            await cancel_audio_stream(notify_client=False)
        except Exception:
            pass
        try:
            await cancel_text_question_task()
        except Exception:
            pass

    # 🧠 Brancher l'orchestrator au FSM : auto-cancel agentic task sur INTERRUPTED
    vfsm.add_listener(agentic_orchestrator.on_state_change)

    # 🛡️ Demarre le watchdog FSM (auto-recover stuck states) — DOIT etre apres vfsm init
    # Disabled: watchdog loop is not started to avoid automatic force_recover warnings.
    # If you want to re-enable later, restore the asyncio.create_task(...) call below.
    # voice_watchdog_task = asyncio.create_task(
    #     watchdog_loop(vfsm, poll_interval_s=5.0, cancel_callback=_vfsm_cancel_callback)
    # )
    voice_watchdog_task = None
    _deps.active_vfsms[session_id] = vfsm  # registre pour /voice/state/{sid} debug
    await vfsm.transition(VoiceState.LISTENING, reason="ws_connected")

    async def send(data: dict):
        nonlocal websocket_closed
        if websocket_closed:
            return False
        async with send_lock:
            if websocket_closed:
                return False
            try:
                await websocket.send_json(data)
                return True
            except (WebSocketDisconnect, RuntimeError) as exc:
                websocket_closed = True
                log.info(f"[{session_id[:8]}] WebSocket fermé pendant send: {exc}")
                return False

    async def handle_post_response(timeout_sec: float = 2.5) -> bool:
        """
        Après que le LLM a répondu.
        Attire interruption utilisateur (VAD).
        Timeout adaptatif selon le profil étudiant (niveau + performance).
        
        Returns: True si utilisateur a interrompu, False si timeout
        """
        if not ctx:
            return False
        
        try:
            # Per-level interrupt-window defaults. The shorter values for
            # advanced levels reflect the operational expectation that
            # higher-level students process information faster — these are
            # UX defaults, not behavioural claims, and are kept stable
            # rather than multiplied by per-student factors.
            BASE_TIMING = {
                "collège":  5.0,
                "lycée":    4.0,
                "licence":  3.5,
                "master":   2.5,
                "doctorat": 2.0,
            }
            timeout_sec = BASE_TIMING.get(session_level, 3.0)
            # The previous version multiplied this base by
            # ``1 + 0.5 × confusion_count`` and ``1 + 0.3 × asks_repeat``.
            # Those coefficients were arbitrary (why 0.5 vs 0.3 vs 0.1?)
            # and accumulated unboundedly with each repeat / confusion,
            # producing minutes-long waits in pathological sessions. They
            # were removed; reintroducing per-student adaptive timing
            # requires a dedicated UX study.
            log.info(f"⏱️  Wait window | level={session_level} → {timeout_sec:.1f}s")
            
            # ⏸️  Transition vers WAITING (court repos)
            await dialogue.transition(ctx.session_id, DialogState.WAITING)
            await send_state(DialogState.WAITING)
            log.info(f"⏸️  [{session_id[:8]}] Waiting for interruption ({timeout_sec:.1f}s)...")
            
            # ⏱️  Attendre interruption utilisateur (durée timeout)
            start_wait = time.time()
            while time.time() - start_wait < timeout_sec:
                # Vérifier si VAD a détecté la parole
                session = await dialogue.get_session(ctx.session_id)
                if session and session.state == DialogState.LISTENING.value:
                    # ✅ Utilisateur a interrompu!
                    log.info(f"🎤 [{session_id[:8]}] User interrupt detected → LISTENING")
                    return True
                
                await asyncio.sleep(0.05)  # Vérifier tous les 50ms
            
            # ⏭️  Timeout écoulé = pas d'interruption → continuer auto
            log.info(f"▶️  [{session_id[:8]}] No interrupt → auto-advancing")
            return False
            
        except Exception as e:
            log.error(f"❌ Error in handle_post_response: {e}")
            return False

    async def send_state(state: DialogState, substep: str = "", details: dict = None, metrics: dict = None, turn_id: int | None = None):
        """
        ✅ ULTIME v2: État ENRICHI COMPLET avec tous les sous-états et métriques temps réel.
        
        Livre un message formaté parfait avec:
        - État principal (8 états)
        - Sous-étape TRÈS détaillée (16+ substeps)
        - Métriques temps réel complètes
        
        Examples:
            await send_state(DialogState.PROCESSING, "rag_search", {}, {"chunks":5, "elapsed":0.3})
            await send_state(DialogState.PROCESSING, "llm_thinking", {}, {"tokens":145, "llm_time":1.2})
            await send_state(DialogState.PROCESSING, "confusion_detected", {"reason":"prosody_slow_speech"}, {"confidence":0.82})
        """
        
        # ═══════════════════════════════════════════════════════════════════════
        # DICTIONNAIRE COMPLET DES 16+ SOUS-ÉTAPES
        # ═══════════════════════════════════════════════════════════════════════
        substep_full = {
            # Audio Capture
            "stt_language_detection": {"emoji": "🌍", "step": "Language Detection", "desc": "Language identification", "detail": "(FR/EN auto-detect)"},
            "prosody_analysis": {"emoji": "📊", "step": "Prosody Analysis", "desc": "Speech rate, hesitations, intonation", "detail": "(Emotion & confusion detection)"},
            
            # RAG Phase
            "rag_search": {"emoji": "🔍", "step": "Document Search", "desc": "Vector search + BM25", "detail": "(Course documents)"},
            "rag_ranking": {"emoji": "📑", "step": "Ranking Results", "desc": "Sort by relevance", "detail": "(Calculated score)"},
            
            # Confusion Detection
            "confusion_analyzing": {"emoji": "🧠", "step": "Confusion Analysis", "desc": "Comprehension check", "detail": "(8-level detection)"},
            "confusion_detected": {"emoji": "🤔", "step": "CONFUSION DETECTED", "desc": "LLM prompt adaptation", "detail": "(Reformulation needed)"},
            "confusion_none": {"emoji": "✅", "step": "No Confusion", "desc": "Clear question", "detail": "(Standard prompt)"},
            
            # Semantic Checking
            "semantic_check": {"emoji": "🔗", "step": "Semantic Check", "desc": "Historical comparison", "detail": "(OpenAI similarity>85%)"},
            "semantic_repeat": {"emoji": "🔄", "step": "Similar Question", "desc": "Repetition detected", "detail": "(Different angle proposed)"},
            
            # LLM Processing
            "llm_thinking": {"emoji": "🧠", "step": "LLM Thinking", "desc": "Response generation", "detail": "(OpenAI/Mistral in progress)"},
            "llm_generating": {"emoji": "✍️", "step": "Generating Response", "desc": "Text construction", "detail": "(Tokens generated)"},
            
            # Streaming Phase
            "streaming_llm": {"emoji": "📡", "step": "LLM Streaming", "desc": "Direct transmission", "detail": "(Complete response)"},
            "tts_generating": {"emoji": "🎙️", "step": "Audio Generation", "desc": "Text-to-speech conversion", "detail": "(Edge TTS in progress)"},
            "tts_streaming": {"emoji": "📢", "step": "Audio Streaming", "desc": "Audio playback", "detail": "(Direct chunks)"},
            
            # Completion
            "response_complete": {"emoji": "🎉", "step": "Response Complete", "desc": "Sent successfully", "detail": "(Awaiting feedback)"},
            "feedback_listening": {"emoji": "👂", "step": "Voice Feedback", "desc": "Say YES or REPEAT", "detail": "(2-3 seconds)"},
        }
        
        # ═══════════════════════════════════════════════════════════════════════
        # MAIN STATES
        # ═══════════════════════════════════════════════════════════════════════
        state_map = {
            DialogState.IDLE: {"emoji": "🛑", "name": "Idle", "description": "No active session"},
            DialogState.INDEXING: {"emoji": "📚", "name": "Indexing", "description": "Course ingestion in progress..."},
            DialogState.PRESENTING: {"emoji": "🎓", "name": "Presenting", "description": "AI presenting content"},
            DialogState.LISTENING: {"emoji": "👂", "name": "Listening", "description": "AI listening to your question..."},
            DialogState.PROCESSING: {"emoji": "⚙️", "name": "Processing", "description": "Analysis and processing..."},
            DialogState.RESPONDING: {"emoji": "🗣️", "name": "Responding", "description": "AI generating response..."},
            DialogState.WAITING: {"emoji": "⏳", "name": "Waiting", "description": "Awaiting feedback..."},
            DialogState.CLARIFICATION: {"emoji": "❓", "name": "Clarification", "description": "Clarification needed..."},
        }
        
        state_info = state_map.get(state, {})
        emoji = state_info.get("emoji", "❓")
        state_name = state_info.get("name", "Unknown")
        
        # ═══════════════════════════════════════════════════════════════════════
        # BUILDING ENRICHED MESSAGE
        # ═══════════════════════════════════════════════════════════════════════
        msg_lines = [f"{emoji} {state_name}"]
        
        # Line 2: State description
        if state_info.get("description"):
            msg_lines.append(f"   {state_info['description']}")
        
        # Line 3-5: Detailed substep
        # ✅ SHOW SUBSTEPS FOR STREAMING REASONING DISPLAY
        if substep and substep in substep_full:
            sub = substep_full[substep]
            msg_lines.append("")  # Empty line
            msg_lines.append(f"{sub['emoji']} {sub['step']}")
            msg_lines.append(f"   → {sub['desc']}")
            if sub.get('detail'):
                msg_lines.append(f"   💭 {sub['detail']}")
        
        # Additional details
        if details and details.get("reason"):
            msg_lines.append(f"   🎯 Reason: {details['reason'].replace('_', ' ').title()}")

        if details:
            detail_items = [
                ("course_title", "📘 Course"),
                ("chapter_title", "📚 Chapter"),
                ("section_title", "🔖 Section"),
                ("slide_title", "🖼️ Slide"),
                ("question_text", "❓ Question"),
                ("transcription", "🎤 STT"),
                ("engine", "🛠️ Engine"),
                ("voice", "🎙️ Voice"),
                ("tts_engine", "🗣️ TTS Engine"),
                ("tts_voice", "🎙️ TTS Voice"),
                ("answer_preview", "💬 Answer"),
            ]
            for key, label in detail_items:
                value = details.get(key)
                if value is None:
                    continue
                value_text = str(value).strip()
                if not value_text:
                    continue
                if len(value_text) > 160:
                    value_text = value_text[:160].rstrip() + "…"
                msg_lines.append(f"   {label}: {value_text}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # REAL-TIME METRICS SECTION 📊
        # ✅ ONLY SHOW METRICS FOR RESPONDING STATE (end of processing)
        # ═══════════════════════════════════════════════════════════════════════
        if metrics and len(metrics) > 0 and state == DialogState.RESPONDING:
            msg_lines.append("")  # Empty line
            msg_lines.append("📊 Complete Metrics:")
            
            # Duration metrics
            if "elapsed" in metrics:
                msg_lines.append(f"   ⏱️  Elapsed: {metrics['elapsed']:.2f}s")
            if "total_time" in metrics:
                msg_lines.append(f"   ⏱️  Total: {metrics['total_time']:.2f}s")
            
            # Quality metrics
            if "confidence" in metrics:
                conf = metrics["confidence"]
                bar = "▓" * int(conf * 10) + "░" * (10 - int(conf * 10))
                msg_lines.append(f"   🎯 Confidence: {bar} {conf:.0%}")
            
            # STT metrics
            if "speech_rate" in metrics:
                msg_lines.append(f"   🎤 Speech Rate: {metrics['speech_rate']:.0f} wpm")
            if "hesitations" in metrics:
                msg_lines.append(f"   💭 Hesitations: {metrics['hesitations']} found")
            if "language" in metrics:
                msg_lines.append(f"   🌍 Language: {metrics['language'].upper()}")
            
            # RAG metrics
            if "chunks" in metrics:
                msg_lines.append(f"   📚 Documents: {metrics['chunks']} found")
            if "retrieval_time" in metrics:
                msg_lines.append(f"   🔍 Retrieval: {metrics['retrieval_time']:.3f}s")
            if "document_score" in metrics:
                msg_lines.append(f"   🎲 Top Score: {metrics['document_score']:.2f}/1.00")
            
            # Confusion metrics
            if "confusion_confidence" in metrics:
                conf = metrics["confusion_confidence"]
                msg_lines.append(f"   🤔 Confusion Level: {conf:.0%}")
            
            # LLM metrics
            if "tokens" in metrics:
                msg_lines.append(f"   🔤 Tokens: {metrics['tokens']} generated")
            if "sentences" in metrics:
                msg_lines.append(f"   📝 Sentences: {metrics['sentences']}")
            if "words" in metrics:
                msg_lines.append(f"   📰 Words: {metrics['words']}")
            if "llm_time" in metrics:
                msg_lines.append(f"   🧠 LLM Time: {metrics['llm_time']:.2f}s")
            
            # TTS metrics
            if "tts_chunks" in metrics:
                msg_lines.append(f"   🎙️  Audio Chunks: {metrics['tts_chunks']}")
            if "audio_duration" in metrics:
                msg_lines.append(f"   🔉 Audio Duration: {metrics['audio_duration']:.1f}s")
            
            # Progress bar
            if "progress" in metrics:
                prog = metrics["progress"]
                bar = "▓" * int(prog * 12) + "░" * (12 - int(prog * 12))
                msg_lines.append(f"   {bar} {prog:.0%}")
        
        display_message = "\n".join(msg_lines)

        # Attempt to compute TTS params to include in state updates for frontend
        tts_params = {}
        try:
            if state in (DialogState.PROCESSING, DialogState.RESPONDING, DialogState.PRESENTING):
                confusion_score = 0.0
                if metrics and isinstance(metrics, dict) and metrics.get("confusion_confidence") is not None:
                    confusion_score = float(metrics.get("confusion_confidence", 0.0))
                elif details and isinstance(details, dict) and details.get("confusion_confidence") is not None:
                    confusion_score = float(details.get("confusion_confidence", 0.0))
                # compute_tts_params may perform I/O (redis) — await safely
                try:
                    tts_params = await compute_tts_params(session_id, base_rate=1.0, confusion_score=confusion_score)
                except Exception:
                    tts_params = {}
        except Exception:
            tts_params = {}

        try:
            record_trace_event({
                "session_id": session_id,
                "state": state.value,
                "state_name": state_name,
                "substep": substep,
                "display_message": display_message,
                "details": details or {},
                "metrics": metrics or {},
                "emoji": emoji,
                **({"turn_id": turn_id} if turn_id is not None else {}),
            })
        except Exception as trace_exc:
            log.debug(f"[{session_id[:8]}] trace event skipped: {trace_exc}")
        
        return await send({
            "type": "state_change",
            "state": state.value,
            "display_message": display_message,
            "substep": substep,
            "emoji": emoji,
            "state_name": state_name,
            "details": details or {},
            "metrics": metrics or {},
            "tts_params": tts_params,
            "substep_full": substep_full,
            "timestamp": time.time(),
            **({"turn_id": turn_id} if turn_id is not None else {}),
        })

    async def cancel_text_question_task() -> None:
        nonlocal text_question_task, active_text_turn_id
        if text_question_task and not text_question_task.done():
            task = text_question_task
            task.cancel()

            def _log_done(done_task: asyncio.Task) -> None:
                try:
                    done_task.result()
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    log.debug(f"[{session_id[:8]}] text question task cancel error: {exc}")

            task.add_done_callback(_log_done)
        text_question_task = None
        active_text_turn_id = 0

    async def process_text_question_turn(
        *,
        turn_id: int,
        content: str,
        lang: str,
        subj: str,
        chunks_with_scores: list,
        rag_time: float,
        avg_score: float,
        is_confused: bool,
        confusion_reason: str,
        question_for_llm: str,
        llm_start: float,
        question_ctx: SessionContext | None,
        presentation_cursor: int,
        current_chapter_title: str,
        current_section_title: str,
        current_chapter_idx: int | None,
        section_index_int: int | None,
    ) -> None:
        nonlocal text_question_task, tts_started_at

        log.info(
            f"[{session_id[:8]}] 📥 process_text_question_turn START "
            f"turn={turn_id} active={active_text_turn_id} "
            f"q={content[:60]!r}"
        )

        try:
            if turn_id != active_text_turn_id:
                log.warning(
                    f"[{session_id[:8]}] ⚠️ EARLY EXIT (entry guard): "
                    f"turn_id={turn_id} != active_text_turn_id={active_text_turn_id}"
                )
                return

            # ✅ STREAMING LLM WITH REAL-TIME METRICS
            ai_response = ""
            llm_confidence = 0.7
            chunk_count = 0
            tokens_generated = 0

            # ── Q&A response cache lookup ──────────────────────────
            # Hash the (question, slide, course, lang) and check Redis.
            # On hit, set ai_response — the LLM pipeline below detects
            # this via `if ai_response:` skip. Saves 3-10 minutes on
            # CPU when the same question is repeated.
            _slide_for_cache = (
                question_ctx.paused_state.get("slide_content", "") if question_ctx else ""
            )
            _course_for_cache = (question_ctx.course_id if question_ctx else "") or ""
            # Hoisted out of the try block below so they're defined on the
            # cache-hit path too — the cache-hit branch raises _SkipLLMPipeline
            # before the original assignments at line 672-673 ran, leaving
            # both variables unbound when downstream code at line ~921
            # (history persistence) tried to read them. UnboundLocalError.
            _qa_course_id = _course_for_cache
            _qa_student_id = (question_ctx.student_id if question_ctx and getattr(question_ctx, 'student_id', None) else "")
            qa_cache_hit = False
            try:
                from cache.qa_cache import get_qa_response
                _cached = await get_qa_response(
                    question=content,
                    slide_content=_slide_for_cache,
                    course_id=_course_for_cache,
                    language=lang,
                )
                if _cached and _cached.get("answer"):
                    ai_response = _cached["answer"]
                    llm_confidence = float(
                        (_cached.get("meta") or {}).get("confidence", 0.7)
                    )
                    qa_cache_hit = True
                    log.info(
                        f"[{session_id[:8]}] ⚡ Q&A cache HIT — "
                        f"skipping LLM pipeline ({len(ai_response)} chars)"
                    )
            except Exception as _cache_exc:
                log.debug(f"[{session_id[:8]}] qa_cache lookup skipped: {_cache_exc}")

            # ── Q&A Graph (LangGraph: Intent → [Rewriter → Retriever] → Responder) ──
            qa_final: dict = {
                "answer": ai_response,
                "confidence": llm_confidence,
                "intent": None,
                "timings": {},
                "actions": [],
                "citations": [],
            }
            try:
                if qa_cache_hit:
                    # Cache hit — bypass the LLM pipeline entirely. The
                    # synthesised qa_final above keeps downstream
                    # telemetry/citations code working with sane empty
                    # defaults.
                    log.info(f"[{session_id[:8]}] ⚡ skipping Q&A graph (cache hit)")
                    raise _SkipLLMPipeline()
                # ✨ Enrich slide_content with vision description (cached on disk per slide).
                # Without this, the LLM only sees OCR text — never the scatter plots,
                # schemas, hand-drawn formulas that make many slides actually meaningful.
                _slide_text_raw = question_ctx.paused_state.get("slide_content", "") if question_ctx else ""
                _slide_image = question_ctx.paused_state.get("slide_path", "") if question_ctx else ""
                _last_slide_content = _slide_text_raw
                if _slide_image:
                    try:
                        from services.vision_describe import describe_slide_image, merge_into_slide_content
                        _vision_desc = await describe_slide_image(
                            _slide_image,
                            (lang or "en")[:2],
                            slide_text=_slide_text_raw,
                        )
                        if _vision_desc:
                            _last_slide_content = merge_into_slide_content(
                                _slide_text_raw, _vision_desc, (lang or "en")[:2],
                            )
                    except Exception as _exc:
                        log.debug(f"[{session_id[:8]}] vision describe skipped: {_exc}")

                # QA Graph orchestration — delegated to the shared runner
                # so audio questions go through the exact same pipeline.
                # See agentic/qa/runner.py for the full sequence (loads
                # persistent history + snapshot, merges with in-memory
                # history, builds qa_state, streams the graph, extracts
                # the cleaned answer).

                # Hook: push intent to the UI as soon as IntentAgent
                # classifies — preserves the old astream-loop behaviour.
                async def _push_intent_to_ui(intent_obj):
                    try:
                        await send({
                            "type": "qa_intent",
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "intent_type": getattr(intent_obj, "type", "question"),
                            "confidence": float(getattr(intent_obj, "confidence", 0.0)),
                            "source": (getattr(intent_obj, "payload", {}) or {}).get("source", ""),
                        })
                        log.info(
                            f"[{session_id[:8]}] 🎯 Intent envoyé au frontend: "
                            f"{getattr(intent_obj, 'type', '?')} "
                            f"(conf={getattr(intent_obj, 'confidence', 0):.2f})"
                        )
                    except Exception as exc:                              # noqa: BLE001
                        log.debug(f"qa_intent send failed: {exc}")

                # Engagement computation stays here — needs session-level
                # counters (questions_in_session, etc.) the runner doesn't have.
                engagement_dict = None
                try:
                    from pedagogy.engagement import EngagementSignals, compute_engagement
                    _now = time.time()
                    _eng = compute_engagement(EngagementSignals(
                        seconds_since_last_interaction=0.0,
                        questions_in_session=questions_in_session,
                        confusions_in_session=(question_ctx.confusion_count if question_ctx else 0),
                        consecutive_passive_slides=consecutive_passive_slides,
                        last_interrupt_latency_ms=last_interrupt_latency_ms,
                        session_age_s=max(0.0, _now - session_started_at),
                    ))
                    engagement_dict = {"score": _eng.score, "label": _eng.label}
                    log.info(f"[{session_id[:8]}] 💡 engagement | {_eng.reason}")
                except Exception as _eng_exc:                             # noqa: BLE001
                    log.debug(f"[{session_id[:8]}] engagement compute skipped: {_eng_exc}")

                from agentic.qa.runner import run_qa_graph
                qa_result = await run_qa_graph(
                    text=question_for_llm or content,
                    session_id=session_id,
                    course_id=_qa_course_id,
                    student_id=_qa_student_id or None,
                    language=lang,
                    chapter_idx=current_chapter_idx,
                    chapter_title=current_chapter_title or "",
                    section_idx=section_index_int,
                    section_title=current_section_title or "",
                    last_slide_content=_last_slide_content,
                    history=history,
                    student_level=(question_ctx.student_level if question_ctx else "lycée"),
                    qa_graph=qa_graph,
                    brain=brain,
                    engagement=engagement_dict,
                    on_intent_classified=_push_intent_to_ui,
                )
                ai_response = qa_result["answer"]
                llm_confidence = qa_result["confidence"]
                qa_final = qa_result["qa_final"]
                qa_intent = qa_result["intent"]
                qa_timings = qa_result["timings"]
                log.info(
                    f"[{session_id[:8]}] 🧠 Q&A Graph done | "
                    f"intent={getattr(qa_intent, 'type', '?') if qa_intent else '?'} | "
                    f"timings={ {k: round(v, 1) for k, v in qa_timings.items()} } | "
                    f"{len(ai_response)} chars | turn={turn_id} active={active_text_turn_id}"
                )
                if not ai_response:
                    raise RuntimeError("Q&A Graph returned empty answer")
            except _SkipLLMPipeline:
                # Cache-hit short-circuit. ai_response is already set,
                # qa_final is already a sane stub. Just continue to TTS.
                pass
            except Exception as llm_exc:
                log.warning(f"[{session_id[:8]}] ⚠️ Q&A Graph failed ({type(llm_exc).__name__}): {llm_exc} → fallback brain.ask_stream")
                # Streamed fallback. We iterate Brain.ask_stream and forward
                # each chunk to the UI as ``answer_text_partial`` so the
                # student sees the answer materialise instead of waiting for
                # a single 60s TTFB on Ollama. Cleaning + TTS still happens
                # on the full accumulated text.
                try:
                    accumulated_chunks: list[str] = []
                    chunk_count = 0
                    async for chunk in brain.ask_stream(
                        question=question_for_llm,
                        reply_language=lang,
                        session_id=session_id,
                    ):
                        if turn_id != active_text_turn_id:
                            # Turn cancelled mid-stream — stop forwarding
                            # tokens. The async generator's ``finally``
                            # will still update history with whatever
                            # arrived. We simply break out.
                            log.info(
                                f"[{session_id[:8]}] ⏹️ stream cancelled "
                                f"(turn_id={turn_id}, active={active_text_turn_id})"
                            )
                            break
                        if not chunk:
                            continue
                        accumulated_chunks.append(chunk)
                        chunk_count += 1
                        try:
                            await send({
                                "type": "answer_text_partial",
                                "text": chunk,
                                "turn_id": turn_id,
                                "subject": "qa",
                            })
                        except Exception as send_exc:
                            log.debug(f"[{session_id[:8]}] partial send skipped: {send_exc}")
                            break
                    raw_streamed = "".join(accumulated_chunks)
                    ai_response = brain._clean_for_speech(raw_streamed)
                    ai_response = brain._dedupe_answer_text(ai_response)
                    llm_confidence = 0.5
                    log.info(
                        f"[{session_id[:8]}] ✅ stream fallback done — "
                        f"{chunk_count} chunks, {len(ai_response)} chars"
                    )
                except Exception as fb_exc:
                    log.error(f"[{session_id[:8]}] ❌ Fallback stream failed: {fb_exc}")
                    ai_response = "Je suis désolé, j'ai rencontré une erreur technique. Veuillez réessayer."
                    llm_confidence = 0.0

            llm_time = time.time() - llm_start

            if turn_id != active_text_turn_id:
                # Diagnostic: surface why the response is being dropped.
                # Was silent before — operator couldn't tell that a 90s
                # LLM call's result got discarded.
                log.warning(
                    f"[{session_id[:8]}] ⚠️ ANSWER DROPPED: "
                    f"turn_id={turn_id} != active_text_turn_id={active_text_turn_id} "
                    f"({len(ai_response)} chars lost). "
                    f"Cause: a new question/event reset active_text_turn_id "
                    f"during the {llm_time:.1f}s LLM call."
                )
                return

            history.append({"role": "user", "content": content})
            history.append({"role": "assistant", "content": ai_response})

            # Also persist to Redis-backed per-(student, course) history
            # so the next session / device / reload picks up where we
            # left off. Best-effort : a Redis hiccup must NOT block the
            # already-delivered answer. Anonymous students (no
            # student_id) fall back to the in-memory ``history`` only —
            # which is exactly the legacy behaviour, so nothing
            # regresses.
            if _qa_student_id and _qa_course_id:
                try:
                    from pedagogy.student_history import append_chat_turn
                    await append_chat_turn(_qa_student_id, _qa_course_id, "user", content)
                    await append_chat_turn(_qa_student_id, _qa_course_id, "assistant", ai_response)
                    log.info(
                        f"[{session_id[:8]}] 💾 chat history persisted to Redis "
                        f"(+1 user, +1 assistant turn)"
                    )
                except Exception as exc:                                  # noqa: BLE001
                    log.debug(f"[{session_id[:8]}] persistent history append skipped: {exc}")

            # ✅ UPDATE STATE: TTS Generation
            num_chunks = max(1, len(ai_response) // 200)
            await send_state(
                DialogState.PROCESSING,
                "tts_text_chunking",
                {"chunks_total": num_chunks},
                {"progress_pct": 78},
                turn_id=turn_id,
            )

            # ✨ Adaptive TTS rate : profile preference × bandit-chosen modulation.
            # The personalization bandit (Phase 1) selects an action whose
            # ``speech_rate`` ∈ {slow, normal, fast} is exposed via the QA
            # graph state in actions[0].payload["bandit_speech_rate"]. We
            # combine it with the user's stored speech_rate preference :
            # final = profile_rate × bandit_multiplier.
            bandit_speech_rate: str | None = None
            try:
                qa_actions = qa_final.get("actions") or []
                if qa_actions:
                    payload = getattr(qa_actions[0], "payload", None) or {}
                    if isinstance(payload, dict):
                        bandit_speech_rate = payload.get("bandit_speech_rate")
            except Exception as exc:                                      # noqa: BLE001
                log.debug(f"[{session_id[:8]}] bandit speech_rate extraction skipped: {exc}")

            # Pull the recent fused confusion score from the session
            # context — written by detect_and_track_confusion. When
            # high (≥ 0.6), the TTS adapter applies a 0.9× slowdown
            # so the next narration is more digestible. Resets
            # automatically once consecutive_clean_turns climbs back.
            _last_conf_score: float | None = None
            try:
                _ctx_for_conf = await dialogue._load(session_id)
                if _ctx_for_conf:
                    _last_conf_score = float(getattr(_ctx_for_conf, "last_confusion_score", 0.0) or 0.0)
                    # Reset the slowdown after 3 consecutive clean turns —
                    # the student has caught up.
                    if int(getattr(_ctx_for_conf, "consecutive_clean_turns", 0) or 0) >= 3:
                        _last_conf_score = 0.0
            except Exception as _conf_exc:                                  # noqa: BLE001
                log.debug(f"[{session_id[:8]}] last_confusion_score read skipped: {_conf_exc}")

            from pedagogy.personalization.tts_adapter import get_edge_tts_rate_with_bandit
            tts_rate = await get_edge_tts_rate_with_bandit(
                session_id,
                bandit_speech_rate=bandit_speech_rate,
                confusion_score=_last_conf_score,
            )
            if _last_conf_score and _last_conf_score >= 0.6:
                log.info(
                    f"[{session_id[:8]}] 🐢 TTS slowdown active "
                    f"(confusion_score={_last_conf_score:.2f}) → rate={tts_rate}"
                )
            audio_bytes, tts_time, tts_engine, tts_voice, mime = await voice.generate_audio_async(
                ai_response,
                language_code=lang,
                rate=tts_rate,
            )

            await send_state(
                DialogState.PROCESSING,
                "tts_generation",
                {"engine": tts_engine, "voice": tts_voice},
                {
                    "audio_bytes": len(audio_bytes) if audio_bytes else 0,
                    "duration_ms": round(tts_time * 1000, 1),
                    "progress_pct": 90,
                },
                turn_id=turn_id,
            )

            if turn_id != active_text_turn_id:
                log.warning(
                    f"[{session_id[:8]}] ⚠️ ANSWER DROPPED post-TTS: "
                    f"turn_id={turn_id} != active_text_turn_id={active_text_turn_id} "
                    f"({len(ai_response)} chars + audio lost)"
                )
                return

            if question_ctx:
                try:
                    await dialogue.transition(question_ctx.session_id, DialogState.RESPONDING)
                except ValueError:
                    return
            response_metrics = {
                "retrieval_time": round(rag_time / 1000.0, 3),
                "chunks": len(chunks_with_scores),
                "document_score": round(avg_score, 2),
                "llm_time": round(llm_time, 2),
                "tts_time": round(tts_time, 2),
                "total_time": round((rag_time / 1000.0) + llm_time + tts_time, 2),
                "tokens": tokens_generated,
                "words": len(ai_response.split()),
                "sentences": chunk_count,
                "confidence": round(llm_confidence, 2),
                "progress_pct": 95,
            }
            await send_state(
                DialogState.RESPONDING,
                "",
                {
                    "question_text": content,
                    "answer_preview": ai_response[:160],
                    "subject": subj,
                    "slide_title": current_section_title or current_chapter_title or "",
                    "chapter_title": current_chapter_title or "",
                    "section_title": current_section_title or "",
                    "tts_engine": tts_engine,
                    "tts_voice": tts_voice,
                },
                response_metrics,
                turn_id=turn_id,
            )

            log.info(f"[{session_id[:8]}] 📤 Envoi answer_text: {len(ai_response)} chars | subj={subj}")
            await send({
                "type": "answer_text",
                "text": ai_response,
                "subject": subj,
                "rag_chunks": len(chunks_with_scores),
                "turn_id": turn_id,
            })
            log.info(f"[{session_id[:8]}] ✅ answer_text envoyé")

            # ── Navigation dispatch ──────────────────────────────────
            # When the IntentAgent classified this turn as ``navigation``
            # the responder emitted ``Action(type='navigate', payload={
            # nav_action, nav_target})`` instead of the legacy generic
            # ``replay_concept`` placeholder. Dispatch the sub-action so
            # the student's "passe à la suivante" / "ralentis" / "va sur
            # K-means" actually moves the slide / changes the rate /
            # jumps to the concept (the spoken acknowledgement is
            # already on its way to TTS — they play in parallel with the
            # slide transition).
            try:
                _qa_actions = qa_final.get("actions") or []
                _first_action = _qa_actions[0] if _qa_actions else None
                _atype = getattr(_first_action, "type", "")
                if _atype == "navigate":
                    _apayload = getattr(_first_action, "payload", None) or {}
                    _nav_action = str(_apayload.get("nav_action", "") or "").strip().lower()
                    _nav_target = str(_apayload.get("nav_target", "") or "").strip()
                    # Look up the current speech_rate so slow_down has a
                    # baseline to multiply (defaults to 1.0 if profile
                    # fetch fails — same default as profile.py).
                    _current_rate = 1.0
                    try:
                        from pedagogy.personalization.profile import (
                            get_or_create_profile, update_profile,
                        )
                        # Pass session defaults so the FIRST creation uses
                        # the authenticated student's preferences instead
                        # of the hardcoded "fr / lycée" of the wrapper.
                        _prof = await get_or_create_profile(
                            session_id,
                            defaults={"language": session_lang, "level": session_level},
                            course_id=(question_ctx.course_id if question_ctx else None),
                        )
                        _current_rate = float(_prof.get("speech_rate", 1.0) or 1.0)
                    except Exception as _prof_exc:                          # noqa: BLE001
                        log.debug(f"[{session_id[:8]}] nav: profile fetch skipped: {_prof_exc}")
                        update_profile = None  # type: ignore[assignment]

                    _nav_result = await _dispatch_nav_action(
                        nav_action=_nav_action,
                        nav_target=_nav_target,
                        ctx=question_ctx or ctx,
                        dialogue=dialogue,
                        send=send,
                        turn_id=turn_id,
                        slide_loader=load_course_slide_context,
                        kg_getter=lambda: __import__(
                            "pedagogy.knowledge_graph", fromlist=["get_or_build"],
                        ).get_or_build(rag),
                        profile_updater=update_profile,
                        current_speech_rate=_current_rate,
                    )
                    log.info(
                        f"[{session_id[:8]}] 🧭 nav dispatch: "
                        f"action={_nav_result.get('action')!r} "
                        f"ok={_nav_result.get('ok')} | {_nav_result.get('detail')}"
                    )
            except Exception as _nav_exc:                                   # noqa: BLE001
                log.warning(
                    f"[{session_id[:8]}] nav dispatch failed: "
                    f"{type(_nav_exc).__name__}: {_nav_exc}"
                )

            # ── Cache the answer (only on fresh generation, not on
            # cache hits — re-storing a cache hit would just refresh
            # the TTL on its own key, no point).
            if not qa_cache_hit and ai_response:
                try:
                    from cache.qa_cache import set_qa_response
                    await set_qa_response(
                        question=content,
                        slide_content=_slide_for_cache,
                        answer=ai_response,
                        course_id=_course_for_cache,
                        language=lang,
                        citations=qa_final.get("citations") or [],
                        meta={"confidence": llm_confidence, "subject": subj},
                    )
                except Exception as _cache_set_exc:
                    log.debug(
                        f"[{session_id[:8]}] qa_cache write skipped: {_cache_set_exc}"
                    )

            if audio_bytes:
                # 🛡️ Mark TTS streaming start (echo suppression gate)
                if tts_started_at == 0.0:
                    tts_started_at = time.time()
                # 🛡️ Append PCM au ring buffer pour echo cross-correlation
                if Config.ENABLE_ECHO_XCORR:
                    pcm = _tts_bytes_to_pcm(audio_bytes, Config.SAMPLE_RATE)
                    if pcm is not None and pcm.size > 0:
                        tts_pcm_history.extend(pcm.tolist())
                await send({
                    "type": "audio_chunk",
                    "data": base64.b64encode(audio_bytes).decode(),
                    "mime": mime,
                    "final": True,
                    "turn_id": turn_id,
                })

            media_stamp = int(time.time() * 1000)
            transcript_payload = {
                "kind": "text_question",
                "session_id": session_id,
                "turn_id": turn_id,
                "question_text": content,
                "answer_text": ai_response,
                "language": lang,
                "subject": subj,
                "chapter_title": current_chapter_title or "",
                "section_title": current_section_title or "",
                "chapter_index": current_chapter_idx,
                "section_index": section_index_int,
                "char_position": presentation_cursor,
                "audio_answer_path": f"turns/answers/{session_id}/{turn_id}_{media_stamp}.{'mp3' if audio_bytes and 'mpeg' in (mime or '') else 'webm'}" if audio_bytes else "",
            }
            await save_media_json(f"turns/transcripts/{session_id}/{turn_id}_{media_stamp}.json", transcript_payload)
            if audio_bytes:
                answer_audio_object = transcript_payload["audio_answer_path"]
                await save_media_bytes(answer_audio_object, audio_bytes, mime or "audio/mpeg")

            await send_state(
                DialogState.RESPONDING,
                "response_complete",
                {
                    "question_text": content,
                    "answer_preview": ai_response[:160],
                    "subject": subj,
                    "slide_title": current_section_title or current_chapter_title or "",
                    "chapter_title": current_chapter_title or "",
                    "section_title": current_section_title or "",
                    "tts_engine": tts_engine,
                    "tts_voice": tts_voice,
                },
                response_metrics,
                turn_id=turn_id,
            )

            if question_ctx:
                try:
                    await dialogue.transition(question_ctx.session_id, DialogState.LISTENING)
                except ValueError:
                    return
            await send_state(DialogState.LISTENING)

            if turn_id != active_text_turn_id:
                return

            # ── Analytics & Recherche ─────────────────────────────────
            try:
                transcript_searcher.index_interaction(
                    session_id=session_id,
                    student_q=content,
                    teacher_a=ai_response,
                    language=lang,
                    course_id="",
                    subject=subj,
                )
                analytics_engine.record_interaction(
                    session_id=session_id,
                    question=content,
                    answer=ai_response,
                    stt_time=0,
                    llm_time=llm_time,
                    tts_time=tts_time,
                    language=lang,
                    subject=subj,
                )
                try:
                    text_profile = await profile_mgr.update_from_interaction(
                        session_id,
                        "qa",
                        topic=(current_section_title or current_chapter_title or subj or ""),
                        confused=is_confused,
                        response_time=rag_time / 1000.0 + llm_time + tts_time,
                        # Real grounding-derived confidence from the
                        # responder (avg citation score, or 0.0 when the
                        # answer wasn't grounded). Replaces the previous
                        # synthesised 0.85/0.35 pair which was just a
                        # restatement of the ``confused`` flag.
                        confidence=llm_confidence,
                        # Bernoulli reward for Thompson sampling: 1 if the
                        # turn was clean, 0 if confusion was detected.
                        reward=1.0 if not is_confused else 0.0,
                        action_taken="reformulate" if is_confused else "answer",
                        course_id=(question_ctx.course_id if question_ctx else None),
                    )
                    if text_profile:
                        student_profile.update(text_profile.to_dict())

                    # Persist the turn with bandit annotations so offline RL
                    # training can rebuild the (state, action, reward) tuples
                    # from Postgres later (see pedagogy.personalization.bandit
                    # .log_extractor).
                    from pedagogy.personalization.bandit.log_extractor import (
                        extra_payload_from_state,
                    )
                    await persist_learning_turn(
                        event_type="qa",
                        question_text=content,
                        answer_text=ai_response,
                        language=lang,
                        subject=subj or current_section_title or current_chapter_title or "",
                        course_id_value=course_id or (question_ctx.course_id if question_ctx and question_ctx.course_id else None),
                        confusion_detected=is_confused,
                        confusion_reason=confusion_reason,
                        action_taken="reformulate" if is_confused else "answer",
                        reward=1.0 if not is_confused else 0.0,
                        stt_time=0.0,
                        llm_time=llm_time,
                        tts_time=tts_time,
                        total_time=rag_time / 1000.0 + llm_time + tts_time,
                        profile_snapshot=text_profile.to_dict() if text_profile else {},
                        concept=current_section_title or current_chapter_title or subj or "",
                        chapter_index=current_chapter_idx,
                        section_index=section_index_int,
                        char_position=presentation_cursor,
                        extra_payload={
                            "source": "text",
                            "rag_chunks": len(chunks_with_scores),
                            "confidence": llm_confidence,
                            **extra_payload_from_state(qa_final),
                        },
                    )
                    # ✨ Bayes posterior update — text question signals reading/visual preference
                    from pedagogy.personalization.learning_style.bayes import fire_signal
                    fire_signal(ctx.student_id if ctx else None, "text_question")
                except Exception as learning_exc:
                    log.debug(f"[{session_id[:8]}] Text learning log skipped: {learning_exc}")

                record_session_event({
                    "session_id": session_id,
                    "language": lang,
                    "question": content,
                    "stt_text": content,
                    "answer": ai_response,
                    "turn_id": turn_id,
                    "stt_time": 0.0,
                    "llm_time": round(llm_time, 2),
                    "tts_time": round(tts_time, 2),
                    "total_time": round(llm_time + tts_time, 2),
                    "meets_kpi": (llm_time + tts_time) < 5.0,
                    "subject": subj,
                    "confusion": is_confused,
                    "confusion_reason": confusion_reason,
                    "source": "text",
                    "slide_title": current_section_title or current_chapter_title or "",
                    "chapter_title": current_chapter_title or "",
                    "section_title": current_section_title or "",
                    "tts_engine": tts_engine,
                    "tts_voice": tts_voice,
                    "chapter_index": current_chapter_idx,
                    "section_index": section_index_int,
                    "char_position": presentation_cursor,
                })
            except Exception as _ae:
                log.debug("analytics error: %s", _ae)
        except asyncio.CancelledError:
            # Surface cancellations: when text_question_task is cancelled
            # (e.g. by another interrupt) mid-flight, the LLM call had
            # already started and may have produced a response that's
            # now being thrown away. Was previously silent.
            log.warning(
                f"[{session_id[:8]}] ⚠️ process_text_question_turn CANCELLED "
                f"(turn={turn_id}). The pending LLM/TTS work is discarded."
            )
            raise
        except Exception as exc:
            log.error(
                f"[{session_id[:8]}] ❌ process_text_question_turn EXCEPTION "
                f"(turn={turn_id}, type={type(exc).__name__}): {exc}",
                exc_info=True,
            )
        finally:
            if asyncio.current_task() is text_question_task:
                text_question_task = None

    async def process_quiz_request(
        *,
        quiz_topic: str,
        lang: str,
        subj: str,
        chunks_with_scores: list,
        question_ctx: SessionContext | None,
        presentation_cursor: int,
        current_chapter_title: str,
        current_section_title: str,
        current_chapter_idx: int | None,
        section_index_int: int | None,
        course_id: str,
        course_title: str,
        course_domain: str,
        slide_path: str,
    ) -> None:
        quiz_topic = (quiz_topic or current_section_title or current_chapter_title or course_title or "Quiz").strip()
        try:
            await cancel_text_question_task()
            await cancel_audio_stream(notify_client=False)
            await cancel_presentation_task(notify_client=False)

            if question_ctx:
                try:
                    await dialogue.transition(question_ctx.session_id, DialogState.PROCESSING)
                except ValueError:
                    pass

            await send_state(
                DialogState.PROCESSING,
                "quiz_preparing",
                {
                    "type": "quiz",
                    "quiz_topic": quiz_topic,
                    "chapter_title": current_chapter_title or "",
                    "section_title": current_section_title or "",
                },
                {
                    "progress_pct": 25,
                    "quiz_topic": quiz_topic,
                },
            )

            log.info(
                "🔍 quiz GENERATE | topic=%r lang=%s level=%s chunks=%d ch=%s sec=%s",
                quiz_topic[:80], lang, session_level,
                len(chunks_with_scores or []),
                current_chapter_title or "?",
                current_section_title or "?",
            )
            quiz_llm_start = time.time()
            quiz_payload, quiz_confidence = await asyncio.to_thread(
                rag.generate_quiz,
                chunks_with_scores,
                question=quiz_topic,
                history=[],
                language=lang,
                student_level=session_level,
                current_chapter_title=current_chapter_title,
                current_section_title=current_section_title,
                question_count=3,
            )
            quiz_llm_time = time.time() - quiz_llm_start
            log.info(
                "🔍 quiz RESULT | %d questions | confidence=%.2f | %.1fs LLM",
                len((quiz_payload or {}).get("questions", [])),
                float(quiz_confidence or 0.0), quiz_llm_time,
            )

            if not isinstance(quiz_payload, dict):
                quiz_payload = {}

            quiz_payload.setdefault("title", "Quiz rapide")
            quiz_payload.setdefault("topic", quiz_topic)
            quiz_payload.setdefault("difficulty", session_level)
            quiz_payload.setdefault("language", lang)
            quiz_payload.setdefault("chapter_title", current_chapter_title or "")
            quiz_payload.setdefault("section_title", current_section_title or "")
            quiz_payload.setdefault("course_title", course_title or "")
            quiz_payload.setdefault("course_domain", course_domain or "")
            quiz_payload.setdefault("slide_title", current_section_title or current_chapter_title or "")
            quiz_payload.setdefault("slide_path", slide_path or "")

            questions = quiz_payload.get("questions") if isinstance(quiz_payload.get("questions"), list) else []
            quiz_payload["question_count"] = len(questions)
            quiz_payload["confidence"] = round(float(quiz_confidence or 0.0), 3)

            await send({
                "type": "quiz_prompt",
                "question": quiz_payload.get("title") or quiz_payload.get("topic") or quiz_topic,
                "quiz": quiz_payload,
                "chapter_title": current_chapter_title or "",
                "section_title": current_section_title or "",
                "course_id": course_id or "",
                "course_title": course_title or "",
                "course_domain": course_domain or "",
                "language": lang,
                "level": session_level,
                "confidence": quiz_payload["confidence"],
            })

            if question_ctx:
                try:
                    await dialogue.transition(question_ctx.session_id, DialogState.LISTENING)
                except ValueError:
                    return

            await send_state(
                DialogState.LISTENING,
                details={
                    "quiz_topic": quiz_payload.get("topic") or quiz_topic,
                    "quiz_questions": quiz_payload.get("question_count", 0),
                    "chapter_title": current_chapter_title or "",
                    "section_title": current_section_title or "",
                },
            )

            try:
                quiz_profile = await profile_mgr.update_from_interaction(
                    session_id,
                    "quiz",
                    topic=quiz_payload.get("topic") or current_section_title or current_chapter_title or subj or "",
                    confused=False,
                    response_time=quiz_llm_time,
                    confidence=None,
                    reward=0.0,
                    action_taken="quiz",
                    course_id=course_id or (question_ctx.course_id if question_ctx and question_ctx.course_id else None),
                )
                if quiz_profile:
                    student_profile.update(quiz_profile.to_dict())

                await persist_learning_turn(
                    event_type="quiz",
                    question_text=quiz_topic,
                    answer_text=f"{quiz_payload.get('title', 'Quiz rapide')} ({quiz_payload.get('question_count', 0)} questions)",
                    language=lang,
                    subject=subj or current_section_title or current_chapter_title or "",
                    course_id_value=course_id or (question_ctx.course_id if question_ctx and question_ctx.course_id else None),
                    confusion_detected=False,
                    action_taken="quiz",
                    reward=0.0,
                    stt_time=0.0,
                    llm_time=quiz_llm_time,
                    tts_time=0.0,
                    total_time=quiz_llm_time,
                    profile_snapshot=quiz_profile.to_dict() if quiz_profile else {},
                    concept=current_section_title or current_chapter_title or subj or "",
                    chapter_index=current_chapter_idx,
                    section_index=section_index_int,
                    char_position=presentation_cursor,
                    extra_payload={
                        "source": "quiz",
                        "quiz": quiz_payload,
                        "quiz_confidence": round(float(quiz_confidence or 0.0), 3),
                    },
                    session_state=DialogState.LISTENING.value,
                )
            except Exception as quiz_log_exc:
                log.debug(f"[{session_id[:8]}] Quiz learning log skipped: {quiz_log_exc}")

            record_session_event({
                "session_id": session_id,
                "language": lang,
                "question": quiz_topic,
                "stt_text": quiz_topic,
                "answer": quiz_payload.get("title", "Quiz rapide"),
                "turn_id": None,
                "stt_time": 0.0,
                "llm_time": round(quiz_llm_time, 2),
                "tts_time": 0.0,
                "total_time": round(quiz_llm_time, 2),
                "meets_kpi": quiz_llm_time < 5.0,
                "subject": subj,
                "confusion": False,
                "confusion_reason": "",
                "source": "quiz",
                "slide_title": current_section_title or current_chapter_title or "",
                "chapter_title": current_chapter_title or "",
                "section_title": current_section_title or "",
                "chapter_index": current_chapter_idx,
                "section_index": section_index_int,
                "char_position": presentation_cursor,
                "quiz_questions": quiz_payload.get("question_count", 0),
            })

            try:
                analytics_engine.record_section(
                    session_id,
                    course_id or "",
                    current_chapter_idx or 0,
                    section_index_int or 0,
                    event_type="quiz",
                    language=lang,
                )
            except Exception as analytics_exc:
                log.debug(f"[{session_id[:8]}] Quiz analytics skipped: {analytics_exc}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(f"Erreur quiz: {e}")
            await send({"type": "error", "message": f"Erreur quiz: {str(e)}"})

    async def set_listening_state() -> None:
        """
        Transition SÉCURISÉE vers LISTENING avec gestion de la machine d'état.
        
        ✅ Si en LISTENING: pas de transition nécessaire
        ✅ Si en PROCESSING: passer par CLARIFICATION d'abord (transition valide)
        ✅ Si dans autre état: transition directe
        """
        nonlocal ctx
        
        if not ctx:
            return
        
        current_state = ctx.state
        
        # ✅ Déjà en LISTENING: rien à faire
        if current_state == DialogState.LISTENING.value:
            return
        
        # ✅ Si en PROCESSING: passer par CLARIFICATION (transition valide: PROCESSING → CLARIFICATION)
        if current_state == DialogState.PROCESSING.value:
            log.info(f"[{session_id[:8]}] 🔄 PROCESSING → CLARIFICATION → LISTENING (transition sécurisée)")
            await dialogue.transition(ctx.session_id, DialogState.CLARIFICATION)
            await send_state(DialogState.CLARIFICATION)
            await send({"type": "message", "text": "Je n'ai rien entendu. Peux-tu répéter?"})
        
        # ✅ Si en PRESENTING: annuler d'abord la présentation
        if current_state == DialogState.PRESENTING.value:
            log.info(f"[{session_id[:8]}] 🛑 PRESENTING → LISTENING (annulation présentation)")
            await cancel_presentation_task(notify_client=False)
        
        # ✅ Maintenant transition sécurisée vers LISTENING
        ctx = await dialogue.get_session(ctx.session_id) or ctx
        if ctx.state != DialogState.LISTENING.value:
            await dialogue.transition(ctx.session_id, DialogState.LISTENING)
        
        await send_state(DialogState.LISTENING)

    # _safe_uuid + persist_learning_turn moved to services/learning_log.py.
    # The wrapper binds learning_session_db_id (mutable closure) + session_id + ctx.course_id.
    _safe_uuid = _safe_uuid_external

    async def persist_learning_turn(**kwargs) -> None:
        await _persist_learning_turn_external(
            learning_session_db_id=learning_session_db_id,
            session_id=session_id,
            ctx_course_id=(ctx.course_id if ctx else None),
            **kwargs,
        )

    def _format_presentation_point() -> tuple[str, str, int]:
        chapter_no = (ctx.chapter_index + 1) if ctx else 0
        section_no = (ctx.section_index + 1) if ctx else 0

        location_bits: list[str] = []
        if chapter_no:
            location_bits.append(f"chapitre {chapter_no}")
        if section_no:
            location_bits.append(f"section {section_no}")

        location_label = ", ".join(location_bits) if location_bits else "point courant"
        total_chars = len(current_presentation_text or "")
        cursor_label = f"{current_presentation_cursor}/{total_chars}" if total_chars else str(current_presentation_cursor)
        return location_label, cursor_label, total_chars

    async def _pause_progress_ticker(start_ts: float, slide_id: str, reason: str) -> None:
        """Logs every 15s/30s/60s/... how long we've been waiting since the pause.

        Cancelled by start_pause_progress_ticker() (on new pause) or by
        stop_pause_progress_ticker() (on resume). Pure observability —
        no DB writes, no side effects.
        """
        # Absolute tick offsets from pause start (seconds). After the last
        # entry, we tick every TAIL_STEP_S to avoid tight-looping on a
        # past timestamp.
        intervals = [15, 30, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1800, 2400, 3000, 3600]
        TAIL_STEP_S = 1800  # every 30 min after 1h
        idx = 0
        try:
            while True:
                if idx < len(intervals):
                    next_at = intervals[idx]
                else:
                    # Hours 1+ : tick every TAIL_STEP_S past the last fixed offset
                    next_at = intervals[-1] + TAIL_STEP_S * (idx - len(intervals) + 1)
                idx += 1
                # Sleep until the next tick (relative to pause start)
                remaining = (start_ts + next_at) - time.time()
                if remaining > 0:
                    await asyncio.sleep(remaining)
                else:
                    # We're already past the scheduled tick (e.g., system was
                    # paused) — fast-forward to the next future tick instead
                    # of tight-looping.
                    elapsed_now = time.time() - start_ts
                    while idx < len(intervals) and intervals[idx] <= elapsed_now:
                        idx += 1
                elapsed = time.time() - start_ts
                m, s = divmod(int(elapsed), 60)
                # Bucket from resume_intelligence
                if elapsed < 10:    bucket = "QUICK"
                elif elapsed < 60:  bucket = "NORMAL"
                elif elapsed < 180: bucket = "GAP"
                else:               bucket = "LONG"
                log.info(
                    f"[{session_id[:8]}] ⏳ WAITING tick | elapsed={elapsed:.0f}s ({m}m{s:02d}s) | "
                    f"bucket={bucket} | reason={reason} | slide={slide_id or '?'}"
                )
        except asyncio.CancelledError:
            elapsed = time.time() - start_ts
            m, s = divmod(int(elapsed), 60)
            log.info(
                f"[{session_id[:8]}] ⏳ WAITING ticker stopped | total_wait={elapsed:.1f}s ({m}m{s:02d}s) | "
                f"reason={reason}"
            )
            raise

    def start_pause_progress_ticker(slide_id: str, reason: str) -> None:
        nonlocal pause_progress_task, pause_started_at
        # Cancel any prior ticker (idempotent — multiple pauses)
        if pause_progress_task and not pause_progress_task.done():
            pause_progress_task.cancel()
        pause_started_at = time.time()
        pause_progress_task = asyncio.create_task(
            _pause_progress_ticker(pause_started_at, slide_id, reason)
        )
        log.info(
            f"[{session_id[:8]}] ⏳ WAITING ticker STARTED | reason={reason} slide={slide_id or '?'} "
            f"| ticks at 15s, 30s, 60s, 90s, 2m, 3m, 4m, 5m, 7m, 10m, 15m, 20m, 30m, 40m, 50m, 1h"
        )

    async def stop_pause_progress_ticker() -> float:
        nonlocal pause_progress_task
        elapsed = (time.time() - pause_started_at) if pause_started_at else 0.0
        if pause_progress_task and not pause_progress_task.done():
            pause_progress_task.cancel()
            try:
                await pause_progress_task
            except (asyncio.CancelledError, Exception):
                pass
        pause_progress_task = None
        return elapsed

    async def record_pause_point(reason: str, notice_prefix: str = "⏸ Point d'arrêt mémorisé") -> str:
        nonlocal ctx

        if not ctx:
            log.info(f"[{session_id[:8]}] ⏸ record_pause_point SKIP | no ctx (reason={reason})")
            return ""

        import time as _time
        from datetime import datetime as _dt
        location_label, cursor_label, total_chars = _format_presentation_point()
        slide_id = ":".join(str(part) for part in current_presentation_key) if current_presentation_key else None
        pause_ts = _time.time()
        pause_iso = _dt.utcnow().isoformat()
        progress_pct = (current_presentation_cursor / total_chars * 100.0) if total_chars else 0.0
        log.info(
            f"[{session_id[:8]}] ⏸ PAUSE START | reason={reason} | slide={slide_id} | "
            f"cursor={cursor_label} ({progress_pct:.1f}%) | timestamp={pause_iso} (ts={pause_ts:.3f})"
        )

        try:
            _save_t0 = _time.time()
            paused_ctx = await dialogue.pause_session(
                ctx.session_id,
                slide_id=slide_id,
                char_offset=current_presentation_cursor,
                presentation_text=current_presentation_text or None,
                presentation_cursor=current_presentation_cursor,
                presentation_key=slide_id,
                slide_title=current_section_title or current_chapter_title or "",
            )
            log.info(
                f"[{session_id[:8]}] ⏸ PAUSE STATE SAVED | took={(_time.time()-_save_t0)*1000:.0f}ms | "
                f"text_len={total_chars} cursor={current_presentation_cursor} → Redis"
            )
            if paused_ctx:
                ctx = paused_ctx
            # Start the live waiting-time ticker
            start_pause_progress_ticker(slide_id or "?", reason)
        except Exception as pause_exc:
            log.warning(f"[{session_id[:8]}] ⏸ pause_session FAILED: {pause_exc}")

        point_text = f"{notice_prefix} — {location_label}, position {cursor_label}. Narration gardée en cache."
        record_checkpoint_event({
            "session_id": session_id,
            "language": session_lang,
            "subject": (ctx.course_analysis.get("course_domain", "") if ctx and ctx.course_analysis else ""),
            "checkpoint_type": "pause",
            "point_text": point_text,
            "location_label": location_label,
            "cursor_label": cursor_label,
            "slide_id": slide_id,
            "slide_title": current_section_title or current_chapter_title or "",
            "chapter_index": ctx.chapter_index if ctx else None,
            "section_index": ctx.section_index if ctx else None,
            "char_position": current_presentation_cursor,
            "reason": reason,
            "source": "checkpoint",
        })
        await send({"type": "system_notice", "text": point_text})

        try:
            await persist_learning_turn(
                event_type="pause",
                question_text="",
                answer_text="",
                language=session_lang,
                subject=(ctx.course_analysis.get("course_domain", "") if ctx and ctx.course_analysis else ""),
                course_id_value=ctx.course_id if ctx else None,
                confusion_detected=False,
                confusion_reason=reason,
                action_taken="pause",
                reward=0.0,
                stt_time=0.0,
                llm_time=0.0,
                tts_time=0.0,
                total_time=0.0,
                profile_snapshot=student_profile.copy(),
                concept=ctx.last_slide_explained if ctx else "",
                chapter_index=ctx.chapter_index if ctx else None,
                section_index=ctx.section_index if ctx else None,
                char_position=current_presentation_cursor,
                extra_payload={
                    "reason": reason,
                    "point_text": point_text,
                    "slide_id": slide_id,
                    "cursor_label": cursor_label,
                    "total_chars": total_chars,
                },
                session_state=DialogState.WAITING.value,
            )
        except Exception as learning_exc:
            log.debug(f"[{session_id[:8]}] Pause learning log skipped: {learning_exc}")

        return point_text

    async def cancel_next_slide_prefetch() -> None:
        nonlocal next_slide_prefetch_task
        if next_slide_prefetch_task and not next_slide_prefetch_task.done():
            next_slide_prefetch_task.cancel()
            try:
                await next_slide_prefetch_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                log.debug(f"[{session_id[:8]}] next slide prefetch cancel error: {exc}")
        next_slide_prefetch_task = None

    async def warm_presentation_audio_cache(narration_text: str, language_code: str, rate: str) -> None:
        if not narration_text.strip():
            return

        for sentence, _, _ in split_sentences_with_spans(narration_text):
            if not sentence.strip():
                continue
            try:
                await synthesize_cached_tts(
                    sentence,
                    language_code=language_code,
                    rate=rate,
                    cache_scope="prefetch",
                )
            except Exception as cache_exc:
                log.debug(f"[{session_id[:8]}] Prefetch TTS skipped: {cache_exc}")

    async def prefetch_next_slide(
        current_slide_key: tuple[str, int, int],
        course_id_value: str,
        chapter_index_value: int,
        section_index_value: int,
        language_code: str,
        student_level_value: str,
        course_summary_value: str,
        rate_value: str,
        current_slide_title: str,
    ) -> None:
        if not course_id_value:
            return

        next_candidates = [
            (chapter_index_value, section_index_value + 1),
            (chapter_index_value + 1, 0),
        ]

        next_slide_ctx = None
        next_chapter_index = None
        next_section_index = None

        for candidate_chapter, candidate_section in next_candidates:
            if candidate_chapter < 0 or candidate_section < 0:
                continue
            next_slide_ctx = await load_course_slide_context(
                course_id_value,
                candidate_chapter,
                candidate_section,
            )
            if next_slide_ctx:
                next_chapter_index = candidate_chapter
                next_section_index = candidate_section
                break

        if not next_slide_ctx or next_chapter_index is None or next_section_index is None:
            return

        next_slide_id = f"{course_id_value}:{next_chapter_index}:{next_section_index}"

        if current_presentation_key != current_slide_key:
            return

        cached_snapshot = await dialogue.load_presentation_snapshot(session_id, next_slide_id)
        next_narration_text = (cached_snapshot or {}).get("presentation_text") or ""
        next_cursor = int((cached_snapshot or {}).get("presentation_cursor") or 0)

        if next_narration_text:
            log.info(
                f"[{session_id[:8]}] 💾 Prefetch cache HIT (slide={next_slide_id}, chars={len(next_narration_text)}) → 0 LLM call"
            )
        else:
            log.info(
                f"[{session_id[:8]}] 🔄 Prefetch cache MISS (slide={next_slide_id}) → Teaching Graph va générer"
            )
            next_narration_text = await explain_slide_focused(
                slide_content=next_slide_ctx.get("content") or "",
                chapter_idx=next_chapter_index,
                chapter_title=next_slide_ctx.get("chapter_title") or "",
                section_title=next_slide_ctx.get("section_title") or "",
                language=language_code,
                student_level=student_level_value,
                course_summary=course_summary_value,
                is_resume=False,
                session_id=session_id,
                course_id=course_id_value,
                section_idx=next_section_index,
                # Slide image (when available) → vision-based concept
                # extraction in the planner replaces text heuristics.
                slide_image_path=next_slide_ctx.get("slide_path") or next_slide_ctx.get("image_url") or "",
            )
            next_narration_text = next_narration_text.strip()
            next_cursor = 0

        if not next_narration_text or current_presentation_key != current_slide_key:
            return

        await dialogue.save_presentation_snapshot(
            session_id,
            next_slide_id,
            next_narration_text,
            presentation_cursor=next_cursor,
            slide_title=next_slide_ctx.get("section_title") or next_slide_ctx.get("chapter_title") or current_slide_title,
        )

        await warm_presentation_audio_cache(next_narration_text, language_code, rate_value)

        log.info(
            f"[{session_id[:8]}] ✅ Next slide prefetched | key={next_slide_id} | chars={len(next_narration_text)}"
        )

    async def schedule_next_slide_prefetch(
        *,
        current_slide_key: tuple[str, int, int],
        course_id_value: str,
        chapter_index_value: int,
        section_index_value: int,
        language_code: str,
        student_level_value: str,
        course_summary_value: str,
        rate_value: str,
        current_slide_title: str,
    ) -> None:
        nonlocal next_slide_prefetch_task

        await cancel_next_slide_prefetch()

        # Toggle: ENABLE_NEXT_SLIDE_PREFETCH=false in .env disables this.
        # Useful on slow CPUs where each Ollama call costs 50-90s — the
        # prefetch otherwise piles up calls in the queue ahead of the
        # student's question, making Q&A unbearably slow.
        import os as _os
        if _os.getenv("ENABLE_NEXT_SLIDE_PREFETCH", "true").lower() != "true":
            log.debug(f"[{session_id[:8]}] next slide prefetch disabled via env")
            return

        async def runner() -> None:
            nonlocal next_slide_prefetch_task
            try:
                await prefetch_next_slide(
                    current_slide_key=current_slide_key,
                    course_id_value=course_id_value,
                    chapter_index_value=chapter_index_value,
                    section_index_value=section_index_value,
                    language_code=language_code,
                    student_level_value=student_level_value,
                    course_summary_value=course_summary_value,
                    rate_value=rate_value,
                    current_slide_title=current_slide_title,
                )
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                log.debug(f"[{session_id[:8]}] next slide prefetch failed: {exc}")
            finally:
                if asyncio.current_task() is next_slide_prefetch_task:
                    next_slide_prefetch_task = None

        next_slide_prefetch_task = asyncio.create_task(runner())

    # synthesize_cached_tts implementation lives in services/presentation.py
    # We bind session_id via this thin wrapper so callers in this WS keep their signature.
    async def synthesize_cached_tts(
        text_to_speak: str, *, language_code: str, rate: str = "+0%",
        cache_scope: str = "presentation",
    ) -> tuple[bytes | None, float, str, str, str | None, bool]:
        return await _synth_cached_tts_external(
            text_to_speak, language_code=language_code, rate=rate,
            cache_scope=cache_scope, session_id=session_id,
        )

    async def _stream_audio(audio_bytes: bytes, mime: str | None, stream_id: int) -> None:
        nonlocal tts_started_at
        chunk_size = 4096
        total_len = len(audio_bytes)
        # 🛡️ Mark TTS streaming start (echo suppression gate) au 1er chunk reel envoye
        if total_len > 0 and tts_started_at == 0.0:
            tts_started_at = time.time()
        # 🛡️ Append PCM au ring buffer pour echo cross-correlation a audio_end
        if total_len > 0 and Config.ENABLE_ECHO_XCORR:
            pcm = _tts_bytes_to_pcm(audio_bytes, Config.SAMPLE_RATE)
            if pcm is not None and pcm.size > 0:
                tts_pcm_history.extend(pcm.tolist())
        for i in range(0, total_len, chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            if not await send({
                "type":      "audio_chunk",
                "stream_id": stream_id,
                "data":      base64.b64encode(chunk).decode(),
                "mime":      mime,
                "final":     (i + chunk_size) >= total_len,
            }):
                return
            # Laisse la boucle événementielle respirer entre chunks
            await asyncio.sleep(0)

    async def cancel_audio_stream(notify_client: bool = False, turn_id: int | None = None) -> None:
        nonlocal audio_stream_task
        if audio_stream_task and not audio_stream_task.done():
            audio_stream_task.cancel()
            try:
                await audio_stream_task
            except asyncio.CancelledError:
                pass
        audio_stream_task = None
        if notify_client:
            payload = {"type": "audio_interrupted", "stream_id": current_stream_id}
            if turn_id is not None:
                payload["turn_id"] = turn_id
            await send(payload)

    async def start_audio_stream(
        audio_bytes: bytes | None,
        mime: str | None,
        auto_listening: bool = True,
    ) -> None:
        nonlocal audio_stream_task, current_stream_id
        await cancel_audio_stream(notify_client=False)

        if not audio_bytes:
            if auto_listening:
                await set_listening_state()
            return

        current_stream_id += 1
        stream_id = current_stream_id

        async def runner() -> None:
            nonlocal audio_stream_task
            try:
                await _stream_audio(audio_bytes, mime, stream_id)
                if auto_listening and not websocket_closed:
                    await set_listening_state()
            except asyncio.CancelledError:
                pass
            except Exception as stream_exc:
                log.error(f"[{session_id[:8]}] audio stream error: {stream_exc}")
            finally:
                if asyncio.current_task() is audio_stream_task:
                    audio_stream_task = None

        audio_stream_task = asyncio.create_task(runner())

    def split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
        import re

        spans: list[tuple[str, int, int]] = []
        for match in re.finditer(r"[^.!?]+(?:[.!?]+|\Z)", text, flags=re.S):
            raw = match.group()
            sentence = raw.strip()
            if not sentence:
                continue
            left_trim = len(raw) - len(raw.lstrip())
            right_trim = len(raw.rstrip())
            spans.append((sentence, match.start() + left_trim, match.start() + right_trim))

        if not spans and text.strip():
            stripped = text.strip()
            start = text.find(stripped)
            spans.append((stripped, max(0, start), max(0, start) + len(stripped)))

        return spans

    async def cancel_presentation_task(notify_client: bool = False) -> None:
        nonlocal presentation_task
        if presentation_task and not presentation_task.done():
            presentation_task.cancel()
            try:
                await presentation_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                log.debug(f"[{session_id[:8]}] presentation task cancel error: {exc}")
        presentation_task = None
        if notify_client:
            await send({"type": "audio_interrupted", "stream_id": current_stream_id})

    # explain_slide_focused implementation lives in services/presentation.py.
    # This wrapper binds the WS-scoped session_id (default) and teaching_graph.
    async def explain_slide_focused(
        slide_content: str,
        chapter_idx: int,
        chapter_title: str,
        section_title: str = "",
        language: str = "fr",
        student_level: str = "lycée",
        course_summary: str = "",
        is_resume: bool = False,
        session_id: str | None = None,
        course_id: str = "",
        section_idx: int = 0,
        on_plan_ready=None,
    ) -> str:
        return await _explain_slide_external(
            slide_content=slide_content,
            chapter_idx=chapter_idx,
            chapter_title=chapter_title,
            section_title=section_title,
            language=language,
            student_level=student_level,
            course_summary=course_summary,
            is_resume=is_resume,
            session_id=session_id,
            course_id=course_id,
            section_idx=section_idx,
            on_plan_ready=on_plan_ready,
            teaching_graph=teaching_graph,
            student_id=(ctx.student_id if ctx and ctx.student_id else None),
        )

# Legacy explain_slide_focused removed — moved to services/presentation.py


    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type", "")

            # ── start_session ─────────────────────────────────────────
            if msg_type == "start_session":
                # ✅ Flexible auth: Only validate if we have a token entry
                # - If a token is registered for this sid: validate strictly
                # - Otherwise: auto-approve unconditionally (legacy SDK)
                token = msg.get("token")

                expected_token = await get_session_token(session_id)
                if expected_token is not None:
                    # Secure flow: validate before consuming
                    if not token or expected_token != token:
                        await send({"type": "error", "message": "Authentication failed: invalid or missing token"})
                        await websocket.close(code=1008, reason="Unauthorized")
                        log.warning(f"[{session_id[:8]}] ❌ WebSocket auth failed (invalid token)")
                        return
                    # Token is valid — consume it (one-time use)
                    await consume_session_token(session_id)
                    log.info(f"[{session_id[:8]}] ✅ WebSocket authenticated (secure token)")
                else:
                    # Legacy flow: No token registered, auto-approve
                    log.info(f"[{session_id[:8]}] ✅ WebSocket auto-approved (legacy SDK)")
                
                lang  = msg.get("language", "fr")
                level = msg.get("level",    "lycée")
                ctx   = await dialogue.create_session(
                    session_id=session_id,
                    language=lang,
                    student_level=level,
                    course_id=(msg.get("course_id") or None),
                )
                session_lang  = lang
                session_level = level

                # ✨ Resolve authenticated student_id from JWT (if provided in start_session)
                # — used by services to fetch StudentProfile + adapt personalization.
                _auth_token = msg.get("auth_token") or msg.get("jwt")
                if _auth_token and ctx:
                    try:
                        from handlers.auth import decode_access_token
                        claims = decode_access_token(_auth_token)
                        ctx.student_id = str(claims.get("sub") or "")
                        if ctx.student_id:
                            # Load the persisted Student row to read
                            # preferred_language and student_level. The
                            # frontend's ``msg.language`` / ``msg.level``
                            # are coarse defaults ; the Student profile
                            # is the canonical source for persisted
                            # preferences (filled at register time).
                            try:
                                import uuid as _uuid
                                from database.init_db import AsyncSessionLocal
                                from database.models import Student
                                from sqlalchemy import select
                                async with AsyncSessionLocal() as _db:
                                    _row = (await _db.execute(
                                        select(Student).where(
                                            Student.id == _uuid.UUID(ctx.student_id),
                                        )
                                    )).scalar_one_or_none()
                                if _row:
                                    # Override session lang/level with the
                                    # student's persisted preferences. The
                                    # client-side defaults (fr / lycée)
                                    # only apply when the student hasn't
                                    # set them in their profile.
                                    if _row.preferred_language:
                                        ctx.language = _row.preferred_language
                                        session_lang = _row.preferred_language
                                        lang = _row.preferred_language
                                    _row_level = getattr(_row, "student_level", None)
                                    if _row_level:
                                        ctx.student_level = _row_level
                                        session_level = _row_level
                                        level = _row_level
                                    log.info(
                                        f"[{session_id[:8]}] 🎓 profile loaded | "
                                        f"language={ctx.language} level={ctx.student_level}"
                                    )
                            except Exception as _profile_exc:                # noqa: BLE001
                                log.debug(
                                    f"[{session_id[:8]}] Student row load skipped: {_profile_exc}"
                                )
                            await dialogue._save(ctx)
                            log.info(f"[{session_id[:8]}] 🎓 Linked to student {ctx.student_id[:8]} ({claims.get('email')})")
                    except Exception as exc:
                        log.debug(f"[{session_id[:8]}] JWT decode skipped: {exc}")
                history.clear()
                last_active_course_id = None  # reset course tracker
                interrupt_audio = False  # 🟢 Réinitialiser le flag
                await cancel_text_question_task()
                active_text_turn_id = 0
                text_turn_seq = 0

                try:
                    course_uuid = _safe_uuid(msg.get("course_id") or (ctx.course_id if ctx else None))
                    async with AsyncSessionLocal() as db:
                        learning_session = await create_learning_session(
                            db,
                            student_id=session_id,
                            course_id=course_uuid,
                            language=lang,
                            level=level,
                        )
                        await update_session_state(
                            db,
                            learning_session.id,
                            DialogState.LISTENING.value,
                            chapter_index=0,
                            section_index=0,
                            char_position=0,
                        )
                        await db.commit()
                        learning_session_db_id = learning_session.id
                except Exception as exc:
                    learning_session_db_id = None
                    log.warning(f"[{session_id[:8]}] ⚠️ Learning session DB init failed: {exc}")

                await send({"type": "session_ready", "session_id": ctx.session_id})
                
                # ✅ NOUVEAU: Transition explicite IDLE → LISTENING
                await dialogue.transition(ctx.session_id, DialogState.LISTENING)
                await send_state(DialogState.LISTENING)
                log.info(f"[{session_id[:8]}] 🚀 Session démarrée | lang={lang} level={level} state=LISTENING")

            # ── audio_chunk — accumule les données audio ───────────────
            elif msg_type == "audio_chunk":
                raw = msg.get("data", "")
                turn_id = int(msg.get("turn_id") or 0)
                if raw:
                    try:
                        decoded = base64.b64decode(raw)
                        log.info(f"   📦 Audio chunk: base64_len={len(raw)} → {len(decoded)} bytes")
                        # ✅ REFRESH ctx (peut être modifié par run_presentation)
                        if ctx:
                            ctx = await dialogue.get_session(session_id) or ctx
                        
                        if ctx and ctx.state in [DialogState.RESPONDING.value, DialogState.PRESENTING.value]:
                            # 🛡️ Audio source gating (Option C — anti-echo loop)
                            # 3 gates avant de valider une vraie interruption utilisateur :
                            #   Gate 1 : suppression window (TTS qui sort par les HP est re-capte)
                            #   Gate 2 : taille du chunk (echo bursts typiquement < 3 KB)
                            #   Gate 3 : debounce (require continuous voice for >= 400ms)
                            now_ts = time.time()
                            # Sanity reset : debounce trop vieux (changement d'etat externe entre 2 chunks)
                            if voice_pending_since is not None and (now_ts - voice_pending_since) > 5.0:
                                voice_pending_since = None
                            in_suppression = (
                                tts_started_at > 0
                                and (now_ts - tts_started_at) < (Config.TTS_SUPPRESSION_WINDOW_S + Config.TTS_ECHO_TAIL_S)
                            )
                            is_tiny = len(decoded) < Config.MIN_INTERRUPT_BYTES

                            fire_interrupt = False
                            if in_suppression or is_tiny:
                                # Probable echo / bruit court → on filtre + reset debounce
                                if voice_pending_since is not None:
                                    voice_pending_since = None
                                log.debug(
                                    f"[{session_id[:8]}] 🔇 Chunk filtre "
                                    f"(suppression={in_suppression}, tiny={is_tiny}, {len(decoded)}B, "
                                    f"+{(now_ts - tts_started_at) if tts_started_at > 0 else 0:.2f}s)"
                                )
                            else:
                                # Hors fenetre suppression + taille suffisante → debounce
                                if voice_pending_since is None:
                                    voice_pending_since = now_ts
                                    log.debug(f"[{session_id[:8]}] 🔍 Voice pending (debounce start)")
                                elif (now_ts - voice_pending_since) >= Config.MIN_INTERRUPT_DURATION_S:
                                    fire_interrupt = True

                            if fire_interrupt:
                                debounce_dur = now_ts - voice_pending_since if voice_pending_since else 0.0
                                log.info(
                                    f"[{session_id[:8]}] ⚡ Interruption confirmee "
                                    f"(debounce={debounce_dur:.2f}s, {len(decoded)}B, source=user)"
                                )
                                interrupt_audio = True
                                # 🛡️ FSM transition : SPEAKING/THINKING → INTERRUPTED
                                await vfsm.transition(VoiceState.INTERRUPTED, reason="user_interrupt")

                                # ✅ Détermine si interruption TRÈS PRÉCOCE (avant streaming)
                                time_since_presenting = time.time() - presentation_start_time if presentation_start_time > 0 else 999
                                if ctx.state == DialogState.PRESENTING.value and time_since_presenting < 1.0:
                                    log.info(f"[{session_id[:8]}] ⚡ TRÈS TÔT interruption ({time_since_presenting:.2f}s après PRESENTING) → annuler préparation")

                                # ✅ CRUCIAL: Sauvegarder le point AVANT d'annuler
                                await record_pause_point("voice_interrupt", notice_prefix="⏸ Point d'arrêt mémorisé (voix)")

                                # ✅ Annuler la présentation EN COURS pour éviter conflit
                                await cancel_presentation_task(notify_client=True)
                                await cancel_audio_stream(notify_client=True, turn_id=turn_id)
                                await cancel_text_question_task()
                                await dialogue.transition(ctx.session_id, DialogState.LISTENING)
                                await send_state(DialogState.LISTENING, turn_id=turn_id)

                                # Reset des gates apres interruption confirmee
                                tts_started_at = 0.0
                                voice_pending_since = None
                                # FSM : INTERRUPTED → LISTENING (clean state apres cancel)
                                await vfsm.transition(VoiceState.LISTENING, reason="post_interrupt_listen")
                                # ✨ Bayes posterior — interrupt during slide = visual/kinesthetic exploration
                                from pedagogy.personalization.learning_style.bayes import fire_signal
                                fire_signal(ctx.student_id if ctx else None, "interrupt_during_slide")
                        audio_buffer.append(decoded)
                    except Exception as e:
                        log.error(f"❌ Failed to decode audio chunk: {e}")

            # ── audio_end — traite l'audio accumulé ───────────────────
            elif msg_type == "audio_end":
                interrupt_audio = False
                turn_id = int(msg.get("turn_id") or 0)
                # 📊 KPI #2 — start measuring response latency (question received)
                from observability.kpi_logger import KPITracker
                KPITracker.get().mark_question_received(session_id, turn_id)
                
                # ✅ REFRESH ctx FROM REDIS (peut être modifié par run_presentation en tâche async)
                if not ctx:
                    ctx = await dialogue.get_session(session_id)
                else:
                    ctx = await dialogue.get_session(session_id) or ctx
                
                # ✅ SÉCURITÉ: Ne jamais traiter audio_end sans session active
                if not ctx:
                    await send({"type": "error", "message": "Aucune session active (démarrez avec start_session)", "turn_id": turn_id})
                    continue
                
                # ✅ SÉCURITÉ: Bloquer transition IDLE → PROCESSING (mais LISTENING OK!)
                if ctx.state == DialogState.IDLE.value:
                    log.warning(f"[{session_id[:8]}] ⚠️ Tentative audio_end en IDLE → ignoré (jamais IDLE → PROCESSING)")
                    await send({"type": "error", "message": "Session en IDLE, démarrage nécessaire", "turn_id": turn_id})
                    continue
                
                if not audio_buffer:
                    await send({"type": "error", "message": "Aucun audio reçu", "turn_id": turn_id})
                    continue
                await cancel_audio_stream(notify_client=False)

                # 🛡️ FSM : LISTENING → THINKING (start pipeline)
                await vfsm.transition(VoiceState.THINKING, reason="audio_end_received")

                # Assembler et convertir l'audio
                full_audio = b"".join(audio_buffer)
                log.info(f"[{session_id[:8]}] 🎙️ Audio buffer assembled: {len(audio_buffer)} chunks = {len(full_audio)} total bytes")
                audio_buffer.clear()

                media_stamp = int(time.time() * 1000)
                question_audio_object = f"turns/questions/{session_id}/{turn_id}_{media_stamp}.webm"
                await save_media_bytes(question_audio_object, full_audio, "audio/webm")

                try:
                    audio_np = audio_bytes_to_numpy(full_audio)
                except RuntimeError as exc:
                    await send({"type": "error", "message": str(exc), "turn_id": turn_id})
                    continue

                # ═══════════════════════════════════════════════════════════════════
                # 🛡️ AUDIO_END GUARDS — Anti-noise + Anti-echo (Option C+)
                # ═══════════════════════════════════════════════════════════════════
                # Avant tout pipeline lourd (Whisper + LLM + TTS), on verifie :
                #   1. Energie audio (RMS) suffisante  → skip silence/bruit
                #   2. Cross-correlation avec TTS recent → skip echo qui passe les gates par-chunk
                # Si l'un des 2 echoue, on abandonne le turn et retourne en LISTENING.
                if len(audio_np) > 0:
                    turn_rms = float(np.sqrt(np.mean(audio_np ** 2)))
                else:
                    turn_rms = 0.0

                if turn_rms < Config.MIN_TURN_ENERGY_RMS:
                    log.info(
                        f"[{session_id[:8]}] 🔇 Turn discarded — RMS={turn_rms:.4f} < {Config.MIN_TURN_ENERGY_RMS} "
                        f"(silence/noise, {len(full_audio)}B)"
                    )
                    await dialogue.transition(ctx.session_id, DialogState.LISTENING)
                    await send_state(DialogState.LISTENING, turn_id=turn_id)
                    continue

                if Config.ENABLE_ECHO_XCORR and len(tts_pcm_history) >= len(audio_np):
                    is_echo, peak = _is_echo_of_recent_tts(audio_np, tts_pcm_history, Config.ECHO_XCORR_THRESHOLD)
                    if is_echo:
                        log.info(
                            f"[{session_id[:8]}] 🔇 Turn discarded — echo xcorr peak={peak:.3f} "
                            f"> {Config.ECHO_XCORR_THRESHOLD} (TTS leak detected)"
                        )
                        await dialogue.transition(ctx.session_id, DialogState.LISTENING)
                        await send_state(DialogState.LISTENING, turn_id=turn_id)
                        continue
                    else:
                        log.debug(f"[{session_id[:8]}] ✓ xcorr peak={peak:.3f} (under threshold {Config.ECHO_XCORR_THRESHOLD})")

                # ═══════════════════════════════════════════════════════════════════
                # 🎤 SILERO VAD: Filter audio with backend voice detection (PyTorch)
                # ═══════════════════════════════════════════════════════════════════
                log.info(f"[{session_id[:8]}] 🎤 Silero VAD: Filtering {len(audio_np)} samples...")
                
                # Break audio into CHUNK_SIZE chunks (512 samples @ 16kHz = 32ms)
                chunk_size = Config.CHUNK_SIZE
                vad_confidence_scores = []
                filtered_chunks = []
                
                for i in range(0, len(audio_np), chunk_size):
                    chunk = audio_np[i:i + chunk_size]
                    
                    # Pad chunk if last one is shorter
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant', constant_values=0.0)
                    
                    # Get speech probability (0.0 = silence, 1.0 = definitely speech)
                    prob = audio_input.get_speech_probability(chunk.astype(np.float32))
                    vad_confidence_scores.append(prob)
                    
                    # Keep chunk if confidence > threshold (0.5 = medium confidence)
                    if prob > Config.SPEECH_THRESHOLD:
                        filtered_chunks.append(chunk[:len(audio_np[i:i + chunk_size])])
                
                # Concatenate filtered chunks
                if filtered_chunks:
                    audio_np = np.concatenate(filtered_chunks)
                    avg_confidence = np.mean(vad_confidence_scores)
                    log.info(
                        f"[{session_id[:8]}] ✅ Silero VAD: "
                        f"Kept {len(filtered_chunks)} / {len(vad_confidence_scores)} chunks, "
                        f"avg_confidence={avg_confidence:.2f}, "
                        f"audio_len {len(audio_np)} samples"
                    )
                else:
                    # No speech detected - send a neutral notice
                    log.info(f"[{session_id[:8]}] ℹ️ Silero VAD: No speech detected (all chunks < threshold)")
                    await send({
                        "type": "system_notice",
                        "text": "Aucune voix détectée.",
                        "turn_id": turn_id,
                    })
                    await dialogue.transition(ctx.session_id, DialogState.LISTENING)
                    await send_state(DialogState.LISTENING, turn_id=turn_id)
                    continue

                # Transition → PROCESSING (sécurisée via state machine)
                await dialogue.transition(ctx.session_id, DialogState.PROCESSING)
                await send_state(DialogState.PROCESSING, turn_id=turn_id)

                # ══════════════════════════════════════════════════════════════
                # 🚀 STREAMING PIPELINE: Real-time LLM → TTS
                # ══════════════════════════════════════════════════════════════
                
                current_stream_id += 1
                stream_id = current_stream_id
                turn_id = int(msg.get("turn_id") or 0)
                response_audio_parts: list[bytes] = []
                response_audio_mime = ""
                
                async def on_text_chunk(sentence: str, full_response: str):
                    chunk_text = (full_response or sentence or "").strip()
                    if not chunk_text:
                        return
                    if not await send({
                        "type": "answer_text",
                        "text": chunk_text,
                        "turn_id": turn_id,
                        "partial": True,
                        "final": False,
                    }):
                        return

                async def on_transcription(text: str, lang: str, confidence: float):
                    await send_state(
                        DialogState.PROCESSING,
                        "stt_transcription",
                        {
                            "transcription": text,
                            "language": lang,
                            "confidence": confidence,
                            "slide_title": current_section_title or current_chapter_title or "",
                            "chapter_title": current_chapter_title or "",
                            "section_title": current_section_title or "",
                        },
                        {"progress_pct": 12, "confidence": confidence},
                        turn_id=turn_id,
                    )
                    if not await send({
                        "type": "transcription",
                        "text": text,
                        "lang": lang,
                        "confidence": confidence,
                        "turn_id": turn_id,
                    }):
                        log.warning(f"[{session_id[:8]}] ⚠️ Transcription non envoyée")
                
                async def on_state_change(substep: str, details: dict = None):
                    """✅ NOUVEAU: Callback pour les mises à jour d'état de la pipeline"""
                    return await send_state(
                        DialogState.PROCESSING,
                        substep=substep,
                        details=details or {},
                        turn_id=turn_id,
                    )
                
                async def on_audio_chunk(audio_bytes: bytes, mime: str):
                    """Stream each sentence's audio as it's generated"""
                    nonlocal response_audio_mime, tts_started_at
                    if audio_bytes:
                        # 🛡️ Mark TTS streaming start (echo suppression gate)
                        if tts_started_at == 0.0:
                            tts_started_at = time.time()
                            # FSM : THINKING → SPEAKING (1er chunk TTS effectivement envoye)
                            await vfsm.transition(VoiceState.SPEAKING, reason="tts_first_chunk")
                            # 📊 KPI #2 — first response audio chunk = end of response latency
                            from observability.kpi_logger import KPITracker
                            _resp_lat = KPITracker.get().mark_first_response_chunk(session_id, turn_id)
                            if _resp_lat is not None:
                                log.info(f"[{session_id[:8]}] 📊 response latency = {_resp_lat:.2f}s")
                        # 🛡️ Append PCM au ring buffer pour echo cross-correlation
                        if Config.ENABLE_ECHO_XCORR:
                            pcm = _tts_bytes_to_pcm(audio_bytes, Config.SAMPLE_RATE)
                            if pcm is not None and pcm.size > 0:
                                tts_pcm_history.extend(pcm.tolist())
                        response_audio_parts.append(audio_bytes)
                        if mime:
                            response_audio_mime = mime
                        chunk_size = 4096
                        total_len = len(audio_bytes)
                        for i in range(0, total_len, chunk_size):
                            # 🚨 Vérifier le flag d'interruption EN TEMPS RÉEL
                            if interrupt_audio:
                                log.info(f"[{session_id[:8]}] 🛑 Audio interruption détectée → STOP streaming")
                                return
                            
                            chunk = audio_bytes[i:i + chunk_size]
                            if not await send({
                                "type":      "audio_chunk",
                                "stream_id": stream_id,
                                "turn_id":   turn_id,
                                "data":      base64.b64encode(chunk).decode(),
                                "mime":      mime,
                                "final":     (i + chunk_size) >= total_len,
                            }):
                                return
                            await asyncio.sleep(0)  # yield control
                
                try:
                    # STT language: auto-detect per utterance. We do NOT force the
                    # course language here — students can ask questions in a
                    # language different from the slides (e.g. English-speaking
                    # student on a course whose analyzer mislabeled the language),
                    # and forcing a wrong language collapses Whisper accuracy on
                    # numbers, acronyms, and proper nouns.
                    course_id_for_rag = ctx.course_id if ctx else None  # ✅ Pass course_id from session context
                    # Bias Whisper with chapter-wide concept VOCABULARY plus a
                    # sliver of current-slide OCR. The glossary (deduped
                    # idea_labels + section_titles from the whole chapter) lets
                    # Whisper recognise terms the student may know from other
                    # slides — e.g. "agentic AI" while paused on the "Pillars
                    # of Modern AI" slide. Never feed the live narration: it's
                    # a coherent sentence and Whisper parrots it back (observed:
                    # every student utterance came out as a slice of the
                    # previous tutor answer). Both the glossary and the OCR
                    # text are telegraphic — no sentence rhythm to mimic.
                    # Total capped at 300 chars to stay under Whisper's
                    # 224-token prompt limit.
                    glossary = ""
                    try:
                        if course_id_for_rag and current_chapter_idx is not None and rag:
                            glossary = rag.get_chapter_vocab(
                                course=course_id_for_rag,
                                chapter_idx=int(current_chapter_idx),
                            )
                    except Exception as exc:
                        log.debug("get_chapter_vocab failed: %s", exc)
                        glossary = ""
                    _stt_slide_text = (
                        ctx.paused_state.get("slide_content", "") if ctx else ""
                    ) or ""
                    parts: list[str] = []
                    if current_chapter_title:
                        parts.append(current_chapter_title)
                    if glossary:
                        parts.append(f"Glossary: {glossary}")
                    if _stt_slide_text:
                        parts.append(_stt_slide_text[:120])
                    slide_context_for_stt = (". ".join(parts))[:300] if parts else None
                    result = await run_pipeline_streaming(
                        audio_np, session_id, history,
                        on_text_chunk=on_text_chunk,
                        on_transcription=on_transcription,
                        on_audio_chunk=on_audio_chunk,
                        on_state_change=on_state_change,  # ✅ NOUVEAU: State updates
                        force_language=None,
                        course_id=course_id_for_rag,  # ✅ Scoped RAG retrieval
                        ctx=ctx,
                        slide_context=slide_context_for_stt,
                        # ✅ Inject dependencies
                        transcriber=transcriber,
                        rag=rag,
                        voice=voice,
                        brain=brain,
                        dialogue=dialogue,
                        csv_logger=csv_logger,
                        stt_logger=stt_logger,
                        qa_graph=qa_graph,  # ✅ Audio now routes through the QA graph
                    )

                    if result.get("no_speech"):
                        await send({
                            "type": "system_notice",
                            "text": result.get("message", "Aucune voix détectée."),
                            "turn_id": turn_id,
                        })
                        await save_media_json(
                            f"turns/transcripts/{session_id}/{turn_id}_{media_stamp}.json",
                            {
                                "kind": "audio_question",
                                "session_id": session_id,
                                "turn_id": turn_id,
                                "question_audio_path": question_audio_object,
                                "question_text": result.get("transcription", {}).get("text", ""),
                                "answer_text": "",
                                "language": result.get("transcription", {}).get("language", session_lang),
                                "subject": result.get("subject", ""),
                                "audio_answer_path": "",
                                "note": result.get("message", "Aucune voix détectée."),
                            },
                        )
                        await dialogue.transition(ctx.session_id, DialogState.LISTENING)
                        await send_state(DialogState.LISTENING, turn_id=turn_id)
                        continue

                    if "error" in result:
                        # ✅ Quand STT échoue: passer par CLARIFICATION (transition valide depuis PROCESSING)
                        await send({"type": "error", "message": result["error"], "turn_id": turn_id})
                        await save_media_json(
                            f"turns/transcripts/{session_id}/{turn_id}_{media_stamp}.json",
                            {
                                "kind": "audio_question",
                                "session_id": session_id,
                                "turn_id": turn_id,
                                "question_audio_path": question_audio_object,
                                "question_text": result.get("transcription", {}).get("text", ""),
                                "answer_text": "",
                                "language": result.get("transcription", {}).get("language", session_lang),
                                "subject": result.get("subject", ""),
                                "audio_answer_path": "",
                                "error": result["error"],
                            },
                        )
                        try:
                            await persist_learning_turn(
                                event_type="error",
                                question_text=result.get("transcription", {}).get("text", ""),
                                answer_text="",
                                language=result.get("transcription", {}).get("language", session_lang),
                                subject=result.get("subject", ""),
                                course_id_value=course_id_for_rag,
                                confusion_detected=bool(result.get("confusion", {}).get("detected", False)),
                                confusion_reason=result.get("confusion", {}).get("reason", ""),
                                action_taken="error",
                                reward=0.0,
                                stt_time=float(result.get("performance", {}).get("stt_time", 0.0)),
                                llm_time=float(result.get("performance", {}).get("llm_time", 0.0)),
                                tts_time=float(result.get("performance", {}).get("tts_time", 0.0)),
                                total_time=float(result.get("performance", {}).get("total_time", 0.0)),
                                profile_snapshot=student_profile.copy(),
                                concept=ctx.last_slide_explained if ctx else "",
                                chapter_index=ctx.chapter_index if ctx else None,
                                section_index=ctx.section_index if ctx else None,
                                char_position=current_presentation_cursor if ctx else None,
                                extra_payload={
                                    "source": "streaming_error",
                                    "message": result["error"],
                                },
                            )
                        except Exception as learning_exc:
                            log.debug(f"[{session_id[:8]}] Error learning log skipped: {learning_exc}")
                        record_session_event({
                            "session_id": session_id,
                            "language": result.get("transcription", {}).get("language", session_lang),
                            "question": result.get("transcription", {}).get("text", ""),
                            "stt_text": result.get("transcription", {}).get("text", ""),
                            "answer": "",
                            "turn_id": turn_id,
                            "stt_time": float(result.get("performance", {}).get("stt_time", 0.0)),
                            "llm_time": float(result.get("performance", {}).get("llm_time", 0.0)),
                            "tts_time": float(result.get("performance", {}).get("tts_time", 0.0)),
                            "total_time": float(result.get("performance", {}).get("total_time", 0.0)),
                            "meets_kpi": False,
                            "subject": result.get("subject", ""),
                            "confusion": bool(result.get("confusion", {}).get("detected", False)),
                            "confusion_reason": result.get("confusion", {}).get("reason", ""),
                            "source": "streaming_error",
                            "slide_title": current_section_title or current_chapter_title or "",
                            "chapter_title": current_chapter_title or "",
                            "section_title": current_section_title or "",
                            "tts_engine": result.get("tts_engine", ""),
                            "tts_voice": result.get("tts_voice", ""),
                            "chapter_index": ctx.chapter_index if ctx else None,
                            "section_index": ctx.section_index if ctx else None,
                            "char_position": current_presentation_cursor if ctx else None,
                        })
                        if ctx:
                            await dialogue.transition(ctx.session_id, DialogState.CLARIFICATION)
                        await send_state(DialogState.CLARIFICATION, turn_id=turn_id)
                        # Puis revenir à LISTENING via set_listening_state()
                        ctx = await dialogue.get_session(session_id) or ctx  # ✅ Refresh ctx
                        await set_listening_state()
                        continue

                    # Check for voice navigation commands to bypass the LLM
                    try:
                        from audio.voice.command_parser import parse_voice_command
                        transcription_text = result.get("transcription", {}).get("text", "")
                        transcription_lang = result.get("transcription", {}).get("language", session_lang)
                        cmd = parse_voice_command(transcription_text, language=transcription_lang)
                    except Exception:
                        cmd = None

                    if cmd:
                        # parse_voice_command now returns either a real match
                        # or None — no confidence floor needed.
                        action = cmd.get("action")
                        log.info(f"[{session_id[:8]}] 🎚️ Voice command detected: {action} (text='{transcription_text}')")
                        # GOTO slide (numeric)
                        if action == "goto":
                            try:
                                slide_num = int(cmd.get("slide") or 0)
                                if slide_num > 0 and ctx and ctx.course_id is not None:
                                    target_idx = max(0, slide_num - 1)
                                    # Save server-side position
                                    try:
                                        await dialogue.save_course_position(
                                            ctx.session_id,
                                            course_id=ctx.course_id,
                                            chapter_index=ctx.chapter_index,
                                            section_index=target_idx,
                                            char_pos=0,
                                        )
                                    except Exception:
                                        pass
                                    # Load slide context and send update
                                    try:
                                        slide_ctx = await load_course_slide_context(ctx.course_id, ctx.chapter_index, target_idx)
                                    except Exception:
                                        slide_ctx = None
                                    await send({
                                        "type": "slide_update",
                                        "presentation_request_id": "",
                                        "slide_type": (slide_ctx or {}).get("slide_type", "section"),
                                        "slide_index": (slide_ctx or {}).get("slide_index", target_idx),
                                        "slide_title": (slide_ctx or {}).get("section_title") or (slide_ctx or {}).get("chapter_title", ""),
                                        "slide_content": (slide_ctx or {}).get("content", ""),
                                        "chapter": (slide_ctx or {}).get("chapter_title", ""),
                                        "chapter_index": (slide_ctx or {}).get("chapter_order", ctx.chapter_index if ctx else 0),
                                        "section_index": target_idx,
                                        "course_id": ctx.course_id if ctx else "",
                                    })
                                    try:
                                        await dialogue.transition(ctx.session_id, DialogState.PRESENTING)
                                    except Exception:
                                        pass
                                    await send_state(DialogState.PRESENTING, details={"slide_title": (slide_ctx or {}).get("section_title", "")})
                                    await set_listening_state()
                                    continue
                            except Exception:
                                pass
                        # Navigation: next / previous / repeat
                        if action == "next":
                            try:
                                if ctx:
                                    await dialogue.next_section(ctx.session_id)
                            except Exception:
                                pass
                            await send({"type": "next_section", "turn_id": turn_id})
                            await set_listening_state()
                            continue
                        if action == "previous":
                            try:
                                if ctx:
                                    await dialogue.prev_section(ctx.session_id)
                            except Exception:
                                pass
                            await send({"type": "prev_section", "turn_id": turn_id})
                            await set_listening_state()
                            continue
                        if action == "repeat":
                            try:
                                if ctx and ctx.course_id is not None:
                                    slide_ctx = await load_course_slide_context(ctx.course_id, ctx.chapter_index, ctx.section_index)
                                else:
                                    slide_ctx = None
                            except Exception:
                                slide_ctx = None
                            # Send slide metadata to client so it can re-present without LLM
                            await send({
                                "type": "slide_update",
                                "presentation_request_id": "",
                                "slide_type": (slide_ctx or {}).get("slide_type", "section"),
                                "slide_index": (slide_ctx or {}).get("slide_index", 0),
                                "slide_title": (slide_ctx or {}).get("section_title") or (slide_ctx or {}).get("chapter_title", ""),
                                "slide_content": (slide_ctx or {}).get("content", ""),
                                "content_original": (slide_ctx or {}).get("content", ""),
                                "image_url": (slide_ctx or {}).get("slide_path", ""),
                                "slide_path": (slide_ctx or {}).get("slide_path", ""),
                                "keywords": (slide_ctx or {}).get("keywords", []),
                                "chapter": (slide_ctx or {}).get("chapter_title", ""),
                                "chapter_index": (slide_ctx or {}).get("chapter_order", 0),
                                "section_index": (slide_ctx or {}).get("slide_index", 0),
                                "section_title": (slide_ctx or {}).get("section_title", ""),
                                "course_id": (slide_ctx or {}).get("course_id", ctx.course_id if ctx else ""),
                                "course_title": (slide_ctx or {}).get("course_title", ""),
                                "course_domain": (slide_ctx or {}).get("course_domain", "general"),
                                "progress_pct": (slide_ctx or {}).get("progress_pct", 0),
                            })
                            await set_listening_state()
                            continue
                        if action == "quiz":
                            # Reuse the quiz flow used for text input
                            try:
                                course_id = str(ctx.course_id if ctx and ctx.course_id else "")
                                current_chapter_idx = (ctx.chapter_index + 1) if ctx and ctx.chapter_index is not None else None
                                current_chapter_title = (ctx.chapter_title if ctx and getattr(ctx, 'chapter_title', None) else "")
                                current_section_title = (ctx.section_title if ctx and getattr(ctx, 'section_title', None) else "")
                                quiz_topic = current_section_title or current_chapter_title or "Quiz"
                                quiz_query = (slide_ctx.get("content") if (slide_ctx := (await load_course_slide_context(course_id, ctx.chapter_index, ctx.section_index) if ctx and ctx.course_id else None)) else current_section_title or current_chapter_title or course_id)
                                quiz_chunks = await asyncio.to_thread(
                                    rag.retrieve_chunks,
                                    quiz_query,
                                    k=Config.RAG_NUM_RESULTS,
                                    current_chapter_idx=current_chapter_idx,
                                    strict_chapter=bool(current_chapter_idx),
                                    course_id=course_id if course_id else None,
                                )
                                if slide_ctx:
                                    from langchain_core.documents import Document
                                    slide_doc = Document(page_content=slide_ctx.get("content",""), metadata={
                                        "course_id": course_id,
                                        "chapter_idx": current_chapter_idx,
                                        "chapter_title": current_chapter_title,
                                        "section_title": current_section_title,
                                        "slide_idx": slide_ctx.get("slide_index"),
                                        "source_file": slide_ctx.get("slide_path",""),
                                    })
                                    quiz_chunks = [(slide_doc, 1.0, f"Current slide: {current_chapter_title} / {current_section_title}")] + quiz_chunks
                                lang = result.get("transcription", {}).get("language", session_lang)
                                subj = detect_subject(transcription_text)
                                await process_quiz_request(
                                    quiz_topic=quiz_topic,
                                    lang=lang,
                                    subj=subj,
                                    chunks_with_scores=quiz_chunks,
                                    question_ctx=ctx,
                                    presentation_cursor=current_presentation_cursor,
                                    current_chapter_title=current_chapter_title,
                                    current_section_title=current_section_title,
                                    current_chapter_idx=current_chapter_idx,
                                    section_index_int=ctx.section_index if ctx else None,
                                    course_id=course_id,
                                    course_title=(slide_ctx or {}).get("course_title", ""),
                                    course_domain=(slide_ctx or {}).get("course_domain", "general"),
                                    slide_path=(slide_ctx or {}).get("slide_path", ""),
                                )
                            except Exception:
                                log.exception("Failed to run quiz via voice command")
                            await set_listening_state()
                            continue
                        if action == "explain":
                            # Treat explain as a repeat/present slide without calling LLM (bypass)
                            try:
                                if ctx and ctx.course_id is not None:
                                    slide_ctx = await load_course_slide_context(ctx.course_id, ctx.chapter_index, ctx.section_index)
                                else:
                                    slide_ctx = None
                                await send({
                                    "type": "slide_update",
                                    "presentation_request_id": "",
                                    "slide_type": (slide_ctx or {}).get("slide_type", "section"),
                                    "slide_index": (slide_ctx or {}).get("slide_index", 0),
                                    "slide_title": (slide_ctx or {}).get("section_title") or (slide_ctx or {}).get("chapter_title", ""),
                                    "slide_content": (slide_ctx or {}).get("content", ""),
                                })
                            except Exception:
                                log.exception("Voice explain failed")
                            await set_listening_state()
                            continue
                        if action == "pause":
                            try:
                                await record_pause_point("voice_pause", notice_prefix="⏸ Point d'arrêt mémorisé (voix)")
                            except Exception:
                                pass
                            try:
                                await cancel_presentation_task(notify_client=True)
                            except Exception:
                                pass
                            try:
                                await cancel_audio_stream(notify_client=True, turn_id=turn_id)
                            except Exception:
                                pass
                            try:
                                await cancel_text_question_task()
                            except Exception:
                                pass
                            if ctx:
                                try:
                                    await dialogue.transition(ctx.session_id, DialogState.LISTENING)
                                except Exception:
                                    pass
                            await send_state(DialogState.LISTENING)
                            continue
                        if action == "resume":
                            try:
                                resumed_ctx = await dialogue.resume_session(ctx.session_id)
                                ctx = resumed_ctx or ctx
                                await send_state(DialogState.PRESENTING, details={"slide_title": ctx and getattr(ctx, 'section_title', '') or ''})
                                await send({"type": "resume_course"})
                            except Exception:
                                log.exception("Failed to resume via voice command")
                            await set_listening_state()
                            continue
                        if action == "flag_confusion":
                            try:
                                await dialogue.mark_confusion_detected(ctx.session_id, reason=transcription_text)
                                await send({"type": "system_notice", "text": "Remarque enregistrée — nous passons en clarification."})
                                await send_state(DialogState.CLARIFICATION)
                            except Exception:
                                log.exception("Failed to flag confusion via voice command")
                            await set_listening_state()
                            continue

                    # Transition → RESPONDING (now streaming)
                    if ctx:
                        await dialogue.transition(ctx.session_id, DialogState.RESPONDING)
                    await send_state(
                        DialogState.RESPONDING,
                        turn_id=turn_id,
                        details={
                            "question_text": result.get("transcription", {}).get("text", ""),
                            "answer_preview": result.get("answer", "")[:160],
                            "subject": result.get("subject", ""),
                            "slide_title": current_section_title or current_chapter_title or "",
                            "chapter_title": current_chapter_title or "",
                            "section_title": current_section_title or "",
                            "tts_engine": result.get("tts_engine", ""),
                            "tts_voice": result.get("tts_voice", ""),
                        },
                    )

                    # Send final full answer text
                    if not await send({"type": "answer_text", "text": result["answer"],
                                "subject": result["subject"], "rag_chunks": result["rag_chunks"], "turn_id": turn_id,
                                "partial": False, "final": True}):
                        continue

                    # Send performance metrics
                    if not await send({"type": "performance", "turn_id": turn_id, **result["performance"]}):
                        continue

                    answer_audio_object = f"turns/answers/{session_id}/{turn_id}_{media_stamp}.{'mp3' if 'mpeg' in (response_audio_mime or '') else 'webm'}"
                    transcript_payload = {
                        "kind": "audio_question",
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "question_audio_path": question_audio_object,
                        "question_text": result.get("transcription", {}).get("text", ""),
                        "answer_text": result.get("answer", ""),
                        "language": result.get("transcription", {}).get("language", session_lang),
                        "subject": result.get("subject", ""),
                        "chapter_title": current_chapter_title or "",
                        "section_title": current_section_title or "",
                        "chapter_index": ctx.chapter_index if ctx else None,
                        "section_index": ctx.section_index if ctx else None,
                        "char_position": current_presentation_cursor if ctx else None,
                        "audio_answer_path": answer_audio_object if response_audio_parts else "",
                    }
                    await save_media_json(f"turns/transcripts/{session_id}/{turn_id}_{media_stamp}.json", transcript_payload)
                    if response_audio_parts:
                        await save_media_bytes(answer_audio_object, b"".join(response_audio_parts), response_audio_mime or "audio/mpeg")

                    try:
                        confusion_detected = bool(result.get("confusion", {}).get("detected", False))
                        total_turn_time = float(result["performance"].get("total_time", 0.0))
                        # Real confidence from the streaming pipeline
                        # (responder grounding score). Falls back to None
                        # when the streaming pipeline didn't surface one
                        # — `update_from_interaction` then ignores the
                        # confidence-based confusion signal.
                        stream_confidence = result.get("confidence")
                        stream_profile = await profile_mgr.update_from_interaction(
                            session_id,
                            "qa",
                            topic=(result.get("subject") or ""),
                            confused=confusion_detected,
                            response_time=total_turn_time,
                            confidence=float(stream_confidence) if stream_confidence is not None else None,
                            reward=1.0 if (not confusion_detected and result["performance"].get("kpi_ok", False)) else 0.0,
                            action_taken="reformulate" if confusion_detected else "answer",
                            course_id=ctx.course_id if ctx else None,
                        )
                        if stream_profile:
                            student_profile.update(stream_profile.to_dict())

                        await persist_learning_turn(
                            event_type="qa",
                            question_text=result.get("transcription", {}).get("text", ""),
                            answer_text=result.get("answer", ""),
                            language=result.get("transcription", {}).get("language", session_lang),
                            subject=result.get("subject", ""),
                            course_id_value=course_id_for_rag,
                            confusion_detected=confusion_detected,
                            confusion_reason=result.get("confusion", {}).get("reason", ""),
                            action_taken="reformulate" if confusion_detected else "answer",
                            reward=1.0 if (not confusion_detected and result["performance"].get("kpi_ok", False)) else 0.0,
                            stt_time=float(result["performance"].get("stt_time", 0.0)),
                            llm_time=float(result["performance"].get("llm_time", 0.0)),
                            tts_time=float(result["performance"].get("tts_time", 0.0)),
                            total_time=total_turn_time,
                            profile_snapshot=stream_profile.to_dict() if stream_profile else {},
                            concept=result.get("subject", ""),
                            chapter_index=ctx.chapter_index if ctx else None,
                            section_index=ctx.section_index if ctx else None,
                            char_position=current_presentation_cursor if ctx else None,
                            extra_payload={
                                "source": "streaming",
                                "rag_chunks": result.get("rag_chunks", 0),
                                "confusion_reason": result.get("confusion", {}).get("reason", ""),
                            },
                        )
                        # ✨ Bayes posterior update — voice question = strong auditory signal
                        from pedagogy.personalization.learning_style.bayes import fire_signal
                        fire_signal(ctx.student_id if ctx else None, "audio_question")
                    except Exception as learning_exc:
                        log.debug(f"[{session_id[:8]}] Streaming learning log skipped: {learning_exc}")
                    record_session_event({
                        "session_id": session_id,
                        "language": result.get("transcription", {}).get("language", session_lang),
                        "question": result.get("transcription", {}).get("text", ""),
                        "stt_text": result.get("transcription", {}).get("text", ""),
                        "answer": result.get("answer", ""),
                        "turn_id": turn_id,
                        "stt_time": float(result["performance"].get("stt_time", 0.0)),
                        "llm_time": float(result["performance"].get("llm_time", 0.0)),
                        "tts_time": float(result["performance"].get("tts_time", 0.0)),
                        "total_time": float(result["performance"].get("total_time", 0.0)),
                        "meets_kpi": bool(result["performance"].get("kpi_ok", False)),
                        "subject": result.get("subject", ""),
                        "confusion": confusion_detected,
                        "confusion_reason": result.get("confusion", {}).get("reason", ""),
                        "source": "streaming",
                        "slide_title": current_section_title or current_chapter_title or "",
                        "chapter_title": current_chapter_title or "",
                        "section_title": current_section_title or "",
                        "tts_engine": result.get("tts_engine", ""),
                        "tts_voice": result.get("tts_voice", ""),
                        "chapter_index": ctx.chapter_index if ctx else None,
                        "section_index": ctx.section_index if ctx else None,
                        "char_position": current_presentation_cursor if ctx else None,
                    })

                    await send_state(
                        DialogState.RESPONDING,
                        "response_complete",
                        {
                            "question_text": result.get("transcription", {}).get("text", ""),
                            "answer_preview": result.get("answer", "")[:160],
                            "subject": result.get("subject", ""),
                            "slide_title": current_section_title or current_chapter_title or "",
                            "chapter_title": current_chapter_title or "",
                            "section_title": current_section_title or "",
                            "tts_engine": result.get("tts_engine", ""),
                            "tts_voice": result.get("tts_voice", ""),
                        },
                        result.get("performance", {}),
                        turn_id=turn_id,
                    )
                    
                    # Signal stream completion
                    if not await send({"type": "audio_stream_end", "stream_id": stream_id, "turn_id": turn_id}):
                        continue

                    # ✅ CRITICAL: Attendre interruption ou auto-avancer
                    await asyncio.sleep(0.5)  # Petit pause après TTS
                    
                    user_interrupted = await handle_post_response(timeout_sec=2.5)
                    
                    if user_interrupted:
                        # ✅ Utilisateur a posé une question → écouter
                        log.info(f"💬 [{session_id[:8]}] Student asked a question")
                        # State est déjà LISTENING (detectable par VAD)
                    else:
                        # ✅ Pas de question → auto-avancer PRÉSENTATION
                        log.info(f"⏭️  [{session_id[:8]}] Auto-advancing to next slide")
                        if ctx:
                            await dialogue.transition(ctx.session_id, DialogState.PRESENTING)
                        await send({"type": "next_section", "turn_id": turn_id})
                        # Laisser le client gérer le chargement du slide suivant

                except Exception as pipeline_exc:
                    log.error(f"[{session_id[:8]}] Pipeline streaming error: {pipeline_exc}", exc_info=True)
                    await send({"type": "error", "message": f"Pipeline error: {str(pipeline_exc)[:100]}", "turn_id": turn_id})
                    await send_state(DialogState.LISTENING, turn_id=turn_id)

            # ── interrupt — l'étudiant coupe l'IA ─────────────────────
            elif msg_type == "interrupt":
                # Default changed from "question" to "pause" : when the
                # FE doesn't tag the reason explicitly, the safer
                # assumption is a manual pause (button click). Old
                # default mislabelled every manual pause as a question
                # in operator logs and downstream learning analytics.
                # The FE now always sends an explicit reason via
                # buildInterruptMsg() — this default just protects
                # against legacy clients.
                interrupt_reason = str(msg.get("reason") or "pause").strip().lower() or "pause"
                turn_id = int(msg.get("turn_id") or 0)
                # 📊 KPI #1 — start measuring interrupt latency (VAD detect)
                from observability.kpi_logger import KPITracker
                _kpi = KPITracker.get()
                _kpi.mark_interrupt_detected(session_id, turn_id)

                # ⛳ Diagnostic — print the FULL interrupt message so we
                # can verify whether the SDK actually sends ``audio_progress``.
                # If you see ``audio_progress=None`` here, the browser is
                # serving an old sdk.js (Ctrl+Shift+R to bypass cache).
                log.info(
                    f"[{session_id[:8]}] ⛳ interrupt msg | "
                    f"audio_progress={msg.get('audio_progress')!r} "
                    f"reason={interrupt_reason!r} "
                    f"current_cursor={current_presentation_cursor} "
                    f"current_text_len={len(current_presentation_text or '')}"
                )

                # ── Audio-progress cursor override ────────────────────
                # The frontend may include ``audio_progress`` in [0, 1]
                # measured from ``currentAudio.currentTime / duration``.
                # When present, it overrides ``current_presentation_cursor``
                # *before* record_pause_point reads it, so the saved
                # paused_state cursor reflects the REAL audio playback
                # position — not the text-streaming end position which
                # is always ``len(narration)`` once streaming finishes.
                # Without this, a pause near the end of a slide saves
                # cursor=len, and the legacy resume logic restarts the
                # whole slide. With this, the saved cursor matches what
                # the student actually heard.
                _audio_progress = msg.get("audio_progress")
                if _audio_progress is not None:
                    _override = _cursor_from_audio_progress(
                        _audio_progress, current_presentation_text or "",
                    )
                    if _override is not None:
                        log.info(
                            f"[{session_id[:8]}] ↪️  Interrupt cursor override "
                            f"{current_presentation_cursor} → {_override} "
                            f"(audio_progress={float(_audio_progress):.2f}, "
                            f"narration_len={len(current_presentation_text or '')})"
                        )
                        current_presentation_cursor = _override

                # ── Navigation reset ─────────────────────────────────
                # When the student clicked next/prev/repeat, they want
                # to GO TO a different slide (or restart the current
                # one), NOT resume the current slide where they paused.
                # We zero the cursor BEFORE record_pause_point so the
                # paused_state doesn't carry forward a stale "end-of-
                # previous-slide" cursor that the cache decision would
                # then apply to the destination slide.
                # Repeat re-presents the same slide from start.
                if interrupt_reason in ("navigation_next", "navigation_prev", "repeat"):
                    log.info(
                        f"[{session_id[:8]}] 🔁 navigation interrupt ({interrupt_reason}) "
                        f"— resetting cursor 0 (current_text length unchanged)"
                    )
                    current_presentation_cursor = 0

                await record_pause_point(interrupt_reason, notice_prefix="⏸ Point d'arrêt mémorisé")
                await cancel_presentation_task(notify_client=True)
                audio_buffer.clear()
                await cancel_audio_stream(notify_client=True, turn_id=turn_id)
                # 📊 KPI #1 — TTS effectively stopped here
                _interrupt_lat = _kpi.mark_tts_stopped(session_id, turn_id)
                if _interrupt_lat is not None:
                    log.info(f"[{session_id[:8]}] 📊 interrupt latency = {_interrupt_lat*1000:.0f}ms")
                    last_interrupt_latency_ms = _interrupt_lat * 1000.0
                # Track interaction for passive-reader detection
                interactions_on_current_slide += 1
                if interrupt_reason == "question":
                    questions_in_session += 1
                await cancel_text_question_task()
                if ctx and interrupt_reason != "pause":
                    await dialogue.transition(ctx.session_id, DialogState.LISTENING)
                    await send_state(DialogState.LISTENING, turn_id=turn_id)
                elif ctx:
                    await send_state(DialogState.WAITING, turn_id=turn_id)
                else:
                    await set_listening_state()
                log.info(f"[{session_id[:8]}] ⚡ Interruption")

            # ── present_section — présenter une section de cours ─────
            elif msg_type == "present_section":
                content_txt = (msg.get("slide_content") or msg.get("content") or "").strip()
                presentation_request_id = str(msg.get("presentation_request_id") or "").strip()
                slide_title = msg.get("slide_title", "")
                slide_content = msg.get("slide_content", "")
                keywords = msg.get("keywords", [])
                chapter = msg.get("chapter", "")
                section_title = msg.get("section_title", "")
                progress_pct = msg.get("progress_pct", 0)
                # The COURSE's language wins for the narration. The
                # frontend's ``present_section`` carries the course's
                # language attribute (filled from the courses table at
                # the time the student selected the course). This is
                # the AUTHORITATIVE language for the narration : a
                # course uploaded in French stays in French regardless
                # of which student opens it. Falls back to the WS
                # session default for legacy SDKs that don't ship a
                # language on the message.
                lang_ps = msg.get("language") or session_lang or "fr"
                course_id = str(msg.get("course_id") or (ctx.course_id if ctx and ctx.course_id else "") or "").strip()
                course_title = msg.get("course_title", "")
                course_domain = msg.get("course_domain", "general")
                chapter_index_raw = msg.get("chapter_index")
                section_index_raw = msg.get("section_index")
                slide_index_raw = msg.get("slide_index")
                slide_path = str(msg.get("slide_path") or msg.get("image_url") or "").strip()
                slide_type = msg.get("slide_type", "section")
                in_course = True

                # ── Course-change observability ──────────────────────
                # We DON'T clear history here — the user wants exchanges
                # to persist across course switches. Cross-course
                # contamination is mitigated upstream by :
                #   1. retriever.py filters chunks by ``course_id``
                #   2. responder course_bound_rule + <<INSTRUCTION_INTERNE>>
                #   3. off-topic guardrail rejecting < 8% lexical overlap
                # Just log the transition so operators can correlate any
                # weird answer with a recent course change.
                if course_id and last_active_course_id and course_id != last_active_course_id:
                    log.info(
                        f"[{session_id[:8]}] 🔄 Course change "
                        f"({last_active_course_id} → {course_id}) — history preserved"
                    )
                if course_id:
                    last_active_course_id = course_id

                # ── Course → language + level lookup ──────────────────
                # The COURSE itself owns the narration language and the
                # academic level. Both were set at upload time by the
                # teacher and live on the ``courses`` table. We refresh
                # session_lang / session_level from the course row so
                # the narration adopts the course's settings, regardless
                # of the student's profile preference. The lookup is
                # done once per (course change) and cached via the
                # last_active_course_id sentinel above.
                if course_id and last_active_course_id == course_id:
                    try:
                        import uuid as _uuid_mod
                        from sqlalchemy import select as _sql_select
                        from database.init_db import AsyncSessionLocal
                        from database.models import Course
                        async with AsyncSessionLocal() as _db:
                            try:
                                _course_uuid = _uuid_mod.UUID(course_id)
                            except (ValueError, TypeError):
                                _course_uuid = None
                            if _course_uuid is not None:
                                _course_row = (await _db.execute(
                                    _sql_select(Course).where(Course.id == _course_uuid)
                                )).scalar_one_or_none()
                                if _course_row:
                                    if _course_row.language:
                                        session_lang = _course_row.language
                                    if _course_row.level:
                                        session_level = _course_row.level
                                    log.info(
                                        f"[{session_id[:8]}] 📚 course settings | "
                                        f"language={session_lang} level={session_level} "
                                        f"(from courses.id={course_id[:8]})"
                                    )
                    except Exception as _course_exc:                       # noqa: BLE001
                        log.debug(f"[{session_id[:8]}] course lookup skipped: {_course_exc}")

                try:
                    chapter_index_int = int(chapter_index_raw) if chapter_index_raw is not None else 0
                except (TypeError, ValueError):
                    chapter_index_int = 0
                try:
                    section_index_int = int(section_index_raw) if section_index_raw is not None else 0
                except (TypeError, ValueError):
                    section_index_int = 0

                # ── Passive-reader detection ─────────────────────────
                # When the student moves to a NEW slide (key change),
                # check whether the PREVIOUS slide received any
                # interaction (pause, question, manual nav). If not,
                # increment the consecutive-passive counter. Reaching
                # ≥ 3 is a strong "lost the student" signal that the
                # engagement scorer reads to push toward "disengaged".
                # MUST run AFTER chapter_index_int / section_index_int
                # are parsed — the detection key uses them.
                _new_slide_key = (course_id or "", chapter_index_int, section_index_int)
                if last_presented_slide_key is not None and last_presented_slide_key != _new_slide_key:
                    if interactions_on_current_slide == 0:
                        consecutive_passive_slides += 1
                        log.info(
                            f"[{session_id[:8]}] 📉 passive slide "
                            f"({consecutive_passive_slides} consecutive without interaction)"
                        )
                    else:
                        if consecutive_passive_slides > 0:
                            log.info(
                                f"[{session_id[:8]}] 📈 student re-engaged after "
                                f"{consecutive_passive_slides} passive slides"
                            )
                        consecutive_passive_slides = 0
                    interactions_on_current_slide = 0
                last_presented_slide_key = _new_slide_key
                try:
                    slide_index_int = int(slide_index_raw) if slide_index_raw is not None else 0
                except (TypeError, ValueError):
                    slide_index_int = 0

                slide_ctx = None
                if course_id:
                    slide_ctx = await load_course_slide_context(course_id, chapter_index_int, section_index_int)
                if slide_ctx:
                    content_txt = (slide_ctx.get("content") or content_txt).strip()
                    slide_title = slide_ctx.get("section_title") or slide_title
                    slide_content = slide_ctx.get("content") or slide_content
                    keywords = slide_ctx.get("keywords") or keywords
                    chapter = slide_ctx.get("chapter_title") or chapter
                    section_title = slide_ctx.get("section_title") or section_title
                    course_title = slide_ctx.get("course_title") or course_title
                    course_domain = slide_ctx.get("course_domain") or course_domain
                    slide_path = slide_ctx.get("slide_path") or slide_path
                    slide_type = slide_ctx.get("slide_type") or slide_type
                    slide_index_int = int(slide_ctx.get("slide_index") or slide_index_int)
                    progress_pct = slide_ctx.get("progress_pct") if slide_ctx.get("progress_pct") is not None else progress_pct

                if not content_txt:
                    continue

                await cancel_presentation_task(notify_client=False)
                await cancel_audio_stream(notify_client=False)

                if ctx:
                    # ✅ Stocker le course_summary pour utilisation dans explain_slide_focused
                    if slide_ctx and slide_ctx.get("course_summary"):
                        ctx.course_summary = slide_ctx["course_summary"]
                        ctx.course_analysis = slide_ctx.get("course_analysis", {})
                    
                    await dialogue.save_course_position(
                        ctx.session_id,
                        course_id=course_id or None,
                        chapter_index=chapter_index_int,
                        section_index=section_index_int,
                        char_pos=0,
                    )
                    await dialogue.transition(ctx.session_id, DialogState.PRESENTING)
                presentation_details = {
                    "course_title": course_title or "",
                    "chapter_title": chapter or "",
                    "section_title": section_title or "",
                    "slide_title": section_title or chapter or "",
                    "slide_index": slide_index_int,
                    "progress_pct": progress_pct,
                }
                await send_state(DialogState.PRESENTING, details=presentation_details)

                requested_slide_key = (course_id or "", chapter_index_int, section_index_int)
                requested_slide_id = ":".join(str(part) for part in requested_slide_key)
                cached_snapshot = None
                if ctx:
                    try:
                        cached_snapshot = await dialogue.load_presentation_snapshot(ctx.session_id, requested_slide_id)
                    except Exception as cache_exc:
                        log.debug(f"[{session_id[:8]}] Presentation cache lookup skipped: {cache_exc}")
                cached_pause_state = ctx.paused_state if ctx else {}

                # Pure decision lives in services.presentation so it's
                # unit-tested without a full WS scaffold. See
                # ``decide_narration_cache_reuse`` for the source ordering :
                # in-memory > Redis snapshot for THIS slide > paused-state
                # match. This replaces the previous inlined logic which
                # had a bug : the Redis snapshot was ignored unless the
                # currently paused slide also matched the requested one
                # (false on every back-navigation through previously-seen
                # slides — the LLM regenerated 60-300s narrations the
                # student had already heard).
                (
                    reuse_cached_narration,
                    cache_source,
                    _cached_text,
                    _cached_cursor,
                ) = _decide_narration_cache_reuse(
                    requested_slide_key=requested_slide_key,
                    current_presentation_key=current_presentation_key,
                    current_presentation_text=current_presentation_text,
                    cached_snapshot=cached_snapshot,
                    cached_pause_state=cached_pause_state,
                )

                if reuse_cached_narration:
                    log.info(
                        f"[{session_id[:8]}] 💾 Narration cache HIT "
                        f"(source={cache_source}, slide={requested_slide_id}, "
                        f"text_len={len(_cached_text)}, resume_cursor={_cached_cursor}) → 0 LLM call"
                    )
                    # The decision function returns the exact text + cursor
                    # to use, already clamped. Just adopt them.
                    current_presentation_text = _cached_text
                    current_presentation_cursor = _cached_cursor
                else:
                    log.info(
                        f"[{session_id[:8]}] 🔄 Narration cache MISS (slide={requested_slide_id}) → Teaching Graph va générer"
                    )
                    current_presentation_text = ""
                    current_presentation_cursor = 0
                current_presentation_key = requested_slide_key
                resume_offset = current_presentation_cursor if reuse_cached_narration else 0
                # Operator-visible : on every resume we want to know the
                # cursor we're starting from. The forward-skip log a few
                # lines down only fires when offset > 0; this one fires
                # always, so a "still 0" cursor is now spotted at a glance.
                log.info(
                    f"[{session_id[:8]}] ↪️  Resume cursor : raw={resume_offset} "
                    f"(narration_len={len(current_presentation_text)}, "
                    f"cache_source={cache_source}, "
                    f"reuse_cached={reuse_cached_narration})"
                )

                # Envoyer la slide immédiatement avec ses métadonnées exactes.
                log.info(
                    "🔍 slide sync | type=%s ch=%s sec=%s title=%r progress=%s%%",
                    slide_type, chapter_index_int, section_index_int,
                    (section_title or chapter or content_txt[:50])[:80],
                    progress_pct,
                )
                if not await send({
                    "type": "slide_update",
                    "presentation_request_id": presentation_request_id,
                    "slide_type": slide_type,
                    "slide_index": slide_index_int,
                    "slide_title": slide_title or content_txt[:60],
                    "slide_content": slide_content or content_txt[:200],
                    "content_original": content_txt,
                    "image_url": slide_path,
                    "slide_path": slide_path,
                    "keywords": keywords,
                    "chapter": chapter,
                    "chapter_index": chapter_index_int,
                    "section_index": section_index_int,
                    "section_title": section_title,
                    "course_id": course_id,
                    "course_title": course_title,
                    "course_domain": course_domain,
                    "progress_pct": progress_pct,
                }):
                    continue

                # Profil étudiant → adapter le débit (TODO: implement speech rate customization)
                try:
                    profile = await profile_mgr.get_or_create(session_id, lang_ps, session_level)
                    # ✅ Stocker profil pour timing adaptatif
                    student_profile.update({
                        "confusion_count": profile.confusion_count,
                        "asks_repeat": profile.asks_repeat,
                    })
                    # TODO: Implement speech rate adaptation via TTS engine
                    # For now, using default speech rate
                    rate_override = "+0%"
                except Exception:
                    rate_override = "+0%"

                async def run_presentation(
                    section_text: str,
                    resume_from: int,
                    reuse_cached: bool,
                ) -> None:
                    nonlocal current_stream_id, current_presentation_text, current_presentation_cursor, presentation_task
                    # ⛳ Marker log : confirms this version of run_presentation
                    # is loaded. If you don't see "RP-v2" on a resume, the
                    # server is still running the old code and needs a
                    # FULL restart (not uvicorn --reload, which doesn't
                    # always pick up changes inside nested closures).
                    log.info(
                        f"[{session_id[:8]}] ⛳ RP-v2 enter | "
                        f"resume_from={resume_from} reuse_cached={reuse_cached} "
                        f"current_text_len={len(current_presentation_text or '')}"
                    )
                    try:
                        narration_text = current_presentation_text if reuse_cached and current_presentation_text else ""
                        if not narration_text:
                            llm_start = time.time()
                            await send_state(
                                DialogState.PRESENTING,
                                "llm_thinking",
                                {"type": "presentation", **presentation_details},
                                {"progress_pct": 20},
                            )
                            try:
                                # ✅ Déterminer si c'est une reprise (pause/resume)
                                is_resuming = resume_offset > 0 and reuse_cached

                                # ✅ Récupérer le course_summary depuis le contexte session
                                course_summary = ctx.course_summary if ctx else ""

                                # Callback to push the plan to the frontend as soon as Planner is done
                                async def _push_plan(plan_obj):
                                    try:
                                        ideas_payload = []
                                        for idea in (getattr(plan_obj, "ideas", None) or []):
                                            ideas_payload.append({
                                                "id": idea.id,
                                                "type": idea.type,
                                                "depth": idea.depth,
                                                "content_brief": idea.content_brief,
                                            })
                                        try:
                                            from observability.dashboard import record_checkpoint_event as _rec
                                            _rec({
                                                "session_id": session_id or "",
                                                "slide_id": requested_slide_id,
                                                "chapter_idx": chapter_index_int,
                                                "section_idx": section_index_int,
                                                "chapter_title": getattr(plan_obj, "chapter_title", chapter),
                                                "section_title": getattr(plan_obj, "section_title", section_title),
                                                "ideas": ideas_payload,
                                                "source": "presentation_plan",
                                            })
                                            log.info(f"[{session_id[:8]}] 📋 Plan enregistré dans dashboard ({len(ideas_payload)} idées)")
                                        except Exception as exc:
                                            log.debug(f"[{session_id[:8]}] presentation_plan record failed: {exc}")
                                    except Exception as exc:
                                        log.debug(f"[{session_id[:8]}] presentation_plan send failed: {exc}")

                                # ✅ NOUVEAU: Utiliser explain_slide_focused (strict + focused + level-adapted)
                                narration_text = await explain_slide_focused(
                                    slide_content=section_text or slide_content or content_txt,
                                    chapter_idx=chapter_index_int,
                                    chapter_title=chapter,
                                    section_title=section_title,
                                    language=lang_ps,
                                    student_level=session_level,
                                    course_summary=course_summary,  # ✅ NOUVEAU
                                    is_resume=is_resuming,  # ✅ NOUVEAU
                                    session_id=session_id,  # ✅ Pass session_id for rate limiting
                                    course_id=course_id or "",
                                    section_idx=section_index_int,
                                    on_plan_ready=_push_plan,
                                    # Slide image → vision concept extraction (planner)
                                    slide_image_path=slide_path or "",
                                )
                            except Exception:
                                # Fallback: brain.ask
                                narration_text, _ = await asyncio.to_thread(
                                    brain.ask,
                                    section_text or content_txt,
                                    reply_language=lang_ps,
                                    session_id=session_id,
                                )  # ✅ Pass session_id
                                narration_text = brain._clean_for_speech(narration_text)
                            narration_text = narration_text.strip()
                            current_presentation_text = narration_text
                            llm_time = time.time() - llm_start
                            if ctx and requested_slide_id:
                                try:
                                    await dialogue.save_presentation_snapshot(
                                        ctx.session_id,
                                        requested_slide_id,
                                        narration_text,
                                        presentation_cursor=resume_from,
                                        slide_title=section_title or chapter or "",
                                    )
                                except Exception as cache_exc:
                                    log.debug(f"[{session_id[:8]}] Presentation snapshot save skipped: {cache_exc}")

                            await schedule_next_slide_prefetch(
                                current_slide_key=requested_slide_key,
                                course_id_value=course_id or "",
                                chapter_index_value=chapter_index_int,
                                section_index_value=section_index_int,
                                language_code=lang_ps,
                                student_level_value=session_level,
                                course_summary_value=ctx.course_summary if ctx else "",
                                rate_value=rate_override,
                                current_slide_title=section_title or chapter or "",
                            )
                            await send_state(
                                DialogState.PRESENTING,
                                "tts_generating",
                                {"engine": "presentation", **presentation_details},
                                {
                                    "llm_time": round(llm_time, 2),
                                    "tokens": len(narration_text.split()),
                                    "progress_pct": 55,
                                },
                            )
                        else:
                            llm_time = 0.0

                        if not narration_text:
                            if await send({"type": "stream_end", "stream_id": current_stream_id}):
                                await set_listening_state()
                            return

                        # Compute effective resume cursor.
                        #
                        # Why this is subtle : the backend sets
                        # ``current_presentation_cursor = len(narration_text)``
                        # once text streaming finishes (line ~3228 below),
                        # but at that moment the TTS audio is often STILL
                        # PLAYING on the client. If the student pauses
                        # at that instant the saved cursor is ``len``
                        # (text done) even though the audio isn't done.
                        #
                        # The legacy code did ``cursor < len else 0``,
                        # restarting the whole slide from char 0 — the
                        # operator-visible 3x repetition bug.
                        #
                        # Fix : when ``cursor >= len`` we rewind to the
                        # start of the LAST sentence and replay just
                        # that sentence. Net effect : the student hears
                        # the end of the slide one more time (continuity
                        # if they hadn't heard it yet, mild repetition
                        # if they had — much less painful than 837-char
                        # full re-play).
                        # ── Pedagogical resume intelligence ────────────
                        # Replaces the old "always rewind to last sentence
                        # on cursor==len" mechanical replay with a 6-intent
                        # / 5-strategy decision. Inputs : cursor position,
                        # narration length, interruption duration. Outputs
                        # : strategy ∈ {CONTINUE, REWIND_SENTENCE,
                        # REWIND_SENTENCE_RECAP, REWIND_SENTENCE_LONG_RECAP,
                        # SKIP_TO_NEXT}.
                        # Pure logic lives in pedagogy.resume_intelligence
                        # so it's unit-tested in isolation.
                        _interruption_dur_s: float | None = None
                        if ctx and ctx.paused_state:
                            _ts = ctx.paused_state.get("timestamp")
                            if isinstance(_ts, (int, float)) and _ts > 0:
                                _interruption_dur_s = max(0.0, time.time() - float(_ts))

                        resume_action = _compose_resume_action(_ResumeCtx(
                            cursor=resume_from,
                            narration_len=len(narration_text),
                            interruption_duration_s=_interruption_dur_s,
                        ))
                        log.info(
                            f"[{session_id[:8]}] 🧠 resume intelligence | {resume_action.reason}"
                        )

                        # Apply the strategy → produces (effective_resume,
                        # remaining_text, resume_recap, [skip_to_next flag]).
                        from services.presentation import (
                            rewind_to_current_sentence_start,
                            build_resume_recap,
                        )

                        resume_recap = ""
                        skip_to_next_slide = False
                        new_start = 0

                        strat = resume_action.strategy

                        if strat == _ResumeStrategy.SKIP_TO_NEXT:
                            # Slide truly finished, student let it run to
                            # the end and didn't linger. Don't replay —
                            # signal stream end + return to listening.
                            # The FE can request the next slide on its
                            # next user action.
                            log.info(
                                f"[{session_id[:8]}] ⏭️  Slide done — skipping replay"
                            )
                            current_presentation_cursor = len(narration_text)
                            if ctx:
                                await dialogue.save_position(ctx.session_id, current_presentation_cursor)
                            if not await send({"type": "stream_end", "stream_id": current_stream_id}):
                                return
                            await set_listening_state()
                            return

                        if strat == _ResumeStrategy.CONTINUE:
                            # Quick pause (<10s, mid-narration) — resume
                            # from exact cursor, no recap, no rewind.
                            effective_resume = max(0, min(resume_from, len(narration_text)))
                            new_start = effective_resume
                            remaining_text = narration_text[effective_resume:]

                        elif strat == _ResumeStrategy.REEXPLAIN_AND_CONTINUE:
                            # Medium / long pause — student likely lost
                            # the thread of the idea. Call the LLM to
                            # rephrase the CURRENT sentence in different
                            # words, then continue with the cached
                            # remainder. Costs one extra LLM call (10-60s
                            # on Ollama) ; we wait synchronously because
                            # the student already paused, the latency is
                            # bounded by the pause duration.
                            from services.presentation import current_sentence_span
                            from agentic.qa._shared import format_history as _fmt_hist  # noqa: F401
                            if resume_from >= len(narration_text) > 0:
                                _re_cursor = _rewind_to_last_sentence_start(narration_text)
                            else:
                                _re_cursor = max(0, resume_from)
                            sent_start, sent_end, sent_text = current_sentence_span(
                                narration_text, _re_cursor,
                            )
                            new_start = sent_end
                            # Continue AFTER the sentence we're going
                            # to re-explain — avoids the student hearing
                            # the same idea twice (re-explained + verbatim).
                            remaining_text = narration_text[sent_end:]
                            effective_resume = sent_start

                            # Build a prompt asking for a 1-2 sentence
                            # rephrasing. Strict on : same idea, simpler
                            # words, no new info, same language.
                            if lang_ps == "fr":
                                _reex_prompt = (
                                    "Tu es un professeur. Reformule UNE SEULE FOIS "
                                    "l'idée suivante en 1 ou 2 phrases plus simples, "
                                    "avec d'autres mots. N'ajoute AUCUNE information "
                                    "qui n'est pas dans la phrase ; n'invente PAS "
                                    "de nom propre. Commence par \"Reprenons : \" et "
                                    "termine par une phrase nette qui mène à la suite.\n\n"
                                    f"PHRASE À REFORMULER : {sent_text}\n\n"
                                    "REFORMULATION :"
                                )
                            else:
                                _reex_prompt = (
                                    "You are a teacher. Rephrase the following idea "
                                    "ONCE in 1-2 simpler sentences, in different words. "
                                    "Do NOT add any information that's not in the "
                                    "sentence ; do NOT invent proper names. Start with "
                                    "\"Let's revisit: \" and end with a clean sentence "
                                    "leading into the next idea.\n\n"
                                    f"SENTENCE TO REPHRASE: {sent_text}\n\n"
                                    "REPHRASED:"
                                )
                            try:
                                _reex_text, _ = await asyncio.to_thread(
                                    brain.ask, _reex_prompt,
                                    reply_language=lang_ps,
                                    session_id=session_id,
                                )
                                _reex_text = (_reex_text or "").strip()
                                # Strip greeting + paren duplicates so the
                                # re-explanation respects the same rules
                                # as the main narration.
                                if _reex_text:
                                    from agentic.teaching.narrator import (
                                        _strip_greeting_opener,
                                        _strip_parenthetical_duplicates,
                                    )
                                    _stripped, _ = _strip_greeting_opener(_reex_text, lang_ps)
                                    _reex_text, _ = _strip_parenthetical_duplicates(_stripped)
                                resume_recap = _reex_text or build_resume_recap(
                                    narration_text, effective_resume, language=lang_ps,
                                    interruption_duration_s=300.0,
                                )
                                log.info(
                                    f"[{session_id[:8]}] 🔁 reexplain | "
                                    f"sentence_span=[{sent_start}:{sent_end}] "
                                    f"({sent_end - sent_start} chars) → "
                                    f"reexplained={len(_reex_text)} chars"
                                )
                            except Exception as _reex_exc:                # noqa: BLE001
                                # LLM unavailable / rate-limited → fall
                                # back to a plain recap of the cached
                                # sentence. Never block the user flow.
                                log.warning(
                                    f"[{session_id[:8]}] reexplain failed ({_reex_exc}) "
                                    f"→ falling back to cached sentence"
                                )
                                remaining_text = narration_text[sent_start:]
                                new_start = sent_start
                                resume_recap = build_resume_recap(
                                    narration_text, effective_resume, language=lang_ps,
                                    interruption_duration_s=60.0,
                                )

                            log.info(
                                f"[{session_id[:8]}] ↩️  resume strategy={strat.value} | "
                                f"sent_span=[{sent_start}:{sent_end}] | "
                                f"remaining={len(remaining_text)}/{len(narration_text)} chars | "
                                f"recap={'reexplained' if resume_recap else 'fallback'}"
                            )

                        else:
                            # All rewind strategies share the same cursor
                            # math : rewind to start of current sentence.
                            # If cursor was at end (REVIEW_REQUEST), we
                            # first rewind to the last sentence start.
                            if resume_from >= len(narration_text) > 0:
                                effective_resume = _rewind_to_last_sentence_start(narration_text)
                            else:
                                effective_resume = max(0, resume_from)

                            sentence_start = rewind_to_current_sentence_start(
                                narration_text, effective_resume,
                            )
                            remaining_text = narration_text[sentence_start:]
                            new_start = sentence_start

                            # Recap depth depends on the strategy. Force a
                            # specific bucket via interruption_duration_s :
                            #   REWIND_SENTENCE             → no recap (just sentence replay)
                            #   REWIND_SENTENCE_RECAP       → medium ("On était sur X.")
                            #   REWIND_SENTENCE_LONG_RECAP  → full recap
                            if strat == _ResumeStrategy.REWIND_SENTENCE:
                                # Quick "Je continue." opener
                                resume_recap = build_resume_recap(
                                    narration_text, effective_resume, language=lang_ps,
                                    interruption_duration_s=5.0,
                                )
                            elif strat == _ResumeStrategy.REWIND_SENTENCE_RECAP:
                                # Medium recap "On était sur X."
                                resume_recap = build_resume_recap(
                                    narration_text, effective_resume, language=lang_ps,
                                    interruption_duration_s=60.0,
                                )
                            elif strat == _ResumeStrategy.REWIND_SENTENCE_LONG_RECAP:
                                # Full recap of the last full sentence
                                resume_recap = build_resume_recap(
                                    narration_text, effective_resume, language=lang_ps,
                                    interruption_duration_s=300.0,
                                )

                            log.info(
                                f"[{session_id[:8]}] ↩️  resume strategy={strat.value} | "
                                f"cursor={resume_from}→sentence_start={new_start} | "
                                f"remaining={len(remaining_text)}/{len(narration_text)} chars | "
                                f"recap={'yes' if resume_recap else 'no'}"
                            )
                        if not remaining_text.strip():
                            current_presentation_cursor = len(narration_text)
                            if ctx:
                                await dialogue.save_position(ctx.session_id, current_presentation_cursor)
                            if not await send({"type": "answer_text", "text": narration_text, "subject": "course", "presentation_request_id": presentation_request_id, "final": True}):
                                return
                            if not await send({"type": "stream_end", "stream_id": current_stream_id}):
                                return
                            await set_listening_state()
                            return

                        current_stream_id += 1
                        stream_id = current_stream_id
                        sentences = split_sentences_with_spans(remaining_text)
                        tts_total_time = 0.0
                        await send_state(
                            DialogState.PRESENTING,
                            "tts_streaming",
                            {"chunks": len(sentences), **presentation_details},
                            {
                                "tts_chunks": len(sentences),
                                "progress_pct": 70,
                            },
                        )

                        # ── Pedagogical resume recap ─────────────────────
                        # Played BEFORE the actual narration resumes. Does
                        # NOT advance current_presentation_cursor — that
                        # cursor tracks position in the narration only.
                        if resume_recap:
                            try:
                                await send({
                                    "type": "answer_text",
                                    "text": resume_recap,
                                    "subject": "course",
                                    "presentation_request_id": presentation_request_id,
                                    "partial": True,
                                    "is_recap": True,
                                })
                                recap_audio, _rt, _, _, recap_mime, _ = await synthesize_cached_tts(
                                    resume_recap,
                                    language_code=lang_ps,
                                    rate=rate_override,
                                    cache_scope="presentation",
                                )
                                if recap_audio:
                                    await _stream_audio(recap_audio, recap_mime, stream_id)
                            except Exception as recap_exc:
                                log.debug(
                                    f"[{session_id[:8]}] resume recap TTS skipped: {recap_exc}"
                                )

                        for sentence, start, end in sentences:
                            if not sentence.strip():
                                continue

                            if not await send({"type": "answer_text", "text": sentence, "subject": "course", "presentation_request_id": presentation_request_id, "partial": True}):
                                return

                            audio_chunk = None
                            tts_piece_time = 0.0
                            mime = None
                            try:
                                audio_chunk, tts_piece_time, _, _, mime, _cache_hit = await synthesize_cached_tts(
                                    sentence,
                                    language_code=lang_ps,
                                    rate=rate_override,
                                    cache_scope="presentation",
                                )
                            except TypeError:
                                audio_chunk, tts_piece_time, _, _, mime, _cache_hit = await synthesize_cached_tts(
                                    sentence,
                                    language_code=lang_ps,
                                    rate="+0%",
                                    cache_scope="presentation",
                                )

                            tts_total_time += tts_piece_time or 0.0

                            if audio_chunk:
                                await _stream_audio(audio_chunk, mime, stream_id)

                            # Cursor in the original narration_text. ``end`` is the
                            # offset within ``remaining_text`` (which starts at
                            # ``new_start`` in the original). Earlier this used
                            # ``effective_resume + end`` which overstated progress
                            # whenever the forward-skip moved past the partial
                            # sentence (new_start > effective_resume).
                            cursor_base = new_start if effective_resume > 0 else effective_resume
                            current_presentation_cursor = min(len(narration_text), cursor_base + end)
                            if ctx:
                                await dialogue.save_position(ctx.session_id, current_presentation_cursor)

                            if ctx and requested_slide_id:
                                try:
                                    await dialogue.save_presentation_snapshot(
                                        ctx.session_id,
                                        requested_slide_id,
                                        narration_text,
                                        presentation_cursor=current_presentation_cursor,
                                        slide_title=section_title or chapter or "",
                                    )
                                except Exception as cache_exc:
                                    log.debug(f"[{session_id[:8]}] Presentation snapshot refresh skipped: {cache_exc}")

                            await asyncio.sleep(0)

                        current_presentation_cursor = len(narration_text)
                        if ctx:
                            await dialogue.save_position(ctx.session_id, current_presentation_cursor)

                        if ctx and requested_slide_id:
                            try:
                                await dialogue.save_presentation_snapshot(
                                    ctx.session_id,
                                    requested_slide_id,
                                    narration_text,
                                    presentation_cursor=current_presentation_cursor,
                                    slide_title=section_title or chapter or "",
                                )
                            except Exception as cache_exc:
                                log.debug(f"[{session_id[:8]}] Presentation snapshot final save skipped: {cache_exc}")

                        await send_state(
                            DialogState.PRESENTING,
                            "response_complete",
                            presentation_details,
                            {
                                "llm_time": round(llm_time, 2),
                                "tts_time": round(tts_total_time, 2),
                                "total_time": round(llm_time + tts_total_time, 2),
                                "tokens": len(narration_text.split()),
                                "words": len(narration_text.split()),
                                "sentences": len(sentences),
                                "progress_pct": 95,
                            },
                        )

                        if not await send({"type": "answer_text", "text": narration_text, "subject": "course", "presentation_request_id": presentation_request_id, "final": True}):
                            return
                        if not await send({"type": "stream_end", "stream_id": stream_id}):
                            return
                        await set_listening_state()
                    except asyncio.CancelledError:
                        if ctx:
                            try:
                                await dialogue.save_position(ctx.session_id, current_presentation_cursor)
                            except Exception:
                                pass
                        raise
                    except Exception as e:
                        log.error(f"Erreur présentation streaming: {e}")
                        await send({"type": "error", "message": f"Erreur présentation: {str(e)}"})
                    finally:
                        nonlocal_presentation_task = presentation_task
                        if nonlocal_presentation_task is asyncio.current_task():
                            presentation_task = None

                # ✅ NOUVEAU: Enregistrer le moment où la présentation COMMENCE
                presentation_start_time = time.time()
                presentation_task = asyncio.create_task(
                    run_presentation(content_txt, resume_offset, reuse_cached_narration)
                )

            # ── text — question texte (depuis l'input HTML) ────────────
            elif msg_type == "text":
                # ✅ REFRESH ctx (peut être modifié par run_presentation)
                if ctx:
                    ctx = await dialogue.get_session(session_id) or ctx
                
                content = msg.get("content", "").strip()
                if not content:
                    continue
                await cancel_audio_stream(notify_client=False)

                content_lower = " ".join(content.lower().split()).strip(" .!?,:;")
                resume_triggers = (
                    "continue",
                    "continuer",
                    "reprendre",
                    "reprends",
                    "reprise",
                    "poursuis",
                    "poursuivre",
                    "go on",
                    "resume",
                    "on continue",
                    "continue le cours",
                    "continuer le cours",
                    "reprendre le cours",
                    "reprends le cours",
                    "on reprend",
                    "reprenons",
                )
                quiz_triggers = (
                    "quiz",
                    "quiz me",
                    "quiz moi",
                    "quiz-moi",
                    "quizz",
                    "test me",
                    "teste-moi",
                    "teste moi",
                    "donne-moi un quiz",
                    "donne moi un quiz",
                    "fais-moi un quiz",
                    "fais moi un quiz",
                    "interroge-moi",
                    "interroge moi",
                )
                quiz_phrase_triggers = (
                    "can do quiz",
                    "do a quiz",
                    "do quiz",
                    "make a quiz",
                    "create a quiz",
                    "generate a quiz",
                    "give me a quiz",
                    "quiz about this course",
                    "quiz on this course",
                    "quiz on the course",
                    "test me on this course",
                    "test me on the course",
                    "can you quiz",
                    "quiz me on",
                    "faire un quiz",
                    "fais un quiz",
                    "donne moi un quiz",
                    "donne-moi un quiz",
                    "cree un quiz",
                    "crée un quiz",
                    "generer un quiz",
                    "générer un quiz",
                    "teste moi sur ce cours",
                    "teste-moi sur ce cours",
                    "interroge moi sur ce cours",
                    "interroge-moi sur ce cours",
                )

                def is_quiz_intent(normalized_text: str) -> bool:
                    if normalized_text in quiz_triggers:
                        return True
                    if any(normalized_text.startswith(trigger + " ") for trigger in quiz_triggers):
                        return True
                    return any(phrase in normalized_text for phrase in quiz_phrase_triggers)

                course_id = str(msg.get("course_id") or (ctx.course_id if ctx and ctx.course_id else "") or "").strip()
                chapter_index_raw = msg.get("chapter_index")
                section_index_raw = msg.get("section_index")
                try:
                    chapter_index_int = int(chapter_index_raw) if chapter_index_raw is not None else None
                except (TypeError, ValueError):
                    chapter_index_int = None
                try:
                    section_index_int = int(section_index_raw) if section_index_raw is not None else None
                except (TypeError, ValueError):
                    section_index_int = None

                current_slide_context = None
                if course_id and chapter_index_int is not None and section_index_int is not None:
                    current_slide_context = await load_course_slide_context(
                        course_id,
                        chapter_index_int,
                        section_index_int,
                    )

                current_slide_content = (msg.get("slide_content") or "").strip()
                current_chapter_title = msg.get("chapter", "")
                current_section_title = msg.get("section_title", msg.get("slide_title", ""))
                course_title = msg.get("course_title", "")
                course_domain = msg.get("course_domain", "general")
                current_chapter_idx = chapter_index_int + 1 if chapter_index_int is not None else None

                if current_slide_context:
                    current_slide_content = (current_slide_context.get("content") or current_slide_content).strip()
                    current_chapter_title = current_slide_context.get("chapter_title") or current_chapter_title
                    current_section_title = current_slide_context.get("section_title") or current_section_title
                    course_title = current_slide_context.get("course_title") or course_title
                    course_domain = current_slide_context.get("course_domain") or course_domain
                    current_chapter_idx = current_slide_context.get("chapter_order") or current_chapter_idx

                is_in_course = msg.get("in_course", in_course or bool(course_id))
                if is_in_course and is_quiz_intent(content_lower):
                    await cancel_text_question_task()
                    active_text_turn_id = 0
                    quiz_topic = current_section_title or current_chapter_title or current_slide_content or course_title or "Quiz"
                    quiz_query = current_slide_content or current_section_title or current_chapter_title or course_title or quiz_topic
                    quiz_chunks = await asyncio.to_thread(
                        rag.retrieve_chunks,
                        quiz_query,
                        k=Config.RAG_NUM_RESULTS,
                        current_chapter_idx=current_chapter_idx,
                        strict_chapter=bool(current_chapter_idx),
                        course_id=course_id if course_id else None,
                    )
                    if current_slide_content:
                        from langchain_core.documents import Document

                        slide_doc = Document(
                            page_content=current_slide_content,
                            metadata={
                                "course_id": course_id,
                                "chapter_idx": current_chapter_idx,
                                "chapter_title": current_chapter_title,
                                "section_title": current_section_title,
                                "slide_idx": current_slide_context.get("slide_index") if current_slide_context else chapter_index_int,
                                "source_file": msg.get("slide_path") or msg.get("image_url") or "",
                            },
                        )
                        quiz_chunks = [(slide_doc, 1.0, f"Current slide: {current_chapter_title} / {current_section_title}")] + quiz_chunks

                    lang = detect_lang_text(content)
                    subj = detect_subject(content)
                    await process_quiz_request(
                        quiz_topic=quiz_topic,
                        lang=lang,
                        subj=subj,
                        chunks_with_scores=quiz_chunks,
                        question_ctx=ctx,
                        presentation_cursor=current_presentation_cursor,
                        current_chapter_title=current_chapter_title,
                        current_section_title=current_section_title,
                        current_chapter_idx=current_chapter_idx,
                        section_index_int=section_index_int,
                        course_id=course_id,
                        course_title=course_title,
                        course_domain=course_domain,
                        slide_path=str(msg.get("slide_path") or msg.get("image_url") or (current_slide_context.get("slide_path") if current_slide_context else "") or ""),
                    )
                    continue
                if is_in_course and content_lower in resume_triggers:
                    await cancel_text_question_task()
                    active_text_turn_id = 0
                    if ctx:
                        import time as _time2
                        from datetime import datetime as _dt2
                        _resume_t0 = _time2.time()
                        # Stop the waiting-time ticker; logs total wait
                        _ticker_wait = await stop_pause_progress_ticker()
                        _m, _s = divmod(int(_ticker_wait), 60)
                        log.info(
                            f"[{ctx.session_id[:8]}] ▶ RESUME TRIGGERED | trigger='{content_lower}' | "
                            f"timestamp={_dt2.utcnow().isoformat()} | "
                            f"wait_total={_ticker_wait:.1f}s ({_m}m{_s:02d}s)"
                        )
                        # ✅ BUG #4 FIX: Retrieve char_offset from paused_state
                        resumed_ctx = await dialogue.resume_session(ctx.session_id)
                        if resumed_ctx and resumed_ctx.paused_state.get("timestamp") is not None:
                            # Restore cursor position from pause point
                            current_presentation_cursor = resumed_ctx.paused_state.get("presentation_cursor", resumed_ctx.paused_state.get("char_offset", 0))
                            cached_text = resumed_ctx.paused_state.get("presentation_text") or ""
                            cached_key = str(resumed_ctx.paused_state.get("presentation_key") or resumed_ctx.paused_state.get("slide_id") or "")
                            resume_slide_id = ":".join(str(part) for part in (course_id, chapter_index_int, section_index_int))
                            if cached_text and cached_key == resume_slide_id:
                                current_presentation_text = cached_text
                            resume_offset = current_presentation_cursor
                            # Compute pause duration from saved timestamp
                            _pause_ts = resumed_ctx.paused_state.get("timestamp")
                            try:
                                _pause_ts_f = float(_pause_ts)
                                _pause_dur_s = max(0.0, _resume_t0 - _pause_ts_f)
                            except (TypeError, ValueError):
                                _pause_dur_s = 0.0
                            _text_len = len(current_presentation_text or "")
                            _progress_pct = (current_presentation_cursor / _text_len * 100.0) if _text_len else 0.0
                            log.info(
                                f"[{ctx.session_id[:8]}] ▶ RESUME RESTORED | "
                                f"pause_duration={_pause_dur_s:.1f}s | "
                                f"cursor={current_presentation_cursor}/{_text_len} ({_progress_pct:.1f}%) | "
                                f"slide_match={cached_key == resume_slide_id} | "
                                f"text_restored={bool(cached_text and cached_key == resume_slide_id)}"
                            )
                            log.info(f"📍 [{ctx.session_id[:8]}] Resume TTS from char {current_presentation_cursor}")
                            record_checkpoint_event({
                                "session_id": ctx.session_id,
                                "language": resumed_ctx.language if resumed_ctx else session_lang,
                                "subject": (ctx.course_analysis.get("course_domain", "") if ctx and ctx.course_analysis else ""),
                                "checkpoint_type": "resume",
                                "point_text": (
                                    f"▶ Reprise au point mémorisé — chapitre {resumed_ctx.chapter_index + 1}, "
                                    f"section {resumed_ctx.section_index + 1}, position {current_presentation_cursor}. "
                                    f"Narration reprise sans nouveau LLM."
                                ),
                                "location_label": f"chapitre {resumed_ctx.chapter_index + 1}, section {resumed_ctx.section_index + 1}",
                                "cursor_label": f"{current_presentation_cursor}/{len(current_presentation_text or '')}",
                                "slide_id": resume_slide_id,
                                "slide_title": current_section_title or current_chapter_title or "",
                                "chapter_index": resumed_ctx.chapter_index,
                                "section_index": resumed_ctx.section_index,
                                "char_position": current_presentation_cursor,
                                "reason": content_lower,
                                "source": "checkpoint",
                            })
                            await send({
                                "type": "system_notice",
                                "text": (
                                    f"▶ Reprise au point mémorisé — chapitre {resumed_ctx.chapter_index + 1}, "
                                    f"section {resumed_ctx.section_index + 1}, position {current_presentation_cursor}. "
                                    f"Narration reprise sans nouveau LLM."
                                ),
                            })
                        ctx = resumed_ctx or ctx
                    await send_state(
                        DialogState.PRESENTING,
                        details={
                            "course_title": course_title or "",
                            "chapter_title": current_chapter_title or "",
                            "section_title": current_section_title or "",
                            "slide_title": current_section_title or current_chapter_title or "",
                            "char_position": current_presentation_cursor,
                        },
                    )
                    await send({"type": "resume_course"})
                    continue

                # ✅ Priorité à la question: stopper immédiatement la présentation en cours
                # pour éviter que la narration du cours continue pendant la réponse.
                # ── ORDRE CRITIQUE ──
                # cancel_text_question_task() réinitialise active_text_turn_id à 0.
                # Donc on doit annuler les anciennes tâches AVANT de set le nouveau
                # turn_id, sinon la nouvelle tâche démarre avec active_text_turn_id=0
                # et son guard d'entrée la fait sortir immédiatement (EARLY EXIT).
                if presentation_task and not presentation_task.done():
                    await cancel_presentation_task(notify_client=True)
                await cancel_text_question_task()

                turn_id = int(msg.get("turn_id") or 0)
                if turn_id <= 0:
                    text_turn_seq += 1
                    turn_id = text_turn_seq
                active_text_turn_id = turn_id

                await cancel_audio_stream(notify_client=True, turn_id=turn_id)

                lang   = detect_lang_text(content)
                subj   = detect_subject(content)
                
                # ✅ UPDATE STATE: Language Detection + Prosody (text: estimated)
                await send_state(
                    DialogState.PROCESSING, 
                    "stt_language_detection", 
                    {"language": lang},
                    {"language": lang},
                    turn_id=turn_id
                )
                
                await send_state(
                    DialogState.PROCESSING,
                    "prosody_analysis",
                    {"type": "text_question"},
                    {"speech_rate": len(content.split())},
                    turn_id=turn_id
                )
                
                # ✅ S29-32: DÉTECTION DE CONFUSION AUTOMATIQUE (généralise audio_pipeline)
                # Inclut: mots-clés, répétition, patterns d'historique + SEMANTIC (si brain fourni)
                # Pour questions texte, créer prosody estimé (pas de voix réelle)
                text_prosody = {
                    "speech_rate": len(content.split()),  # Approximation: un mot = 1 mpm
                    "hesitation_count": 0,  # Pas accessible en texte
                    "markers": [],
                    "confidence": 0.0,  # Pas de signal prosodique en texte
                }
                
                # ✅ Helper function to emit micro-states during confusion detection
                async def emit_confusion_micro_state(state_name: str, metrics: dict):
                    """Wrapper for sending confusion micro-states with proper state"""
                    if metrics and metrics != {}:
                        await send_state(DialogState.PROCESSING, state_name, {}, metrics, turn_id=turn_id)
                
                is_confused, confusion_reason, q_hash, confusion_count = await dialogue.detect_and_track_confusion(
                    session_id=session_id,
                    question_text=content,
                    language=lang,
                    history=history,  # ← Inclure l'historique pour pattern detection
                    brain=brain,      # ← NOUVEAU: Embeddings sémantiques
                    prosody=text_prosody,  # ← Pour questions texte: dummy/estimé
                    on_state_change=emit_confusion_micro_state,  # ✅ Pass callback for micro-states
                )
                
                # RAG retrieval is performed inside the QA graph
                # (agentic/qa/retriever.py:132). Previously this block ran a
                # second retrieval here purely for telemetry — results were
                # never used for answer generation, costing ~3-8s per text
                # question. The current slide is passed to the graph via
                # last_slide_content in qa_state below.
                rag_start = time.time()
                await send_state(DialogState.PROCESSING, "rag_search", {}, {}, turn_id=turn_id)
                chunks_with_scores: list = []
                rag_time = (time.time() - rag_start) * 1000
                avg_score = 0.0

                if ctx:
                    await dialogue.transition(ctx.session_id, DialogState.PROCESSING)
                
                # ✅ UPDATE STATE: RAG Search Complete with Metrics
                await send_state(DialogState.PROCESSING, "rag_search", {}, {
                    "chunks_found": len(chunks_with_scores),
                    "avg_score": round(avg_score, 3),
                    "duration_ms": round(rag_time, 1),
                    "progress_pct": 35
                }, turn_id=turn_id)

                # ✅ UPDATE STATE: Confusion Detection (if any)
                if is_confused:
                    await send_state(DialogState.PROCESSING, "confusion_detected", 
                                   {"reason": confusion_reason}, 
                                   {"confidence": 0.85 if is_confused else 0.1},
                                   turn_id=turn_id)

                llm_start   = time.time()
                
                # ✅ UPDATE STATE: LLM Thinking
                await send_state(DialogState.PROCESSING, "llm_thinking", 
                               {"chunks": len(chunks_with_scores)},
                               {"chunks_processed": len(chunks_with_scores), "progress_pct": 50},
                               turn_id=turn_id)
                
                # ✅ TRY RAG + FALLBACK pour les questions texte (comme le streaming)
                # Construire le prompt : normal ou reformulation si confusion
                last_slide = ctx.last_slide_explained if ctx else ""
                
                if is_confused:
                    # Prompt spécial reformulation pour questions texte
                    question_for_llm = dialogue.build_confusion_prompt(
                        original_question=content,
                        language=lang,
                        last_slide_content=last_slide,
                    )
                    log.info(f"[{session_id[:8]}] 📝 Prompt reformulation appliqué au LLM (texte)")
                else:
                    question_for_llm = content
                
                text_question_task = asyncio.create_task(
                    process_text_question_turn(
                        turn_id=turn_id,
                        content=content,
                        lang=lang,
                        subj=subj,
                        chunks_with_scores=chunks_with_scores,
                        rag_time=rag_time,
                        avg_score=avg_score,
                        is_confused=is_confused,
                        confusion_reason=confusion_reason,
                        question_for_llm=question_for_llm,
                        llm_start=llm_start,
                        question_ctx=ctx,
                        presentation_cursor=current_presentation_cursor,
                        current_chapter_title=current_chapter_title,
                        current_section_title=current_section_title,
                        current_chapter_idx=current_chapter_idx,
                        section_index_int=section_index_int,
                    )
                )
                continue

            # ── quiz — générer un quiz structuré ─────────────────────
            elif msg_type == "quiz":
                if ctx:
                    ctx = await dialogue.get_session(session_id) or ctx

                course_id = str(msg.get("course_id") or (ctx.course_id if ctx and ctx.course_id else "") or "").strip()
                chapter_index_raw = msg.get("chapter_index")
                section_index_raw = msg.get("section_index")
                try:
                    chapter_index_int = int(chapter_index_raw) if chapter_index_raw is not None else None
                except (TypeError, ValueError):
                    chapter_index_int = None
                try:
                    section_index_int = int(section_index_raw) if section_index_raw is not None else None
                except (TypeError, ValueError):
                    section_index_int = None

                current_slide_context = None
                if course_id and chapter_index_int is not None and section_index_int is not None:
                    current_slide_context = await load_course_slide_context(
                        course_id,
                        chapter_index_int,
                        section_index_int,
                    )

                current_slide_content = (msg.get("slide_content") or msg.get("content") or "").strip()
                current_chapter_title = msg.get("chapter", "")
                current_section_title = msg.get("section_title", msg.get("slide_title", ""))
                course_title = msg.get("course_title", "")
                course_domain = msg.get("course_domain", "general")
                current_chapter_idx = chapter_index_int + 1 if chapter_index_int is not None else None

                if current_slide_context:
                    current_slide_content = (current_slide_context.get("content") or current_slide_content).strip()
                    current_chapter_title = current_slide_context.get("chapter_title") or current_chapter_title
                    current_section_title = current_slide_context.get("section_title") or current_section_title
                    course_title = current_slide_context.get("course_title") or course_title
                    course_domain = current_slide_context.get("course_domain") or course_domain
                    current_chapter_idx = current_slide_context.get("chapter_order") or current_chapter_idx

                if not current_slide_content:
                    current_slide_content = current_section_title or current_chapter_title or course_title or ""

                quiz_topic = current_section_title or current_chapter_title or current_slide_content or course_title or "Quiz"
                quiz_query = current_slide_content or current_section_title or current_chapter_title or course_title or quiz_topic

                quiz_chunks = await asyncio.to_thread(
                    rag.retrieve_chunks,
                    quiz_query,
                    k=Config.RAG_NUM_RESULTS,
                    current_chapter_idx=current_chapter_idx,
                    strict_chapter=bool(current_chapter_idx),
                    course_id=course_id if course_id else None,
                )

                if current_slide_content:
                    from langchain_core.documents import Document

                    slide_doc = Document(
                        page_content=current_slide_content,
                        metadata={
                            "course_id": course_id,
                            "chapter_idx": current_chapter_idx,
                            "chapter_title": current_chapter_title,
                            "section_title": current_section_title,
                            "slide_idx": current_slide_context.get("slide_index") if current_slide_context else chapter_index_int,
                            "source_file": msg.get("slide_path") or msg.get("image_url") or "",
                        },
                    )
                    quiz_chunks = [(slide_doc, 1.0, f"Current slide: {current_chapter_title} / {current_section_title}")] + quiz_chunks

                lang = detect_lang_text(quiz_query)
                subj = detect_subject(quiz_query)
                await process_quiz_request(
                    quiz_topic=quiz_topic,
                    lang=lang,
                    subj=subj,
                    chunks_with_scores=quiz_chunks,
                    question_ctx=ctx,
                    presentation_cursor=current_presentation_cursor,
                    current_chapter_title=current_chapter_title,
                    current_section_title=current_section_title,
                    current_chapter_idx=current_chapter_idx,
                    section_index_int=section_index_int,
                    course_id=course_id,
                    course_title=course_title,
                    course_domain=course_domain,
                    slide_path=str(msg.get("slide_path") or msg.get("image_url") or (current_slide_context.get("slide_path") if current_slide_context else "") or ""),
                )
                continue

            # ── next_section — avancer dans le cours ──────────────────
            elif msg_type == "next_section":
                if ctx:
                    await dialogue.next_section(ctx.session_id)
                await send({"type": "section_changed", "direction": "next"})

            # ── record dashboard event ───────────────────────────────
            # (done after each complete interaction)

            # ── ping / keepalive ───────────────────────────────────────
            elif msg_type == "ping":
                await send({"type": "pong"})

    except WebSocketDisconnect:
        websocket_closed = True
        log.info(f"🔌 WebSocket déconnecté : {session_id[:8]}")
    except Exception as exc:
        log.error(f"❌ WebSocket error [{session_id[:8]}]: {exc}", exc_info=True)
        if not websocket_closed:
            try:
                await send({"type": "error", "message": str(exc)})
            except Exception:
                pass
    finally:
        # 🛡️ Stop watchdog FSM + retire du registre debug
        if voice_watchdog_task is not None and not voice_watchdog_task.done():
            voice_watchdog_task.cancel()
            try:
                await voice_watchdog_task
            except (asyncio.CancelledError, Exception):
                pass
        await vfsm.transition(VoiceState.IDLE, reason="ws_disconnect")
        _deps.active_vfsms.pop(session_id, None)
        await cancel_next_slide_prefetch()
        await cancel_presentation_task(notify_client=False)
        await cancel_audio_stream(notify_client=False)
        await cancel_text_question_task()
        # Stop the waiting-time ticker if it was running (logs final wait)
        if pause_progress_task is not None and not pause_progress_task.done():
            await stop_pause_progress_ticker()
