
# ── UTF-8 self-bootstrap ──────────────────────────────────────────────
# Re-exec the interpreter with PYTHONUTF8=1 if it isn't already set.
# Why : on Windows the default codec is cp1252, which makes the
# `unstructured` library raise ``UnicodeDecodeError: 'utf-8' codec
# can't decode byte 0xe2…`` on PDFs containing UTF-8 multi-byte
# characters (smart quotes, em-dashes, accents). PYTHONUTF8 is read
# at interpreter start, not at runtime — so a plain ``os.environ[…] =
# "1"`` is too late. The cleanest fix is to detect the missing flag
# and exec a fresh interpreter with -X utf8. This makes ``python
# main.py`` work without any launcher script.
import os as _os
import sys as _sys
if _os.environ.get("PYTHONUTF8") != "1":
    _os.environ["PYTHONUTF8"] = "1"
    _os.environ["PYTHONIOENCODING"] = "utf-8"
    _os.execv(_sys.executable, [_sys.executable, "-X", "utf8", *_sys.argv])
# ──────────────────────────────────────────────────────────────────────


"""
╔══════════════════════════════════════════════════════════════════════╗
║        SMART TEACHER — WebSocket Server (Streaming Audio)          ║
║                                                                      ║
║  Ce fichier est le point d'entrée PRINCIPAL du projet.              ║
║  Il remplace server.py pour la production.                          ║
║                                                                      ║
║  Routes WebSocket :                                                  ║
║    WS  /ws/{session_id}    — pipeline vocal temps réel              ║
║                                                                      ║
║  Routes REST :                                                        ║
║    POST /session            — créer une session avec token auth     ║
║    POST /ask                — texte → réponse                       ║
║    POST /ingest             — indexer des fichiers dans le RAG      ║
║    GET  /rag/stats          — statistiques RAG                      ║
║    GET  /session/{id}       — état de la session                    ║
║    GET  /health             — healthcheck complet                   ║
║                                                                      ║
║  Messages WebSocket (client → serveur) :                            ║
║    {"type": "start_session",  "token": "...", "language": "fr"}   ║
║    {"type": "audio_chunk",    "data": "<base64>"}                   ║
║    {"type": "audio_end"}                                            ║
║    {"type": "interrupt"}                                            ║
║    {"type": "next_section"}                                         ║
║    {"type": "text",           "content": "..."}                     ║
║    {"type": "quiz"}                                                ║
║                                                                      ║
║  Messages WebSocket (serveur → client) :                            ║
║    {"type": "session_ready",  "session_id": "..."}                  ║
║    {"type": "transcription",  "text": "...", "lang": "fr"}          ║
║    {"type": "answer_text",    "text": "..."}                        ║
║    {"type": "audio_chunk",    "data": "<base64>", "mime": "..."}    ║
║    {"type": "state_change",   "state": "PROCESSING"}                ║
║    {"type": "error",          "message": "..."}                     ║
║    {"type": "quiz_prompt",    "quiz": {...}}                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── stdlib ─────────────────────────────────────────────────────────────
import logging
import socket
from contextlib import asynccontextmanager
from pathlib import Path

# ── third-party ────────────────────────────────────────────────────────
from fastapi import (
    FastAPI, Request,
)
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

# ── core / config ──────────────────────────────────────────────────────
from core.config import Config

# ── audio (STT, VAD, TTS, voice FSM) ───────────────────────────────────
from audio.audio_input import AudioInput
from audio.transcriber import Transcriber
from audio.tts import VoiceEngine

# ── AI / RAG ───────────────────────────────────────────────────────────
from ai.llm import Brain
from rag.multimodal_rag import MultiModalRAG
from rag.ingestion_manager import IngestionManager

# ── pedagogy (dialogue, profile, slides, course analysis) ─────────────
from pedagogy.dialogue import DialogueManager
from pedagogy.personalization.profile import ProfileManager
from pedagogy.slide_sync import SlideSynchronizer

# ── handlers (WS pipeline + REST helpers) ─────────────────────────────

# ── storage / observability ────────────────────────────────────────────
from observability.analytics import get_analytics
from observability.dashboard import (
    router as dashboard_router,
)
from observability.logger import CsvLogger
from observability.stt_logger import STTLogger
from storage.media_storage import get_storage
from storage.transcript_search import get_searcher

# ── database ───────────────────────────────────────────────────────────
from database.init_db import check_db_connection, create_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SmartTeacher.Main")

FAVICON_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0%' stop-color='#7c6dfa'/><stop offset='100%' stop-color='#00e5b0'/></linearGradient></defs>
<rect width='64' height='64' rx='16' fill='#0b0d16'/><rect x='14' y='14' width='36' height='36' rx='10' fill='url(#g)' opacity='.95'/>
<path d='M22 24h20v4H22zm0 8h20v4H22zm0 8h14v4H22z' fill='#ffffff'/></svg>"""


# ══════════════════════════════════════════════════════════════════════
#  INITIALISATION DES SERVICES
# ══════════════════════════════════════════════════════════════════════
# Services Docker requis (lancer via `docker-compose up -d`) :
#   PostgreSQL (5432), Redis (6379), Qdrant (6333),
#   Elasticsearch (9200), MinIO (9000), Ollama (11434, fallback LLM)

log.info("🤖 SMART TEACHER — démarrage (WebSocket + REST)")
log.info("💡 Services requis : PostgreSQL · Redis · Qdrant · Elasticsearch · MinIO · Ollama")

Config.validate()

transcriber = Transcriber()
brain       = Brain()
voice       = VoiceEngine()
rag         = MultiModalRAG(
    db_dir=Config.RAG_DB_DIR,
    force_local_embeddings=not Config.RAG_ENABLED,
)
# Eagerly load the cross-encoder reranker now (~20-43s on CPU) so the
# first Q&A turn doesn't pay this cost. Without this, every fresh
# server boot makes the first student wait 30+ extra seconds. Failure
# is non-fatal — RAG falls back to heuristic ranking.
rag.warmup()

# Pre-load the SIGHT confusion model (xlm-roberta-base, ~25s on CPU).
# Without this, the first student interruption pays ~25s loading the
# tokenizer + transformer + classifier head. Pre-loading here moves
# that cost from "user-visible latency" to "server boot time".
# Returns None if the bundle file is missing — non-fatal.
try:
    from pedagogy.confusion.detector import get_confusion_detector
    _confusion_detector = get_confusion_detector()
    if _confusion_detector is not None:
        log.info("✅ SIGHT confusion model preloaded")
except Exception as _exc:
    log.warning(f"⚠️ SIGHT confusion preload skipped: {_exc}")
csv_logger  = CsvLogger()
stt_logger  = STTLogger()
dialogue    = DialogueManager()
profile_mgr   = ProfileManager()
slide_sync    = SlideSynchronizer()
audio_input = AudioInput()  # ✅ Silero VAD for backend voice detection
media_storage = get_storage()
transcript_searcher = get_searcher()
analytics_engine    = get_analytics()
ingestion_manager   = IngestionManager()

# ── Agentic LangGraph: Teaching Graph (Phase A — Planner → Narrator) ──
from agentic import build_teaching_graph, build_qa_graph
teaching_graph = build_teaching_graph(brain, rag=rag, profile_mgr=profile_mgr)
qa_graph = build_qa_graph(brain, rag=rag)

# 🧠 Orchestrator central : bridge FSM ↔ Agentic graphs (cancellable, per-session)
from agentic.orchestrator import AgenticOrchestrator
agentic_orchestrator = AgenticOrchestrator(qa_graph=qa_graph, teaching_graph=teaching_graph)

# ── Register services in deps registry so routes/*.py and services/*.py
#    can retrieve them via deps.get_*() without circular imports.
import deps as _deps
_deps.register_services(
    transcriber=transcriber,
    brain=brain,
    voice=voice,
    rag=rag,
    dialogue=dialogue,
    profile_mgr=profile_mgr,
    csv_logger=csv_logger,
    stt_logger=stt_logger,
    transcript_searcher=transcript_searcher,
    analytics_engine=analytics_engine,
    agentic_orchestrator=agentic_orchestrator,
    ingestion_manager=ingestion_manager,
    slide_sync=slide_sync,
    media_storage=media_storage,
    teaching_graph=teaching_graph,
    qa_graph=qa_graph,
    audio_input=audio_input,
)

from core.diagnostics import log_backend_diagnostics

log.info("✅ Modules prêts")


# ══════════════════════════════════════════════════════════════════════
#  APPLICATION FASTAPI — lifespan + middleware + static + routers
# ══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation légère au démarrage, avec dégradation si PostgreSQL est indisponible."""

    # ── Silence Windows asyncio + asyncpg socket-teardown noise ──────────
    # On Windows, the Proactor event loop sometimes logs ConnectionResetError
    # tracebacks when asyncpg closes pooled connections after a successful
    # commit (the remote already closed; our shutdown(SHUT_RDWR) fails).
    # Data is fine; only the shutdown step is noisy. We swallow it here.
    import asyncio
    loop = asyncio.get_running_loop()
    _default_handler = loop.get_exception_handler() or loop.default_exception_handler
    def _handler(loop, context):
        if isinstance(context.get("exception"), ConnectionResetError):
            return
        _default_handler(context)
    loop.set_exception_handler(_handler)

    try:
        if await check_db_connection():
            await create_tables()
            log.info("✅ PostgreSQL tables créées/validées")
        else:
            log.info("ℹ️ PostgreSQL indisponible au démarrage — mode dégradé activé")
    except Exception as exc:
        log.info(f"ℹ️ PostgreSQL non disponible au démarrage ({exc}) — mode dégradé activé")

    await log_backend_diagnostics()

    yield


app = FastAPI(
    title="Smart Teacher API",
    description="Professeur IA Vocal — WebSocket + REST | STT+RAG+LLM+TTS",
    version="3.0.0",
    lifespan=lifespan,
)


# ── Middleware ─────────────────────────────────────────────────────────

@app.middleware("http")
async def disable_html_cache(request: Request, call_next):
    """No-cache pour le HTML — évite les soucis de hot-reload côté front."""
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static files ──────────────────────────────────────────────────────

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
if Path("media").exists():
    app.mount("/media", StaticFiles(directory="media"), name="media")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_icon():
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


# ── Modular routers (routes/*.py) ─────────────────────────────────────
# Toutes les routes REST vivent dans routes/*.py. Ajouter un nouveau
# domaine = créer routes/<nom>.py + ajouter la ligne ci-dessous.

from routes.admin              import router as admin_router
from routes.analytics           import router as analytics_router
from routes.auth                import router as auth_router
from routes.course              import router as course_router
from routes.dashboard_services  import router as dashboard_services_router
from routes.health              import router as health_router
from routes.practice            import router as practice_router
from routes.rest                import router as rest_router
from routes.search              import router as search_router
from routes.session             import router as session_router
from routes.student             import router as student_router
from routes.voice               import router as voice_router

for _router in (
    dashboard_router,           # /dashboard (observability)
    auth_router,                # /auth/register, /login, /me, /logout
    session_router,             # /session/{id}/profile, /tts_params
    analytics_router,           # /analytics/* + /kpi
    search_router,              # /search/*
    voice_router,                # /voice/*, /agentic/*
    health_router,              # /health, /rag/*, /cache/*, /debug/*
    practice_router,            # /concept/*/practice, /practice/*/submit
    student_router,             # /student/{id}/* + /student/me/*
    admin_router,               # /admin/students, /timeline, /learning-styles
    course_router,              # /course/build, /list, /structure, etc.
    rest_router,                # /session POST, /ask, /ingest
    dashboard_services_router,  # /dashboard/services
):
    app.include_router(_router)


# ── WebSocket router (handlers/ws.py) ─────────────────────────────────
# The /ws/{session_id} pipeline lives entirely in handlers/ws.py — it pulls
# the shared services from `deps` at the start of each connection.
from handlers.ws import router as ws_router
app.include_router(ws_router)


# ══════════════════════════════════════════════════════════════════════
#  Route racine — info + healthcheck léger
# ══════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status":    "running",
        "rag_ready": rag.is_ready,
        "tts":       Config.TTS_PROVIDER,
        "model":     Config.GPT_MODEL,
        "websocket": "ws://<host>:8000/ws/{session_id}",
    }


# ══════════════════════════════════════════════════════════════════════
#  Lancement (uvicorn)
# ══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import uvicorn

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
            probe_socket.bind((Config.SERVER_HOST, Config.SERVER_PORT))
    except OSError as exc:
        log.error(
            "🚫 Port %s déjà utilisé sur %s:%s (%s) → arrêtez l'autre instance ou changez SERVER_PORT",
            Config.SERVER_PORT,
            Config.SERVER_HOST,
            Config.SERVER_PORT,
            exc,
        )
        raise SystemExit(1)

    log.info("")
    log.info("🌐 Interface UI       : http://localhost:" + str(Config.SERVER_PORT) + "/static/index.html")
    log.info("📚 Swagger Docs       : http://localhost:" + str(Config.SERVER_PORT) + "/docs")
    log.info("🔌 WebSocket tunnels  : ws://localhost:" + str(Config.SERVER_PORT) + "/ws/{session_id}")
    log.info("")
    log.info("✅ Serveur actif — Appuyez sur Ctrl+C pour arrêter")
    log.info("")
    uvicorn.run(app, host=Config.SERVER_HOST, port=Config.SERVER_PORT, reload=False)
