"""Smart Teacher — Configuration"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


class Config:
    """Smart Teacher Configuration"""

    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    # OPENAI_BASE_URL — override the OpenAI API host. Empty/unset → real OpenAI.
    # Set to e.g. "https://api.groq.com/openai/v1" to route through Groq
    # (OpenAI-API compatible) without changing call sites. The OpenAI Python
    # SDK and langchain-openai both honor this; we also pass it explicitly to
    # every OpenAI/ChatOpenAI instantiation for visibility.
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL") or None
    ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")

    # Database
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "smart_teacher")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "admin")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "secret")

    DATABASE_URL: str = (
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))

    # ── Cache & session TTLs (seconds) ─────────────────────────────────
    # Sémantique :
    #   - SESSION_TOKEN_TTL : token WS one-time. Très court (juste le temps
    #     de la connexion initiale). Volé/perdu = client doit refaire /session.
    #   - HTTP_HISTORY_TTL  : historique conversationnel REST /ask. Aligné sur
    #     SESSION_TTL pour cohérence (si dialogue expire, history aussi).
    #   - SESSION_TTL       : état dialogue WS (ctx, baseline étudiant, slides
    #     en cours). 1h = "rallumer son onglet en revenant du déjeuner reprend
    #     où on en était".
    #   - PRESENTATION_SNAPSHOT_TTL : alias de SESSION_TTL (snapshot lié à la session).
    #   - TTS_CACHE_TTL     : audio TTS pré-généré, peut survivre à la session
    #     (réutilisable cross-session). 24h = balance entre cache hit et coût RAM.
    #   - NARRATION_CACHE_TTL : narration LLM cross-session par slide. Long (7j)
    #     car invalide seulement si le contenu du cours change.
    #   - EMBEDDING_CACHE_TTL : embeddings BGE/OpenAI. 24h suffisant — recompute
    #     léger si miss.
    #
    # ⚠️ Drift potentiel : NARRATION (7j) > SESSION (1h) → un narration peut
    # être servie à un nouvel étudiant après que la session originale est morte.
    # C'est intentionnel (cache cross-session par design).
    SESSION_TOKEN_TTL: int        = int(os.getenv("SESSION_TOKEN_TTL", "300"))           # 5 min
    HTTP_HISTORY_TTL: int         = int(os.getenv("HTTP_HISTORY_TTL", "3600"))           # 1 h
    SESSION_TTL: int              = int(os.getenv("SESSION_TTL", "3600"))                # 1 h
    PRESENTATION_SNAPSHOT_TTL: int = int(os.getenv("PRESENTATION_SNAPSHOT_TTL", "3600")) # 1 h (= SESSION_TTL)
    TTS_CACHE_TTL: int            = int(os.getenv("TTS_CACHE_TTL", "86400"))             # 24 h
    NARRATION_CACHE_TTL: int      = int(os.getenv("NARRATION_CACHE_TTL", "604800"))      # 7 j
    EMBEDDING_CACHE_TTL: int      = int(os.getenv("EMBEDDING_CACHE_TTL", "86400"))       # 24 h

    # Audio
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 512
    SPEECH_THRESHOLD: float = 0.3        # Lowered from 0.5 → better speech detection
    SILENCE_DURATION: float = 1.0        # Reduced from 1.5 → faster response
    MAX_AUDIO_DURATION: float = 30.0

    # STT (Whisper)
    STT_BACKEND: str = os.getenv("STT_BACKEND", "faster-whisper")
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE: str = os.getenv("WHISPER_COMPUTE", "int8")
    WHISPER_THREADS: int = int(os.getenv("WHISPER_THREADS", "4"))
    STT_MIN_AUDIO_SEC: float = float(os.getenv("STT_MIN_AUDIO_SEC", "0.1"))  # Reduced from 0.45 - trim_silence was removing too much
    STT_BEAM_SIZE: int = int(os.getenv("STT_BEAM_SIZE", "3"))

    # LLM (GPT)
    GPT_MODEL: str = os.getenv("GPT_MODEL", "gpt-4o-mini")
    GPT_MAX_TOKENS: int = int(os.getenv("GPT_MAX_TOKENS", "400"))
    GPT_TEMPERATURE: float = float(os.getenv("GPT_TEMPERATURE", "0.7"))
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))

    # RAG (Retrieval-Augmented Generation)
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true ").lower() == "true"  # Disabled temporarily (OpenAI quota)
    RAG_NUM_RESULTS: int = int(os.getenv("RAG_NUM_RESULTS", "5"))
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    RAG_DB_DIR: str = os.getenv("RAG_DB_DIR", "data/multimodal_db")
    RAG_USE_RERANKER: bool = os.getenv("RAG_USE_RERANKER", "true").lower() == "true"
    RAG_RERANKER_MODEL: str = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RAG_RERANKER_TOP_N: int = int(os.getenv("RAG_RERANKER_TOP_N", "15"))  # candidats RRF passes au reranker
    # Idea-level chunking via LLM (sous-decoupage des title-chunks en idees atomiques)
    RAG_USE_IDEA_CHUNKING: bool = os.getenv("RAG_USE_IDEA_CHUNKING", "true").lower() == "true"
    RAG_IDEAS_PER_CHUNK_MAX: int = int(os.getenv("RAG_IDEAS_PER_CHUNK_MAX", "6"))
    RAG_IDEA_MIN_LENGTH: int = int(os.getenv("RAG_IDEA_MIN_LENGTH", "30"))
    # Parallelisme des appels LLM par-section/par-idee pendant l'ingestion (idea chunking + summaries).
    # I/O-bound (OpenAI/Ollama) → threads. Trop haut = rate-limit; trop bas = lent.
    RAG_LLM_PARALLELISM: int = max(1, int(os.getenv("RAG_LLM_PARALLELISM", "6")))
    # Fusion des sections trop courtes avant idea-chunking (evite 1 appel LLM pour 2 phrases)
    RAG_FUSION_MIN_CHARS: int = int(os.getenv("RAG_FUSION_MIN_CHARS", "300"))
    RAG_FUSION_MAX_CHARS: int = int(os.getenv("RAG_FUSION_MAX_CHARS", "2400"))
    # KG-augmented retrieval : etendre le top-1 chunk avec prereqs / examples /
    # le concept illustre (Phase 1 — voir RetrieverAgent._augment_with_kg).
    # OFF = retrieval classique BM25+dense+rerank uniquement (pas d'expansion).
    RAG_USE_GRAPH_EXPANSION: bool = os.getenv("RAG_USE_GRAPH_EXPANSION", "true").lower() == "true"

    # Confusion detection (SIGHT)
    # config.py lives in core/, so the project root is one level up
    CONFUSION_MODEL_PATH: str = os.getenv(
        "CONFUSION_MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "dataset" / "sight-main" / "data" / "processed" / "confusion_model_final.pth"),
    )

    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "smart_teacher_multimodal")

    # TTS
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "edge")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "default")
    TTS_MODEL: str = os.getenv("TTS_MODEL", "eleven_multilingual_v2")
    TTS_OUTPUT_FORMAT: str = os.getenv("TTS_OUTPUT_FORMAT", "mp3_22050_32")

    EDGE_VOICES = {
        "fr": "fr-FR-DeniseNeural",
        "en": "en-US-JennyNeural",
    }

    # Paths
    COURSES_DIR: str = os.getenv("COURSES_DIR", "courses")
    MEDIA_DIR: str = os.getenv("MEDIA_DIR", "media")
    SLIDES_DIR: str = f"{MEDIA_DIR}/slides"
    LOGS_DIR: str = os.getenv("LOGS_DIR", "logs")
    DATABASE_DIR: str = "database"

    # Server
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))

    # Performance KPIs
    MAX_RESPONSE_TIME: float = 5.0
    # MAX_INTERRUPTION_LATENCY: float = 0.5  # [UNUSED v1.0]
    # TARGET_WER_FR: float = 0.10  # [UNUSED v1.0]
    # TARGET_WER_EN: float = 0.10  # [UNUSED v1.0]
    TARGET_RTF: float = 0.50

    # Logs
    CSV_LOG_FILE: str = os.getenv("CSV_LOG_FILE", "logs/metrics.csv")
    STT_LOG_FILE: str = os.getenv("STT_LOG_FILE", "logs/stt_metrics.csv")

    # Anti-echo (Option C) — gates pour empecher la re-capture du TTS par le micro
    TTS_SUPPRESSION_WINDOW_S: float = float(os.getenv("TTS_SUPPRESSION_WINDOW_S", "1.5"))
    TTS_ECHO_TAIL_S: float          = float(os.getenv("TTS_ECHO_TAIL_S", "0.5"))
    MIN_INTERRUPT_DURATION_S: float = float(os.getenv("MIN_INTERRUPT_DURATION_S", "0.4"))
    MIN_INTERRUPT_BYTES: int        = int(os.getenv("MIN_INTERRUPT_BYTES", "3500"))
    MIN_TURN_ENERGY_RMS: float      = float(os.getenv("MIN_TURN_ENERGY_RMS", "0.015"))
    ENABLE_ECHO_XCORR: bool         = os.getenv("ENABLE_ECHO_XCORR", "true").lower() == "true"
    ECHO_XCORR_THRESHOLD: float     = float(os.getenv("ECHO_XCORR_THRESHOLD", "0.55"))
    ECHO_XCORR_BUFFER_S: float      = float(os.getenv("ECHO_XCORR_BUFFER_S", "5.0"))

    # Analytics / ClickHouse
    CLICKHOUSE_HOST: str       = os.getenv("CLICKHOUSE_HOST", "localhost")
    CLICKHOUSE_PORT: int       = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    CLICKHOUSE_DB: str         = os.getenv("CLICKHOUSE_DB", "smart_teacher")
    CLICKHOUSE_USER: str       = os.getenv("CLICKHOUSE_USER", "default")
    CLICKHOUSE_PASSWORD: str   = os.getenv("CLICKHOUSE_PASSWORD", "")
    ANALYTICS_CSV_DIR: str     = os.getenv("ANALYTICS_CSV_DIR", "./analytics")

    # Storage / MinIO
    MINIO_ENDPOINT: str        = os.getenv("MINIO_ENDPOINT", "")
    MINIO_ACCESS_KEY: str      = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str      = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str          = os.getenv("MINIO_BUCKET", "smart-teacher")
    MINIO_SECURE: bool         = os.getenv("MINIO_SECURE", "false").lower() == "true"
    LOCAL_MEDIA_DIR: str       = os.getenv("LOCAL_MEDIA_DIR", "./media")

    # ── JWT auth ───────────────────────────────────────────────────────
    # Sémantique :
    #   - JWT_SECRET_KEY : clé HS256 — DOIT être changée en prod via env var
    #   - JWT_ALGORITHM  : HS256 par défaut (HMAC SHA-256, simple)
    #   - JWT_EXPIRATION_HOURS : durée de vie du token (24h par défaut)
    JWT_SECRET_KEY: str        = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_smart_teacher_2024")
    JWT_ALGORITHM: str         = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS: int  = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

    # OCR / Vision LLM (intelligent ingester)
    OCR_MIN_IMAGE_BYTES: int   = int(os.getenv("OCR_MIN_IMAGE_BYTES", "2000"))
    OCR_MIN_IMAGE_DIM_PX: int  = int(os.getenv("OCR_MIN_IMAGE_DIM_PX", "50"))
    USE_VISION_LLM: bool       = os.getenv("USE_VISION_LLM", "false").lower() == "true"
    VISION_LLM_MODEL: str      = os.getenv("VISION_LLM_MODEL", "gpt-4o-mini")
    LIBREOFFICE_BIN: str       = os.getenv("LIBREOFFICE_BIN", "libreoffice")

    # Slide-level vision description (services/vision_describe.py)
    # Each slide PNG is described once and cached on disk forever (MD5 key).
    VISION_DESCRIBE_ENABLED: bool = os.getenv("VISION_DESCRIBE_ENABLED", "true").lower() == "true"
    OLLAMA_VISION_MODEL: str   = os.getenv("OLLAMA_VISION_MODEL", "llava")
    # Text model for the Ollama fallback used by Brain, RAG answer/quiz
    # generation, and course classification. Must match a model already
    # pulled in your Ollama instance (`ollama list`). The init_ollama.sh
    # bootstrap pulls this same name, so changing it here changes both.
    OLLAMA_TEXT_MODEL: str     = os.getenv("OLLAMA_TEXT_MODEL", "qwen2.5")
    OLLAMA_URL: str            = os.getenv("OLLAMA_URL", "http://localhost:11434")
    # Threads per Ollama inference request. Ollama doesn't read this
    # env var on its own — we pass it as ``options.num_thread`` in every
    # /api/generate call. Values:
    #   - 0 (default, "auto")  → Ollama picks half the available cores
    #   - N >= 1              → explicit cap (use cores you actually want
    #                           to dedicate to LLM inference)
    # On CPU-only machines, setting this to (physical_cores - 1) gives
    # the biggest speedup without starving the rest of the system.
    OLLAMA_NUM_THREADS: int    = int(os.getenv("OLLAMA_NUM_THREADS", "0"))

    # ── Operational kill-switches ─────────────────────────────────────
    # Force-disable OpenAI everywhere : Brain.ask falls straight through
    # to the Ollama fallback, vision_describe skips the OpenAI provider,
    # and Config.validate() does not require the API key. Useful when the
    # OpenAI account is not provisioned yet (no billing, prepaid quota
    # exhausted, offline development).
    DISABLE_OPENAI: bool       = os.getenv("DISABLE_OPENAI", "false").lower() == "true"
    # Title detection : when small Ollama vision models (llava 7B, etc.)
    # hallucinate titles ("Introduction à la rive", "Introduction à la
    # recherche d'agrément"), force the pipeline to use the deterministic
    # structural extractor (`_extract_main_concept`) instead. The vision
    # call is skipped entirely — saves time, avoids wrong titles.
    DISABLE_VISION_TITLES: bool = os.getenv("DISABLE_VISION_TITLES", "false").lower() == "true"
    # Verbose per-slide ingestion logging : prints raw OCR excerpt, every
    # LLM call (prompt prefix + response excerpt), and the title-decision
    # tree (vision result, structural fallback, why). Default off because
    # it's noisy in production; turn on to debug a course that ingests
    # with bad titles.
    INGESTION_VERBOSE_LOGS: bool = os.getenv("INGESTION_VERBOSE_LOGS", "false").lower() == "true"


    @classmethod
    def validate(cls) -> None:
        """Check critical parameters at startup"""
        errors = []
        # OpenAI is only required when DISABLE_OPENAI is false. With the
        # kill-switch on, the system uses Ollama for both LLM and vision
        # and the missing API key is expected, not an error.
        if not cls.OPENAI_API_KEY and not cls.DISABLE_OPENAI:
            errors.append("OPENAI_API_KEY missing in .env (set DISABLE_OPENAI=true to use Ollama only)")
        if cls.TTS_PROVIDER == "elevenlabs" and not cls.ELEVENLABS_API_KEY:
            errors.append("ELEVENLABS_API_KEY missing (required if TTS_PROVIDER=elevenlabs)")

        if errors:
            print("\n" + "=" * 60)
            print("❌ CONFIGURATION ERRORS")
            print("=" * 60)
            for err in errors:
                print(f"   • {err}")
            print("\n💡 Create .env file:")
            print("   OPENAI_API_KEY=sk-...")
            print("   ELEVENLABS_API_KEY=... (optional)")
            print("=" * 60 + "\n")
            sys.exit(1)
        print("✅ Configuration validated")

    @classmethod
    def print_info(cls) -> None:
        """Display active configuration"""
        print("\n" + "=" * 60)
        print("⚙️  SMART TEACHER — CONFIGURATION")
        print("=" * 60)
        print(f"\n🎙️  AUDIO:  {cls.SAMPLE_RATE}Hz | chunk={cls.CHUNK_SIZE} | silence={cls.SILENCE_DURATION}s")
        print(f"🧠 STT:    {cls.STT_BACKEND} | Whisper {cls.WHISPER_MODEL_SIZE} | {cls.WHISPER_DEVICE} | {cls.WHISPER_COMPUTE}")
        print(f"💬 LLM:    {cls.GPT_MODEL} | max_tokens={cls.GPT_MAX_TOKENS} | history={cls.MAX_HISTORY_TURNS}")
        print(f"🔊 TTS:    {cls.TTS_PROVIDER}")
        print(f"📚 RAG:    {cls.RAG_DB_DIR} | top-k={cls.RAG_NUM_RESULTS}")
        print(f"🌐 Server: {cls.SERVER_HOST}:{cls.SERVER_PORT}")
        print(f"\n🎯 KPIs:   response<{cls.MAX_RESPONSE_TIME}s | RTF<{cls.TARGET_RTF}")
        print("=" * 60 + "\n")
