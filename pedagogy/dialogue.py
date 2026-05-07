"""Smart Teacher — Dialogue State Machine"""

import base64
import hashlib
import json
import logging
import time
import uuid
from functools import lru_cache
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional

import redis.asyncio as aioredis

from core.config import Config

log = logging.getLogger("SmartTeacher.Dialogue")


# NOTE: ``_get_semantic_embedder`` and ``_is_openai_embedding_model``
# were removed. Their only consumer (``check_semantic_repetition``) is
# also gone — both belonged to the rule-based confusion detection layer
# (cosine similarity > 0.85 to flag "repeated questions" as confusion),
# which has been replaced by SIGHT-only detection.


@lru_cache(maxsize=1)
def _get_sight_confusion_predictor():
    try:
        from pedagogy.confusion.detector import predict_confusion
        return predict_confusion
    except Exception as exc:
        log.warning("SIGHT confusion model unavailable: %s", exc)
        return None

_redis: Optional[aioredis.Redis] = None

async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        host = __import__("os").getenv("REDIS_HOST", "localhost")
        port = int(__import__("os").getenv("REDIS_PORT", 6379))
        _redis = aioredis.Redis(host=host, port=port, decode_responses=True)
    return _redis


class DialogState(str, Enum):
    IDLE        = "IDLE"          # Aucune session active
    INDEXING    = "INDEXING"      # Ingestion de cours en cours (bloqué)
    PRESENTING  = "PRESENTING"    # L'IA présente le cours (TTS en cours)
    LISTENING   = "LISTENING"     # L'IA écoute l'étudiant (VAD actif)
    PROCESSING  = "PROCESSING"    # STT + RAG + LLM en cours
    RESPONDING  = "RESPONDING"    # TTS réponse en cours
    WAITING     = "WAITING"       # En attente de compréhension (demande si OK)
    CLARIFICATION = "CLARIFICATION"  # ✅ NOUVEAU: Étudiant demande clarification/revenir slide


# Transitions valides — Machine d'état REAL WORLD (prod-ready)
VALID_TRANSITIONS: dict[DialogState, list[DialogState]] = {
    DialogState.IDLE:       [DialogState.INDEXING, DialogState.PRESENTING, DialogState.LISTENING],
    DialogState.INDEXING:   [DialogState.IDLE, DialogState.PRESENTING],
    DialogState.PRESENTING: [DialogState.LISTENING, DialogState.WAITING, DialogState.CLARIFICATION],  # ✅ Peut clarifier pendant présentation
    DialogState.LISTENING:  [DialogState.PROCESSING, DialogState.PRESENTING, DialogState.CLARIFICATION],  # ✅ Demander clarif pendant écoute
    DialogState.PROCESSING: [DialogState.RESPONDING, DialogState.CLARIFICATION, DialogState.IDLE, DialogState.LISTENING, DialogState.PRESENTING],
    DialogState.RESPONDING: [DialogState.PRESENTING, DialogState.WAITING, DialogState.LISTENING, DialogState.CLARIFICATION],  # ✅ Question pendant réponse
    DialogState.WAITING:    [DialogState.PRESENTING, DialogState.LISTENING, DialogState.IDLE, DialogState.CLARIFICATION],
    DialogState.CLARIFICATION: [DialogState.RESPONDING, DialogState.PRESENTING, DialogState.LISTENING],  # ✅ Répondre à clarif, retour à présentation
}


@dataclass
class SessionContext:
    """Complete session state (serialized to Redis)"""
    session_id:      str  = field(default_factory=lambda: str(uuid.uuid4()))
    state:           str  = DialogState.IDLE.value
    language:        str  = "fr"
    student_level:   str  = "lycée"    # collège | lycée | université

    # Position dans le cours
    course_id:       Optional[str] = None
    chapter_index:   int  = 0
    section_index:   int  = 0
    char_position:   int  = 0          # Position dans le texte de la section (legacy, char-based resume)
    last_narrated_idea_id: str = ""    # ✨ Idea-based resume — granularité grammaticale propre
                                        # Set par narrator après chaque idée TTS-streamée. Au resume,
                                        # on reprend à plan.ideas[index_of(last_narrated_idea_id) + 1].
    student_id: str = ""               # ✨ UUID du compte étudiant authentifié (extrait du JWT à start_session)
                                        # Permet aux agents de fetch StudentProfile + adapter la personnalisation.
    
    # ✅ NOUVEAU: Course metadata (analyse du cours)
    course_summary:  str  = ""         # Résumé court du cours (pour LLM)
    course_analysis: dict = field(default_factory=dict)  # Analyse complète (lang, level, etc)
    last_slide_explained: str = ""     # Slide précédente expliquée (pour continuité)
    # Slide-level pedagogical continuity (read by services.presentation
    # before each Teaching Graph run; written after the run succeeds).
    # Allow the narrator to open with a bridge ("we just saw X — now Y")
    # instead of restarting cold on every slide.
    last_concept_explained: str = ""   # Main concept narrated on prior slide
    last_narration_summary: str = ""   # 1-2 sentence recap of prior narration
    
    # ── Détection de confusion ─────────────────────────────────────
    confusion_count:      int  = 0          # Nombre de confusions détectées cette session
    last_question_hash:   str  = ""         # Hash de la dernière question (pour détecter répétitions)
    repeated_question_count: int = 0        # Fois qu'une même question a été posée
    
    # ✅ NOUVEAU (Couche #1): Profil adaptatif étudiant pour seuils dynamiques
    student_baseline: dict = field(default_factory=lambda: {
        "avg_speech_rate": 120.0,              # mots/min (baseline francophone)
        "avg_question_length": 8,              # mots typiques par question
        "avg_questions_per_turn": 1.2,         # Questions par tour
        "hesitation_baseline": 1.0,            # Hésitations moyennes
        "confusion_threshold_multiplier": 1.0, # Adaptatif (1.0 = normal, <1 = sujet confus)
        "turns_analyzed": 0,                   # Nombre de turns análysés
    })
    
    # 🔴 PAUSE/REPRISE — Sauvegarder position exacte lors pause
    paused_state:    dict = field(default_factory=lambda: {
        "is_paused": False,
        "slide_id": None,              # Quel slide?
        "char_offset": 0,              # Position (caractères)
        "timestamp": None,             # Quand?
    })
    
    # Blocking pendant ingestion
    is_indexing:     bool = False      # True = en cours d'indexation, bloque les questions
    indexing_progress: int = 0         # Pourcentage 0-100

    # Historique conversationnel (max 10 messages)
    history:         list = field(default_factory=list)

    # Métriques
    total_turns:     int   = 0
    interruptions:   int   = 0
    created_at:      float = field(default_factory=time.time)
    last_activity:   float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "SessionContext":
        d = json.loads(data)
        return cls(**d)

    def add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > Config.MAX_HISTORY_TURNS * 2:
            self.history = self.history[2:]
        self.last_activity = time.time()


# ══════════════════════════════════════════════════════════════════════
#  GESTIONNAIRE DE DIALOGUE
# ══════════════════════════════════════════════════════════════════════

# TTL aliases (read from Config — single source of truth in core/config.py)
SESSION_TTL = Config.SESSION_TTL
PRESENTATION_SNAPSHOT_TTL = Config.PRESENTATION_SNAPSHOT_TTL
TTS_CACHE_TTL = Config.TTS_CACHE_TTL


# ══════════════════════════════════════════════════════════════════════
#  SEEN CONCEPTS — Redis SET per session (anti-repetition runtime)
# ══════════════════════════════════════════════════════════════════════
#  Pour chaque session, on tient un set Redis des idea_id deja vus par
#  l'eleve. Permet au Responder d'adapter son prompt :
#    - idea_id absent → mode explication complete
#    - idea_id present → mode revision rapide
#  Cle : seen_ideas:{session_id}, TTL aligne sur SESSION_TTL.

async def mark_idea_seen(session_id: str, idea_id: str) -> None:
    """Marque idea_id comme vu dans cette session (Redis SET)."""
    if not session_id or not idea_id:
        return
    try:
        r = await get_redis()
        key = f"seen_ideas:{session_id}"
        await r.sadd(key, idea_id)
        await r.expire(key, SESSION_TTL)
    except Exception as exc:
        log.debug(f"mark_idea_seen failed: {exc}")


async def mark_ideas_seen(session_id: str, idea_ids: list[str] | set[str]) -> None:
    """Bulk : marque plusieurs idea_id comme vus (1 round-trip Redis)."""
    if not session_id or not idea_ids:
        return
    ids = [i for i in idea_ids if i]
    if not ids:
        return
    try:
        r = await get_redis()
        key = f"seen_ideas:{session_id}"
        await r.sadd(key, *ids)
        await r.expire(key, SESSION_TTL)
    except Exception as exc:
        log.debug(f"mark_ideas_seen failed: {exc}")


async def is_idea_seen(session_id: str, idea_id: str) -> bool:
    """True si idea_id deja vu dans cette session."""
    if not session_id or not idea_id:
        return False
    try:
        r = await get_redis()
        return bool(await r.sismember(f"seen_ideas:{session_id}", idea_id))
    except Exception as exc:
        log.debug(f"is_idea_seen failed: {exc}")
        return False


async def get_seen_ideas(session_id: str) -> set[str]:
    """Retourne l'ensemble complet des idea_id vus dans cette session."""
    if not session_id:
        return set()
    try:
        r = await get_redis()
        members = await r.smembers(f"seen_ideas:{session_id}")
        # decode_responses=True dans get_redis() → strings deja decodees
        return {m for m in members if m}
    except Exception as exc:
        log.debug(f"get_seen_ideas failed: {exc}")
        return set()


async def clear_seen_ideas(session_id: str) -> None:
    """Vide le set des idea_id vus (utile pour forcer re-explication)."""
    if not session_id:
        return
    try:
        r = await get_redis()
        await r.delete(f"seen_ideas:{session_id}")
    except Exception as exc:
        log.debug(f"clear_seen_ideas failed: {exc}")


# ── Reformulation prompt builder ──────────────────────────────────────
#
# Single template per language. The previous version had 7 templates per
# language dispatched by a ``reason`` string ("keyword", "repeated",
# "too_short", "pattern_*", "sight_model") — but every ``reason`` other
# than "sight_model" was produced by rule-based detectors (keyword
# matching, hash-equal hashing, history pattern thresholds, similarity
# heuristics) that have all been removed. With SIGHT-only detection
# the dispatcher had a single live branch and 6 dead ones.
#
# Choosing the *right* reformulation strategy (analogy / decomposition /
# example / socratic / recap / simpler words) is a learned-policy
# problem, not a string-keyed lookup. That work belongs to the
# binôme's reformulation graph (StrategyClassifier → Planner →
# Generator). Until that ships, this module exposes the minimal
# template "rephrase simpler with a concrete example" used by both
# the audio pipeline and the QA graph responder.

_REFORM_INSTRUCTION = {
    "fr": "L'étudiant n'a pas compris. Reformule l'explication précédente "
          "différemment, avec un exemple concret et des mots plus simples.",
    "en": "The student didn't understand. Rephrase the explanation "
          "differently, using a concrete example and simpler words.",
}


def compose_reformulation_prompt(
    original_question: str,
    language: str,
    last_slide_content: str = "",
    history_text: str = "",
) -> str:
    """Build a reformulation prompt for the LLM when confusion is detected.

    Args:
        original_question : the student's utterance that triggered confusion.
        language          : "fr" | "en".
        last_slide_content: optional excerpt of the slide currently shown
                            (truncated to 400 chars).
        history_text      : optional formatted history of recent exchanges.

    Returns:
        A prompt string ready to feed into Brain.ask() / brain.present().
    """
    lang_key = language[:2] if language and language[:2] in _REFORM_INSTRUCTION else "fr"
    parts = [_REFORM_INSTRUCTION[lang_key]]
    if last_slide_content:
        slide_label = "Contenu de la slide précédente :" if lang_key == "fr" else "Previous slide content:"
        parts.append(f"\n{slide_label}\n{last_slide_content[:400]}")
    if history_text:
        hist_label = "Échanges récents (continuité) :" if lang_key == "fr" else "Recent exchanges (continuity):"
        parts.append(f"\n{hist_label}\n{history_text}")
    if original_question:
        q_label = "Question de l'étudiant :" if lang_key == "fr" else "Student question:"
        parts.append(f"\n{q_label} {original_question}")

    return "\n".join(parts)


class DialogueManager:
    """
    Gère l'état de la conversation entre l'étudiant et l'IA.

    Utilisation typique (WebSocket) :
        manager = DialogueManager()

        # Créer une session
        ctx = await manager.create_session(language="fr")

        # Démarrer la présentation du cours
        await manager.transition(ctx.session_id, DialogState.PRESENTING)
        text = await manager.get_current_section_text(ctx.session_id, course_sections)

        # L'étudiant interrompt → transition LISTENING
        await manager.handle_interruption(ctx.session_id)

        # L'étudiant a fini de parler → PROCESSING
        await manager.transition(ctx.session_id, DialogState.PROCESSING)

        # Après la réponse → reprendre le cours (PRESENTING)
        await manager.transition(ctx.session_id, DialogState.PRESENTING)
    """

    # ── Gestion des sessions Redis ────────────────────────────────────
    async def _save(self, ctx: SessionContext) -> None:
        r = await get_redis()
        await r.setex(f"session:{ctx.session_id}", SESSION_TTL, ctx.to_json())

    async def _load(self, session_id: str) -> Optional[SessionContext]:
        r = await get_redis()
        data = await r.get(f"session:{session_id}")
        if not data:
            return None
        return SessionContext.from_json(data)

    async def _delete(self, session_id: str) -> None:
        r = await get_redis()
        await r.delete(f"session:{session_id}")

    @staticmethod
    def _presentation_snapshot_key(session_id: str, slide_id: str) -> str:
        return f"presentation:snapshot:{session_id}:{slide_id}"

    @staticmethod
    def _tts_cache_key(text: str, language: str, rate: str, provider: str, voice_name: str) -> str:
        normalized = "|".join([
            provider.strip().lower(),
            voice_name.strip().lower(),
            language.strip().lower(),
            rate.strip().lower(),
            text.strip(),
        ])
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return f"presentation:tts:{digest}"

    # ── Cycle de vie de la session ────────────────────────────────────
    async def create_session(
        self,
        session_id:    Optional[str] = None,
        language:      str = "fr",
        student_level: str = "lycée",
        course_id:     Optional[str] = None,
    ) -> SessionContext:
        ctx = SessionContext(
            session_id=session_id or str(uuid.uuid4()),
            language=language,
            student_level=student_level,
            course_id=course_id,
        )
        await self._save(ctx)
        log.info(f"✅ Session créée : {ctx.session_id[:8]} | lang={language}")
        return ctx

    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        return await self._load(session_id)

    async def end_session(self, session_id: str) -> None:
        await self._delete(session_id)
        log.info(f"🔚 Session terminée : {session_id[:8]}")

    # ── Transitions d'état ────────────────────────────────────────────
    async def transition(
        self, session_id: str, new_state: DialogState
    ) -> Optional[SessionContext]:
        ctx = await self._load(session_id)
        if not ctx:
            log.warning(f"Session {session_id[:8]} introuvable")
            return None

        current = DialogState(ctx.state)
        
        # Si on est déjà dans l'état cible, ne rien faire (pas d'erreur)
        if current == new_state:
            log.debug(f"[{session_id[:8]}] Déjà en {current.value}, ignorer transition")
            return ctx
        
        allowed = VALID_TRANSITIONS.get(current, [])

        if new_state not in allowed:
            msg = f"Transition invalide : {current.value} → {new_state.value} (autorisées : {[s.value for s in allowed]})"
            log.error(f"🔴 BLOCKED: {msg}")
            raise ValueError(msg)

        ctx.state = new_state.value
        ctx.last_activity = time.time()
        await self._save(ctx)
        log.info(f"✅ [{session_id[:8]}] {current.value} → {new_state.value}")
        return ctx

    # ── Pause/Reprise ────────────────────────────────────────────────────────
    async def pause_session(
        self,
        session_id: str,
        slide_id: Optional[str] = None,
        char_offset: int = 0,
        presentation_text: Optional[str] = None,
        presentation_cursor: Optional[int] = None,
        presentation_key: Optional[str] = None,
        slide_title: Optional[str] = None,
        last_idea_id: Optional[str] = None,  # ✨ resume-by-idea: dernière idée narrée
        slide_path: Optional[str] = None,    # ✨ image PNG pour vision (Q&A)
        slide_content: Optional[str] = None, # ✨ texte OCR de la slide (Q&A)
    ) -> Optional[SessionContext]:
        """Save exact position when pausing.

        char_offset = legacy char-based resume (peut couper au milieu d'un mot).
        last_idea_id = idea-based resume propre (au prochain idea complet).
        slide_path = chemin du PNG rendu — needed by Q&A handler so it
        can describe the slide visually (charts, diagrams, formulas) when
        the student asks a question. Fallback "" if no image available.
        slide_content = OCR/extracted text of the slide — Q&A merges
        this with the vision description for grounding.
        Both are stored; the resumer picks whichever is available.
        """
        ctx = await self._load(session_id)
        if not ctx:
            return None

        # Pick the most recent idea_id known (caller arg > ctx)
        effective_idea_id = last_idea_id or ctx.last_narrated_idea_id or ""

        # Preserve slide_path/slide_content from previous paused_state if
        # the caller didn't supply new values — avoids losing the image
        # path on every pause/resume cycle.
        prev_state = ctx.paused_state or {}
        effective_slide_path = (
            slide_path
            if slide_path is not None
            else prev_state.get("slide_path", "")
        ) or ""
        effective_slide_content = (
            slide_content
            if slide_content is not None
            else prev_state.get("slide_content", "")
        ) or ""

        # Last-resort: if still no slide_path (caller didn't pass it AND
        # no prior paused state), derive it from course_id + slide_id.
        # Most callers don't pass slide_path explicitly — this single
        # DB lookup keeps the Q&A vision path working without forcing
        # every pause caller to thread the path through. Failure is
        # silent (vision will just be skipped).
        if not effective_slide_path and ctx.course_id and slide_id:
            try:
                from services.course_slides import load_course_slide_context
                parts = str(slide_id).split(":")
                if len(parts) >= 3:
                    ch_idx = int(parts[1])
                    sec_idx = int(parts[2])
                    slide_ctx = await load_course_slide_context(
                        ctx.course_id, ch_idx, sec_idx,
                    )
                    if slide_ctx:
                        effective_slide_path = slide_ctx.get("slide_path") or ""
                        if not effective_slide_content:
                            effective_slide_content = slide_ctx.get("content") or ""
            except Exception as exc:
                log.debug(f"pause_session: slide_path lookup skipped: {exc}")

        ctx.paused_state = {
            "is_paused": True,
            "slide_id": slide_id,
            "char_offset": char_offset,
            "last_idea_id": effective_idea_id,  # ✨ for clean idea-based resume
            "timestamp": time.time(),
            "presentation_text": presentation_text or "",
            "presentation_cursor": max(0, presentation_cursor if presentation_cursor is not None else char_offset),
            "presentation_key": presentation_key or slide_id or "",
            "slide_title": slide_title or "",
            "slide_path": effective_slide_path,           # ✨ for Q&A vision
            "slide_content": effective_slide_content,     # ✨ OCR text for Q&A
            "presentation_text_len": len(presentation_text or ""),
        }
        ctx.char_position = char_offset
        if effective_idea_id:
            ctx.last_narrated_idea_id = effective_idea_id
        ctx.interruptions += 1
        ctx.state = DialogState.WAITING.value
        ctx.last_activity = time.time()
        await self._save(ctx)

        if slide_id:
            await self.save_presentation_snapshot(
                session_id=session_id,
                slide_id=slide_id,
                presentation_text=presentation_text or "",
                presentation_cursor=ctx.char_position,
                slide_title=slide_title or "",
            )

        log.info(f"⏸️  [{session_id[:8]}] Paused at offset {char_offset} | slide={slide_id or 'unknown'}")
        return ctx

    async def save_narration_progress(
        self, session_id: str, idea_id: str,
    ) -> Optional[SessionContext]:
        """Track the latest TTS-streamed idea for clean resume-by-idea.

        Called by the narrator after each idea finishes streaming. The next
        pause/resume uses this to skip ahead to the next idea (clean grammar)
        instead of resuming mid-character.
        """
        if not idea_id:
            return None
        ctx = await self._load(session_id)
        if not ctx:
            return None
        if ctx.last_narrated_idea_id == idea_id:
            return ctx  # idempotent
        ctx.last_narrated_idea_id = idea_id
        ctx.last_activity = time.time()
        await self._save(ctx)
        log.debug(f"[{session_id[:8]}] narration progress: idea={idea_id}")
        return ctx

    async def resume_session(self, session_id: str) -> Optional[SessionContext]:
        """Resume exactly at pause point (not from beginning!)"""
        ctx = await self._load(session_id)
        if not ctx:
            return None
        
        if not ctx.paused_state.get("is_paused"):
            log.warning(f"[{session_id[:8]}] Session is not paused")
            return ctx
        
        offset = ctx.paused_state.get("presentation_cursor", ctx.paused_state.get("char_offset", 0))
        ctx.paused_state["is_paused"] = False
        ctx.paused_state["char_offset"] = offset
        ctx.paused_state["presentation_cursor"] = offset
        ctx.char_position = offset
        ctx.state = DialogState.PRESENTING.value
        ctx.last_activity = time.time()
        await self._save(ctx)
        log.info(f"▶️  [{session_id[:8]}] Resumed from offset {offset}")
        return ctx

    # ── Interruption ──────────────────────────────────────────────────
    async def handle_interruption(self, session_id: str) -> Optional[SessionContext]:
        """
        When VAD detects student speaking during presentation.
        PRESENTING → LISTENING (valid)
        RESPONDING must NOT go directly to LISTENING
        """
        ctx = await self._load(session_id)
        if not ctx:
            return None

        if ctx.state in {DialogState.PRESENTING.value, DialogState.RESPONDING.value}:
            return await self.pause_session(
                session_id,
                slide_id=ctx.paused_state.get("slide_id"),
                char_offset=ctx.char_position,
            )
        else:
            log.debug(f"[{session_id[:8]}] Interrupt ignored (state={ctx.state})")

        return ctx

    # ── Navigation dans le cours ──────────────────────────────────────
    async def next_section(self, session_id: str) -> Optional[SessionContext]:
        ctx = await self._load(session_id)
        if not ctx:
            return None
        ctx.section_index  += 1
        ctx.char_position   = 0
        await self._save(ctx)
        return ctx

    async def prev_section(self, session_id: str) -> Optional[SessionContext]:
        ctx = await self._load(session_id)
        if not ctx:
            return None
        ctx.section_index  = max(0, ctx.section_index - 1)
        ctx.char_position  = 0
        await self._save(ctx)
        return ctx

    async def save_position(self, session_id: str, char_pos: int) -> None:
        """Sauvegarde la position de lecture pour reprendre après interruption."""
        ctx = await self._load(session_id)
        if ctx:
            ctx.char_position = char_pos
            needs_snapshot_update = bool(ctx.paused_state.get("presentation_text") and ctx.paused_state.get("slide_id"))
            if ctx.paused_state.get("presentation_text") and ctx.paused_state.get("slide_id"):
                ctx.paused_state["char_offset"] = char_pos
                ctx.paused_state["presentation_cursor"] = char_pos
            await self._save(ctx)

            if needs_snapshot_update:
                await self.save_presentation_snapshot(
                    session_id=session_id,
                    slide_id=str(ctx.paused_state.get("slide_id")),
                    presentation_text=ctx.paused_state.get("presentation_text") or "",
                    presentation_cursor=char_pos,
                    slide_title=str(ctx.paused_state.get("slide_title") or ""),
                )

    async def save_course_position(
        self,
        session_id: str,
        course_id: Optional[str] = None,
        chapter_index: Optional[int] = None,
        section_index: Optional[int] = None,
        char_pos: Optional[int] = None,
    ) -> None:
        """Sauvegarde la position courante du cours et de la lecture."""
        ctx = await self._load(session_id)
        if not ctx:
            return

        if course_id is not None:
            ctx.course_id = course_id
        if chapter_index is not None:
            ctx.chapter_index = max(0, chapter_index)
        if section_index is not None:
            ctx.section_index = max(0, section_index)
        if char_pos is not None:
            ctx.char_position = max(0, char_pos)

        ctx.last_activity = time.time()
        await self._save(ctx)

    async def save_presentation_snapshot(
        self,
        session_id: str,
        slide_id: str,
        presentation_text: str,
        presentation_cursor: int = 0,
        slide_title: str = "",
    ) -> None:
        """Persist the generated presentation text and current cursor in Redis."""
        if not slide_id:
            return

        payload = {
            "session_id": session_id,
            "slide_id": slide_id,
            "presentation_text": presentation_text or "",
            "presentation_cursor": max(0, presentation_cursor),
            "slide_title": slide_title or "",
            "presentation_text_len": len(presentation_text or ""),
            "updated_at": time.time(),
        }
        r = await get_redis()
        await r.setex(
            self._presentation_snapshot_key(session_id, slide_id),
            PRESENTATION_SNAPSHOT_TTL,
            json.dumps(payload, ensure_ascii=False),
        )

    async def load_presentation_snapshot(self, session_id: str, slide_id: str) -> Optional[dict]:
        """Load a previously cached presentation snapshot for a slide."""
        if not slide_id:
            return None

        r = await get_redis()
        raw = await r.get(self._presentation_snapshot_key(session_id, slide_id))
        if not raw:
            return None

        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return None

    async def save_tts_phrase_cache(
        self,
        text: str,
        audio_bytes: bytes,
        *,
        language: str,
        rate: str,
        provider: str,
        voice_name: str,
        mime: str = "audio/mpeg",
        metadata: Optional[dict] = None,
    ) -> str:
        """Cache a synthesized TTS phrase in Redis so repeated phrases are reused."""
        if not text or not audio_bytes:
            return ""

        payload = {
            "text": text,
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            "mime": mime or "audio/mpeg",
            "language": language,
            "rate": rate,
            "provider": provider,
            "voice_name": voice_name,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        cache_key = self._tts_cache_key(text, language, rate, provider, voice_name)
        r = await get_redis()
        await r.setex(cache_key, TTS_CACHE_TTL, json.dumps(payload, ensure_ascii=False))
        return cache_key

    async def load_tts_phrase_cache(
        self,
        text: str,
        *,
        language: str,
        rate: str,
        provider: str,
        voice_name: str,
    ) -> Optional[dict]:
        """Return cached TTS audio for a phrase if available."""
        if not text:
            return None

        cache_key = self._tts_cache_key(text, language, rate, provider, voice_name)
        r = await get_redis()
        raw = await r.get(cache_key)
        if not raw:
            return None

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None

            audio_b64 = data.get("audio_b64") or ""
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
            return {
                **data,
                "audio_bytes": audio_bytes,
                "cache_key": cache_key,
            }
        except Exception:
            return None

    async def get_resume_text(
        self, session_id: str, section_text: str
    ) -> str:
        """
        Retourne le texte restant à partir de la position de reprise.
        Ajoute une phrase de transition naturelle.
        """
        ctx = await self._load(session_id)
        if not ctx:
            return section_text

        pos = ctx.char_position
        if pos <= 0 or pos >= len(section_text):
            return section_text

        remaining = section_text[pos:].strip()
        transitions = {
            "fr": "Comme je le disais, ",
            "en": "As I was saying, ",
        }
        prefix = transitions.get(ctx.language, "Continuing... ")
        return prefix + remaining

    # ── Mise à jour de l'historique ───────────────────────────────────
    async def add_to_history(
        self, session_id: str, role: str, content: str
    ) -> None:
        ctx = await self._load(session_id)
        if ctx:
            ctx.add_to_history(role, content)
            ctx.total_turns += 1 if role == "assistant" else 0
            await self._save(ctx)

    # ── Détection d'incompréhension ───────────────────────────────────
    async def update_student_baseline(
        self, session_id: str, question: str, prosody: dict
    ) -> None:
        """
        ✅ NOUVEAU: Apprendre le profil étudiant pour adapter les seuils.
        
        Après chaque question, updater la baseline pour mieux prédire confusion.
        """
        ctx = await self._load(session_id)
        if not ctx:
            return
        
        baseline = ctx.student_baseline
        
        # Mise à jour exponentielle (nouvelles données pèsent moins que l'historique)
        alpha = 0.1  # learning rate

        new_speech_rate = prosody.get("speech_rate", baseline["avg_speech_rate"])
        baseline["avg_speech_rate"] = (
            baseline["avg_speech_rate"] * (1 - alpha) + new_speech_rate * alpha
        )
        
        new_q_len = len(question.split())
        baseline["avg_question_length"] = (
            baseline["avg_question_length"] * (1 - alpha) + new_q_len * alpha
        )
        
        new_hesitations = prosody.get("hesitation_count", 0)
        baseline["hesitation_baseline"] = (
            baseline["hesitation_baseline"] * (1 - alpha) + new_hesitations * alpha
        )
        
        baseline["turns_analyzed"] += 1
        
        if baseline["turns_analyzed"] % 5 == 0:  # Log tous les 5 turns
            log.info(
                f"[{session_id[:8]}] 📊 Student baseline updated: "
                f"speech_rate={baseline['avg_speech_rate']:.0f} wpm, "
                f"q_len={baseline['avg_question_length']:.1f} words, "
                f"hesitations={baseline['hesitation_baseline']:.1f}"
            )
        
        await self._save(ctx)

    def detect_confusion(self, text: str, language: str = "fr") -> tuple[bool, str]:
        """SIGHT-only confusion detection.

        Returns ``(is_confused, reason)`` where ``reason`` is either
        ``"sight_model"`` (the SIGHT classifier flagged the utterance as
        confused) or the empty string.

        # What used to be here

        Four stacked layers ran before SIGHT:
          0. ``feedback_negative`` / ``feedback_positive`` keyword dicts
             (FR/EN) — substring matching with hand-picked confidence.
          1. SIGHT model — kept.
          2. ``confusion_keywords`` keyword dict (FR/EN) — substring
             matching, ~25 hand-curated phrases.
          3. ``word_count <= 3 + "?"`` heuristic — magic threshold (why
             3 vs 2 vs 4? no empirical basis).

        All non-SIGHT layers were rule-based. They masked SIGHT's actual
        accuracy (every ``keyword`` hit short-circuited the model) and
        inflated false positives (any short question with "?" was tagged
        confused). Removing them leaves the calibrated learned classifier
        as the single source of truth — its threshold is loaded from the
        training bundle, not hand-picked.
        """
        t = (text or "").strip()
        if not t:
            return False, ""

        sight_predict = _get_sight_confusion_predictor()
        if sight_predict is None:
            # Model unavailable → fail closed (assume not confused).
            # The previous keyword/heuristic fallbacks were removed;
            # without SIGHT we have no learned signal to substitute.
            return False, ""

        prediction = sight_predict(t)
        if prediction is not None and bool(prediction.confused):
            return True, "sight_model"
        return False, ""

    # NOTE: ``detect_confusion_from_history`` was removed. It used five
    # hand-picked thresholds (``short_question_count >= 3``, ``len(q.split())
    # <= 3``, ``similarity > 0.6``, ``similar_count >= 2``,
    # ``len(user_questions) >= 7``) plus ``SequenceMatcher`` lexical ratio
    # to detect "confusion patterns" across the dialogue history. Every
    # threshold was a guess. SIGHT is the single source of truth now;
    # if turn-level confusion misses a slow-burning pattern, the right
    # fix is to train SIGHT on dialogue context, not to bolt on heuristics.

    def build_confusion_prompt(
        self,
        original_question: str,
        language: str,
        last_slide_content: str = "",
    ) -> str:
        """Back-compat wrapper around the module-level ``compose_reformulation_prompt``.
        The instance method exists so audio_pipeline / ws.py can call it
        through the DialogueManager singleton."""
        return compose_reformulation_prompt(
            original_question=original_question,
            language=language,
            last_slide_content=last_slide_content,
        )

    # NOTE: ``check_semantic_repetition`` was removed. It computed cosine
    # similarity between the current question and the last five user
    # turns and flagged confusion when ``max_sim >= 0.85``. Embeddings
    # are learned, but the 0.85 threshold was hand-picked. Question
    # repetition is also a noisy proxy for confusion (a curious student
    # may rephrase a question, an emphatic one may repeat it). SIGHT is
    # the single source of truth now.

    async def detect_and_track_confusion(
        self,
        session_id: str,
        question_text: str,
        language: str = "fr",
        history: list[dict] = None,        # accepted for back-compat, no longer used
        brain=None,                         # accepted for back-compat, no longer used
        prosody: dict = None,               # accepted for back-compat, no longer used
        on_state_change=None,
    ) -> tuple[bool, str, str, int]:
        """SIGHT-only confusion detection + Redis-side bookkeeping.

        Returns ``(is_confused, reason, q_hash, confusion_count)`` where
        ``reason`` is either ``"sight_model"`` or the empty string.

        # What used to be here

        Eight stacked detection layers (keywords, feedback patterns,
        SIGHT, hash repetition, history patterns, semantic repetition,
        prosody markers, adaptive thresholds). All non-SIGHT layers were
        either rule-based (keyword dicts, hash equality) or used
        hand-picked thresholds (similarity > 0.6 / 0.85, word_count <= 3,
        question_count >= 7, ...). They masked SIGHT's real performance
        and stacked false positives. SIGHT is now the single source of
        truth.

        ``history``, ``brain`` and ``prosody`` are kept in the signature
        so existing call sites in ``audio_pipeline.py`` and ``ws.py`` keep
        working without changes; they're ignored by the body. They can
        be wired back in if SIGHT is later trained on dialogue context
        or fused with acoustic features.
        """
        import hashlib, re, time

        start_ts = time.time()
        is_confused, reason = self.detect_confusion(question_text, language)

        # Hash for telemetry / dedup at the storage layer (not for
        # decision-making — the rule-based "if hash equals previous
        # then confused" branch was removed).
        normalized = re.sub(r"[^\w\s]", "", (question_text or "").lower()).strip()
        q_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]

        ctx = await self._load(session_id)
        if not ctx:
            log.warning(f"[{session_id[:8]}] Session not found for confusion detection")
            return (is_confused, reason, q_hash, 0)

        ctx.last_question_hash = q_hash
        if is_confused:
            ctx.confusion_count += 1
        await self._save(ctx)

        duration_ms = round((time.time() - start_ts) * 1000, 1)

        if on_state_change:
            try:
                await on_state_change("confusion_sight", {
                    "is_confused": is_confused,
                    "reason": reason,
                    "duration_ms": duration_ms,
                    "status": "complete",
                    "progress_pct": 70,
                })
            except Exception:
                pass

        if is_confused:
            log.info(
                f"[{session_id[:8]}] 🤔 SIGHT confusion detected "
                f"(total={ctx.confusion_count}, {duration_ms}ms)"
            )

        return (is_confused, reason, q_hash, ctx.confusion_count)

    # ── Clarification ────────────────────────────────────────────────
    async def mark_confusion_detected(self, session_id: str, reason: str = "") -> Optional[SessionContext]:
        """✅ Quand la confusion est détectée → transition vers CLARIFICATION"""
        ctx = await self._load(session_id)
        if not ctx:
            return None
        
        try:
            await self.transition(session_id, DialogState.CLARIFICATION)
            log.info(f"🤔 [{session_id[:8]}] CLARIFICATION needed: {reason}")
        except ValueError as e:
            log.warning(f"[{session_id[:8]}] Cannot transition to CLARIFICATION: {e}")
        
        return ctx
    
    async def resume_from_clarification(self, session_id: str) -> Optional[SessionContext]:
        """✅ Après la clarification, retour à l'écoute ou présentation"""
        ctx = await self._load(session_id)
        if not ctx:
            return None
        
        try:
            # Priorité : retour à LISTENING (étudiant peut poser d'autres questions)
            await self.transition(session_id, DialogState.LISTENING)
        except ValueError:
            try:
                # Sinon retour à PRESENTING
                await self.transition(session_id, DialogState.PRESENTING)
            except ValueError:
                log.warning(f"[{session_id[:8]}] Cannot resume from CLARIFICATION")
        
        return ctx

    # ── Stats ──────────────────────────────────────────────────────────
    async def get_stats(self, session_id: str) -> dict:
        ctx = await self._load(session_id)
        if not ctx:
            return {}
        return {
            "session_id":    session_id,
            "state":         ctx.state,
            "language":      ctx.language,
            "total_turns":   ctx.total_turns,
            "interruptions": ctx.interruptions,
            "chapter":       ctx.chapter_index,
            "section":       ctx.section_index,
            "char_position": ctx.char_position,
            "paused":        bool(ctx.paused_state.get("is_paused")),
            "pause_slide_id": ctx.paused_state.get("slide_id"),
            "pause_offset":  ctx.paused_state.get("char_offset", 0),
            "pause_presentation_key": ctx.paused_state.get("presentation_key"),
            "pause_presentation_len": ctx.paused_state.get("presentation_text_len", 0),
            "pause_has_presentation": bool(ctx.paused_state.get("presentation_text")),
            "history_len":   len(ctx.history),
            "uptime_min":    round((time.time() - ctx.created_at) / 60, 1),
        }
