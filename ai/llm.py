"""Smart Teacher — LLM Module (OpenAI GPT + Local LLM Fallback)"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import AsyncIterator, Optional

import requests
from openai import OpenAI

from core.config import Config
from ai.local_llm import LocalLLMFallback
from ai.prompt_rules import CROSSLANG_RULE_EN, CROSSLANG_RULE_FR

log = logging.getLogger("SmartTeacher.LLM")


# ── Slide-narration disk cache ────────────────────────────────────────
# brain.present() generates a slide narration via LLM (~5-30s). Without
# a cache, every revisit of the same slide pays this cost again. Key
# = md5(content+lang+level+domain+chapter+section); value = narration
# text. Lives next to the vision_describe cache.

_NARRATION_CACHE_DIR: Path = (
    Path(getattr(Config, "LOGS_DIR", "logs")).parent / "cache" / "narrations"
)
try:
    _NARRATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception as _exc:                                               # noqa: BLE001
    log.warning("Could not create narration cache dir %s: %s", _NARRATION_CACHE_DIR, _exc)


def _narration_key(
    content: str,
    lang: str,
    level: str,
    domain: Optional[str],
    chapter_title: str,
    section_title: str,
) -> str:
    h = hashlib.md5()
    parts = [content, lang, level, domain or "", chapter_title, section_title]
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def _narration_cache_read(key: str) -> Optional[str]:
    p = _NARRATION_CACHE_DIR / f"{key}.txt"
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").rstrip("\n")
    except Exception as exc:                                            # noqa: BLE001
        log.debug("narration cache read failed: %s", exc)
        return None


def _narration_cache_write(key: str, text: str) -> None:
    try:
        (_NARRATION_CACHE_DIR / f"{key}.txt").write_text(text, encoding="utf-8")
    except Exception as exc:                                            # noqa: BLE001
        log.debug("narration cache write failed: %s", exc)


_FALLBACK_PROMPTS = {
    "en": (
        "You are Smart Teacher, an AI tutor that adapts to the student's level. "
        "You ALWAYS answer as if you are SPEAKING in class - never writing. "
        "Be concise (max 4 natural sentences) unless more detail is requested. "
        "Use precise technical terminology without dumbing it down. "
        "NEVER use markdown, bullet points, or LaTeX. "
        "If an idea has already been stated, do not repeat it with slightly different wording. "
        "If the course content is in a different language than your answer, parse the source "
        "naturally and answer in the requested language; on first mention of a key technical "
        "term keep the slide's original wording in parentheses."
    ),
    "fr": (
        "Tu es Smart Teacher, un tuteur IA qui s'adapte au niveau de l'étudiant. "
        "Tu réponds TOUJOURS comme si tu PARLAIS en cours - jamais comme un texte écrit. "
        "Sois concis (max 4 phrases naturelles) sauf si plus de détails sont demandés. "
        "Utilise la terminologie technique précise. "
        "JAMAIS de markdown, listes, ni LaTeX. "
        "Si une idée a déjà été dite, ne la répète pas avec des mots proches. "
        "Si le contenu du cours est dans une langue différente de ta réponse, comprends la "
        "source naturellement et réponds dans la langue demandée ; à la première mention "
        "d'un terme technique clé, conserve l'écriture originale de la slide entre parenthèses."
    ),
}


_FALLBACK_PRESENTATION_PROMPTS = {
    "en": (
        "You are SMART TEACHER, an AI tutor presenting a slide to students. "
        "You comment on the slide content as a pedagogue, NOT as the slide's human author.\n"
        "- NEVER say 'I am [name]' or 'Welcome to [Course]'. You are not the slide author.\n"
        "- Keep acronyms exactly as the slide writes them. Do NOT invent expansions for "
        "acronyms you don't already know — that's a hallucination.\n"
        "- If a continuity block is provided, OPEN with one bridging sentence to the previous slide.\n"
        "- NEVER read word for word. Rephrase pedagogically.\n"
        "- ZERO markdown / LaTeX. 3 to 5 natural sentences. English only."
    ),
    "fr": (
        "Tu es SMART TEACHER, un tuteur IA qui présente une slide à des étudiants. "
        "Tu commentes le contenu en tant que pédagogue, PAS en tant qu'auteur humain.\n"
        "- NE JAMAIS dire 'Je suis [nom]' ou 'Bienvenue à [Cours]'. Tu n'es pas l'auteur.\n"
        "- Conserve les acronymes EXACTEMENT comme la slide les écrit. NE PAS inventer "
        "d'expansion pour un acronyme que tu ne connais pas — c'est une hallucination.\n"
        "- Si un bloc continuité est fourni, OUVRE par une phrase de transition.\n"
        "- NE LIS JAMAIS mot pour mot. Reformule pédagogiquement.\n"
        "- ZÉRO markdown / LaTeX. 3 à 5 phrases naturelles. Français uniquement."
    ),
}

def get_system_prompt(domain: str = None, language: str = "en") -> str:
    """Génère un prompt système dynamique basé sur le domaine."""
    try:
        from core.domains_config import DOMAINS
    except ImportError:
        log.warning("⚠️  domains_config not available, using fallback prompts")
        return _FALLBACK_PROMPTS.get(language, _FALLBACK_PROMPTS["en"])

    lang = language.lower()[:2] if language else "en"

    domain_meta = DOMAINS.get(domain) if domain else None
    if isinstance(domain_meta, dict):
        domain_desc = domain_meta.get("description", "specialty subjects")
        domain_name = domain_meta.get("name", domain)
    elif domain:
        domain_desc = "specialty subjects"
        domain_name = domain
    else:
        domain_desc = "specialty subjects"
        domain_name = "diverse topics"

    # Math notation verbalization is delegated to ``audio/math_speech.py``
    # (deterministic post-processor with full FR/EN unicode lexicon +
    # comparison operators + LaTeX command rewriting). The prompt only
    # needs a short reminder that math goes in plain words — exhaustive
    # rules in the prompt would just duplicate that module's logic and
    # waste tokens on every call.
    math_reminder_en = (
        " Math notation must be written in plain spoken English: no LaTeX, "
        "no dollar signs, no caret/underscore markers."
    )
    math_reminder_fr = (
        " Les notations mathématiques doivent être en français parlé : pas de LaTeX, "
        "pas de dollars, pas d'indices/exposants typographiques."
    )

    # Cross-language rule (shared with presentation prompt + rewriter
    # via ai.prompt_rules — single source of truth). The leading space
    # matches the previous in-prompt style: concatenated after the rest
    # of the system rule.
    crosslang_rule_en = " " + CROSSLANG_RULE_EN
    crosslang_rule_fr = " " + CROSSLANG_RULE_FR

    # Course-bound rule — Smart Teacher is a tutor restricted to the
    # course material, NOT a generic chatbot. When a course context is
    # provided, the answer MUST come from it; if the question is
    # off-topic, the model must refuse honestly instead of paraphrasing
    # general/Wikipedia knowledge. Stronger phrasing in the responder
    # prompt itself, but we also mirror it here so the system rule
    # cannot quietly override the user-level rule.
    course_bound_rule_en = (
        " COURSE-BOUND RULE: you only answer from the course material provided in "
        "this conversation (slides, course context, retrieved chunks). You are NOT "
        "an encyclopedic chatbot — do not pad answers with general knowledge or "
        "Wikipedia-style content. If the question is off-topic relative to that "
        "material, say so honestly (e.g. 'this point is not covered in this "
        "course') rather than answering from outside knowledge."
    )
    course_bound_rule_fr = (
        " RÈGLE TUTEUR DE COURS : tu ne réponds QU'À PARTIR du matériel de cours "
        "fourni dans cette conversation (slides, contexte de cours, morceaux "
        "récupérés). Tu n'es PAS un chatbot encyclopédique — n'enrichis pas tes "
        "réponses avec des connaissances générales ou de type Wikipédia. Si la "
        "question est hors-sujet par rapport à ce matériel, dis-le honnêtement "
        "(ex : 'ce point n'est pas abordé dans ce cours') plutôt que de répondre "
        "depuis des connaissances extérieures."
    )

    prompts_map = {
        "en": (
            f"You are Smart Teacher, an AI tutor specialised in {domain_desc}. "
            f"You adapt your level of detail and vocabulary to the student you are speaking to "
            f"(school, undergraduate, Master, doctoral — whichever the session indicates). "
            f"You ALWAYS answer as if you are SPEAKING in class - never writing. "
            f"Be concise (max 4 natural sentences) unless more detail is requested. "
            f"Use precise technical terminology appropriate to {domain_name} without dumbing it down. "
            f"NEVER use markdown, bullet points, or LaTeX. "
            f"Keep acronyms exactly as the slide writes them — do NOT invent expansions for "
            f"acronyms you don't already know. "
            f"If the same idea appears more than once, merge it into one explanation."
            f"{course_bound_rule_en}"
            f"{crosslang_rule_en}"
            f"{math_reminder_en}"
        ),
        "fr": (
            f"Tu es Smart Teacher, un tuteur IA expert en {domain_desc}. "
            f"Tu adaptes ton niveau de détail et ton vocabulaire à l'étudiant à qui tu parles "
            f"(lycée, licence, Master, doctorat — selon ce que la session indique). "
            f"Tu réponds TOUJOURS comme si tu PARLAIS en cours - jamais comme un texte écrit. "
            f"Sois concis (max 4 phrases naturelles) sauf si plus de détails sont demandés. "
            f"Utilise la terminologie technique précise en {domain_name}. "
            f"JAMAIS de markdown, listes, ni LaTeX. "
            f"Conserve les acronymes EXACTEMENT comme la slide les écrit — N'INVENTE PAS "
            f"d'expansion pour un acronyme que tu ne connais pas. "
            f"Si une même idée apparaît plusieurs fois, fusionne-la en une seule explication."
            f"{course_bound_rule_fr}"
            f"{crosslang_rule_fr}"
            f"{math_reminder_fr}"
        ),
    }

    return prompts_map.get(lang, prompts_map["en"])


def get_presentation_prompt(
    domain: str = None,
    language: str = "en",
    chapter_title: str = "",
    student_level: str = "",
) -> str:
    """Génère un prompt de présentation centré sur la slide courante."""
    try:
        from core.domains_config import DOMAINS
    except ImportError:
        log.warning("⚠️  domains_config not available, using fallback presentation prompts")
        return _FALLBACK_PRESENTATION_PROMPTS.get(language, _FALLBACK_PRESENTATION_PROMPTS["en"])

    lang = language.lower()[:2] if language else "en"

    domain_meta = DOMAINS.get(domain) if domain else None
    if isinstance(domain_meta, dict):
        domain_name = domain_meta.get("name", domain)
    elif domain:
        domain_name = domain
    else:
        domain_name = "this field"

    chapter_ctx = f"\nThis content is from the chapter: '{chapter_title}'." if chapter_title else ""

    # Audience descriptor — derived from student_level so the prompt
    # adapts to who is listening (lycée / université / Master / etc.)
    # without hardcoding "M2". Defaults to a generic phrasing.
    level_norm = (student_level or "").strip().lower()
    if "lycée" in level_norm or "lycee" in level_norm or "high school" in level_norm:
        audience_en = "high-school students"
        audience_fr = "lycéens"
    elif "master" in level_norm or "m2" in level_norm or "m1" in level_norm:
        audience_en = "Master's level students"
        audience_fr = "étudiants en Master"
    elif "doctor" in level_norm or "phd" in level_norm or "doctorat" in level_norm:
        audience_en = "doctoral students"
        audience_fr = "doctorants"
    elif "univers" in level_norm or "license" in level_norm or "licence" in level_norm:
        audience_en = "university students"
        audience_fr = "étudiants universitaires"
    elif level_norm:
        audience_en = level_norm
        audience_fr = level_norm
    else:
        audience_en = "students"
        audience_fr = "étudiants"

    prompts_map = {
        "en": (
            f"You are SMART TEACHER, an AI tutor specialised in {domain_name}, presenting a lecture to {audience_en}. "
            f"You explain the slide content as a teacher commenting on the material, NOT as the human author of the slides.{chapter_ctx}\n\n"
            "IDENTITY RULES (critical — break these and you fail):\n"
            "- You are 'Smart Teacher'. NEVER impersonate the slide author.\n"
            "- NEVER say 'I am [name]', 'My name is...', 'I am your instructor', or 'Welcome to [Course Name]'.\n"
            "- If the slide contains an instructor's name, an office number, a university affiliation "
            "or other identifying contact info, DO NOT repeat them — those belong to the slide's author, "
            "not to you. Mention briefly that the course is from such-and-such institution only if it "
            "actually helps the student situate the material, then move on to the content.\n"
            "- DO NOT open every slide with 'Welcome' or 'Today we will'. Only the first slide of the "
            "session may use a single-sentence opening; subsequent slides must continue from the previous one.\n\n"
            "TERMINOLOGY RULES (critical):\n"
            "- Keep acronyms EXACTLY as the slide writes them. Do NOT invent or guess an expansion. "
            "If you don't know what an acronym stands for, just use it as-is — the slide author chose it. "
            "An invented expansion for an unfamiliar acronym is a hallucination and worse than saying nothing.\n"
            "- Keep proper nouns and brand names verbatim (no transliteration, no 'translation').\n"
            "- " + CROSSLANG_RULE_EN + "\n\n"
            "CONTINUITY RULES:\n"
            "- If a 🔗 CONTINUITY block is provided in the input, OPEN with ONE bridging sentence "
            "linking the previous concept to the current one (e.g. 'Having just covered X, "
            "we now turn to Y'). Do NOT re-define the previous concept.\n"
            "- If no continuity block is given, this is the FIRST slide — open with a brief topical hook.\n\n"
            "PRESENTATION RULES:\n"
            "- Focus ONLY on the current slide. Do not preview future slides.\n"
            "- NEVER read the text word for word — rephrase in your own pedagogical words.\n"
            "- ZERO markdown: no **, no #, no bullets, no lists.\n"
            "- Keep ALL technical terminology from this domain.\n"
            "- If the slide repeats the same idea, synthesise it once.\n"
            "- 3 to 5 natural sentences. Only in English.\n\n"
            "MATH NOTATION:\n"
            "- Math is delegated to a deterministic post-processor — your job is just to "
            "write math in plain spoken English. NEVER produce raw LaTeX, dollar signs, "
            "carets (^), underscores (_), or curly braces in your output."
        ),
        "fr": (
            f"Tu es SMART TEACHER, un tuteur IA spécialisé en {domain_name}, qui présente un cours à des {audience_fr}. "
            f"Tu expliques le contenu de la slide en tant que pédagogue qui commente le matériel, "
            f"PAS en tant qu'auteur humain des slides.{chapter_ctx}\n\n"
            "RÈGLES D'IDENTITÉ (critiques — les enfreindre = échec) :\n"
            "- Tu es 'Smart Teacher'. NE JAMAIS te faire passer pour l'auteur de la slide.\n"
            "- NE JAMAIS dire 'Je suis [nom]', 'Je m'appelle...', 'Je suis votre enseignant', "
            "'Bienvenue à [Nom du Cours]'.\n"
            "- Si la slide mentionne un nom d'enseignant, un numéro de bureau, une affiliation "
            "universitaire ou autres coordonnées, NE LES RÉPÈTE PAS — ce sont les infos "
            "de l'auteur, pas les tiennes. Mentionne brièvement l'institution seulement si "
            "ça aide l'étudiant à situer le matériel, puis passe au contenu.\n"
            "- N'ouvre PAS chaque slide par 'Bienvenue' ou 'Aujourd'hui nous allons'. "
            "Seule la PREMIÈRE slide de la session peut avoir une ouverture ; les suivantes "
            "doivent ENCHAÎNER avec la précédente.\n\n"
            "RÈGLES TERMINOLOGIQUES (critiques) :\n"
            "- Conserve les acronymes EXACTEMENT comme la slide les écrit. NE PAS inventer "
            "ou deviner une expansion. Si tu ne sais pas ce que signifie un acronyme, "
            "utilise-le tel quel — l'auteur de la slide l'a choisi. Une expansion inventée "
            "pour un acronyme inconnu est une hallucination, et pire que de ne rien dire.\n"
            "- Conserve les noms propres et marques verbatim (pas de translittération, pas de 'traduction').\n"
            "- " + CROSSLANG_RULE_FR + "\n\n"
            "RÈGLES DE CONTINUITÉ :\n"
            "- Si un bloc 🔗 CONTINUITÉ est fourni en entrée, OUVRE par UNE phrase de transition "
            "qui relie le concept précédent au nouveau (ex: 'Après avoir vu X, "
            "nous abordons maintenant Y'). NE re-définis PAS le concept précédent.\n"
            "- Si aucun bloc de continuité n'est fourni, c'est la PREMIÈRE slide — "
            "ouvre par une accroche topique brève.\n\n"
            "RÈGLES DE PRÉSENTATION :\n"
            "- Focus UNIQUEMENT sur la slide courante. Ne pas anticiper les slides suivantes.\n"
            "- NE JAMAIS lire mot pour mot — reformule avec tes propres mots pédagogiques.\n"
            "- ZÉRO markdown : pas de **, pas de #, pas de tirets, pas de listes.\n"
            "- Conserve TOUS les termes techniques du domaine.\n"
            "- Si la slide répète la même idée, synthétise une fois.\n"
            "- 3 à 5 phrases naturelles. Uniquement en français.\n\n"
            "NOTATION MATHÉMATIQUE :\n"
            "- Le math est traité par un post-processeur déterministe — ton job est juste "
            "d'écrire les notations en français parlé. NE PRODUIS JAMAIS de LaTeX brut, "
            "dollars, accents circonflexes (^), underscores (_) ou accolades dans ta sortie."
        ),
    }

    return prompts_map.get(lang, prompts_map["en"])


def _extract_json_payload(raw_text: str) -> dict[str, object] | None:
    if not raw_text:
        return None

    cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


class Brain:
    def __init__(self):
        self.client: OpenAI | None = None
        self.fallback: LocalLLMFallback | None = None
        self.history: list[dict] = []
        self.max_history_len = Config.MAX_HISTORY_TURNS * 2
        
        # ✅ Rate limiting: per-session throttler (session_id -> {last_call_time, call_count, minute_reset_time})
        self.session_throttlers: dict[str, dict] = {}
        self.min_call_interval = 1.0  # Minimum 1 second between calls per session
        self.max_calls_per_minute = 10  # Maximum 10 calls per minute per session

        # Honour the DISABLE_OPENAI kill-switch: skip the client init
        # entirely so the rest of the file (which checks ``self.client``
        # on every code path) routes straight to Ollama.
        if Config.DISABLE_OPENAI:
            log.info("ℹ️ DISABLE_OPENAI=true → Brain runs Ollama-only")
        elif Config.OPENAI_API_KEY:
            try:
                self.client = OpenAI(
                    api_key=Config.OPENAI_API_KEY,
                    base_url=Config.OPENAI_BASE_URL,
                    max_retries=0,
                )
                _host = Config.OPENAI_BASE_URL or "api.openai.com (default)"
                log.info(f"✅ OpenAI client ready → {_host}")
            except Exception as exc:
                log.error(f"❌ OpenAI erreur: {exc}")
        else:
            log.info("ℹ️ OpenAI non configuré (fallback Ollama utilisé)")

        # Chaîne LLM : OpenAI → Ollama. Modèle pris dans Config.OLLAMA_TEXT_MODEL.
        self.fallback = LocalLLMFallback(model=Config.OLLAMA_TEXT_MODEL)

    @staticmethod
    def _should_disable_openai(exc: Exception) -> bool:
        message = f"{exc.__class__.__module__}:{exc.__class__.__name__}:{exc}".lower()
        return any(
            token in message
            for token in (
                "insufficient_quota",
                "quota",
                "429",
                "rate limit",
                "ratelimit",
                "authentication",
                "unauthorized",
                "invalid_api_key",
            )
        )

    def _disable_openai(self, reason: str) -> None:
        if self.client is not None:
            self.client = None
            log.info("ℹ️ OpenAI désactivé pour cette session (%s) → Ollama prioritaire", reason)

    def clear_memory(self):
        self.history = []
        log.info("🧠 Mémoire effacée")
    
    def _check_rate_limit(self, session_id: str | None = None) -> tuple[bool, str]:
        """
        ✅ Check if LLM call is allowed for this session.
        
        Returns:
            (allowed: bool, reason: str)
            - If allowed, reason is empty string
            - If not allowed, reason explains why
        """
        if not session_id:
            # No session_id provided, always allow (for backward compatibility)
            return True, ""
        
        now = time.time()
        
        if session_id not in self.session_throttlers:
            # First call for this session
            self.session_throttlers[session_id] = {
                "last_call_time": now,
                "call_count": 0,
                "minute_reset_time": now,
            }
            return True, ""
        
        throttler = self.session_throttlers[session_id]
        
        # 1. Check minimum interval between calls (1 second)
        time_since_last_call = now - throttler["last_call_time"]
        if time_since_last_call < self.min_call_interval:
            wait_time = self.min_call_interval - time_since_last_call
            reason = f"Rate limited: wait {wait_time:.1f}s (min interval: {self.min_call_interval}s)"
            return False, reason
        
        # 2. Check per-minute call limit
        time_since_minute_reset = now - throttler["minute_reset_time"]
        if time_since_minute_reset > 60:
            # Reset minute counter
            throttler["call_count"] = 0
            throttler["minute_reset_time"] = now
        
        if throttler["call_count"] >= self.max_calls_per_minute:
            reason = f"Per-minute limit reached ({self.max_calls_per_minute} calls/min)"
            return False, reason
        
        # All checks passed - update throttler
        throttler["last_call_time"] = now
        throttler["call_count"] += 1
        return True, ""


    # Class-level lock that serialises ALL Ollama calls process-wide.
    # Ollama on CPU is single-stream per model — concurrent /api/generate
    # calls queue inside Ollama anyway, but they hold connections and
    # confuse callers' timing. With this lock, only one Ollama call is
    # in flight at a time; the others wait their turn cleanly. Combined
    # with timeout=None, this means a slow call doesn't fail under load,
    # it just delays the next one.
    _ollama_lock = __import__("threading").Lock()

    def _call_ollama_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ):
        """Stream tokens from Ollama's /api/generate.

        Yields each token chunk as it arrives. The caller can forward
        chunks to the WebSocket client incrementally so the student
        sees the answer materialise word-by-word instead of waiting
        for the full response.

        Why streaming matters on CPU
        ----------------------------
        On a 60-second mistral response, the student would otherwise
        stare at a spinner for 60 seconds. With streaming, the first
        word lands in ~1-2s and the answer ticks in steadily — the
        same total wait, but radically better UX (students perceive
        latency as the time to FIRST byte, not last byte).

        The HTTP connection is held inside the global ``_ollama_lock``
        because Ollama is single-stream per model and we don't want
        two callers fighting for tokens.
        """
        if not self.fallback or not self.fallback.available:
            return
        with Brain._ollama_lock:
            log.info("🖥️ Appel Ollama (stream)...")
            payload = {
                "model": self.fallback.model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": True,
            }
            try:
                response = requests.post(
                    f"{self.fallback.base_url}/api/generate",
                    json=payload,
                    timeout=None,
                    stream=True,
                )
                if response.status_code != 200:
                    log.error(f"❌ Ollama stream HTTP {response.status_code}")
                    return
                full_text_chars = 0
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        full_text_chars += len(token)
                        yield token
                    if chunk.get("done"):
                        break
                log.info(f"✅ Ollama stream done ({full_text_chars} chars)")
            except Exception as exc:                                        # noqa: BLE001
                log.error(f"❌ Ollama stream failed: {exc}")
                return

    def _call_ollama_sync(self, prompt: str, temperature: float = 0.7, max_tokens: int = 400) -> str | None:
        """Appel synchrone à Ollama via HTTP, sérialisé par lock global."""
        if not self.fallback or not self.fallback.available:
            return None

        with Brain._ollama_lock:
            try:
                log.info("🖥️ Appel Ollama synchrone...")
                payload = {
                    "model": self.fallback.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": False,
                }
                response = requests.post(
                    f"{self.fallback.base_url}/api/generate",
                    json=payload,
                    timeout=None,
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "").strip()
                    if answer:
                        log.info(f"✅ Ollama réponse : {len(answer)} chars")
                        return answer
                    log.warning("⚠️  Ollama réponse vide")
                    return None

                log.error(f"❌ Ollama HTTP {response.status_code}")
                return None
            except requests.exceptions.Timeout:
                log.error("❌ Ollama request failed or was interrupted")
                return None
            except requests.exceptions.ConnectionError:
                log.error("❌ Ollama connexion échouée — Lancez: docker-compose up -d ollama")
                return None
            except Exception as exc:
                log.error(f"❌ Ollama erreur : {exc}")
                return None

    def ask(
        self,
        question: str,
        course_context: str = "",
        reply_language: str | None = None,
        chapter_idx: int | None = None,
        chapter_title: str = "",
        section_title: str = "",
        domain: str | None = None,
        session_id: str | None = None,  # ✅ For rate limiting
    ) -> tuple[str, float]:
        """Répond à une question de l'étudiant."""
        # ✅ Check rate limit before proceeding
        allowed, reason = self._check_rate_limit(session_id)
        if not allowed:
            log.warning(f"[{session_id[:8] if session_id else 'NA'}] ⚠️  {reason}")
            return "Trop rapide! Attendez une seconde antes de poser une autre question.", 0.0
        
        start = time.time()
        lang = (reply_language or "en").lower()[:2]

        system_content = get_system_prompt(domain, lang)

        ch_ctx = ""
        if chapter_title:
            if lang == "en":
                ch_ctx = f"\nWe are currently in: '{chapter_title}'."
            elif lang == "fr":
                ch_ctx = f"\nNous sommes actuellement dans : '{chapter_title}'."
            else:
                ch_ctx = f"\nنحن الآن في : '{chapter_title}'."
            if section_title:
                ch_ctx += f" Section: '{section_title}'." if lang == "en" else f" Section : '{section_title}'."

        system_content += ch_ctx

        if course_context:
            sep = "─" * 40
            system_content += f"\n\n{sep}\nCOURSE CONTEXT:\n{course_context}\n{sep}"

        messages = [
            {"role": "system", "content": system_content},
            *self.history,
            {"role": "user", "content": question},
        ]

        # 1️⃣  OpenAI (LLM principal)
        if self.client:
            try:
                log.info("🤖 Tentative OpenAI...")
                response = self.client.chat.completions.create(
                    model=Config.GPT_MODEL,
                    messages=messages,
                    temperature=Config.GPT_TEMPERATURE,
                    max_tokens=Config.GPT_MAX_TOKENS,
                )
                answer = self._clean_for_speech(response.choices[0].message.content)
                answer = self._dedupe_answer_text(answer)
                self.history.append({"role": "user", "content": question})
                self.history.append({"role": "assistant", "content": answer})
                if len(self.history) > self.max_history_len:
                    self.history = self.history[2:]
                duration = time.time() - start
                log.info(f"✅ OpenAI OK | {duration:.2f}s | lang={lang} | {len(answer)} chars")
                return answer, duration
            except Exception as openai_err:
                if self._should_disable_openai(openai_err):
                    self._disable_openai(str(openai_err))
                log.warning(f"⚠️  OpenAI échoué: {openai_err} → Tentative Ollama...")

        # 2️⃣  Ollama (fallback local)
        if self.fallback and self.fallback.available:
            log.info("🖥️ Ollama fallback activé (sans timeout)...")
            lang_instruction = {
                "en": "\n\n[!!!CRITICAL!!!] You MUST respond ONLY in English. Any response in French or other languages is forbidden. ONLY English.",
                "fr": "\n\n[!!!CRITIQUE!!!] Tu DOIS répondre UNIQUEMENT en français. Aucune réponse en anglais ou autre langue. SEULEMENT du français.",
            }
            fallback_prompt = f"{system_content}{lang_instruction.get(lang, lang_instruction['en'])}\n\nQuestion/Prompt: {question}"
            answer = self._call_ollama_sync(
                prompt=fallback_prompt,
                temperature=Config.GPT_TEMPERATURE,
                max_tokens=250,
            )
            if answer:
                answer = self._clean_for_speech(answer)
                answer = self._dedupe_answer_text(answer)
                self.history.append({"role": "user", "content": question})
                self.history.append({"role": "assistant", "content": answer})
                if len(self.history) > self.max_history_len:
                    self.history = self.history[2:]
                duration = time.time() - start
                log.info(f"✅ Ollama OK | {duration:.2f}s | {len(answer)} chars")
                return answer, duration

        log.error("❌ LLM indisponible (OpenAI + Ollama échoué)")
        return "Je n'ai pas pu générer de réponse pour le moment. Réessayez dans un instant.", time.time() - start

    async def ask_stream(
        self,
        question: str,
        course_context: str = "",
        reply_language: str | None = None,
        chapter_title: str = "",
        section_title: str = "",
        domain: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Async generator yielding raw token chunks for Q&A.

        Streams from OpenAI when available, falls back to Ollama. Caller
        accumulates chunks for TTS / history; partial chunks can be sent
        to the client as ``answer_text_partial`` events so the student
        sees the answer materialise word-by-word.

        Why a queue + thread bridge
        ---------------------------
        The OpenAI SDK and our ``_call_ollama_stream`` are both blocking
        sync generators (they hold ``requests`` connections). We can't
        ``await`` them directly from asyncio code. Pattern used here:
        run the producer in a worker thread and pipe tokens through an
        ``asyncio.Queue`` that the async caller drains.
        """
        allowed, _ = self._check_rate_limit(session_id)
        if not allowed:
            yield "Trop rapide! Attendez une seconde antes de poser une autre question."
            return

        lang = (reply_language or "en").lower()[:2]
        system_content = get_system_prompt(domain, lang)

        if chapter_title:
            if lang == "en":
                system_content += f"\nWe are currently in: '{chapter_title}'."
            elif lang == "fr":
                system_content += f"\nNous sommes actuellement dans : '{chapter_title}'."
            if section_title:
                system_content += (
                    f" Section: '{section_title}'." if lang == "en"
                    else f" Section : '{section_title}'."
                )

        if course_context:
            sep = "─" * 40
            system_content += f"\n\n{sep}\nCOURSE CONTEXT:\n{course_context}\n{sep}"

        messages = [
            {"role": "system", "content": system_content},
            *self.history,
            {"role": "user", "content": question},
        ]

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()
        full_chunks: list[str] = []

        def _push(item):
            asyncio.run_coroutine_threadsafe(queue.put(item), loop)

        def producer():
            # 1) OpenAI streaming
            if self.client:
                try:
                    log.info("🤖 Tentative OpenAI (stream)...")
                    stream = self.client.chat.completions.create(
                        model=Config.GPT_MODEL,
                        messages=messages,
                        temperature=Config.GPT_TEMPERATURE,
                        max_tokens=Config.GPT_MAX_TOKENS,
                        stream=True,
                    )
                    for chunk in stream:
                        try:
                            token = chunk.choices[0].delta.content or ""
                        except Exception:
                            token = ""
                        if token:
                            full_chunks.append(token)
                            _push(token)
                    _push(SENTINEL)
                    return
                except Exception as openai_err:
                    if self._should_disable_openai(openai_err):
                        self._disable_openai(str(openai_err))
                    log.warning(f"⚠️  OpenAI stream échoué: {openai_err} → Ollama stream...")

            # 2) Ollama streaming fallback
            if self.fallback and self.fallback.available:
                lang_instruction = {
                    "en": "\n\n[!!!CRITICAL!!!] You MUST respond ONLY in English.",
                    "fr": "\n\n[!!!CRITIQUE!!!] Tu DOIS répondre UNIQUEMENT en français.",
                }
                fallback_prompt = (
                    f"{system_content}{lang_instruction.get(lang, lang_instruction['en'])}"
                    f"\n\nQuestion/Prompt: {question}"
                )
                try:
                    for token in self._call_ollama_stream(
                        prompt=fallback_prompt,
                        temperature=Config.GPT_TEMPERATURE,
                        max_tokens=500,
                    ):
                        if token:
                            full_chunks.append(token)
                            _push(token)
                except Exception as exc:
                    log.error(f"❌ Ollama stream producer failed: {exc}")
                _push(SENTINEL)
                return

            log.error("❌ ask_stream: aucun LLM disponible")
            _push(SENTINEL)

        t = threading.Thread(target=producer, name="brain-ask-stream", daemon=True)
        t.start()

        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                yield item
        finally:
            # Update history with the full accumulated answer (cleaned)
            if full_chunks:
                full_text = "".join(full_chunks)
                cleaned = self._clean_for_speech(full_text)
                cleaned = self._dedupe_answer_text(cleaned)
                self.history.append({"role": "user", "content": question})
                self.history.append({"role": "assistant", "content": cleaned})
                if len(self.history) > self.max_history_len:
                    self.history = self.history[2:]

    def label_confusion(
        self,
        text: str,
        domain: str = "",
        module: str = "",
        language: str = "en",
    ) -> tuple[str, float, str]:
        """Label a student text as confused or not_confused using the LLM."""
        text = (text or "").strip()
        if not text:
            return "not_confused", 0.0, "empty input"

        lang = (language or "en").lower()[:2]
        system_prompts = {
            "en": (
                "You are a strict annotation assistant for the Smart Teacher dataset. "
                "Decide whether the text expresses confusion. "
                "confused = explicit lack of understanding, request to re-explain, says lost/confused/stuck, or cannot follow the explanation. "
                "not_confused = normal question, factual question, statement of understanding, neutral remark, or administrative text. "
                "A question mark alone does NOT mean confused. "
                "Return only JSON with keys label, confidence, and reason. "
                "label must be confused or not_confused. confidence must be a number between 0 and 1."
            ),
            "fr": (
                "Tu es un annotateur strict pour le dataset Smart Teacher. "
                "Decide si le texte exprime une confusion. "
                "confused = manque de comprehension explicite, demande de reexplication, texte perdu, confus, bloque, ou incapacite a suivre l'explication. "
                "not_confused = question normale, question factuelle, phrase de comprehension, remarque neutre, ou texte administratif. "
                "Un point d'interrogation seul ne veut pas dire confused. "
                "Retourne uniquement du JSON avec les cles label, confidence et reason. "
                "label doit etre confused ou not_confused. confidence doit etre un nombre entre 0 et 1."
            ),
        }
        system_prompt = system_prompts.get(lang, system_prompts["en"])

        user_prompt = (
            f"Text: {text}\n"
            f"Language: {lang}\n"
            f"Domain: {domain or 'general'}\n"
            f"Module: {module or 'general'}\n\n"
            "Return one JSON object only."
        )

        raw_response = None

        # 1️⃣  OpenAI (LLM principal)
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=Config.GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=120,
                )
                raw_response = (response.choices[0].message.content or "").strip()
            except Exception as openai_err:
                if self._should_disable_openai(openai_err):
                    self._disable_openai(str(openai_err))
                log.warning(f"⚠️  LLM labeling OpenAI failed: {openai_err} → fallback Ollama...")

        # 2️⃣  Ollama (fallback local)
        if not raw_response and self.fallback and self.fallback.available:
            raw_response = self._call_ollama_sync(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                temperature=0.0,
                max_tokens=120,
            )

        if raw_response:
            payload = _extract_json_payload(raw_response)
            if payload:
                label = str(payload.get("label", "")).strip().lower().replace(" ", "_").replace("-", "_")
                if label in {"confused", "not_confused"}:
                    try:
                        confidence = float(payload.get("confidence", 0.0))
                    except Exception:
                        confidence = 0.0
                    confidence = max(0.0, min(1.0, confidence))
                    reason = str(payload.get("reason", "")).strip() or "LLM label"
                    return label, confidence, reason

            normalized = raw_response.lower()
            if "not_confused" in normalized or "not confused" in normalized:
                return "not_confused", 0.55, "parsed from raw LLM response"
            if "confused" in normalized:
                return "confused", 0.55, "parsed from raw LLM response"

        # LLM (OpenAI + Ollama) both unavailable or returned unparseable
        # output. Fail closed (assume not_confused) — the previous
        # keyword-based ``detect_confusion`` fallback was removed.
        return "not_confused", 0.0, "LLM unavailable, no fallback signal"

    def present(
        self,
        section_content: str,
        language: str = "en",
        student_level: str = "université",
        chapter_idx: int | None = None,
        chapter_title: str = "",
        section_title: str = "",
        domain: str | None = None,
        session_id: str | None = None,  # ✅ For rate limiting
    ) -> tuple[str, float]:
        """Présente une slide ou section de cours oralement."""
        # ✅ Check rate limit before proceeding
        allowed, reason = self._check_rate_limit(session_id)
        if not allowed:
            log.warning(f"[{session_id[:8] if session_id else 'NA'}] ⚠️  {reason}")
            return "", 0.0
        
        if not section_content or not section_content.strip():
            log.warning("⚠️  Slide content vide — rien à expliquer")
            return "", 0.0

        if not self.client and not (self.fallback and self.fallback.available):
            log.error("❌ No LLM available — returning raw content")
            return self._clean_for_speech(section_content), 0.0

        start = time.time()
        lang = (language or "en").lower()[:2]

        # Disk cache hit → skip LLM entirely. Same content + same language +
        # same level/domain/chapter/section produces the same narration, so
        # revisiting a slide returns instantly instead of re-paying 5-30s.
        cache_key = _narration_key(
            section_content, lang, student_level, domain,
            chapter_title or "", section_title or "",
        )
        cached = _narration_cache_read(cache_key)
        if cached is not None:
            log.info(f"✅ present cache HIT | {len(cached)} chars | key={cache_key[:8]}")
            return cached, time.time() - start

        # Level is now baked into the prompt via student_level — the
        # audience descriptor inside get_presentation_prompt adapts the
        # opening line ("teaching to lycéens / Master's / doctoral...")
        # without hardcoded if/else branches.
        system_content = get_presentation_prompt(
            domain, lang, chapter_title, student_level=student_level,
        )

        # 1️⃣  OpenAI (LLM principal)
        if self.client:
            try:
                log.info("🤖 Presentation: OpenAI...")
                response = self.client.chat.completions.create(
                    model=Config.GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": f"Content to present:\n\n{section_content}"},
                    ],
                    temperature=0.7,
                    # Bumped from 400 → 700 because slides with a definition
                    # plus a worked example (e.g. Trimmed Mean, Median) were
                    # consistently truncating mid-explanation, producing
                    # cliffhanger narrations like "…we will define next."
                    # 700 fits a 4-5 idea plan without leaving change on the
                    # table for slides that already finish under budget.
                    max_tokens=700,
                )
                answer = self._clean_for_speech(response.choices[0].message.content.strip())
                answer = self._dedupe_answer_text(answer)
                duration = time.time() - start
                if answer:
                    _narration_cache_write(cache_key, answer)
                log.info(f"✅ OpenAI present OK | {duration:.2f}s | ch={chapter_idx} | {len(answer)} chars")
                return answer, duration
            except Exception as openai_err:
                if self._should_disable_openai(openai_err):
                    self._disable_openai(str(openai_err))
                log.warning(f"⚠️  OpenAI presentation failed: {openai_err} → Trying Ollama...")

        # 2️⃣  Ollama (fallback local)
        if self.fallback and self.fallback.available:
            log.info("🖥️ Presentation: Ollama fallback (sans timeout)...")
            lang_instruction = {
                "en": "\n\n*** IMPORTANT: You MUST respond ONLY in English. Do NOT respond in French. ***",
                "fr": "\n\n*** IMPORTANT: Tu DOIS répondre UNIQUEMENT en français. Ne réponds pas en anglais. ***",
            }
            fallback_prompt = f"{system_content}{lang_instruction.get(lang, lang_instruction['en'])}\n\nContent to present:\n\n{section_content}"
            answer = self._call_ollama_sync(
                prompt=fallback_prompt,
                temperature=0.7,
                max_tokens=500,  # bumped from 300 — see OpenAI branch above
            )
            if answer:
                answer = self._clean_for_speech(answer)
                answer = self._dedupe_answer_text(answer)
                duration = time.time() - start
                _narration_cache_write(cache_key, answer)
                log.info(f"✅ Ollama present OK | {duration:.2f}s | {len(answer)} chars")
                return answer, duration

        log.warning("⚠️  All LLMs failed → returning raw content")
        return self._clean_for_speech(section_content), time.time() - start

    def chat(self, content: str, language: str = "en") -> str:
        """Alias rétrocompatible."""
        text, _ = self.present(section_content=content, language=language)
        return text

    def _dedupe_answer_text(self, text: str) -> str:
        clean_text = text.strip()
        if not clean_text:
            return clean_text

        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        kept_sentences: list[str] = []
        seen_signatures: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            signature = re.sub(r"[^\w\sÀ-ÿ]+", " ", sentence.lower())
            signature = re.sub(r"\s+", " ", signature).strip()
            if not signature:
                continue

            if any(SequenceMatcher(None, signature, seen).ratio() >= 0.9 for seen in seen_signatures[-4:]):
                continue

            kept_sentences.append(sentence)
            seen_signatures.append(signature)

        deduped = " ".join(kept_sentences).strip()
        return deduped or clean_text

    def _clean_for_speech(self, text: str, language: str = "fr") -> str:
        """Render the LLM output as a clean spoken-text string for TTS.

        Pipeline :
          1. Verbalize math notation via ``audio.math_speech.to_speech``
             — LaTeX blocks become spoken phrases (``$x^2$`` → "x squared"
             / "x au carré"), unicode symbols become words. The previous
             version DELETED LaTeX blocks (``re.sub(r'\\$..\\$', '', ...)``)
             which dropped the math content silently. Verbalizing keeps
             the meaning.
          2. Strip markdown formatting (headings, bullets, emphasis, code
             fences) since TTS doesn't render structure.

        # Note on the removed ``dm_terms`` allowlist

        A previous version protected a hardcoded list of nine data-mining
        compound terms (``k-NN``, ``t-SNE``, ``XGBoost``, …) behind
        placeholders so the markdown-strip regex couldn't damage them.
        The list was rule-based : any new technical term — and the
        course is full of them — required editing this function. The
        markdown regex only damages tokens that contain markdown
        delimiters (``*``, ``_``, ``\\command``), which standard
        technical names don't, so the protection was paranoid rather
        than load-bearing. Removing it deletes the maintenance burden
        without measurable behavioural impact.
        """
        # Verbalize math notation BEFORE stripping markdown — the math
        # speech module needs the LaTeX delimiters (``$``, ``\(``…) intact
        # to detect blocks. After this call, residual backslash commands
        # are gone, so the markdown-strip pass below is purely cosmetic.
        try:
            from audio.math_speech import to_speech as _math_to_speech
            text = _math_to_speech(text, lang=(language or "fr")[:2])
        except Exception as exc:                                          # noqa: BLE001
            log.debug("math_speech.to_speech failed: %s — falling back to strip", exc)
            # Failsafe: drop LaTeX blocks rather than echoing raw symbols
            text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
            text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
            text = re.sub(r'\\\(.*?\\\)', '', text, flags=re.DOTALL)
            text = re.sub(r'\$[^$\n]+\$', '', text)
            text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
            text = re.sub(r'\\[a-zA-Z]+', '', text)

        # Markdown / structural cleanup (no longer touches math)
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
        text = re.sub(r'_{1,3}([^_\n]+)_{1,3}', r'\1', text)
        text = re.sub(r'^\s*[-•–—]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+[.)]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\\n|\\t|\\r', ' ', text)
        text = text.replace('\\', '')
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)

        return text.strip()
