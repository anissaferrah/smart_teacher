"""LLM Router — appel unifie OpenAI + Ollama avec preference par-call.

Pourquoi : avant, `multimodal_rag._invoke_llm_text` (OpenAI-first) et
`concept_extractor._call_llm` (Ollama-first) etaient deux helpers divergents
qui faisaient la meme chose avec des priorites inverses, sans documentation
sur le pourquoi du choix.

Design : 1 classe, parametre `prefer` par-call. Le caller choisit explicitement
sa priorite (qualite vs cout) et c'est documente dans le code.

Threadsafe : utilise depuis un ThreadPoolExecutor (Fix #3 — parallelisation
idea-chunking). Les compteurs `stats` sont proteges par un lock leger.

Metrics : compte les calls / erreurs / fallbacks par backend pour qu'un
operateur voie en logs si Ollama porte 80% du trafic parce qu'OpenAI est
down depuis 3 jours, au lieu de l'apprendre via une facture etrangement basse.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("SmartTeacher.LLMRouter")


# Tokens d'erreur OpenAI qui justifient un disable persistent (vs. transient)
_OPENAI_PERMANENT_ERROR_TOKENS = (
    "quota",
    "rate limit",
    "ratelimit",
    "429",
    "insufficient_quota",
    "authentication",
    "invalid_api_key",
)


@dataclass
class LLMStats:
    """Compteurs threadsafe pour observabilite des fallbacks."""

    openai_calls: int = 0
    openai_errors: int = 0
    groq_calls: int = 0
    groq_errors: int = 0
    ollama_calls: int = 0
    ollama_errors: int = 0
    fallback_events: int = 0           # calls ou la preference a echoue → bascule
    both_failed: int = 0               # calls ou tous les backends ont echoue
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "openai_calls":     self.openai_calls,
                "openai_errors":    self.openai_errors,
                "groq_calls":       self.groq_calls,
                "groq_errors":      self.groq_errors,
                "ollama_calls":     self.ollama_calls,
                "ollama_errors":    self.ollama_errors,
                "fallback_events":  self.fallback_events,
                "both_failed":      self.both_failed,
            }


class LLMRouter:
    """Routeur LLM avec preference par-call et fallback symetrique.

    Usage :
        router = LLMRouter()
        # Pour idea-chunking / summaries : qualite > cout
        text = router.invoke(prompt, prefer="openai", max_tokens=1200)
        # Pour validation / enrichment de masse : cout > qualite (anti-quota)
        text = router.invoke(prompt, prefer="ollama", max_tokens=600)

    Si `prefer` echoue, l'autre backend est essaye. Retourne None si les
    deux echouent (le caller decide du fallback metier).
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        ollama_model: str = "mistral",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.openai_model = openai_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip("/")
        # Disable persistent : on cesse d'essayer OpenAI pour le reste de la session
        # apres une erreur "permanente" (quota, auth). Reset = restart process.
        self._openai_disabled_reason: Optional[str] = None
        self._disable_lock = threading.Lock()
        self.stats = LLMStats()
        # Groq config — read at init time so we know whether to include it
        # in the fallback chain. Without an API key, Groq is silently
        # skipped (same pattern as OpenAI when its key is missing).
        self._groq_enabled: bool = False
        self._groq_model: str = "llama-3.3-70b-versatile"
        self._groq_base_url: str = "https://api.groq.com/openai/v1"
        self._groq_api_key: Optional[str] = None
        try:
            from core.config import Config
            if getattr(Config, "DISABLE_OPENAI", False):
                self._openai_disabled_reason = "DISABLE_OPENAI=true (config)"
                log.info("LLMRouter : OpenAI désactivé via Config.DISABLE_OPENAI")
            self._groq_api_key = getattr(Config, "GROQ_API_KEY", None) or None
            self._groq_model = getattr(Config, "GROQ_MODEL", self._groq_model)
            self._groq_base_url = getattr(Config, "GROQ_BASE_URL", self._groq_base_url)
            self._groq_enabled = bool(self._groq_api_key)
            if self._groq_enabled:
                log.info(
                    "LLMRouter : 🟢 Groq activé | model=%s base=%s "
                    "(rapide + fiable, fallback principal avant Ollama)",
                    self._groq_model, self._groq_base_url,
                )
            else:
                log.info(
                    "LLMRouter : Groq désactivé (GROQ_API_KEY manquant) — "
                    "fallback ira directement vers Ollama"
                )
        except Exception:
            # Config import failure shouldn't take down the router; if
            # the flag can't be read, default to "OpenAI enabled" which
            # is the historical behaviour.
            pass

    # ── Public ──────────────────────────────────────────────────────────

    def invoke(
        self,
        prompt: str,
        prefer: str = "openai",
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> Optional[str]:
        """Essaye `prefer` puis les autres backends dans l'ordre de fallback.

        Fallback chain (skips disabled / unconfigured backends automatically):
          - prefer="openai" → openai → groq → ollama
          - prefer="groq"   → groq   → openai → ollama
          - prefer="ollama" → ollama → groq → openai
        Returns texte non-vide ou None si tout a échoué.
        """
        prefer = prefer.lower().strip()
        if prefer not in {"openai", "groq", "ollama"}:
            prefer = "openai"

        # Build the full chain : `prefer` first, then the others in a
        # quality-preserving order. We do NOT pre-filter disabled backends
        # here — each ``_call_*`` short-circuits when its backend is
        # unavailable (returns None → fallback). Pre-filtering would
        # break tests that mock ``_call_openai`` to inject responses.
        if prefer == "openai":
            chain = ["openai", "groq", "ollama"]
        elif prefer == "groq":
            chain = ["groq", "openai", "ollama"]
        else:                                # ollama
            chain = ["ollama", "groq", "openai"]

        first_try = chain[0]
        first_failed = False
        for idx, backend in enumerate(chain):
            text = self._call(backend, prompt, temperature, max_tokens)
            if text:
                if first_failed:
                    log.info(f"LLM fallback {first_try} → {backend} succeeded")
                return text
            # Mark the first-backend failure exactly once, regardless of
            # whether any later backend ends up rescuing the call. This
            # matches the historical metric semantics where a single
            # ``fallback_events`` count means "the preferred backend
            # didn't answer this turn" (independent of recovery).
            if idx == 0:
                first_failed = True
                with self.stats._lock:
                    self.stats.fallback_events += 1

        with self.stats._lock:
            self.stats.both_failed += 1
        log.warning(f"LLM router : all backends failed (prefer={prefer}, tried={chain})")
        return None

    @property
    def openai_disabled_reason(self) -> Optional[str]:
        return self._openai_disabled_reason

    def disable_openai(self, reason: str) -> None:
        """Force-disable OpenAI pour le reste de la session (test / config externe)."""
        with self._disable_lock:
            if not self._openai_disabled_reason:
                self._openai_disabled_reason = reason
                log.warning(f"OpenAI disabled : {reason}")

    # ── Internals ───────────────────────────────────────────────────────

    def _call(
        self,
        backend: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[str]:
        if backend == "openai":
            return self._call_openai(prompt, temperature, max_tokens)
        if backend == "groq":
            return self._call_groq(prompt, temperature, max_tokens)
        return self._call_ollama(prompt, temperature, max_tokens)

    def _call_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[str]:
        if self._openai_disabled_reason:
            return None
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            with self.stats._lock:
                self.stats.openai_calls += 1
            llm = ChatOpenAI(
                model=self.openai_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=0,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            text = (response.content or "").strip()
            return text or None
        except Exception as exc:  # noqa: BLE001
            with self.stats._lock:
                self.stats.openai_errors += 1
            if self._is_permanent_openai_error(exc):
                self.disable_openai(str(exc))
            log.debug(f"OpenAI invoke failed: {exc}")
            return None

    def _call_groq(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[str]:
        """Call Groq via the OpenAI-compatible endpoint.

        Uses `langchain_openai.ChatOpenAI` with a custom `base_url` and the
        Groq API key — the same path as OpenAI but pointed at Groq's
        OpenAI-compatible inference server. Logs prompt/output stats with
        the 🟢 emoji so operators can spot Groq calls in the terminal.
        """
        if not self._groq_enabled:
            return None
        import time as _t
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            with self.stats._lock:
                self.stats.groq_calls += 1
            t0 = _t.time()
            log.info(
                "🟢 GROQ generate START | model=%s | prompt_chars=%d (~%d tokens) | "
                "temp=%.2f max_tokens=%d",
                self._groq_model, len(prompt), len(prompt) // 4,
                temperature, max_tokens,
            )
            llm = ChatOpenAI(
                model=self._groq_model,
                api_key=self._groq_api_key,
                base_url=self._groq_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=0,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            elapsed = _t.time() - t0
            text = (response.content or "").strip()
            log.info(
                "🟢 GROQ generate DONE | model=%s | took=%.2fs | out_chars=%d",
                self._groq_model, elapsed, len(text),
            )
            return text or None
        except Exception as exc:    # noqa: BLE001
            with self.stats._lock:
                self.stats.groq_errors += 1
            log.warning(f"🟢 GROQ invoke failed: {exc}")
            return None

    def _call_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[str]:
        try:
            import requests

            with self.stats._lock:
                self.stats.ollama_calls += 1
            # Ollama's /api/generate expects model parameters inside the
            # ``options`` dict, not at the top level. The previous version
            # passed temperature/num_predict at the top level, which Ollama
            # silently ignored — sampling stayed at the model defaults.
            options: dict = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            # Honour OLLAMA_NUM_THREADS when set (>0) so a CPU-only host
            # can dedicate more cores to inference. 0 = "auto" = let
            # Ollama pick (default = half the cores).
            try:
                from core.config import Config as _Cfg
                n_threads = int(getattr(_Cfg, "OLLAMA_NUM_THREADS", 0) or 0)
                if n_threads > 0:
                    options["num_thread"] = n_threads
            except Exception:
                pass
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }
            # Timeout : Ollama CPU peut etre tres lent. None = laisse mouliner
            # (les callers parallelises bornent leur propre patience).
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=None,
            )
            if response.status_code != 200:
                with self.stats._lock:
                    self.stats.ollama_errors += 1
                log.debug(f"Ollama HTTP {response.status_code}")
                return None
            text = (response.json().get("response", "") or "").strip()
            return text or None
        except Exception as exc:  # noqa: BLE001
            with self.stats._lock:
                self.stats.ollama_errors += 1
            log.debug(f"Ollama invoke failed: {exc}")
            return None

    @staticmethod
    def _is_permanent_openai_error(exc: Exception) -> bool:
        message = f"{exc.__class__.__module__}:{exc.__class__.__name__}:{exc}".lower()
        return any(token in message for token in _OPENAI_PERMANENT_ERROR_TOKENS)


# ── Singleton partage (lazy) ────────────────────────────────────────────

_default_router: Optional[LLMRouter] = None
_default_lock = threading.Lock()


def get_default_router() -> LLMRouter:
    """Retourne le router par defaut (singleton lazy, threadsafe)."""
    global _default_router
    if _default_router is None:
        with _default_lock:
            if _default_router is None:
                _default_router = LLMRouter()
    return _default_router
