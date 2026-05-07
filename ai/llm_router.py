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
    ollama_calls: int = 0
    ollama_errors: int = 0
    fallback_events: int = 0           # calls ou la preference a echoue → bascule
    both_failed: int = 0               # calls ou les 2 backends ont echoue
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "openai_calls": self.openai_calls,
                "openai_errors": self.openai_errors,
                "ollama_calls": self.ollama_calls,
                "ollama_errors": self.ollama_errors,
                "fallback_events": self.fallback_events,
                "both_failed": self.both_failed,
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
        ollama_model: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        if ollama_model is None:
            from core.config import Config as _Cfg
            ollama_model = _Cfg.OLLAMA_TEXT_MODEL
        self.openai_model = openai_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip("/")
        # Disable persistent : on cesse d'essayer OpenAI pour le reste de la session
        # apres une erreur "permanente" (quota, auth). Reset = restart process.
        self._openai_disabled_reason: Optional[str] = None
        self._disable_lock = threading.Lock()
        self.stats = LLMStats()
        # Honour the global DISABLE_OPENAI kill-switch at construction
        # time. Without this, every caller that asked for prefer="openai"
        # would burn one failing OpenAI attempt before falling back to
        # Ollama — wasting time AND polluting the logs with
        # ``LLM fallback openai → ollama succeeded`` on every call.
        try:
            from core.config import Config
            if getattr(Config, "DISABLE_OPENAI", False):
                self._openai_disabled_reason = "DISABLE_OPENAI=true (config)"
                log.info("LLMRouter : OpenAI désactivé via Config.DISABLE_OPENAI → Ollama only")
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
        """Essaye `prefer` puis l'autre backend. Retourne texte non-vide ou None."""
        prefer = prefer.lower().strip()
        if prefer not in {"openai", "ollama"}:
            prefer = "openai"

        first, second = (prefer, "ollama" if prefer == "openai" else "openai")

        text = self._call(first, prompt, temperature, max_tokens)
        if text:
            return text

        # Fallback
        with self.stats._lock:
            self.stats.fallback_events += 1
        text = self._call(second, prompt, temperature, max_tokens)
        if text:
            log.info(f"LLM fallback {first} → {second} succeeded")
            return text

        with self.stats._lock:
            self.stats.both_failed += 1
        log.warning(f"LLM router : both backends failed (prefer={prefer})")
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
            from core.config import Config as _Cfg
            llm = ChatOpenAI(
                model=self.openai_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=0,
                base_url=_Cfg.OPENAI_BASE_URL,
                api_key=_Cfg.OPENAI_API_KEY,
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
