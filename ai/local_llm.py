"""
Local LLM Fallback Module
Si OpenAI plante, utilise Ollama (Mistral) en local
Installation: ollama run mistral (ou otra modelo)
URL: http://localhost:11434
"""

import logging
import requests
from typing import Optional

log = logging.getLogger("SmartTeacher.LocalLLM")


class LocalLLMFallback:
    """Fallback vers Ollama local si OpenAI indisponible"""
    
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialise le fallback LLM local
        
        Args:
            model: Modèle Ollama ("mistral", "neural-chat", "orca-mini", etc.)
            base_url: URL du serveur Ollama
        """
        self.model = model
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/generate"
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Vérifier si Ollama est accessible"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=None  # Pas de timeout pour laisser Ollama démarrer à son rythme
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                is_available = any(self.model in name for name in model_names)
                if is_available:
                    log.info(f"✅ Ollama actif - modèle '{self.model}' chargé")
                else:
                    log.warning(f"⚠️ Service Ollama actif, modèle '{self.model}' non présent")
                    log.info(f"   Modèles disponibles: {model_names}")
                return is_available
        except requests.exceptions.ConnectionError:
            log.warning(f"⚠️ Ollama non connecté sur {self.base_url}")
            log.warning("   👇 Lancez: docker-compose up -d ollama")
        except Exception as e:
            log.warning(f"⚠️ Ollama check (échec): {e}")
        
        return False
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[str]:
        """
        Générer réponse avec Ollama en local
        
        Args:
            prompt: Question/instruction
            temperature: Créativité (0.0-1.0)
            max_tokens: Longueur max réponse
            
        Returns:
            Texte généré ou None si erreur
        """
        
        if not self.available:
            return None

        import time as _ot
        try:
            _opts = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            try:
                from core.config import Config as _Cfg
                _n_threads = int(getattr(_Cfg, "OLLAMA_NUM_THREADS", 0) or 0)
                if _n_threads > 0:
                    _opts["num_thread"] = _n_threads
            except Exception:
                pass
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": _opts,
            }

            log.info(
                "🦙 OLLAMA generate START | model=%s | prompt_chars=%d (~%d tokens) | "
                "temp=%.2f max_tokens=%d threads=%s | endpoint=%s",
                self.model, len(prompt), len(prompt) // 4,
                temperature, max_tokens, _opts.get("num_thread", "default"),
                self.endpoint,
            )
            _t0 = _ot.time()

            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=None
            )

            elapsed = _ot.time() - _t0

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                # Ollama returns these stats when stream=false
                eval_count = int(result.get("eval_count") or 0)
                eval_dur_ns = int(result.get("eval_duration") or 0)
                prompt_eval_count = int(result.get("prompt_eval_count") or 0)
                prompt_eval_dur_ns = int(result.get("prompt_eval_duration") or 0)
                load_dur_ns = int(result.get("load_duration") or 0)
                tokens_per_s = (eval_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0.0

                if generated_text:
                    log.info(
                        "🦙 OLLAMA generate DONE | model=%s | "
                        "prompt_tokens=%d eval_tokens=%d | %.0f tok/s | "
                        "load=%.0fms prompt_eval=%.0fms eval=%.0fms total=%.2fs | "
                        "out_chars=%d",
                        self.model, prompt_eval_count, eval_count, tokens_per_s,
                        load_dur_ns / 1e6, prompt_eval_dur_ns / 1e6, eval_dur_ns / 1e6,
                        elapsed, len(generated_text),
                    )
                    return generated_text
                else:
                    log.warning(
                        "🦙 OLLAMA generate EMPTY | model=%s | took=%.2fs | "
                        "prompt_tokens=%d eval_tokens=%d",
                        self.model, elapsed, prompt_eval_count, eval_count,
                    )
                    return None
            else:
                log.error(
                    "🦙 OLLAMA generate HTTP_FAIL | model=%s | status=%d | took=%.2fs | body=%s",
                    self.model, response.status_code, elapsed, response.text[:200],
                )
                return None

        except requests.exceptions.Timeout:
            log.error(f"🦙 OLLAMA generate TIMEOUT | model={self.model}")
            return None
        except requests.exceptions.ConnectionError:
            log.error(f"🦙 OLLAMA generate CONN_ERR | endpoint={self.endpoint}")
            self.available = False
            return None
        except Exception as e:
            log.error(f"🦙 OLLAMA generate ERR | {e}")
            return None
    
    async def generate_educational(
        self,
        question: str,
        subject: str = "general",
        level: str = "intermediate"
    ) -> Optional[str]:
        """
        Générer réponse pédagogique avec Ollama
        
        Args:
            question: Question de l'étudiant
            subject: Sujet (math, sciences, histoire, etc.)
            level: Niveau (beginner, intermediate, advanced)
            
        Returns:
            Explication pédagogique
        """
        
        # Prompt pédagogique spécialisé
        system_prompt = f"""Tu es un professeur pédagogique en {subject}.
Réponds clairement et pédagogiquement au niveau {level}.
Utilise des exemples concrets.
Reste concis (max 300 mots)."""
        
        full_prompt = f"{system_prompt}\n\nÉtudiant: {question}\n\nProfesseur:"
        
        return await self.generate(
            prompt=full_prompt,
            temperature=0.6,  # Plus déterministe pour éducation
            max_tokens=300
        )
    
    async def health_check(self) -> dict:
        """Vérifier santé du serveur Ollama"""
        return {
            "available": self.available,
            "model": self.model,
            "endpoint": self.endpoint,
            "status": "✅ Ready" if self.available else "❌ Unavailable"
        }
