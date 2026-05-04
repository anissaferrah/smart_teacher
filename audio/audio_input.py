"""
╔══════════════════════════════════════════════════════════════════════╗
║           SMART TEACHER — Module Audio Input (VAD + Micro)         ║
║                                                                      ║
║  Capture le son du micro et détecte automatiquement la parole       ║
║  via Silero VAD (modèle IA léger, 1 MB).                            ║
║                                                                      ║
║  AMÉLIORATIONS vs version initiale :                                 ║
║    ✅ Méthode record_until_silence() — boucle complète intégrée     ║
║    ✅ Callback d'interruption : appelé dès que la parole commence   ║
║    ✅ Protection contre les buffers infinis (MAX_AUDIO_DURATION)     ║
║    ✅ Logs structurés                                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import logging
import time
from typing import Callable

import numpy as np
import torch

from core.config import Config

log = logging.getLogger("SmartTeacher.AudioInput")

try:
    import sounddevice as sd
except OSError:
    sd = None
    log.warning("sounddevice unavailable (no PortAudio) — server-side mic capture disabled (browser handles capture)")


class AudioInput:
    """
    Capture audio microphone + Voice Activity Detection (VAD Silero).

    Utilisation simple :
        audio_input = AudioInput()
        audio_array = audio_input.record_until_silence()
        # audio_array est prêt pour Transcriber.transcribe()

    Utilisation avec callback d'interruption :
        def on_speech_start():
            print("Parole détectée — IA muette")

        audio_array = audio_input.record_until_silence(
            on_speech_start=on_speech_start
        )
    """

    def __init__(self):
        log.info("Chargement du modèle VAD Silero…")
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        log.info("✅ VAD Silero chargé")

    # ──────────────────────────────────────────────────────────────────
    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Retourne la probabilité (0.0–1.0) que le chunk contienne de la parole.

        Args:
            audio_chunk: Tableau numpy float32 de Config.CHUNK_SIZE samples

        Returns:
            Probabilité de voix (0 = silence, 1 = parole certaine)
        """
        tensor = torch.from_numpy(audio_chunk)
        return float(self.model(tensor, Config.SAMPLE_RATE).item())

    # ──────────────────────────────────────────────────────────────────
    def create_stream(self) -> "sd.InputStream":
        """Crée un flux d'entrée microphone (non démarré)."""
        return sd.InputStream(
            samplerate=Config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=Config.CHUNK_SIZE,
            latency="low",
        )

    # ──────────────────────────────────────────────────────────────────
    def record_until_silence(
        self,
        on_speech_start: Callable | None = None,
        min_speech_duration: float = 0.3,
    ) -> np.ndarray | None:
        """
        Enregistre depuis le micro jusqu'à ce qu'un silence soit détecté.

        Logique :
          1. Attend que la parole commence (VAD > SPEECH_THRESHOLD)
          2. Appelle on_speech_start() si fourni (pour couper le TTS)
          3. Accumule l'audio tant que la parole continue
          4. Retourne le buffer quand SILENCE_DURATION secondes de silence

        Args:
            on_speech_start:      Callback appelé dès que la parole est détectée
            min_speech_duration:  Durée minimale de parole pour déclencher (s)

        Returns:
            np.ndarray float32 à 16 kHz, ou None si rien capturé.
        """
        speech_started    = False
        silence_start     = None
        audio_buffer      = []
        recording_start   = None

        log.info("🎙️  En écoute… (parle pour commencer)")

        with self.create_stream() as stream:
            while True:
                chunk, _ = stream.read(Config.CHUNK_SIZE)
                chunk = chunk.flatten()

                prob = self.get_speech_probability(chunk)
                is_speech = prob >= Config.SPEECH_THRESHOLD

                # ── Parole détectée ───────────────────────────────────
                if is_speech:
                    if not speech_started:
                        speech_started  = True
                        recording_start = time.time()
                        silence_start   = None
                        log.info("🗣️  Parole détectée — enregistrement démarré")

                        # Callback interruption (coupe le TTS en cours)
                        if on_speech_start:
                            try:
                                on_speech_start()
                            except Exception as exc:
                                log.warning(f"on_speech_start callback error: {exc}")

                    audio_buffer.append(chunk.copy())
                    silence_start = None  # réinitialise le compteur de silence

                # ── Silence ───────────────────────────────────────────
                elif speech_started:
                    audio_buffer.append(chunk.copy())

                    if silence_start is None:
                        silence_start = time.time()

                    # Silence suffisamment long → fin de l'énoncé
                    elif time.time() - silence_start >= Config.SILENCE_DURATION:
                        log.info("🔇  Silence détecté — fin de l'enregistrement")
                        break

                # ── Sécurité : durée max ──────────────────────────────
                if speech_started and recording_start:
                    if time.time() - recording_start >= Config.MAX_AUDIO_DURATION:
                        log.warning("⚠️  Durée max atteinte — envoi forcé")
                        break

        if not audio_buffer:
            return None

        audio = np.concatenate(audio_buffer, axis=0)

        # Vérifie qu'il y a assez de parole réelle
        duration = len(audio) / Config.SAMPLE_RATE
        if duration < min_speech_duration:
            log.debug(f"Audio trop court ({duration:.2f}s) — ignoré")
            return None

        log.info(f"✅ Audio capturé : {duration:.2f}s")
        return audio