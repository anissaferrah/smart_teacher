"""Smart Teacher — Text-to-Speech (Edge-TTS + ElevenLabs)"""

import logging
import re
import time
from typing import Optional, Tuple

import edge_tts

from core.config import Config

log = logging.getLogger("SmartTeacher.TTS")

EDGE_VOICES: dict = {
    "fr": {
        "female": "fr-FR-DeniseNeural",
        "male": "fr-FR-HenriNeural",
    },
    "en": {
        "female": "en-US-JennyNeural",
        "male": "en-US-GuyNeural",
    },
}

DEFAULT_VOICE: str = "en-US-JennyNeural"
ELEVENLABS_VOICES: dict = {
    "fr": "pNInz6obpgDQGcFmaJgB",  # Adam (French)
    "en": "21m00Tcm4TlvDq8ikWAM",  # Rachel (English)
}


# ════════════════════════════════════════════════════════════════════════
# VOICE ENGINE
# ════════════════════════════════════════════════════════════════════════


class VoiceEngine:
    """
    Unified Text-to-Speech engine supporting Edge-TTS and ElevenLabs.
    
    Automatically selects best available provider and includes fallback
    logic to ensure audio generation always succeeds.
    
    Attributes
    ----------
    provider : str
        Active TTS provider ("edge" or "elevenlabs")
    gender : str
        Preferred voice gender ("female" or "male")
    voice_name : str
        Name/ID of currently selected voice
    voice_id : str
        Alternative alias for voice_name (backward compatibility)
    """

    def __init__(self) -> None:
        """Initialize TTS engine with configured provider and fallback."""
        self.provider: str = Config.TTS_PROVIDER  # "edge" | "elevenlabs"
        self.gender: str = "female"
        self.voice_name: str = DEFAULT_VOICE
        self.voice_id: str = DEFAULT_VOICE  # Backward compatibility
        
        self._el_client: Optional[object] = None
        
        if self.provider == "elevenlabs":
            self._init_elevenlabs()
        
        log.info(f"✅ TTS provider initialized: {self.provider}")

    def _init_elevenlabs(self) -> None:
        """Initialize ElevenLabs client if API key is available."""
        if not Config.ELEVENLABS_API_KEY:
            log.warning("⚠️ ELEVENLABS_API_KEY not set — falling back to Edge-TTS")
            self.provider = "edge"
            return
        
        try:
            from elevenlabs.client import ElevenLabs
            self._el_client = ElevenLabs(api_key=Config.ELEVENLABS_API_KEY)
            log.info("✅ ElevenLabs client initialized")
        except Exception as exc:
            log.warning(f"⚠️ ElevenLabs initialization failed: {exc} — falling back to Edge-TTS")
            self.provider = "edge"

    def select_voice(self, index: int = 0, gender: str = "female") -> None:
        """
        Select voice by gender.
        
        Parameters
        ----------
        index : int, optional
            Voice index (currently unused, for future expansion)
        gender : str, optional
            Voice gender: "female" (default) or "male"
        """
        self.gender = gender
        log.info(f"Voice gender selected: {gender}")

    def set_voice(self, voice_id: str, voice_name: Optional[str] = None) -> None:
        """
        Explicitly set voice ID and name.
        
        Parameters
        ----------
        voice_id : str
            Voice identifier (Edge-TTS neural name or ElevenLabs ID)
        voice_name : str, optional
            Display name for the voice
        """
        self.voice_id = voice_id
        self.voice_name = voice_name or voice_id
        log.info(f"Voice set: {self.voice_name}")

    def get_cache_signatures(self, language_code: Optional[str] = None) -> list[tuple[str, str]]:
        """Return cache signatures for the active provider and its fallback voice."""
        if self.provider == "elevenlabs" and self._el_client:
            lang = (language_code or "en")[:2].lower()
            elevenlabs_voice = ELEVENLABS_VOICES.get(lang, ELEVENLABS_VOICES["en"])
            return [
                ("elevenlabs", elevenlabs_voice),
                ("edge", self._pick_edge_voice(language_code)),
            ]

        return [("edge", self._pick_edge_voice(language_code))]

    def get_available_voices(self) -> list:
        """
        List all available voices across all providers.
        
        Returns
        -------
        list[dict]
            List of voice dicts with keys: id, name, lang, gender
        """
        voices = []
        for lang, genders in EDGE_VOICES.items():
            for gender, voice_id in genders.items():
                voices.append({
                    "id": voice_id,
                    "name": voice_id,
                    "lang": lang,
                    "gender": gender
                })
        return voices

    async def generate_audio_async(
        self,
        text: str,
        language_code: Optional[str] = None,
        rate: str = "+0%"
    ) -> Tuple[Optional[bytes], float, str, str, Optional[str]]:
        """
        Generate speech audio asynchronously.
        
        Attempts primary provider first, then falls back to Edge-TTS
        if configured provider fails or returns None.
        
        Parameters
        ----------
        text : str
            Text to synthesize
        language_code : str, optional
            ISO 639-1 language code ("fr", "en").
            Defaults to English if not specified.
        rate : str, optional
            Speech rate in Edge-TTS format (e.g., "-20%", "+15%").
            Default: "+0%" (normal speed)
        
        Returns
        -------
        tuple[bytes | None, float, str, str, str | None]
            - audio_bytes: MP3 audio (None if synthesis failed)
            - duration_s: Generation time in seconds
            - engine: Provider name ("edge", "elevenlabs", or "none")
            - voice_name: Actual voice used
            - mime_type: Audio MIME type ("audio/mpeg" or None)
        
        Examples
        --------
        >>> audio, duration, engine, voice, mime = \\
        ...     await voice.generate_audio_async(
        ...         "Bonjour!",
        ...         language_code="fr"
        ...     )
        >>> print(f"Generated {len(audio)} bytes in {duration:.2f}s using {engine}")
        """
        if not text or len(text.strip()) < 2:
            return None, 0.0, "none", "none", None
        
        safe_rate = self._sanitize_rate(rate)
        
        # Try configured provider first
        if self.provider == "elevenlabs" and self._el_client:
            result = await self._synthesize_elevenlabs(text, language_code, safe_rate)
            if result[0] is not None:
                return result
            log.warning("ElevenLabs synthesis failed — falling back to Edge-TTS")
        
        # Fallback to Edge-TTS
        return await self._synthesize_edge(text, language_code, safe_rate)

    async def _synthesize_edge(
        self,
        text: str,
        language_code: Optional[str],
        rate: str
    ) -> Tuple[Optional[bytes], float, str, str, str]:
        """
        Synthesize audio using Edge-TTS (Microsoft Neural).
        
        Parameters
        ----------
        text : str
            Text to synthesize
        language_code : str, optional
            Language code for voice selection
        rate : str
            Speech rate (e.g., "+0%", "-15%")
        
        Returns
        -------
        tuple
            (audio_bytes, duration_s, engine_name, voice_name, mime_type)
        """
        voice = self._pick_edge_voice(language_code)
        start_time = time.time()
        
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
            audio_bytes = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            
            duration = time.time() - start_time
            log.info(
                f"✅ Edge-TTS: {duration:.2f}s | voice={voice} | "
                f"rate={rate} | {len(audio_bytes)} bytes"
            )
            # Return "edge" (not "edge_tts") so the engine name matches
            # get_cache_signatures() — keeps the TTS phrase cache from
            # writing twice under both names (~2× Redis usage per phrase).
            return audio_bytes, duration, "edge", voice, "audio/mpeg"
        
        except Exception as exc:
            log.error(f"❌ Edge-TTS synthesis failed: {exc}")
            return None, 0.0, "none", "none", None

    async def _synthesize_elevenlabs(
        self,
        text: str,
        language_code: Optional[str],
        rate: str
    ) -> Tuple[Optional[bytes], float, str, str, Optional[str]]:
        """
        Synthesize audio using ElevenLabs API.
        
        Parameters
        ----------
        text : str
            Text to synthesize
        language_code : str, optional
            Language code for voice selection
        rate : str
            Speech rate (Edge format, not used by ElevenLabs)
        
        Returns
        -------
        tuple
            (audio_bytes, duration_s, engine_name, voice_id, mime_type)
        """
        lang = (language_code or "en")[:2].lower()
        voice_id = ELEVENLABS_VOICES.get(lang, ELEVENLABS_VOICES["en"])
        start_time = time.time()
        
        try:
            import asyncio
            audio_bytes = await asyncio.to_thread(
                self._synthesize_elevenlabs_sync,
                text,
                voice_id
            )
            duration = time.time() - start_time
            log.info(
                f"✅ ElevenLabs: {duration:.2f}s | voice_id={voice_id} | {len(audio_bytes)} bytes"
            )
            return audio_bytes, duration, "elevenlabs", voice_id, "audio/mpeg"
        
        except Exception as exc:
            log.error(f"❌ ElevenLabs synthesis failed: {exc}")
            return None, 0.0, "none", "none", None

    def _synthesize_elevenlabs_sync(self, text: str, voice_id: str) -> bytes:
        """
        Synchronous wrapper for ElevenLabs text-to-speech.
        
        Intended to be called from async context via asyncio.to_thread().
        """
        audio_iter = self._el_client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=Config.TTS_MODEL,
            output_format=Config.TTS_OUTPUT_FORMAT,
        )
        return b"".join(audio_iter)

    def _pick_edge_voice(self, language_code: Optional[str]) -> str:
        """
        Select appropriate Edge-TTS voice based on language and gender preference.
        
        Parameters
        ----------
        language_code : str, optional
            ISO 639-1 language code
        
        Returns
        -------
        str
            Edge-TTS neural voice name (e.g., "fr-FR-DeniseNeural")
        """
        lang = (language_code or "en")[:2].lower()
        lang_voices = EDGE_VOICES.get(lang, EDGE_VOICES["en"])
        return lang_voices.get(self.gender, lang_voices["female"])

    def _sanitize_rate(self, rate: Optional[str]) -> str:
        """
        Validate and sanitize speech rate string.
        
        Edge-TTS expects format: "-20%", "+15%", "+0%", etc.
        
        Parameters
        ----------
        rate : str, optional
            Rate string to validate
        
        Returns
        -------
        str
            Validated rate string or "+0%" if invalid
        """
        if not rate:
            return "+0%"
        value = str(rate).strip()
        if re.fullmatch(r"[+-]\d{1,3}%", value):
            return value
        return "+0%"

    def speak_local(
        self,
        text: str,
        language_code: Optional[str] = None
    ) -> float:
        """
        Synchronous wrapper for testing or blocking contexts.
        
        Internally runs async speech generation and returns duration.
        
        Parameters
        ----------
        text : str
            Text to synthesize
        language_code : str, optional
            ISO 639-1 language code
        
        Returns
        -------
        float
            Synthesis duration in seconds
        """
        import asyncio
        _, duration_s, *_ = asyncio.run(
            self.generate_audio_async(text, language_code)
        )
        return duration_s
