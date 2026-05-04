"""Echo cancellation helpers — TTS PCM decode + cross-correlation detection.

Extracted from main.py to keep the WS handler module focused on flow control.
"""

import logging
from collections import deque

import numpy as np

log = logging.getLogger("SmartTeacher.audio.echo_helpers")


def tts_bytes_to_pcm(audio_bytes: bytes, target_sr: int) -> np.ndarray | None:
    """Decode TTS audio (mp3/wav/webm) en PCM mono float32 a `target_sr` Hz.

    Retourne un numpy array normalise [-1, 1] ou None si decode echoue.
    Utilise pour le ring buffer de cross-correlation echo.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return None
    try:
        from io import BytesIO
        from pydub import AudioSegment
        seg = AudioSegment.from_file(BytesIO(audio_bytes))
        seg = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
        samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
        return samples.astype(np.float32) / 32768.0
    except Exception as exc:
        log.debug(f"TTS PCM decode failed: {exc}")
        return None


def is_echo_of_recent_tts(
    user_audio: np.ndarray,
    tts_pcm_history: deque,
    threshold: float,
) -> tuple[bool, float]:
    """Detecte si user_audio est l'echo de TTS recent via cross-correlation FFT.

    Returns (is_echo, peak_corr). peak_corr est la correlation Pearson |r|
    sur le shift d'alignement optimal.
    """
    if len(tts_pcm_history) < len(user_audio):
        return (False, 0.0)
    try:
        from scipy.signal import correlate
        tts = np.array(tts_pcm_history, dtype=np.float32)
        user = user_audio.astype(np.float32)
        xcorr = correlate(tts, user, mode="valid", method="fft")
        if len(xcorr) == 0:
            return (False, 0.0)
        best_shift = int(np.argmax(np.abs(xcorr)))
        matched = tts[best_shift : best_shift + len(user)]
        if len(matched) != len(user):
            return (False, 0.0)
        u = user - user.mean()
        m = matched - matched.mean()
        denom = float(np.sqrt(np.sum(u ** 2) * np.sum(m ** 2)))
        if denom == 0.0:
            return (False, 0.0)
        peak = float(abs(np.sum(u * m) / denom))
        return (peak > threshold, peak)
    except Exception as exc:
        log.debug(f"xcorr failed: {exc}")
        return (False, 0.0)
