"""TTS parameter adapter guided by user profile.

Returns TTS parameters (rate, pitch, voice) tuned by the user's stored
preferences. Lightweight and synchronous.

# Note on confusion-driven modulation

A previous version multiplied the rate by ``max(0.6, 1.0 - 0.35 ×
confusion_score)`` and the pitch by ``(1.0 + 0.1 × confusion_score)``
to "slow the voice when the student is confused". The coefficients
0.6 / 0.35 / 0.1 had no UX research or empirical backing, so they were
removed. ``confusion_score`` is still accepted in the signature for
back-compat but doesn't currently modify the TTS output. If this UX
behaviour is reintroduced, the magnitudes must come from a study
(perceptual rate-limit thresholds + user satisfaction A/B test), not
hand-picked.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

from pedagogy.personalization.profile import get_or_create_profile

log = logging.getLogger("services.personalization.tts_adapter")


async def compute_tts_params(
    session_id: str,
    base_rate: float = 1.0,
    confusion_score: float = 0.0,   # accepted for back-compat, not used
) -> Dict[str, Any]:
    """Return TTS params from the user's stored preferences."""
    profile = await get_or_create_profile(session_id)
    prefs = profile.get("preferences", {}) if profile else {}

    rate  = float(prefs.get("speech_rate", base_rate))
    pitch = float(prefs.get("pitch", 1.0))
    voice = prefs.get("voice", "default")

    params = {"rate": round(rate, 2), "pitch": round(pitch, 2), "voice": voice}
    log.debug("Computed tts params for %s: %s", session_id, params)
    return params


def rate_float_to_edge_str(rate: float) -> str:
    """Convert a multiplier (0.7..1.3) into the Edge-TTS rate string format.

    Edge-TTS accepts rates as percentage strings: '+0%', '+15%', '-20%', etc.
    Multiplier 1.0 → '+0%' ; 0.85 → '-15%' ; 1.20 → '+20%'.
    Clamped to [-50%, +50%] (Edge limits, imposed by the external API).
    """
    pct = int(round((float(rate) - 1.0) * 100))
    pct = max(-50, min(50, pct))
    return f"{pct:+d}%"


async def get_edge_tts_rate(session_id: str, confusion_score: float = 0.0) -> str:
    """One-call helper: Profile → Edge-TTS rate string.

    Reads the student profile and returns a rate string ready for
    ``voice.generate_audio_async(rate=...)``. Falls back to '+0%' on any
    error (no profile / Redis down / etc.).
    """
    try:
        params = await compute_tts_params(session_id, confusion_score=confusion_score)
        return rate_float_to_edge_str(params["rate"])
    except Exception as exc:
        log.debug("get_edge_tts_rate fallback to +0%% (%s)", exc)
        return "+0%"


# ── Bandit-driven speech rate ──────────────────────────────────────────
#
# When the personalization bandit selects an action, the action's
# ``speech_rate`` field is one of {slow, normal, fast}. Edge-TTS expects
# a percentage offset on top of the user's base rate, so we map the
# discrete bandit categories to numeric multipliers here.
#
# The 0.15 offset (≈ 15% slower / faster) is the per-turn editorial
# default for Phase 1: large enough to be perceptually noticeable but
# inside Edge-TTS's natural-prosody band (-50 % .. +50 %). It's expected
# to be replaced by an empirically-calibrated per-student offset once
# Phase 2 (simulator) and Phase 3 (offline RL) collect enough samples
# to fit the JND curve. Documented as a tunable Phase-1 default rather
# than a magic number — adjust here, not at call sites.

_BANDIT_RATE_OFFSET = 0.15

_BANDIT_RATE_MULTIPLIERS: dict[str, float] = {
    "slow":   1.0 - _BANDIT_RATE_OFFSET,
    "normal": 1.0,
    "fast":   1.0 + _BANDIT_RATE_OFFSET,
}


def bandit_rate_multiplier(speech_rate: str | None) -> float:
    """Map a bandit ``speech_rate`` category to a numeric multiplier.

    Returns ``1.0`` (no change) for an unknown / missing category.
    """
    if not speech_rate:
        return 1.0
    return _BANDIT_RATE_MULTIPLIERS.get(speech_rate.strip().lower(), 1.0)


async def get_edge_tts_rate_with_bandit(
    session_id: str,
    bandit_speech_rate: str | None = None,
) -> str:
    """Profile-based rate × bandit modulation → Edge-TTS rate string.

    Pipeline :
      1. Read the student's stored preference (``speech_rate`` in profile).
      2. If the bandit selected a category, multiply by
         ``bandit_rate_multiplier(category)``.
      3. Convert to the Edge-TTS percentage format.

    Falls back to ``"+0%"`` on any error so TTS never breaks.
    """
    try:
        params = await compute_tts_params(session_id)
        base_rate = float(params.get("rate", 1.0))
        final_rate = base_rate * bandit_rate_multiplier(bandit_speech_rate)
        return rate_float_to_edge_str(final_rate)
    except Exception as exc:                                              # noqa: BLE001
        log.debug("get_edge_tts_rate_with_bandit fallback to +0%% (%s)", exc)
        return "+0%"
