"""TTS parameter adapter — minimal pass-through.

All automatic adaptations (level-based defaults, bandit modulation,
confusion-driven slowdown) have been removed. The TTS rate is fixed at
1.0 (= "+0%" in Edge-TTS) unless the student has explicitly set a
``speech_rate`` in their profile preferences.

Public API kept for back-compat with existing call sites :
  - compute_tts_params(session_id, ...)            → dict
  - rate_float_to_edge_str(rate)                   → str (e.g. "+0%")
  - get_edge_tts_rate(session_id, ...)             → str
  - get_edge_tts_rate_with_bandit(session_id, ...) → str
  - bandit_rate_multiplier(speech_rate)            → 1.0 always
  - confusion_rate_multiplier(score)               → 1.0 always
  - default_rate_for_level(level)                  → 1.0 always
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from pedagogy.personalization.profile import get_or_create_profile

log = logging.getLogger("services.personalization.tts_adapter")


# ── No-op helpers (kept so callers don't break) ────────────────────────

def default_rate_for_level(level: str | None) -> float:    # noqa: ARG001
    """Always returns 1.0 — level-based adaptation removed."""
    return 1.0


def bandit_rate_multiplier(speech_rate: str | None) -> float:    # noqa: ARG001
    """Always returns 1.0 — bandit-driven rate adaptation removed."""
    return 1.0


def confusion_rate_multiplier(confusion_score: float | None) -> float:    # noqa: ARG001
    """Always returns 1.0 — confusion-driven slowdown removed."""
    return 1.0


# ── Core API ────────────────────────────────────────────────────────────

def rate_float_to_edge_str(rate: float) -> str:
    """Convert a multiplier (0.5..1.5) into an Edge-TTS rate string.

    Edge-TTS accepts rates as percentage strings: '+0%', '+15%', '-20%'.
    Multiplier 1.0 → '+0%' ; 0.85 → '-15%' ; 1.20 → '+20%'.
    Clamped to [-50%, +50%] (Edge limits).
    """
    pct = int(round((float(rate) - 1.0) * 100))
    pct = max(-50, min(50, pct))
    return f"{pct:+d}%"


async def compute_tts_params(
    session_id: str,
    base_rate: float = 1.0,                # noqa: ARG001 (back-compat)
    confusion_score: float = 0.0,          # noqa: ARG001 (back-compat)
) -> Dict[str, Any]:
    """Return TTS params, honouring only the student's explicit preference.

    Resolution :
      - If ``preferences.speech_rate`` is set in the profile → use it.
      - Otherwise → 1.0 (neutral, "+0%" in Edge-TTS).
    """
    rate: float = 1.0
    pitch: float = 1.0
    voice: str = "default"
    rate_source: str = "default"
    manual_override: bool = False

    try:
        profile = await get_or_create_profile(session_id)
        prefs = profile.get("preferences", {}) if profile else {}
        if "speech_rate" in prefs and prefs["speech_rate"] is not None:
            rate = float(prefs["speech_rate"])
            rate_source = "user_preference"
        pitch = float(prefs.get("pitch", 1.0))
        voice = prefs.get("voice", "default")
        manual_override = bool(prefs.get("manual_rate_override", False))
    except Exception as exc:    # noqa: BLE001
        log.debug("compute_tts_params fallback to defaults (%s)", exc)

    return {
        "rate":            round(rate, 2),
        "pitch":           round(pitch, 2),
        "voice":           voice,
        "rate_source":     rate_source,
        "manual_override": manual_override,
    }


async def get_edge_tts_rate(
    session_id: str,
    confusion_score: float = 0.0,          # noqa: ARG001 (back-compat)
) -> str:
    """Profile → Edge-TTS rate string. Falls back to '+0%' on any error."""
    try:
        params = await compute_tts_params(session_id)
        return rate_float_to_edge_str(params["rate"])
    except Exception as exc:    # noqa: BLE001
        log.debug("get_edge_tts_rate fallback to +0%% (%s)", exc)
        return "+0%"


async def get_edge_tts_rate_with_bandit(
    session_id: str,
    bandit_speech_rate: str | None = None,    # noqa: ARG001 (back-compat)
    confusion_score: float | None = None,     # noqa: ARG001 (back-compat)
) -> str:
    """Identical to :func:`get_edge_tts_rate` — bandit + confusion are no-ops.

    Kept under this name so existing call sites in handlers/ws.py keep
    importing it without churn.
    """
    return await get_edge_tts_rate(session_id)
