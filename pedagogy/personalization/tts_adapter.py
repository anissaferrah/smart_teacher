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


# Default speech rate inferred from the student's academic level when no
# explicit preference exists. Pedagogical reasoning :
#  - "collège" (middle school) — younger learners benefit from slower
#    pacing so they can process new vocabulary without being overwhelmed.
#  - "lycée" (high school) — neutral baseline, similar to native everyday
#    speech (~150 wpm in French → multiplier 1.0 = "+0%" in Edge-TTS).
#  - "université" / "master" / "doctorat" — adults handle a slightly
#    faster delivery without comprehension loss, and many actually prefer
#    it to keep engagement up over long lectures.
# These are *defaults*; once the student manually sets their preferred
# rate via the UI (preferences.speech_rate + preferences.manual_rate_override
# = True), this layer is bypassed entirely. See get_edge_tts_rate_with_bandit.
_LEVEL_DEFAULT_RATE: dict[str, float] = {
    "collège":     0.92,    # ~ -8 %
    "college":     0.92,    # ASCII variant
    "lycée":       1.00,
    "lycee":       1.00,
    "université":  1.05,    # ~ +5 %
    "universite":  1.05,
    "licence":     1.05,
    "master":      1.05,
    "m1":          1.05,
    "m2":          1.05,
    "doctorat":    1.05,
    "phd":         1.05,
}


def default_rate_for_level(level: str | None) -> float:
    """Return a sensible default multiplier for a student's academic level.

    Falls back to 1.0 (neutral) on unknown / missing level.
    """
    if not level:
        return 1.0
    return _LEVEL_DEFAULT_RATE.get(level.strip().lower(), 1.0)


async def compute_tts_params(
    session_id: str,
    base_rate: float = 1.0,
    confusion_score: float = 0.0,   # accepted for back-compat, not used
) -> Dict[str, Any]:
    """Return TTS params from the user's stored preferences.

    Resolution order for ``rate`` :
      1. ``preferences.speech_rate`` (student-set explicit preference)
      2. Default derived from ``profile.level`` via _LEVEL_DEFAULT_RATE
      3. ``base_rate`` argument (legacy, usually 1.0)
    """
    profile = await get_or_create_profile(session_id)
    prefs = profile.get("preferences", {}) if profile else {}
    level = (profile.get("level") if profile else None) or "lycée"

    # Layer 1 (explicit preference) > Layer 2 (level default) > Layer 3 (caller default)
    if "speech_rate" in prefs and prefs.get("speech_rate") is not None:
        rate = float(prefs["speech_rate"])
        rate_source = "user_preference"
    else:
        rate = default_rate_for_level(level)
        rate_source = f"level_default({level})"
    pitch = float(prefs.get("pitch", 1.0))
    voice = prefs.get("voice", "default")

    params = {
        "rate":         round(rate, 2),
        "pitch":        round(pitch, 2),
        "voice":        voice,
        "rate_source":  rate_source,
        "manual_override": bool(prefs.get("manual_rate_override", False)),
    }
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


# ── Confusion-driven adaptive slowdown ───────────────────────────────
# When the fused confusion score (text + prosody) stays high across
# recent turns, the next narration slows down so the student can catch
# up. We use a STEPPED multiplier rather than a continuous one so the
# behaviour is predictable in operator logs : either we slow down by
# 10 % or we don't, no in-between.
#
# Why 0.6 threshold + 0.9 multiplier :
#   - 0.6 matches the "confidently confused" zone of the fused score
#     (both SIGHT and prosody contributing).
#   - 0.9 = -10 % is the smallest perceptually meaningful slowdown
#     (Goldman-Eisler 1968 ; rate JND ≈ 5-7 % for trained listeners,
#     12-15 % for untrained). We use 10 % so the change is felt
#     without being jarring.

CONFUSION_TTS_SLOWDOWN_THRESHOLD = 0.6
CONFUSION_TTS_SLOWDOWN_FACTOR    = 0.9


def confusion_rate_multiplier(confusion_score: float | None) -> float:
    """Stepped rate multiplier driven by recent fused confusion score.

    confusion_score >= 0.6 → 0.9 (slow down by 10 %).
    Otherwise              → 1.0 (no change).

    The stepped behaviour is intentional : a continuous map would mean
    every utterance has a slightly different rate, making the audio
    feel jittery to the student.
    """
    if confusion_score is None:
        return 1.0
    try:
        score = float(confusion_score)
    except (TypeError, ValueError):
        return 1.0
    if score >= CONFUSION_TTS_SLOWDOWN_THRESHOLD:
        return CONFUSION_TTS_SLOWDOWN_FACTOR
    return 1.0


async def get_edge_tts_rate_with_bandit(
    session_id: str,
    bandit_speech_rate: str | None = None,
    confusion_score: float | None = None,
) -> str:
    """Profile-based rate × bandit modulation × confusion-driven slowdown.

    Pipeline :
      1. Read the student's stored preference (``speech_rate`` in profile),
         falling back to a level-derived default (collège slower, université
         neutral/faster) when no explicit preference exists.
      2. If ``preferences.manual_rate_override == True`` (student has
         explicitly chosen their rate via the UI), SKIP both the bandit
         modulation AND the confusion-driven slowdown — the student knows
         what they want, the system shouldn't fight them.
      3. Otherwise multiply by ``bandit_rate_multiplier(category)`` and
         ``confusion_rate_multiplier(score)``.
      4. Convert to the Edge-TTS percentage format.

    Falls back to ``"+0%"`` on any error so TTS never breaks.
    """
    try:
        params = await compute_tts_params(session_id)
        base_rate = float(params.get("rate", 1.0))
        manual_override = bool(params.get("manual_override", False))
        rate_source = params.get("rate_source", "?")

        if manual_override:
            # Student-controlled rate : ignore bandit + confusion. The UI
            # surfaces a slider/switch that flips this flag along with the
            # ``speech_rate`` value. We respect their choice unconditionally.
            log.info(
                "🎚️ tts rate USER-OVERRIDE | rate=%.2f (source=%s) → %s "
                "(bandit=%s and confusion=%s ignored)",
                base_rate, rate_source, rate_float_to_edge_str(base_rate),
                bandit_speech_rate or "none",
                f"{float(confusion_score):.2f}" if confusion_score is not None else "none",
            )
            return rate_float_to_edge_str(base_rate)

        bandit_mult = bandit_rate_multiplier(bandit_speech_rate)
        confusion_mult = confusion_rate_multiplier(confusion_score)
        final_rate = base_rate * bandit_mult * confusion_mult
        # Detail log : if anything other than the trivial 1.0×1.0×1.0
        # was applied, the operator sees exactly which layer fired.
        if bandit_mult != 1.0 or confusion_mult != 1.0 or rate_source != "user_preference":
            log.info(
                "🎚️ tts rate composed | profile=%.2f (%s) × bandit=%.2f (%s) × "
                "confusion=%.2f (score=%s) = %.2f → %s",
                base_rate, rate_source,
                bandit_mult, bandit_speech_rate or "none",
                confusion_mult,
                f"{float(confusion_score):.2f}" if confusion_score is not None else "none",
                final_rate,
                rate_float_to_edge_str(final_rate),
            )
        return rate_float_to_edge_str(final_rate)
    except Exception as exc:                                              # noqa: BLE001
        log.debug("get_edge_tts_rate_with_bandit fallback to +0%% (%s)", exc)
        return "+0%"
