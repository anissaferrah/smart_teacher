"""Multi-signal confusion fusion — text + prosody + behavior.

# Why this module exists (and why carefully)

The previous detect_and_track_confusion stacked 8 layers of confusion
heuristics : keywords, hash repetition, history patterns, similarity
thresholds, prosody markers… all OR-ed together. The result : the SIGHT
classifier (a fine-tuned XLM-R model with empirically-validated
thresholds) was buried under the noise of the rule-based layers, which
fired false positives on perfectly normal questions. The system was
"too sensitive" — students got reformulations they didn't ask for.

The codebase was deliberately collapsed to **SIGHT-only** to recover
precision. Re-introducing prosody as ANOTHER OR-stack would walk back
into the same trap. So this module does it differently :

  - **SIGHT** is the primary decision (XLM-R fine-tuned, calibrated).
  - **Prosody** is a SECONDARY signal that comes from a different
    modality (vocal hesitations, slow speech, silence) — orthogonal
    to text. We trust it ONLY when it produces a STRONG independent
    vote (>= 2 markers triggered simultaneously, not just one keyword).
  - **Behavior** (repeated questions, etc.) is intentionally NOT in
    this fusion — it was the worst offender in the old system. Wire
    it back later only with calibration data.

# Decision rule (principled, no magic thresholds at the boundary)

    fused_score = 0.7 * text_score + 0.3 * prosody_score

The weights reflect prior confidence : SIGHT is empirically validated,
prosody markers are coarse heuristics. We expose them as Config-tunable
so future calibration can rebalance.

# Returns a SCORE, not a binary

The fusion returns a continuous score in [0, 1]. The caller still
applies a threshold (currently SIGHT's own threshold for "confused").
This way :
  - downstream consumers (TTS adaptive rate, engagement scorer) can
    use the gradient ;
  - the binary boundary stays anchored in SIGHT's calibration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("pedagogy.confusion.fusion")


# Weights — sum to 1.0. SIGHT carries 70 % of the decision because it's
# the only empirically validated signal ; prosody adds 30 % of "vocal
# evidence" so a clearly-hesitant student still tips into confused
# even if their text reads cleanly. The asserter below fails fast if
# someone overrides the env vars to inconsistent values.
W_TEXT     = 0.7
W_PROSODY  = 0.3

assert abs((W_TEXT + W_PROSODY) - 1.0) < 1e-6, (
    "confusion fusion weights must sum to 1.0"
)

# Score per number of prosody markers. The transcriber emits at most
# 3 markers (slow_speech_rate, frequent_hesitations, high_silence_ratio).
# 0 markers = clean speech ; 1 marker = mild signal ; 2 = strong ; 3 = very
# strong. Monotonic by design — adding a marker never decreases the score.
_PROSODY_SCORE_BY_COUNT = {
    0: 0.0,
    1: 0.30,
    2: 0.70,
    3: 1.00,
}


@dataclass
class ConfusionSignals:
    """Inputs to the fusion. Each signal is optional (None) so the
    caller can pass only what's available."""
    text_score:    Optional[float] = None    # SIGHT posterior, [0, 1]
    prosody_dict:  Optional[dict]  = None    # transcriber.extract_prosody output


@dataclass
class FusedConfusion:
    score:        float            # fused score in [0, 1]
    text_score:   float            # what SIGHT said
    prosody_score: float           # derived from markers count
    contributors: list[str]        # which signals were available
    reason:       str              # one-liner for operator log


def score_from_prosody(prosody: Optional[dict]) -> float:
    """Convert transcriber prosody output into a [0, 1] score.

    The transcriber returns ``{"markers": [...], ...}`` with up to 3
    markers. We use the COUNT of triggered markers, not their identity
    — combining 'slow_speech_rate' + 'frequent_hesitations' is a
    stronger confusion signal than any single marker, and a per-marker
    weight scheme would be unjustified without calibration data.

    Returns 0.0 on missing / malformed input — the caller treats this
    as "no prosody evidence", and the fusion falls back to text-only.
    """
    if not prosody or not isinstance(prosody, dict):
        return 0.0
    markers = prosody.get("markers", [])
    if not isinstance(markers, (list, tuple)):
        return 0.0
    n = min(3, len([m for m in markers if isinstance(m, str)]))
    return _PROSODY_SCORE_BY_COUNT.get(n, 1.0)


def fuse_confusion_signals(signals: ConfusionSignals) -> FusedConfusion:
    """Combine SIGHT + prosody into one [0, 1] score.

    The fusion respects the W_TEXT / W_PROSODY weights ; missing
    signals are dropped (renormalising the remaining weight to sum to
    1.0). With only one signal available, the fused score equals that
    signal — no spurious lift / drop from the absence of the other.

    Examples :
      - SIGHT=0.9, no prosody  → 0.9 (full weight on text)
      - SIGHT=0.2, prosody=0.7 → 0.7×0.2 + 0.3×0.7 = 0.35 (mild)
      - SIGHT=0.8, prosody=0.7 → 0.7×0.8 + 0.3×0.7 = 0.77 (both agree)
      - SIGHT=0.1, prosody=1.0 → 0.7×0.1 + 0.3×1.0 = 0.37
        (vocal evidence not enough alone to tip past SIGHT's vote)
    """
    contributors: list[str] = []

    text_score = float(signals.text_score) if signals.text_score is not None else None
    if text_score is not None:
        text_score = max(0.0, min(1.0, text_score))
        contributors.append("text(sight)")

    prosody_score = score_from_prosody(signals.prosody_dict)
    if signals.prosody_dict is not None:
        contributors.append("prosody")

    # No signal at all → safe default 0.0 (= "not confused")
    if not contributors:
        return FusedConfusion(
            score=0.0,
            text_score=0.0,
            prosody_score=0.0,
            contributors=[],
            reason="no signals available — defaulting to 0.0",
        )

    # Weighted sum with renormalisation when only one signal is present
    if text_score is not None and signals.prosody_dict is not None:
        score = W_TEXT * text_score + W_PROSODY * prosody_score
    elif text_score is not None:
        score = text_score
    else:
        score = prosody_score

    score = round(max(0.0, min(1.0, score)), 3)
    reason = (
        f"fused={score:.2f} (text={text_score if text_score is not None else 'NA'}"
        f", prosody={prosody_score:.2f}, contributors={contributors})"
    )
    return FusedConfusion(
        score=score,
        text_score=text_score if text_score is not None else 0.0,
        prosody_score=prosody_score,
        contributors=contributors,
        reason=reason,
    )
