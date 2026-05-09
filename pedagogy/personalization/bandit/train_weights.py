"""Train reward weights from offline simulation data via Ridge regression.

# What the weights mean pedagogically

The bandit's per-turn reward is a weighted sum of three signals:

  - ``W_CONFUSION``  — penalty for student confusion. The dominant
                       signal in pedagogical literature (Bloom 1968,
                       Sweller 1985) — confusion is the strongest
                       indicator that the explanation missed.
  - ``W_MASTERY``    — reward for measurable mastery progress on the
                       targeted concept.
  - ``W_ENGAGEMENT`` — reward for the student staying with the session
                       (no bail-out, follow-up question, navigation).

The current production weights (0.50 / 0.40 / 0.10) reflect an
editorial choice. This script fits a Ridge regression on the
simulation rollout (``turn_history.json``) to *learn* which mix of
the three observable signals best predicts long-horizon learning
gain, then writes those weights to ``learned_weights.json`` for
``reward.py`` to load.

# Target: long-horizon mastery delta, not stored reward

Earlier versions of this script regressed on the per-turn ``reward``
field stored in ``turn_history.json``. That field was itself
``HARDCODED_W·X`` — so the regression was fitting a linear formula on
the output of the same linear formula and trivially recovered R²=1
plus the original weights (modulo Ridge bias). A tautology, not a
finding.

The honest target is something the formula did NOT generate. We use
the cumulative mastery gain over an ``H``-turn lookahead window:

    y[i] = mastery_after[i + H] - mastery_before[i]

This asks "given the (confusion, mastery_signal, engagement) trio
observed at turn i, how much mastery does this student accumulate over
the next H turns?". Features are still the per-turn signals; the
target is the downstream learning outcome they're supposed to predict.

Two operational details:

  - **Group by archetype first.** ``turn_history.json`` is a flat
    concatenation of per-archetype runs. A naive ``i + H`` lookup at
    a seam reads "FastLearner's t=480 features → DeepThinker's t=0
    mastery", which is nonsense. We bucket by ``archetype`` and only
    look ahead within the same bucket.
  - **Drop the trailing H turns.** They have no future to look at.

# Why Ridge instead of plain OLS

Ridge (alpha=1.0) shrinks coefficients toward zero, which is the
right inductive bias when features are correlated (confusion absence
and engagement co-occur strongly in well-fitting arms). It also
prevents the optimizer from over-allocating weight to whichever
feature happens to have slightly more variance in this finite sample.

Run standalone with::

    python -m pedagogy.personalization.bandit.train_weights
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


HISTORY_FILE = Path(__file__).resolve().parent / "turn_history.json"
OUT_FILE     = Path(__file__).resolve().parent / "learned_weights.json"

# Normalizer for the mastery_delta feature. Set to the empirical
# maximum per-turn gain across all archetypes (FastLearner.GAIN_MAX =
# 0.15). Using the empirical max instead of reward.py's conservative
# 0.30 cap keeps the feature in [0, 1] without throwing away
# resolution at the high end of observed gains.
MAX_DELTA   = 0.15
RIDGE_ALPHA = 1.0

# Lookahead horizon (turns) for the mastery-gain target. Tuned to
# match the bandit's effective decision window — long enough that a
# single noisy turn doesn't dominate the target, short enough to stay
# inside one mastery-decay cycle (50 turns) most of the time. If R²
# comes out < 0.1 the window is too long for the noise floor; try 10.
LOOKAHEAD_H = 20

HARDCODED = {
    "W_CONFUSION":  0.50,
    "W_MASTERY":    0.40,
    "W_ENGAGEMENT": 0.10,
}


def _features_row(record: dict) -> tuple[float, float, float]:
    """Map one turn record to its three reward features in [0, 1]."""
    confusion_signal  = 0.0 if record["confusion_detected"] else 1.0
    mastery_signal    = max(0.0, min(1.0, record["mastery_delta"] / MAX_DELTA))
    engagement_signal = 1.0 if record["engaged"] else 0.0
    return confusion_signal, mastery_signal, engagement_signal


def _build_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for Ridge.

    X[:, [0, 1, 2]] = (confusion_signal, mastery_signal, engagement_signal)
    per turn — same as before.

    y[i] = mastery_after[i + LOOKAHEAD_H] - mastery_before[i],
    computed *within each archetype's slice* of the history. This
    breaks the circular dependency on the stored reward field, which
    was itself ``HARDCODED·X`` and made the regression a tautology.
    """
    # Bucket by archetype so the lookahead window never crosses the
    # seam between two archetypes' rollouts.
    by_archetype: dict[str, list[dict]] = {}
    for r in records:
        by_archetype.setdefault(r["archetype"], []).append(r)

    rows_X: list[tuple[float, float, float]] = []
    rows_y: list[float] = []
    for arche_records in by_archetype.values():
        # Drop the trailing LOOKAHEAD_H turns: no future to look at.
        usable = len(arche_records) - LOOKAHEAD_H
        if usable <= 0:
            continue
        for i in range(usable):
            now    = arche_records[i]
            future = arche_records[i + LOOKAHEAD_H]
            rows_X.append(_features_row(now))
            rows_y.append(future["mastery_after"] - now["mastery_before"])

    X = np.asarray(rows_X, dtype=float)
    y = np.asarray(rows_y, dtype=float)
    return X, y


def _normalize_to_simplex(coefs: np.ndarray) -> np.ndarray:
    """Project a non-negative coefficient vector onto the unit simplex
    (sum = 1). Falls back to the uniform vector if all coefs are zero."""
    coefs = np.clip(coefs, 0.0, None)
    s = coefs.sum()
    if s <= 0:
        return np.full_like(coefs, 1.0 / len(coefs))
    return coefs / s


def _weighted_reward(X: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    w = np.array([weights["W_CONFUSION"], weights["W_MASTERY"], weights["W_ENGAGEMENT"]])
    return X @ w


def main() -> None:
    with HISTORY_FILE.open(encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    X, y = _build_features(records)

    # ``positive=True`` keeps the weights non-negative — required for
    # the simplex normalization step and consistent with the
    # interpretation that all three signals are *positive* indicators
    # of a good turn (confusion is signed via ``1 - confusion_detected``).
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False, positive=True)
    model.fit(X, y)

    coefs = _normalize_to_simplex(model.coef_)
    learned = {
        "W_CONFUSION":  float(coefs[0]),
        "W_MASTERY":    float(coefs[1]),
        "W_ENGAGEMENT": float(coefs[2]),
    }

    y_pred = model.predict(X)
    r2 = float(r2_score(y, y_pred))

    out = {
        "W_CONFUSION":      round(learned["W_CONFUSION"], 4),
        "W_MASTERY":        round(learned["W_MASTERY"], 4),
        "W_ENGAGEMENT":     round(learned["W_ENGAGEMENT"], 4),
        "r2_score":         round(r2, 4),
        "trained_on_samples": int(X.shape[0]),
        "trained_on_turns":   len(records),
        "lookahead_h":        LOOKAHEAD_H,
        "target":             "mastery_after[t+H] - mastery_before[t]",
        "note": "Learned via Ridge regression. Predicts H-turn mastery gain from per-turn features.",
    }
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ── Report ──────────────────────────────────────────────────────
    print(
        f"Trained on {X.shape[0]} samples (lookahead H={LOOKAHEAD_H}) "
        f"from {len(records)} turns in {HISTORY_FILE.name}"
    )
    print(f"R^2 on training data: {r2:.4f}")
    print()
    print("Weights (sum = 1.0):")
    print(f"  {'':<14}{'hardcoded':>12}{'learned':>10}{'delta':>10}")
    for name in ("W_CONFUSION", "W_MASTERY", "W_ENGAGEMENT"):
        h = HARDCODED[name]
        l = learned[name]
        print(f"  {name:<14}{h:>12.4f}{l:>10.4f}{l - h:>+10.4f}")
    print()

    # Per-archetype mean reward, both weight sets applied to the SAME
    # feature matrix so the comparison isolates the effect of weights.
    # We use the FULL per-turn feature matrix (one row per record) for
    # this diagnostic — it's a per-turn reward comparison, independent
    # of the lookahead window used for training.
    archetypes: list[str] = []
    for r in records:
        if r["archetype"] not in archetypes:
            archetypes.append(r["archetype"])

    X_full = np.asarray([_features_row(r) for r in records], dtype=float)
    arche_arr = np.asarray([r["archetype"] for r in records])

    print("Per-archetype mean reward (per-turn, all records):")
    print(f"  {'Archetype':<22}{'hardcoded':>12}{'learned':>10}{'delta':>10}")
    hc_all = _weighted_reward(X_full, HARDCODED)
    lr_all = _weighted_reward(X_full, learned)
    for arche in archetypes:
        mask = arche_arr == arche
        hc = float(hc_all[mask].mean())
        lr = float(lr_all[mask].mean())
        print(f"  {arche:<22}{hc:>12.4f}{lr:>10.4f}{lr - hc:>+10.4f}")

    print()
    print(f"Wrote {OUT_FILE.name}")


if __name__ == "__main__":
    main()
