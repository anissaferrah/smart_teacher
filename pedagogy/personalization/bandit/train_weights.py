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
simulation rollout (``turn_history.json``) to *learn* what weights the
observed reward signal is actually consistent with — useful as a
sanity check before swapping production weights, and as a starting
point for an empirical ablation study.

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

HARDCODED = {
    "W_CONFUSION":  0.50,
    "W_MASTERY":    0.40,
    "W_ENGAGEMENT": 0.10,
}


def _build_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y). X[:, [0, 1, 2]] = (confusion_signal, mastery_signal,
    engagement_signal). y = stored per-turn reward.

    ``mastery_signal`` is floored at 0 (mastery_delta can be negative
    in principle if the simulator ever regresses; in practice the
    archetypes always gain, but we keep the guard) and capped at 1.
    """
    n = len(records)
    X = np.zeros((n, 3), dtype=float)
    y = np.zeros(n, dtype=float)
    for i, r in enumerate(records):
        X[i, 0] = 0.0 if r["confusion_detected"] else 1.0
        delta_norm = r["mastery_delta"] / MAX_DELTA
        X[i, 1] = max(0.0, min(1.0, delta_norm))
        X[i, 2] = 1.0 if r["engaged"] else 0.0
        y[i] = r["reward"]
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
        "trained_on_turns": len(records),
        "note": "Learned via Ridge regression on simulation data. Replaces hardcoded weights in reward.py.",
    }
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ── Report ──────────────────────────────────────────────────────
    print(f"Trained on {len(records)} turns from {HISTORY_FILE.name}")
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
    # feature matrix so the comparison isolates the effect of weights
    # (not feature normalization).
    archetypes: list[str] = []
    for r in records:
        if r["archetype"] not in archetypes:
            archetypes.append(r["archetype"])

    print("Per-archetype mean reward:")
    print(f"  {'Archetype':<22}{'hardcoded':>12}{'learned':>10}{'delta':>10}")
    hc_all = _weighted_reward(X, HARDCODED)
    lr_all = _weighted_reward(X, learned)
    for arche in archetypes:
        mask = np.array([r["archetype"] == arche for r in records])
        hc = float(hc_all[mask].mean())
        lr = float(lr_all[mask].mean())
        print(f"  {arche:<22}{hc:>12.4f}{lr:>10.4f}{lr - hc:>+10.4f}")

    print()
    print(f"Wrote {OUT_FILE.name}")


if __name__ == "__main__":
    main()
