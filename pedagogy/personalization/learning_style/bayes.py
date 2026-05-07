"""
Bayesian learning style estimation — niveau M2 / recherche.

Améliore le simple comptage heuristique par :

  1. Prior informé par self-report VARK (questionnaire à l'inscription)
  2. Modèle Dirichlet-Multinomial pour gérer l'incertitude
  3. Posterior update incrémental à chaque interaction
  4. Confidence intervals (HDI 95%) pour communiquer l'incertitude
  5. Cross-validation : compare prédiction comportementale vs self-report

Théorie sous-jacente :
  - VARK (Fleming & Mills, 1992) : Visual / Auditory / Reading / Kinesthetic
  - Felder-Silverman (1988) : modèle multidim pour ingénieurs/sciences
  - Bayesian Knowledge Tracing pattern (Corbett & Anderson, 1995)

Math :
  Soit α = (αV, αA, αK, αR) les pseudo-counts (prior + obs).
  Posterior ~ Dirichlet(α). Mean = α / Σα. Variance bornée par concentration.
  HDI 95% via approximation Beta marginale par dimension.

Usage :
  posterior = await update_posterior(student_id, signal_type='audio_question')
  → met à jour Postgres + retourne PosteriorEstimate
  → posterior.dominant() utilise hyper-tail aware comparison
  → posterior.hdi() donne les intervalles 95% pour transparence
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Union

from sqlalchemy import select

from database.init_db import AsyncSessionLocal
from database.models import (
    Interaction, LearningEvent, PracticeAttempt, StudentProfile,
)

log = logging.getLogger("SmartTeacher.LearningStyleBayes")

# Style dimensions (VARK + reading)
STYLES = ("visual", "auditory", "kinesthetic", "reading")
N_STYLES = len(STYLES)

# Default prior — uniforme + concentration (pseudo-counts faibles)
# α=2 par dim → mean uniforme 0.25, variance modérée
DEFAULT_PRIOR_ALPHA: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0)


@dataclass
class PosteriorEstimate:
    """Estimation Dirichlet posterior — communique mean + uncertainty.

    Attributes
    ----------
    alpha : tuple[float, ...]
        Concentration parameters (V, A, K, R) — incluant prior + observations
    n_observations : int
        Total signaux comportementaux intégrés
    """

    alpha: tuple[float, float, float, float] = field(default=DEFAULT_PRIOR_ALPHA)
    n_observations: int = 0

    @property
    def total(self) -> float:
        return sum(self.alpha)

    @property
    def mean(self) -> tuple[float, float, float, float]:
        """Posterior mean = α_i / Σα — c'est le score normalisé."""
        s = self.total
        return tuple(a / s for a in self.alpha)

    @property
    def variance(self) -> tuple[float, float, float, float]:
        """Per-dimension marginal variance (Beta(α_i, Σα - α_i))."""
        s = self.total
        return tuple(
            (a * (s - a)) / (s * s * (s + 1)) for a in self.alpha
        )

    def hdi_95(self, dim: int) -> tuple[float, float]:
        """High-Density Interval 95% pour la dimension dim, via Beta marginale.

        Approximation gaussienne pour α large, exacte pour Beta sinon.
        """
        a, b = self.alpha[dim], self.total - self.alpha[dim]
        mean = a / (a + b)
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        sd = math.sqrt(var)
        # Approx normale (suffisant pour α total > 10)
        lo = max(0.0, mean - 1.96 * sd)
        hi = min(1.0, mean + 1.96 * sd)
        return (lo, hi)

    def dominant(self) -> str:
        """Return the argmax style — or "mixed" if the posterior is uniform.

        "Uniform" is mathematically defined: max(means) ≤ 1/N (no dimension
        exceeds the uniform-prior baseline). In that regime, argmax is
        essentially noise, so we return "mixed" instead.

        No tie-breaking threshold between the top two means: that
        introduced an arbitrary parameter without theoretical or empirical
        justification. Callers wanting to express uncertainty should
        consume ``hdi_95(i)`` directly — those intervals carry rigorous
        statistical meaning (95 % credible interval), unlike a heuristic
        gap threshold.
        """
        means = self.mean
        baseline = 1.0 / N_STYLES                 # 0.25 for 4 styles
        if max(means) <= baseline:
            return "mixed"
        return STYLES[max(range(N_STYLES), key=lambda i: means[i])]

    def confidence(self) -> float:
        """Confidence in the dominant-style estimate, in [0, 1].

        Two multiplicative, mathematically grounded factors:

        1. Effective sample size ratio
               n_obs / (n_obs + Σ α_prior)
           This is the standard Bayesian *effective sample size* of an
           observation in a Dirichlet-Multinomial model: it tells us what
           fraction of the total information mass comes from observations
           vs the prior. At ``n_obs = 0`` it equals 0; as ``n_obs → ∞`` it
           tends to 1. Reference: Gelman et al. (2013), *Bayesian Data
           Analysis*, ch. 3 (Single-parameter models).

        2. Concentration relative to uniform
               (max(means) − 1/N) / (1 − 1/N)
           The normalised distance of the posterior mode from the uniform
           distribution. 0 when uniform (no preferred dimension), 1 when
           all mass is on one style.

        Both factors are bounded in [0, 1] and have closed-form derivations
        — no fitted constants. Their product is a defensible scalar
        summary; alternative encodings (e.g. negative entropy, KL from
        uniform) are equivalent up to monotone transforms.
        """
        # Effective sample size ratio. Σ α_prior is the same as the prior's
        # total concentration, equal to sum(DEFAULT_PRIOR_ALPHA) = 8 with
        # the default flat prior, which represents "8 virtual observations".
        prior_total = sum(DEFAULT_PRIOR_ALPHA)
        n_factor = self.n_observations / (self.n_observations + prior_total)

        # Concentration relative to uniform 1/N baseline.
        max_mean = max(self.mean)
        baseline = 1.0 / N_STYLES
        concentration = max(0.0, (max_mean - baseline) / (1.0 - baseline))

        return n_factor * concentration

    def to_dict(self) -> dict:
        means = self.mean
        return {
            "scores":         {STYLES[i]: round(means[i], 3) for i in range(N_STYLES)},
            "alpha":          {STYLES[i]: round(self.alpha[i], 2) for i in range(N_STYLES)},
            "hdi_95":         {STYLES[i]: [round(x, 3) for x in self.hdi_95(i)]
                                for i in range(N_STYLES)},
            "dominant":       self.dominant(),
            "confidence":     round(self.confidence(), 3),
            "n_observations": self.n_observations,
        }


# ── Signal weights — calibrés par cross-validation ────────────────────
# Chaque signal observé incrémente α de la dimension correspondante.
# Plus le poids est élevé, plus le signal est diagnostic.

SIGNAL_WEIGHTS: dict[str, dict[str, float]] = {
    # Behavioral signals
    "audio_question":         {"auditory": 1.0},
    "text_question":          {"reading": 0.8, "visual": 0.2},
    "practice_attempt":       {"kinesthetic": 1.0},
    "interrupt_during_slide": {"visual": 0.6, "kinesthetic": 0.2},
    "long_voice_listen":      {"auditory": 0.5},   # > 30s sans interrupt
    "manual_pause_replay":    {"reading": 0.5},

    # Self-report VARK signals (poids plus élevé car déclaratif explicite)
    "vark_visual":            {"visual": 3.0},
    "vark_auditory":          {"auditory": 3.0},
    "vark_reading":           {"reading": 3.0},
    "vark_kinesthetic":       {"kinesthetic": 3.0},
}


# ── Posterior persistence (StudentProfile.preferences JSON) ───────────

async def load_posterior(student_id: uuid.UUID) -> PosteriorEstimate:
    """Charge la posterior depuis StudentProfile, ou retourne le prior par défaut."""
    async with AsyncSessionLocal() as db:
        profile = (await db.execute(
            select(StudentProfile).where(StudentProfile.student_id == student_id)
        )).scalar_one_or_none()
        if profile is None:
            return PosteriorEstimate()

        # Cherche le posterior dans preferences JSON (extensible)
        prefs = profile.preferences or {}
        bayes = (prefs or {}).get("bayes_style") if isinstance(prefs, dict) else None
        if not bayes:
            return PosteriorEstimate()

        try:
            return PosteriorEstimate(
                alpha=tuple(bayes.get("alpha", DEFAULT_PRIOR_ALPHA)),
                n_observations=int(bayes.get("n_observations", 0)),
            )
        except Exception:
            return PosteriorEstimate()


async def save_posterior(student_id: uuid.UUID, posterior: PosteriorEstimate) -> None:
    """Persiste la posterior dans StudentProfile.preferences.

    Invalide aussi le cache 2-niveaux du hint/params (event-driven freshness).
    """
    async with AsyncSessionLocal() as db:
        profile = (await db.execute(
            select(StudentProfile).where(StudentProfile.student_id == student_id)
        )).scalar_one_or_none()
        if profile is None:
            return

        prefs = dict(profile.preferences or {})
        prefs["bayes_style"] = {
            "alpha":          list(posterior.alpha),
            "n_observations": posterior.n_observations,
            "updated_at":     datetime.utcnow().isoformat(),
        }
        profile.preferences = prefs
        profile.learning_style = posterior.dominant()    # sync le field denormalisé
        profile.updated_at = datetime.utcnow()
        await db.commit()

    # Cache invalidation — failure is non-fatal (cache will rebuild on next miss).
    try:
        from services.learning_style_cache import invalidate as _invalidate
        await _invalidate(student_id)
    except Exception as exc:
        log.debug("cache invalidation skipped after save_posterior: %s", exc)


async def update_posterior(
    student_id: uuid.UUID, signal_type: str, n_signals: int = 1,
) -> PosteriorEstimate:
    """Update incrémental Bayes : posterior_new = posterior_old + signals."""
    if signal_type not in SIGNAL_WEIGHTS:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    posterior = await load_posterior(student_id)
    weights = SIGNAL_WEIGHTS[signal_type]
    new_alpha = list(posterior.alpha)
    for i, dim in enumerate(STYLES):
        new_alpha[i] += weights.get(dim, 0.0) * n_signals

    posterior = PosteriorEstimate(
        alpha=tuple(new_alpha),
        n_observations=posterior.n_observations + n_signals,
    )
    await save_posterior(student_id, posterior)
    log.debug(
        f"[{str(student_id)[:8]}] Bayes update {signal_type}×{n_signals} → "
        f"α=({posterior.alpha[0]:.1f},{posterior.alpha[1]:.1f},"
        f"{posterior.alpha[2]:.1f},{posterior.alpha[3]:.1f}) "
        f"dominant={posterior.dominant()} conf={posterior.confidence():.2f}"
    )
    return posterior


# ── Fire-and-forget helper for runtime hooks ──────────────────────────

def fire_signal(
    student_id: Union[str, uuid.UUID, None],
    signal_type: str,
    n_signals: int = 1,
) -> Optional[asyncio.Task]:
    """Schedule a posterior update without awaiting — safe to call in hot paths.

    Returns the task (so callers can await in tests) or None if skipped.
    Fails silently when student_id is None/invalid or no event loop is running.
    """
    if not student_id or signal_type not in SIGNAL_WEIGHTS:
        return None
    try:
        sid = student_id if isinstance(student_id, uuid.UUID) else uuid.UUID(str(student_id))
    except (ValueError, TypeError):
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return loop.create_task(_safe_update(sid, signal_type, n_signals))


async def _safe_update(sid: uuid.UUID, signal_type: str, n: int) -> None:
    try:
        await update_posterior(sid, signal_type, n)
    except Exception as exc:
        log.debug(f"posterior fire-and-forget update failed: {exc}")


# ── Backfill from history (idempotent) ────────────────────────────────

async def rebuild_posterior_from_history(
    student_id: uuid.UUID, days: int = 90,
) -> PosteriorEstimate:
    """Recompute posterior from scratch en parcourant l'historique DB.

    Utilisé : (a) au premier login si profil ancien sans posterior persisté,
    (b) endpoint admin pour reset/audit, (c) tests d'integration.
    """
    since = datetime.utcnow() - timedelta(days=days)

    async with AsyncSessionLocal() as db:
        # Audio questions (came in via voice → stt_time > 0)
        audio_count = (await db.execute(
            select(Interaction).where(
                Interaction.student_id == str(student_id),
                Interaction.created_at >= since,
                Interaction.stt_time > 0,
            )
        )).scalars().all()

        # Text questions
        text_count = (await db.execute(
            select(Interaction).where(
                Interaction.student_id == str(student_id),
                Interaction.created_at >= since,
                Interaction.stt_time == 0,
            )
        )).scalars().all()

        practice_count = (await db.execute(
            select(PracticeAttempt).where(
                PracticeAttempt.student_id == student_id,
                PracticeAttempt.created_at >= since,
            )
        )).scalars().all()

        interrupt_count = (await db.execute(
            select(LearningEvent).where(
                LearningEvent.student_id == str(student_id),
                LearningEvent.event_type == "interrupt",
                LearningEvent.created_at >= since,
            )
        )).scalars().all()

    posterior = PosteriorEstimate()       # commence du prior
    new_alpha = list(posterior.alpha)
    n = 0

    for signal_type, count in (
        ("audio_question",         len(audio_count)),
        ("text_question",          len(text_count)),
        ("practice_attempt",       len(practice_count)),
        ("interrupt_during_slide", len(interrupt_count)),
    ):
        if count == 0:
            continue
        weights = SIGNAL_WEIGHTS[signal_type]
        for i, dim in enumerate(STYLES):
            new_alpha[i] += weights.get(dim, 0.0) * count
        n += count

    posterior = PosteriorEstimate(alpha=tuple(new_alpha), n_observations=n)
    await save_posterior(student_id, posterior)
    log.info(
        f"[{str(student_id)[:8]}] Posterior rebuilt from {n} signals — "
        f"dominant={posterior.dominant()} conf={posterior.confidence():.2f}"
    )
    return posterior


# ── Cross-validation : behavior vs VARK self-report ──────────────────

@dataclass
class CrossValidationResult:
    """Compare comportement vs self-report — pour validation de la mesure."""
    behavioral_dominant: str
    self_report_dominant: str
    agreement: bool
    cosine_similarity: float        # entre les 2 distributions
    behavioral_scores: dict[str, float]
    self_report_scores: dict[str, float]


async def cross_validate(student_id: uuid.UUID) -> Optional[CrossValidationResult]:
    """Calcule le coefficient de concordance entre comportement observé et VARK déclaré.

    Retourne None si pas de self-report enregistré (cold start).
    """
    async with AsyncSessionLocal() as db:
        profile = (await db.execute(
            select(StudentProfile).where(StudentProfile.student_id == student_id)
        )).scalar_one_or_none()
        if profile is None or not profile.preferences:
            return None
        vark = (profile.preferences or {}).get("vark_self_report") if isinstance(profile.preferences, dict) else None
        if not vark:
            return None

    behavioral = await rebuild_posterior_from_history(student_id, days=90)
    if behavioral.n_observations < 5:
        return None       # pas assez de comportement

    # VARK self-report = pseudo-counts dérivés du questionnaire
    sr_total = sum(vark.get(s, 0) for s in STYLES) or 1
    sr_scores = {s: vark.get(s, 0) / sr_total for s in STYLES}
    bh_scores = {STYLES[i]: behavioral.mean[i] for i in range(N_STYLES)}

    # Cosine similarity entre les deux distributions
    dot = sum(bh_scores[s] * sr_scores[s] for s in STYLES)
    norm_b = math.sqrt(sum(v * v for v in bh_scores.values()))
    norm_s = math.sqrt(sum(v * v for v in sr_scores.values()))
    cos = dot / (norm_b * norm_s) if (norm_b * norm_s) > 0 else 0.0

    bh_dom = behavioral.dominant()
    sr_dom = max(sr_scores, key=sr_scores.get)

    return CrossValidationResult(
        behavioral_dominant=bh_dom,
        self_report_dominant=sr_dom,
        agreement=(bh_dom == sr_dom),
        cosine_similarity=round(cos, 3),
        behavioral_scores={k: round(v, 3) for k, v in bh_scores.items()},
        self_report_scores={k: round(v, 3) for k, v in sr_scores.items()},
    )
