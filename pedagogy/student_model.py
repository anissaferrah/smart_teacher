"""StudentModel — single consolidated view of an enrolled student.

Why this exists
---------------
Today, the Q&A responder, the teaching planner, and the bandit each
fetch student-state slices independently :

  - ``MasteryRepo.get_scores_bulk(student_id, course_id, idea_ids)``
  - ``get_or_create_profile(session_id, course_id=...)``
  - ``PersonalizationEngine.get_context(student_id, course_id)``
  - ``learning_style.bayes.load_posterior(student_id)``
  - ``ConfusionDetector.score(...)``

Each of those is best-effort, async, and has its own failure mode. The
result is that every consumer sprinkles try/except blocks around the
same five fetches, with subtly different fallback values — a maintenance
trap. ``StudentModel`` exposes ONE async ``fetch(student_id, course_id)``
that returns a frozen snapshot, with sensible defaults filled in for
every missing piece.

The class is intentionally **read-only**. Writes still go through their
canonical store (``MasteryRepo.record_clean``, ``ProfileManager.save``,
etc.) ; this aggregator is for *reading* the consolidated state at the
top of a turn or planner call.

Design choices
--------------

  - **Dataclass, not Pydantic** : matches the rest of ``agentic/schemas.py``
    and avoids a runtime validation cost on every turn.
  - **Always returns a value** : every field has a default, so even with
    ``redis``/``postgres`` down the model is usable. The ``ok`` field
    flags whether the fetch was complete or fell back to defaults.
  - **No I/O in ``__init__``** : construction is cheap. All I/O is in
    the ``fetch()`` classmethod.
  - **Per-(student, course) scoping** : every read is scoped by
    ``course_id`` where the underlying store supports it (mastery,
    profile). The Bayesian style posterior is currently per-student
    only — flagged for a future migration when we have real per-course
    behavioural data to justify the split.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("pedagogy.student_model")


# ── Constants ───────────────────────────────────────────────────────────
# Default mastery for unseen concepts. 0.5 = "unknown / 50/50" — the
# Bayesian Beta posterior with Laplace's rule of succession on (1, 1)
# prior gives exactly this value with zero observations. Not a tuning
# knob ; it's the prior mean. See ``pedagogy.mastery_repo``.
_DEFAULT_MASTERY = 0.5

# Default learning style when no posterior is available. "mixed" is the
# explicit "we don't know yet" label — preferred over picking a fake
# dominant style on a flat prior.
_DEFAULT_STYLE = "mixed"


# ── Dataclass ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StudentModel:
    """Frozen snapshot of one student's state on one course.

    Frozen because consumers should NOT mutate this — writes go through
    their canonical stores. If a consumer needs an updated view, they
    re-call ``StudentModel.fetch(...)``.
    """

    # ── Identity ────────────────────────────────────────────────────
    student_id: str
    course_id: Optional[str]      # None for cross-course / cold start

    # ── Mastery (per concept, Bayesian Beta posterior mean) ─────────
    # Empty when no concepts have been touched yet OR when MasteryRepo
    # is unreachable. Use ``mastery_for(idea_ids)`` to read defensively.
    mastery: dict[str, float] = field(default_factory=dict)

    # ── Style (Dirichlet-Multinomial posterior + heuristic fallback) ─
    # ``style_dominant``  : argmax of the posterior, or "mixed" on flat prior.
    # ``style_scores``    : full V/A/R/K distribution (sums to 1).
    # ``style_confidence``: in [0, 1], reflects evidence × concentration.
    style_dominant: str = _DEFAULT_STYLE
    style_scores: dict[str, float] = field(default_factory=dict)
    style_confidence: float = 0.0

    # ── Pacing (per-course, derived from PracticeAttempt window) ───
    # ``pace`` : "slow" / "normal" / "fast" / "auto" (when undetermined)
    # ``avg_response_time_s`` : empirical mean over the recent window
    # ``confusion_rate``      : incorrect / total over the same window
    pace: str = "normal"
    avg_response_time_s: float = 0.0
    confusion_rate: float = 0.0

    # ── Engagement (session-level, runtime signal) ─────────────────
    # Currently a placeholder — session_manager doesn't yet expose a
    # consolidated engagement metric. Wired in fetch() as a hook so
    # downstream consumers can already read it ; the value is 1.0
    # (engaged) by default until we add proper signal extraction
    # (interruption rate, click-through, time-on-slide).
    engagement: float = 1.0

    # ── Cognitive preferences (echoed from the SQL profile row) ────
    # These mirror columns in ``database.models.StudentProfile`` so the
    # narrator / responder doesn't need to fetch the row separately.
    preferred_explanation_depth: str = "balanced"     # concise|balanced|detailed
    preferred_difficulty: str = "intermediate"
    speech_rate: float = 1.0

    # ── Diagnostics ─────────────────────────────────────────────────
    # ``ok`` is False when at least one source failed silently. Useful
    # for downstream consumers that want to fall back gracefully (e.g.
    # disable bandit if mastery couldn't be fetched).
    ok: bool = True
    fetched_at: float = 0.0
    sources_failed: tuple[str, ...] = ()

    # ── Convenience accessors ──────────────────────────────────────

    def mastery_for(self, idea_ids: list[str], default: float = _DEFAULT_MASTERY) -> dict[str, float]:
        """Return mastery for a list of ideas, falling back to ``default`` per missing key.

        Used by the bandit + responder which call this with the idea_ids
        of the retrieved chunks for a turn.
        """
        return {iid: self.mastery.get(iid, default) for iid in idea_ids}

    def avg_mastery(self, idea_ids: list[str] | None = None) -> float:
        """Mean mastery over ``idea_ids`` (or all known concepts).

        Returns ``_DEFAULT_MASTERY`` when there's nothing to average.
        Used by the bandit's ``mastery_score`` context feature.
        """
        if idea_ids is not None:
            scored = self.mastery_for(idea_ids)
            return sum(scored.values()) / len(scored) if scored else _DEFAULT_MASTERY
        if not self.mastery:
            return _DEFAULT_MASTERY
        return sum(self.mastery.values()) / len(self.mastery)

    def is_struggling(self, threshold: float = 0.30) -> bool:
        """Heuristic flag. Threshold matches the engine's tone-router cutoff
        (0.30 confusion_rate → encouraging tone). Documented there ;
        kept consistent here so consumers don't drift apart.
        """
        return self.confusion_rate > threshold

    def to_dict(self) -> dict:
        """JSON-friendly view, used by /student/me endpoints + analytics."""
        return {
            "student_id": self.student_id,
            "course_id": self.course_id,
            "mastery_avg": round(self.avg_mastery(), 3),
            "mastery_count": len(self.mastery),
            "style": {
                "dominant": self.style_dominant,
                "scores": {k: round(v, 3) for k, v in self.style_scores.items()},
                "confidence": round(self.style_confidence, 3),
            },
            "pace": self.pace,
            "avg_response_time_s": round(self.avg_response_time_s, 2),
            "confusion_rate": round(self.confusion_rate, 3),
            "engagement": round(self.engagement, 3),
            "preferred_explanation_depth": self.preferred_explanation_depth,
            "preferred_difficulty": self.preferred_difficulty,
            "speech_rate": round(self.speech_rate, 2),
            "ok": self.ok,
            "sources_failed": list(self.sources_failed),
            "fetched_at": round(self.fetched_at, 3),
        }


# ── Async fetch ─────────────────────────────────────────────────────────

def _coerce_uuid(value) -> Optional[uuid.UUID]:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


async def fetch_student_model(
    student_id: str,
    course_id: Optional[str] = None,
    *,
    idea_ids: Optional[list[str]] = None,
) -> StudentModel:
    """Build the consolidated :class:`StudentModel` for ``(student, course)``.

    Pulls in parallel from :
      1. ``MasteryRepo.get_scores_bulk`` (only when ``idea_ids`` provided —
         we don't want to scan the whole concept graph on every turn).
      2. ``ProfileManager`` for the cognitive prefs (Redis dataclass).
      3. ``PersonalizationEngine.get_context`` for derived pace/depth/tone.
      4. ``learning_style.bayes.load_posterior`` for V/A/R/K.

    All four reads are best-effort and run concurrently. Failures are
    captured in ``sources_failed`` ; the returned model is always usable.
    """
    started = time.time()
    failed: list[str] = []

    # ── Run all reads in parallel ─────────────────────────────────
    mastery_task = _safe_fetch_mastery(student_id, course_id, idea_ids or [])
    profile_task = _safe_fetch_profile(student_id, course_id)
    context_task = _safe_fetch_context(student_id, course_id)
    posterior_task = _safe_fetch_posterior(student_id)

    mastery, profile, context, posterior = await asyncio.gather(
        mastery_task, profile_task, context_task, posterior_task,
        return_exceptions=False,        # the helpers swallow their own errors
    )

    # ── Mastery slice ─────────────────────────────────────────────
    if mastery is None:
        failed.append("mastery")
        mastery_dict: dict[str, float] = {}
    else:
        mastery_dict = mastery

    # ── Profile slice (Redis dataclass) ──────────────────────────
    if profile is None:
        failed.append("profile")
        speech_rate = 1.0
        preferred_depth = "balanced"
        preferred_difficulty = "intermediate"
    else:
        speech_rate = float(profile.get("speech_rate", 1.0))
        preferred_depth = str(profile.get("preferred_explanation_depth", "balanced"))
        # ProfileManager doesn't carry preferred_difficulty (that's on
        # the SQL row only). The PersonalizationContext below picks it
        # up. Default here for the Redis-only fallback path.
        preferred_difficulty = "intermediate"

    # ── Personalization context (DB-backed pace/confusion/depth) ─
    if context is None:
        failed.append("personalization_context")
        pace = "normal"
        avg_rt = 0.0
        confusion_rate = 0.0
        # depth/difficulty already defaulted
    else:
        pace = context.pace
        avg_rt = float(context.avg_response_time_s)
        confusion_rate = float(context.confusion_rate)
        if not preferred_depth or preferred_depth == "balanced":
            preferred_depth = context.explanation_depth
        preferred_difficulty = context.preferred_difficulty or preferred_difficulty

    # ── Style posterior ──────────────────────────────────────────
    if posterior is None:
        failed.append("style_posterior")
        style_dominant = _DEFAULT_STYLE
        style_scores: dict[str, float] = {}
        style_confidence = 0.0
    else:
        style_dominant = posterior.dominant()
        # ``mean`` is a tuple (V, A, K, R) ; pair with STYLES for dict.
        from pedagogy.personalization.learning_style.bayes import STYLES
        means = posterior.mean
        style_scores = {STYLES[i]: float(means[i]) for i in range(len(STYLES))}
        style_confidence = float(posterior.confidence())

    model = StudentModel(
        student_id=student_id,
        course_id=course_id,
        mastery=mastery_dict,
        style_dominant=style_dominant,
        style_scores=style_scores,
        style_confidence=style_confidence,
        pace=pace,
        avg_response_time_s=avg_rt,
        confusion_rate=confusion_rate,
        engagement=1.0,                       # hook, no signal yet
        preferred_explanation_depth=preferred_depth,
        preferred_difficulty=preferred_difficulty,
        speech_rate=speech_rate,
        ok=not failed,
        sources_failed=tuple(failed),
        fetched_at=started,
    )

    # One-line operator-visible summary per fetch. Compact on purpose —
    # this fires once per turn (per Q&A response) so we want it present
    # in logs without flooding. The ``ok`` flag tells the operator at
    # a glance whether the snapshot is complete or fell back to defaults.
    elapsed_ms = (time.time() - started) * 1000
    status = "✅" if model.ok else "⚠️"
    log.info(
        "👤 StudentModel %s  student=%s course=%s  "
        "mastery_avg=%.2f (%d ideas)  style=%s (conf=%.2f)  "
        "pace=%s  confusion=%.2f  depth=%s  [%dms%s]",
        status,
        student_id[:8] if student_id else "?",
        (course_id or "_global")[:8],
        model.avg_mastery(),
        len(model.mastery),
        model.style_dominant,
        model.style_confidence,
        model.pace,
        model.confusion_rate,
        model.preferred_explanation_depth,
        int(elapsed_ms),
        ("" if model.ok else f", failed={list(model.sources_failed)}"),
    )
    return model


# ── Per-source safe wrappers ────────────────────────────────────────────
# Each one returns ``None`` on failure so the caller above can record
# the source name in ``sources_failed`` without try/except clutter at
# every site.

async def _safe_fetch_mastery(
    student_id: str,
    course_id: Optional[str],
    idea_ids: list[str],
) -> Optional[dict[str, float]]:
    if not idea_ids:
        return {}                              # nothing to fetch is success, not failure
    try:
        from pedagogy.mastery_repo import MasteryRepo
        return await MasteryRepo.get_scores_bulk(student_id, course_id, idea_ids)
    except Exception as exc:                                                # noqa: BLE001
        log.debug("StudentModel : mastery fetch failed (%s)", exc)
        return None


async def _safe_fetch_profile(
    student_id: str,
    course_id: Optional[str],
) -> Optional[dict]:
    try:
        from pedagogy.personalization.profile import get_or_create_profile
        return await get_or_create_profile(student_id, course_id=course_id)
    except Exception as exc:                                                # noqa: BLE001
        log.debug("StudentModel : profile fetch failed (%s)", exc)
        return None


async def _safe_fetch_context(student_id: str, course_id: Optional[str]):
    try:
        from pedagogy.personalization.engine import PersonalizationEngine
        return await PersonalizationEngine.get_context(student_id, course_id=course_id)
    except Exception as exc:                                                # noqa: BLE001
        log.debug("StudentModel : personalization context fetch failed (%s)", exc)
        return None


async def _safe_fetch_posterior(student_id: str):
    sid = _coerce_uuid(student_id)
    if sid is None:
        return None
    try:
        from pedagogy.personalization.learning_style.bayes import load_posterior
        return await load_posterior(sid)
    except Exception as exc:                                                # noqa: BLE001
        log.debug("StudentModel : posterior fetch failed (%s)", exc)
        return None


# ── Convenience class-method wrapper ────────────────────────────────────
# Lets consumers write ``StudentModel.fetch(...)`` instead of importing
# the standalone function — matches the rest of the codebase where
# data classes carry their own fetcher.

async def _fetch_classmethod(
    cls,
    student_id: str,
    course_id: Optional[str] = None,
    *,
    idea_ids: Optional[list[str]] = None,
) -> "StudentModel":
    return await fetch_student_model(student_id, course_id, idea_ids=idea_ids)


# Bind as a class-level fetch — keeps the dataclass frozen-friendly
# (assigning to .fetch on the class doesn't violate frozen=True).
StudentModel.fetch = classmethod(_fetch_classmethod)        # type: ignore[attr-defined]
