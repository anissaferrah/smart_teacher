"""
Smart Teacher — Student Profile Management Module.

Manages student learning profiles with adaptive personalization:
    - Learning style and preferences (visual, auditory, mixed)
    - Comprehension level (collège, lycée, université)
    - Speech rate adaptation based on confusion/difficulty
    - Topic mastery and confusion tracking
    - Automatic system prompt customization for LLM
    - Persistent storage in Redis with 30-day TTL

Usage:
    profile_mgr = ProfileManager()
    student = await profile_mgr.get_or_create("student_123", language="fr", level="lycée")
    
    # Record interaction
    student.record_confusion("k-means clustering")
    student.record_question()
    await profile_mgr.save(student)
    
    # Adapt LLM behavior
    system_prompt_extension = student.get_system_prompt_additions()
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional

import redis.asyncio as aioredis

log = logging.getLogger("SmartTeacher.Profile")

# ════════════════════════════════════════════════════════════════════════
# REDIS CONNECTION
# ════════════════════════════════════════════════════════════════════════

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """
    Get or create Redis connection (singleton pattern).
    
    Returns
    -------
    aioredis.Redis
        Connected Redis client
    """
    global _redis
    if _redis is None:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        _redis = aioredis.Redis(host=host, port=port, decode_responses=True)
    return _redis


# ════════════════════════════════════════════════════════════════════════
# STUDENT PROFILE DATACLASS
# ════════════════════════════════════════════════════════════════════════


@dataclass
class StudentProfile:
    """Student learning profile (Redis-backed, complement to the SQL row).

    Field names are aligned with ``database.models.StudentProfile`` so the
    same value can be cross-referenced between the two stores without a
    rename layer:
      - ``avg_response_time_s`` (was ``avg_response_time``)
      - ``preferred_explanation_depth`` (was ``detail_level``)
      - ``preferences`` JSON bag — same semantic as the SQL column
    Dead fields ``concept_mastery`` and ``recent_confusion_score`` (legacy
    in-RAM mastery store, replaced by ``mastery_repo`` Beta posterior)
    have been removed; old Redis blobs are accepted by ``from_json``
    which silently drops unknown keys.
    """

    student_id: str
    name: str = "Étudiant"
    language: str = "fr"
    level: str = "lycée"

    # Learning preferences (mirror the SQL column names)
    learning_style: str = "mixed"
    speech_rate: float = 1.0                            # TTS-only, Redis-only
    preferred_explanation_depth: str = "balanced"        # was `detail_level`

    # Session statistics
    total_sessions: int = 0
    total_questions: int = 0
    confusion_count: int = 0
    avg_response_time_s: float = 0.0                    # was `avg_response_time`

    # Topic tracking
    difficult_topics: List[str] = field(default_factory=list)
    mastered_topics: List[str] = field(default_factory=list)
    last_response_time: float = 0.0
    last_action: str = ""

    # Behavior tracking (auto-detected)
    asks_examples: int = 0
    asks_repeat: int = 0
    interruptions: int = 0

    # Free-form bag matching the SQL `preferences` column. Producers must
    # namespace their keys (vark.*, bayes.*, etc.) so they don't collide.
    preferences: Dict[str, object] = field(default_factory=dict)

    # Audit
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """
        Serialize profile to JSON string.
        
        Returns
        -------
        str
            JSON representation of profile
        """
        return json.dumps(asdict(self))

    def to_dict(self) -> dict:
        """Return the profile as a plain dictionary."""
        return asdict(self)

    # Renames applied during the alignment with the SQL column names.
    # Old Redis blobs use the LHS keys; we map them to the RHS dataclass
    # attributes during deserialization.
    _LEGACY_RENAMES = {
        "avg_response_time": "avg_response_time_s",
        "detail_level": "preferred_explanation_depth",
    }

    @classmethod
    def from_json(cls, data: str) -> "StudentProfile":
        """Deserialize profile from JSON, tolerating legacy field names.

        Old blobs (predating the SQL/Redis alignment) are accepted: legacy
        keys are renamed to their new names, and unknown keys (e.g. the
        retired ``concept_mastery`` and ``recent_confusion_score``) are
        silently dropped instead of crashing ``__init__``.
        """
        raw = json.loads(data)
        if not isinstance(raw, dict):
            raise ValueError("StudentProfile JSON must decode to a dict")
        # Rename legacy keys, then keep only fields the dataclass declares.
        for old, new in cls._LEGACY_RENAMES.items():
            if old in raw and new not in raw:
                raw[new] = raw.pop(old)
            elif old in raw:
                raw.pop(old)  # both present → trust the new key, discard old
        valid = {f.name for f in fields(cls)}
        cleaned = {k: v for k, v in raw.items() if k in valid}
        return cls(**cleaned)

    def record_confusion(self, topic: str = "") -> None:
        """Record student confusion on a topic.

        Updates the confusion counter and tags the topic as difficult.
        The previous version also wrote ``concept_mastery[topic] -= 0.15``
        and ``recent_confusion_score = 0.75`` — both magnitudes were
        hand-picked, so the in-RAM mastery delta has been removed in
        favour of the Bayesian posterior in ``mastery_repo``.
        """
        self.confusion_count += 1
        if topic and topic not in self.difficult_topics:
            self.difficult_topics.append(topic)
        self.updated_at = time.time()

    def record_mastery(self, topic: str) -> None:
        """Record student mastery of a topic.

        Moves the topic from ``difficult_topics`` to ``mastered_topics``.
        The previous version also wrote ``concept_mastery[topic] += 0.2``
        and decayed ``recent_confusion_score`` by 0.1 — both magnitudes
        were hand-picked, so they were removed.
        """
        if topic not in self.mastered_topics:
            self.mastered_topics.append(topic)
        if topic in self.difficult_topics:
            self.difficult_topics.remove(topic)
        self.updated_at = time.time()

    def record_interaction(self) -> None:
        """
        Record completion of one student-teacher interaction.
        
        Updates question counter and timestamp.
        """
        self.total_questions += 1
        self.updated_at = time.time()

    # NOTE: ``adapt_speech_rate`` was removed. The previous version slowed
    # the TTS output based on confusion counters with hand-picked thresholds
    # (recent_confusion_score >= 0.7, confusion_count > 5, asks_repeat > 3)
    # and slowdown deltas (-0.10, -0.05). None of the magnitudes had
    # empirical or theoretical backing. The user's stored ``speech_rate``
    # preference is now the single source of truth, consumed directly by
    # ``pedagogy.personalization.tts_adapter.compute_tts_params``.

    def get_system_prompt_additions(self) -> str:
        """
        Generate LLM system prompt customization based on profile.
        
        Creates additional instructions for the language model to
        personalize responses according to student characteristics.
        
        Returns
        -------
        str
            System prompt extension or empty string if not applicable
        
        Examples
        --------
        >>> profile = StudentProfile(student_id="s1", level="collège")
        >>> prompt = profile.get_system_prompt_additions()
        >>> # Returns: "Utilise un vocabulaire simple, évite le jargon technique."
        """
        parts: List[str] = []

        # Adapt to education level
        if self.level == "collège":
            parts.append("Utilise un vocabulaire simple, évite le jargon technique.")
        elif self.level == "université":
            parts.append("Tu peux utiliser un vocabulaire technique et approfondi.")

        # Adapt explanation depth (DB column name is the canonical one;
        # the Redis dataclass mirrors it under the same name).
        if self.preferred_explanation_depth == "concise":
            parts.append("Sois concis et va à l'essentiel.")
        elif self.preferred_explanation_depth == "detailed":
            parts.append("Donne des explications détaillées avec étapes.")

        # Emphasize difficult topics
        if self.difficult_topics:
            topics_str = ", ".join(self.difficult_topics[-3:])
            parts.append(
                f"L'étudiant a des difficultés avec : {topics_str}. "
                f"Sois particulièrement clair sur ces sujets."
            )

        # Note learning preference. ``> 0`` is intentional: the previous
        # threshold of ``> 2`` was a hand-picked cutoff for "the student
        # really likes examples" without any empirical basis. Treating the
        # counter as a binary "has the student ever asked for examples?"
        # signal removes the arbitrary magnitude.
        if self.asks_examples > 0:
            parts.append(
                "Cet étudiant apprécie les exemples concrets. Inclus-en systématiquement."
            )

        return " ".join(parts)


# ════════════════════════════════════════════════════════════════════════
# PROFILE MANAGER
# ════════════════════════════════════════════════════════════════════════


class ProfileManager:
    """
    Manages persistent student profiles in Redis.
    
    Provides CRUD operations with automatic TTL management for temporary
    profile data storage during active learning sessions.
    
    Attributes
    ----------
    PROFILE_TTL : int
        Redis key expiration time in seconds (30 days)
    """

    PROFILE_TTL: int = 86400 * 30  # 30 days in seconds

    # Profiles are keyed per (student, course). The same student can have
    # very different pace / confusion / preferred depth across subjects
    # (e.g. fast on Python, slow on linguistics), so a single global
    # profile would average those out and degrade adaptation. ``_global``
    # is the cross-course default bucket — used for the legacy callers
    # that don't yet thread a course through, and as a sane fallback the
    # very first time a course is opened.
    _NO_COURSE = "_global"

    @staticmethod
    def _course_key(course_id: str | None) -> str:
        cid = (course_id or "").strip()
        return cid if cid else ProfileManager._NO_COURSE

    @classmethod
    def _redis_key(cls, student_id: str, course_id: str | None) -> str:
        return f"profile:{student_id}:{cls._course_key(course_id)}"

    async def get_or_create(
        self,
        student_id: str,
        language: str = "fr",
        level: str = "lycée",
        course_id: str | None = None,
    ) -> StudentProfile:
        """
        Retrieve existing profile or create new one for ``(student, course)``.

        Attempts to load from Redis; creates fresh profile if not found.
        When the per-course bucket is empty, the per-student ``_global``
        bucket is used as a seed so the student's cross-course preferences
        (VARK self-report, broad pace, etc.) carry over to a new course
        instead of being discarded.
        """
        r = await get_redis()
        key = self._redis_key(student_id, course_id)

        try:
            data = await r.get(key)
            if data:
                log.debug(f"Loaded profile from Redis: {student_id}/{self._course_key(course_id)}")
                return StudentProfile.from_json(data)
        except Exception as exc:
            log.warning(f"Failed to load profile from Redis: {exc}")

        # Seed from the cross-course default if it exists, so newly opened
        # courses inherit the student's broad preferences instead of starting
        # from zero. Only happens when course_id is real (not _global itself).
        seed: StudentProfile | None = None
        if self._course_key(course_id) != self._NO_COURSE:
            try:
                global_data = await r.get(self._redis_key(student_id, None))
                if global_data:
                    seed = StudentProfile.from_json(global_data)
            except Exception:
                seed = None

        if seed is not None:
            # Fresh per-course copy: keep static prefs (style, level, language)
            # but reset session-level counters so per-course signals start clean.
            profile = StudentProfile(
                student_id=student_id,
                name=seed.name,
                language=seed.language,
                level=seed.level,
                learning_style=seed.learning_style,
                speech_rate=seed.speech_rate,
                preferred_explanation_depth=seed.preferred_explanation_depth,
            )
            log.info(
                "Created course-scoped profile %s/%s seeded from global",
                student_id, self._course_key(course_id),
            )
        else:
            profile = StudentProfile(student_id=student_id, language=language, level=level)
            log.info(
                "Created new profile %s/%s (lang=%s, level=%s)",
                student_id, self._course_key(course_id), language, level,
            )
        await self.save(profile, course_id=course_id)
        return profile

    async def save(self, profile: StudentProfile, course_id: str | None = None) -> None:
        """Persist profile to the ``(student, course)`` bucket with TTL."""
        try:
            r = await get_redis()
            await r.setex(
                self._redis_key(profile.student_id, course_id),
                self.PROFILE_TTL,
                profile.to_json()
            )
            log.debug(
                "Saved profile to Redis: %s/%s",
                profile.student_id, self._course_key(course_id),
            )
        except Exception as exc:
            log.warning(f"Failed to save profile to Redis: {exc}")

    async def update_from_interaction(
        self,
        student_id: str,
        interaction_type: str,
        topic: str = "",
        confused: bool = False,
        response_time: float | None = None,
        confidence: float | None = None,
        reward: float | None = None,
        action_taken: str = "",
        course_id: str | None = None,
    ) -> Optional[StudentProfile]:
        """
        Update profile based on student interaction type.
        
        Automatically tracks student behavior patterns to adjust
        personalization and adaptive content.
        
        Parameters
        ----------
        student_id : str
            Student identifier
        interaction_type : str
            Type of interaction: "repeat", "give_example", "explain_again", 
            "interrupt", or normal "question"
        topic : str, optional
            Topic related to interaction
        
        Returns
        -------
        StudentProfile or None
            Updated profile, or None if save fails
        """
        profile = await self.get_or_create(student_id, course_id=course_id)
        profile.last_action = action_taken or interaction_type
        
        if interaction_type == "repeat":
            profile.asks_repeat += 1
        elif interaction_type == "give_example":
            profile.asks_examples += 1
        elif interaction_type == "explain_again":
            profile.record_confusion(topic)
        elif interaction_type == "interrupt":
            profile.interruptions += 1

        if response_time is not None:
            response_time = max(0.0, float(response_time))
            profile.last_response_time = response_time
            # True running mean over total_questions (which has not been
            # incremented yet — record_interaction does that below).
            # The previous version used EWMA with alpha = 0.15 (effective
            # window ~7 samples), but the smoothing constant was
            # hand-picked. A uniform-weight running mean has no tuning
            # knob and answers the literal question "average response
            # time so far".
            n = profile.total_questions
            if n <= 0:
                profile.avg_response_time_s = round(response_time, 3)
            else:
                profile.avg_response_time_s = round(
                    (profile.avg_response_time_s * n + response_time) / (n + 1), 3
                )

        # 0.5 is the Bayes-optimal decision boundary for a calibrated
        # binary classifier under equal prior and equal misclassification
        # cost — i.e. "more likely confused than not". This is a
        # theoretically grounded threshold, not a tuning knob.
        low_confidence = confidence is not None and float(confidence) < 0.5
        if confused or low_confidence:
            profile.record_confusion(topic)
        elif topic and reward is not None and float(reward) > 0:
            # Positive reward = mastery signal. The previous version
            # graded ``boost = 0.15 if reward > 0 else 0.05`` and lifted
            # mastery once the in-RAM dict crossed 0.85; both have been
            # removed because the magnitudes were hand-picked and the
            # canonical mastery store is ``mastery_repo`` (Beta posterior).
            profile.record_mastery(topic)

        profile.record_interaction()

        await self.save(profile, course_id=course_id)
        return profile

    async def update_from_session(
        self,
        student_id: str,
        interaction_type: str,
        topic: str = "",
        confused: bool = False,
        response_time: float | None = None,
        confidence: float | None = None,
        reward: float | None = None,
        action_taken: str = "",
        course_id: str | None = None,
    ) -> Optional[StudentProfile]:
        """Backward-compatible alias used by older call sites."""
        return await self.update_from_interaction(
            student_id=student_id,
            interaction_type=interaction_type,
            topic=topic,
            confused=confused,
            response_time=response_time,
            confidence=confidence,
            reward=reward,
            action_taken=action_taken,
            course_id=course_id,
        )


# ════════════════════════════════════════════════════════════════════════
# Module-level convenience helpers (singleton ProfileManager)
# Previously located in services/personalization/profile_manager.py
# ════════════════════════════════════════════════════════════════════════

_PM: Optional["ProfileManager"] = None


def _get_pm() -> "ProfileManager":
    global _PM
    if _PM is None:
        _PM = ProfileManager()
    return _PM


async def get_or_create_profile(
    session_id: str,
    defaults: Dict[str, object] | None = None,
    course_id: str | None = None,
) -> Dict[str, object]:
    """Return a profile dict for ``(session_id, course_id)`` (Redis-backed).

    ``course_id`` may be left None for legacy callers — they fall back to
    the cross-course ``_global`` bucket.
    """
    pm = _get_pm()
    profile = await pm.get_or_create(
        session_id,
        language=(defaults or {}).get("language", "fr"),
        level=(defaults or {}).get("level", "lycée"),
        course_id=course_id,
    )
    return profile.to_dict() if isinstance(profile, StudentProfile) else profile


async def update_profile(
    session_id: str,
    patch: Dict[str, object],
    course_id: str | None = None,
) -> Dict[str, object]:
    """Patch profile fields and persist for ``(session_id, course_id)``."""
    pm = _get_pm()
    try:
        profile = await pm.get_or_create(session_id, course_id=course_id)
        for k, v in (patch or {}).items():
            if hasattr(profile, k):
                setattr(profile, k, v)
        await pm.save(profile, course_id=course_id)
        return profile.to_dict()
    except Exception as exc:
        log.warning("Failed to update profile %s: %s", session_id, exc)
        return {}
