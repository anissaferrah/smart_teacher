"""Personalization package — single source of truth for student adaptation.

# Layout

    pedagogy/personalization/
    ├── learning_style/      ← cognitive style modeling (V/A/R/K)
    │   ├── heuristic.py     ← v1 fallback (cold start)
    │   ├── bayes.py         ← M2 Dirichlet-Multinomial model
    │   ├── vark.py          ← psychometric questionnaire
    │   └── params.py        ← style → narrator directives
    ├── profile.py           ← StudentProfile (in-RAM model)
    ├── engine.py            ← PersonalizationEngine (compose prompt prefix)
    └── tts_adapter.py       ← TTS rate adaptation per style/confusion

The 2-level cache (Redis + in-process LRU) for resolved hints/params lives
in ``services.learning_style_cache`` because it's cross-cutting cache
infrastructure, not personalization domain logic.

# Public API

Common imports for callers:

    from pedagogy.personalization import (
        # Bayes M2
        load_posterior, fire_signal, update_posterior, cross_validate,
        # Heuristic v1 (fallback)
        compute_learning_style, style_to_prompt_hint,
        # VARK
        score_responses, serialize_for_frontend,
        # Params
        style_to_params,
        # Engine + Profile
        PersonalizationEngine, get_or_create_profile,
    )
"""
# Sub-package re-exports (most-used symbols)
from pedagogy.personalization.learning_style import (
    # Heuristic v1
    StyleScores,
    compute_learning_style,
    update_student_learning_style,
    style_to_prompt_hint,
    # Bayes M2
    DEFAULT_PRIOR_ALPHA,
    PosteriorEstimate,
    SIGNAL_WEIGHTS,
    STYLES,
    CrossValidationResult,
    cross_validate,
    fire_signal,
    load_posterior,
    rebuild_posterior_from_history,
    save_posterior,
    update_posterior,
    # VARK
    VARKQuestion,
    VARK_QUESTIONS,
    score_responses,
    serialize_for_frontend,
)
from pedagogy.personalization.engine import (
    PersonalizationContext,
    PersonalizationEngine,
)

__all__ = [
    # Learning style — heuristic
    "StyleScores", "compute_learning_style", "update_student_learning_style",
    "style_to_prompt_hint",
    # Learning style — bayes
    "DEFAULT_PRIOR_ALPHA", "PosteriorEstimate", "SIGNAL_WEIGHTS", "STYLES",
    "CrossValidationResult", "cross_validate", "fire_signal", "load_posterior",
    "rebuild_posterior_from_history", "save_posterior", "update_posterior",
    # VARK
    "VARKQuestion", "VARK_QUESTIONS", "score_responses", "serialize_for_frontend",
    # Engine
    "PersonalizationContext", "PersonalizationEngine",
]
