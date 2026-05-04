"""Learning-style sub-package.

Three layers:
  - heuristic : v1 fallback (StyleScores, compute_learning_style) — used at
                cold start before the Bayesian model has enough signal
  - bayes     : M2 Dirichlet-Multinomial model (PosteriorEstimate, fire_signal,
                update_posterior, cross_validate)
  - vark      : 8-item psychometric questionnaire used to seed the Bayes prior

The infrastructure cache (in-process LRU + Redis) lives at
``services.learning_style_cache`` because it's cross-cutting infrastructure,
not learning-style domain logic.

# Note on style → narrator parameters

A previous version exposed ``style_to_params(style) → {max_sentences,
must_include, structure, concreteness}`` which mapped each cognitive style
to fixed numeric directives (e.g. visual → 5 sentences, analogy first).
That mapping was made up — the numbers were not derived from any
empirical study or theoretical model — so it was removed. The narrator
now receives only the prose hint via ``style_to_prompt_hint``; if we
introduce structured directives again, they must be backed by data
(A/B test sweep across configurations).
"""
from pedagogy.personalization.learning_style.heuristic import (
    StyleScores,
    compute_learning_style,
    update_student_learning_style,
    style_to_prompt_hint,
)
from pedagogy.personalization.learning_style.bayes import (
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
)
from pedagogy.personalization.learning_style.vark import (
    VARKQuestion,
    VARK_QUESTIONS,
    score_responses,
    serialize_for_frontend,
)

__all__ = [
    # heuristic v1
    "StyleScores", "compute_learning_style", "update_student_learning_style",
    "style_to_prompt_hint",
    # bayes M2
    "DEFAULT_PRIOR_ALPHA", "PosteriorEstimate", "SIGNAL_WEIGHTS", "STYLES",
    "CrossValidationResult", "cross_validate", "fire_signal", "load_posterior",
    "rebuild_posterior_from_history", "save_posterior", "update_posterior",
    # vark
    "VARKQuestion", "VARK_QUESTIONS", "score_responses", "serialize_for_frontend",
]
