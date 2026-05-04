"""Phase 1 — Contextual Thompson sampling bandit for pedagogical strategy selection.

Quick start :

    from pedagogy.personalization.bandit import (
        ContextualThompsonBandit, context_from_profile,
        compute_reward, TurnOutcome,
    )

    bandit = ContextualThompsonBandit()

    # 1. Build context from the student's profile
    context = context_from_profile(
        learning_style="visual",
        avg_response_time_s=22.0,
        mastery_score=0.4,
    )

    # 2. Pick an action
    action = bandit.select(context)
    # action.strategy ∈ {analogy, example, decomposition, socratic, recap, simpler_words}
    # action.speech_rate ∈ {slow, normal, fast}

    # 3. Apply it (inject prompt fragment, set TTS rate, etc.)
    # ...

    # 4. Observe outcome and compute reward
    reward = compute_reward(TurnOutcome(
        confusion_detected=False,
        mastery_before=0.40,
        mastery_after=0.55,
        engaged=True,
    ))

    # 5. Update the bandit
    bandit.update(context, action, reward)

# Reading order

  - ``strategies.py`` : action space (Strategy × SpeechRate = 18 arms)
  - ``thompson.py``   : Beta posterior + selection + update
  - ``reward.py``     : pedagogical objective (confusion + mastery + engagement)

# References

  - Russo et al. 2018, *A Tutorial on Thompson Sampling*
  - Agrawal & Goyal 2013, *Further Optimal Regret Bounds for Thompson Sampling*
  - Bloom 1968, *Learning for Mastery*
  - Sweller 1985, *Cognitive Load Theory*
"""
from pedagogy.personalization.bandit.controller import (
    BanditController,
    TurnDecision,
)
from pedagogy.personalization.bandit.reward import (
    TurnOutcome,
    W_CONFUSION,
    W_ENGAGEMENT,
    W_MASTERY,
    compute_reward,
)
from pedagogy.personalization.bandit.strategies import (
    SpeechRate,
    Strategy,
    StrategyAction,
    all_actions,
    strategy_prompt,
)
from pedagogy.personalization.bandit.thompson import (
    ArmPosterior,
    ContextBucket,
    ContextualThompsonBandit,
    context_from_profile,
    discretize_mastery,
    discretize_response_time,
)

from pedagogy.personalization.bandit.offline import (
    EpisodeDataset,
    LoggedEpisode,
    ips_clipped_off_policy_value,
    ips_off_policy_value,
    policy_from_bandit,
    train_offline,
    train_test_split,
)
from pedagogy.personalization.bandit.log_extractor import (
    event_to_episode,
    extra_payload_from_state,
    extract_episodes_from_postgres,
    extract_to_jsonl,
)
from pedagogy.personalization.bandit.simulator import (
    Episode,
    SimulatedStudent,
    bootstrap_bandit,
    generate_episodes,
    simulate_episode,
)

__all__ = [
    # strategies
    "Strategy", "SpeechRate", "StrategyAction", "all_actions", "strategy_prompt",
    # thompson
    "ArmPosterior", "ContextBucket", "ContextualThompsonBandit",
    "context_from_profile", "discretize_mastery", "discretize_response_time",
    # reward
    "TurnOutcome", "compute_reward",
    "W_CONFUSION", "W_MASTERY", "W_ENGAGEMENT",
    # controller
    "BanditController", "TurnDecision",
    # simulator
    "SimulatedStudent", "Episode", "simulate_episode", "generate_episodes",
    "bootstrap_bandit",
    # offline / Phase 3
    "LoggedEpisode", "EpisodeDataset",
    "train_offline", "train_test_split",
    "ips_off_policy_value", "ips_clipped_off_policy_value",
    "policy_from_bandit",
    # log extractor (Postgres → LoggedEpisode)
    "event_to_episode", "extra_payload_from_state",
    "extract_episodes_from_postgres", "extract_to_jsonl",
]
