"""Tests for the personalization bandit (Phase 1)."""
from __future__ import annotations

import pytest

from pedagogy.personalization.bandit import (
    ArmPosterior,
    ContextBucket,
    ContextualThompsonBandit,
    SpeechRate,
    Strategy,
    StrategyAction,
    TurnOutcome,
    all_actions,
    compute_reward,
    context_from_profile,
    discretize_mastery,
    discretize_response_time,
    strategy_prompt,
)


# ════════════════════════════════════════════════════════════════════
# Strategies (action space)
# ════════════════════════════════════════════════════════════════════

class TestStrategies:
    def test_action_space_size(self):
        """6 strategies × 3 speech rates = 18 arms."""
        actions = list(all_actions())
        assert len(actions) == 18

    def test_arm_id_is_stable(self):
        a = StrategyAction(Strategy.ANALOGY, SpeechRate.SLOW)
        assert a.arm_id == "analogy:slow"

    def test_arm_id_roundtrip(self):
        a = StrategyAction(Strategy.SOCRATIC, SpeechRate.FAST)
        b = StrategyAction.from_arm_id(a.arm_id)
        assert a == b

    def test_action_space_unique_arm_ids(self):
        ids = {a.arm_id for a in all_actions()}
        assert len(ids) == 18  # no collisions

    def test_strategy_prompt_fr_and_en(self):
        fr = strategy_prompt(Strategy.ANALOGY, "fr")
        en = strategy_prompt(Strategy.ANALOGY, "en")
        # Different phrasings, both non-empty, both reference "analogy"
        assert fr != en
        assert "ANALOGIE" in fr
        assert "ANALOGY" in en

    def test_strategy_prompt_unknown_lang_falls_back_to_en(self):
        out = strategy_prompt(Strategy.EXAMPLE, "zz")
        assert out  # non-empty
        # Default fallback returns the EN entry (not "zz" entry)
        assert "EXAMPLE" in out


# ════════════════════════════════════════════════════════════════════
# Context discretization
# ════════════════════════════════════════════════════════════════════

class TestContextDiscretization:
    def test_mastery_buckets(self):
        assert discretize_mastery(0.0) == "low"
        assert discretize_mastery(0.39) == "low"
        assert discretize_mastery(0.40) == "medium"
        assert discretize_mastery(0.84) == "medium"
        assert discretize_mastery(0.85) == "high"
        assert discretize_mastery(1.0) == "high"

    def test_response_time_buckets(self):
        # Boundaries follow the existing PersonalizationEngine logic
        assert discretize_response_time(5.0) == "fast"
        assert discretize_response_time(14.9) == "fast"
        assert discretize_response_time(15.0) == "normal"
        assert discretize_response_time(60.0) == "normal"
        assert discretize_response_time(60.1) == "slow"

    def test_context_from_profile(self):
        ctx = context_from_profile(
            learning_style="visual",
            avg_response_time_s=22.0,
            mastery_score=0.5,
        )
        assert ctx.learning_style == "visual"
        assert ctx.pace == "normal"
        assert ctx.mastery_level == "medium"

    def test_bucket_key_uniqueness(self):
        a = ContextBucket("visual", "slow", "low")
        b = ContextBucket("visual", "slow", "high")
        assert a.bucket_key != b.bucket_key


# ════════════════════════════════════════════════════════════════════
# ArmPosterior (Beta(α, β) updates)
# ════════════════════════════════════════════════════════════════════

class TestArmPosterior:
    def test_default_prior_is_uniform_beta_1_1(self):
        p = ArmPosterior()
        assert p.alpha == 1.0
        assert p.beta == 1.0
        assert p.mean == 0.5
        assert p.n_pulls == 0

    def test_update_with_reward_one_increments_alpha(self):
        p = ArmPosterior()
        p.update(1.0)
        assert p.alpha == 2.0
        assert p.beta == 1.0
        assert p.n_pulls == 1
        # Mean now skewed positive
        assert p.mean > 0.5

    def test_update_with_reward_zero_increments_beta(self):
        p = ArmPosterior()
        p.update(0.0)
        assert p.alpha == 1.0
        assert p.beta == 2.0
        assert p.mean < 0.5

    def test_update_with_continuous_reward(self):
        p = ArmPosterior()
        p.update(0.7)  # fractional Bernoulli observation
        assert p.alpha == pytest.approx(1.7)
        assert p.beta == pytest.approx(1.3)

    def test_update_clips_out_of_range_reward(self):
        p = ArmPosterior()
        p.update(1.5)  # clipped to 1.0
        assert p.alpha == pytest.approx(2.0)
        p.update(-0.5)  # clipped to 0.0
        assert p.beta == pytest.approx(2.0)

    def test_sample_in_unit_interval(self):
        p = ArmPosterior(alpha=5.0, beta=3.0)
        for _ in range(100):
            assert 0.0 <= p.sample() <= 1.0


# ════════════════════════════════════════════════════════════════════
# ContextualThompsonBandit
# ════════════════════════════════════════════════════════════════════

class TestContextualThompsonBandit:
    def test_select_returns_valid_action(self):
        bandit = ContextualThompsonBandit(rng_seed=42)
        ctx = ContextBucket("visual", "normal", "medium")
        action = bandit.select(ctx)
        # Must be one of the 18 arms
        assert action in list(all_actions())

    def test_select_creates_posteriors_lazily(self):
        bandit = ContextualThompsonBandit(rng_seed=42)
        # Initially, no posteriors stored
        assert len(bandit.posteriors) == 0
        ctx = ContextBucket("visual", "normal", "medium")
        bandit.select(ctx)
        # After selection, every candidate arm got a posterior
        assert len(bandit.posteriors) == 18

    def test_update_mutates_correct_posterior(self):
        bandit = ContextualThompsonBandit(rng_seed=42)
        ctx = ContextBucket("visual", "normal", "medium")
        action = StrategyAction(Strategy.ANALOGY, SpeechRate.NORMAL)
        bandit.update(ctx, action, reward=1.0)
        post = bandit.posteriors[bandit._key(ctx, action)]
        assert post.alpha == 2.0
        assert post.beta == 1.0
        # Other arms untouched
        other = StrategyAction(Strategy.EXAMPLE, SpeechRate.NORMAL)
        if bandit._key(ctx, other) in bandit.posteriors:
            other_post = bandit.posteriors[bandit._key(ctx, other)]
            assert other_post.alpha == 1.0
            assert other_post.beta == 1.0

    def test_bandit_converges_to_best_arm(self):
        """If one arm consistently rewards 1.0 and others 0.0, the bandit
        should learn to pick the good arm."""
        bandit = ContextualThompsonBandit(rng_seed=42)
        ctx = ContextBucket("visual", "normal", "medium")
        good_arm = StrategyAction(Strategy.ANALOGY, SpeechRate.NORMAL)

        # Train: 200 turns, reward=1 if action == good_arm else 0
        for _ in range(200):
            chosen = bandit.select(ctx)
            r = 1.0 if chosen == good_arm else 0.0
            bandit.update(ctx, chosen, r)

        # The best arm by posterior mean should be the good one
        assert bandit.best_arm_for(ctx) == good_arm

    def test_isolated_contexts_dont_leak(self):
        """Two different contexts should maintain independent posteriors."""
        bandit = ContextualThompsonBandit(rng_seed=42)
        ctx_visual = ContextBucket("visual", "normal", "medium")
        ctx_audio  = ContextBucket("auditory", "normal", "medium")
        action = StrategyAction(Strategy.ANALOGY, SpeechRate.NORMAL)
        # Update visual context
        bandit.update(ctx_visual, action, reward=1.0)
        # Audio context posterior should still be at the prior
        post_audio = bandit._post(ctx_audio, action)
        assert post_audio.alpha == 1.0
        assert post_audio.beta == 1.0

    def test_serialization_roundtrip(self):
        bandit = ContextualThompsonBandit(rng_seed=42)
        ctx = ContextBucket("visual", "normal", "medium")
        action = StrategyAction(Strategy.ANALOGY, SpeechRate.NORMAL)
        bandit.update(ctx, action, reward=1.0)
        bandit.update(ctx, action, reward=0.5)

        data = bandit.to_dict()
        restored = ContextualThompsonBandit.from_dict(data)

        # Posteriors preserved
        original_post = bandit._post(ctx, action)
        restored_post = restored._post(ctx, action)
        assert original_post.alpha == restored_post.alpha
        assert original_post.beta == restored_post.beta


# ════════════════════════════════════════════════════════════════════
# Reward function
# ════════════════════════════════════════════════════════════════════

class TestReward:
    def test_perfect_outcome_yields_max_reward(self):
        outcome = TurnOutcome(
            confusion_detected=False,
            mastery_before=0.4, mastery_after=0.7,  # +0.30 (saturated)
            engaged=True,
        )
        # confusion=1.0 × 0.5 + mastery=1.0 × 0.4 + engagement=1.0 × 0.1 = 1.0
        assert compute_reward(outcome) == pytest.approx(1.0)

    def test_worst_outcome_yields_zero_reward(self):
        outcome = TurnOutcome(
            confusion_detected=True,
            mastery_before=0.5, mastery_after=0.5,
            engaged=False,
        )
        # confusion=0.0 + mastery=0.0 + engagement=0.0 = 0.0
        assert compute_reward(outcome) == 0.0

    def test_confusion_dominates(self):
        """Confusion alone (no mastery, no engagement) still gives 0.5."""
        outcome = TurnOutcome(
            confusion_detected=False,
            mastery_before=0.5, mastery_after=0.5,
            engaged=False,
        )
        # confusion=1.0 × 0.5 = 0.5
        assert compute_reward(outcome) == pytest.approx(0.5)

    def test_negative_mastery_delta_clamped_to_zero(self):
        """A drop in mastery (rare but possible) doesn't make the
        reward negative — it's just zero on the mastery axis."""
        outcome = TurnOutcome(
            confusion_detected=False,
            mastery_before=0.6, mastery_after=0.4,  # -0.20 → clamped to 0
            engaged=True,
        )
        # confusion=1.0 × 0.5 + mastery=0.0 + engagement=1.0 × 0.1 = 0.6
        assert compute_reward(outcome) == pytest.approx(0.6)

    def test_reward_in_unit_interval(self):
        """All reward outputs must be in [0, 1] for the Beta posterior."""
        outcomes = [
            TurnOutcome(False, 0.0, 0.0, False),
            TurnOutcome(True, 0.0, 1.0, True),
            TurnOutcome(False, 0.5, 0.55, True),
        ]
        for o in outcomes:
            r = compute_reward(o)
            assert 0.0 <= r <= 1.0


# ════════════════════════════════════════════════════════════════════
# BanditController (start_turn / end_turn lifecycle)
# ════════════════════════════════════════════════════════════════════

import asyncio
from unittest.mock import patch, AsyncMock

from pedagogy.personalization.bandit import BanditController, TurnDecision
from pedagogy.personalization.bandit.repo import (
    PendingDecision,
    _reset_for_tests,
)


class TestBanditController:
    def setup_method(self) -> None:
        # Each test starts with a fresh in-process bandit instance
        _reset_for_tests()

    def _patch_redis_calls(self, pending_storage: dict | None = None):
        """Mock the Redis-touching helpers in repo.py with in-memory dict."""
        store = pending_storage if pending_storage is not None else {}

        async def fake_record_pending(session_id, decision):
            store[session_id] = decision

        async def fake_consume_pending(session_id):
            return store.pop(session_id, None)

        async def fake_load_state():
            return None  # Always start fresh

        async def fake_save_state():
            return None  # No-op

        return patch.multiple(
            "pedagogy.personalization.bandit.repo",
            record_pending=AsyncMock(side_effect=fake_record_pending),
            consume_pending=AsyncMock(side_effect=fake_consume_pending),
            _load_state=AsyncMock(side_effect=fake_load_state),
            save_state=AsyncMock(side_effect=fake_save_state),
        ), patch.multiple(
            "pedagogy.personalization.bandit.controller",
            record_pending=AsyncMock(side_effect=fake_record_pending),
            consume_pending=AsyncMock(side_effect=fake_consume_pending),
            maybe_save_periodically=AsyncMock(),
        )

    def test_start_turn_returns_valid_decision(self):
        store: dict = {}
        repo_patch, ctrl_patch = self._patch_redis_calls(store)
        with repo_patch, ctrl_patch:
            ctrl = BanditController()
            decision = asyncio.run(ctrl.start_turn(
                session_id="s1",
                learning_style="visual",
                avg_response_time_s=22.0,
                mastery_score=0.5,
                primary_concept="concept_knn",
                language="fr",
            ))
        assert isinstance(decision, TurnDecision)
        assert decision.ok is True
        assert decision.action in list(all_actions())
        # Prompt fragment must be a non-empty FR string
        assert len(decision.prompt_fragment) > 10

    def test_start_turn_records_pending_decision(self):
        store: dict = {}
        repo_patch, ctrl_patch = self._patch_redis_calls(store)
        with repo_patch, ctrl_patch:
            ctrl = BanditController()
            asyncio.run(ctrl.start_turn(
                session_id="s1",
                learning_style="auditory",
                avg_response_time_s=18.0,
                mastery_score=0.6,
                primary_concept="concept_recursion",
                language="fr",
            ))
            # Pending decision was stashed
            assert "s1" in store
            pending = store["s1"]
            assert pending.context.learning_style == "auditory"
            assert pending.primary_concept == "concept_recursion"
            assert pending.mastery_before == 0.6

    def test_end_turn_consumes_pending_and_updates_bandit(self):
        store: dict = {}
        repo_patch, ctrl_patch = self._patch_redis_calls(store)
        with repo_patch, ctrl_patch:
            ctrl = BanditController()
            # 1. Start a turn
            decision = asyncio.run(ctrl.start_turn(
                session_id="s1",
                learning_style="visual",
                avg_response_time_s=22.0,
                mastery_score=0.5,
                primary_concept="concept_knn",
                language="fr",
            ))
            arm_id_chosen = decision.action.arm_id
            # Verify pending exists
            assert "s1" in store
            # 2. End the turn — successful outcome
            updated = asyncio.run(ctrl.end_turn(
                session_id="s1",
                confusion_detected=False,
                mastery_after=0.7,
                engaged=True,
            ))
            assert updated is True
            # Pending was consumed
            assert "s1" not in store
            # Bandit posterior was updated for the chosen arm
            from pedagogy.personalization.bandit.repo import _bandit_instance
            assert _bandit_instance is not None
            # Find the posterior for the arm we chose
            arm_keys = [k for k in _bandit_instance.posteriors if k.endswith(f"|{arm_id_chosen}")]
            assert len(arm_keys) >= 1
            post = _bandit_instance.posteriors[arm_keys[0]]
            # alpha should have grown (positive reward)
            assert post.alpha > 1.0

    def test_end_turn_no_pending_returns_false(self):
        """If no pending decision exists (first turn of a session), end_turn
        returns False without raising."""
        store: dict = {}
        repo_patch, ctrl_patch = self._patch_redis_calls(store)
        with repo_patch, ctrl_patch:
            ctrl = BanditController()
            updated = asyncio.run(ctrl.end_turn(
                session_id="never_started",
                confusion_detected=False,
                mastery_after=0.5,
            ))
            assert updated is False

    def test_full_episode_converges(self):
        """Multi-turn: start → end → start → end... With consistent positive
        rewards on one arm, the bandit's preferred arm for that bucket should
        converge."""
        store: dict = {}
        repo_patch, ctrl_patch = self._patch_redis_calls(store)
        with repo_patch, ctrl_patch:
            ctrl = BanditController()
            # 30 turns for a "visual / normal pace / medium mastery" student
            for i in range(30):
                decision = asyncio.run(ctrl.start_turn(
                    session_id=f"s_{i}",  # different session each turn — pending is per-session
                    learning_style="visual",
                    avg_response_time_s=22.0,
                    mastery_score=0.5,
                    language="fr",
                ))
                # Reward = 1 if action was analogy*, else 0 (synthetic environment)
                got_analogy = decision.action.strategy.value == "analogy"
                asyncio.run(ctrl.end_turn(
                    session_id=f"s_{i}",
                    confusion_detected=not got_analogy,
                    mastery_after=0.7 if got_analogy else 0.5,
                    engaged=True,
                ))
            # After training, bandit's best arm for the bucket should
            # have strategy=analogy
            from pedagogy.personalization.bandit.repo import _bandit_instance
            from pedagogy.personalization.bandit import ContextBucket
            bucket = ContextBucket("visual", "normal", "medium")
            best = _bandit_instance.best_arm_for(bucket)
            assert best is not None
            assert best.strategy.value == "analogy"


# ════════════════════════════════════════════════════════════════════
# TTS adapter — bandit speech_rate plumbing
# ════════════════════════════════════════════════════════════════════

class TestTTSBanditPlumbing:
    def test_bandit_rate_multiplier_known_categories(self):
        from pedagogy.personalization.tts_adapter import bandit_rate_multiplier
        # slow → 0.85, normal → 1.0, fast → 1.15
        assert bandit_rate_multiplier("slow") < 1.0
        assert bandit_rate_multiplier("normal") == 1.0
        assert bandit_rate_multiplier("fast") > 1.0

    def test_bandit_rate_multiplier_unknown_falls_back_to_one(self):
        from pedagogy.personalization.tts_adapter import bandit_rate_multiplier
        assert bandit_rate_multiplier(None) == 1.0
        assert bandit_rate_multiplier("") == 1.0
        assert bandit_rate_multiplier("weird") == 1.0

    def test_rate_float_to_edge_str_clamped(self):
        from pedagogy.personalization.tts_adapter import rate_float_to_edge_str
        # +0% on rate=1.0
        assert rate_float_to_edge_str(1.0) == "+0%"
        # 0.85 → -15%
        assert rate_float_to_edge_str(0.85) == "-15%"
        # Clamping at -50%, +50%
        assert rate_float_to_edge_str(0.1) == "-50%"
        assert rate_float_to_edge_str(2.0) == "+50%"


# ════════════════════════════════════════════════════════════════════
# Simulator (Phase 2)
# ════════════════════════════════════════════════════════════════════

class TestSimulator:
    def test_simulated_student_random_creates_valid_bucket(self):
        from pedagogy.personalization.bandit import SimulatedStudent
        import random as _random
        rng = _random.Random(0)
        s = SimulatedStudent.random(rng=rng)
        assert s.bucket.learning_style in {"visual", "auditory", "kinesthetic", "reading", "mixed"}
        assert s.bucket.pace in {"slow", "normal", "fast"}
        assert s.bucket.mastery_level in {"low", "medium", "high"}

    def test_preferences_normalized(self):
        from pedagogy.personalization.bandit import SimulatedStudent
        import random as _random
        s = SimulatedStudent.random(rng=_random.Random(0))
        assert s.preference  # non-empty
        # Sum to 1 (normalized distribution)
        total = sum(s.preference.values())
        assert abs(total - 1.0) < 1e-6

    def test_visual_student_biases_toward_analogy(self):
        """Visual students should have higher preference for ANALOGY-type
        arms than for SOCRATIC arms (a heuristic the simulator encodes)."""
        from pedagogy.personalization.bandit import (
            SimulatedStudent, Strategy, ContextBucket,
        )
        from pedagogy.personalization.bandit.simulator import _build_preference
        import random as _random
        # Build many visual students and average preferences
        rng = _random.Random(0)
        bucket = ContextBucket("visual", "normal", "medium")
        analogy_w = 0.0
        socratic_w = 0.0
        for _ in range(50):
            pref = _build_preference(bucket, rng)
            for action, w in pref.items():
                if action.strategy == Strategy.ANALOGY:
                    analogy_w += w
                if action.strategy == Strategy.SOCRATIC:
                    socratic_w += w
        # On average, analogy should beat socratic for visual learners
        assert analogy_w > socratic_w

    def test_bootstrap_bandit_populates_posteriors(self):
        from pedagogy.personalization.bandit import (
            ContextualThompsonBandit, bootstrap_bandit,
        )
        bandit = ContextualThompsonBandit(rng_seed=0)
        n = bootstrap_bandit(bandit, n_students=10, turns_per_student=5, seed=0)
        # 10 × 5 = 50 episodes
        assert n == 50
        assert bandit.total_pulls() == 50
        # Multiple buckets should have been touched
        assert len(bandit.posteriors) > 0


# ════════════════════════════════════════════════════════════════════
# Offline RL utilities (Phase 3)
# ════════════════════════════════════════════════════════════════════

class TestOfflineTraining:
    def test_logged_episode_roundtrip(self):
        from pedagogy.personalization.bandit import (
            LoggedEpisode, ContextBucket, StrategyAction, Strategy, SpeechRate,
        )
        ep = LoggedEpisode(
            context=ContextBucket("visual", "normal", "medium"),
            action=StrategyAction(Strategy.ANALOGY, SpeechRate.NORMAL),
            reward=0.8,
            propensity=0.05,
        )
        d = ep.to_dict()
        restored = LoggedEpisode.from_dict(d)
        assert restored.context == ep.context
        assert restored.action == ep.action
        assert restored.reward == 0.8
        assert restored.propensity == 0.05

    def test_train_offline_applies_updates(self):
        from pedagogy.personalization.bandit import (
            ContextualThompsonBandit, LoggedEpisode, EpisodeDataset,
            ContextBucket, StrategyAction, Strategy, SpeechRate, train_offline,
        )
        bandit = ContextualThompsonBandit()
        ds = EpisodeDataset()
        ctx = ContextBucket("visual", "normal", "medium")
        action = StrategyAction(Strategy.ANALOGY, SpeechRate.NORMAL)
        for _ in range(10):
            ds.append(LoggedEpisode(context=ctx, action=action, reward=1.0))
        n = train_offline(bandit, ds)
        assert n == 10
        # The arm should now have alpha = 1 + 10 = 11
        post = bandit._post(ctx, action)
        assert post.alpha == pytest.approx(11.0)

    def test_train_test_split(self):
        from pedagogy.personalization.bandit import (
            EpisodeDataset, LoggedEpisode, ContextBucket, StrategyAction,
            Strategy, SpeechRate, train_test_split,
        )
        ds = EpisodeDataset()
        ctx = ContextBucket("visual", "normal", "medium")
        action = StrategyAction(Strategy.ANALOGY, SpeechRate.NORMAL)
        for i in range(100):
            ds.append(LoggedEpisode(context=ctx, action=action, reward=float(i % 2)))
        train, test = train_test_split(ds, test_ratio=0.2, seed=0)
        assert len(train) + len(test) == 100
        assert len(test) == 20

    def test_jsonl_roundtrip(self, tmp_path):
        from pedagogy.personalization.bandit import (
            EpisodeDataset, LoggedEpisode, ContextBucket, StrategyAction,
            Strategy, SpeechRate,
        )
        ds = EpisodeDataset()
        ctx = ContextBucket("auditory", "fast", "high")
        action = StrategyAction(Strategy.SOCRATIC, SpeechRate.FAST)
        ds.append(LoggedEpisode(context=ctx, action=action, reward=0.9, propensity=0.3))
        path = tmp_path / "logs.jsonl"
        ds.to_jsonl(path)
        loaded = EpisodeDataset.from_jsonl(path)
        assert len(loaded) == 1
        assert loaded.episodes[0].context == ctx
        assert loaded.episodes[0].action == action

    def test_ips_off_policy_value(self):
        """Sanity : a policy that ALWAYS picks the logged action gets
        the average reward; one that NEVER matches gets 0."""
        from pedagogy.personalization.bandit import (
            EpisodeDataset, LoggedEpisode, ContextBucket, StrategyAction,
            Strategy, SpeechRate, ips_off_policy_value,
        )
        ds = EpisodeDataset()
        ctx = ContextBucket("visual", "normal", "medium")
        always_action = StrategyAction(Strategy.ANALOGY, SpeechRate.NORMAL)
        other_action = StrategyAction(Strategy.SOCRATIC, SpeechRate.SLOW)
        for _ in range(10):
            ds.append(LoggedEpisode(context=ctx, action=always_action, reward=1.0, propensity=1.0))
        # Policy that matches → V = 1.0
        v_match = ips_off_policy_value(lambda c: always_action, ds)
        assert v_match == pytest.approx(1.0)
        # Policy that never matches → V = 0
        v_miss = ips_off_policy_value(lambda c: other_action, ds)
        assert v_miss == 0.0

    def test_policy_from_bandit_trained_picks_best(self):
        from pedagogy.personalization.bandit import (
            ContextualThompsonBandit, ContextBucket, StrategyAction, Strategy,
            SpeechRate, policy_from_bandit,
        )
        bandit = ContextualThompsonBandit()
        ctx = ContextBucket("visual", "normal", "medium")
        winner = StrategyAction(Strategy.ANALOGY, SpeechRate.NORMAL)
        loser = StrategyAction(Strategy.SOCRATIC, SpeechRate.SLOW)
        # Train : winner gets 50 successes, loser gets 50 failures
        for _ in range(50):
            bandit.update(ctx, winner, 1.0)
            bandit.update(ctx, loser, 0.0)
        pi = policy_from_bandit(bandit)
        # Best arm by mean for this bucket should be the winner
        assert pi(ctx) == winner


# ════════════════════════════════════════════════════════════════════
# Log extractor (Postgres logs → LoggedEpisode)
# ════════════════════════════════════════════════════════════════════

class TestLogExtractor:
    def test_extra_payload_from_state_with_bandit_action(self):
        from pedagogy.personalization.bandit.log_extractor import extra_payload_from_state
        from agentic.schemas import Action
        qa_final = {
            "actions": [Action(type="answer", payload={
                "intent": "question",
                "memory_mode": "first_contact",
                "grounded": True,
                "bandit_strategy": "analogy",
                "bandit_speech_rate": "slow",
                "bandit_context": "visual|slow|low",
            })],
        }
        out = extra_payload_from_state(qa_final)
        assert out["bandit_strategy"] == "analogy"
        assert out["bandit_speech_rate"] == "slow"
        assert out["bandit_context"] == "visual|slow|low"

    def test_extra_payload_from_state_empty_for_no_actions(self):
        from pedagogy.personalization.bandit.log_extractor import extra_payload_from_state
        assert extra_payload_from_state({}) == {}
        assert extra_payload_from_state({"actions": []}) == {}
        assert extra_payload_from_state(None) == {}

    def test_extra_payload_from_state_empty_for_action_without_bandit(self):
        from pedagogy.personalization.bandit.log_extractor import extra_payload_from_state
        from agentic.schemas import Action
        qa_final = {
            "actions": [Action(type="continue", payload={"reason": "feedback_repeat"})],
        }
        # No bandit_* fields → empty extra payload
        assert extra_payload_from_state(qa_final) == {}

    def test_event_to_episode_full_row(self):
        from pedagogy.personalization.bandit.log_extractor import event_to_episode
        row = {
            "student_state": {
                "learning_style":     "visual",
                "avg_response_time":  22.0,
                "mastery_score":      0.5,
            },
            "event_payload": {
                "bandit_strategy":    "analogy",
                "bandit_speech_rate": "normal",
                "bandit_propensity":  0.05,
            },
            "reward": 0.85,
            "confusion_score": 0.0,
        }
        ep = event_to_episode(row)
        assert ep is not None
        assert ep.context.learning_style == "visual"
        assert ep.context.pace == "normal"
        assert ep.context.mastery_level == "medium"
        assert ep.action.strategy.value == "analogy"
        assert ep.action.speech_rate.value == "normal"
        assert ep.reward == 0.85
        assert ep.propensity == 0.05

    def test_event_to_episode_skips_row_without_action(self):
        from pedagogy.personalization.bandit.log_extractor import event_to_episode
        row = {
            "student_state": {"learning_style": "visual", "avg_response_time": 22.0, "mastery_score": 0.5},
            "event_payload": {},  # no bandit fields
            "reward": 1.0,
        }
        # No bandit_strategy → can't reconstruct an action → None
        assert event_to_episode(row) is None

    def test_event_to_episode_skips_row_without_style(self):
        from pedagogy.personalization.bandit.log_extractor import event_to_episode
        row = {
            "student_state": {},  # no learning_style
            "event_payload": {"bandit_strategy": "analogy", "bandit_speech_rate": "normal"},
        }
        assert event_to_episode(row) is None

    def test_event_to_episode_falls_back_to_confusion_score_for_reward(self):
        from pedagogy.personalization.bandit.log_extractor import event_to_episode
        row = {
            "student_state": {"learning_style": "auditory", "avg_response_time": 18.0, "mastery_score": 0.6},
            "event_payload": {"bandit_strategy": "socratic", "bandit_speech_rate": "fast"},
            "reward": None,
            "confusion_score": 0.3,
        }
        ep = event_to_episode(row)
        assert ep is not None
        # reward = 1 - 0.3 = 0.7
        assert ep.reward == pytest.approx(0.7)

    def test_event_to_episode_invalid_strategy_returns_none(self):
        from pedagogy.personalization.bandit.log_extractor import event_to_episode
        row = {
            "student_state": {"learning_style": "visual", "avg_response_time": 22.0, "mastery_score": 0.5},
            "event_payload": {"bandit_strategy": "invented_action", "bandit_speech_rate": "normal"},
        }
        # Unknown strategy enum value → None (not a crash)
        assert event_to_episode(row) is None
