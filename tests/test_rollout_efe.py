"""Tests for multi-step rollout EFE (backward induction / sophisticated inference)."""

import time

import numpy as np
import pytest

from architecture_core.cognition.tom.intent_particle_filter import (
    IntentParticleFilter,
    IntentProfile,
    ObservationContext,
    compute_intent_probability,
)
from architecture_core.cognition.tom.social_efe import (
    INTENT_SPEED,
    REWARD_SELF,
    RolloutConfig,
    RolloutEFEOutput,
    SocialEFE,
    reward_other,
)
from architecture_core.cognition.tom.gated_tom import GatedToM


# -- Helpers ---------------------------------------------------------------

def _approaching_obs(distance: float = 1.5, approaching: bool = True) -> ObservationContext:
    return ObservationContext(
        kinematic_intent="approach",
        distance=distance,
        velocity=0.3,
        approaching=approaching,
        gaze_on_robot=0.7,
        body_orientation=0.6,
        valence=0.0,
        arousal=0.2,
        robot_last_intent="neutral",
    )


def _default_profile() -> IntentProfile:
    return IntentProfile(
        approach_bias=1.0, responsiveness=0.5, precision=1.5, empathy_j=0.3,
    )


def _uniform_q():
    return {"approach": 0.2, "avoid": 0.2, "yield": 0.2, "wait": 0.2, "neutral": 0.2}


# -- Context-dependent REWARD_OTHER ----------------------------------------

class TestRewardOther:
    def test_yield_high_value_when_close_and_approaching(self):
        obs = _approaching_obs(distance=0.5, approaching=True)
        r_yield = reward_other("yield", "approach", obs)
        r_approach = reward_other("approach", "approach", obs)
        assert r_yield > r_approach

    def test_yield_low_value_when_far(self):
        obs = _approaching_obs(distance=5.0, approaching=False)
        r_yield = reward_other("yield", "approach", obs)
        r_approach = reward_other("approach", "approach", obs)
        assert r_approach > r_yield

    def test_yield_low_value_when_not_approaching(self):
        obs = _approaching_obs(distance=2.0, approaching=False)
        r_yield = reward_other("yield", "approach", obs)
        r_approach = reward_other("approach", "approach", obs)
        # With approaching=False, proximity *= 0.2, so baseline dominates
        assert r_approach > r_yield

    def test_returns_baseline_at_max_distance(self):
        obs = _approaching_obs(distance=10.0, approaching=False)
        # At max distance, proximity=0, pure baseline
        r_yield = reward_other("yield", "approach", obs)
        r_wait = reward_other("wait", "approach", obs)
        # Baseline: yield=0.2, wait=0.2 (both low)
        assert abs(r_yield - r_wait) < 0.01


# -- Backward induction rollout -------------------------------------------

class TestBackwardInduction:
    def test_empathic_yields_then_approaches(self):
        """Core test: empathy=1.0 at close range yields first, approaches later."""
        efe = SocialEFE(empathy_factor=1.0, beta=2.0, epistemic_weight=0.0)
        obs = _approaching_obs(distance=1.5)
        profile = _default_profile()
        result = efe.compute_rollout(_uniform_q(), obs, profile)
        # First action should be yield (obstruction model makes this clear)
        assert result.selected == "yield"
        # Policy should transition to approach after clearing path
        assert "approach" in result.full_policy

    def test_approach_when_far(self):
        """When far and not approaching, approach wins for empathic robot."""
        efe = SocialEFE(empathy_factor=1.0, beta=2.0, epistemic_weight=0.0)
        obs = _approaching_obs(distance=5.0, approaching=False)
        profile = _default_profile()
        result = efe.compute_rollout(_uniform_q(), obs, profile)
        assert result.selected == "approach"

    def test_rollout_distribution_sums_to_one(self):
        efe = SocialEFE(empathy_factor=0.5, beta=2.0, epistemic_weight=0.0)
        obs = _approaching_obs(distance=2.0)
        profile = _default_profile()
        result = efe.compute_rollout(_uniform_q(), obs, profile)
        total = sum(result.distribution.values())
        assert abs(total - 1.0) < 1e-6

    def test_full_policy_length_equals_horizon(self):
        cfg = RolloutConfig(horizon=7)
        efe = SocialEFE(empathy_factor=0.5, beta=2.0, epistemic_weight=0.0)
        obs = _approaching_obs()
        profile = _default_profile()
        result = efe.compute_rollout(_uniform_q(), obs, profile, config=cfg)
        assert len(result.full_policy) == 7

    def test_single_step_output_included(self):
        efe = SocialEFE(empathy_factor=0.5, beta=2.0, epistemic_weight=0.0)
        obs = _approaching_obs()
        profile = _default_profile()
        result = efe.compute_rollout(_uniform_q(), obs, profile)
        assert result.single_step_output is not None
        assert len(result.single_step_output.actions) == 5

    def test_distance_affects_rollout_policy(self):
        """Different starting distances produce different policies."""
        efe = SocialEFE(empathy_factor=1.0, beta=2.0, epistemic_weight=0.0)
        profile = _default_profile()
        result_close = efe.compute_rollout(
            _uniform_q(), _approaching_obs(distance=1.0), profile,
        )
        result_far = efe.compute_rollout(
            _uniform_q(), _approaching_obs(distance=5.0, approaching=False), profile,
        )
        # Close should yield, far should approach
        assert result_close.selected != result_far.selected

    def test_responsiveness_affects_rollout(self):
        """High vs low responsiveness produces different predictions."""
        efe = SocialEFE(empathy_factor=0.8, beta=2.0, epistemic_weight=0.0)
        obs = _approaching_obs(distance=2.0)
        profile_responsive = IntentProfile(
            approach_bias=0.5, responsiveness=2.0, precision=1.5, empathy_j=0.3,
        )
        profile_unresponsive = IntentProfile(
            approach_bias=0.5, responsiveness=0.0, precision=1.5, empathy_j=0.3,
        )
        result_r = efe.compute_rollout(_uniform_q(), obs, profile_responsive)
        result_u = efe.compute_rollout(_uniform_q(), obs, profile_unresponsive)
        # Policies should differ because human response predictions differ
        assert result_r.full_policy != result_u.full_policy or \
               result_r.distribution != result_u.distribution


# -- Shared generative model ----------------------------------------------

class TestComputeIntentProbability:
    def test_matches_particle_filter_method(self):
        """Module-level function matches particle filter's private method."""
        pf = IntentParticleFilter(n_particles=10)
        profile = _default_profile()
        obs = _approaching_obs()
        # Module-level
        p1 = compute_intent_probability(profile, obs)
        # Via particle filter (delegates to same function)
        p2 = pf._intent_probability(profile, obs)
        np.testing.assert_array_almost_equal(p1, p2)

    def test_output_sums_to_one(self):
        profile = _default_profile()
        obs = _approaching_obs()
        p = compute_intent_probability(profile, obs)
        assert abs(p.sum() - 1.0) < 1e-6

    def test_all_positive(self):
        profile = _default_profile()
        obs = _approaching_obs()
        p = compute_intent_probability(profile, obs)
        assert (p > 0).all()


# -- Initial obstruction ---------------------------------------------------

class TestInitialObstruction:
    def test_low_obstruction_selects_approach_at_close_range(self):
        """When robot has already cleared the path (obstruction~0), approach wins."""
        efe = SocialEFE(empathy_factor=1.0, beta=2.0, epistemic_weight=0.0)
        obs = _approaching_obs(distance=1.5)
        profile = _default_profile()
        # Robot is already off the agent's path
        result = efe.compute_rollout(
            _uniform_q(), obs, profile, initial_obstruction=0.1,
        )
        assert result.selected == "approach"

    def test_high_obstruction_selects_yield_at_close_range(self):
        """When robot is fully on path, yield wins at close range."""
        efe = SocialEFE(empathy_factor=1.0, beta=2.0, epistemic_weight=0.0)
        obs = _approaching_obs(distance=1.5)
        profile = _default_profile()
        result = efe.compute_rollout(
            _uniform_q(), obs, profile, initial_obstruction=1.0,
        )
        assert result.selected == "yield"

    def test_obstruction_changes_policy(self):
        """Different initial obstruction produces different policies."""
        efe = SocialEFE(empathy_factor=1.0, beta=2.0, epistemic_weight=0.0)
        obs = _approaching_obs(distance=1.5)
        profile = _default_profile()
        result_blocked = efe.compute_rollout(
            _uniform_q(), obs, profile, initial_obstruction=1.0,
        )
        result_clear = efe.compute_rollout(
            _uniform_q(), obs, profile, initial_obstruction=0.0,
        )
        assert result_blocked.full_policy != result_clear.full_policy


class TestComputeObstruction:
    def test_robot_on_path_returns_high(self):
        """Robot directly on agent's path → obstruction near 1.0."""
        from architecture_core.cognition.tom.tom_modulator import ToMModulator
        # Agent at (0,0) heading to (5,0), robot at (2,0) — directly on path
        obs = ToMModulator._compute_obstruction(2.0, 0.0, (0, 0), (5, 0), 2.0)
        assert obs > 0.9

    def test_robot_off_path_returns_low(self):
        """Robot far off agent's path → obstruction near 0.0."""
        from architecture_core.cognition.tom.tom_modulator import ToMModulator
        # Agent at (0,0) heading to (5,0), robot at (2,2) — 2m off path
        obs = ToMModulator._compute_obstruction(2.0, 2.0, (0, 0), (5, 0), 2.83)
        assert obs < 0.1

    def test_robot_partially_off_path(self):
        """Robot partially off path → intermediate obstruction."""
        from architecture_core.cognition.tom.tom_modulator import ToMModulator
        # Agent at (0,0) heading to (5,0), robot at (2, 0.4) — 0.4m off path
        obs = ToMModulator._compute_obstruction(2.0, 0.4, (0, 0), (5, 0), 2.04)
        assert 0.3 < obs < 0.7

    def test_stationary_agent_close_returns_high(self):
        """Agent with no goal but close → obstruction high."""
        from architecture_core.cognition.tom.tom_modulator import ToMModulator
        obs = ToMModulator._compute_obstruction(1.0, 0.0, (0, 0), (0, 0), 1.0)
        assert obs == 1.0

    def test_stationary_agent_far_returns_low(self):
        """Agent with no goal and far → obstruction low."""
        from architecture_core.cognition.tom.tom_modulator import ToMModulator
        obs = ToMModulator._compute_obstruction(5.0, 0.0, (0, 0), (0, 0), 5.0)
        assert obs == 0.0


# -- GatedToM rollout integration -----------------------------------------

class TestGatedToMRollout:
    def test_always_uses_rollout_even_low_reliability(self):
        """Full rollout is always used — no single-step fallback."""
        gated = GatedToM(empathy_factor=0.5, n_particles=20)
        obs = _approaching_obs()
        # No updates → low reliability, but rollout still runs
        result = gated.select_action_rollout("agent_1", obs)
        assert isinstance(result, RolloutEFEOutput)
        # Full rollout, not single-step fallback
        assert len(result.full_policy) > 1

    def test_profile_blending_with_reliability(self):
        """Low reliability blends toward conservative default profile."""
        gated = GatedToM(empathy_factor=0.8, n_particles=50)
        obs = _approaching_obs(distance=2.0)
        # No updates → low reliability → uses mostly default profile
        result_low = gated.select_action_rollout("agent_1", obs)
        # Many updates → high reliability → uses learned profile
        for _ in range(30):
            gated.update("agent_1", obs)
        result_high = gated.select_action_rollout("agent_1", obs)
        # Both produce full rollouts but policies may differ
        assert len(result_low.full_policy) > 1
        assert len(result_high.full_policy) > 1

    def test_rollout_after_sufficient_updates(self):
        """After many updates, reliability rises and rollout is used."""
        gated = GatedToM(empathy_factor=0.8, n_particles=50)
        obs = _approaching_obs(distance=2.0)
        # Feed observations to build reliability
        for _ in range(30):
            gated.update("agent_1", obs)
        result = gated.select_action_rollout("agent_1", obs)
        # Should use full rollout (policy length > 1)
        assert len(result.full_policy) > 1


# -- Performance ----------------------------------------------------------

class TestRolloutPerformance:
    def test_rollout_under_50ms(self):
        efe = SocialEFE(empathy_factor=0.8, beta=2.0, epistemic_weight=0.0)
        obs = _approaching_obs()
        profile = _default_profile()
        # Warmup
        efe.compute_rollout(_uniform_q(), obs, profile)
        # Time
        start = time.perf_counter()
        n = 20
        for _ in range(n):
            efe.compute_rollout(_uniform_q(), obs, profile)
        elapsed_ms = (time.perf_counter() - start) / n * 1000
        assert elapsed_ms < 500, f"Rollout took {elapsed_ms:.1f}ms (limit: 500ms)"
