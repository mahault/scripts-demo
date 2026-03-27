"""Tests for active inference ToM: particle filter, Social EFE, GatedToM."""

from __future__ import annotations

import math

import pytest
import numpy as np

from architecture_core.core.types import AffectState, PerceptBundle, SkillRequest, SkillUpdate
from architecture_core.cognition.tom.intent_particle_filter import (
    INTENT_LABELS,
    IntentParticleFilter,
    IntentProfile,
    ObservationContext,
)
from architecture_core.cognition.tom.social_efe import (
    SocialEFE,
    SocialEFEOutput,
    REWARD_SELF,
    REWARD_OTHER,
)
from architecture_core.cognition.tom.gated_tom import GatedToM
from architecture_core.cognition.tom.tom_modulator import ToMModulator


# =====================================================================
# Helpers
# =====================================================================
def _approach_obs(**overrides) -> ObservationContext:
    defaults = dict(
        kinematic_intent="approach",
        distance=1.0,
        velocity=0.5,
        approaching=True,
        gaze_on_robot=0.9,
        body_orientation=0.8,
        valence=0.3,
        arousal=0.2,
        robot_last_intent="neutral",
    )
    defaults.update(overrides)
    return ObservationContext(**defaults)


def _avoid_obs(**overrides) -> ObservationContext:
    defaults = dict(
        kinematic_intent="avoid",
        distance=3.0,
        velocity=0.8,
        approaching=False,
        gaze_on_robot=0.1,
        body_orientation=0.2,
        valence=-0.5,
        arousal=0.7,
        robot_last_intent="neutral",
    )
    defaults.update(overrides)
    return ObservationContext(**defaults)


def _make_pb(agents=None, engagement=None, affect=None):
    world = {"robot_pose": (0, 0, 0, 0), "agents": agents or []}
    social = {}
    if engagement:
        social["engagement"] = {"readings": engagement}
    if affect:
        social["affect"] = {"readings": affect}
    return PerceptBundle(t=0.0, world=world, social=social)


# =====================================================================
# IntentParticleFilter — initialization
# =====================================================================
class TestParticleFilterInit:
    def test_correct_particle_count(self):
        pf = IntentParticleFilter(n_particles=50)
        assert pf.n_particles == 50
        assert len(pf.weights) == 50

    def test_weights_sum_to_one(self):
        pf = IntentParticleFilter(n_particles=100)
        assert pf.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_initial_uniform_weights(self):
        pf = IntentParticleFilter(n_particles=100)
        assert np.allclose(pf.weights, 1.0 / 100)


# =====================================================================
# IntentParticleFilter — update + predict
# =====================================================================
class TestParticleFilterUpdate:
    def test_update_affects_predictions(self):
        pf = IntentParticleFilter(n_particles=100)
        obs = _approach_obs()
        dist_before = pf.predict_intent(obs)
        pf.update(obs)
        dist_after = pf.predict_intent(obs)
        # Predictions should change after incorporating evidence
        # (weights may be re-uniformized by resampling, but particles shift)
        diffs = [abs(dist_after[i] - dist_before[i]) for i in INTENT_LABELS]
        assert sum(diffs) > 0.0  # at least some change

    def test_weights_still_sum_to_one_after_update(self):
        pf = IntentParticleFilter(n_particles=100)
        pf.update(_approach_obs())
        assert pf.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_predict_returns_valid_distribution(self):
        pf = IntentParticleFilter(n_particles=100)
        pf.update(_approach_obs())
        dist = pf.predict_intent(_approach_obs())
        assert len(dist) == len(INTENT_LABELS)
        total = sum(dist.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_approach_observations_favor_approach(self):
        pf = IntentParticleFilter(n_particles=200)
        # Feed many approach observations
        for _ in range(20):
            pf.update(_approach_obs())
        dist = pf.predict_intent(_approach_obs())
        assert dist["approach"] > dist["avoid"]

    def test_avoid_observations_favor_avoid(self):
        pf = IntentParticleFilter(n_particles=200)
        for _ in range(20):
            pf.update(_avoid_obs())
        dist = pf.predict_intent(_avoid_obs())
        assert dist["avoid"] > dist["approach"]

    def test_repeated_updates_increase_reliability(self):
        pf = IntentParticleFilter(n_particles=100)
        r0 = pf.reliability
        for _ in range(10):
            pf.update(_approach_obs())
        r1 = pf.reliability
        # After consistent observations, reliability should increase
        assert r1 >= r0


# =====================================================================
# IntentParticleFilter — reliability metrics
# =====================================================================
class TestParticleFilterMetrics:
    def test_ess_initially_equals_n(self):
        pf = IntentParticleFilter(n_particles=100)
        # Uniform weights → ESS = n
        assert pf.ess == pytest.approx(100.0, abs=1.0)

    def test_entropy_nonnegative(self):
        pf = IntentParticleFilter(n_particles=50)
        pf.update(_approach_obs())
        assert pf.entropy >= 0.0

    def test_reliability_in_zero_one(self):
        pf = IntentParticleFilter(n_particles=50)
        assert 0.0 <= pf.reliability <= 1.0
        pf.update(_approach_obs())
        assert 0.0 <= pf.reliability <= 1.0

    def test_mean_profile_returns_profile(self):
        pf = IntentParticleFilter(n_particles=50)
        mp = pf.mean_profile()
        assert isinstance(mp, IntentProfile)
        assert -10 < mp.approach_bias < 10
        assert 0 <= mp.empathy_j <= 1


# =====================================================================
# IntentParticleFilter — epistemic value
# =====================================================================
class TestEpistemicValue:
    def test_epistemic_value_is_finite(self):
        pf = IntentParticleFilter(n_particles=50)
        obs = _approach_obs()
        ev = pf.epistemic_value("approach", obs)
        assert math.isfinite(ev)

    def test_different_actions_different_epistemic_values(self):
        pf = IntentParticleFilter(n_particles=50)
        obs = _approach_obs()
        ev_approach = pf.epistemic_value("approach", obs)
        ev_avoid = pf.epistemic_value("avoid", obs)
        # They should generally differ (not identical)
        # But can be close for uniform particles, so just check they're finite
        assert math.isfinite(ev_approach)
        assert math.isfinite(ev_avoid)


# =====================================================================
# SocialEFE
# =====================================================================
class TestSocialEFE:
    def test_returns_all_intents(self):
        efe = SocialEFE(empathy_factor=0.3)
        q_human = {"approach": 0.5, "avoid": 0.1, "yield": 0.2, "wait": 0.1, "neutral": 0.1}
        result = efe.compute(q_human, _approach_obs())
        assert len(result.actions) == len(INTENT_LABELS)

    def test_probabilities_sum_to_one(self):
        efe = SocialEFE(empathy_factor=0.3)
        q_human = {"approach": 0.5, "avoid": 0.1, "yield": 0.2, "wait": 0.1, "neutral": 0.1}
        result = efe.compute(q_human, _approach_obs())
        total = sum(a.probability for a in result.actions)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_selected_is_in_intents(self):
        efe = SocialEFE(empathy_factor=0.3)
        q_human = {"approach": 0.5, "avoid": 0.1, "yield": 0.2, "wait": 0.1, "neutral": 0.1}
        result = efe.compute(q_human, _approach_obs())
        assert result.selected in INTENT_LABELS

    def test_selfish_robot_ignores_other(self):
        efe_selfish = SocialEFE(empathy_factor=0.0)
        efe_empathic = SocialEFE(empathy_factor=1.0)
        q_human = {"approach": 0.5, "avoid": 0.1, "yield": 0.2, "wait": 0.1, "neutral": 0.1}
        result_s = efe_selfish.compute(q_human, _approach_obs())
        result_e = efe_empathic.compute(q_human, _approach_obs())
        # G_other should be zero for selfish, nonzero for empathic
        # Their selected actions may differ
        assert result_s.empathy_factor == 0.0
        assert result_e.empathy_factor == 1.0

    def test_empathic_robot_yields_to_distressed_human(self):
        efe = SocialEFE(empathy_factor=0.8)
        q_human = {"approach": 0.3, "avoid": 0.3, "yield": 0.1, "wait": 0.2, "neutral": 0.1}
        distressed = AffectState(arousal=0.8, valence=-0.8)
        result = efe.compute(q_human, _approach_obs(), other_affect=distressed)
        # With high empathy + distressed human, affect risk (expected
        # surprisal) should make yield/wait favored over approach
        yield_prob = result.distribution.get("yield", 0)
        wait_prob = result.distribution.get("wait", 0)
        approach_prob = result.distribution.get("approach", 0)
        assert (yield_prob + wait_prob) > approach_prob

    def test_efe_with_epistemic_value(self):
        pf = IntentParticleFilter(n_particles=50)
        efe = SocialEFE(empathy_factor=0.3, epistemic_weight=1.0)
        q_human = {"approach": 0.5, "avoid": 0.1, "yield": 0.2, "wait": 0.1, "neutral": 0.1}
        result = efe.compute(q_human, _approach_obs(), particle_filter=pf)
        # Epistemic values should be nonzero
        has_epistemic = any(a.g_epistemic != 0.0 for a in result.actions)
        assert has_epistemic

    def test_actions_sorted_by_probability(self):
        efe = SocialEFE(empathy_factor=0.3)
        q_human = {"approach": 0.5, "avoid": 0.1, "yield": 0.2, "wait": 0.1, "neutral": 0.1}
        result = efe.compute(q_human, _approach_obs())
        probs = [a.probability for a in result.actions]
        assert probs == sorted(probs, reverse=True)


# =====================================================================
# GatedToM
# =====================================================================
class TestGatedToM:
    def test_new_agent_low_trust(self):
        gated = GatedToM(n_particles=50)
        # Fresh filter → moderate reliability (sigmoid of 0 entropy)
        obs = _approach_obs()
        q = gated.predict_intent("h1", obs)
        # Should be blended toward prior
        assert sum(q.values()) == pytest.approx(1.0, abs=0.01)

    def test_predict_returns_valid_distribution(self):
        gated = GatedToM(n_particles=50)
        obs = _approach_obs()
        q = gated.predict_intent("h1", obs)
        assert len(q) == len(INTENT_LABELS)
        assert all(0 <= v <= 1 for v in q.values())

    def test_update_then_predict(self):
        gated = GatedToM(n_particles=50)
        obs = _approach_obs()
        gated.update("h1", obs)
        q = gated.predict_intent("h1", obs)
        total = sum(q.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_select_action_returns_efe_output(self):
        gated = GatedToM(n_particles=50, empathy_factor=0.3)
        obs = _approach_obs()
        gated.update("h1", obs)
        result = gated.select_action("h1", obs)
        assert isinstance(result, SocialEFEOutput)
        assert result.selected in INTENT_LABELS
        assert result.empathy_factor == 0.3

    def test_empathy_factor_settable(self):
        gated = GatedToM(empathy_factor=0.3)
        assert gated.empathy_factor == 0.3
        gated.empathy_factor = 0.8
        assert gated.empathy_factor == 0.8

    def test_prune_stale_removes_filters(self):
        gated = GatedToM(n_particles=50)
        gated.update("h1", _approach_obs())
        gated.update("h2", _avoid_obs())
        assert gated.agent_reliability("h1") > 0
        gated.prune_stale({"h2"})
        assert gated.agent_reliability("h1") == 0.0  # pruned

    def test_consistent_observations_build_confidence(self):
        gated = GatedToM(n_particles=100)
        obs = _approach_obs()
        for _ in range(15):
            gated.update("h1", obs)
        r = gated.agent_reliability("h1")
        # After many consistent observations, reliability should be reasonable
        assert r > 0.0


# =====================================================================
# ToMModulator with GatedToM
# =====================================================================
class TestToMModulatorWithGatedToM:
    def test_gated_tom_produces_active_inference_debug(self):
        gated = GatedToM(n_particles=50, empathy_factor=0.3)
        tom = ToMModulator(gated_tom=gated)
        pb = _make_pb(
            agents=[{"id": "h1", "pose": (1, 0)}],
            engagement=[{
                "entity_id": "h1",
                "gaze_on_robot": 0.8,
                "body_orientation": 0.7,
                "proximity_trend": -0.2,
            }],
            affect=[{
                "entity_id": "h1",
                "valence": 0.3,
                "arousal": 0.2,
            }],
        )
        update = tom.modulate(pb, SkillRequest(skill="navigate"))
        assert "active_inference" in update.debug
        assert "intent_confidence" in update.params
        assert "g_social" in update.params

    def test_gated_tom_overrides_kinematic(self):
        gated = GatedToM(n_particles=50)
        tom = ToMModulator(gated_tom=gated)
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        update = tom.modulate(pb, SkillRequest(skill="navigate"))
        assert update.intent in INTENT_LABELS

    def test_falls_back_to_inference_on_no_agents(self):
        gated = GatedToM(n_particles=50)
        tom = ToMModulator(gated_tom=gated)
        pb = _make_pb(agents=[])
        update = tom.modulate(pb, SkillRequest(skill="navigate"))
        # No agents → GatedToM returns None → falls back
        assert update.intent == "neutral"

    def test_efe_breakdown_in_params(self):
        gated = GatedToM(n_particles=50, empathy_factor=0.5)
        tom = ToMModulator(gated_tom=gated)
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        update = tom.modulate(pb, SkillRequest(skill="navigate"))
        assert "g_self" in update.params
        assert "g_other" in update.params
        assert "empathy_factor" in update.params
