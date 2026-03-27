"""Tests for fused intent inference."""

from __future__ import annotations

import math

import pytest

from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate
from architecture_core.cognition.tom.intent_inference import (
    INTENT_LABELS,
    AgentIntentBelief,
    IntentHypothesis,
    IntentInference,
    _affect_likelihood,
    _engagement_likelihood,
    _kinematic_likelihood,
)
from architecture_core.cognition.tom.tom_modulator import ToMModulator


# =====================================================================
# Helpers
# =====================================================================
def _make_pb(
    agents=None, engagement_readings=None, affect_readings=None,
):
    world = {"robot_pose": (0, 0, 0, 0), "agents": agents or []}
    social = {}
    if engagement_readings is not None:
        social["engagement"] = {"readings": engagement_readings}
    if affect_readings is not None:
        social["affect"] = {"readings": affect_readings}
    return PerceptBundle(t=0.0, world=world, social=social)


# =====================================================================
# Likelihood functions
# =====================================================================
class TestKinematicLikelihood:
    def test_dominant_intent(self):
        lik = _kinematic_likelihood("approach")
        assert lik["approach"] == 0.8
        assert lik["avoid"] == 0.05

    def test_sums_to_about_one(self):
        lik = _kinematic_likelihood("yield")
        total = sum(lik.values())
        assert total == pytest.approx(1.0, abs=0.01)


class TestEngagementLikelihood:
    def test_high_gaze_favors_approach(self):
        lik = _engagement_likelihood(gaze=0.9, body_orient=0.9, proximity_trend=-0.3)
        assert lik["approach"] > lik["avoid"]

    def test_low_gaze_favors_avoid(self):
        lik = _engagement_likelihood(gaze=0.1, body_orient=0.1, proximity_trend=0.3)
        assert lik["avoid"] > lik["approach"]


class TestAffectLikelihood:
    def test_positive_valence_favors_approach(self):
        lik = _affect_likelihood(valence=0.8, arousal=-0.3)
        assert lik["approach"] > lik["avoid"]

    def test_negative_valence_high_arousal_favors_avoid(self):
        lik = _affect_likelihood(valence=-0.8, arousal=0.8)
        assert lik["avoid"] > lik["approach"]


# =====================================================================
# IntentInference
# =====================================================================
class TestIntentInference:
    def test_no_agents_empty_result(self):
        inf = IntentInference()
        pb = _make_pb(agents=[])
        beliefs = inf.infer(pb)
        assert beliefs == []

    def test_single_agent_returns_one_belief(self):
        inf = IntentInference()
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        beliefs = inf.infer(pb, kinematic_intent="approach")
        assert len(beliefs) == 1
        assert beliefs[0].entity_id == "h1"

    def test_distribution_sums_to_one(self):
        inf = IntentInference()
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        beliefs = inf.infer(pb, kinematic_intent="approach")
        total = sum(h.probability for h in beliefs[0].hypotheses)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_kinematic_approach_dominates_without_other_signals(self):
        inf = IntentInference()
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        beliefs = inf.infer(pb, kinematic_intent="approach")
        assert beliefs[0].most_likely == "approach"

    def test_engagement_shifts_intent(self):
        inf = IntentInference()
        # Kinematic says neutral, but engagement strongly says approach
        pb = _make_pb(
            agents=[{"id": "h1", "pose": (1, 0)}],
            engagement_readings=[{
                "entity_id": "h1",
                "gaze_on_robot": 0.95,
                "body_orientation": 0.9,
                "proximity_trend": -0.4,
                "score": 0.9,
            }],
        )
        beliefs = inf.infer(pb, kinematic_intent="neutral")
        # With strong engagement, approach should be boosted
        approach_prob = next(
            h.probability for h in beliefs[0].hypotheses if h.intent == "approach"
        )
        avoid_prob = next(
            h.probability for h in beliefs[0].hypotheses if h.intent == "avoid"
        )
        assert approach_prob > avoid_prob

    def test_negative_affect_shifts_toward_avoid(self):
        inf = IntentInference()
        pb = _make_pb(
            agents=[{"id": "h1", "pose": (1, 0)}],
            affect_readings=[{
                "entity_id": "h1",
                "valence": -0.8,
                "arousal": 0.8,
            }],
        )
        beliefs = inf.infer(pb, kinematic_intent="neutral")
        avoid_prob = next(
            h.probability for h in beliefs[0].hypotheses if h.intent == "avoid"
        )
        approach_prob = next(
            h.probability for h in beliefs[0].hypotheses if h.intent == "approach"
        )
        assert avoid_prob > approach_prob

    def test_script_prior_biases_distribution(self):
        inf = IntentInference()
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        # Handover script strongly favors approach
        beliefs_handover = inf.infer(pb, kinematic_intent="neutral", situation_type="handover")
        # Corridor encounter favors yield
        beliefs_corridor = inf.infer(pb, kinematic_intent="neutral", situation_type="corridor_encounter")

        approach_handover = next(
            h.probability for h in beliefs_handover[0].hypotheses if h.intent == "approach"
        )
        approach_corridor = next(
            h.probability for h in beliefs_corridor[0].hypotheses if h.intent == "approach"
        )
        # Handover context should give higher approach probability
        assert approach_handover > approach_corridor

    def test_confidence_and_entropy(self):
        inf = IntentInference()
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        beliefs = inf.infer(pb, kinematic_intent="approach")
        b = beliefs[0]
        assert 0.0 <= b.confidence <= 1.0
        assert b.entropy >= 0.0

    def test_hypotheses_sorted_by_probability(self):
        inf = IntentInference()
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        beliefs = inf.infer(pb, kinematic_intent="approach")
        probs = [h.probability for h in beliefs[0].hypotheses]
        assert probs == sorted(probs, reverse=True)

    def test_evidence_sources_tracked(self):
        inf = IntentInference()
        pb = _make_pb(
            agents=[{"id": "h1", "pose": (1, 0)}],
            engagement_readings=[{
                "entity_id": "h1",
                "gaze_on_robot": 0.5,
                "body_orientation": 0.5,
                "proximity_trend": 0.0,
            }],
            affect_readings=[{
                "entity_id": "h1",
                "valence": 0.0,
                "arousal": 0.0,
            }],
        )
        beliefs = inf.infer(pb, kinematic_intent="neutral")
        sources = beliefs[0].evidence_sources
        assert "kinematic" in sources
        assert "engagement" in sources
        assert "affect" in sources
        assert "script_prior" in sources

    def test_multi_agent(self):
        inf = IntentInference()
        pb = _make_pb(agents=[
            {"id": "h1", "pose": (1, 0)},
            {"id": "h2", "pose": (3, 0)},
        ])
        beliefs = inf.infer(pb, kinematic_intent="approach")
        assert len(beliefs) == 2
        ids = {b.entity_id for b in beliefs}
        assert ids == {"h1", "h2"}

    def test_custom_channel_weights(self):
        # Heavy engagement weight, zero kinematic
        inf = IntentInference(channel_weights={
            "kinematic": 0.0,
            "engagement": 2.0,
            "affect": 0.0,
            "script_prior": 0.0,
        })
        pb = _make_pb(
            agents=[{"id": "h1", "pose": (1, 0)}],
            engagement_readings=[{
                "entity_id": "h1",
                "gaze_on_robot": 0.95,
                "body_orientation": 0.95,
                "proximity_trend": -0.5,
            }],
        )
        beliefs = inf.infer(pb, kinematic_intent="avoid")
        # Even though kinematic says avoid, engagement weight dominates
        assert beliefs[0].most_likely == "approach"


# =====================================================================
# ToMModulator with IntentInference
# =====================================================================
class TestToMModulatorWithInference:
    def test_no_inference_returns_kinematic(self):
        tom = ToMModulator()
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        update = tom.modulate(pb, SkillRequest(skill="navigate"))
        assert update.intent == "neutral"  # no adapter = neutral

    def test_inference_overrides_intent(self):
        inf = IntentInference()
        tom = ToMModulator(intent_inference=inf)
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        update = tom.modulate(pb, SkillRequest(skill="navigate"))
        # With inference but no adapter, kinematic is "neutral"
        # Inference still produces a fused result
        assert update.intent in INTENT_LABELS
        assert "intent_confidence" in update.params
        assert "intent_entropy" in update.params

    def test_situation_type_forwarded(self):
        inf = IntentInference()
        tom = ToMModulator(intent_inference=inf)
        tom.set_situation_type("handover")
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        update = tom.modulate(pb, SkillRequest(skill="navigate"))
        # Handover context should bias toward approach
        dist = update.params.get("intent_distribution", {})
        assert dist.get("approach", 0) > dist.get("avoid", 0)

    def test_graceful_degradation_on_error(self):
        class BrokenInference:
            def infer(self, *args, **kwargs):
                raise RuntimeError("test error")

        tom = ToMModulator(intent_inference=BrokenInference())
        pb = _make_pb(agents=[{"id": "h1", "pose": (1, 0)}])
        update = tom.modulate(pb, SkillRequest(skill="navigate"))
        # Should fall back to kinematic (neutral)
        assert update.intent == "neutral"
        assert "error" in update.debug
