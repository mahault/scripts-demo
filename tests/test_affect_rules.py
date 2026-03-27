"""Tests for affect-modulated norm rules."""

from __future__ import annotations

import pytest

from architecture_core.core.types import (
    AffectState,
    PerceptBundle,
    SkillRequest,
)
from architecture_core.cognition.norms.affect_rules import (
    AffectModulatedSpeedRule,
    DistressVetoRule,
)


# =====================================================================
# Helpers
# =====================================================================
def _dummy_pb():
    return PerceptBundle(
        t=0.0,
        world={"robot_pose": (0, 0, 0, 0), "agents": []},
    )


def _dummy_req():
    return SkillRequest(skill="navigate")


# =====================================================================
# AffectModulatedSpeedRule
# =====================================================================
class TestAffectModulatedSpeedRule:
    def test_neutral_affect_no_reduction(self):
        rule = AffectModulatedSpeedRule(base_max_speed=0.5)
        rule.set_affect(AffectState(arousal=0.0, valence=0.0))
        c = rule.evaluate(_dummy_pb(), _dummy_req())
        assert c.hard["max_linear_speed"] == 0.5
        assert c.veto is None

    def test_positive_valence_no_reduction(self):
        rule = AffectModulatedSpeedRule(base_max_speed=0.5)
        rule.set_affect(AffectState(arousal=0.0, valence=0.5))
        c = rule.evaluate(_dummy_pb(), _dummy_req())
        assert c.hard["max_linear_speed"] == 0.5

    def test_negative_valence_reduces_speed(self):
        rule = AffectModulatedSpeedRule(base_max_speed=0.5, valence_threshold=-0.3)
        rule.set_affect(AffectState(arousal=0.0, valence=-0.6))
        c = rule.evaluate(_dummy_pb(), _dummy_req())
        assert c.hard["max_linear_speed"] < 0.5

    def test_extreme_negative_valence_minimum_speed(self):
        rule = AffectModulatedSpeedRule(
            base_max_speed=0.5, valence_threshold=-0.3, min_speed_fraction=0.3,
        )
        rule.set_affect(AffectState(arousal=0.0, valence=-1.0))
        c = rule.evaluate(_dummy_pb(), _dummy_req())
        assert c.hard["max_linear_speed"] >= 0.5 * 0.3 - 0.01  # allow rounding

    def test_at_threshold_no_reduction(self):
        rule = AffectModulatedSpeedRule(base_max_speed=0.5, valence_threshold=-0.3)
        rule.set_affect(AffectState(arousal=0.0, valence=-0.3))
        c = rule.evaluate(_dummy_pb(), _dummy_req())
        assert c.hard["max_linear_speed"] == 0.5


# =====================================================================
# DistressVetoRule
# =====================================================================
class TestDistressVetoRule:
    def test_neutral_no_veto(self):
        rule = DistressVetoRule()
        rule.set_affect(AffectState(arousal=0.0, valence=0.0))
        c = rule.evaluate(_dummy_pb(), _dummy_req())
        assert c.veto is None

    def test_extreme_distress_vetoes(self):
        rule = DistressVetoRule(valence_threshold=-0.8, arousal_threshold=0.8)
        rule.set_affect(AffectState(arousal=0.9, valence=-0.9))
        c = rule.evaluate(_dummy_pb(), _dummy_req())
        assert c.veto is not None
        assert "Distress veto" in c.veto

    def test_low_valence_alone_insufficient(self):
        rule = DistressVetoRule(valence_threshold=-0.8, arousal_threshold=0.8)
        rule.set_affect(AffectState(arousal=0.3, valence=-0.9))
        c = rule.evaluate(_dummy_pb(), _dummy_req())
        assert c.veto is None

    def test_high_arousal_alone_insufficient(self):
        rule = DistressVetoRule(valence_threshold=-0.8, arousal_threshold=0.8)
        rule.set_affect(AffectState(arousal=0.9, valence=0.2))
        c = rule.evaluate(_dummy_pb(), _dummy_req())
        assert c.veto is None

    def test_at_boundary_no_veto(self):
        rule = DistressVetoRule(valence_threshold=-0.8, arousal_threshold=0.8)
        rule.set_affect(AffectState(arousal=0.8, valence=-0.8))
        c = rule.evaluate(_dummy_pb(), _dummy_req())
        assert c.veto is None  # needs strictly < and >
