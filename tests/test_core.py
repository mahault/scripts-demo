"""Unit tests for the social-layer core modules."""

from __future__ import annotations

import math

import pytest

from architecture_core.core.types import (
    NormativeConstraints,
    PerceptBundle,
    SkillRequest,
    SkillUpdate,
)
from architecture_core.core.status import Status
from architecture_core.safety.geometry import (
    Circle,
    Rect,
    bearing,
    circle_circle_distance,
    circle_rect_overlap,
    clamp_position,
    point_distance,
    point_in_rect,
)
from architecture_core.safety.constraints import (
    DistanceConstraint,
    KeepoutConstraint,
    SafetyConstraintSet,
    SpeedConstraint,
)
from architecture_core.perception.augmentations.proxemics import (
    ProxemicReading,
    ProxemicsAugmentation,
    classify_zone,
)
from architecture_core.cognition.norms.rules import (
    PersonalSpaceRule,
    SpeedLimitRule,
    KeepoutZoneRule,
)
from architecture_core.cognition.norms.norm_engine import NormEngine
from architecture_core.safety.shield import SafetyShield
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.script_types import (
    ScriptSequence,
    ScriptStep,
)


# -------------------------------------------------------------------
# Geometry
# -------------------------------------------------------------------
class TestGeometry:
    def test_point_distance(self):
        assert point_distance(0, 0, 3, 4) == pytest.approx(5.0)
        assert point_distance(1, 1, 1, 1) == pytest.approx(0.0)

    def test_circle_circle_distance(self):
        a = Circle(0, 0, 1.0)
        b = Circle(5, 0, 1.0)
        assert circle_circle_distance(a, b) == pytest.approx(3.0)

    def test_circle_circle_overlap(self):
        a = Circle(0, 0, 1.0)
        b = Circle(1.5, 0, 1.0)
        assert circle_circle_distance(a, b) < 0  # overlap

    def test_point_in_rect(self):
        r = Rect(0, 0, 2, 3)
        assert point_in_rect(1, 1, r) is True
        assert point_in_rect(5, 5, r) is False

    def test_circle_rect_overlap(self):
        c = Circle(3.0, 1.5, 1.5)
        r = Rect(0, 0, 2, 3)
        assert circle_rect_overlap(c, r) is True

    def test_circle_rect_no_overlap(self):
        c = Circle(10, 10, 0.5)
        r = Rect(0, 0, 2, 3)
        assert circle_rect_overlap(c, r) is False

    def test_clamp_position(self):
        zones = [Rect(0, 0, 2, 2)]
        x, y = clamp_position(1.0, 1.0, zones, agent_radius=0.0)
        # Should be pushed outside the zone
        assert not point_in_rect(x, y, zones[0])

    def test_bearing(self):
        assert bearing(0, 0, 1, 0) == pytest.approx(0.0)
        assert bearing(0, 0, 0, 1) == pytest.approx(math.pi / 2)


# -------------------------------------------------------------------
# Constraints data types
# -------------------------------------------------------------------
class TestConstraints:
    def test_safety_constraint_set(self):
        s = SafetyConstraintSet()
        s.distance.append(DistanceConstraint("agent_1", 0.5, 0.8))
        s.keepout.append(KeepoutConstraint(Rect(0, 0, 1, 1), "restricted"))
        s.speed.append(SpeedConstraint(max_linear=0.5))
        assert len(s.distance) == 1
        assert len(s.keepout) == 1
        assert s.speed[0].max_linear == 0.5


# -------------------------------------------------------------------
# Proxemics
# -------------------------------------------------------------------
class TestProxemics:
    def test_classify_zone(self):
        assert classify_zone(0.3) == "intimate"
        assert classify_zone(0.8) == "personal"
        assert classify_zone(2.0) == "social"
        assert classify_zone(10.0) == "public"

    def test_augment_no_agents(self):
        aug = ProxemicsAugmentation()
        result = aug.augment({"robot_pose": (0, 0, 0, 0), "agents": []})
        prox = result["proxemics"]
        assert prox["closest_distance"] == float("inf")
        assert prox["closest_zone"] == "public"
        assert prox["num_in_personal"] == 0

    def test_augment_with_agent(self):
        aug = ProxemicsAugmentation()
        result = aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [{"id": "a1", "pose": (0.3, 0)}],
        })
        prox = result["proxemics"]
        assert prox["closest_distance"] == pytest.approx(0.3)
        assert prox["closest_zone"] == "intimate"
        assert prox["num_in_personal"] == 1
        assert len(prox["readings"]) == 1


# -------------------------------------------------------------------
# Norm Rules
# -------------------------------------------------------------------
class TestNormRules:
    def _make_pb(self, closest_distance):
        return PerceptBundle(
            t=0,
            world={},
            social={
                "proxemics": {
                    "closest_distance": closest_distance,
                    "closest_zone": "intimate" if closest_distance < 0.45 else "personal",
                    "readings": [],
                    "num_in_personal": 1,
                }
            },
        )

    def test_personal_space_veto(self):
        rule = PersonalSpaceRule(min_distance=0.5, veto_distance=0.3)
        pb = self._make_pb(0.2)
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert result.veto is not None

    def test_personal_space_hard_constraint(self):
        rule = PersonalSpaceRule(min_distance=0.5, veto_distance=0.3)
        pb = self._make_pb(0.4)
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert result.veto is None
        assert "min_agent_distance" in result.hard

    def test_personal_space_no_constraint(self):
        rule = PersonalSpaceRule(min_distance=0.5, veto_distance=0.3)
        pb = self._make_pb(2.0)
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert result.veto is None
        assert len(result.hard) == 0

    def test_speed_limit(self):
        rule = SpeedLimitRule(max_speed=0.3)
        pb = PerceptBundle(t=0, social={})
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert result.hard["max_linear_speed"] == 0.3


# -------------------------------------------------------------------
# Norm Engine (merging)
# -------------------------------------------------------------------
class TestNormEngine:
    def test_merge_most_restrictive(self):
        engine = NormEngine(rules=[
            SpeedLimitRule(max_speed=0.5),
            SpeedLimitRule(max_speed=0.3),
        ])
        pb = PerceptBundle(t=0, social={})
        result = engine.evaluate(pb, SkillRequest(skill="navigate"))
        assert result.hard["max_linear_speed"] == pytest.approx(0.3)

    def test_first_veto_wins(self):
        engine = NormEngine(rules=[
            PersonalSpaceRule(min_distance=0.5, veto_distance=0.3),
            SpeedLimitRule(max_speed=0.5),
        ])
        pb = PerceptBundle(
            t=0,
            social={"proxemics": {"closest_distance": 0.2}},
        )
        result = engine.evaluate(pb, SkillRequest(skill="navigate"))
        assert result.veto is not None
        # Speed limit still merged
        assert "max_linear_speed" in result.hard


# -------------------------------------------------------------------
# Safety Shield
# -------------------------------------------------------------------
class TestSafetyShield:
    def test_emergency_stop(self):
        shield = SafetyShield(emergency_stop_distance=0.3)
        pb = PerceptBundle(
            t=0,
            social={"proxemics": {"closest_distance": 0.2}},
        )
        update = SkillUpdate(intent="approach", params={"speed_scale": 1.0})
        norms = NormativeConstraints()
        result = shield.apply(pb, SkillRequest(skill="nav"), update, norms)
        assert result.intent == "wait"
        assert result.params["speed_scale"] == 0.0
        assert result.constraints.get("emergency_stop") is True

    def test_speed_cap(self):
        shield = SafetyShield(emergency_stop_distance=0.3)
        pb = PerceptBundle(
            t=0,
            social={"proxemics": {"closest_distance": 5.0}},
        )
        update = SkillUpdate(intent="approach", params={"speed_scale": 1.0})
        norms = NormativeConstraints(hard={"max_linear_speed": 0.4})
        result = shield.apply(pb, SkillRequest(skill="nav"), update, norms)
        assert result.params["speed_scale"] == pytest.approx(0.4)

    def test_passthrough_when_safe(self):
        shield = SafetyShield(emergency_stop_distance=0.3)
        pb = PerceptBundle(
            t=0,
            social={"proxemics": {"closest_distance": 10.0}},
        )
        update = SkillUpdate(intent="approach", params={"speed_scale": 0.8})
        norms = NormativeConstraints()
        result = shield.apply(pb, SkillRequest(skill="nav"), update, norms)
        assert result.params["speed_scale"] == pytest.approx(0.8)


# -------------------------------------------------------------------
# Script Manager
# -------------------------------------------------------------------
class TestScriptManager:
    def test_default_fallback(self):
        mgr = ScriptManager()
        pb = PerceptBundle(t=0)
        req = mgr.select(pb, None)
        assert req.skill == "navigate"

    def test_sequence_advancement(self):
        seq = ScriptSequence(
            name="test",
            steps=[
                ScriptStep(SkillRequest(skill="nav", goal={"x": 1})),
                ScriptStep(SkillRequest(skill="nav", goal={"x": 2})),
                ScriptStep(SkillRequest(skill="nav", goal={"x": 3})),
            ],
        )
        mgr = ScriptManager(sequence=seq)
        pb = PerceptBundle(t=0)

        assert mgr.select(pb, None).goal == {"x": 1}
        mgr.notify_status("SUCCESS")
        assert mgr.select(pb, None).goal == {"x": 2}
        mgr.notify_status("SUCCESS")
        assert mgr.select(pb, None).goal == {"x": 3}

    def test_sequence_loop(self):
        seq = ScriptSequence(
            name="loop",
            steps=[
                ScriptStep(SkillRequest(skill="nav", goal={"x": 1})),
                ScriptStep(SkillRequest(skill="nav", goal={"x": 2})),
            ],
            loop=True,
        )
        mgr = ScriptManager(sequence=seq)
        pb = PerceptBundle(t=0)

        assert mgr.select(pb, None).goal == {"x": 1}
        mgr.notify_status("SUCCESS")
        assert mgr.select(pb, None).goal == {"x": 2}
        mgr.notify_status("SUCCESS")
        # Should loop back
        assert mgr.select(pb, None).goal == {"x": 1}

    def test_is_complete(self):
        seq = ScriptSequence(
            name="done",
            steps=[ScriptStep(SkillRequest(skill="nav", goal={"x": 1}))],
        )
        mgr = ScriptManager(sequence=seq)
        assert mgr.is_complete is False
        mgr.notify_status("SUCCESS")
        assert mgr.is_complete is True
