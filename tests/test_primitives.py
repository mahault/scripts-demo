"""Tests for script primitives and repertoire types."""

from __future__ import annotations

import math

import pytest

from architecture_core.core.types import AffectState, SkillRequest
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    ScriptPrimitive,
    ScriptTrajectory,
    TrajectoryStep,
    TransitionEntry,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary


# =====================================================================
# RepertoireTypes — dataclasses
# =====================================================================
class TestScriptPrimitive:
    def test_default_fields(self):
        p = ScriptPrimitive(name="test", skill_template=SkillRequest(skill="nav"))
        assert p.name == "test"
        assert p.precondition_situations == []
        assert p.postcondition_situation == ""
        assert p.expected_affect is None
        assert p.typical_duration_s == 5.0
        assert p.deontic_default == "permitted"

    def test_custom_fields(self):
        p = ScriptPrimitive(
            name="yield-pass",
            skill_template=SkillRequest(skill="navigate"),
            precondition_situations=["corridor"],
            postcondition_situation="open_area",
            expected_affect=AffectState(valence=0.1, arousal=-0.1),
            typical_duration_s=3.0,
            deontic_default="obligatory",
        )
        assert p.postcondition_situation == "open_area"
        assert p.deontic_default == "obligatory"


class TestTrajectoryStep:
    def test_defaults(self):
        ts = TrajectoryStep(t=1.0, primitive_name="approach-greet")
        assert ts.outcome == "RUNNING"
        assert ts.violation_kl == 0.0
        assert ts.human_intent == "neutral"


class TestScriptTrajectory:
    def test_empty_trajectory(self):
        traj = ScriptTrajectory(script_name="test")
        assert traj.step_count == 0
        assert traj.success_rate == 0.0

    def test_success_rate(self):
        steps = [
            TrajectoryStep(t=0, primitive_name="a", outcome="SUCCESS"),
            TrajectoryStep(t=1, primitive_name="b", outcome="FAILURE"),
            TrajectoryStep(t=2, primitive_name="c", outcome="SUCCESS"),
        ]
        traj = ScriptTrajectory(script_name="test", steps=steps)
        assert traj.step_count == 3
        assert traj.success_rate == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_all_success(self):
        steps = [
            TrajectoryStep(t=i, primitive_name="a", outcome="SUCCESS")
            for i in range(5)
        ]
        traj = ScriptTrajectory(script_name="test", steps=steps)
        assert traj.success_rate == 1.0


class TestTransitionEntry:
    def test_zero_total(self):
        te = TransitionEntry()
        assert te.probability == 0.0

    def test_probability(self):
        te = TransitionEntry(count=3, total_from=10)
        assert te.probability == pytest.approx(0.3)


class TestScriptPattern:
    def test_default_weak(self):
        sp = ScriptPattern(name="test", precision=0.1)
        assert sp.is_strong is False
        assert sp.reliability < 0.5  # Low precision → low reliability

    def test_high_precision_high_reliability(self):
        sp = ScriptPattern(name="test", precision=3.0)
        assert sp.reliability > 0.9

    def test_reliability_bounded(self):
        sp_low = ScriptPattern(name="lo", precision=0.01)
        sp_high = ScriptPattern(name="hi", precision=100.0)
        assert 0.0 <= sp_low.reliability <= 1.0
        assert 0.0 <= sp_high.reliability <= 1.0

    def test_reliability_monotonic(self):
        r_prev = 0.0
        for p in [0.1, 0.5, 1.0, 2.0, 5.0]:
            sp = ScriptPattern(name="t", precision=p)
            assert sp.reliability >= r_prev
            r_prev = sp.reliability


class TestRepertoireConfig:
    def test_defaults(self):
        cfg = RepertoireConfig()
        assert cfg.strong_precision_threshold == 2.0
        assert cfg.max_composition_length == 6
        assert cfg.n_particles == 50


# =====================================================================
# PrimitiveLibrary
# =====================================================================
class TestPrimitiveLibrary:
    def test_default_primitives_registered(self):
        lib = PrimitiveLibrary()
        assert len(lib) == 13
        assert "approach-greet" in lib.names
        assert "yield-pass" in lib.names
        assert "handover-extend" in lib.names

    def test_get_existing(self):
        lib = PrimitiveLibrary()
        p = lib.get("yield-pass")
        assert p is not None
        assert p.name == "yield-pass"

    def test_get_missing(self):
        lib = PrimitiveLibrary()
        assert lib.get("nonexistent") is None

    def test_register_custom(self):
        lib = PrimitiveLibrary(register_defaults=False)
        assert len(lib) == 0
        lib.register(ScriptPrimitive(
            name="custom",
            skill_template=SkillRequest(skill="custom_skill"),
        ))
        assert len(lib) == 1
        assert lib.get("custom") is not None

    def test_get_applicable_filters_by_situation(self):
        lib = PrimitiveLibrary()
        applicable = lib.get_applicable("corridor_encounter")
        names = [p.name for p in applicable]
        assert "yield-pass" in names
        assert "wait-acknowledge" in names
        assert "avoid-reroute" in names

    def test_get_applicable_empty_preconditions_always_match(self):
        lib = PrimitiveLibrary(register_defaults=False)
        lib.register(ScriptPrimitive(
            name="universal",
            skill_template=SkillRequest(skill="any"),
            precondition_situations=[],  # empty = always applicable
        ))
        applicable = lib.get_applicable("any_situation")
        assert len(applicable) == 1

    def test_all_primitives(self):
        lib = PrimitiveLibrary()
        assert len(lib.all_primitives) == 13

    def test_no_defaults(self):
        lib = PrimitiveLibrary(register_defaults=False)
        assert len(lib) == 0

    def test_overwrite_on_register(self):
        lib = PrimitiveLibrary(register_defaults=False)
        lib.register(ScriptPrimitive(
            name="test", skill_template=SkillRequest(skill="v1"),
        ))
        lib.register(ScriptPrimitive(
            name="test", skill_template=SkillRequest(skill="v2"),
        ))
        assert len(lib) == 1
        assert lib.get("test").skill_template.skill == "v2"


# =====================================================================
# ScriptStep backward compatibility
# =====================================================================
class TestScriptStepPrimitiveName:
    def test_default_none(self):
        from architecture_core.cognition.scripts.script_types import ScriptStep
        step = ScriptStep(request=SkillRequest(skill="nav"))
        assert step.primitive_name is None

    def test_set_primitive_name(self):
        from architecture_core.cognition.scripts.script_types import ScriptStep
        step = ScriptStep(
            request=SkillRequest(skill="nav"),
            primitive_name="yield-pass",
        )
        assert step.primitive_name == "yield-pass"
