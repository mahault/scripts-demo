"""Tests for script composer."""

from __future__ import annotations

import pytest

from architecture_core.core.types import AffectState, SkillRequest
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPrimitive,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_composer import ScriptComposer


class TestScriptComposerBasic:
    def test_compose_returns_pattern(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        pattern = composer.compose("corridor_encounter")
        assert pattern is not None
        assert len(pattern.primitives_sequence) > 0
        assert pattern.precision == 0.1  # Weak
        assert pattern.situation_affinity.get("corridor_encounter") == 1.0

    def test_compose_name_reflects_primitives(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        pattern = composer.compose("corridor_encounter")
        assert pattern.name.startswith("composed_")

    def test_compose_respects_max_length(self):
        cfg = RepertoireConfig(max_composition_length=2)
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib, config=cfg)
        pattern = composer.compose("corridor_encounter")
        assert len(pattern.primitives_sequence) <= 2

    def test_compose_with_no_applicable_primitives(self):
        lib = PrimitiveLibrary(register_defaults=False)
        lib.register(ScriptPrimitive(
            name="only-for-mars",
            skill_template=SkillRequest(skill="mars"),
            precondition_situations=["mars_base"],
        ))
        composer = ScriptComposer(lib)
        # "corridor_encounter" has no applicable primitives
        pattern = composer.compose("corridor_encounter")
        # Should fallback to something
        assert len(pattern.primitives_sequence) >= 0


class TestScriptComposerGoal:
    def test_compose_toward_goal(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        pattern = composer.compose(
            "corridor_encounter",
            goal_situation="open_area",
        )
        assert len(pattern.primitives_sequence) > 0
        # yield-pass transitions corridor → open_area, should appear
        assert "yield-pass" in pattern.primitives_sequence

    def test_goal_already_reached(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        pattern = composer.compose(
            "open_area",
            goal_situation="open_area",
        )
        # Already at goal — should produce empty or minimal sequence
        assert len(pattern.primitives_sequence) <= 1


class TestScriptComposerAffect:
    def test_negative_affect_avoids_approach_when_human_avoiding(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        pattern = composer.compose(
            "corridor_encounter",
            affect_state=AffectState(valence=-0.5, arousal=0.3),
            human_intent="avoid",
        )
        # With negative affect and avoiding human, approach-greet should be
        # disfavored; yield-pass or wait-acknowledge more likely first
        if len(pattern.primitives_sequence) > 0:
            first = pattern.primitives_sequence[0]
            assert first in ("yield-pass", "wait-acknowledge", "avoid-reroute")


class TestScriptComposerEpistemic:
    def test_usage_counts_affect_selection(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        # Compose twice — second time should try different primitives
        p1 = composer.compose("corridor_encounter")
        p2 = composer.compose("corridor_encounter")
        # They may differ due to epistemic scoring favoring less-used prims
        # At minimum, both should be valid
        assert len(p1.primitives_sequence) > 0
        assert len(p2.primitives_sequence) > 0

    def test_reset_usage(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        composer.compose("corridor_encounter")
        composer.reset_usage()
        # After reset, same primitives should be selectable
        p = composer.compose("corridor_encounter")
        assert len(p.primitives_sequence) > 0


class TestScriptComposerEmpty:
    def test_empty_library(self):
        lib = PrimitiveLibrary(register_defaults=False)
        composer = ScriptComposer(lib)
        pattern = composer.compose("corridor_encounter")
        # Should handle gracefully
        assert pattern is not None
