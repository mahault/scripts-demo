"""Tests for script repertoire."""

from __future__ import annotations

import pytest

from architecture_core.core.types import AffectState
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    TransitionEntry,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire


# =====================================================================
# Helpers
# =====================================================================
def _corridor_pattern() -> ScriptPattern:
    return ScriptPattern(
        name="corridor_yield",
        primitives_sequence=["wait-acknowledge", "yield-pass", "approach-greet"],
        precision=1.5,
        situation_affinity={"corridor_encounter": 0.9},
        transition_counts={
            "wait-acknowledge": {
                "yield-pass": TransitionEntry(count=8, total_from=10),
            },
            "yield-pass": {
                "approach-greet": TransitionEntry(count=7, total_from=10),
            },
        },
    )


def _belief(main: str = "corridor_encounter") -> dict:
    return {main: 0.8, "open_area": 0.2}


# =====================================================================
# Basic lifecycle
# =====================================================================
class TestScriptRepertoireBasic:
    def test_empty_repertoire(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        assert rep.pattern_count == 0
        assert rep.strong_count == 0

    def test_initial_patterns(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        assert rep.pattern_count == 1
        assert "corridor_yield" in rep.patterns

    def test_get_best_pattern(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        best = rep.get_best_pattern("corridor_encounter")
        assert best is not None
        assert best.name == "corridor_yield"

    def test_get_best_pattern_no_match(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        best = rep.get_best_pattern("mars_base")
        assert best is None


# =====================================================================
# Situation recognition + trajectory start
# =====================================================================
class TestScriptRepertoireSituation:
    def test_on_situation_recognized_starts_trajectory(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        name = rep.on_situation_recognized(_belief(), t=0.0)
        assert name is not None
        assert rep.tracker.is_recording

    def test_on_situation_recognized_selects_existing_pattern(self):
        lib = PrimitiveLibrary()
        p = _corridor_pattern()
        p.precision = 3.0  # High enough to pass reliability threshold
        rep = ScriptRepertoire(lib, initial_patterns=[p])
        name = rep.on_situation_recognized(_belief(), t=0.0)
        assert name == "corridor_yield"

    def test_on_situation_recognized_composes_when_no_match(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)  # No initial patterns
        name = rep.on_situation_recognized(_belief(), t=0.0)
        assert name is not None
        assert rep.pattern_count > 0  # Composed pattern added


# =====================================================================
# Step completion + inference
# =====================================================================
class TestScriptRepertoireSteps:
    def test_on_step_completed_records(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        rep.on_situation_recognized(_belief(), t=0.0)
        rep.on_step_completed(
            t=1.0,
            primitive_name="wait-acknowledge",
            situation_belief=_belief(),
            outcome="SUCCESS",
        )
        assert rep.tracker.current.step_count == 1

    def test_on_step_completed_updates_inference(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        rep.on_situation_recognized(_belief(), t=0.0)
        rep.on_step_completed(
            t=1.0,
            primitive_name="wait-acknowledge",
            outcome="SUCCESS",
        )
        # Inference should have been updated
        assert rep.trajectory_free_energy > 0.0


# =====================================================================
# Script completion + precision update
# =====================================================================
class TestScriptRepertoireCompletion:
    def test_on_script_completed_finalises(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        rep.on_situation_recognized(_belief(), t=0.0)
        rep.on_step_completed(t=1.0, primitive_name="wait-acknowledge", outcome="SUCCESS")
        rep.on_step_completed(t=2.0, primitive_name="yield-pass", outcome="SUCCESS")
        rep.on_script_completed("SUCCESS")
        assert not rep.tracker.is_recording
        assert len(rep.tracker.completed_trajectories) == 1

    def test_precision_increases_on_good_trajectory(self):
        lib = PrimitiveLibrary()
        p = _corridor_pattern()
        initial_precision = p.precision
        rep = ScriptRepertoire(lib, initial_patterns=[p])

        # Run a successful trajectory
        rep.on_situation_recognized(_belief(), t=0.0)
        for i, prim in enumerate(["wait-acknowledge", "yield-pass", "approach-greet"]):
            rep.on_step_completed(t=float(i), primitive_name=prim, outcome="SUCCESS")
        rep.on_script_completed("SUCCESS")

        updated = rep.patterns["corridor_yield"]
        assert updated.trajectory_count >= 1
        # Precision should have changed
        assert updated.precision != initial_precision


# =====================================================================
# Consolidation
# =====================================================================
class TestScriptRepertoireConsolidation:
    def test_pattern_promoted_to_strong(self):
        cfg = RepertoireConfig(
            strong_precision_threshold=2.0,
            min_trajectories_for_promotion=2,
            max_free_energy_for_promotion=5.0,
        )
        lib = PrimitiveLibrary()
        p = ScriptPattern(
            name="test_pattern",
            primitives_sequence=["yield-pass"],
            precision=1.9,
            trajectory_count=1,
            mean_free_energy=1.0,
            situation_affinity={"corridor_encounter": 0.9},
        )
        rep = ScriptRepertoire(lib, config=cfg, initial_patterns=[p])

        # Run two good trajectories to push precision over threshold
        for _ in range(3):
            rep.on_situation_recognized(_belief(), t=0.0)
            rep.on_step_completed(t=1.0, primitive_name="yield-pass", outcome="SUCCESS")
            rep.on_script_completed("SUCCESS")

        updated = rep.patterns["test_pattern"]
        assert updated.trajectory_count >= 2
        # Should eventually become strong
        assert updated.is_strong or updated.precision >= cfg.strong_precision_threshold

    def test_strong_count(self):
        lib = PrimitiveLibrary()
        p1 = ScriptPattern(name="strong1", precision=3.0, is_strong=True)
        p2 = ScriptPattern(name="weak1", precision=0.5, is_strong=False)
        rep = ScriptRepertoire(lib, initial_patterns=[p1, p2])
        assert rep.strong_count == 1


# =====================================================================
# Pattern to sequence conversion
# =====================================================================
class TestPatternToSequence:
    def test_converts_to_script_sequence(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        seq = rep.pattern_to_sequence(_corridor_pattern())
        assert seq.name == "corridor_yield"
        assert len(seq.steps) == 3
        assert seq.steps[0].primitive_name == "wait-acknowledge"
        assert seq.steps[1].primitive_name == "yield-pass"
        assert seq.steps[2].primitive_name == "approach-greet"

    def test_steps_have_correct_skills(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        p = _corridor_pattern()
        seq = rep.pattern_to_sequence(p)
        # wait-acknowledge uses navigate skill
        assert seq.steps[0].request.skill == "navigate"

    def test_steps_are_flexible(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        p = _corridor_pattern()
        seq = rep.pattern_to_sequence(p)
        for step in seq.steps:
            assert step.gate == "flexible"


# =====================================================================
# Decay
# =====================================================================
class TestPrecisionDecay:
    def test_decay_reduces_precision(self):
        lib = PrimitiveLibrary()
        p = ScriptPattern(name="test", precision=1.0)
        rep = ScriptRepertoire(lib, initial_patterns=[p])
        rep.decay_precision(t=1.0)
        assert rep.patterns["test"].precision < 1.0

    def test_decay_respects_floor(self):
        cfg = RepertoireConfig(weak_precision_floor=0.05, precision_decay_rate=10.0)
        lib = PrimitiveLibrary()
        p = ScriptPattern(name="test", precision=0.1)
        rep = ScriptRepertoire(lib, config=cfg, initial_patterns=[p])
        rep.decay_precision(t=1.0)
        assert rep.patterns["test"].precision >= cfg.weak_precision_floor

    def test_strong_scripts_decay_slower(self):
        lib = PrimitiveLibrary()
        p_strong = ScriptPattern(name="strong", precision=2.0, is_strong=True)
        p_weak = ScriptPattern(name="weak", precision=2.0, is_strong=False)
        rep = ScriptRepertoire(lib, initial_patterns=[p_strong, p_weak])
        rep.decay_precision(t=1.0)
        assert rep.patterns["strong"].precision > rep.patterns["weak"].precision
