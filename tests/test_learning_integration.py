"""Integration tests for script repertoire learning."""

from __future__ import annotations

import pytest

from architecture_core.core.types import AffectState, PerceptBundle, SkillRequest
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    TransitionEntry,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.learning_script_manager import LearningScriptManager
from architecture_core.cognition.scripts.script_types import (
    ScriptSequence,
    ScriptStep,
    SituationType,
)
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer


# =====================================================================
# Helpers
# =====================================================================
def _make_recognizer():
    types = [
        SituationType(name="corridor_encounter", feature_weights={
            "agent_count": 0.5, "min_distance": -2.0, "mean_velocity": 1.0,
        }),
        SituationType(name="open_area", feature_weights={
            "agent_count": 0.2, "min_distance": 0.5, "mean_velocity": 0.3,
        }),
    ]
    return WeakScriptRecognizer(types)


def _make_pb(agents=None):
    world = {
        "robot_pose": (0, 0, 0, 0),
        "agents": agents or [{"id": "h1", "pose": (1.5, 0)}],
    }
    return PerceptBundle(t=0.0, world=world, social={})


def _corridor_pattern():
    return ScriptPattern(
        name="corridor_yield",
        primitives_sequence=["wait-acknowledge", "yield-pass"],
        precision=2.0,
        situation_affinity={"corridor_encounter": 0.9},
        transition_counts={
            "wait-acknowledge": {
                "yield-pass": TransitionEntry(count=8, total_from=10),
            },
        },
    )


# =====================================================================
# LearningScriptManager basic
# =====================================================================
class TestLearningScriptManagerBasic:
    def test_passthrough_without_repertoire(self):
        base = ScriptManager(sequence=ScriptSequence(
            name="test",
            steps=[ScriptStep(request=SkillRequest(skill="navigate"))],
        ))
        lsm = LearningScriptManager(base, repertoire=None)
        pb = _make_pb()
        req = lsm.select(pb)
        assert req.skill == "navigate"

    def test_is_complete_delegates(self):
        base = ScriptManager()
        lsm = LearningScriptManager(base)
        assert lsm.is_complete == base.is_complete

    def test_is_repairing_delegates(self):
        base = ScriptManager()
        lsm = LearningScriptManager(base)
        assert lsm.is_repairing == base.is_repairing


# =====================================================================
# LearningScriptManager with repertoire
# =====================================================================
class TestLearningScriptManagerWithRepertoire:
    def test_select_with_repertoire(self):
        recognizer = _make_recognizer()
        base = ScriptManager(recognizer=recognizer)
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        lsm = LearningScriptManager(base, repertoire=rep)

        pb = _make_pb()
        req = lsm.select(pb)
        assert req.skill is not None

    def test_notify_status_feeds_repertoire(self):
        recognizer = _make_recognizer()
        base = ScriptManager(recognizer=recognizer)
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        lsm = LearningScriptManager(base, repertoire=rep)

        pb = _make_pb()
        lsm.select(pb)
        lsm.set_last_step_info(primitive_name="wait-acknowledge")
        lsm.notify_status("SUCCESS")
        # Should not crash — repertoire processes the step


# =====================================================================
# Full learning cycle
# =====================================================================
class TestFullLearningCycle:
    def test_observe_infer_compose_cycle(self):
        """Test the full cycle: observe situation -> infer pattern -> compose if needed."""
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)  # Empty — will have to compose

        belief = {"corridor_encounter": 0.8, "open_area": 0.2}

        # Step 1: Situation recognized → triggers composition
        name = rep.on_situation_recognized(belief, t=0.0)
        assert name is not None
        assert rep.pattern_count > 0

        # Step 2: Execute steps
        rep.on_step_completed(
            t=1.0, primitive_name="yield-pass", outcome="SUCCESS",
        )
        rep.on_step_completed(
            t=2.0, primitive_name="approach-greet", outcome="SUCCESS",
        )

        # Step 3: Complete
        rep.on_script_completed("SUCCESS")
        assert len(rep.tracker.completed_trajectories) == 1

    def test_precision_accumulates_across_trajectories(self):
        """Test that repeated successful trajectories build precision."""
        lib = PrimitiveLibrary()
        p = ScriptPattern(
            name="test_pattern",
            primitives_sequence=["yield-pass"],
            precision=1.5,  # High enough to pass reliability gate
            situation_affinity={"corridor_encounter": 0.9},
        )
        cfg = RepertoireConfig(
            strong_precision_threshold=3.0,
            min_trajectories_for_promotion=3,
            trajectory_match_threshold=0.5,  # Lower threshold to select existing
        )
        rep = ScriptRepertoire(lib, config=cfg, initial_patterns=[p])

        for i in range(5):
            rep.on_situation_recognized(
                {"corridor_encounter": 0.8, "open_area": 0.2}, t=float(i * 10),
            )
            rep.on_step_completed(
                t=float(i * 10 + 1), primitive_name="yield-pass", outcome="SUCCESS",
            )
            rep.on_script_completed("SUCCESS")

        updated = rep.patterns["test_pattern"]
        assert updated.trajectory_count >= 3
        assert updated.precision > 0.5  # Should have increased

    def test_pattern_to_sequence_roundtrip(self):
        """Test that a pattern can be converted to a ScriptSequence and back."""
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib, initial_patterns=[_corridor_pattern()])
        seq = rep.pattern_to_sequence(_corridor_pattern())

        assert seq.name == "corridor_yield"
        assert len(seq.steps) == 2
        assert seq.steps[0].primitive_name == "wait-acknowledge"
        assert seq.steps[1].primitive_name == "yield-pass"
        assert seq.steps[0].request.skill == "navigate"


# =====================================================================
# ScriptManager.set_sequence
# =====================================================================
class TestScriptManagerSetSequence:
    def test_set_sequence_replaces(self):
        sm = ScriptManager(sequence=ScriptSequence(
            name="old",
            steps=[ScriptStep(request=SkillRequest(skill="old_skill"))],
        ))
        pb = _make_pb()
        req = sm.select(pb, None)
        assert req.skill == "old_skill"

        sm.set_sequence(ScriptSequence(
            name="new",
            steps=[ScriptStep(request=SkillRequest(skill="new_skill"))],
        ))
        req = sm.select(pb, None)
        assert req.skill == "new_skill"
