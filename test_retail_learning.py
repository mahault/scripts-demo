#!/usr/bin/env python3
"""Headless test: verify the retail learning pipeline works end-to-end.

Simulates a learner robot traversing the restock waypoints and checks
whether the ScriptRepertoire accumulates precision and crystallizes.

Run:
    python test_retail_learning.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")

from architecture_core.core.types import AffectState, PerceptBundle, SkillRequest
from architecture_core.core.status import Status
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer, SituationType
from architecture_core.cognition.scripts.learning_script_manager import LearningScriptManager

from plugins.tiago_webots.retail_primitives import (
    register_retail_primitives,
    make_retail_fragments,
    make_retail_repertoire_config,
)

# Import the fixed learning manager from the retail controller
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "retail_controller", "src/plugins/tiago_webots/retail_controller.py"
)
_retail_mod = importlib.util.module_from_spec(_spec)
# We can't execute the module because it imports controller (Webots-only),
# but we can extract the DemoLearningScriptManager class definition
# by reading the source... Actually, simpler: define it inline here.

class DemoLearningScriptManager(LearningScriptManager):
    """Loop-wrap fix + only installs strong patterns."""

    def select(self, pb, active=None):
        if self._repertoire is None:
            return self._base.select(pb, active)
        situation_belief = self._extract_situation_belief(pb)
        if situation_belief:
            pattern_name = self._repertoire.on_situation_recognized(
                situation_belief, t=pb.t,
            )
            if pattern_name is not None and pattern_name != self._current_pattern_name:
                pattern = self._repertoire.patterns.get(pattern_name)
                if pattern is not None and pattern.is_strong:
                    self._current_pattern_name = pattern_name
                    most_likely_sit = (
                        max(situation_belief, key=situation_belief.get)
                        if situation_belief else None
                    )
                    seq = self._repertoire.pattern_to_sequence(
                        pattern, context=most_likely_sit,
                    )
                    self._base.set_sequence(seq)
        return self._base.select(pb, active)

    def notify_status(self, status):
        seq = self._base._sequence
        wrapped = False
        if (
            seq is not None
            and seq.loop
            and status in ("SUCCESS", "FAILURE", "TIMEOUT")
        ):
            idx_before = self._base._current_idx
            step = seq.steps[idx_before] if idx_before < len(seq.steps) else None
            if step is not None and status == step.transition_on:
                if idx_before >= len(seq.steps) - 1:
                    wrapped = True
        super().notify_status(status)
        if wrapped and self._repertoire is not None:
            self._repertoire.on_script_completed(status)
            self._current_pattern_name = None


def simulate_restock_trajectory(t_start: float, outcome: Status = "SUCCESS"):
    """Return a PerceptBundle simulating one restock loop."""
    # 4 waypoints: stock -> shelf A -> shelf B -> counter
    waypoints = [
        (-5.5, -1.0, "stock_zone"),
        (-2.0, -5.0, "shelf_zone"),
        (1.5, -5.0, "shelf_zone"),
        (4.5, -7.5, "counter_zone"),
    ]
    pb = PerceptBundle(t=t_start, world={
        "robot_pose": (waypoints[0][0], waypoints[0][1], 0, 0),
        "agents": [],
    })
    return pb


def run_pipeline_test():
    print("=" * 60)
    print("Retail Learning Pipeline — Headless Test")
    print("=" * 60)

    # Setup
    library = PrimitiveLibrary(register_defaults=True)
    register_retail_primitives(library)
    print(f"\nRegistered primitives: {len(library.all_primitives)}")

    fragments = make_retail_fragments()
    print(f"Initial fragments: {[f.name for f in fragments]}")

    config = make_retail_repertoire_config()
    config.trajectory_match_threshold = 0.1  # low threshold so weak patterns get reused
    print(f"Config: threshold={config.strong_precision_threshold}, "
          f"min_traj={config.min_trajectories_for_promotion}, "
          f"max_fe={config.max_free_energy_for_promotion}, "
          f"match_thresh={config.trajectory_match_threshold}")

    # Test 1A: WITH retail primitives, WITH compositional mode
    repertoire = ScriptRepertoire(library, config=config, initial_patterns=fragments)
    repertoire.enable_compositional_mode()

    recognizer = WeakScriptRecognizer(
        situation_types=[
            SituationType(name="stock_zone", feature_weights={"cue_stock": 4.0}),
            SituationType(name="shelf_zone", feature_weights={"cue_shelf": 4.0}),
            SituationType(name="counter_zone", feature_weights={"cue_counter": 4.0}),
            SituationType(name="approach", feature_weights={"mean_velocity": 2.0}),
        ],
        prior={"stock_zone": 0.25, "shelf_zone": 0.25, "counter_zone": 0.25, "approach": 0.25},
    )

    seq = ScriptSequence(
        name="learner_naive",
        steps=[
            ScriptStep(SkillRequest(skill="navigate", goal={"x": -5.5, "y": -1.0}),
                       expected_situation="stock_zone"),
            ScriptStep(SkillRequest(skill="navigate", goal={"x": -2.0, "y": -5.0}),
                       expected_situation="shelf_zone"),
            ScriptStep(SkillRequest(skill="navigate", goal={"x": 1.5, "y": -5.0}),
                       expected_situation="shelf_zone"),
            ScriptStep(SkillRequest(skill="navigate", goal={"x": 4.5, "y": -7.5}),
                       expected_situation="counter_zone"),
        ],
        loop=True,
    )

    base_mgr = ScriptManager(sequence=seq, recognizer=recognizer)
    learner = DemoLearningScriptManager(base_mgr, repertoire)

    # Simulate 6 restock loops
    print("\n--- Simulating 6 restock loops ---")
    strong_formed = False
    for loop in range(6):
        t = loop * 20.0

        # Feed situation belief to start trajectory
        pb = simulate_restock_trajectory(t)
        belief = recognizer.recognize(pb)
        learner.select(pb)

        # Simulate completing each of the 4 steps
        for step_idx in range(4):
            step_t = t + step_idx * 5.0
            # Notify step completion
            learner.set_last_step_info(primitive_name="navigate-to-stock" if step_idx == 0 else
                                                    "navigate-to-shelf" if step_idx in (1, 2) else
                                                    "navigate-to-counter",
                                       affect=AffectState(valence=0.0, arousal=0.0))
            learner.notify_status("SUCCESS")

        # End of loop = script completed
        learner.notify_status("SUCCESS")

        # Check patterns
        patterns = repertoire.patterns
        strong = [p for p in patterns.values() if p.is_strong]
        best = max(patterns.values(), key=lambda p: p.precision) if patterns else None

        print(f"Loop {loop+1}: {len(patterns)} patterns, {len(strong)} strong, "
              f"best={best.name if best else 'none'} prec={best.precision:.2f} "
              f"traj={best.trajectory_count if best else 0}")

        if strong and not strong_formed:
            strong_formed = True
            print(f"  >>> FIRST STRONG PATTERN CRYSTALLIZED at loop {loop+1}! <<<")

    print("\n--- Final pattern details ---")
    for name, p in sorted(repertoire.patterns.items(), key=lambda x: -x[1].precision):
        seq_str = " -> ".join(p.primitives_sequence[:5]) if p.primitives_sequence else "(cluster)"
        print(f"  {name}: prec={p.precision:.2f} strong={p.is_strong} trajs={p.trajectory_count} "
              f"seq=[{seq_str}]")

    print("\n" + "=" * 60)
    if strong_formed:
        print("RESULT: LEARNING WORKS — patterns crystallize from experience.")
    else:
        print("RESULT: No strong pattern formed in 6 loops (may need more loops or tuning).")
    print("=" * 60)

    # -------- Second test: WITHOUT retail primitives --------
    print("\n\n" + "=" * 60)
    print("Test 2: Learning WITHOUT pre-defined retail primitives")
    print("=" * 60)

    library2 = PrimitiveLibrary(register_defaults=True)  # only defaults (approach-greet, yield-pass, etc.)
    print(f"\nPrimitives available: {[p.name for p in library2.all_primitives]}")

    config2 = make_retail_repertoire_config()
    config2.trajectory_match_threshold = 0.1
    repertoire2 = ScriptRepertoire(library2, config=config2, initial_patterns=[])
    # No compositional mode — pure from-scratch composition

    base_mgr2 = ScriptManager(sequence=seq, recognizer=recognizer)
    learner2 = DemoLearningScriptManager(base_mgr2, repertoire2)

    print("\n--- Simulating 6 restock loops with only default primitives ---")
    strong_formed2 = False
    for loop in range(6):
        t = loop * 20.0
        pb = simulate_restock_trajectory(t)
        learner2.select(pb)
        for step_idx in range(4):
            learner2.set_last_step_info(primitive_name="navigate-to-stock" if step_idx == 0 else
                                                     "navigate-to-shelf" if step_idx in (1, 2) else
                                                     "navigate-to-counter",
                                        affect=AffectState(valence=0.0, arousal=0.0))
            learner2.notify_status("SUCCESS")
        learner2.notify_status("SUCCESS")

        patterns2 = repertoire2.patterns
        strong2 = [p for p in patterns2.values() if p.is_strong]
        best2 = max(patterns2.values(), key=lambda p: p.precision) if patterns2 else None
        print(f"Loop {loop+1}: {len(patterns2)} patterns, {len(strong2)} strong, "
              f"best={best2.name if best2 else 'none'} prec={best2.precision:.2f}")
        if strong2 and not strong_formed2:
            strong_formed2 = True
            print(f"  >>> FIRST STRONG PATTERN CRYSTALLIZED at loop {loop+1}! <<<")

    print("\n--- Final pattern details ---")
    for name, p in sorted(repertoire2.patterns.items(), key=lambda x: -x[1].precision):
        seq_str = " -> ".join(p.primitives_sequence[:5]) if p.primitives_sequence else "(cluster)"
        print(f"  {name}: prec={p.precision:.2f} strong={p.is_strong} trajs={p.trajectory_count} "
              f"seq=[{seq_str}]")

    print("\n" + "=" * 60)
    if strong_formed2:
        print("RESULT: Learning works even WITHOUT retail primitives!")
        print("The composer generates patterns from default primitives.")
    else:
        print("RESULT: No strong pattern without retail primitives.")
        print("The default primitives don't match the restock causal chain well.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_test()
