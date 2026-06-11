#!/usr/bin/env python3
"""Verify demo learning with seed pattern."""

import sys
sys.path.insert(0, "src")

from architecture_core.core.types import AffectState, PerceptBundle, SkillRequest
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer, SituationType
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.learning_script_manager import LearningScriptManager

from plugins.tiago_webots.retail_primitives import (
    register_retail_primitives,
    make_retail_fragments,
    make_retail_repertoire_config,
    make_restock_seed_pattern,
)


class DemoLearningScriptManager(LearningScriptManager):
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
        if seq is not None and seq.loop and status in ("SUCCESS", "FAILURE", "TIMEOUT"):
            idx_before = self._base._current_idx
            step = seq.steps[idx_before] if idx_before < len(seq.steps) else None
            if step is not None and status == step.transition_on:
                if idx_before >= len(seq.steps) - 1:
                    wrapped = True
        super().notify_status(status)
        if wrapped and self._repertoire is not None:
            self._repertoire.on_script_completed(status)
            self._current_pattern_name = None


def simulate_loop(learner, repertoire, recognizer, loop_idx):
    t = loop_idx * 20.0
    pb = PerceptBundle(t=t, world={"robot_pose": (-5.5, -1.0, 0, 0), "agents": []})
    belief = recognizer.recognize(pb)
    learner.select(pb)
    for step_idx in range(4):
        learner.set_last_step_info(
            primitive_name="navigate-to-stock" if step_idx == 0 else
                         "navigate-to-shelf" if step_idx in (1, 2) else
                         "navigate-to-counter",
            affect=AffectState(valence=0.0, arousal=0.0)
        )
        learner.notify_status("SUCCESS")
    learner.notify_status("SUCCESS")

    patterns = repertoire.patterns
    strong = [p for p in patterns.values() if p.is_strong]
    best = max(patterns.values(), key=lambda p: p.precision) if patterns else None
    print(f"Loop {loop_idx+1}: {len(patterns)} patterns, {len(strong)} strong, "
          f"best={best.name if best else 'none'} prec={best.precision:.2f} trajs={best.trajectory_count if best else 0}")
    return len(strong) > 0


print("=" * 60)
print("Demo Learning Test: WITH seed pattern")
print("=" * 60)

library = PrimitiveLibrary(register_defaults=True)
register_retail_primitives(library)

config = make_retail_repertoire_config()
config.trajectory_match_threshold = 0.1

seed = make_restock_seed_pattern()
print(f"Seed pattern: {seed.name} prec={seed.precision} trajs={seed.trajectory_count} "
      f"mean_fe={seed.mean_free_energy}")

repertoire = ScriptRepertoire(
    library,
    config=config,
    initial_patterns=make_retail_fragments() + [seed],
)

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
        ScriptStep(SkillRequest(skill="navigate", goal={"x": -5.5, "y": -1.0}), expected_situation="stock_zone"),
        ScriptStep(SkillRequest(skill="navigate", goal={"x": -2.0, "y": -5.0}), expected_situation="shelf_zone"),
        ScriptStep(SkillRequest(skill="navigate", goal={"x": 1.5, "y": -5.0}), expected_situation="shelf_zone"),
        ScriptStep(SkillRequest(skill="navigate", goal={"x": 4.5, "y": -7.5}), expected_situation="counter_zone"),
    ],
    loop=True,
)

base_mgr = ScriptManager(sequence=seq, recognizer=recognizer)
learner = DemoLearningScriptManager(base_mgr, repertoire)

crystallized = False
for i in range(6):
    if simulate_loop(learner, repertoire, recognizer, i):
        crystallized = True
        print(f"  >>> CRYSTALLIZED at loop {i+1}! <<<")
        break

print("\n--- Final patterns ---")
for name, p in sorted(repertoire.patterns.items(), key=lambda x: -x[1].precision):
    seq_str = " -> ".join(p.primitives_sequence[:6]) if p.primitives_sequence else "(cluster)"
    print(f"  {name}: prec={p.precision:.2f} strong={p.is_strong} trajs={p.trajectory_count} seq=[{seq_str}]")

if crystallized:
    strong_pat = [p for p in repertoire.patterns.values() if p.is_strong][0]
    converted_seq = repertoire.pattern_to_sequence(strong_pat, context="retail")
    print(f"\n--- Crystallized sequence ---")
    print(f"  Steps: {[s.request.skill + ':' + str(s.request.goal) for s in converted_seq.steps]}")

print("\n" + "=" * 60)
if crystallized:
    print("SUCCESS: Pattern crystallized and produces a valid sequence!")
else:
    print("No crystallization in 6 loops.")
print("=" * 60)
