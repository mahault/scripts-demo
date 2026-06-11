"""Test observational learning from scratch.

The learner starts with NO pre-defined retail primitives and NO seed pattern.
It watches a mock teacher, segments its behavior into primitives, discovers
a pattern from the segmented trajectory, and crystallizes it after 3 loops.
"""

from __future__ import annotations

import json

from architecture_core.core.types import AffectState
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer
from architecture_core.cognition.scripts.script_types import SituationType
from plugins.tiago_webots.retail_primitives import make_retail_repertoire_config
from plugins.tiago_webots.observation import TeacherObserver


# ------------------------------------------------------------------
# Mock teacher that simulates state transitions and position
# ------------------------------------------------------------------
class MockTeacherNode:
    """Mocks a Webots node with translation and customData fields."""

    def __init__(self):
        self._pos = (0.0, 0.0, 0.0)
        self._custom_data = ""

    def getField(self, name):
        return self

    def getSFVec3f(self):
        return self._pos

    def getSFString(self):
        return self._custom_data

    def set_pos(self, x, y):
        self._pos = (x, y, 0.0)

    def set_state(self, waypoint, state, loop=0):
        self._custom_data = json.dumps({"waypoint": waypoint, "state": state, "loop": loop})


class MockRobot:
    """Mocks a Webots Supervisor robot."""

    def __init__(self):
        self._teacher = MockTeacherNode()

    def getFromDef(self, name):
        return self._teacher


# ------------------------------------------------------------------
# Teacher timeline generator
# ------------------------------------------------------------------
WAYPOINT_COORDS = {
    0: (-5.5, -1.0),   # stock
    1: (-2.0, -5.0),   # shelf_a
    2: (1.5, -5.0),    # shelf_b
    3: (4.5, -7.5),    # counter
}


def generate_teacher_timeline(n_loops=3):
    """Generate a sequence of (pos, state_json) tuples simulating teacher behavior.

    Each loop: stock → shelf_a → shelf_b → counter → stock
    At each waypoint: WORK(90) → WORK(45) → WORK(15)
    At counter: WORK(90)=wait → WORK(45)=look-around → WORK(15)=settle
    Navigation: state="NAV(SUCCESS)"
    """
    timeline = []
    for loop in range(n_loops):
        # Stock work
        timeline.append((WAYPOINT_COORDS[0], json.dumps({"waypoint": 0, "state": "WORK(90)", "loop": loop})))
        timeline.append((WAYPOINT_COORDS[0], json.dumps({"waypoint": 0, "state": "WORK(45)", "loop": loop})))
        timeline.append((WAYPOINT_COORDS[0], json.dumps({"waypoint": 0, "state": "WORK(15)", "loop": loop})))
        # Nav to shelf_a
        timeline.append((WAYPOINT_COORDS[1], json.dumps({"waypoint": 1, "state": "NAV(SUCCESS)", "loop": loop})))
        # Shelf_a work
        timeline.append((WAYPOINT_COORDS[1], json.dumps({"waypoint": 1, "state": "WORK(90)", "loop": loop})))
        timeline.append((WAYPOINT_COORDS[1], json.dumps({"waypoint": 1, "state": "WORK(45)", "loop": loop})))
        timeline.append((WAYPOINT_COORDS[1], json.dumps({"waypoint": 1, "state": "WORK(15)", "loop": loop})))
        # Nav to shelf_b
        timeline.append((WAYPOINT_COORDS[2], json.dumps({"waypoint": 2, "state": "NAV(SUCCESS)", "loop": loop})))
        # Shelf_b work
        timeline.append((WAYPOINT_COORDS[2], json.dumps({"waypoint": 2, "state": "WORK(90)", "loop": loop})))
        timeline.append((WAYPOINT_COORDS[2], json.dumps({"waypoint": 2, "state": "WORK(45)", "loop": loop})))
        timeline.append((WAYPOINT_COORDS[2], json.dumps({"waypoint": 2, "state": "WORK(15)", "loop": loop})))
        # Nav to counter
        timeline.append((WAYPOINT_COORDS[3], json.dumps({"waypoint": 3, "state": "NAV(SUCCESS)", "loop": loop})))
        # Counter work
        timeline.append((WAYPOINT_COORDS[3], json.dumps({"waypoint": 3, "state": "WORK(90)", "loop": loop})))
        timeline.append((WAYPOINT_COORDS[3], json.dumps({"waypoint": 3, "state": "WORK(45)", "loop": loop})))
        timeline.append((WAYPOINT_COORDS[3], json.dumps({"waypoint": 3, "state": "WORK(15)", "loop": loop})))
        # Nav back to stock (completes loop)
        timeline.append((WAYPOINT_COORDS[0], json.dumps({"waypoint": 0, "state": "NAV(SUCCESS)", "loop": loop})))
    return timeline


# ------------------------------------------------------------------
# Main test
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Observational Learning FROM SCRATCH")
    print("=" * 60)

    # Setup: empty patterns, default primitives only
    library = PrimitiveLibrary(register_defaults=True)
    config = make_retail_repertoire_config()
    repertoire = ScriptRepertoire(
        library,
        config=config,
        initial_patterns=[],
        observational_mode=True,
    )

    recognizer = WeakScriptRecognizer(
        situation_types=[
            SituationType(name="stock_zone", feature_weights={"cue_stock": 4.0}),
            SituationType(name="shelf_zone", feature_weights={"cue_shelf": 4.0}),
            SituationType(name="counter_zone", feature_weights={"cue_counter": 4.0}),
        ]
    )

    mock = MockRobot()
    observer = TeacherObserver(mock, teacher_name="Worker_T", library=library)

    # Generate 3 loops + 1 extra tick so the final navigate segment can complete
    timeline = generate_teacher_timeline(n_loops=3)
    # Add completion tick for the last loop's navigate segment
    timeline.append((WAYPOINT_COORDS[0], json.dumps({"waypoint": 0, "state": "WORK(90)", "loop": 3})))
    print(f"\n--- Simulating {len(timeline)} ticks of teacher behavior ---\n")

    for t, (pos, state_json) in enumerate(timeline, 1):
        mock._teacher.set_pos(pos[0], pos[1])
        mock._teacher._custom_data = state_json

        status = observer.tick(t=t, repertoire=repertoire, recognizer=recognizer)

        strong = [p for p in repertoire.patterns.values() if p.is_strong]
        best = max(repertoire.patterns.values(), key=lambda p: p.precision) if repertoire.patterns else None
        best_name = best.name if best else "None"
        best_prec = best.precision if best else 0.0

        if status.get("segment_type"):
            print(f"Tick {t:3d}: wp={status['last_wp'] or '---':8s} "
                  f"seg={status['segment_type']:12s} "
                  f"loops={status['loop_count']} "
                  f"strong={len(strong)} "
                  f"best={best_name:20s} prec={best_prec:.2f} "
                  f"prims={len(library.names)}")

    print()
    print("--- Final patterns ---")
    for name, p in sorted(repertoire.patterns.items(), key=lambda x: -x[1].precision):
        seq = " -> ".join(p.primitives_sequence)
        print(f"  {name}: prec={p.precision:.2f} strong={p.is_strong} "
              f"trajs={p.trajectory_count} seq=[{seq}]")

    print()
    print("--- Discovered primitives ---")
    for pname in sorted(library.names):
        prim = library.get(pname)
        if prim and not pname.startswith("approach") and not pname.startswith("yield") and not pname.startswith("wait") and not pname.startswith("avoid"):
            print(f"  {pname}: skill={prim.skill_template.skill} "
                  f"dur={prim.typical_duration_s:.1f}s")

    # Verify crystallization
    strong = [p for p in repertoire.patterns.values() if p.is_strong]
    learned_seq = None
    if strong:
        best = max(strong, key=lambda p: p.precision)
        learned_seq = repertoire.pattern_to_sequence(best, context="retail")

    print()
    if learned_seq:
        print("--- Learned sequence ---")
        for i, step in enumerate(learned_seq.steps, 1):
            goal = step.request.goal
            print(f"  Step {i}: {step.request.skill} -> {goal}")
        print()
        print("=" * 60)
        print("SUCCESS: Crystallized from scratch with auto-segmentation!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("FAIL: No crystallization from scratch.")
        print("=" * 60)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
