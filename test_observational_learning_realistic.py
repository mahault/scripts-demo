"""Realistic observational learning test.

Simulates stale customData, position jitter, and realistic timing
to verify robustness before Webots integration.
"""

from __future__ import annotations

import json
import random

from architecture_core.core.types import AffectState
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer
from architecture_core.cognition.scripts.script_types import SituationType
from plugins.tiago_webots.retail_primitives import make_retail_repertoire_config
from plugins.tiago_webots.observation import TeacherObserver


class MockTeacherNode:
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


class MockRobot:
    def __init__(self):
        self._teacher = MockTeacherNode()

    def getFromDef(self, name):
        return self._teacher


WAYPOINT_COORDS = {
    0: (-5.5, -1.0),
    1: (-2.0, -5.0),
    2: (1.5, -5.0),
    3: (4.5, -7.5),
}


def generate_realistic_timeline(n_loops=3, publish_interval=5, jitter_sigma=0.05):
    """Generate a realistic teacher timeline with stale customData and position jitter."""
    timeline = []
    rng = random.Random(42)

    for loop in range(n_loops):
        def add_phase(wp_idx, state, duration_ticks):
            base_x, base_y = WAYPOINT_COORDS[wp_idx]
            for dt in range(duration_ticks):
                x = base_x + rng.gauss(0, jitter_sigma)
                y = base_y + rng.gauss(0, jitter_sigma)
                if dt % publish_interval == 0:
                    data = json.dumps({"waypoint": wp_idx, "state": state, "loop": loop})
                else:
                    data = None
                timeline.append(((x, y), data, wp_idx, state))

        # Stock
        add_phase(0, "WORK(90)", 2)
        add_phase(0, "WORK(45)", 3)
        add_phase(0, "WORK(15)", 3)
        # Nav to shelf_a
        for t in range(3):
            frac = (t + 1) / 3.0
            x = WAYPOINT_COORDS[0][0] + frac * (WAYPOINT_COORDS[1][0] - WAYPOINT_COORDS[0][0])
            y = WAYPOINT_COORDS[0][1] + frac * (WAYPOINT_COORDS[1][1] - WAYPOINT_COORDS[0][1])
            x += rng.gauss(0, jitter_sigma)
            y += rng.gauss(0, jitter_sigma)
            data = json.dumps({"waypoint": 1, "state": "NAV(SUCCESS)", "loop": loop}) if t == 2 else None
            timeline.append(((x, y), data, 1, "NAV"))
        # Shelf_a
        add_phase(1, "WORK(90)", 2)
        add_phase(1, "WORK(45)", 3)
        add_phase(1, "WORK(15)", 3)
        # Nav to shelf_b
        for t in range(3):
            frac = (t + 1) / 3.0
            x = WAYPOINT_COORDS[1][0] + frac * (WAYPOINT_COORDS[2][0] - WAYPOINT_COORDS[1][0])
            y = WAYPOINT_COORDS[1][1] + frac * (WAYPOINT_COORDS[2][1] - WAYPOINT_COORDS[1][1])
            x += rng.gauss(0, jitter_sigma)
            y += rng.gauss(0, jitter_sigma)
            data = json.dumps({"waypoint": 2, "state": "NAV(SUCCESS)", "loop": loop}) if t == 2 else None
            timeline.append(((x, y), data, 2, "NAV"))
        # Shelf_b
        add_phase(2, "WORK(90)", 2)
        add_phase(2, "WORK(45)", 3)
        add_phase(2, "WORK(15)", 3)
        # Nav to counter
        for t in range(3):
            frac = (t + 1) / 3.0
            x = WAYPOINT_COORDS[2][0] + frac * (WAYPOINT_COORDS[3][0] - WAYPOINT_COORDS[2][0])
            y = WAYPOINT_COORDS[2][1] + frac * (WAYPOINT_COORDS[3][1] - WAYPOINT_COORDS[2][1])
            x += rng.gauss(0, jitter_sigma)
            y += rng.gauss(0, jitter_sigma)
            data = json.dumps({"waypoint": 3, "state": "NAV(SUCCESS)", "loop": loop}) if t == 2 else None
            timeline.append(((x, y), data, 3, "NAV"))
        # Counter
        add_phase(3, "WORK(90)", 2)
        add_phase(3, "WORK(45)", 3)
        add_phase(3, "WORK(15)", 3)
        # Nav back to stock
        for t in range(3):
            frac = (t + 1) / 3.0
            x = WAYPOINT_COORDS[3][0] + frac * (WAYPOINT_COORDS[0][0] - WAYPOINT_COORDS[3][0])
            y = WAYPOINT_COORDS[3][1] + frac * (WAYPOINT_COORDS[0][1] - WAYPOINT_COORDS[3][1])
            x += rng.gauss(0, jitter_sigma)
            y += rng.gauss(0, jitter_sigma)
            data = json.dumps({"waypoint": 0, "state": "NAV(SUCCESS)", "loop": loop}) if t == 2 else None
            timeline.append(((x, y), data, 0, "NAV"))

    # Extra tick to complete final navigate segment
    x, y = WAYPOINT_COORDS[0]
    timeline.append(((x + rng.gauss(0, jitter_sigma), y + rng.gauss(0, jitter_sigma)),
                     json.dumps({"waypoint": 0, "state": "WORK(90)", "loop": n_loops}), 0, "WORK"))
    return timeline


def main():
    print("=" * 60)
    print("Realistic Observational Learning Test")
    print("(stale customData + position jitter + intermediate positions)")
    print("=" * 60)

    config = make_retail_repertoire_config()
    recognizer = WeakScriptRecognizer(
        situation_types=[
            SituationType(name="stock_zone", feature_weights={"cue_stock": 4.0}),
            SituationType(name="shelf_zone", feature_weights={"cue_shelf": 4.0}),
            SituationType(name="counter_zone", feature_weights={"cue_counter": 4.0}),
        ]
    )
    mock = MockRobot()

    for test_name, publish_interval, jitter_sigma in [
        ("perfect", 1, 0.0),
        ("moderate_stale", 3, 0.03),
        ("heavy_stale", 5, 0.08),
        ("noisy_position", 1, 0.15),
    ]:
        print(f"\n--- Test: {test_name} (publish_every={publish_interval}, jitter={jitter_sigma}) ---")

        library = PrimitiveLibrary(register_defaults=True)
        repertoire = ScriptRepertoire(
            library, config=config, initial_patterns=[], observational_mode=True,
        )
        observer = TeacherObserver(mock, teacher_name="Worker_T", library=library)

        timeline = generate_realistic_timeline(n_loops=3, publish_interval=publish_interval, jitter_sigma=jitter_sigma)

        segments_seen = 0
        for t, (pos, data, expected_wp, expected_state) in enumerate(timeline, 1):
            mock._teacher.set_pos(pos[0], pos[1])
            if data is not None:
                mock._teacher._custom_data = data
            status = observer.tick(t=t, repertoire=repertoire, recognizer=recognizer)
            if status.get("segment_type"):
                segments_seen += 1

        strong = [p for p in repertoire.patterns.values() if p.is_strong]
        best = max(repertoire.patterns.values(), key=lambda p: p.precision) if repertoire.patterns else None

        print(f"  Timeline ticks: {len(timeline)}, Segments detected: {segments_seen}")
        print(f"  Patterns: {len(repertoire.patterns)}, Strong: {len(strong)}")
        print(f"  Primitives: {len(library.names)}")
        if best:
            print(f"  Best: {best.name[:50]}... prec={best.precision:.2f} strong={best.is_strong} trajs={best.trajectory_count}")
        if strong:
            print("  >>> CRYSTALLIZED <<<")
        else:
            print("  >>> NOT CRYSTALLIZED <<<")

    print("\n" + "=" * 60)
    print("All realistic tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
