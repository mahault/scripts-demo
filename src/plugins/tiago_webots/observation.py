"""Observational learning for the retail demo.

Uses Webots Supervisor API to watch the teacher robot (Worker_T),
segment its trajectory into discrete behavior primitives, and feed
synthetic observations into the learner's ScriptRepertoire.

The learner can discover primitives by segmentation, assemble patterns
from observed sequences, and crystallize a script from observation alone —
without any pre-defined retail primitives or seed patterns.
"""

from __future__ import annotations

from typing import Optional

from architecture_core.core.types import AffectState
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep

from plugins.tiago_webots.segmentation_engine import (
    BehaviorSegment,
    SegmentationEngine,
)


# Spatial waypoints (used for loop-detection and situation inference)
WAYPOINTS = {
    "stock": (-5.5, -1.0),
    "shelf_a": (-2.0, -5.0),
    "shelf_b": (1.5, -5.0),
    "counter": (4.5, -7.5),
}

# Mapping from waypoint name to situation
WAYPOINT_SITUATION = {
    "stock": "stock_zone",
    "shelf_a": "shelf_zone",
    "shelf_b": "shelf_zone",
    "counter": "counter_zone",
}

ARRIVAL_THRESHOLD = 0.8


class TeacherObserver:
    """Watches the teacher robot and feeds segmented observations to a repertoire.

    Uses the SegmentationEngine to discover primitives dynamically from
    teacher state transitions (read via Supervisor API customData).
    """

    def __init__(
        self,
        robot,
        teacher_name: str = "Worker_T",
        library: Optional[PrimitiveLibrary] = None,
    ) -> None:
        self._robot = robot
        self._teacher_name = teacher_name
        self._teacher_node = None
        self._segmentation = SegmentationEngine(library=library)

        # Loop tracking
        self._loop_count = 0
        self._step_count = 0
        self._started = False
        self._last_waypoint: Optional[str] = None

    # ------------------------------------------------------------------
    # Teacher node access
    # ------------------------------------------------------------------
    def _get_teacher_node(self):
        if self._teacher_node is None:
            self._teacher_node = self._robot.getFromDef(self._teacher_name)
        return self._teacher_node

    def _get_teacher_pos(self):
        node = self._get_teacher_node()
        if node is None:
            return None
        try:
            pos = node.getField("translation").getSFVec3f()
            return (pos[0], pos[1])
        except Exception:
            return None

    def _get_teacher_state_json(self) -> Optional[str]:
        node = self._get_teacher_node()
        if node is None:
            return None
        try:
            return node.getField("customData").getSFString()
        except Exception:
            return None

    def _nearest_waypoint(self, pos):
        best_name = None
        best_dist = 1e9
        for name, (wx, wy) in WAYPOINTS.items():
            d = ((pos[0] - wx) ** 2 + (pos[1] - wy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name, best_dist

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def tick(
        self,
        t: float,
        repertoire: ScriptRepertoire,
        recognizer,
    ) -> dict:
        """Watch the teacher for one tick.

        Returns a status dict with keys:
            observing, at_wp, last_wp, loop_count, step_count, teacher_pos,
            segment_type, discovered_primitives
        """
        pos = self._get_teacher_pos()
        state_json = self._get_teacher_state_json()

        status = {
            "observing": pos is not None,
            "at_wp": False,
            "last_wp": self._last_waypoint,
            "loop_count": self._loop_count,
            "step_count": self._step_count,
            "teacher_pos": pos,
            "segment_type": None,
            "discovered_primitives": self._segmentation.get_discovered_primitive_names(),
        }

        if pos is None:
            return status

        wp_name, nearest_dist = self._nearest_waypoint(pos)
        if nearest_dist < ARRIVAL_THRESHOLD:
            self._last_waypoint = wp_name
            status["at_wp"] = True
            status["last_wp"] = wp_name

        # Feed to segmentation engine
        completed_segment = self._segmentation.track_tick(t, pos, state_json)

        if completed_segment is not None and repertoire is not None:
            status["segment_type"] = completed_segment.segment_type
            self._handle_completed_segment(t, completed_segment, repertoire, wp_name)

        return status

    def _handle_completed_segment(
        self,
        t: float,
        segment: BehaviorSegment,
        repertoire: ScriptRepertoire,
        current_wp: Optional[str],
    ) -> None:
        """Process a completed behavior segment: discover primitive, feed to repertoire."""
        # Skip unknown/unclassified segments (stale data, transitions, etc.)
        if segment.segment_type == "unknown":
            return

        primitive = self._segmentation.get_or_create_primitive(segment)
        situation = WAYPOINT_SITUATION.get(current_wp or segment.waypoint or "", "approach")
        situation_belief = {situation: 1.0}

        if not self._started:
            # First segment ever — start trajectory
            repertoire.on_situation_recognized(situation_belief, t=t)
            self._started = True

        # Record the observed step
        repertoire.on_step_completed(
            t=t,
            primitive_name=primitive.name,
            situation_belief=situation_belief,
            outcome="SUCCESS",
            affect_before=AffectState(valence=0.0, arousal=0.0),
        )
        self._step_count += 1

        # Detect full loop: returned to stock after visiting counter
        if segment.segment_type == "navigate" and current_wp == "stock" and self._step_count >= 8:
            # A full restock loop has at least ~8 segments
            repertoire.on_script_completed("SUCCESS")
            self._loop_count += 1
            self._step_count = 0
            # Reset segmentation for next loop
            self._segmentation.reset()
            self._started = False

    def reset(self) -> None:
        """Reset observation state (e.g. when switching to execution)."""
        self._segmentation.reset()
        self._last_waypoint = None
        self._loop_count = 0
        self._step_count = 0
        self._started = False


def build_learned_sequence(repertoire: ScriptRepertoire) -> ScriptSequence | None:
    """Extract the best strong pattern from the repertoire and convert to a ScriptSequence."""
    strong = [p for p in repertoire.patterns.values() if p.is_strong]
    if not strong:
        return None
    best = max(strong, key=lambda p: p.precision)
    return repertoire.pattern_to_sequence(best, context="retail")
