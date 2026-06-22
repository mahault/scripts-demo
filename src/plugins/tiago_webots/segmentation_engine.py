"""Behavior segmentation engine for observational learning.

Segments continuous teacher behavior into discrete primitives by detecting
change points in state, position, and velocity. Discovered primitives are
registered dynamically in a PrimitiveLibrary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from architecture_core.core.types import AffectState, SkillRequest
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.repertoire_types import ScriptPrimitive


@dataclass
class BehaviorSegment:
    """A discovered segment of continuous teacher behavior."""

    start_t: float
    end_t: float
    segment_type: str  # e.g. "navigate", "extend-arm", "retract-arm", "wait"
    dominant_feature: str  # what signal triggered this segment
    situation_context: str = ""  # inferred situation
    waypoint: Optional[str] = None  # associated waypoint if any
    target_xy: Optional[Tuple[float, float]] = None  # teacher's nav target


class SegmentationEngine:
    """Segments teacher behavior into discrete actions.

    Uses teacher state (from customData) as the primary segmentation signal,
    with position/velocity as secondary confirmation.
    """

    # State prefixes that indicate different behavior modes
    NAV_PREFIX = "NAV"
    WORK_PREFIX = "WORK"
    SOCIAL_PREFIX = "SOCIAL"

    def __init__(self, library: Optional[PrimitiveLibrary] = None) -> None:
        self._library = library
        self._discovered_primitives: Dict[str, ScriptPrimitive] = {}
        self._current_segment: Optional[BehaviorSegment] = None
        self._completed_segments: List[BehaviorSegment] = []
        self._segment_counter = 0

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    def track_tick(
        self,
        t: float,
        pos: Tuple[float, float],
        teacher_state_json: Optional[str] = None,
    ) -> Optional[BehaviorSegment]:
        """Process one tick of teacher observation.

        Returns a completed segment if a segment boundary was crossed.
        """
        state, waypoint_idx, target_xy = self._parse_teacher_state(teacher_state_json)
        segment_type = self._classify_state(state, waypoint_idx)
        situation = self._waypoint_idx_to_situation(waypoint_idx)
        waypoint_name = self._waypoint_idx_to_name(waypoint_idx)

        if self._current_segment is None:
            # First observation — start first segment
            self._current_segment = BehaviorSegment(
                start_t=t,
                end_t=t,
                segment_type=segment_type,
                dominant_feature="initial_observation",
                situation_context=situation,
                waypoint=waypoint_name,
                target_xy=target_xy,
            )
            return None

        if segment_type != self._current_segment.segment_type:
            # Segment boundary crossed — finalize current, start new
            self._current_segment.end_t = t
            completed = self._current_segment
            self._completed_segments.append(completed)

            self._current_segment = BehaviorSegment(
                start_t=t,
                end_t=t,
                segment_type=segment_type,
                dominant_feature="state_transition",
                situation_context=situation,
                waypoint=waypoint_name,
                target_xy=target_xy,
            )
            return completed

        # Same segment — update end time and keep the latest nav target
        self._current_segment.end_t = t
        if target_xy is not None:
            self._current_segment.target_xy = target_xy
        return None

    # ------------------------------------------------------------------
    # Segment access
    # ------------------------------------------------------------------
    def flush_segments(self) -> List[BehaviorSegment]:
        """Return and clear all completed segments."""
        result = list(self._completed_segments)
        self._completed_segments.clear()
        return result

    def get_all_segments(self) -> List[BehaviorSegment]:
        """Return all segments including the current in-progress one."""
        result = list(self._completed_segments)
        if self._current_segment is not None:
            result.append(self._current_segment)
        return result

    def reset(self) -> None:
        """Reset engine state."""
        self._current_segment = None
        self._completed_segments.clear()

    # ------------------------------------------------------------------
    # Primitive discovery
    # ------------------------------------------------------------------
    def get_or_create_primitive(self, segment: BehaviorSegment) -> ScriptPrimitive:
        """Get existing primitive for segment type, or create and cache a new one."""
        prim_name = self._segment_to_primitive_name(segment)

        if prim_name in self._discovered_primitives:
            return self._discovered_primitives[prim_name]

        primitive = self._build_primitive(segment, prim_name)
        self._discovered_primitives[prim_name] = primitive

        if self._library is not None:
            self._library.register(primitive)

        return primitive

    def get_discovered_primitive_names(self) -> List[str]:
        """Return names of all discovered primitives."""
        return list(self._discovered_primitives.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_teacher_state(
        teacher_state_json: Optional[str],
    ) -> Tuple[str, int, Optional[Tuple[float, float]]]:
        """Parse customData JSON for state string, waypoint index, nav target."""
        if not teacher_state_json:
            return "", -1, None
        try:
            data = json.loads(teacher_state_json)
            tx, ty = data.get("tx"), data.get("ty")
            target = (float(tx), float(ty)) if tx is not None and ty is not None else None
            return data.get("state", ""), data.get("waypoint", -1), target
        except (json.JSONDecodeError, TypeError, ValueError):
            return "", -1, None

    @classmethod
    def _classify_state(cls, state: str, waypoint_idx: int) -> str:
        """Map teacher state string to a segment type."""
        if state.startswith(cls.NAV_PREFIX):
            return "navigate"

        if state.startswith(cls.SOCIAL_PREFIX):
            # "SOCIAL:yield" / "SOCIAL:greet" / "SOCIAL:handover" — the social
            # act becomes its own observable segment, so the learner discovers
            # it as a primitive (obs_yield, obs_greet, ...) in the script.
            parts = state.split(":", 1)
            return parts[1] if len(parts) > 1 and parts[1] else "social"

        if state.startswith(cls.WORK_PREFIX):
            # Parse dwell remaining from "WORK(N)"
            try:
                dwell = int(state.split("(")[1].split(")")[0])
            except (IndexError, ValueError):
                dwell = 60

            # Counter (waypoint 3) has different behavior
            if waypoint_idx == 3:
                if dwell > 60:
                    return "wait"
                elif dwell > 30:
                    return "look-around"
                else:
                    return "settle"

            # Stock (0) or shelves (1, 2)
            if dwell > 60:
                return "approach"
            elif dwell > 30:
                return "extend-arm"
            else:
                return "retract-arm"

        return "unknown"

    @staticmethod
    def _waypoint_idx_to_name(idx: int) -> Optional[str]:
        mapping = {0: "stock", 1: "shelf_a", 2: "shelf_b", 3: "counter"}
        return mapping.get(idx)

    @staticmethod
    def _waypoint_idx_to_situation(idx: int) -> str:
        mapping = {
            0: "stock_zone",
            1: "shelf_zone",
            2: "shelf_zone",
            3: "counter_zone",
        }
        return mapping.get(idx, "unknown")

    @staticmethod
    def _segment_to_primitive_name(segment: BehaviorSegment) -> str:
        """Generate a deterministic primitive name from a segment.

        Navigate segments are made destination-specific (keyed by their target
        waypoint) so the learner discovers one primitive per move and can
        retrace the real route, instead of collapsing every navigation into a
        single goal-less primitive.
        """
        base = f"obs_{segment.segment_type.replace('-', '_')}"
        if segment.segment_type == "navigate" and segment.target_xy is not None:
            return f"{base}_{segment.target_xy[0]:.1f}_{segment.target_xy[1]:.1f}"
        return base

    @classmethod
    def _build_primitive(
        cls, segment: BehaviorSegment, prim_name: str
    ) -> ScriptPrimitive:
        """Construct a ScriptPrimitive from a behavior segment."""
        duration = max(1.0, segment.end_t - segment.start_t)

        # Only the navigate segments are real base motion: give each its learned
        # destination so the learner retraces the route.  Every other segment is
        # a stationary dwell (arm/head gesture or a pause) and maps to the
        # manipulate skill, which completes on its own — mapping these to a
        # zero-speed navigate (the old behaviour) sent the learner to the origin
        # and never returned SUCCESS.
        if segment.segment_type == "navigate":
            goal = ({"x": segment.target_xy[0], "y": segment.target_xy[1]}
                    if segment.target_xy is not None else {"x": 0.0, "y": 0.0})
            skill = SkillRequest(
                skill="navigate",
                goal=goal,
                params={
                    "intent": "approach",
                    "speed_scale": 0.6,
                    "goal_tolerance": 0.6,
                },
            )
        elif segment.segment_type == "extend-arm":
            skill = SkillRequest(skill="manipulate", params={"action": "extend_arm"})
        elif segment.segment_type == "retract-arm":
            skill = SkillRequest(skill="manipulate", params={"action": "retract_arm"})
        elif segment.segment_type == "look-around":
            skill = SkillRequest(skill="manipulate", params={"action": "look_around"})
        else:  # approach / wait / settle / unknown — stationary dwell
            skill = SkillRequest(skill="manipulate", params={"action": "wait"})

        return ScriptPrimitive(
            name=prim_name,
            skill_template=skill,
            precondition_situations=[segment.situation_context],
            postcondition_situation=segment.situation_context,
            typical_duration_s=duration,
            expected_affect=AffectState(valence=0.0, arousal=0.0),
        )
