"""Movement-based behavior segmentation for observational learning.

The learner watches the teacher and segments its behavior into discrete
primitives using ONLY observable kinematics — the teacher's position over time
(speed), its arm-joint angle (reaching vs not), and its distance to visible
fixtures.  It never reads the teacher's published state string, nav target, or
social flags: how the teacher decides to move is its own business; we observe
only the motion.

Inferred primitives:
  * ``move`` — the teacher travelled and came to rest at a fixture.  The
    destination (where it stopped) and the fixture *zone* (stock / shelf /
    counter, from the nearest landmark) are read off the trajectory.
  * ``reach`` — the teacher, at rest at a fixture, raised its arm (a stocking /
    hand-over gesture), read from the arm-joint angle.

Stops away from any fixture (e.g. stepping aside to let a shopper pass) produce
no primitive, so reactive social acts stay out of the crystallising routine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from architecture_core.core.types import AffectState, SkillRequest
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.repertoire_types import ScriptPrimitive


# Visible static landmarks the watcher can localise against (fixture centres).
FIXTURES: Dict[str, Tuple[float, float]] = {
    "stock": (-5.5, -1.0),
    "shelf_a": (-2.0, -5.75),
    "shelf_b": (1.5, -5.75),
    "counter": (4.5, -6.85),
}
# Coarse zone per fixture — keying moves by zone lets the learner generalise
# "go to a shelf" across shelf A and shelf B.
FIXTURE_ZONE: Dict[str, str] = {
    "stock": "stock", "shelf_a": "shelf", "shelf_b": "shelf", "counter": "counter",
}

STOP_SPEED = 0.18        # m/s — below this the teacher is "at rest"
FIXTURE_NEAR = 1.5       # m — within this of a fixture centre counts as "at" it
REACH_ARM = 0.95         # rad — arm angle above this = reaching (walking peaks ~0.7)
ARRIVE_TICKS = 4         # consecutive at-rest ticks before we call it an arrival
REACH_TICKS = 3          # consecutive arm-up ticks before we call it a reach


@dataclass
class BehaviorSegment:
    """A discovered segment of teacher behavior, inferred from motion."""

    start_t: float
    end_t: float
    segment_type: str               # "move" | "reach"
    zone: str = "area"              # stock / shelf / counter
    waypoint: Optional[str] = None  # nearest fixture name
    target_xy: Optional[Tuple[float, float]] = None  # where the move ended


class SegmentationEngine:
    """Segments teacher behavior from observed kinematics."""

    def __init__(self, library: Optional[PrimitiveLibrary] = None) -> None:
        self._library = library
        self._discovered_primitives: Dict[str, ScriptPrimitive] = {}
        # Arrival/departure state machine
        self._at_fixture: Optional[str] = None     # fixture we are currently "at"
        self._slow_ticks = 0
        self._arm_ticks = 0
        self._reached_here = False                 # already emitted a reach at this stop
        self._move_start_t = 0.0                   # when the current travel began
        self._t = 0.0

    # ------------------------------------------------------------------
    # Tracking — fed observable kinematics only
    # ------------------------------------------------------------------
    def track_tick(
        self,
        t: float,
        pos: Tuple[float, float],
        speed: float,
        arm_angle: float,
    ) -> Optional[BehaviorSegment]:
        """Process one tick of observation; return a completed segment or None."""
        self._t = t
        fname, fdist = self._nearest_fixture(pos)
        at_fixture = (speed < STOP_SPEED) and (fdist < FIXTURE_NEAR)

        # Hysteresis on "at rest near a fixture".
        self._slow_ticks = self._slow_ticks + 1 if at_fixture else 0
        arrived = self._slow_ticks >= ARRIVE_TICKS

        # Hysteresis on "arm raised".
        self._arm_ticks = self._arm_ticks + 1 if arm_angle > REACH_ARM else 0
        arm_up = self._arm_ticks >= REACH_TICKS

        # --- Arrival: we just settled at a fixture we were travelling to ---
        if arrived and self._at_fixture is None:
            self._at_fixture = fname
            self._reached_here = False
            return BehaviorSegment(
                start_t=self._move_start_t, end_t=t, segment_type="move",
                zone=FIXTURE_ZONE.get(fname, "area"), waypoint=fname,
                target_xy=(pos[0], pos[1]),
            )

        # --- Reach: at a fixture and the arm goes up (once per stop) ---
        if self._at_fixture is not None and arm_up and not self._reached_here:
            self._reached_here = True
            return BehaviorSegment(
                start_t=t, end_t=t, segment_type="reach",
                zone=FIXTURE_ZONE.get(self._at_fixture, "area"),
                waypoint=self._at_fixture, target_xy=(pos[0], pos[1]),
            )

        # --- Departure: started moving again, a new travel leg begins ---
        if self._at_fixture is not None and speed >= STOP_SPEED and fdist >= FIXTURE_NEAR:
            self._at_fixture = None
            self._reached_here = False
            self._move_start_t = t

        return None

    # ------------------------------------------------------------------
    # Primitive discovery
    # ------------------------------------------------------------------
    def get_or_create_primitive(self, segment: BehaviorSegment) -> ScriptPrimitive:
        prim_name = self._segment_to_primitive_name(segment)
        if prim_name in self._discovered_primitives:
            return self._discovered_primitives[prim_name]
        primitive = self._build_primitive(segment, prim_name)
        self._discovered_primitives[prim_name] = primitive
        if self._library is not None:
            self._library.register(primitive)
        return primitive

    def get_discovered_primitive_names(self) -> List[str]:
        return list(self._discovered_primitives.keys())

    def reset(self) -> None:
        self._at_fixture = None
        self._slow_ticks = 0
        self._arm_ticks = 0
        self._reached_here = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _nearest_fixture(pos: Tuple[float, float]) -> Tuple[str, float]:
        best, best_d = "area", 1e9
        for name, (fx, fy) in FIXTURES.items():
            d = ((pos[0] - fx) ** 2 + (pos[1] - fy) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, name
        return best, best_d

    @staticmethod
    def _segment_to_primitive_name(segment: BehaviorSegment) -> str:
        if segment.segment_type == "move":
            return f"obs_move_{segment.zone}"     # zone-keyed -> generalises
        return "obs_reach"

    @classmethod
    def _build_primitive(
        cls, segment: BehaviorSegment, prim_name: str
    ) -> ScriptPrimitive:
        duration = max(1.0, segment.end_t - segment.start_t)
        situation = f"{segment.zone}_zone"
        if segment.segment_type == "move" and segment.target_xy is not None:
            skill = SkillRequest(
                skill="navigate",
                goal={"x": segment.target_xy[0], "y": segment.target_xy[1]},
                params={"intent": "approach", "speed_scale": 0.7,
                        # Generous tolerance: the observed stop sits deep against a
                        # fixture/corner the no-physics teacher could reach but the
                        # wheeled learner cannot squeeze into — accept arriving near it.
                        "goal_tolerance": 0.8},
            )
        else:  # reach
            skill = SkillRequest(skill="manipulate", params={"action": "extend_arm"})
        return ScriptPrimitive(
            name=prim_name,
            skill_template=skill,
            precondition_situations=[situation],
            postcondition_situation=situation,
            typical_duration_s=duration,
            expected_affect=AffectState(valence=0.0, arousal=0.0),
        )
