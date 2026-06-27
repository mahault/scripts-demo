"""Observational learning for the retail demo.

Watches the teacher (Worker_T) and feeds movement-segmented observations into the
learner's ScriptRepertoire so it can crystallise a script from observation alone.

The learner reads ONLY observable kinematics — the teacher's position over time
(=> speed) and its arm-joint angle (=> reaching).  It never reads the teacher's
published state string, nav target, or social flags.  How the teacher *decides*
to move is its own business; the learner infers the routine from the *motion*.
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


class TeacherObserver:
    """Watches the teacher and feeds movement-segmented observations to a repertoire.

    Observable signals only: teacher position (=> speed) and right-arm joint angle
    (=> reaching).  Segments come out as ``move`` (travelled and stopped at a
    fixture) and ``reach`` (raised the arm at a fixture); the repertoire crystallises
    the recurring sequence into a script.
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

        # Loop / trajectory tracking
        self._loop_count = 0
        self._step_count = 0
        self._started = False

        # Observed-motion state
        self._last_pos = None
        self._last_t = None
        self._speed = 0.0
        self._dt_default = robot.getBasicTimeStep() / 1000.0

    # ------------------------------------------------------------------
    # Teacher node access (position + arm angle ONLY)
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
            p = node.getField("translation").getSFVec3f()
            return (p[0], p[1])
        except Exception:
            return None

    def _get_teacher_arm(self) -> float:
        """Observed right-arm joint angle — high when the teacher is reaching."""
        node = self._get_teacher_node()
        if node is None:
            return 0.0
        try:
            f = node.getField("rightArmAngle")
            return f.getSFFloat() if f else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def tick(self, t: float, repertoire: ScriptRepertoire, recognizer) -> dict:
        pos = self._get_teacher_pos()
        status = {
            "observing": pos is not None,
            "loop_count": self._loop_count,
            "step_count": self._step_count,
            "teacher_pos": pos,
            "last_wp": self._segmentation._at_fixture,
            "segment_type": None,
            "speed": round(self._speed, 2),
            "discovered_primitives": self._segmentation.get_discovered_primitive_names(),
        }
        if pos is None:
            return status

        # Observed speed from successive positions (EMA-smoothed against gait jitter).
        if self._last_pos is not None:
            dt = (t - self._last_t) if (self._last_t is not None and t > self._last_t) \
                else self._dt_default
            d = ((pos[0] - self._last_pos[0]) ** 2
                 + (pos[1] - self._last_pos[1]) ** 2) ** 0.5
            inst = d / dt if dt > 1e-6 else 0.0
            self._speed = 0.6 * self._speed + 0.4 * inst
        self._last_pos, self._last_t = pos, t

        arm = self._get_teacher_arm()
        seg = self._segmentation.track_tick(t, pos, self._speed, arm)
        if seg is not None and repertoire is not None:
            status["segment_type"] = seg.segment_type
            self._handle_completed_segment(t, seg, repertoire)
        return status

    def _handle_completed_segment(
        self, t: float, segment: BehaviorSegment, repertoire: ScriptRepertoire
    ) -> None:
        """Record an observed move/reach primitive; detect a full restock loop."""
        primitive = self._segmentation.get_or_create_primitive(segment)
        situation = f"{segment.zone}_zone"
        situation_belief = {situation: 1.0}

        if not self._started:
            repertoire.on_situation_recognized(situation_belief, t=t)
            self._started = True

        repertoire.on_step_completed(
            t=t,
            primitive_name=primitive.name,
            situation_belief=situation_belief,
            outcome="SUCCESS",
            affect_before=AffectState(valence=0.0, arousal=0.0),
        )
        self._step_count += 1

        # A full restock loop = observed return to the stock landmark after real
        # work (pick at stock -> out to a fixture -> place -> home).
        if (segment.segment_type == "move" and segment.zone == "stock"
                and self._step_count >= 3):
            repertoire.on_script_completed("SUCCESS")
            self._loop_count += 1
            self._step_count = 0
            self._segmentation.reset()
            self._started = False

    def reset(self) -> None:
        """Reset observation state (e.g. when switching to execution)."""
        self._segmentation.reset()
        self._loop_count = 0
        self._step_count = 0
        self._started = False
        self._last_pos = None
        self._last_t = None
        self._speed = 0.0


def build_learned_sequence(repertoire: ScriptRepertoire) -> ScriptSequence | None:
    """Extract the best strong pattern from the repertoire as a ScriptSequence."""
    strong = [p for p in repertoire.patterns.values() if p.is_strong]
    if not strong:
        return None
    best = max(strong, key=lambda p: p.precision)
    seq = repertoire.pattern_to_sequence(best, context="retail")

    # Drop degenerate navigate steps: a navigate primitive discovered without a
    # target defaults to the world origin (0,0), which the learner cannot reach
    # and wedges on — a long dead freeze.  Keep only real moves.
    cleaned = []
    for step in seq.steps:
        if step.request.skill == "navigate":
            g = step.request.goal or {}
            if abs(g.get("x", 0.0)) < 1e-6 and abs(g.get("y", 0.0)) < 1e-6:
                continue
        cleaned.append(step)
    seq.steps = cleaned
    return seq
