"""Clear-table manager: ToM-driven multi-object clearing.

Wraps a ScriptManager and TaskPlanner to provide continuous
object-clearing behavior.  On each deliberation tick the manager:

1. Queries the GatedToM for the other agent's predicted intent
2. If the current script sequence is complete, asks the TaskPlanner
   to select the next object via EFE rollout
3. If the EFE says "wait" (observation phase), emits a wait request
4. Otherwise generates a nav→pick→nav→place sequence and loads it

All coordination emerges from the same active inference machinery
used for navigation — no explicit communication between robots.

Robot-agnostic: imports only from architecture_core.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from architecture_core.core.types import Intent, PerceptBundle, SkillRequest
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep
from architecture_core.cognition.tom.gated_tom import GatedToM
from architecture_core.cognition.tom.intent_particle_filter import (
    IntentProfile,
    ObservationContext,
)
from architecture_core.cognition.planning.task_planner import (
    ObjectState,
    TaskPlanner,
)
from architecture_core.cognition.planning.commitment_inference import CommitmentInference
from architecture_core.cognition.planning.affordance import (
    FurnitureItem,
)
from architecture_core.safety.geometry import Rect


class ClearTableManager:
    """Orchestrates multi-object table clearing via ToM + EFE.

    Parameters
    ----------
    planner : TaskPlanner
        EFE-driven object selector.
    base : ScriptManager
        Underlying script execution engine.
    gated_tom : GatedToM
        Theory of Mind module for intent prediction.
    """

    def __init__(
        self,
        planner: TaskPlanner,
        base: ScriptManager,
        gated_tom: GatedToM,
    ) -> None:
        self._planner = planner
        self._base = base
        self._tom = gated_tom
        self._commitment = CommitmentInference()
        self._planner._commitment = self._commitment
        self._awaiting_observation = False
        self._observe_ticks = 0
        self._max_observe_ticks = 3  # deliberation cycles to observe

        # Deadlock-driven yield penalties: persistent yielding + high
        # commitment evidence → p(deadlock) rises → preference penalty
        # increases → EFE naturally shifts to alternative objects.
        self._yield_count: Dict[str, int] = {}           # obj_id → consecutive yield ticks
        self._yield_penalties: Dict[str, float] = {}     # obj_id → penalty

        # Motion tracking for velocity/approaching estimation
        self._prev_other_dist: Dict[str, float] = {}    # other_id → last distance
        self._prev_t: float = 0.0                        # last timestamp

        # Robot's own arousal for precision coupling in task-level EFE
        self._self_arousal: Optional[float] = None

    # ------------------------------------------------------------------
    # ScriptManager-compatible interface (delegation)
    # ------------------------------------------------------------------
    @property
    def is_complete(self) -> bool:
        return self._planner.is_task_complete

    @property
    def is_repairing(self) -> bool:
        return self._base.is_repairing

    def notify_status(self, status) -> None:
        self._base.notify_status(status)

    def check_violation(self, pb: PerceptBundle, timestamp: float = 0.0):
        return self._base.check_violation(pb, timestamp=timestamp)

    def set_self_arousal(self, arousal: Optional[float]) -> None:
        """Set robot's own arousal for precision coupling in task-level EFE."""
        self._self_arousal = arousal

    def set_sequence(self, sequence: ScriptSequence) -> None:
        self._base.set_sequence(sequence)

    # Expose recognizer for executive.py situation-type detection
    @property
    def _recognizer(self):
        return self._base._recognizer

    # ------------------------------------------------------------------
    # Core: ToM-driven task selection
    # ------------------------------------------------------------------
    def select(
        self,
        pb: PerceptBundle,
        active: Optional[SkillRequest],
    ) -> SkillRequest:
        """Main deliberation entry point.

        1. Extract robot pose + other agent from PerceptBundle
        2. Query GatedToM for intent predictions
        3. If current sequence done → run EFE-based object selection
        4. Delegate step execution to ScriptManager

        Object status consistency: pb.world["objects"] must include
        accurate "status" ("on_table"/"held"/"placed") and "held_by"
        fields for coordination to stabilize.  Without this, robots
        may redundantly target held objects.  This is a controller/
        perception responsibility, not architecture_core.
        """
        # Update object states from perception
        objects_data = pb.world.get("objects", [])
        if objects_data:
            objects = [
                ObjectState(
                    id=o["id"],
                    type=o.get("type", "unknown"),
                    position=tuple(o["position"]),
                    table=o.get("table", "unknown"),
                    status=o.get("status", "on_table"),
                    held_by=o.get("held_by"),
                )
                for o in objects_data
            ]
            self._planner.update_objects(objects)

        # Extract positions
        robot_pose = pb.world.get("robot_pose", (0, 0, 0, 0))
        robot_pos = (robot_pose[0], robot_pose[1])

        agents = pb.world.get("agents", [])
        other_pos = None
        other_id = None
        if agents:
            ap = agents[0].get("pose", (0, 0))
            other_pos = (ap[0], ap[1])
            other_id = agents[0].get("id", "other")
            other_heading = agents[0].get("alpha", 0.0)
            self._planner.update_other_agent(other_id, other_pos, other_heading)

        # Check if we need to select a new object
        if self._base.is_complete or active is None:
            return self._select_next_object(
                pb, robot_pos, other_pos, other_id,
            )

        # Delegate to base ScriptManager for step execution
        return self._base.select(pb, active)

    def _select_next_object(
        self,
        pb: PerceptBundle,
        robot_pos: Tuple[float, float],
        other_pos: Optional[Tuple[float, float]],
        other_id: Optional[str],
    ) -> SkillRequest:
        """Use ToM + EFE to select next object and generate sequence."""
        # Build observation context from perception
        obs = self._build_obs_context(pb, robot_pos, other_pos, other_id)

        # Query ToM for other agent's predicted intent
        q_other: Dict[str, float] = {
            "approach": 0.2, "avoid": 0.2, "yield": 0.2,
            "wait": 0.2, "neutral": 0.2,
        }
        profile = IntentProfile(
            approach_bias=0.3, responsiveness=1.0,
            precision=1.0, empathy_j=0.3,
        )

        if other_id is not None:
            q_other = self._tom.predict_intent(other_id, obs)
            pf = self._tom.get_or_create_filter(other_id)
            rel = pf.reliability
            learned = pf.mean_profile()
            # Blend with default based on reliability
            from architecture_core.cognition.tom.gated_tom import _DEFAULT_PROFILE
            profile = IntentProfile(
                approach_bias=rel * learned.approach_bias + (1 - rel) * _DEFAULT_PROFILE.approach_bias,
                responsiveness=rel * learned.responsiveness + (1 - rel) * _DEFAULT_PROFILE.responsiveness,
                precision=rel * learned.precision + (1 - rel) * _DEFAULT_PROFILE.precision,
                empathy_j=rel * learned.empathy_j + (1 - rel) * _DEFAULT_PROFILE.empathy_j,
            )

        # Observation phase: high-empathy robot waits when uncertain
        if self._awaiting_observation:
            self._observe_ticks += 1
            if self._observe_ticks >= self._max_observe_ticks:
                self._awaiting_observation = False
                self._observe_ticks = 0
            else:
                return SkillRequest(
                    skill="navigate",
                    goal={"x": robot_pos[0], "y": robot_pos[1]},
                )

        # Decay yield penalties each deliberation tick
        for obj_id in list(self._yield_penalties):
            self._yield_penalties[obj_id] *= 0.9
            if self._yield_penalties[obj_id] < 0.1:
                del self._yield_penalties[obj_id]

        # Extract furniture from perception for affordance reasoning
        furniture_items = self._extract_furniture(pb)

        # EFE-driven object selection (affordance-aware)
        obj, rollout = self._planner.select_next(
            robot_pos=robot_pos,
            obs=obs,
            q_other=q_other,
            profile=profile,
            obstacles=furniture_items,
            yield_penalties=self._yield_penalties,
            other_pos=other_pos,
            self_arousal=self._self_arousal,
        )

        if obj is None:
            # All done or no objects — stay put
            return SkillRequest(
                skill="navigate",
                goal={"x": robot_pos[0], "y": robot_pos[1]},
            )

        # Deadlock evidence: persistent yielding + contention → penalty
        if rollout is not None and rollout.selected in ("yield", "wait"):
            obj_id = obj.id
            self._yield_count[obj_id] = self._yield_count.get(obj_id, 0) + 1
            # p(deadlock) from persistence × contention strength
            n_yields = self._yield_count[obj_id]
            p_contend = (
                self._commitment.get_belief(other_id, obj_id)
                if other_id else 0.0
            )
            p_deadlock = min(1.0, (n_yields / 10.0) * max(0.3, p_contend))
            if p_deadlock > 0.1:
                self._yield_penalties[obj_id] = p_deadlock * 3.0
        else:
            # Not yielding → clear count
            if obj is not None:
                self._yield_count.pop(obj.id, None)

        # Check if EFE says to observe first (wait at step 0)
        if rollout is not None and rollout.selected == "wait":
            self._awaiting_observation = True
            self._observe_ticks = 0
            return SkillRequest(
                skill="navigate",
                goal={"x": robot_pos[0], "y": robot_pos[1]},
            )

        # Generate and load sequence
        seq = self._planner.generate_sequence(obj)
        approach = self._planner._current_approach
        print(f"  [CTM] target={obj.id} pos={obj.position}"
              f" approach={approach}"
              f" steps={len(seq.steps)}"
              f" robot=({robot_pos[0]:.2f},{robot_pos[1]:.2f})")
        self._base.set_sequence(seq)
        return self._base.select(pb, None)

    def _build_obs_context(
        self,
        pb: PerceptBundle,
        robot_pos: Tuple[float, float],
        other_pos: Optional[Tuple[float, float]],
        other_id: Optional[str] = None,
    ) -> ObservationContext:
        """Build ObservationContext from PerceptBundle."""
        distance = 10.0
        approaching = False
        velocity = 0.0

        if other_pos is not None:
            dx = other_pos[0] - robot_pos[0]
            dy = other_pos[1] - robot_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)

            # Velocity from distance delta (positive = closing)
            oid = other_id or "unknown"
            dt = max(0.01, pb.t - self._prev_t) if self._prev_t > 0 else 0.1
            prev_dist = self._prev_other_dist.get(oid, distance)
            velocity = max(0.0, (prev_dist - distance) / dt)
            approaching = velocity > 0.02  # actual motion, not threshold

            self._prev_other_dist[oid] = distance
            self._prev_t = pb.t

        # Extract affect if available
        valence = 0.0
        arousal = 0.0
        affect = pb.social.get("affect", {})
        if affect:
            valence = affect.get("valence", 0.0)
            arousal = affect.get("arousal", 0.0)

        # Extract gaze/body orientation if available
        gaze = pb.social.get("engagement", {}).get("gaze_on_robot", 0.5)
        body_ori = pb.social.get("engagement", {}).get("body_orientation", 0.5)

        return ObservationContext(
            kinematic_intent="approach",
            distance=distance,
            velocity=velocity,
            approaching=approaching,
            gaze_on_robot=gaze,
            body_orientation=body_ori,
            valence=valence,
            arousal=arousal,
            robot_last_intent="neutral",
        )

    @staticmethod
    def _extract_furniture(pb: PerceptBundle) -> list:
        """Convert raw furniture dicts from perception to FurnitureItem."""
        furniture_data = pb.world.get("furniture", [])
        items = []
        for f in furniture_data:
            keepout = Rect(
                f.get("keepout_x_min", 0),
                f.get("keepout_y_min", 0),
                f.get("keepout_x_max", 0),
                f.get("keepout_y_max", 0),
            )
            items.append(FurnitureItem(
                id=f["id"],
                type=f.get("type", "unknown"),
                position=tuple(f["position"]),
                rotation=f.get("rotation", 0.0),
                width=f.get("width", 0.5),
                depth=f.get("depth", 0.5),
                mass=f.get("mass", 100.0),
                movable=f.get("movable", False),
                keepout=keepout,
            ))
        return items
