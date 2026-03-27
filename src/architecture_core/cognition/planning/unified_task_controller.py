"""Unified active inference task controller.

Replaces ClearTableManager + ScriptManager with a single POMDP-based
controller where task phases emerge from posterior beliefs and actions
are selected by EFE minimisation.  Single inference loop per tick —
no state machine, no explicit SUCCESS-based transitions.

Same interface as ClearTableManager so it drops in at the Executive level.

Key differences from ClearTableManager:
  - active=None does NOT trigger reselection (beliefs drive policy)
  - Phase transitions are distance-conditioned (smooth sigmoid, not discrete)
  - VFE feeds EmpathicModulator (actual F, not G proxy)
  - Policy entropy H[q(pi)] from EFE softmax (self-consistent)
  - Safe yield waypoints with world-bounds clamping and hysteresis
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from architecture_core.core.types import PerceptBundle, SkillRequest
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
from architecture_core.cognition.planning.affordance import FurnitureItem
from architecture_core.cognition.planning.generative_model import (
    TaskPhase,
    HandState,
    TargetMode,
    TaskPolicy,
    N_STATES,
    N_POLICIES,
    Observation,
    state_index,
    state_factors,
    discretize_distance,
    discretize_commitment,
    build_A_matrices,
    build_B_matrices,
    build_C_vectors,
    build_D_vector,
    update_B_from_distances,
    ObjZObs,
    TargetObs,
    SkillOutcomeObs,
)
from architecture_core.cognition.planning.variational_engine import (
    belief_update,
    evaluate_policies_multistep,
    select_policy,
)
from architecture_core.safety.geometry import Rect


class UnifiedTaskController:
    """POMDP-based task controller for multi-object table clearing.

    Parameters
    ----------
    planner : TaskPlanner
        EFE-driven object selector (reused for which-object decisions).
    gated_tom : GatedToM
        Theory of Mind module for intent prediction.
    empathy_factor : float
        Empathy weight [0, 1] for social preference scaling.
    world_bounds : tuple
        (x_min, x_max, y_min, y_max) for safe yield waypoint generation.
    beta : float
        Base softmax precision for policy selection.
    """

    # EMA smoothing for VFE (damps jitter from Webots perception noise)
    _VFE_ALPHA = 0.3
    # Yield hysteresis: persist target for this many ticks
    _YIELD_PERSIST_TICKS = 5
    # Belief thresholds for phase transitions
    _DONE_THRESHOLD = 0.8
    _LOST_THRESHOLD = 0.7
    # AT_OBJECT threshold for PICKUP feasibility
    _AT_OBJ_THRESHOLD = 0.3

    def __init__(
        self,
        planner: TaskPlanner,
        gated_tom: GatedToM,
        empathy_factor: float = 0.0,
        world_bounds: Tuple[float, float, float, float] = (-8.5, 0.5, -6.0, 0.5),
        beta: float = 4.0,
    ) -> None:
        self._planner = planner
        self._tom = gated_tom
        self._commitment = CommitmentInference()
        self._planner._commitment = self._commitment
        self._empathy_factor = empathy_factor
        self._beta = beta

        # Generative model
        self._A = build_A_matrices()
        self._B_base = build_B_matrices()
        self._C = build_C_vectors(empathy_factor)
        self._D = build_D_vector()

        # Belief state
        self._q_s = self._D.copy()
        self._prev_policy: Optional[int] = None
        self._target_obj: Optional[ObjectState] = None
        self._target_approach: Optional[Tuple[float, float]] = None

        # VFE / policy entropy (for EmpathicModulator)
        self._vfe_raw = 0.0
        self._vfe_smooth = 0.0
        self._policy_entropy_val = 0.0
        self._self_arousal: Optional[float] = None

        # Yield waypoint safety
        self._world_bounds = world_bounds
        self._yield_target: Optional[Tuple[float, float]] = None
        self._yield_persist_counter = 0

        # Motion tracking (from ClearTableManager, reused for obs context)
        self._prev_other_dist: Dict[str, float] = {}
        self._prev_t: float = 0.0

        # Yield penalties (deadlock evidence, passed to TaskPlanner)
        self._yield_count: Dict[str, int] = {}
        self._yield_penalties: Dict[str, float] = {}

        # Skill outcome observation (stored by notify_status, consumed by next tick)
        self._last_skill_outcome: int = int(SkillOutcomeObs.NONE)
        # Track last emitted skill type so notify_status only stores
        # outcomes for manipulation skills (pick_place).  Navigation
        # outcomes are already captured by distance observations —
        # feeding nav SUCCESS into the skill_outcome modality would
        # incorrectly shift beliefs toward TRANSPORT/DONE.
        self._last_emitted_skill: str = "navigate"

        # Skill commitment (HAIF "plan-then-execute" pattern):
        # Once PICKUP or PLACE is selected, keep returning the same
        # SkillRequest until notify_status() reports terminal status.
        self._committed_req: Optional[SkillRequest] = None
        self._committed_phase: Optional[str] = None  # "pick" or "place"

        # Cache other agent position for use in RESELECT
        self._last_other_pos: Optional[Tuple[float, float]] = None
        self._last_other_id: Optional[str] = None

        # Accumulated RESELECT evidence (requires sustained contested/lost)
        self._reselect_evidence: float = 0.0
        self._RESELECT_EVIDENCE_THRESHOLD = 0.60
        self._RESELECT_DECAY = 0.85

        # Pickup failure tracking
        self._pickup_failures: Dict[str, int] = {}
        self._unavailable_until: Dict[str, int] = {}
        self._tick_counter: int = 0
        self._MAX_PICKUP_FAILURES = 3
        self._UNAVAILABLE_TICKS = 50  # ~10 seconds at 5Hz deliberation

        # Recognizer ref (for Executive situation detection)
        self._recognizer_ref = None

    # ------------------------------------------------------------------
    # ScriptManager-compatible interface
    # ------------------------------------------------------------------
    @property
    def is_complete(self) -> bool:
        return self._planner.is_task_complete

    @property
    def is_repairing(self) -> bool:
        return False

    _STATUS_TO_OBS = {"SUCCESS": 1, "FAILURE": 2, "TIMEOUT": 3}

    def notify_status(self, status) -> None:
        """Store skill outcome and clear commitment on terminal status.

        Only manipulation outcomes (pick_place) are stored as observations.
        Navigation outcomes are captured by distance observations instead.

        Commitment is cleared on any terminal status so the next select()
        call will run full inference again.
        """
        if self._last_emitted_skill == "pick_place":
            self._last_skill_outcome = self._STATUS_TO_OBS.get(status, 0)

        # Clear commitment on terminal status
        if self._committed_req is not None:
            print(f"  [UTC] commitment released: {status}")
            # Track failures per target
            if status in ("FAILURE", "TIMEOUT") and self._target_obj is not None:
                obj_id = self._target_obj.id
                self._pickup_failures[obj_id] = self._pickup_failures.get(obj_id, 0) + 1
                if self._pickup_failures[obj_id] >= self._MAX_PICKUP_FAILURES:
                    self._unavailable_until[obj_id] = (
                        self._tick_counter + self._UNAVAILABLE_TICKS
                    )
                    print(f"  [UTC] {obj_id} marked unavailable"
                          f" for {self._UNAVAILABLE_TICKS} ticks")
            self._committed_req = None
            self._committed_phase = None

    def check_violation(self, pb: PerceptBundle, timestamp: float = 0.0):
        return None

    def set_self_arousal(self, arousal: Optional[float]) -> None:
        self._self_arousal = arousal

    @property
    def _recognizer(self):
        return self._recognizer_ref

    # New properties for EmpathicModulator
    @property
    def vfe(self) -> float:
        """Smoothed VFE for EmpathicModulator (replaces G proxy)."""
        return self._vfe_smooth

    @property
    def policy_entropy(self) -> float:
        """H[q(pi)] from EFE softmax — drives arousal."""
        return self._policy_entropy_val

    # ------------------------------------------------------------------
    # Core: POMDP-based task control
    # ------------------------------------------------------------------
    def select(
        self,
        pb: PerceptBundle,
        active: Optional[SkillRequest],
    ) -> SkillRequest:
        """Main deliberation entry point.

        Unlike ClearTableManager, active=None does NOT trigger reselection.
        Target persistence is internal belief state, not Executive's active_req.
        """
        # 1. Update object states from perception
        objects_data = pb.world.get("objects", [])
        objects: List[ObjectState] = []
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

        # 2. Tick counter + expire unavailable targets
        self._tick_counter += 1
        expired = [oid for oid, until in self._unavailable_until.items()
                   if self._tick_counter >= until]
        for oid in expired:
            del self._unavailable_until[oid]
            self._pickup_failures.pop(oid, None)

        # 3. Extract positions
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

        # Cache for RESELECT
        self._last_other_pos = other_pos
        self._last_other_id = other_id

        # 3. Feed commitment inference
        if other_id and other_pos and objects:
            obj_positions = {
                o.id: o.position for o in objects if o.status == "on_table"
            }
            if obj_positions:
                self._commitment.update(
                    other_id, other_pos,
                    agents[0].get("alpha", 0.0),
                    obj_positions,
                )

        # 4. If no target → select one via TaskPlanner
        if self._target_obj is None:
            self._select_target(pb, robot_pos, other_pos, other_id)
            if self._target_obj is None:
                # All done or no objects — hold position
                return SkillRequest(
                    skill="navigate",
                    goal={"x": robot_pos[0], "y": robot_pos[1]},
                )

        # 5. Honour skill commitment (HAIF "plan-then-execute" pattern)
        # Once PICKUP or PLACE is emitted, keep returning the same request
        # until notify_status() clears the commitment.
        if self._committed_req is not None:
            return self._committed_req

        # 6. Run inference loop
        return self._inference_tick(pb, robot_pos, other_pos, other_id, objects)

    # ------------------------------------------------------------------
    # Target selection (delegates to TaskPlanner)
    # ------------------------------------------------------------------
    def _select_target(
        self,
        pb: PerceptBundle,
        robot_pos: Tuple[float, float],
        other_pos: Optional[Tuple[float, float]],
        other_id: Optional[str],
    ) -> None:
        """Use TaskPlanner.select_next() to choose which object to pursue."""
        obs = self._build_obs_context(pb, robot_pos, other_pos, other_id)
        q_other, profile = self._get_tom_predictions(other_id, obs)

        # Decay yield penalties
        for obj_id in list(self._yield_penalties):
            self._yield_penalties[obj_id] *= 0.9
            if self._yield_penalties[obj_id] < 0.1:
                del self._yield_penalties[obj_id]

        # Penalise targets with repeated failures (effectively unavailable)
        for obj_id, until in self._unavailable_until.items():
            if self._tick_counter < until:
                self._yield_penalties[obj_id] = max(
                    self._yield_penalties.get(obj_id, 0.0), 100.0,
                )

        furniture = self._extract_furniture(pb)

        obj, rollout = self._planner.select_next(
            robot_pos=robot_pos,
            obs=obs,
            q_other=q_other,
            profile=profile,
            obstacles=furniture,
            yield_penalties=self._yield_penalties,
            other_pos=other_pos,
            self_arousal=self._self_arousal,
        )

        if obj is not None:
            self._target_obj = obj
            self._target_approach = self._planner._current_approach or obj.position
            # Reset beliefs for new target
            self._q_s = self._D.copy()
            self._prev_policy = None
            self._yield_target = None
            self._yield_persist_counter = 0
            self._reselect_evidence = 0.0
            print(f"  [UTC] target={obj.id} pos={obj.position}"
                  f" approach={self._target_approach}"
                  f" robot=({robot_pos[0]:.2f},{robot_pos[1]:.2f})")

    # ------------------------------------------------------------------
    # POMDP inference loop
    # ------------------------------------------------------------------
    def _inference_tick(
        self,
        pb: PerceptBundle,
        robot_pos: Tuple[float, float],
        other_pos: Optional[Tuple[float, float]],
        other_id: Optional[str],
        objects: List[ObjectState],
    ) -> SkillRequest:
        """Run one tick of the POMDP inference loop."""

        # a. Extract observations
        obs = self._extract_observations(pb, robot_pos, objects, other_id)

        # b. Distance-condition B matrices
        d_obj = self._distance_to_target(robot_pos)
        d_drop = self._distance_to_dropoff(robot_pos)
        B = update_B_from_distances(self._B_base, d_obj, d_drop)

        # c. Predict prior: P(s_t | s_{t-1}, pi)
        if self._prev_policy is not None and self._prev_policy in B:
            prior = B[self._prev_policy] @ self._q_s
        else:
            prior = self._q_s.copy()

        # d. Belief update
        posterior, F = belief_update(prior, obs, self._A)
        self._q_s = posterior

        # e. Smooth VFE
        self._vfe_raw = F
        self._vfe_smooth = (
            self._VFE_ALPHA * F + (1.0 - self._VFE_ALPHA) * self._vfe_smooth
        )

        # f. Build social G (from SocialEFE for task-relevant policies)
        social_G = self._compute_social_G(pb, robot_pos, other_pos, other_id)

        # g. Build policy mask
        mask = self._build_policy_mask(pb, robot_pos)

        # h. Evaluate EFE (T=2 lookahead: NAV_OBJ sees downstream PICKUP value)
        G = evaluate_policies_multistep(posterior, B, self._C, self._A, social_G)

        # i. Apply mask
        if mask is not None:
            G[~mask] = 1e10

        # j. Select policy
        arousal = self._self_arousal if self._self_arousal is not None else 0.5
        selected, q_pi, H_pi = select_policy(G, self._beta, arousal)

        # k. Store
        self._prev_policy = selected
        self._policy_entropy_val = H_pi

        # l. Debug: log selected policy and belief state
        policy_name = TaskPolicy(selected).name if selected < len(TaskPolicy) else f"?{selected}"
        p_done = sum(
            self._q_s[state_index(TaskPhase.DONE, h, m)]
            for h in HandState for m in TargetMode
        )
        p_lost = sum(
            self._q_s[state_index(p, h, TargetMode.LOST)]
            for p in TaskPhase for h in HandState
        )
        G_str = " ".join(f"{TaskPolicy(i).name}={G[i]:.2f}" for i in range(N_POLICIES))
        print(f"  [UTC-DBG] policy={policy_name} obs=({obs.d_obj},{obs.d_drop},{obs.arm},{obs.hold},{obs.obj_z},{obs.social},{obs.target},{obs.skill_outcome})"
              f" p_done={p_done:.3f} p_lost={p_lost:.3f}")
        print(f"  [UTC-DBG] G: {G_str}")

        # m. Check for phase completion / target loss
        if self._check_done():
            print(f"  [UTC-DBG] DONE triggered — clearing target")
            return SkillRequest(
                skill="navigate",
                goal={"x": robot_pos[0], "y": robot_pos[1]},
            )
        if self._check_lost():
            print(f"  [UTC-DBG] LOST triggered — clearing target")
            return SkillRequest(
                skill="navigate",
                goal={"x": robot_pos[0], "y": robot_pos[1]},
            )

        # n. Track yield counts for deadlock evidence
        self._update_yield_tracking(selected, other_id)

        # o. Execute selected policy
        return self._execute_policy(selected, pb, robot_pos)

    # ------------------------------------------------------------------
    # Observation extraction
    # ------------------------------------------------------------------
    def _extract_observations(
        self,
        pb: PerceptBundle,
        robot_pos: Tuple[float, float],
        objects: List[ObjectState],
        other_id: Optional[str],
    ) -> Observation:
        """Build discrete observation vector from PerceptBundle."""
        # d_obj: distance to target object
        d_obj = self._distance_to_target(robot_pos)
        o_d_obj = discretize_distance(d_obj)

        # d_drop: distance to dropoff
        d_drop = self._distance_to_dropoff(robot_pos)
        o_d_drop = discretize_distance(d_drop)

        # arm: proximity to TARGET object (not any object — arm_at_target
        # from sensors is not target-specific, causing false PICKUP triggers)
        o_arm = 1 if d_obj < 0.6 else 0

        # hold: holding object
        o_hold = 1 if pb.world.get("held_object", False) else 0

        # obj_z: target object z-height status
        o_obj_z = self._target_obj_z_status(objects)

        # social: commitment belief for target
        o_social = 0
        if other_id and self._target_obj:
            belief = self._commitment.get_belief(other_id, self._target_obj.id)
            o_social = discretize_commitment(belief)

        # target: object availability
        o_target = self._target_availability(objects)

        # skill_outcome: last skill result (stored by notify_status)
        o_skill = self._last_skill_outcome
        self._last_skill_outcome = int(SkillOutcomeObs.NONE)  # consume

        return Observation(
            d_obj=o_d_obj,
            d_drop=o_d_drop,
            arm=o_arm,
            hold=o_hold,
            obj_z=o_obj_z,
            social=o_social,
            target=o_target,
            skill_outcome=o_skill,
        )

    def _target_obj_z_status(self, objects: List[ObjectState]) -> int:
        """Map target object status to ObjZObs."""
        if self._target_obj is None:
            return int(ObjZObs.TABLE)
        for o in objects:
            if o.id == self._target_obj.id:
                if o.status == "held":
                    return int(ObjZObs.HELD)
                elif o.status == "placed":
                    return int(ObjZObs.PLACED)
                return int(ObjZObs.TABLE)
        return int(ObjZObs.TABLE)

    def _target_availability(self, objects: List[ObjectState]) -> int:
        """Map target object availability to TargetObs."""
        if self._target_obj is None:
            return int(TargetObs.AVAILABLE)
        for o in objects:
            if o.id == self._target_obj.id:
                if o.status == "held" and o.held_by is not None:
                    # Check if held by OTHER agent
                    return int(TargetObs.HELD_OTHER)
                if o.status == "placed":
                    return int(TargetObs.GONE)
                return int(TargetObs.AVAILABLE)
        # Object not found in list
        return int(TargetObs.GONE)

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------
    def _distance_to_target(self, robot_pos: Tuple[float, float]) -> float:
        if self._target_approach is None:
            return 10.0
        dx = robot_pos[0] - self._target_approach[0]
        dy = robot_pos[1] - self._target_approach[1]
        return math.sqrt(dx * dx + dy * dy)

    def _distance_to_dropoff(self, robot_pos: Tuple[float, float]) -> float:
        drop = self._planner.drop_off
        dx = robot_pos[0] - drop[0]
        dy = robot_pos[1] - drop[1]
        return math.sqrt(dx * dx + dy * dy)

    # ------------------------------------------------------------------
    # Social EFE integration
    # ------------------------------------------------------------------
    # Policy → social intent mapping for SocialEFE
    _POLICY_INTENT = {
        TaskPolicy.NAV_OBJ: "approach",
        TaskPolicy.YIELD: "yield",
        TaskPolicy.WAIT: "wait",
    }

    def _compute_social_G(
        self,
        pb: PerceptBundle,
        robot_pos: Tuple[float, float],
        other_pos: Optional[Tuple[float, float]],
        other_id: Optional[str],
    ) -> Dict[int, float]:
        """Compute social EFE terms for task-relevant policies.

        Uses single-step SocialEFE (not multi-step rollout) so the scale
        is naturally compatible with the task POMDP's base EFE (~0.5–5.0).
        """
        social_G: Dict[int, float] = {}
        if other_pos is None or other_id is None:
            return social_G

        obs = self._build_obs_context(pb, robot_pos, other_pos, other_id)
        q_other, profile = self._get_tom_predictions(other_id, obs)

        try:
            output = self._planner._efe.compute(
                q_human=q_other,
                obs=obs,
                self_arousal=self._self_arousal,
            )
            g_by_intent = {a.intent: a.g_social for a in output.actions}
            for pi, intent in self._POLICY_INTENT.items():
                social_G[int(pi)] = g_by_intent.get(intent, 0.0)
        except Exception:
            pass

        return social_G

    # ------------------------------------------------------------------
    # Policy mask
    # ------------------------------------------------------------------
    def _build_policy_mask(
        self,
        pb: PerceptBundle,
        robot_pos: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        """Build boolean mask: True = feasible, False = masked."""
        mask = np.ones(N_POLICIES, dtype=bool)

        # PICKUP only if beliefs support AT_OBJECT + EMPTY
        p_at_obj_empty = sum(
            self._q_s[state_index(TaskPhase.AT_OBJECT, HandState.EMPTY, m)]
            for m in TargetMode
        )
        if p_at_obj_empty < self._AT_OBJ_THRESHOLD:
            mask[int(TaskPolicy.PICKUP)] = False

        # NAV_DROP only if beliefs support HOLDING (any phase)
        # From APPROACH/EMPTY, NAV_DROP is identity (no advance defined)
        # but gets g_social=0, so it unfairly beats NAV_OBJ.
        p_holding = sum(
            self._q_s[state_index(p, HandState.HOLDING, m)]
            for p in TaskPhase for m in TargetMode
        )
        if p_holding < self._AT_OBJ_THRESHOLD:
            mask[int(TaskPolicy.NAV_DROP)] = False

        # PLACE only if beliefs support AT_DROPOFF + HOLDING
        p_at_drop_hold = sum(
            self._q_s[state_index(TaskPhase.AT_DROPOFF, HandState.HOLDING, m)]
            for m in TargetMode
        )
        if p_at_drop_hold < self._AT_OBJ_THRESHOLD:
            mask[int(TaskPolicy.PLACE)] = False

        # RESELECT only if sustained evidence of contention or target loss.
        # Accumulate evidence over multiple ticks to prevent single-tick
        # triggers from noisy observations.
        p_contested_lost = sum(
            self._q_s[state_index(p, h, m)]
            for p in TaskPhase for h in HandState
            for m in (TargetMode.CONTESTED, TargetMode.LOST)
        )
        self._reselect_evidence = (
            self._RESELECT_DECAY * self._reselect_evidence
            + (1.0 - self._RESELECT_DECAY) * p_contested_lost
        )
        if self._reselect_evidence < self._RESELECT_EVIDENCE_THRESHOLD:
            mask[int(TaskPolicy.RESELECT)] = False

        # YIELD only if safe waypoint exists
        furniture = pb.world.get("furniture", [])
        other_pos = None
        agents = pb.world.get("agents", [])
        if agents:
            ap = agents[0].get("pose", (0, 0))
            other_pos = (ap[0], ap[1])

        yield_tgt = self._get_safe_yield_target(robot_pos, other_pos, furniture)
        if yield_tgt is None:
            mask[int(TaskPolicy.YIELD)] = False

        return mask

    # ------------------------------------------------------------------
    # Safe yield waypoint generation
    # ------------------------------------------------------------------
    def _get_safe_yield_target(
        self,
        robot_pos: Tuple[float, float],
        other_pos: Optional[Tuple[float, float]],
        furniture: list,
    ) -> Optional[Tuple[float, float]]:
        """Compute a safe yield target with hysteresis.

        Returns None if no safe target exists (robot cornered).
        """
        # Hysteresis: reuse previous target if within persistence window
        if (
            self._yield_target is not None
            and self._yield_persist_counter < self._YIELD_PERSIST_TICKS
        ):
            self._yield_persist_counter += 1
            return self._yield_target

        if other_pos is None:
            self._yield_target = None
            return None

        x_min, x_max, y_min, y_max = self._world_bounds
        margin = 0.5

        # Perpendicular sidestep from bearing to other robot
        bearing = math.atan2(
            other_pos[1] - robot_pos[1],
            other_pos[0] - robot_pos[0],
        )

        best_target = None
        best_clearance = -1.0

        for sign in (+1, -1):
            perp = bearing + sign * math.pi / 2
            raw_x = robot_pos[0] + 0.6 * math.cos(perp)
            raw_y = robot_pos[1] + 0.6 * math.sin(perp)

            # Clamp to world interior
            tx = max(x_min + margin, min(x_max - margin, raw_x))
            ty = max(y_min + margin, min(y_max - margin, raw_y))

            # Check against furniture keepout zones
            in_keepout = False
            for f in furniture:
                kx_min = f.get("keepout_x_min", 0)
                ky_min = f.get("keepout_y_min", 0)
                kx_max = f.get("keepout_x_max", 0)
                ky_max = f.get("keepout_y_max", 0)
                if kx_min <= tx <= kx_max and ky_min <= ty <= ky_max:
                    in_keepout = True
                    break

            if in_keepout:
                continue

            # Compute clearance (min distance to any boundary/keepout)
            clearance = min(
                tx - (x_min + margin),
                (x_max - margin) - tx,
                ty - (y_min + margin),
                (y_max - margin) - ty,
            )
            for f in furniture:
                kx_min = f.get("keepout_x_min", 0)
                ky_min = f.get("keepout_y_min", 0)
                kx_max = f.get("keepout_x_max", 0)
                ky_max = f.get("keepout_y_max", 0)
                cx = max(kx_min, min(kx_max, tx))
                cy = max(ky_min, min(ky_max, ty))
                d = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
                clearance = min(clearance, d)

            if clearance > best_clearance:
                best_clearance = clearance
                best_target = (tx, ty)

        self._yield_target = best_target
        self._yield_persist_counter = 0
        return best_target

    # ------------------------------------------------------------------
    # Phase completion / target loss checks
    # ------------------------------------------------------------------
    def _check_done(self) -> bool:
        """Check if beliefs indicate task phase DONE for current target."""
        p_done = sum(
            self._q_s[state_index(TaskPhase.DONE, h, m)]
            for h in HandState for m in TargetMode
        )
        if p_done > self._DONE_THRESHOLD:
            obj_id = self._target_obj.id if self._target_obj else None
            if obj_id:
                self._yield_count.pop(obj_id, None)
                self._pickup_failures.pop(obj_id, None)
            self._target_obj = None
            self._target_approach = None
            self._q_s = self._D.copy()
            self._prev_policy = None
            self._yield_target = None
            self._reselect_evidence = 0.0
            return True
        return False

    def _check_lost(self) -> bool:
        """Check if beliefs indicate target is LOST."""
        p_lost = sum(
            self._q_s[state_index(p, h, TargetMode.LOST)]
            for p in TaskPhase for h in HandState
        )
        if p_lost > self._LOST_THRESHOLD:
            self._target_obj = None
            self._target_approach = None
            self._q_s = self._D.copy()
            self._prev_policy = None
            self._yield_target = None
            self._reselect_evidence = 0.0
            return True
        return False

    # ------------------------------------------------------------------
    # Yield tracking / deadlock evidence
    # ------------------------------------------------------------------
    def _update_yield_tracking(self, selected: int, other_id: Optional[str]) -> None:
        if self._target_obj is None:
            return
        obj_id = self._target_obj.id

        if selected in (int(TaskPolicy.YIELD), int(TaskPolicy.WAIT)):
            self._yield_count[obj_id] = self._yield_count.get(obj_id, 0) + 1
            n_yields = self._yield_count[obj_id]
            p_contend = (
                self._commitment.get_belief(other_id, obj_id) if other_id else 0.0
            )
            p_deadlock = min(1.0, (n_yields / 10.0) * max(0.3, p_contend))
            if p_deadlock > 0.1:
                self._yield_penalties[obj_id] = p_deadlock * 3.0
        else:
            self._yield_count.pop(obj_id, None)

    # ------------------------------------------------------------------
    # Policy execution
    # ------------------------------------------------------------------
    def _execute_policy(
        self,
        selected: int,
        pb: PerceptBundle,
        robot_pos: Tuple[float, float],
    ) -> SkillRequest:
        """Map selected policy to SkillRequest."""
        if selected == int(TaskPolicy.NAV_OBJ):
            self._last_emitted_skill = "navigate"
            if self._target_approach is None:
                return SkillRequest(
                    skill="navigate",
                    goal={"x": robot_pos[0], "y": robot_pos[1]},
                )
            return SkillRequest(
                skill="navigate",
                goal={"x": self._target_approach[0], "y": self._target_approach[1]},
            )

        elif selected == int(TaskPolicy.PICKUP):
            self._last_emitted_skill = "pick_place"
            obj_id = self._target_obj.id if self._target_obj else ""
            obj_pos = self._target_obj.position if self._target_obj else ()
            req = SkillRequest(
                skill="pick_place",
                goal={"object": obj_id, "position": obj_pos},
                params={"mode": "pick"},
            )
            self._committed_req = req
            self._committed_phase = "pick"
            print(f"  [UTC] COMMITTED to PICKUP {obj_id}")
            return req

        elif selected == int(TaskPolicy.NAV_DROP):
            self._last_emitted_skill = "navigate"
            drop = self._planner.drop_off
            return SkillRequest(
                skill="navigate",
                goal={"x": drop[0], "y": drop[1]},
            )

        elif selected == int(TaskPolicy.PLACE):
            self._last_emitted_skill = "pick_place"
            drop = self._planner.drop_off
            obj_id = self._target_obj.id if self._target_obj else ""
            req = SkillRequest(
                skill="pick_place",
                goal={"object": obj_id, "position": (drop[0], drop[1])},
                params={"mode": "place"},
            )
            self._committed_req = req
            self._committed_phase = "place"
            print(f"  [UTC] COMMITTED to PLACE {obj_id}")
            return req

        elif selected == int(TaskPolicy.YIELD):
            self._last_emitted_skill = "navigate"
            tgt = self._yield_target or robot_pos
            return SkillRequest(
                skill="navigate",
                goal={"x": tgt[0], "y": tgt[1]},
                params={"speed_scale": 0.3},
            )

        elif selected == int(TaskPolicy.WAIT):
            self._last_emitted_skill = "navigate"
            # Use approach_pos (not current_pos) to avoid instant SUCCESS
            # from dist_to_goal ≈ 0.  Speed control comes from ToM/IntentPolicy.
            tgt = self._target_approach or robot_pos
            return SkillRequest(
                skill="navigate",
                goal={"x": tgt[0], "y": tgt[1]},
            )

        elif selected == int(TaskPolicy.RESELECT):
            self._last_emitted_skill = "navigate"
            # Clear current target and immediately select a new one
            old_target_id = self._target_obj.id if self._target_obj else None
            self._target_obj = None
            self._target_approach = None
            self._q_s = self._D.copy()
            self._prev_policy = None
            self._yield_target = None
            self._reselect_evidence = 0.0

            # Immediately select new target (avoid navigate-to-self deadlock)
            self._select_target(
                pb, robot_pos, self._last_other_pos, self._last_other_id,
            )

            if (self._target_obj is not None
                    and self._target_obj.id != old_target_id):
                print(f"  [UTC] RESELECT -> new target {self._target_obj.id}")
                return SkillRequest(
                    skill="navigate",
                    goal={"x": self._target_approach[0],
                          "y": self._target_approach[1]},
                )
            elif self._target_obj is not None:
                # Same target re-selected — wait instead of navigate-to-self
                print(f"  [UTC] RESELECT -> same target, waiting")
                tgt = self._target_approach or robot_pos
                return SkillRequest(
                    skill="navigate",
                    goal={"x": tgt[0], "y": tgt[1]},
                )
            else:
                # No targets available — hold position
                return SkillRequest(
                    skill="navigate",
                    goal={"x": robot_pos[0], "y": robot_pos[1]},
                )

        # Fallback
        self._last_emitted_skill = "navigate"
        return SkillRequest(
            skill="navigate",
            goal={"x": robot_pos[0], "y": robot_pos[1]},
        )

    # ------------------------------------------------------------------
    # Belief manipulation helpers
    # ------------------------------------------------------------------
    def _nudge_beliefs(
        self,
        target_phase: int,
        target_hand: int,
        strength: float,
    ) -> None:
        """Shift probability mass toward a specific phase/hand combination."""
        mass = 0.0
        for m in TargetMode:
            s = state_index(target_phase, target_hand, m)
            mass += self._q_s[s]

        if mass < 0.99:
            transfer = strength * (1.0 - mass)
            scale = 1.0 - transfer
            self._q_s *= scale
            for m in TargetMode:
                s = state_index(target_phase, target_hand, m)
                self._q_s[s] += transfer / len(TargetMode)
            self._q_s = np.clip(self._q_s, 1e-16, None)
            self._q_s /= self._q_s.sum()

    def _nudge_target_mode(self, target_mode: int, strength: float) -> None:
        """Shift probability mass toward a specific target_mode."""
        current_mass = sum(
            self._q_s[state_index(p, h, target_mode)]
            for p in TaskPhase for h in HandState
        )
        if current_mass < 0.99:
            transfer = strength * (1.0 - current_mass)
            for s in range(N_STATES):
                p, h, m = state_factors(s)
                if m == target_mode:
                    self._q_s[s] += transfer / (N_STATES // len(TargetMode))
                else:
                    self._q_s[s] *= (1.0 - transfer)
            self._q_s = np.clip(self._q_s, 1e-16, None)
            self._q_s /= self._q_s.sum()

    def _phase_hand_marginal(self, hand: int) -> float:
        """Marginal probability of a hand state."""
        return sum(
            self._q_s[state_index(p, hand, m)]
            for p in TaskPhase for m in TargetMode
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def belief_summary(self) -> str:
        """Human-readable summary of dominant belief."""
        best_s = int(np.argmax(self._q_s))
        p, h, m = state_factors(best_s)
        phase_name = TaskPhase(p).name
        hand_name = HandState(h).name
        mode_name = TargetMode(m).name
        prob = self._q_s[best_s]
        return f"{phase_name}({prob:.2f})"

    # ------------------------------------------------------------------
    # ToM + observation context (reused from ClearTableManager)
    # ------------------------------------------------------------------
    def _get_tom_predictions(
        self,
        other_id: Optional[str],
        obs: ObservationContext,
    ) -> Tuple[Dict[str, float], IntentProfile]:
        """Get ToM predictions for the other agent."""
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
            from architecture_core.cognition.tom.gated_tom import _DEFAULT_PROFILE
            profile = IntentProfile(
                approach_bias=rel * learned.approach_bias + (1 - rel) * _DEFAULT_PROFILE.approach_bias,
                responsiveness=rel * learned.responsiveness + (1 - rel) * _DEFAULT_PROFILE.responsiveness,
                precision=rel * learned.precision + (1 - rel) * _DEFAULT_PROFILE.precision,
                empathy_j=rel * learned.empathy_j + (1 - rel) * _DEFAULT_PROFILE.empathy_j,
            )

        return q_other, profile

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

            oid = other_id or "unknown"
            dt = max(0.01, pb.t - self._prev_t) if self._prev_t > 0 else 0.1
            prev_dist = self._prev_other_dist.get(oid, distance)
            velocity = max(0.0, (prev_dist - distance) / dt)
            approaching = velocity > 0.02

            self._prev_other_dist[oid] = distance
            self._prev_t = pb.t

        valence = 0.0
        arousal = 0.0
        affect = pb.social.get("affect", {})
        if affect:
            valence = affect.get("valence", 0.0)
            arousal = affect.get("arousal", 0.0)

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
