"""Task-level planning via Expected Free Energy.

Selects which object to pick next by evaluating the full EFE for each
candidate policy π_x = "pursue object x":

  G(π_x) = G_risk + G_information + G_social

- G_risk = −ln p(o|C):  Boltzmann preference over cost-to-go.
  Cost-to-go = nav + clearing + drop-off distance (+ yield penalty
  as a temporary preference shift away from recently contested objects).
  Mapped to risk via inverse temperature β: G_risk = β · cost.

- G_information = H[p(o|s)] − I(o;s):  Net information term.
  Ambiguity (outcome entropy) and epistemic value (info gain about
  self-model precision) are coupled — both depend on the same hidden
  state (clearing action precision).  Ambiguity pushes away from
  uncertain outcomes; info gain pulls toward them for exploration.
  Combined as: H[Bernoulli(q_success)] + epistemic_value.

- G_social = V*(step 0) from SocialEFE backward induction.  Already a
  proper EFE with empathy-weighted social reward, collision cost, and
  discounted future value.  Commitment inference feeds in via the
  initial_obstruction parameter — high inferred commitment of the other
  agent to this object → high obstruction → SocialEFE naturally penalizes
  via collision cost and favours yield/wait → higher G_social.

Affordance-aware planning:
- Computes approach positions at arm_reach (not raw object positions)
- Evaluates clearing actions (push, grab-and-pull) via EFE
- Generates sequences with optional obstacle-clearing steps

All imports are robot-agnostic (architecture_core only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from architecture_core.core.types import Intent, SkillRequest
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep
from architecture_core.cognition.tom.intent_particle_filter import (
    IntentProfile,
    ObservationContext,
)
from architecture_core.cognition.tom.social_efe import (
    RolloutConfig,
    RolloutEFEOutput,
    SocialEFE,
)
from architecture_core.cognition.planning.affordance import (
    ApproachCandidate,
    ClearingAction,
    ClearingPlan,
    EmbodimentModel,
    FurnitureItem,
    SceneGraph,
    compute_approach_candidates,
)
from architecture_core.cognition.planning.commitment_inference import (
    CommitmentInference,
)
from architecture_core.safety.geometry import Rect, point_distance

_EPS = 1e-10


def _binary_entropy(p: float) -> float:
    """H[Bernoulli(p)] — outcome uncertainty for binary success/fail."""
    p = max(_EPS, min(1.0 - _EPS, p))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


# =====================================================================
# Object state
# =====================================================================
@dataclass
class ObjectState:
    """State of one manipulable object in the environment."""
    id: str                             # e.g. "orange_1"
    type: str                           # e.g. "Orange"
    position: Tuple[float, float]       # world (x, y)
    table: str                          # e.g. "dining" or "coffee"
    status: str = "on_table"            # "on_table", "held", "placed"
    held_by: Optional[str] = None       # robot name or None


# Default table regions (apartment world)
DEFAULT_TABLE_REGIONS: Dict[str, Dict] = {
    "dining": {"center": (-1.074, -4.944), "radius": 0.8},
    "coffee": {"center": (-7.163, -2.555), "radius": 0.8},
}


# =====================================================================
# TaskPlanner
# =====================================================================
class TaskPlanner:
    """EFE-driven task planner for multi-object clearing.

    Uses the same SocialEFE rollout as navigation to evaluate each
    candidate object.  The rollout naturally captures:
    - Social cost: other robot near the object → high EFE
    - Collision risk: path crosses other robot's trajectory
    - Affect: stressed → prefer farther/safer objects
    - Empathy: high-empathy robot avoids objects that would force
      the other robot to yield

    Parameters
    ----------
    drop_off : tuple[float, float]
        Position where objects should be placed.
    efe : SocialEFE
        Same EFE instance used for navigation rollout.
    empathy_factor : float
        Robot's own empathy (for contention weighting).
    table_regions : dict | None
        Table name → {"center": (x,y), "radius": float}.
    """

    def __init__(
        self,
        drop_off: Tuple[float, float],
        efe: SocialEFE,
        empathy_factor: float = 0.0,
        table_regions: Optional[Dict[str, Dict]] = None,
        embodiment: Optional[EmbodimentModel] = None,
        commitment: Optional[CommitmentInference] = None,
        beta_pragmatic: float = 0.3,
    ) -> None:
        self.drop_off = drop_off
        self._efe = efe
        self._empathy = empathy_factor
        self._table_regions = table_regions or DEFAULT_TABLE_REGIONS
        self._embodiment = embodiment or EmbodimentModel()
        self._commitment = commitment
        self._beta_pragmatic = beta_pragmatic
        self._other_agent_id: Optional[str] = None
        self._objects: List[ObjectState] = []
        self._current_target: Optional[ObjectState] = None
        self._current_approach: Optional[Tuple[float, float]] = None
        self._current_clearing: List[ClearingPlan] = []

    # ------------------------------------------------------------------
    # Object state management
    # ------------------------------------------------------------------
    def update_objects(self, objects: List[ObjectState]) -> None:
        """Refresh object states from sensors."""
        self._objects = list(objects)

    @property
    def available_objects(self) -> List[ObjectState]:
        """Objects still on a table (not held or placed)."""
        return [o for o in self._objects if o.status == "on_table"]

    @property
    def is_task_complete(self) -> bool:
        """True when all objects have been placed."""
        return len(self._objects) > 0 and all(
            o.status == "placed" for o in self._objects
        )

    @property
    def current_target(self) -> Optional[ObjectState]:
        return self._current_target

    def update_other_agent(
        self,
        agent_id: str,
        agent_pos: Tuple[float, float],
        agent_heading: float,
    ) -> None:
        """Feed observable cues about the other agent to commitment inference."""
        self._other_agent_id = agent_id
        if self._commitment is not None:
            objects = {
                o.id: o.position for o in self._objects if o.status == "on_table"
            }
            if objects:
                self._commitment.update(agent_id, agent_pos, agent_heading, objects)

    # ------------------------------------------------------------------
    # EFE-driven object selection
    # ------------------------------------------------------------------
    def select_next(
        self,
        robot_pos: Tuple[float, float],
        obs: ObservationContext,
        q_other: Dict[str, float],
        profile: IntentProfile,
        initial_obstruction: float = 1.0,
        rollout_config: Optional[RolloutConfig] = None,
        obstacles: Optional[List[FurnitureItem]] = None,
        yield_penalties: Optional[Dict[str, float]] = None,
        other_pos: Optional[Tuple[float, float]] = None,
        self_arousal: Optional[float] = None,
    ) -> Tuple[Optional[ObjectState], Optional[RolloutEFEOutput]]:
        """Select the best (object, approach_pos) by minimising EFE.

        G(π_x) = G_risk + G_information + G_social

        For each available object:
          1. Compute approach candidates at arm_reach distance
          2. For each feasible candidate:
             - G_risk = β · cost_to_go  (−ln p(o|C), Boltzmann preference)
             - G_information = H[Bernoulli(q_success)] + epistemic_value
             - G_social = V* from SocialEFE backward induction
          3. Select (object, candidate) with lowest total EFE

        Returns
        -------
        (selected_object, rollout_output) or (None, None) if no objects.
        rollout_output.selected indicates immediate action:
          - "wait" -> robot defers (observation phase)
          - "approach" -> robot commits
        """
        available = self.available_objects
        if not available:
            self._current_target = None
            self._current_approach = None
            self._current_clearing = []
            return None, None

        obstacles = obstacles or []

        # Build scene graph: infer object-surface relations and
        # check reachability (arm_reach vs surface size)
        scene = SceneGraph(furniture=obstacles, objects=available)
        self._scene = scene

        best_obj: Optional[ObjectState] = None
        best_rollout: Optional[RolloutEFEOutput] = None
        best_efe = float("inf")
        best_approach: Optional[Tuple[float, float]] = None
        best_clearing: List[ClearingPlan] = []

        for obj in available:
            # Scene graph reachability: can the arm reach this object
            # from any edge of its support surface?
            if not scene.reachable(obj.position, self._embodiment):
                continue

            # Compute approach candidates using affordance module
            candidates = compute_approach_candidates(
                object_pos=obj.position,
                obstacles=obstacles,
                embodiment=self._embodiment,
                robot_pos=robot_pos,
            )

            # No feasible candidates — object is physically blocked
            if not candidates:
                continue

            for cand in candidates:
                if not cand.clear:
                    continue

                approach_pos = (cand.x, cand.y)

                # ---- G_risk: −ln p(o|C) via Boltzmann preference ----
                # Preferences over task completion:
                #   p(efficient_completion | C) ∝ exp(−β · cost)
                # Expected surprisal:
                #   G_risk = −ln p(o|C) = β · cost + ln(Z)
                # Z is constant across candidates → cancels in argmin.
                #
                # Yield penalties enter as temporary preference shifts
                # (decay back to baseline over time).
                dist_to_drop = point_distance(
                    approach_pos[0], approach_pos[1],
                    self.drop_off[0], self.drop_off[1],
                )
                penalty = 0.0
                if yield_penalties:
                    penalty = yield_penalties.get(obj.id, 0.0)
                cost_to_go = (
                    cand.nav_cost + dist_to_drop
                    + cand.clearing_cost + penalty
                )
                g_risk = self._beta_pragmatic * cost_to_go

                # ---- G_information: H[p(o|s)] − I(o;s) ----
                # Ambiguity (outcome entropy) and info gain (epistemic)
                # are coupled through the same hidden state (precision).
                # Combined: entropy pushes away from uncertain outcomes,
                # info gain pulls toward them for exploration.
                q_success = self._success_probability(cand)
                g_information = _binary_entropy(q_success) + cand.epistemic_value

                # ---- G_social: SocialEFE rollout (proper EFE) ----
                # V*(step 0) from backward induction.  All terms are
                # expected surprisal: cross_entropy(p_outcome, p_pref)
                # for self-progress, other-comfort, collision, affect.
                # Commitment feeds in via obstruction.
                task_obstruction = self._estimate_obstruction(
                    robot_pos, approach_pos, obs, obj_id=obj.id,
                    other_pos=other_pos,
                )
                task_obs = ObservationContext(
                    kinematic_intent=obs.kinematic_intent,
                    distance=cand.nav_cost,
                    velocity=obs.velocity,
                    approaching=obs.approaching,
                    gaze_on_robot=obs.gaze_on_robot,
                    body_orientation=obs.body_orientation,
                    valence=obs.valence,
                    arousal=obs.arousal,
                    robot_last_intent="approach",
                )
                rollout = self._efe.compute_rollout(
                    q_human=q_other,
                    obs=task_obs,
                    mean_profile=profile,
                    config=rollout_config,
                    initial_obstruction=task_obstruction,
                    self_arousal=self_arousal,
                )
                g_social = rollout.value

                # ---- Total EFE ----
                # Contention flows through g_social: high task_obstruction
                # (from commitment + geometry) → SocialEFE rollout naturally
                # penalizes via collision risk and comfort, weighted by empathy.
                # No separate contention term needed — it's already in the
                # outcome model via obstruction → p(collision) → G_collision.
                total_efe = g_risk + g_information + g_social

                if total_efe < best_efe:
                    best_efe = total_efe
                    best_obj = obj
                    best_rollout = rollout
                    best_approach = approach_pos
                    best_clearing = list(cand.clearing_plans)

        self._current_target = best_obj
        self._current_approach = best_approach
        self._current_clearing = best_clearing
        return best_obj, best_rollout

    def _success_probability(self, cand: ApproachCandidate) -> float:
        """Predicted success probability from self-model precision.

        Maps clearing action precisions through a sigmoid to get
        p(success | hidden_state, policy).  No clearing actions needed
        → high baseline.  Push/grab needed → depends on precision.
        """
        if not cand.clearing_plans:
            return 0.95  # direct pick, high confidence

        # Joint probability: all clearing actions must succeed
        p = 0.95
        for plan in cand.clearing_plans:
            if plan.action == ClearingAction.PUSH_FORWARD:
                prec = self._embodiment.push_precision
            elif plan.action == ClearingAction.GRAB_AND_PULL:
                prec = self._embodiment.grip_precision
            else:
                continue
            # Sigmoid: p(success) = σ(κ·precision − b)
            # κ=3.0, b=1.0 gives ~0.27 at precision=0.1, ~0.88 at precision=1.0
            p *= 1.0 / (1.0 + math.exp(-(3.0 * prec - 1.0)))
        return p

    def _estimate_obstruction(
        self,
        robot_pos: Tuple[float, float],
        object_pos: Tuple[float, float],
        obs: ObservationContext,
        obj_id: Optional[str] = None,
        other_pos: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Estimate contention: how likely the other robot is pursuing this object.

        Combines geometric evidence (other→object distance ratio) with
        inferred commitment (behavioral cues from CommitmentInference).
        Both channels feed into SocialEFE's obstruction parameter so
        empathy weighting handles the trade-off naturally.

        High contention → high obstruction → SocialEFE penalizes via
        collision cost and favours yield/wait → higher G_social.
        """
        physical = 0.0

        if other_pos is not None:
            # Distance from other agent to this object
            d_other_obj = point_distance(
                other_pos[0], other_pos[1],
                object_pos[0], object_pos[1],
            )
            # Distance from self to this object
            d_self_obj = point_distance(
                robot_pos[0], robot_pos[1],
                object_pos[0], object_pos[1],
            )

            if d_other_obj < d_self_obj and d_other_obj < 3.0:
                # Other is closer to object than self — potential contention
                ratio = max(0.0, 1.0 - d_other_obj / max(d_self_obj, 0.01))
                physical = ratio
                if obs.approaching:
                    physical = min(1.0, physical * 1.5)
        elif obs.distance <= 5.0:
            # Fallback: no other_pos available, use small baseline
            physical = 0.1

        # Commitment-based obstruction: behavioral evidence that the
        # other agent is pursuing this specific object
        commitment = 0.0
        if (
            self._commitment is not None
            and self._other_agent_id is not None
            and obj_id is not None
        ):
            commitment = self._commitment.get_belief(
                self._other_agent_id, obj_id,
            )

        return max(physical, commitment)

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------
    def generate_sequence(self, obj: ObjectState) -> ScriptSequence:
        """Generate a clearing + nav + pick + nav + place sequence.

        Uses self._current_approach and self._current_clearing from
        the most recent select_next() call.

        Steps:
          [optional: clearing steps for each obstacle]
          1. Navigate to approach position (NOT raw object position)
          2. Pick object
          3. Navigate to drop-off position
          4. Place object
        """
        steps: List[ScriptStep] = []

        # Phase 1: Clear obstacles (if any)
        for plan in self._current_clearing:
            if plan.action == ClearingAction.PUSH_FORWARD:
                # Navigate behind the obstacle
                steps.append(ScriptStep(
                    request=SkillRequest(
                        skill="navigate",
                        goal={"x": plan.nav_to[0], "y": plan.nav_to[1]},
                    ),
                    transition_on="SUCCESS",
                    primitive_name="approach-obstacle",
                ))
                # Drive through slowly (push)
                steps.append(ScriptStep(
                    request=SkillRequest(
                        skill="navigate",
                        goal={"x": plan.push_through[0], "y": plan.push_through[1]},
                        params={"speed_scale": 0.3},
                    ),
                    transition_on="SUCCESS",
                    primitive_name="push-obstacle",
                ))

            elif plan.action == ClearingAction.GRAB_AND_PULL:
                # Navigate to front of obstacle (within arm reach)
                steps.append(ScriptStep(
                    request=SkillRequest(
                        skill="navigate",
                        goal={"x": plan.nav_to[0], "y": plan.nav_to[1]},
                    ),
                    transition_on="SUCCESS",
                    primitive_name="approach-obstacle",
                ))
                # Pick the obstacle (grab)
                steps.append(ScriptStep(
                    request=SkillRequest(
                        skill="pick_place",
                        goal={
                            "object": plan.obstacle.id,
                            "position": plan.grasp_target,
                        },
                        params={"mode": "pick"},
                    ),
                    transition_on="SUCCESS",
                    primitive_name="grab-obstacle",
                ))
                # Reverse to pull position
                steps.append(ScriptStep(
                    request=SkillRequest(
                        skill="navigate",
                        goal={"x": plan.pull_to[0], "y": plan.pull_to[1]},
                    ),
                    transition_on="SUCCESS",
                    primitive_name="pull-obstacle",
                ))
                # Place the obstacle aside (release)
                steps.append(ScriptStep(
                    request=SkillRequest(
                        skill="pick_place",
                        goal={
                            "object": plan.obstacle.id,
                            "position": plan.pull_to,
                        },
                        params={"mode": "place"},
                    ),
                    transition_on="SUCCESS",
                    primitive_name="release-obstacle",
                ))

        # Phase 2: Navigate to approach position
        approach = self._current_approach or obj.position
        steps.append(ScriptStep(
            request=SkillRequest(
                skill="navigate",
                goal={"x": approach[0], "y": approach[1]},
            ),
            transition_on="SUCCESS",
            primitive_name="approach-greet",
        ))

        # Phase 3: Pick target object
        steps.append(ScriptStep(
            request=SkillRequest(
                skill="pick_place",
                goal={"object": obj.id, "position": obj.position},
                params={"mode": "pick"},
            ),
            transition_on="SUCCESS",
            primitive_name="pick-object",
        ))

        # Phase 4: Navigate to drop-off
        steps.append(ScriptStep(
            request=SkillRequest(
                skill="navigate",
                goal={"x": self.drop_off[0], "y": self.drop_off[1]},
            ),
            transition_on="SUCCESS",
            primitive_name="approach-greet",
        ))

        # Phase 5: Place at drop-off
        steps.append(ScriptStep(
            request=SkillRequest(
                skill="pick_place",
                goal={
                    "object": obj.id,
                    "position": (self.drop_off[0], self.drop_off[1]),
                },
                params={"mode": "place"},
            ),
            transition_on="SUCCESS",
            primitive_name="place-object",
        ))

        return ScriptSequence(name=f"clear_{obj.id}", steps=steps)
