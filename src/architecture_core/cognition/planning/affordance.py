"""Embodied affordance reasoning via active inference.

The robot's generative model includes its own embodiment as a self-model
and the environment's affordance structure.  Every decision is driven by
Expected Free Energy minimization:

  G_total = G_pragmatic + G_epistemic + G_social

- G_pragmatic: path cost + clearing effort (goal-directed)
- G_epistemic: information gain about self-model capabilities
- G_social: empathy-weighted EFE rollout (computed externally by TaskPlanner)

Active inference framing (cf. HAIF — Pezzato, VERSES AI):
  Self-model (A-matrix)    = EmbodimentModel (what I can do)
  Environment model        = FurnitureItem (what the world affords)
  Scene graph              = SceneGraph (support/containment relations)
  Affordance prior (D)     = action types: navigate_around, push, grab_and_pull
  Epistemic drive          = precision per capability; low → explore, high → exploit
  Prediction error         = clearing failure → decrease precision

Hierarchical inference (HAIF-style):
  1. Preferred observation: end-effector at object position
  2. Self-model inference:  given arm_reach, body must be at table edge
  3. Scene graph:           support surface is part of reaching affordance,
                            not an obstacle to avoid

Robot-agnostic: imports only from architecture_core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from architecture_core.safety.geometry import (
    Circle,
    Rect,
    circle_rect_overlap,
    point_distance,
    point_in_rect,
)


# =====================================================================
# Clearing action enum
# =====================================================================
class ClearingAction(Enum):
    """Affordance actions the agent can take to clear an obstacle."""

    NAVIGATE_AROUND = "navigate_around"
    PUSH_FORWARD = "push_forward"
    GRAB_AND_PULL = "grab_and_pull"


# =====================================================================
# Self-model (A-matrix)
# =====================================================================
@dataclass
class EmbodimentModel:
    """Robot's self-model of its physical capabilities.

    Encodes what the agent *can* do -- its action repertoire constrained
    by physical form.  Precision tracks confidence in each capability
    and drives the epistemic term in the clearing EFE.
    """

    arm_reach: float = 0.6
    body_radius: float = 0.27
    min_clearance: float = 0.35
    max_push_mass: float = 5.0
    grip_strength: float = 5.0

    # Precision per capability (starts low = uncertain)
    push_precision: float = 0.1
    grip_precision: float = 0.1
    epistemic_weight: float = 0.3

    def update_precision(
        self,
        action: ClearingAction,
        success: bool,
        increment: float = 0.2,
        decrement: float = 0.3,
    ) -> None:
        """Prediction error updates self-model precision.

        Success -> increase precision (exploit).
        Failure -> decrease precision (revise model).
        """
        if action == ClearingAction.PUSH_FORWARD:
            if success:
                self.push_precision += increment
            else:
                self.push_precision = max(0.01, self.push_precision - decrement)
        elif action == ClearingAction.GRAB_AND_PULL:
            if success:
                self.grip_precision += increment
            else:
                self.grip_precision = max(0.01, self.grip_precision - decrement)


# =====================================================================
# Environment model
# =====================================================================
@dataclass
class FurnitureItem:
    """Perceived furniture item with physical properties.

    Part of the environment model -- what the world affords.
    """

    id: str
    type: str
    position: Tuple[float, float]
    rotation: float
    width: float
    depth: float
    mass: float
    movable: bool
    keepout: Rect

    @property
    def physical_rect(self) -> Rect:
        """Physical bounding box (no navigation padding)."""
        hw = self.width / 2.0
        hd = self.depth / 2.0
        return Rect(
            self.position[0] - hw, self.position[1] - hd,
            self.position[0] + hw, self.position[1] + hd,
        )


# =====================================================================
# Scene graph
# =====================================================================
class SceneGraph:
    """Relational model of the environment for affordance reasoning.

    Infers support relations (object ON surface) from spatial
    containment: if an object's position is inside a furniture item's
    physical bounding box, that furniture supports the object.

    Hierarchical inference (cf. HAIF):
      Preferred observation: end-effector at object position.
      Self-model:            arm_reach determines required body position.
      Scene graph:           support surface constrains WHERE the body
                             must stand (at table edge, not on table).
    """

    def __init__(
        self,
        furniture: List[FurnitureItem],
        objects: Optional[List] = None,
    ) -> None:
        self._furniture: Dict[str, FurnitureItem] = {
            f.id: f for f in furniture
        }
        self._support: Dict[str, str] = {}
        if objects:
            self._infer_support(objects)

    def _infer_support(self, objects: List) -> None:
        """Infer support relations from spatial containment.

        An object whose position falls inside a furniture item's
        physical bounding box is supported by that furniture.
        """
        for obj in objects:
            pos = obj.position if hasattr(obj, "position") else (0, 0)
            obj_id = obj.id if hasattr(obj, "id") else str(id(obj))
            for fid, furn in self._furniture.items():
                if point_in_rect(pos[0], pos[1], furn.physical_rect):
                    self._support[obj_id] = fid
                    break

    def support_surface(self, object_id: str) -> Optional[FurnitureItem]:
        """Get the furniture item supporting this object."""
        fid = self._support.get(object_id)
        return self._furniture.get(fid) if fid else None

    def reachable(
        self,
        object_pos: Tuple[float, float],
        embodiment: EmbodimentModel,
    ) -> bool:
        """Check if an object is reachable from ANY edge of its surface.

        The robot's body must stand outside the physical footprint
        (at least body_radius clearance) and the arm must reach
        the object (distance <= arm_reach).
        """
        support_id = None
        for fid, furn in self._furniture.items():
            if point_in_rect(object_pos[0], object_pos[1], furn.physical_rect):
                support_id = fid
                break
        if support_id is None:
            return True  # no surface → freely reachable

        furn = self._furniture[support_id]
        pr = furn.physical_rect
        # Minimum distance from object to nearest edge
        dx_min = min(
            abs(object_pos[0] - pr.x_min),
            abs(object_pos[0] - pr.x_max),
        )
        dy_min = min(
            abs(object_pos[1] - pr.y_min),
            abs(object_pos[1] - pr.y_max),
        )
        edge_dist = min(dx_min, dy_min)
        # Body stands at edge + body_radius, arm reaches back arm_reach
        return edge_dist + embodiment.body_radius <= embodiment.arm_reach

    @property
    def support_relations(self) -> Dict[str, str]:
        return dict(self._support)

    @property
    def furniture(self) -> Dict[str, FurnitureItem]:
        return dict(self._furniture)


# =====================================================================
# Clearing plan
# =====================================================================
@dataclass
class ClearingPlan:
    """EFE-scored plan for clearing a specific obstacle."""

    action: ClearingAction
    obstacle: FurnitureItem
    efe: float
    nav_to: Tuple[float, float]
    push_through: Optional[Tuple[float, float]] = None
    grasp_target: Optional[Tuple[float, float]] = None
    pull_to: Optional[Tuple[float, float]] = None


# =====================================================================
# Approach candidate
# =====================================================================
@dataclass
class ApproachCandidate:
    """One candidate position from which the robot could reach an object.

    EFE components stored per candidate for TaskPlanner to combine with
    G_social.
    """

    x: float
    y: float
    angle: float
    clear: bool
    blocked_by: List[FurnitureItem] = field(default_factory=list)
    clearing_plans: List[ClearingPlan] = field(default_factory=list)
    nav_cost: float = 0.0
    clearing_cost: float = 0.0
    epistemic_value: float = 0.0


# =====================================================================
# Approach candidate computation
# =====================================================================
def compute_approach_candidates(
    object_pos: Tuple[float, float],
    obstacles: List[FurnitureItem],
    embodiment: EmbodimentModel,
    robot_pos: Tuple[float, float],
    n_candidates: int = 8,
) -> List[ApproachCandidate]:
    """Generate and evaluate approach positions around an object.

    Hierarchical inference (HAIF-style):
      Preferred observation = end-effector at object_pos
      Required body state   = at arm_reach distance from object
      Constraint            = body must not overlap any physical furniture

    All obstacles (including the object's support table) are checked
    against their physical footprint, not the padded navigation keepout.
    The arm bridges the gap between the body position and the object.

    1. Place *n_candidates* in a circle at arm_reach distance
    2. For each candidate check body collision with physical footprints
    3. Blocked by immovable -> not clear
    4. Blocked by movable -> compute clearing plans via select_clearing_action
    5. Return feasible candidates sorted by nav_cost + clearing_cost
    """
    candidates: List[ApproachCandidate] = []
    angle_step = 2.0 * math.pi / n_candidates

    for i in range(n_candidates):
        angle = i * angle_step
        cx = object_pos[0] + embodiment.arm_reach * math.cos(angle)
        cy = object_pos[1] + embodiment.arm_reach * math.sin(angle)

        # Collision check against PHYSICAL footprint, not padded keepout.
        # Every obstacle (including the support table) blocks the body.
        # The arm extends arm_reach beyond the body — that's how the
        # robot reaches objects on the table while standing at its edge.
        # Using the physical rect avoids double-counting body_radius
        # (keepout padding already includes it).
        robot_circle = Circle(cx, cy, embodiment.body_radius)
        immovable_block = False
        movable_blockers: List[FurnitureItem] = []

        for obs in obstacles:
            if circle_rect_overlap(robot_circle, obs.physical_rect):
                if obs.movable:
                    movable_blockers.append(obs)
                else:
                    immovable_block = True
                    break

        if immovable_block:
            candidates.append(ApproachCandidate(
                x=cx, y=cy, angle=angle, clear=False,
            ))
            continue

        # Nav cost: distance from robot to this candidate
        nav_cost = point_distance(robot_pos[0], robot_pos[1], cx, cy)

        # If movable obstacles block, compute clearing plans
        clearing_plans: List[ClearingPlan] = []
        clearing_cost = 0.0
        epistemic_value = 0.0

        if movable_blockers:
            # Select best clearing action for each blocker
            for blocker in movable_blockers:
                plans = select_clearing_action(
                    obstacle=blocker,
                    approach_pos=(cx, cy),
                    robot_pos=robot_pos,
                    embodiment=embodiment,
                    other_obstacles=[o for o in obstacles if o.id != blocker.id],
                )
                if plans:
                    best = plans[0]
                    clearing_plans.append(best)
                    clearing_cost += best.efe
                else:
                    # No feasible clearing action
                    immovable_block = True
                    break

        if immovable_block:
            candidates.append(ApproachCandidate(
                x=cx, y=cy, angle=angle, clear=False,
            ))
            continue

        # Epistemic value from best clearing plans
        for plan in clearing_plans:
            epistemic_value += _epistemic_for_action(
                plan.action, embodiment,
            )

        candidates.append(ApproachCandidate(
            x=cx,
            y=cy,
            angle=angle,
            clear=True,
            blocked_by=movable_blockers,
            clearing_plans=clearing_plans,
            nav_cost=nav_cost,
            clearing_cost=clearing_cost,
            epistemic_value=epistemic_value,
        ))

    # Sort feasible candidates by total pragmatic cost
    feasible = [c for c in candidates if c.clear]
    feasible.sort(key=lambda c: c.nav_cost + c.clearing_cost + c.epistemic_value)
    return feasible


# =====================================================================
# Clearing action selection (EFE over affordances)
# =====================================================================
_EFFORT_COSTS = {
    ClearingAction.NAVIGATE_AROUND: 0.0,
    ClearingAction.PUSH_FORWARD: 0.3,
    ClearingAction.GRAB_AND_PULL: 0.5,
}


def _epistemic_for_action(
    action: ClearingAction,
    embodiment: EmbodimentModel,
) -> float:
    """G_epistemic for a clearing action.

    Low precision -> large negative value -> drives exploration.
    Navigate-around has zero epistemic value (nothing to learn).
    """
    if action == ClearingAction.NAVIGATE_AROUND:
        return 0.0
    elif action == ClearingAction.PUSH_FORWARD:
        return -embodiment.epistemic_weight / (1.0 + embodiment.push_precision)
    elif action == ClearingAction.GRAB_AND_PULL:
        return -embodiment.epistemic_weight / (1.0 + embodiment.grip_precision)
    return 0.0


def select_clearing_action(
    obstacle: FurnitureItem,
    approach_pos: Tuple[float, float],
    robot_pos: Tuple[float, float],
    embodiment: EmbodimentModel,
    other_obstacles: List[FurnitureItem],
) -> List[ClearingPlan]:
    """EFE-based selection over clearing affordances for one obstacle.

    For each action:  G = G_pragmatic(effort) + G_epistemic(info gain)

    NAVIGATE_AROUND is not included here -- it is implicit: if no
    clearing action is feasible, the candidate is marked as not-clear
    and the robot selects a different (clear) candidate position.

    Returns feasible ClearingPlans sorted by total EFE (lowest first).
    Empty list means no clearing is feasible for this obstacle.
    """
    plans: List[ClearingPlan] = []

    # --- PUSH_FORWARD ---
    if obstacle.mass <= embodiment.max_push_mass:
        push = _try_push(obstacle, approach_pos, embodiment, other_obstacles)
        if push is not None:
            g_prag = _EFFORT_COSTS[ClearingAction.PUSH_FORWARD]
            g_epist = _epistemic_for_action(ClearingAction.PUSH_FORWARD, embodiment)
            plans.append(ClearingPlan(
                action=ClearingAction.PUSH_FORWARD,
                obstacle=obstacle,
                efe=g_prag + g_epist,
                nav_to=push[0],
                push_through=push[1],
            ))

    # --- GRAB_AND_PULL ---
    if obstacle.mass <= embodiment.grip_strength:
        pull = _try_pull(obstacle, approach_pos, embodiment, other_obstacles)
        if pull is not None:
            g_prag = _EFFORT_COSTS[ClearingAction.GRAB_AND_PULL]
            g_epist = _epistemic_for_action(ClearingAction.GRAB_AND_PULL, embodiment)
            plans.append(ClearingPlan(
                action=ClearingAction.GRAB_AND_PULL,
                obstacle=obstacle,
                efe=g_prag + g_epist,
                nav_to=pull[0],
                grasp_target=pull[1],
                pull_to=pull[2],
            ))

    plans.sort(key=lambda p: p.efe)
    return plans


# =====================================================================
# Push position computation
# =====================================================================
def compute_push_position(
    obstacle: FurnitureItem,
    approach_pos: Tuple[float, float],
    embodiment: EmbodimentModel,
    push_distance: float = 0.5,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute (start_pos, drive_through_pos) for pushing obstacle aside.

    The robot pushes from the side OPPOSITE to the approach direction,
    displacing the obstacle away from the approach position.
    """
    # Direction from obstacle to approach position (push toward this)
    dx = approach_pos[0] - obstacle.position[0]
    dy = approach_pos[1] - obstacle.position[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.01:
        dx, dy = 1.0, 0.0
        dist = 1.0
    dx /= dist
    dy /= dist

    # Start position: behind obstacle (opposite of approach direction)
    standoff = max(obstacle.width, obstacle.depth) / 2.0 + embodiment.body_radius + 0.1
    start_x = obstacle.position[0] - dx * standoff
    start_y = obstacle.position[1] - dy * standoff

    # Drive-through: past obstacle center in push direction
    through_x = obstacle.position[0] + dx * push_distance
    through_y = obstacle.position[1] + dy * push_distance

    return (start_x, start_y), (through_x, through_y)


# =====================================================================
# Pull position computation
# =====================================================================
def compute_pull_position(
    obstacle: FurnitureItem,
    approach_pos: Tuple[float, float],
    embodiment: EmbodimentModel,
    pull_distance: float = 0.5,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Compute (nav_to, grasp_point, pull_to) for grab-and-pull.

    Robot approaches from the APPROACH side, grasps nearest edge,
    then reverses to pull obstacle toward itself (away from approach pos).
    """
    # Direction from obstacle to approach position
    dx = approach_pos[0] - obstacle.position[0]
    dy = approach_pos[1] - obstacle.position[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.01:
        dx, dy = 1.0, 0.0
        dist = 1.0
    dx /= dist
    dy /= dist

    # Grasp point: nearest edge of obstacle toward approach
    edge_dist = max(obstacle.width, obstacle.depth) / 2.0
    grasp_x = obstacle.position[0] + dx * edge_dist
    grasp_y = obstacle.position[1] + dy * edge_dist

    # Nav-to: arm_reach distance from grasp point (approach side)
    nav_x = grasp_x + dx * embodiment.arm_reach
    nav_y = grasp_y + dy * embodiment.arm_reach

    # Pull-to: behind nav position (robot reverses)
    pull_x = nav_x + dx * pull_distance
    pull_y = nav_y + dy * pull_distance

    return (nav_x, nav_y), (grasp_x, grasp_y), (pull_x, pull_y)


# =====================================================================
# Internal helpers for clearing feasibility
# =====================================================================
def _try_push(
    obstacle: FurnitureItem,
    approach_pos: Tuple[float, float],
    embodiment: EmbodimentModel,
    other_obstacles: List[FurnitureItem],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Check push feasibility and return positions if feasible.

    Push requires clear space BEHIND the obstacle (opposite approach).
    """
    start, through = compute_push_position(obstacle, approach_pos, embodiment)

    # Check that start position and through position don't collide
    # with immovable obstacles
    start_circle = Circle(start[0], start[1], embodiment.body_radius)
    through_circle = Circle(through[0], through[1], embodiment.body_radius)

    for other in other_obstacles:
        if not other.movable:
            if (circle_rect_overlap(start_circle, other.keepout)
                    or circle_rect_overlap(through_circle, other.keepout)):
                return None

    return start, through


def _try_pull(
    obstacle: FurnitureItem,
    approach_pos: Tuple[float, float],
    embodiment: EmbodimentModel,
    other_obstacles: List[FurnitureItem],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """Check grab-and-pull feasibility and return positions if feasible.

    Pull requires clear space IN FRONT of obstacle (approach side).
    """
    nav_to, grasp, pull_to = compute_pull_position(
        obstacle, approach_pos, embodiment,
    )

    # Check that nav and pull positions don't collide with immovable obstacles
    nav_circle = Circle(nav_to[0], nav_to[1], embodiment.body_radius)
    pull_circle = Circle(pull_to[0], pull_to[1], embodiment.body_radius)

    for other in other_obstacles:
        if not other.movable:
            if (circle_rect_overlap(nav_circle, other.keepout)
                    or circle_rect_overlap(pull_circle, other.keepout)):
                return None

    return nav_to, grasp, pull_to
