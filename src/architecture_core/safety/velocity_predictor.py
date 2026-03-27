"""Velocity-based trajectory prediction and time-to-collision (TTC).

Pure computation module used by both PredictiveCollisionRule (NormEngine)
and SafetyShield. Robot-agnostic; no hardware imports.

TTC is computed via the standard quadratic on relative position/velocity:
    |dp + dv*t|^2 = combined_radius^2
Smallest positive root = time to collision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from architecture_core.core.types import PerceptBundle
from architecture_core.safety.geometry import point_distance


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class EntityState:
    """Position and velocity of a single entity."""
    entity_id: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.25


@dataclass
class CollisionRisk:
    """Predicted collision between two entities."""
    entity_a: str
    entity_b: str
    ttc: float
    min_distance: float
    closing_speed: float
    collision_point: Tuple[float, float]
    risk_type: str  # "robot_agent" | "robot_obstacle" | "agent_agent"


@dataclass
class CollisionPrediction:
    """Full prediction result for one tick."""
    risks: List[CollisionRisk] = field(default_factory=list)
    min_robot_ttc: float = float("inf")
    min_agent_agent_ttc: float = float("inf")
    robot_velocity: Tuple[float, float] = (0.0, 0.0)
    timestamp: float = 0.0


# =====================================================================
# TTC computation
# =====================================================================

def compute_ttc(
    x1: float, y1: float, vx1: float, vy1: float, r1: float,
    x2: float, y2: float, vx2: float, vy2: float, r2: float,
) -> Tuple[float, float, Tuple[float, float]]:
    """Compute time-to-collision between two circular entities.

    Returns:
        (ttc, closing_speed, collision_point)
        ttc is float('inf') if no collision predicted.
    """
    dpx = x2 - x1
    dpy = y2 - y1
    dvx = vx2 - vx1
    dvy = vy2 - vy1

    combined_r = r1 + r2

    dist = math.sqrt(dpx * dpx + dpy * dpy)

    # Closing speed: positive means entities are getting closer
    if dist > 1e-9:
        closing_speed = -(dpx * dvx + dpy * dvy) / dist
    else:
        closing_speed = 0.0

    # Already overlapping
    if dist < combined_r:
        return (0.0, closing_speed, ((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    # Quadratic: a*t^2 + b*t + c = 0
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (dpx * dvx + dpy * dvy)
    c = dpx * dpx + dpy * dpy - combined_r * combined_r

    if a < 1e-12:
        # No relative motion — can't collide (already checked overlap)
        return (float("inf"), closing_speed, (0.0, 0.0))

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return (float("inf"), closing_speed, (0.0, 0.0))

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Smallest positive root
    ttc = float("inf")
    if t1 > 1e-6:
        ttc = t1
    elif t2 > 1e-6:
        ttc = t2

    # Collision point (midpoint at time of contact)
    if ttc < float("inf"):
        cx = (x1 + vx1 * ttc + x2 + vx2 * ttc) / 2.0
        cy = (y1 + vy1 * ttc + y2 + vy2 * ttc) / 2.0
        collision_point = (cx, cy)
    else:
        collision_point = (0.0, 0.0)

    return (ttc, closing_speed, collision_point)


def speed_reduction_factor(
    ttc: float,
    ttc_stop: float = 0.5,
    ttc_caution: float = 3.0,
) -> float:
    """Compute speed reduction factor from TTC.

    Linear ramp: 0.0 at ttc_stop, 1.0 at ttc_caution.
    """
    if ttc <= ttc_stop:
        return 0.0
    if ttc >= ttc_caution:
        return 1.0
    return (ttc - ttc_stop) / (ttc_caution - ttc_stop)


# =====================================================================
# Main predictor class
# =====================================================================

class VelocityPredictor:
    """Predict collisions from PerceptBundle data.

    Estimates robot velocity from pose deltas when not provided.
    Enumerates all entity pairs for collision checking.
    """

    def __init__(
        self,
        robot_radius: float = 0.25,
        default_agent_radius: float = 0.25,
        max_horizon: float = 5.0,
    ) -> None:
        self.robot_radius = robot_radius
        self.default_agent_radius = default_agent_radius
        self.max_horizon = max_horizon

        self._prev_robot_pos: Optional[Tuple[float, float]] = None
        self._prev_t: Optional[float] = None

    def predict(self, pb: PerceptBundle) -> CollisionPrediction:
        """Compute all collision predictions for this tick."""
        robot_pose = pb.world.get("robot_pose", (0, 0, 0, 0))
        rx, ry = robot_pose[0], robot_pose[1]

        rvx, rvy = self._estimate_robot_velocity(rx, ry, pb.t)

        # Build entity list from agents
        entities: List[EntityState] = []
        for agent in pb.world.get("agents", []):
            pose = agent.get("pose", (0, 0))
            vel = agent.get("velocity")
            if vel is not None and isinstance(vel, (list, tuple)) and len(vel) >= 2:
                avx, avy = float(vel[0]), float(vel[1])
            else:
                avx, avy = 0.0, 0.0
            radius = agent.get("radius", self.default_agent_radius)
            entities.append(EntityState(
                entity_id=agent.get("id", "unknown"),
                x=pose[0], y=pose[1],
                vx=avx, vy=avy,
                radius=radius,
            ))

        risks: List[CollisionRisk] = []

        # Robot vs each agent
        for ent in entities:
            ttc, closing, cp = compute_ttc(
                rx, ry, rvx, rvy, self.robot_radius,
                ent.x, ent.y, ent.vx, ent.vy, ent.radius,
            )
            if ttc <= self.max_horizon:
                dist = point_distance(rx, ry, ent.x, ent.y)
                risks.append(CollisionRisk(
                    entity_a="robot",
                    entity_b=ent.entity_id,
                    ttc=ttc,
                    min_distance=dist,
                    closing_speed=closing,
                    collision_point=cp,
                    risk_type="robot_agent",
                ))

        # Robot vs obstacles
        for obs in pb.world.get("obstacles", []):
            opose = obs.get("pose", (0, 0))
            oradius = obs.get("radius", 0.3)
            ttc, closing, cp = compute_ttc(
                rx, ry, rvx, rvy, self.robot_radius,
                opose[0], opose[1], 0.0, 0.0, oradius,
            )
            if ttc <= self.max_horizon:
                dist = point_distance(rx, ry, opose[0], opose[1])
                risks.append(CollisionRisk(
                    entity_a="robot",
                    entity_b=obs.get("id", "obstacle"),
                    ttc=ttc,
                    min_distance=dist,
                    closing_speed=closing,
                    collision_point=cp,
                    risk_type="robot_obstacle",
                ))

        # Agent vs agent (all pairs)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                ei, ej = entities[i], entities[j]
                ttc, closing, cp = compute_ttc(
                    ei.x, ei.y, ei.vx, ei.vy, ei.radius,
                    ej.x, ej.y, ej.vx, ej.vy, ej.radius,
                )
                if ttc <= self.max_horizon:
                    dist = point_distance(ei.x, ei.y, ej.x, ej.y)
                    risks.append(CollisionRisk(
                        entity_a=ei.entity_id,
                        entity_b=ej.entity_id,
                        ttc=ttc,
                        min_distance=dist,
                        closing_speed=closing,
                        collision_point=cp,
                        risk_type="agent_agent",
                    ))

        # Sort by TTC ascending
        risks.sort(key=lambda r: r.ttc)

        # Summaries
        robot_risks = [r for r in risks
                       if r.risk_type in ("robot_agent", "robot_obstacle")]
        agent_risks = [r for r in risks if r.risk_type == "agent_agent"]

        return CollisionPrediction(
            risks=risks,
            min_robot_ttc=robot_risks[0].ttc if robot_risks else float("inf"),
            min_agent_agent_ttc=agent_risks[0].ttc if agent_risks else float("inf"),
            robot_velocity=(rvx, rvy),
            timestamp=pb.t,
        )

    def _estimate_robot_velocity(
        self, rx: float, ry: float, t: float,
    ) -> Tuple[float, float]:
        """Estimate robot velocity from pose deltas."""
        if self._prev_robot_pos is None or self._prev_t is None:
            self._prev_robot_pos = (rx, ry)
            self._prev_t = t
            return (0.0, 0.0)

        dt = t - self._prev_t
        if dt < 1e-6:
            return (0.0, 0.0)

        vx = (rx - self._prev_robot_pos[0]) / dt
        vy = (ry - self._prev_robot_pos[1]) / dt

        self._prev_robot_pos = (rx, ry)
        self._prev_t = t

        return (vx, vy)
