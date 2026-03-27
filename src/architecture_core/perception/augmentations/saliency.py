"""Saliency augmentation.

Identifies the most attention-worthy targets in the environment
based on proximity, velocity, novelty, and threat level.  Results
merge into ``pb.attention["saliency"]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from architecture_core.safety.geometry import point_distance


# =====================================================================
# Data classes
# =====================================================================
@dataclass
class SaliencyTarget:
    entity_id: str
    position: tuple             # (x, y)
    score: float                # [0, 1] composite salience
    distance: float
    velocity: float             # m/s — estimated from pose deltas
    category: str               # "agent" | "obstacle" | "goal" | "hazard"
    is_novel: bool              # first time seen this tick cycle


# =====================================================================
# Per-entity velocity estimator
# =====================================================================
class _EntityTracker:
    def __init__(self, position: tuple) -> None:
        self.prev_pos = position
        self.velocity = 0.0
        self.ticks_seen = 0

    def update(self, position: tuple, dt: float) -> float:
        if dt > 0:
            dx = position[0] - self.prev_pos[0]
            dy = position[1] - self.prev_pos[1]
            self.velocity = math.sqrt(dx * dx + dy * dy) / dt
        self.prev_pos = position
        self.ticks_seen += 1
        return self.velocity


# =====================================================================
# Public augmentation
# =====================================================================
class SaliencyAugmentation:
    """Compute per-entity salience and inject into ``pb.attention``.

    Salience factors (weighted sum):
        * Proximity — closer = more salient
        * Velocity  — faster = more salient (potential threat / relevance)
        * Novelty   — newly appeared entities get a salience boost
        * Threat    — hazards always high salience

    Sensor contract — ``world`` must contain at least ``robot_pose`` and
    ``agents``.  Optional: ``hazards`` (list of (x,y,hx,hy) tuples).
    """

    section = "attention"

    # Weight configuration
    W_PROXIMITY = 0.35
    W_VELOCITY = 0.25
    W_NOVELTY = 0.20
    W_THREAT = 0.20

    def __init__(self, novelty_boost_ticks: int = 5) -> None:
        self._trackers: Dict[str, _EntityTracker] = {}
        self._prev_ids: Set[str] = set()
        self._novelty_boost_ticks = novelty_boost_ticks
        self._last_t: Optional[float] = None

    def augment(self, world: Dict[str, Any]) -> Dict[str, Any]:
        robot_pose = world.get("robot_pose", (0, 0, 0, 0))
        rx, ry = robot_pose[0], robot_pose[1]
        t = world.get("timestamp", 0.0)
        dt = (t - self._last_t) if self._last_t is not None else 0.1
        dt = max(dt, 1e-4)
        self._last_t = t

        targets: List[SaliencyTarget] = []
        current_ids: Set[str] = set()

        # --- agents ---
        for agent in world.get("agents", []):
            eid = agent.get("id", "unknown")
            current_ids.add(eid)
            pose = agent.get("pose", (0, 0))
            dist = point_distance(rx, ry, pose[0], pose[1])

            if eid not in self._trackers:
                self._trackers[eid] = _EntityTracker(pose)
            vel = self._trackers[eid].update(pose, dt)
            is_novel = self._trackers[eid].ticks_seen <= self._novelty_boost_ticks

            score = self._compute_score(dist, vel, is_novel, is_threat=False)
            targets.append(SaliencyTarget(
                entity_id=eid,
                position=pose,
                score=round(score, 3),
                distance=round(dist, 3),
                velocity=round(vel, 3),
                category="agent",
                is_novel=is_novel,
            ))

        # --- hazards ---
        for i, haz in enumerate(world.get("hazards", [])):
            hid = f"hazard_{i}"
            current_ids.add(hid)
            hx, hy = haz[0], haz[1]
            dist = point_distance(rx, ry, hx, hy)
            is_novel = hid not in self._prev_ids

            score = self._compute_score(dist, 0.0, is_novel, is_threat=True)
            targets.append(SaliencyTarget(
                entity_id=hid,
                position=(hx, hy),
                score=round(score, 3),
                distance=round(dist, 3),
                velocity=0.0,
                category="hazard",
                is_novel=is_novel,
            ))

        # Prune stale trackers
        stale = [k for k in self._trackers if k not in current_ids]
        for k in stale:
            del self._trackers[k]
        self._prev_ids = current_ids

        # Sort by salience (highest first)
        targets.sort(key=lambda t: t.score, reverse=True)

        return {
            "saliency": {
                "targets": [
                    {
                        "entity_id": t.entity_id,
                        "position": t.position,
                        "score": t.score,
                        "distance": t.distance,
                        "velocity": t.velocity,
                        "category": t.category,
                        "is_novel": t.is_novel,
                    }
                    for t in targets
                ],
                "most_salient": targets[0].entity_id if targets else None,
                "most_salient_score": targets[0].score if targets else 0.0,
                "num_targets": len(targets),
            }
        }

    # ------------------------------------------------------------------
    def _compute_score(
        self, dist: float, vel: float, is_novel: bool, is_threat: bool,
    ) -> float:
        prox = _proximity_salience(dist)
        vel_s = _velocity_salience(vel)
        nov = 1.0 if is_novel else 0.0
        threat = 1.0 if is_threat else 0.0

        score = (
            self.W_PROXIMITY * prox
            + self.W_VELOCITY * vel_s
            + self.W_NOVELTY * nov
            + self.W_THREAT * threat
        )
        return max(0.0, min(1.0, score))


# =====================================================================
# Scoring helpers
# =====================================================================
def _proximity_salience(distance: float) -> float:
    """Inverse-distance salience.  Saturates at 0.3 m, decays past 5 m."""
    if distance <= 0.3:
        return 1.0
    if distance >= 5.0:
        return 0.0
    return 1.0 - (distance - 0.3) / 4.7


def _velocity_salience(velocity: float) -> float:
    """Faster entities are more salient.  Saturates at 1.5 m/s."""
    return min(1.0, velocity / 1.5)
