"""Engagement augmentation.

Estimates per-agent engagement level from gaze direction, interaction
duration, proximity trends, and body orientation.  Results merge into
``pb.social["engagement"]``.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from architecture_core.safety.geometry import point_distance


# =====================================================================
# Data classes
# =====================================================================
@dataclass
class EngagementReading:
    entity_id: str
    gaze_on_robot: float        # [0, 1] probability of gaze toward robot
    interaction_duration: float  # seconds since first detected
    proximity_trend: float      # m/s — negative = approaching
    body_orientation: float     # [0, 1] — 1 = facing robot directly
    level: str                  # "high" | "medium" | "low" | "none"
    score: float                # [0, 1] composite


# =====================================================================
# Per-agent tracker
# =====================================================================
class _AgentEngagementTracker:
    """Tracks engagement signals for one agent over time."""

    def __init__(self, entity_id: str, first_distance: float) -> None:
        self.entity_id = entity_id
        self.first_seen: float = _time.monotonic()
        self.prev_distance: Optional[float] = first_distance
        self.prev_time: float = self.first_seen

    def update(
        self,
        distance: float,
        gaze: float,
        body_orient: float,
        now: float,
    ) -> EngagementReading:
        dt = now - self.prev_time if now > self.prev_time else 1e-3
        duration = now - self.first_seen

        # Proximity trend (m/s): negative means getting closer
        if self.prev_distance is not None:
            trend = (distance - self.prev_distance) / dt
        else:
            trend = 0.0
        # Clamp to avoid spikes from noisy jumps
        trend = max(-2.0, min(2.0, trend))

        self.prev_distance = distance
        self.prev_time = now

        # Composite score (weighted sum)
        score = (
            0.35 * gaze
            + 0.25 * body_orient
            + 0.20 * _proximity_factor(distance)
            + 0.10 * _approach_factor(trend)
            + 0.10 * _duration_factor(duration)
        )
        score = max(0.0, min(1.0, score))

        if score >= 0.7:
            level = "high"
        elif score >= 0.4:
            level = "medium"
        elif score >= 0.15:
            level = "low"
        else:
            level = "none"

        return EngagementReading(
            entity_id=self.entity_id,
            gaze_on_robot=round(gaze, 2),
            interaction_duration=round(duration, 1),
            proximity_trend=round(trend, 3),
            body_orientation=round(body_orient, 2),
            level=level,
            score=round(score, 3),
        )


# =====================================================================
# Public augmentation
# =====================================================================
class EngagementAugmentation:
    """Compute per-agent engagement and inject into ``pb.social``.

    Sensor contract — each agent dict in ``world["agents"]`` may carry::

        "engagement_cues": {
            "gaze_on_robot": float,     # [0, 1]
            "body_orientation": float,  # [0, 1]  — 1 = facing robot
        }

    ``pose`` and ``id`` are always expected.  If ``engagement_cues`` is
    absent, gaze and orientation default to 0.5 (unknown).
    """

    section = "social"

    def __init__(self) -> None:
        self._trackers: Dict[str, _AgentEngagementTracker] = {}

    def augment(self, world: Dict[str, Any]) -> Dict[str, Any]:
        robot_pose = world.get("robot_pose", (0, 0, 0, 0))
        rx, ry = robot_pose[0], robot_pose[1]
        now = _time.monotonic()

        agents = world.get("agents", [])
        readings: List[EngagementReading] = []

        seen_ids: set = set()
        for agent in agents:
            eid = agent.get("id", "unknown")
            seen_ids.add(eid)
            pose = agent.get("pose", (0, 0))
            dist = point_distance(rx, ry, pose[0], pose[1])

            cues = agent.get("engagement_cues", {})
            gaze = float(cues.get("gaze_on_robot", 0.5))
            body = float(cues.get("body_orientation", 0.5))

            if eid not in self._trackers:
                self._trackers[eid] = _AgentEngagementTracker(eid, dist)

            reading = self._trackers[eid].update(dist, gaze, body, now)
            readings.append(reading)

        # Prune stale trackers (agents no longer visible)
        stale = [k for k in self._trackers if k not in seen_ids]
        for k in stale:
            del self._trackers[k]

        max_score = max((r.score for r in readings), default=0.0)
        most_engaged = max(readings, key=lambda r: r.score).entity_id if readings else None

        return {
            "engagement": {
                "readings": [
                    {
                        "entity_id": r.entity_id,
                        "gaze_on_robot": r.gaze_on_robot,
                        "interaction_duration": r.interaction_duration,
                        "proximity_trend": r.proximity_trend,
                        "body_orientation": r.body_orientation,
                        "level": r.level,
                        "score": r.score,
                    }
                    for r in readings
                ],
                "max_engagement_score": round(max_score, 3),
                "most_engaged_entity": most_engaged,
                "num_engaged": sum(1 for r in readings if r.level in ("high", "medium")),
            }
        }


# =====================================================================
# Scoring helpers
# =====================================================================
def _proximity_factor(distance: float) -> float:
    """Closer agents are more engaging.  Saturates inside 0.5 m."""
    if distance <= 0.5:
        return 1.0
    if distance >= 5.0:
        return 0.0
    return 1.0 - (distance - 0.5) / 4.5


def _approach_factor(trend: float) -> float:
    """Approaching (negative trend) is more engaging than withdrawing."""
    if trend <= -0.3:
        return 1.0
    if trend >= 0.3:
        return 0.0
    return 0.5 - trend / 0.6


def _duration_factor(seconds: float) -> float:
    """Longer interaction means higher engagement.  Saturates at 30 s."""
    return min(1.0, seconds / 30.0)
