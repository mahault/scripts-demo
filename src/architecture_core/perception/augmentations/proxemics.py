"""architecture_core/perception/augmentations/proxemics.py

Compute proxemic-zone information from world state using Hall's
interpersonal distance zones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from architecture_core.safety.geometry import bearing, point_distance

# Hall's proxemic distances (metres)
INTIMATE = 0.45
PERSONAL = 1.2
SOCIAL = 3.6
PUBLIC = 7.6


@dataclass
class ProxemicReading:
    entity_id: str
    distance: float
    zone: str           # "intimate" | "personal" | "social" | "public"
    bearing: float      # radians from robot heading


def classify_zone(distance: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    t = thresholds or {"intimate": INTIMATE, "personal": PERSONAL, "social": SOCIAL}
    if distance < t["intimate"]:
        return "intimate"
    if distance < t["personal"]:
        return "personal"
    if distance < t["social"]:
        return "social"
    return "public"


class ProxemicsAugmentation:
    """Compute proxemic readings and inject them into ``pb.social``."""

    section = "social"  # tells the pipeline where to merge results

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        self.thresholds = thresholds or {
            "intimate": INTIMATE,
            "personal": PERSONAL,
            "social": SOCIAL,
        }

    @classmethod
    def from_profile(cls, profile: object) -> "ProxemicsAugmentation":
        """Create from a CulturalProfile."""
        return cls(thresholds={
            "intimate": profile.intimate_distance,
            "personal": profile.personal_distance,
            "social": profile.social_distance,
        })

    def augment(self, world: Dict[str, Any]) -> Dict[str, Any]:
        robot_pose = world.get("robot_pose", (0, 0, 0))
        rx, ry = robot_pose[0], robot_pose[1]
        robot_heading = robot_pose[3] if len(robot_pose) > 3 else (
            robot_pose[2] if len(robot_pose) > 2 else 0.0
        )

        agents = world.get("agents", [])
        readings: List[ProxemicReading] = []

        for agent in agents:
            pose = agent.get("pose", (0, 0))
            ax, ay = pose[0], pose[1]
            dist = point_distance(rx, ry, ax, ay)
            b = bearing(rx, ry, ax, ay) - robot_heading
            # normalise to [-pi, pi]
            b = math.atan2(math.sin(b), math.cos(b))
            zone = classify_zone(dist, self.thresholds)
            readings.append(ProxemicReading(
                entity_id=agent.get("id", "unknown"),
                distance=dist,
                zone=zone,
                bearing=b,
            ))

        closest_dist = min((r.distance for r in readings), default=float("inf"))
        closest_zone = classify_zone(closest_dist, self.thresholds) if readings else "public"

        return {
            "proxemics": {
                "readings": readings,
                "closest_distance": closest_dist,
                "closest_zone": closest_zone,
                "num_in_personal": sum(
                    1 for r in readings if r.zone in ("intimate", "personal")
                ),
            }
        }
