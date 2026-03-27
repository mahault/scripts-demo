"""Dynamic keepout zones from perception.

Creates keepout zones from detected obstacles and hazards in the
PerceptBundle, with configurable safety margins. Vetoes actions
if the robot is currently inside a keepout zone.
"""

from __future__ import annotations

import math

from architecture_core.cognition.norms.rules import NormRule
from architecture_core.core.types import NormativeConstraints, PerceptBundle, SkillRequest


class DynamicKeepoutRule(NormRule):
    """Create keepout zones from perceived obstacles and hazards."""

    name = "dynamic_keepout"

    def __init__(
        self,
        obstacle_margin: float = 0.5,
        hazard_margin: float = 1.0,
    ) -> None:
        self.obstacle_margin = obstacle_margin
        self.hazard_margin = hazard_margin

    def evaluate(
        self, pb: PerceptBundle, candidate: SkillRequest
    ) -> NormativeConstraints:
        constraints = NormativeConstraints()
        zones = []

        # Obstacles from perception
        for obs in pb.world.get("obstacles", []):
            pose = obs.get("pose", (0, 0))
            radius = obs.get("radius", 0.3) + self.obstacle_margin
            zones.append({
                "center": pose,
                "radius": radius,
                "source": "obstacle",
            })

        # Hazards from perception
        for haz in pb.world.get("hazards", []):
            pose = haz.get("pose", (0, 0))
            radius = haz.get("radius", 0.5) + self.hazard_margin
            zones.append({
                "center": pose,
                "radius": radius,
                "source": "hazard",
            })

        if zones:
            constraints.hard["keepout_zones"] = zones

        # Veto if robot is inside any zone
        robot_pose = pb.world.get("robot_pose", (0, 0, 0))
        rx, ry = robot_pose[0], robot_pose[1]
        for z in zones:
            cx, cy = z["center"][0], z["center"][1]
            dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            if dist < z["radius"]:
                constraints.veto = (
                    f"Inside keepout zone ({z['source']})"
                )
                break

        return constraints
