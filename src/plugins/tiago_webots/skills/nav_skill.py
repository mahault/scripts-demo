"""TIAGo navigation skill wrapping the low-level driver.

Implements the ``Skill`` interface so the Executive can manage its
lifecycle without knowing about Webots motors.
"""

from __future__ import annotations

import math

from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate
from architecture_core.core.status import Status
from architecture_core.skills.base import Skill

from plugins.tiago_webots.robot.driver import TiagoDriver


class TiagoNavSkill(Skill):
    name = "navigate"

    def __init__(self, driver: TiagoDriver) -> None:
        self.driver = driver
        self.goal_x: float = 0.0
        self.goal_y: float = 0.0
        self.goal_tolerance: float = 0.3

    # ------------------------------------------------------------------
    # Skill interface
    # ------------------------------------------------------------------
    def start(self, req: SkillRequest) -> None:
        self.goal_x = req.goal.get("x", 0.0)
        self.goal_y = req.goal.get("y", 0.0)
        self.goal_tolerance = req.params.get("goal_tolerance", 0.3)
        self._tick_count = 0
        print(f"  [NAV] start goal=({self.goal_x:.2f},{self.goal_y:.2f})")

    def tick(self, pb: PerceptBundle, update: SkillUpdate) -> Status:
        # Emergency stop from safety shield
        if update.constraints.get("emergency_stop"):
            self.driver.stop()
            return "RUNNING"

        speed_scale = update.params.get("speed_scale", 1.0)
        if speed_scale <= 0.0:
            self.driver.stop()
            return "RUNNING"

        # Respect stop_distance: hold position if an agent is within range
        stop_dist = update.params.get("stop_distance", 0.0)
        if stop_dist > 0:
            pose = pb.world.get("robot_pose", (0, 0, 0, 0))
            for agent in pb.world.get("agents", []):
                ap = agent.get("pose", (0, 0))
                ad = math.sqrt((pose[0] - ap[0]) ** 2 + (pose[1] - ap[1]) ** 2)
                if ad < stop_dist:
                    self.driver.stop()
                    return "RUNNING"

        # Use ToM-planned target if available, else use goal directly
        target_x = update.params.get("target_x", self.goal_x)
        target_y = update.params.get("target_y", self.goal_y)

        # Current pose from perception
        pose = pb.world.get("robot_pose", (0, 0, 0, 0))
        cx, cy = pose[0], pose[1]
        heading = pose[3] if len(pose) > 3 else 0.0

        # Extract obstacles for reactive avoidance (AABB nearest-point)
        obstacle_list = []
        for f in pb.world.get("furniture", []):
            fp = f.get("position", (0, 0))
            fw = f.get("width", 0.5)
            fd = f.get("depth", 0.5)
            rot = f.get("rotation", 0.0)
            # Compute axis-aligned half-extents after rotation
            hw, hd = fw / 2.0, fd / 2.0
            cos_r = abs(math.cos(rot))
            sin_r = abs(math.sin(rot))
            aabb_hw = hw * cos_r + hd * sin_r
            aabb_hd = hw * sin_r + hd * cos_r
            # Nearest point on AABB to robot position (edge-based repulsion)
            nx = max(fp[0] - aabb_hw, min(cx, fp[0] + aabb_hw))
            ny = max(fp[1] - aabb_hd, min(cy, fp[1] + aabb_hd))
            obstacle_list.append((nx, ny, 0.0))
        # Also avoid other agents
        for agent in pb.world.get("agents", []):
            ap = agent.get("pose", (0, 0))
            obstacle_list.append((ap[0], ap[1], 0.3))

        # Drive toward target with obstacle avoidance
        reached = self.driver.navigate_to_target(
            cx, cy, heading, target_x, target_y, speed_scale,
            obstacles=obstacle_list,
        )

        # Check if the actual goal (not intermediate target) has been reached
        dist_to_goal = math.sqrt((self.goal_x - cx) ** 2 + (self.goal_y - cy) ** 2)

        self._tick_count += 1
        if self._tick_count % 60 == 0:
            n_obs = len(obstacle_list)
            close_obs = sum(1 for ox, oy, _ in obstacle_list
                           if math.sqrt((cx-ox)**2 + (cy-oy)**2) < 0.5)
            print(f"  [NAV] pos=({cx:.2f},{cy:.2f}) h={heading:.2f}"
                  f" tgt=({target_x:.2f},{target_y:.2f})"
                  f" goal=({self.goal_x:.2f},{self.goal_y:.2f})"
                  f" d={dist_to_goal:.2f} reached={reached}"
                  f" obs={n_obs}(close={close_obs}) spd={speed_scale:.2f}"
                  f" stop_d={stop_dist:.1f}")

        if dist_to_goal < self.goal_tolerance:
            self.driver.stop()
            print(f"  [NAV] SUCCESS dist={dist_to_goal:.2f}")
            return "SUCCESS"

        return "RUNNING"

    def stop(self, reason: str = "") -> None:
        self.driver.stop()
