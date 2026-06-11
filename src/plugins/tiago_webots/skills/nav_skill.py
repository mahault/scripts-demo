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

    # Body + arm clearance used for path planning and reactive avoidance.
    ROBOT_RADIUS = 0.27
    FURNITURE_CLEARANCE = 0.25

    def __init__(self, driver: TiagoDriver) -> None:
        self.driver = driver
        self.goal_x: float = 0.0
        self.goal_y: float = 0.0
        self.goal_tolerance: float = 0.3
        self._detour_target: tuple[float, float] | None = None
        self._escape_furniture: dict | None = None

    # ------------------------------------------------------------------
    # Skill interface
    # ------------------------------------------------------------------
    def start(self, req: SkillRequest) -> None:
        self.goal_x = req.goal.get("x", 0.0)
        self.goal_y = req.goal.get("y", 0.0)
        self.goal_tolerance = req.params.get("goal_tolerance", 0.3)
        self._detour_target = None
        self._escape_furniture = None
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

        # Build furniture list with AABB half-extents and inflated keepout.
        furniture = []
        for f in pb.world.get("furniture", []):
            fp = f.get("position", (0, 0))
            fw = f.get("width", 0.5)
            fd = f.get("depth", 0.5)
            rot = f.get("rotation", 0.0)
            hw, hd = fw / 2.0, fd / 2.0
            cos_r = abs(math.cos(rot))
            sin_r = abs(math.sin(rot))
            aabb_hw = hw * cos_r + hd * sin_r
            aabb_hd = hw * sin_r + hd * cos_r
            margin = self.ROBOT_RADIUS + self.FURNITURE_CLEARANCE
            furniture.append({
                "position": fp,
                "aabb_hw": aabb_hw,
                "aabb_hd": aabb_hd,
                "margin": margin,
                "xmin": fp[0] - aabb_hw - margin,
                "xmax": fp[0] + aabb_hw + margin,
                "ymin": fp[1] - aabb_hd - margin,
                "ymax": fp[1] + aabb_hd + margin,
            })

        # If the direct path to the goal is blocked, plan a detour waypoint.
        # The detour is recomputed each tick so the robot can adapt as it moves.
        self._detour_target = self._plan_detour(cx, cy, self.goal_x, self.goal_y, furniture)
        target_x, target_y = self.goal_x, self.goal_y
        if self._detour_target is not None:
            target_x, target_y = self._detour_target
            # If we are escaping an inflated obstacle, don't declare the
            # detour reached until we are clearly outside that obstacle.
            if self._escape_furniture is not None:
                f = self._escape_furniture
                inside = (f["xmin"] <= cx <= f["xmax"] and f["ymin"] <= cy <= f["ymax"])
                if inside:
                    reached = False

        # Extract obstacles for reactive avoidance (AABB nearest-point).
        obstacle_list = []
        for f in furniture:
            fp = f["position"]
            nx = max(fp[0] - f["aabb_hw"], min(cx, fp[0] + f["aabb_hw"]))
            ny = max(fp[1] - f["aabb_hd"], min(cy, fp[1] + f["aabb_hd"]))
            obstacle_list.append((nx, ny, f["margin"]))
        # Also avoid other agents
        for agent in pb.world.get("agents", []):
            ap = agent.get("pose", (0, 0))
            obstacle_list.append((ap[0], ap[1], 0.4))

        # Drive toward target.  Furniture avoidance is handled globally by the
        # detour planner; only use reactive repulsion for dynamic agents so the
        # robots do not get pushed off their collision-free routes.
        agent_obstacles = [(ap[0], ap[1], 0.4) for agent in pb.world.get("agents", [])
                           for ap in [agent.get("pose", (0, 0))]]
        reached = self.driver.navigate_to_target(
            cx, cy, heading, target_x, target_y, speed_scale,
            obstacles=agent_obstacles,
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

    def _compute_escape_target(
        self,
        x1: float,
        y1: float,
        f: dict,
    ) -> tuple[float, float]:
        """Return a point well outside the inflated AABB, preferring the
        direction that also makes progress toward the current goal."""
        dx_w = x1 - f["xmin"]
        dx_e = f["xmax"] - x1
        dy_s = y1 - f["ymin"]
        dy_n = f["ymax"] - y1

        # Choose the escape direction that also points toward the goal
        gx, gy = self.goal_x, self.goal_y
        candidates = [
            (dx_w, (f["xmin"] - 0.6, y1), x1 - f["xmin"]),
            (dx_e, (f["xmax"] + 0.6, y1), f["xmax"] - x1),
            (dy_s, (x1, f["ymin"] - 0.6), y1 - f["ymin"]),
            (dy_n, (x1, f["ymax"] + 0.6), f["ymax"] - y1),
        ]

        # Filter to directions that move toward the goal
        toward_goal = []
        for _, (tx, ty), _ in candidates:
            if (tx - x1) * (gx - x1) + (ty - y1) * (gy - y1) > 0:
                toward_goal.append((tx, ty))

        if toward_goal:
            # Pick the one closest to the goal
            best = min(toward_goal, key=lambda p: (p[0] - gx) ** 2 + (p[1] - gy) ** 2)
            return best

        # Fallback: closest boundary
        min_d = min(dx_w, dx_e, dy_s, dy_n)
        if min_d == dx_w:
            return (f["xmin"] - 0.6, y1)
        if min_d == dx_e:
            return (f["xmax"] + 0.6, y1)
        if min_d == dy_s:
            return (x1, f["ymin"] - 0.6)
        return (x1, f["ymax"] + 0.6)

    # ------------------------------------------------------------------
    # Simple detour planner
    # ------------------------------------------------------------------
    def _plan_detour(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        furniture: list[dict],
    ) -> tuple[float, float] | None:
        """Return an intermediate waypoint if the direct path is blocked.

        Uses the inflated AABB of each furniture item.  When the straight
        line from start to goal cuts through an AABB, generate candidate
        waypoints just outside the four AABB corners and pick the one that
        gives the shortest two-segment path without hitting another AABB.

        If the robot is already inside an inflated AABB (e.g., it cut a
        corner too tightly), return the nearest point on the AABB boundary
        so the robot escapes before continuing toward the goal.
        """
        # Escape first: if we are inside any inflated AABB, move to the
        # closest point on its boundary plus a generous outward nudge.
        if self._escape_furniture is not None:
            f = self._escape_furniture
            inside = f["xmin"] <= x1 <= f["xmax"] and f["ymin"] <= y1 <= f["ymax"]
            if inside:
                return self._compute_escape_target(x1, y1, f)
            self._escape_furniture = None

        for f in furniture:
            if f["xmin"] <= x1 <= f["xmax"] and f["ymin"] <= y1 <= f["ymax"]:
                self._escape_furniture = f
                return self._compute_escape_target(x1, y1, f)

        # Find the first blocking furniture along the direct path
        blocker = None
        min_t = 2.0
        for f in furniture:
            t = self._line_aabb_intersection_t(
                x1, y1, x2, y2,
                f["xmin"], f["ymin"], f["xmax"], f["ymax"],
            )
            if t is not None and t < min_t:
                min_t = t
                blocker = f
        if blocker is None:
            return None

        # Candidate waypoints: the four corners of the inflated AABB
        candidates = [
            (blocker["xmin"], blocker["ymin"]),
            (blocker["xmin"], blocker["ymax"]),
            (blocker["xmax"], blocker["ymin"]),
            (blocker["xmax"], blocker["ymax"]),
        ]

        best = None
        best_cost = float("inf")
        for cx, cy in candidates:
            # The candidate must be reachable from the robot and the goal
            # reachable from the candidate without crossing any inflated AABB.
            if self._line_aabb_intersection_t(
                x1, y1, cx, cy,
                blocker["xmin"], blocker["ymin"], blocker["xmax"], blocker["ymax"],
            ) is not None:
                continue
            if self._line_aabb_intersection_t(
                cx, cy, x2, y2,
                blocker["xmin"], blocker["ymin"], blocker["xmax"], blocker["ymax"],
            ) is not None:
                continue
            # Also avoid crossing any other furniture inflated AABB
            blocked = False
            for f in furniture:
                if f is blocker:
                    continue
                if (self._line_aabb_intersection_t(
                        x1, y1, cx, cy,
                        f["xmin"], f["ymin"], f["xmax"], f["ymax"]) is not None
                    or self._line_aabb_intersection_t(
                        cx, cy, x2, y2,
                        f["xmin"], f["ymin"], f["xmax"], f["ymax"]) is not None):
                    blocked = True
                    break
            if blocked:
                continue
            cost = math.hypot(cx - x1, cy - y1) + math.hypot(x2 - cx, y2 - cy)
            if cost < best_cost:
                best_cost = cost
                best = (cx, cy)

        return best

    @staticmethod
    def _line_aabb_intersection_t(
        x1: float, y1: float,
        x2: float, y2: float,
        xmin: float, ymin: float,
        xmax: float, ymax: float,
    ) -> float | None:
        """Return the smallest t in [0,1] where the segment enters the AABB."""
        dx = x2 - x1
        dy = y2 - y1
        t_min = 0.0
        t_max = 1.0

        # X slab
        if abs(dx) < 1e-9:
            if x1 < xmin or x1 > xmax:
                return None
        else:
            tx1 = (xmin - x1) / dx
            tx2 = (xmax - x1) / dx
            if tx1 > tx2:
                tx1, tx2 = tx2, tx1
            t_min = max(t_min, tx1)
            t_max = min(t_max, tx2)
            if t_min > t_max:
                return None

        # Y slab
        if abs(dy) < 1e-9:
            if y1 < ymin or y1 > ymax:
                return None
        else:
            ty1 = (ymin - y1) / dy
            ty2 = (ymax - y1) / dy
            if ty1 > ty2:
                ty1, ty2 = ty2, ty1
            t_min = max(t_min, ty1)
            t_max = min(t_max, ty2)
            if t_min > t_max:
                return None

        return t_min if t_min <= 1.0 else None

    def stop(self, reason: str = "") -> None:
        self.driver.stop()
