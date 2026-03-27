"""architecture_core/cognition/tom/models/tom_planner_adapter.py

Adapter around the existing tom_planner.py from the Alignment-experiments
repository.  Converts PerceptBundle/SkillRequest into the planner's
interface and maps the planner output to a social-layer Intent.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Tuple

from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate


class ToMPlannerAdapter:
    """Thin translation layer around ``tom_planner.ToMPlanner``.

    The adapter:
    1. Extracts positions from PerceptBundle
    2. Calls ``planner.plan(my_x, my_y, other_x, other_y, ...)``
    3. Classifies the resulting waypoint into an abstract Intent
    4. Returns a ``SkillUpdate`` with intent + target params
    """

    def __init__(
        self,
        agent_id: int,
        goal_x: float,
        goal_y: float,
        alpha: float,
        planner_path: Optional[str] = None,
    ) -> None:
        if planner_path and planner_path not in sys.path:
            sys.path.insert(0, planner_path)

        from tom_planner import ToMPlanner, configure  # type: ignore[import]

        self._ToMPlanner = ToMPlanner
        self._configure = configure
        self._agent_id = agent_id
        self.alpha = alpha
        self.goal_x = goal_x
        self.goal_y = goal_y
        # Planner created eagerly with default grid; configure_arena()
        # rebuilds it if the grid changes.
        self._planner = ToMPlanner(
            agent_id=agent_id,
            goal_x=goal_x,
            goal_y=goal_y,
            alpha=alpha,
        )

    def configure_arena(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        hazards: Optional[List[Dict[str, Any]]] = None,
        n_x: Optional[int] = None,
        n_y: Optional[int] = None,
    ) -> None:
        """Configure the planner's discretisation grid, then rebuild the planner."""
        self._configure(x_min, x_max, y_min, y_max, hazards=hazards,
                        n_x=n_x, n_y=n_y)
        # Rebuild planner so its matrices match the new grid dimensions
        self._planner = self._ToMPlanner(
            agent_id=self._agent_id,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            alpha=self.alpha,
        )

    def infer_intent(
        self, pb: PerceptBundle, req: SkillRequest
    ) -> SkillUpdate:
        world = pb.world
        robot_pose = world.get("robot_pose", (0, 0, 0, 0))
        my_x, my_y = robot_pose[0], robot_pose[1]

        agents = world.get("agents", [])
        if not agents:
            return SkillUpdate(
                intent="approach",
                params={"target_x": req.goal.get("x", 0),
                        "target_y": req.goal.get("y", 0)},
                debug="no other agents, approach",
            )

        other = agents[0]
        other_x, other_y = other["pose"][0], other["pose"][1]
        other_goal = other.get("goal", (0.0, 0.0))
        other_alpha = other.get("alpha", 0.5)

        target_x, target_y, debug_str = self._planner.plan(
            my_x=my_x,
            my_y=my_y,
            other_x=other_x,
            other_y=other_y,
            other_goal_x=other_goal[0],
            other_goal_y=other_goal[1],
            other_alpha=other_alpha,
        )

        intent = self._classify_intent(
            my_x, my_y, target_x, target_y,
            other_x, other_y,
            req.goal.get("x", 0), req.goal.get("y", 0),
        )

        return SkillUpdate(
            intent=intent,
            params={"target_x": target_x, "target_y": target_y},
            debug=debug_str,
        )

    @staticmethod
    def _classify_intent(
        mx: float, my: float,
        tx: float, ty: float,
        ox: float, oy: float,
        gx: float, gy: float,
    ) -> str:
        dist_goal_now = math.hypot(gx - mx, gy - my)
        dist_goal_tgt = math.hypot(gx - tx, gy - ty)
        dist_other_now = math.hypot(ox - mx, oy - my)
        dist_other_tgt = math.hypot(ox - tx, oy - ty)

        toward_goal = dist_goal_tgt < dist_goal_now - 0.05
        away_from_goal = dist_goal_tgt > dist_goal_now + 0.05
        away_from_other = dist_other_tgt > dist_other_now + 0.05

        move_dist = math.hypot(tx - mx, ty - my)
        if move_dist < 0.05:
            return "wait"
        if toward_goal and not away_from_other:
            return "approach"
        if away_from_goal and away_from_other:
            return "yield"
        if not toward_goal and away_from_other:
            return "avoid"
        if toward_goal:
            return "approach"
        return "neutral"
