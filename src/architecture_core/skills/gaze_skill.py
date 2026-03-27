"""Robot-agnostic gaze control skill.

Controls where the robot looks. Modes: look_at_agent, look_at_point,
scan (sweep scene), and avert (look away from target).

Gaze is critical for social signaling -- approach gaze conveys engagement,
avert gaze signals discomfort or deference.

Actions are expressed through SkillUpdate.params; actual head motor control
is delegated to hook methods that TIAGo (or other plugins) can override.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from architecture_core.cognition.tom.intent_policy import IntentPolicy
from architecture_core.core.status import Status
from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate
from architecture_core.skills.base import Skill


class GazeSkill(Skill):
    """Gaze direction control with multiple modes."""

    name = "gaze"

    STABILIZE_TICKS = 3  # ticks at target before SUCCESS

    def __init__(self) -> None:
        self._mode: str = "look_at_point"
        self._target: Any = None  # agent ID, (x,y,z) point, or direction
        self._stabilized_ticks: int = 0
        self._scan_index: int = 0
        self._phase: str = "idle"
        self._elapsed: float = 0.0
        self._timeout: float = 10.0
        self._last_t: float = 0.0

    # ------------------------------------------------------------------
    # Skill interface
    # ------------------------------------------------------------------
    def start(self, req: SkillRequest) -> None:
        self._mode = req.params.get("mode", "look_at_point")
        self._target = req.goal.get("target")
        self._timeout = req.timeout_s
        self._stabilized_ticks = 0
        self._scan_index = 0
        self._phase = "tracking"
        self._elapsed = 0.0
        self._last_t = 0.0

    def tick(self, pb: PerceptBundle, update: SkillUpdate) -> Status:
        # Track elapsed time
        if self._last_t > 0.0:
            self._elapsed += pb.t - self._last_t
        self._last_t = pb.t

        # Emergency stop
        if update.constraints.get("emergency_stop"):
            return "RUNNING"

        # Timeout
        if self._elapsed > self._timeout:
            return "FAILURE"

        # Intent-driven gaze override
        if update.params.get("gaze_avert"):
            return self._tick_avert(pb, update)

        if self._mode == "look_at_agent":
            return self._tick_look_at_agent(pb, update)
        elif self._mode == "look_at_point":
            return self._tick_look_at_point(pb, update)
        elif self._mode == "scan":
            return self._tick_scan(pb, update)
        elif self._mode == "avert":
            return self._tick_avert(pb, update)

        return "RUNNING"

    def stop(self, reason: str = "") -> None:
        self._phase = "idle"
        self._stabilized_ticks = 0
        self._elapsed = 0.0

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------
    def _tick_look_at_agent(
        self, pb: PerceptBundle, update: SkillUpdate
    ) -> Status:
        agent_pos = self._find_agent_position(pb, self._target)
        if agent_pos is None:
            self._stabilized_ticks = 0
            return "RUNNING"

        gaze_target = agent_pos
        update.params["gaze_target"] = gaze_target
        self._on_gaze_update(gaze_target)

        self._stabilized_ticks += 1
        if self._stabilized_ticks >= self.STABILIZE_TICKS:
            return "SUCCESS"
        return "RUNNING"

    def _tick_look_at_point(
        self, pb: PerceptBundle, update: SkillUpdate
    ) -> Status:
        if self._target is None:
            return "FAILURE"

        gaze_target = self._target
        update.params["gaze_target"] = gaze_target
        self._on_gaze_update(gaze_target)

        self._stabilized_ticks += 1
        if self._stabilized_ticks >= self.STABILIZE_TICKS:
            return "SUCCESS"
        return "RUNNING"

    def _tick_scan(self, pb: PerceptBundle, update: SkillUpdate) -> Status:
        targets = pb.attention.get("salient_targets", [])
        if not targets:
            # No targets to scan — just complete
            return "SUCCESS"

        if self._scan_index >= len(targets):
            return "SUCCESS"

        gaze_target = targets[self._scan_index]
        update.params["gaze_target"] = gaze_target
        self._on_gaze_update(gaze_target)

        self._stabilized_ticks += 1
        if self._stabilized_ticks >= self.STABILIZE_TICKS:
            self._scan_index += 1
            self._stabilized_ticks = 0

        return "RUNNING"

    def _tick_avert(self, pb: PerceptBundle, update: SkillUpdate) -> Status:
        # Look away from target agent
        agent_pos = self._find_agent_position(pb, self._target)
        robot_pose = pb.world.get("robot_pose", (0, 0, 0))

        if agent_pos is not None:
            # Compute direction away from agent
            rx, ry = robot_pose[0], robot_pose[1]
            ax, ay = agent_pos[0], agent_pos[1]
            dx, dy = rx - ax, ry - ay
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                avert_target = (rx + dx / dist, ry + dy / dist, 0.0)
            else:
                avert_target = (rx, ry - 1.0, 0.0)  # look down
        else:
            # Default: look down
            rx, ry = robot_pose[0], robot_pose[1]
            avert_target = (rx, ry, -1.0)

        update.params["gaze_target"] = avert_target
        self._on_gaze_update(avert_target)

        self._stabilized_ticks += 1
        if self._stabilized_ticks >= self.STABILIZE_TICKS:
            return "SUCCESS"
        return "RUNNING"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_agent_position(
        self, pb: PerceptBundle, agent_id: Any
    ) -> Optional[Tuple[float, ...]]:
        agents = pb.world.get("agents", [])
        if not agents:
            agents = pb.social.get("agents", [])
        for a in agents:
            if isinstance(a, dict) and a.get("id") == agent_id:
                return a.get("pose", a.get("position"))
        return None

    # ------------------------------------------------------------------
    # Hook methods (no-op here, overridden by plugin subclasses)
    # ------------------------------------------------------------------
    def _on_gaze_update(self, target: Any) -> None:
        """Called when gaze target is computed. Override for motor control."""


class GazeIntentPolicy(IntentPolicy):
    """Maps social intents to gaze-specific parameters."""

    def approach(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["gaze_tracking"] = True
        base.params["track_persistence"] = 1.0
        return base

    def avoid(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["gaze_avert"] = True
        base.params["avert_direction"] = "down"
        return base

    def yield_(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["gaze_tracking"] = True
        base.params["track_persistence"] = 0.5
        return base

    def wait(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["gaze_hold"] = True
        return base
