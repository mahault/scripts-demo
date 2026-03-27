"""TIAGo-specific gaze skill using head pan/tilt driver."""

from __future__ import annotations

import math
from typing import Any

from architecture_core.skills.gaze_skill import GazeSkill

from plugins.tiago_webots.robot.driver import TiagoDriver


class TiagoGazeSkill(GazeSkill):
    """Gaze control with TIAGo head pan/tilt hardware."""

    def __init__(self, driver: TiagoDriver) -> None:
        super().__init__()
        self.driver = driver

    def _on_gaze_update(self, target: Any) -> None:
        if isinstance(target, (tuple, list)) and len(target) >= 2:
            # Simple conversion: target (x, y, z) → pan/tilt angles
            # This is a stub — real IK would use the robot's current pose
            x, y = float(target[0]), float(target[1])
            pan = math.atan2(y, x) if (x != 0 or y != 0) else 0.0
            z = float(target[2]) if len(target) > 2 else 0.0
            dist = math.sqrt(x * x + y * y)
            tilt = math.atan2(-z, dist) if dist > 0 else 0.0
            self.driver.set_head_pan_tilt(pan, tilt)
