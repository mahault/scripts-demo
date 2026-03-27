"""architecture_core/perception/sensors/base_sensors.py

Abstract sensor interface.  Plugins implement this so that the
PerceptionPipeline can read robot-agnostic data.

Recommended keys returned by ``read()``:
    robot_pose  : tuple (x, y, theta) or (x, y, z, heading)
    agents      : list of {"id": str, "pose": (x, y), "velocity": (vx, vy)}
    obstacles   : list of {"pose": (x, y), "radius": float}
    timestamp   : float
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class SensorInterface(ABC):
    @abstractmethod
    def read(self) -> Dict[str, Any]:
        """Return current sensor readings as a dict."""
        ...
