"""architecture_core/safety/constraints.py

Constraint data types produced by the norm engine and consumed
by the safety shield.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from architecture_core.safety.geometry import Rect


@dataclass
class DistanceConstraint:
    """Minimum distance to maintain from a specific entity."""
    entity_id: str
    min_distance: float  # metres
    current_distance: float = 0.0


@dataclass
class KeepoutConstraint:
    """A spatial zone the robot must not enter."""
    zone: Rect
    reason: str = ""


@dataclass
class SpeedConstraint:
    """Maximum speed (global or zone-based)."""
    max_linear: float = float("inf")   # m/s
    max_angular: float = float("inf")  # rad/s
    reason: str = ""


@dataclass
class SafetyConstraintSet:
    """Collected constraints for the safety shield to enforce."""
    distance: List[DistanceConstraint] = field(default_factory=list)
    keepout: List[KeepoutConstraint] = field(default_factory=list)
    speed: List[SpeedConstraint] = field(default_factory=list)
