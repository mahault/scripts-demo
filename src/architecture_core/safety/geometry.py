"""architecture_core/safety/geometry.py

Pure 2D geometry utilities used by the safety shield, norm engine,
and proxemics augmentation.  No robot dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Circle:
    """An agent or obstacle footprint."""
    x: float
    y: float
    radius: float


@dataclass
class Rect:
    """An axis-aligned keepout zone."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def point_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def circle_circle_distance(a: Circle, b: Circle) -> float:
    """Surface-to-surface distance (negative means overlap)."""
    return point_distance(a.x, a.y, b.x, b.y) - a.radius - b.radius


def point_in_rect(x: float, y: float, r: Rect) -> bool:
    """Check if point is inside a rectangle."""
    return r.x_min <= x <= r.x_max and r.y_min <= y <= r.y_max


def circle_rect_overlap(c: Circle, r: Rect) -> bool:
    """Check if a circle overlaps a rectangle."""
    # Nearest point on rect to circle centre
    cx = max(r.x_min, min(c.x, r.x_max))
    cy = max(r.y_min, min(c.y, r.y_max))
    return point_distance(c.x, c.y, cx, cy) <= c.radius


def clamp_position(
    x: float,
    y: float,
    keepout_zones: List[Rect],
    agent_radius: float = 0.25,
) -> Tuple[float, float]:
    """Push a position out of any keepout zone (nearest-edge projection)."""
    for zone in keepout_zones:
        inflated = Rect(
            zone.x_min - agent_radius,
            zone.y_min - agent_radius,
            zone.x_max + agent_radius,
            zone.y_max + agent_radius,
        )
        if point_in_rect(x, y, inflated):
            # Push to nearest edge
            dx_min = abs(x - inflated.x_min)
            dx_max = abs(x - inflated.x_max)
            dy_min = abs(y - inflated.y_min)
            dy_max = abs(y - inflated.y_max)
            nearest = min(dx_min, dx_max, dy_min, dy_max)
            if nearest == dx_min:
                x = inflated.x_min - 0.01
            elif nearest == dx_max:
                x = inflated.x_max + 0.01
            elif nearest == dy_min:
                y = inflated.y_min - 0.01
            else:
                y = inflated.y_max + 0.01
    return x, y


def bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    """Bearing angle in radians from (x1,y1) toward (x2,y2)."""
    return math.atan2(y2 - y1, x2 - x1)
