"""Webots Supervisor-based sensor reading for TIAGo.

Extracted from tiago_empathic.py (lines 127-249).  Implements the
social-layer ``SensorInterface`` so the perception pipeline can
consume it without knowing about Webots.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from architecture_core.perception.sensors.base_sensors import SensorInterface


class TiagoWebotsSensors(SensorInterface):
    """Reads world state via the Webots Supervisor API."""

    def __init__(self, robot, self_node, name: str,
                 observe_agent_internals: bool = True) -> None:
        self.robot = robot
        self.self_node = self_node
        self.name = name
        # When False, this agent may read only the *observable* state of other
        # agents — their position — never their customData (intended goal, role
        # priority, published state).  The learner sets this False so it cannot
        # access any internal process of the humans it watches.
        self.observe_agent_internals = observe_agent_internals

        # Discover other robots and world geometry once at startup
        self._other_robots: List[Dict[str, Any]] = []
        # Legacy single-robot attrs (for backward compat)
        self.other_node: Optional[Any] = None
        self.other_name: Optional[str] = None
        self.other_alpha: float = 0.5

        self._find_all_robots()
        self._hazards = self._discover_hazards()
        self._cues = self._discover_deontic_cues()
        self._arena_bounds = self._get_arena_bounds()

    # ------------------------------------------------------------------
    # SensorInterface
    # ------------------------------------------------------------------
    def read(self) -> Dict[str, Any]:
        x, y, z = self._get_position()
        heading = self._get_heading()
        agents: List[Dict[str, Any]] = []

        for other in self._other_robots:
            node = other["node"]
            try:
                pos = node.getPosition()
                ox, oy = pos[0], pos[1]
            except Exception:
                continue
            # Only the agent's position is observable.  Reading its intended goal
            # from customData would be accessing an internal process — forbidden
            # unless this agent is allowed to (e.g. the scripted worker/shopper).
            ogx, ogy = ox, oy
            if self.observe_agent_internals:
                try:
                    cd = node.getField("customData")
                    if cd:
                        parts = cd.getSFString().strip().split(",")
                        if len(parts) >= 2:
                            ogx, ogy = float(parts[0]), float(parts[1])
                except Exception:
                    pass
            agents.append({
                "id": other["name"],
                "pose": (ox, oy),
                "goal": (ogx, ogy),
                "alpha": other["alpha"],
            })

        return {
            "robot_pose": (x, y, z, heading),
            "agents": agents,
            "hazards": self._hazards,
            "deontic_cues": self._read_deontic_cues(x, y),
            "arena_bounds": self._arena_bounds,
            "timestamp": self.robot.getTime(),
        }

    # ------------------------------------------------------------------
    # Discovery helpers (run once)
    # ------------------------------------------------------------------
    def _find_all_robots(self) -> None:
        """Discover ALL other actors in the scene.

        Actors are either wheeled robots (``Tiago``) or walking humans
        (``PedestrianAgent``) — both carry a ``name`` and a ``customData``
        string, so the rest of the pipeline treats them identically.
        """
        root = self.robot.getRoot()
        children = root.getField("children")
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            type_name = node.getTypeName()
            if type_name == "Tiago" or "Pedestrian" in type_name:
                name_field = node.getField("name")
                if name_field:
                    rname = name_field.getSFString()
                    if rname != self.name:
                        alpha = 0.5
                        if self.observe_agent_internals:
                            cd = node.getField("customData")
                            if cd:
                                parts = cd.getSFString().strip().split(",")
                                if len(parts) >= 3:
                                    alpha = float(parts[2])
                        self._other_robots.append({
                            "node": node,
                            "name": rname,
                            "alpha": alpha,
                        })
                        # Keep legacy attrs pointing to first found
                        if self.other_node is None:
                            self.other_node = node
                            self.other_name = rname
                            self.other_alpha = alpha

    def _discover_hazards(self) -> List[Tuple[float, float, float, float]]:
        hazards: List[Tuple[float, float, float, float]] = []
        root = self.robot.getRoot()
        children = root.getField("children")
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            try:
                type_name = node.getTypeName()
            except Exception:
                continue
            if type_name == "HazardObstacle":
                try:
                    pos = node.getField("translation").getSFVec3f()
                    size = node.getField("size").getSFVec3f()
                    hazards.append((pos[0], pos[1], size[0] / 2.0, size[1] / 2.0))
                except Exception:
                    pass
        return hazards

    def _discover_deontic_cues(self) -> List[Any]:
        """Discover cue nodes by name convention.

        Any node with a name starting with 'CUE_' is treated as a deontic cue.
        This makes world-authoring simple: drop a Solid with name
        'CUE_QUEUE_HERE' or 'CUE_STAFF_ONLY' in the scene.
        """
        cues: List[Any] = []
        root = self.robot.getRoot()
        children = root.getField("children")
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            try:
                name_field = node.getField("name")
                if not name_field:
                    continue
                n = name_field.getSFString()
                if n and n.startswith("CUE_"):
                    cues.append(node)
            except Exception:
                continue
        return cues

    def _read_deontic_cues(self, rx: float, ry: float) -> List[Dict[str, Any]]:
        """Return cue dicts with an 'active' flag based on distance."""
        out: List[Dict[str, Any]] = []
        for node in self._cues:
            try:
                name = node.getField("name").getSFString()
                pos = node.getField("translation").getSFVec3f()
                dx, dy = pos[0] - rx, pos[1] - ry
                d = math.sqrt(dx * dx + dy * dy)

                # Simple activation radius: within 4m means the cue is readable.
                active = d <= 4.0

                # Parse cue type from name: CUE_QUEUE_HERE -> queue_here
                cue_type = name[4:].lower()

                out.append({
                    "name": name,
                    "type": cue_type,
                    "pose": (pos[0], pos[1]),
                    "distance": d,
                    "active": active,
                })
            except Exception:
                continue
        return out

    def _get_arena_bounds(self) -> Tuple[float, float, float, float]:
        root = self.robot.getRoot()
        children = root.getField("children")
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            try:
                type_name = node.getTypeName()
            except Exception:
                continue
            if type_name == "RectangleArena":
                try:
                    fs = node.getField("floorSize").getSFVec2f()
                    hx, hy = fs[0] / 2.0, fs[1] / 2.0
                    return (-hx, hx, -hy, hy)
                except Exception:
                    pass
        return (-2.5, 2.5, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Per-tick helpers
    # ------------------------------------------------------------------
    def _get_position(self) -> Tuple[float, float, float]:
        if self.self_node:
            pos = self.self_node.getPosition()
            return pos[0], pos[1], pos[2]
        return 0.0, 0.0, 0.0

    def _get_heading(self) -> float:
        if self.self_node:
            rot = self.self_node.getOrientation()
            return math.atan2(rot[3], rot[0])
        return 0.0

    def _get_other_position(self) -> Tuple[Optional[float], Optional[float]]:
        if self.other_node:
            pos = self.other_node.getPosition()
            return pos[0], pos[1]
        return None, None

    def _get_other_goal(self) -> Tuple[float, float]:
        if self.other_node and self.observe_agent_internals:
            cd = self.other_node.getField("customData")
            if cd:
                parts = cd.getSFString().strip().split(",")
                if len(parts) >= 2:
                    return float(parts[0]), float(parts[1])
        return 0.0, 0.0
