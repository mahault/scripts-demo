"""Object-aware sensors for table-clearing tasks.

Extends TiagoWebotsSensors with:
- Discovery of manipulable objects (Orange, Apple, Can) via Supervisor API
- Real-time tracking of object positions and statuses
- Supervisor-based grasping and releasing (teleportation)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from plugins.tiago_webots.perception.sensors import TiagoWebotsSensors


# Object types to discover in the world tree
OBJECT_TYPES = {"Orange", "Apple", "Can"}

# Solid nodes with these name prefixes are also treated as manipulable objects.
# This lets us place custom basket / tote / generic grocery items in the world.
SOLID_OBJECT_PREFIXES = ("ITEM_", "BASKET_", "TOTE_")

# Surface regions for classifying objects (apartment + retail demo worlds)
TABLE_REGIONS = {
    "dining": {"center": (-1.074, -4.944), "radius": 1.0, "height": 0.74},
    "coffee": {"center": (-7.163, -2.555), "radius": 1.0, "height": 0.53},
    # Retail demo regions
    "stock":    {"center": (-5.5, -1.0),  "radius": 2.2, "height": 0.35},
    "shelf_a":  {"center": (-2.0, -5.0),  "radius": 1.0, "height": 0.40},
    "shelf_b":  {"center": (1.5, -5.0),   "radius": 1.0, "height": 0.40},
    "counter":  {"center": (4.5, -7.5),   "radius": 1.2, "height": 0.85},
    "entrance": {"center": (0.0, -8.5),   "radius": 1.2, "height": 0.01},
}

# Proximity threshold for grasping (robot must be this close to object)
GRASP_PROXIMITY = 0.9

# Furniture types with default footprint and physics properties
FURNITURE_TYPES = {
    "Chair":       {"width": 0.5, "depth": 0.5, "mass": 3.0, "movable": True},
    "WoodenChair": {"width": 0.5, "depth": 0.5, "mass": 4.0, "movable": True},
    "Armchair":    {"width": 0.8, "depth": 0.8, "mass": 25.0, "movable": False},
    "Sofa":        {"width": 2.0, "depth": 1.0, "mass": 40.0, "movable": False},
    "Table":       {"width": 1.2, "depth": 0.8, "mass": 15.0, "movable": False},
    "Desk":        {"width": 1.5, "depth": 0.7, "mass": 20.0, "movable": False},
    "Cabinet":     {"width": 1.0, "depth": 0.4, "mass": 30.0, "movable": False},
    # Walls/doors/windows — nav must see these to avoid them
    "Wall":        {"width": 0.3, "depth": 1.0, "mass": 999.0, "movable": False},
    "Window":      {"width": 0.3, "depth": 1.0, "mass": 999.0, "movable": False},
    "Door":        {"width": 0.3, "depth": 1.0, "mass": 999.0, "movable": False},
    "Fridge":      {"width": 0.7, "depth": 0.7, "mass": 80.0, "movable": False},
    "Oven":        {"width": 0.6, "depth": 0.6, "mass": 50.0, "movable": False},
}

# Keepout padding: body_radius + min_clearance
_KEEPOUT_PADDING = 0.27 + 0.35


class TiagoObjectSensors(TiagoWebotsSensors):
    """Extended sensors with object discovery and manipulation."""

    def __init__(self, robot, self_node, name: str) -> None:
        super().__init__(robot, self_node, name)
        self._object_nodes: List[Dict[str, Any]] = []
        self._held_object: Optional[Dict[str, Any]] = None
        self._placed_ids: set = set()
        self._furniture_data: List[Dict[str, Any]] = []
        self._manipulation_target_id: Optional[str] = None
        self._discover_objects()
        self._discover_furniture()

    # ------------------------------------------------------------------
    # SensorInterface override
    # ------------------------------------------------------------------
    def read(self) -> Dict[str, Any]:
        """Extended read: adds objects, furniture, held_object, arm_at_target."""
        data = super().read()
        data["objects"] = self._read_objects()
        data["furniture"] = self._furniture_data
        data["held_object"] = self._held_object is not None
        data["held_object_id"] = self._held_object["id"] if self._held_object else None
        data["arm_at_target"] = self._check_arm_proximity()
        return data

    # ------------------------------------------------------------------
    # Object discovery (run once at startup)
    # ------------------------------------------------------------------
    def _discover_objects(self) -> None:
        """Walk world tree, find manipulable objects, record initial state."""
        root = self.robot.getRoot()
        children = root.getField("children")

        for i in range(children.getCount()):
            node = children.getMFNode(i)
            try:
                type_name = node.getTypeName()
            except Exception:
                continue

            if not self._is_manipulable(node, type_name):
                continue

            name_field = node.getField("name")
            obj_name = name_field.getSFString() if name_field else type_name.lower()

            pos = node.getField("translation").getSFVec3f()
            table = self._classify_table(pos[0], pos[1])

            # Basket / generic solid objects get a friendly type label
            object_type = type_name
            if type_name == "Solid" and any(obj_name.startswith(p) for p in SOLID_OBJECT_PREFIXES):
                if obj_name.startswith("BASKET_"):
                    object_type = "Basket"
                else:
                    object_type = "Item"

            # Use DEF name if available, else construct from type + index
            obj_id = f"{object_type.lower()}_{obj_name}"

            self._object_nodes.append({
                "id": obj_id,
                "type": object_type,
                "node": node,
                "initial_pos": (pos[0], pos[1], pos[2]),
                "table": table,
            })

    def _is_manipulable(self, node, type_name: str) -> bool:
        """Return True if this node should be tracked as a manipulable object."""
        if type_name in OBJECT_TYPES:
            return True
        if type_name == "Solid":
            name_field = node.getField("name")
            if name_field:
                name = name_field.getSFString()
                if any(name.startswith(p) for p in SOLID_OBJECT_PREFIXES):
                    return True
        return False

    def _classify_table(self, x: float, y: float) -> str:
        """Determine which table an object belongs to based on proximity."""
        for name, region in TABLE_REGIONS.items():
            cx, cy = region["center"]
            dx = x - cx
            dy = y - cy
            if math.sqrt(dx * dx + dy * dy) < region["radius"]:
                return name
        return "unknown"

    # ------------------------------------------------------------------
    # Furniture discovery (run once at startup)
    # ------------------------------------------------------------------
    def _discover_furniture(self) -> None:
        """Walk world tree, find furniture items, compute keepout zones."""
        root = self.robot.getRoot()
        children = root.getField("children")
        idx = 0

        for i in range(children.getCount()):
            node = children.getMFNode(i)
            try:
                type_name = node.getTypeName()
            except Exception:
                continue

            if type_name not in FURNITURE_TYPES:
                continue

            props = FURNITURE_TYPES[type_name]
            pos = node.getField("translation").getSFVec3f()

            # Try to get rotation (axis-angle format in Webots)
            rotation = 0.0
            try:
                rot_field = node.getField("rotation")
                if rot_field:
                    rot = rot_field.getSFRotation()
                    # Webots rotation: [ax, ay, az, angle]
                    # For floor items, rotation is around Z axis
                    rotation = rot[3] if abs(rot[2]) > 0.5 else 0.0
            except Exception:
                pass

            # Try to read actual dimensions from the node fields.
            # Cabinet uses depth/outerThickness/rowsHeights/columnsWidths.
            # Table/Wall provide a size field.
            width = props["width"]
            depth = props["depth"]
            try:
                if type_name == "Cabinet":
                    depth = node.getField("depth").getSFFloat()
                    outer_thickness = node.getField("outerThickness").getSFFloat()
                    columns_widths = node.getField("columnsWidths").getMFFloat()
                    width = 2.0 * outer_thickness + sum(columns_widths)
                else:
                    size_field = node.getField("size")
                    if size_field:
                        size = size_field.getSFVec3f()
                        width = size[0]
                        depth = size[1]
            except Exception:
                pass

            # Compute axis-aligned bounding box (rotate footprint)
            hw, hd = width / 2.0, depth / 2.0
            cos_r = abs(math.cos(rotation))
            sin_r = abs(math.sin(rotation))
            aabb_hw = hw * cos_r + hd * sin_r
            aabb_hd = hw * sin_r + hd * cos_r

            # Add keepout padding
            padded_hw = aabb_hw + _KEEPOUT_PADDING
            padded_hd = aabb_hd + _KEEPOUT_PADDING

            furn_id = f"{type_name.lower()}_{idx}"
            idx += 1

            self._furniture_data.append({
                "id": furn_id,
                "type": type_name,
                "position": (pos[0], pos[1]),
                "rotation": rotation,
                "width": width,
                "depth": depth,
                "mass": props["mass"],
                "movable": props["movable"],
                "keepout_x_min": pos[0] - padded_hw,
                "keepout_y_min": pos[1] - padded_hd,
                "keepout_x_max": pos[0] + padded_hw,
                "keepout_y_max": pos[1] + padded_hd,
            })

    # ------------------------------------------------------------------
    # Per-tick object reading
    # ------------------------------------------------------------------

    # Z-height thresholds for globally observable object status
    _HELD_Z_THRESHOLD = 0.85     # gripper height ~0.95
    _TABLE_Z_TOLERANCE = 0.15    # tolerance around table surface

    def _read_objects(self) -> List[Dict[str, Any]]:
        """Read current positions and statuses of all objects.

        Status is inferred from z-height (globally observable) rather than
        local bookkeeping.  This lets both robots see when the other has
        grasped or placed an object — no communication needed.
        """
        agent_positions = self._get_all_agent_positions()
        objects = []

        for obj_info in self._object_nodes:
            obj_id = obj_info["id"]
            node = obj_info["node"]

            # Current 3D position
            try:
                pos = node.getField("translation").getSFVec3f()
                position = (pos[0], pos[1])
                z = pos[2]
            except Exception:
                position = obj_info["initial_pos"][:2]
                z = obj_info["initial_pos"][2]

            # Infer status from z-height (globally observable)
            status, held_by = self._infer_object_status(
                obj_id, position, z, obj_info["table"], agent_positions,
            )

            objects.append({
                "id": obj_id,
                "type": obj_info["type"],
                "position": position,
                "table": obj_info["table"],
                "status": status,
                "held_by": held_by,
            })

        return objects

    def _infer_object_status(
        self,
        obj_id: str,
        position: Tuple[float, float],
        z: float,
        table: str,
        agent_positions: List[Tuple[str, float, float]],
    ) -> Tuple[str, Optional[str]]:
        """Infer object status from z-height (globally observable).

        - z > HELD_Z_THRESHOLD → "held", held_by = nearest agent
        - z ≈ table height → "on_table"
        - otherwise → "placed"
        """
        if z > self._HELD_Z_THRESHOLD:
            held_by = self._nearest_agent(position, agent_positions)
            return "held", held_by

        table_info = TABLE_REGIONS.get(table)
        if table_info:
            if abs(z - table_info["height"]) < self._TABLE_Z_TOLERANCE:
                return "on_table", None

        return "placed", None

    def _get_all_agent_positions(self) -> List[Tuple[str, float, float]]:
        """Collect positions of self + other robots."""
        positions = []
        # Self
        self_pos = self._get_position()
        positions.append((self.name, self_pos[0], self_pos[1]))
        # Other robot
        ox, oy = self._get_other_position()
        if ox is not None and self.other_name is not None:
            positions.append((self.other_name, ox, oy))
        return positions

    @staticmethod
    def _nearest_agent(
        obj_pos: Tuple[float, float],
        agent_positions: List[Tuple[str, float, float]],
    ) -> Optional[str]:
        """Find the nearest agent to an object (2D distance)."""
        best_name = None
        best_dist = float("inf")
        for name, ax, ay in agent_positions:
            dx = obj_pos[0] - ax
            dy = obj_pos[1] - ay
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name

    def set_manipulation_target(self, object_id: Optional[str]) -> None:
        """Set which object the pick/place skill is targeting.

        When set, _check_arm_proximity only returns True for THIS object,
        preventing premature grasping of nearby non-target objects.
        """
        self._manipulation_target_id = object_id

    def _check_arm_proximity(self) -> bool:
        """Check if robot is close enough to the manipulation target.

        Target-specific: only returns True when near the SPECIFIC target
        object (set via set_manipulation_target), not any nearby object.
        Falls back to any-object check if no target is set.
        """
        if not self._object_nodes:
            return False

        robot_pos = self._get_position()

        # Target-specific check (preferred)
        if self._manipulation_target_id is not None:
            for obj_info in self._object_nodes:
                if obj_info["id"] != self._manipulation_target_id:
                    continue
                if obj_info["id"] in self._placed_ids:
                    return False
                if self._held_object and self._held_object["id"] == obj_info["id"]:
                    return True
                try:
                    pos = obj_info["node"].getField("translation").getSFVec3f()
                    dx = robot_pos[0] - pos[0]
                    dy = robot_pos[1] - pos[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    return dist < GRASP_PROXIMITY
                except Exception:
                    return False
            return False

        # Fallback: any on-table object (backward-compatible)
        for obj_info in self._object_nodes:
            if obj_info["id"] in self._placed_ids:
                continue
            if self._held_object and self._held_object["id"] == obj_info["id"]:
                continue
            try:
                pos = obj_info["node"].getField("translation").getSFVec3f()
                dx = robot_pos[0] - pos[0]
                dy = robot_pos[1] - pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < GRASP_PROXIMITY:
                    return True
            except Exception:
                pass

        return False

    # ------------------------------------------------------------------
    # Supervisor-based grasping
    # ------------------------------------------------------------------
    def supervisor_grasp(self, object_id: str) -> bool:
        """Teleport object to robot's gripper position (Supervisor API).

        Returns True if successful, False if object not found or already held.
        """
        if self._held_object is not None:
            return False

        for obj_info in self._object_nodes:
            if obj_info["id"] == object_id:
                robot_pos = self._get_position()
                node = obj_info["node"]

                # Teleport to robot's gripper position (slightly in front)
                heading = self._get_heading()
                gripper_x = robot_pos[0] + 0.3 * math.cos(heading)
                gripper_y = robot_pos[1] + 0.3 * math.sin(heading)
                gripper_z = 0.95  # TIAGo gripper height when extended

                try:
                    trans_field = node.getField("translation")
                    trans_field.setSFVec3f([gripper_x, gripper_y, gripper_z])
                    # Keep the carried object upright and kill residual velocity
                    rot_field = node.getField("rotation")
                    if rot_field:
                        rot_field.setSFRotation([0, 0, 1, 0])
                    node.resetPhysics()
                    self._held_object = obj_info
                    return True
                except Exception:
                    return False

        return False

    def supervisor_release(
        self, position: Tuple[float, float, float],
    ) -> bool:
        """Teleport held object to the specified position.

        Returns True if successful, False if not holding anything.
        """
        if self._held_object is None:
            return False

        node = self._held_object["node"]
        try:
            trans_field = node.getField("translation")
            trans_field.setSFVec3f(list(position))
            # Place upright and zero velocity so the object settles cleanly
            rot_field = node.getField("rotation")
            if rot_field:
                rot_field.setSFRotation([0, 0, 1, 0])
            node.resetPhysics()
            self._placed_ids.add(self._held_object["id"])
            self._held_object = None
            return True
        except Exception:
            return False

    def update_held_position(self) -> None:
        """Keep held object attached to robot's gripper each tick."""
        if self._held_object is None:
            return

        robot_pos = self._get_position()
        heading = self._get_heading()
        gripper_x = robot_pos[0] + 0.3 * math.cos(heading)
        gripper_y = robot_pos[1] + 0.3 * math.sin(heading)
        gripper_z = 0.95

        try:
            node = self._held_object["node"]
            trans_field = node.getField("translation")
            trans_field.setSFVec3f([gripper_x, gripper_y, gripper_z])
            # Cancel drift/roll that builds up while being carried
            rot_field = node.getField("rotation")
            if rot_field:
                rot_field.setSFRotation([0, 0, 1, 0])
            node.resetPhysics()
        except Exception:
            pass

    @property
    def held_object_id(self) -> Optional[str]:
        """ID of currently held object, or None."""
        return self._held_object["id"] if self._held_object else None

    def find_nearest_object(
        self, robot_pos: Tuple[float, float, float],
    ) -> Optional[str]:
        """Find the nearest on-table object to the robot."""
        best_id = None
        best_dist = float("inf")

        for obj_info in self._object_nodes:
            obj_id = obj_info["id"]
            if obj_id in self._placed_ids:
                continue
            if self._held_object and self._held_object["id"] == obj_id:
                continue

            try:
                pos = obj_info["node"].getField("translation").getSFVec3f()
                dx = robot_pos[0] - pos[0]
                dy = robot_pos[1] - pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = obj_id
            except Exception:
                pass

        return best_id

    def find_object_in_region(
        self, region: str,
    ) -> Optional[str]:
        """Return the ID of an available object currently in *region*."""
        for obj_info in self._object_nodes:
            obj_id = obj_info["id"]
            if obj_id in self._placed_ids:
                continue
            if self._held_object and self._held_object["id"] == obj_id:
                continue
            if obj_info["table"] == region:
                return obj_id
        return None

    def find_object_by_type(
        self, object_type: str,
    ) -> Optional[str]:
        """Return the ID of an available object of the given type."""
        for obj_info in self._object_nodes:
            obj_id = obj_info["id"]
            if obj_info["type"] != object_type:
                continue
            if obj_id in self._placed_ids:
                continue
            if self._held_object and self._held_object["id"] == obj_id:
                continue
            return obj_id
        return None

    def reset_placed(self) -> None:
        """Clear the placed-id set so objects can be grasped again.

        Useful when respawning stock for an endless demo loop.
        """
        self._placed_ids.clear()

    def reset_all_objects(self) -> None:
        """Teleport every tracked object back to its initial position.

        This respawns stock / resets the scene for the next demo loop.
        """
        self._placed_ids.clear()
        self._held_object = None
        for obj_info in self._object_nodes:
            try:
                node = obj_info["node"]
                node.getField("translation").setSFVec3f(list(obj_info["initial_pos"]))
            except Exception:
                pass
