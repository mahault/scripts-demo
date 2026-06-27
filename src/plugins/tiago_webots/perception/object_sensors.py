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

# Solid nodes with these prefixes act as containers: items dropped inside are
# tracked with an offset and teleported to follow the container each tick.
CONTAINER_PREFIXES = ("BASKET_", "CONTAINER_", "TRAY_", "CART_")

# Default capacity for discovered containers
DEFAULT_CONTAINER_CAPACITY = 6

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

# Height of the TIAGo gripper when the arm is extended (used for held objects)
GRIPPER_Z = 1.05

# Objects above this z are considered to be held / carried.
_HELD_Z_THRESHOLD = 1.0

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

    # Re-export module-level threshold so instance methods can reference it.
    _HELD_Z_THRESHOLD = _HELD_Z_THRESHOLD

    def __init__(self, robot, self_node, name: str,
                 observe_agent_internals: bool = True) -> None:
        super().__init__(robot, self_node, name,
                         observe_agent_internals=observe_agent_internals)
        self._object_nodes: List[Dict[str, Any]] = []
        self._held_object: Optional[Dict[str, Any]] = None
        # Smooth (non-teleport) manipulation: items glide to the hand on grasp
        # and to their destination on release over a fraction of a second.
        self._attach: Optional[Dict[str, Any]] = None    # in-progress grasp
        self._transits: List[Dict[str, Any]] = []         # in-progress releases
        self._transiting_ids: set = set()
        self._dt = robot.getBasicTimeStep() / 1000.0
        self._placed_ids: set = set()
        self._furniture_data: List[Dict[str, Any]] = []
        self._manipulation_target_id: Optional[str] = None
        # container_id -> {"node": ..., "capacity": int, "items": [{id, offset}]}
        self._containers: Dict[str, Dict[str, Any]] = {}
        # Container node names so they are not mistaken for contained items
        # during startup discovery.
        self._container_names: set = set()
        self._discover_objects()
        self._discover_containers()
        self._initialize_container_contents()
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
        data["containers"] = self._container_state()
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
                "name": obj_name,
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

    @staticmethod
    def _is_container(name: str) -> bool:
        """Return True if a Solid name identifies it as a container."""
        return any(name.startswith(p) for p in CONTAINER_PREFIXES)

    def _discover_containers(self) -> None:
        """Register dedicated container nodes (BASKET_, CONTAINER_, etc.).

        Containers are discovered separately from manipulable objects so that
        trays and baskets are not mistaken for graspable grocery items.
        """
        root = self.robot.getRoot()
        children = root.getField("children")
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            try:
                type_name = node.getTypeName()
            except Exception:
                continue
            if type_name != "Solid":
                continue
            name_field = node.getField("name")
            if not name_field:
                continue
            obj_name = name_field.getSFString()
            if self._is_container(obj_name):
                self._register_container(obj_name, node)

    def _register_container(self, container_id: str, node) -> None:
        """Track a container node and any items already inside it."""
        try:
            pos = node.getField("translation").getSFVec3f()
        except Exception:
            return
        self._container_names.add(container_id)
        self._containers[container_id] = {
            "node": node,
            "position": (pos[0], pos[1], pos[2]),
            "capacity": DEFAULT_CONTAINER_CAPACITY,
            "items": [],
        }

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
    # Containers
    # ------------------------------------------------------------------
    def _container_state(self) -> Dict[str, Any]:
        """Return a serializable snapshot of container contents."""
        return {
            cid: {
                "capacity": c["capacity"],
                "count": len(c["items"]),
                "items": [it["id"] for it in c["items"]],
            }
            for cid, c in self._containers.items()
        }

    def _container_bounds(self, container: Dict[str, Any]) -> Optional[Tuple]:
        """Read axis-aligned half-extents of a container's bounding box."""
        try:
            bo_field = container["node"].getField("boundingObject")
            bo_node = bo_field.getSFNode()
            size_field = bo_node.getField("size")
            size = size_field.getSFVec3f()
            return (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        except Exception:
            return None

    def _is_inside_container(
        self,
        position: Tuple[float, float, float],
        container: Dict[str, Any],
    ) -> bool:
        """Check if a world position is inside a container's bounding box."""
        bounds = self._container_bounds(container)
        if bounds is None:
            return False
        cx, cy, cz = container["position"]
        hx, hy, hz = bounds
        # Treat the upper half as the usable interior (items sit above center)
        return (
            cx - hx <= position[0] <= cx + hx
            and cy - hy <= position[1] <= cy + hy
            and cz <= position[2] <= cz + hz
        )

    def _initialize_container_contents(self) -> None:
        """After discovery, assign any items already inside containers to them."""
        for obj_info in self._object_nodes:
            if self._is_container(obj_info.get("name", "")):
                continue
            try:
                pos = obj_info["node"].getField("translation").getSFVec3f()
            except Exception:
                continue
            for cid, container in self._containers.items():
                if self._is_inside_container(pos, container):
                    self._add_to_container(obj_info["id"], cid)
                    break

    def _obj_position(self, obj_id: str) -> Optional[Tuple[float, float, float]]:
        """Look up an object's current world position."""
        for obj_info in self._object_nodes:
            if obj_info["id"] == obj_id:
                try:
                    pos = obj_info["node"].getField("translation").getSFVec3f()
                    return (pos[0], pos[1], pos[2])
                except Exception:
                    return None
        return None

    def _add_to_container(self, item_id: str, container_id: str) -> bool:
        """Register an item as contained and compute its local offset."""
        container = self._containers.get(container_id)
        if container is None:
            return False
        if len(container["items"]) >= container["capacity"]:
            return False
        pos = self._obj_position(item_id)
        if pos is None:
            return False
        cx, cy, cz = container["position"]
        offset = (pos[0] - cx, pos[1] - cy, pos[2] - cz)
        container["items"].append({"id": item_id, "offset": offset})
        return True

    def _remove_from_container(self, item_id: str) -> bool:
        """Remove an item from whichever container holds it."""
        for container in self._containers.values():
            before = len(container["items"])
            container["items"] = [it for it in container["items"] if it["id"] != item_id]
            if len(container["items"]) < before:
                return True
        return False

    def update_containers(self) -> None:
        """Teleport contained items so they follow their container each tick.

        If a container is currently being carried (its z is above the held
        threshold), skip it.  The robot that is holding the container will move
        its contents, so other robots do not lag one frame behind or fight
        over the item positions.
        """
        # Advance any items mid-glide to their released destination first.
        self.update_transits()

        for container in self._containers.values():
            try:
                pos = container["node"].getField("translation").getSFVec3f()
                container["position"] = (pos[0], pos[1], pos[2])
            except Exception:
                continue
            cx, cy, cz = container["position"]
            # A held container is updated by the robot that holds it.
            if cz > self._HELD_Z_THRESHOLD:
                continue
            for item in container["items"]:
                ox, oy, oz = item["offset"]
                new_pos = (cx + ox, cy + oy, cz + oz)
                if self._held_object and self._held_object["id"] == item["id"]:
                    continue
                # Don't fight an item still gliding into the container.
                if item["id"] in self._transiting_ids:
                    continue
                for obj_info in self._object_nodes:
                    if obj_info["id"] != item["id"]:
                        continue
                    try:
                        obj_info["node"].getField("translation").setSFVec3f(list(new_pos))
                        obj_info["node"].resetPhysics()
                    except Exception:
                        pass
                    break

    def find_container_in_region(self, region: str) -> Optional[str]:
        """Return the ID of a container currently in *region*."""
        for cid, container in self._containers.items():
            pos = container["position"]
            if self._classify_table(pos[0], pos[1]) == region:
                return cid
        return None

    def find_container_by_id(self, container_id: str) -> Optional[Dict[str, Any]]:
        """Return container info by ID."""
        return self._containers.get(container_id)

    def get_container_count(self, container_id: str) -> int:
        """Number of items currently inside a container."""
        container = self._containers.get(container_id)
        return len(container["items"]) if container else 0

    def _container_slot_offset(self, container: Dict[str, Any], index: int) -> Tuple[float, float]:
        """Return a deterministic (x, y) offset for the *index* item in a container.

        Lays items out in a small grid so they do not all stack at the center.
        """
        bounds = self._container_bounds(container)
        hx = bounds[0] if bounds else 0.1
        hy = bounds[1] if bounds else 0.1
        cols = 3
        col = index % cols
        row = index // cols
        x = (col - 1) * (hx * 0.8)
        y = (row - 0.5) * (hy * 0.6)
        return (x, y)

    def supervisor_release_into_container(self, container_id: str) -> bool:
        """Release the held object into a container at the next free slot."""
        if self._held_object is None:
            return False
        container = self._containers.get(container_id)
        if container is None:
            return False
        if len(container["items"]) >= container["capacity"]:
            return False

        cx, cy, cz = container["position"]
        bounds = self._container_bounds(container)
        hz = bounds[2] if bounds else 0.05
        # Drop the item just above the container's interior bottom so it sits
        # naturally (on a tray floor or in the bottom of the basket).
        z_drop = cz - hz + 0.05
        index = len(container["items"])
        sx, sy = self._container_slot_offset(container, index)
        position = (cx + sx, cy + sy, z_drop)

        item_id = self._held_object["id"]
        # Freeze the item so it does not roll or bounce out of the container.
        if not self.supervisor_release(position, mark_placed=False, freeze=True):
            return False
        self._add_to_container(item_id, container_id)
        return True

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
                node = obj_info["node"]
                try:
                    # Record where the item starts so it can glide to the hand
                    # (smooth pick-up) rather than popping there instantly.
                    start = list(node.getField("translation").getSFVec3f())
                    # If it was sitting in a container, remove it from there
                    self._remove_from_container(object_id)
                    # Unfreeze in case the item was a frozen static prop
                    self._unfreeze_object(obj_info)
                    self._held_object = obj_info
                    self._attach = {"start": start, "progress": 0.0}
                    return True
                except Exception:
                    return False

        return False

    def _freeze_object(self, obj_info: Dict[str, Any]) -> bool:
        """Make an object static so it cannot roll or jitter.

        Uses zero mass instead of removing the physics node; this keeps the
        node attached and avoids the object disappearing on reset.
        """
        try:
            node = obj_info["node"]
            physics_field = node.getField("physics")
            if physics_field is None:
                return False
            physics_node = physics_field.getSFNode()
            if physics_node is None:
                return False
            mass_field = physics_node.getField("mass")
            if mass_field is None:
                return False
            # Remember the original mass so unfreeze can restore it cleanly.
            if "mass" not in obj_info:
                obj_info["mass"] = mass_field.getSFFloat()
            mass_field.setSFFloat(0.0)
            return True
        except Exception:
            return False

    def _unfreeze_object(self, obj_info: Dict[str, Any]) -> bool:
        """Restore an object's physics so it can be moved/carried again."""
        try:
            node = obj_info["node"]
            physics_field = node.getField("physics")
            if physics_field is None:
                return False
            physics_node = physics_field.getSFNode()
            if physics_node is None:
                return False
            mass_field = physics_node.getField("mass")
            if mass_field is None:
                return False
            original_mass = obj_info.get("mass")
            if original_mass is not None:
                mass_field.setSFFloat(original_mass)
            else:
                mass_field.setSFFloat(0.1)
            return True
        except Exception:
            return False

    def supervisor_release(
        self,
        position: Tuple[float, float, float],
        mark_placed: bool = True,
        freeze: bool = False,
    ) -> bool:
        """Teleport held object to the specified position.

        Returns True if successful, False if not holding anything.
        """
        if self._held_object is None:
            return False

        obj = self._held_object
        node = obj["node"]
        try:
            start = list(node.getField("translation").getSFVec3f())
            # Glide the item to its destination over RELEASE_TIME instead of
            # popping it there.  Freeze (if any) is applied on arrival.
            self._transits.append({
                "node": node, "obj": obj, "start": start, "end": list(position),
                "progress": 0.0, "freeze": freeze,
            })
            self._transiting_ids.add(obj["id"])
            if mark_placed:
                self._placed_ids.add(obj["id"])
            self._held_object = None
            self._attach = None
            return True
        except Exception:
            return False

    ATTACH_TIME = 0.45   # s — glide a grasped item into the hand
    RELEASE_TIME = 0.5   # s — glide a released item to its destination

    @staticmethod
    def _ease(p: float) -> float:
        p = max(0.0, min(1.0, p))
        return p * p * (3.0 - 2.0 * p)   # smoothstep

    def update_held_position(self) -> None:
        """Keep held object at the gripper; glide it in on a fresh grasp."""
        if self._held_object is None:
            return

        robot_pos = self._get_position()
        heading = self._get_heading()
        gripper = [robot_pos[0] + 0.3 * math.cos(heading),
                   robot_pos[1] + 0.3 * math.sin(heading),
                   GRIPPER_Z]

        try:
            node = self._held_object["node"]
            trans_field = node.getField("translation")
            if self._attach is not None and self._attach["progress"] < 1.0:
                # Smoothly travel from where it was picked up into the hand.
                e = self._ease(self._attach["progress"])
                s = self._attach["start"]
                pos = [s[i] * (1.0 - e) + gripper[i] * e for i in range(3)]
                self._attach["progress"] += self._dt / self.ATTACH_TIME
                trans_field.setSFVec3f(pos)
            else:
                self._attach = None
                trans_field.setSFVec3f(gripper)
                rot_field = node.getField("rotation")
                if rot_field:
                    rot_field.setSFRotation([0, 0, 1, 0])
            node.resetPhysics()
        except Exception:
            pass

    def update_transits(self) -> None:
        """Advance in-flight released items gliding to their destination."""
        for tr in self._transits[:]:
            node = tr["node"]
            tr["progress"] += self._dt / self.RELEASE_TIME
            e = self._ease(tr["progress"])
            s, en = tr["start"], tr["end"]
            try:
                pos = [s[i] * (1.0 - e) + en[i] * e for i in range(3)]
                node.getField("translation").setSFVec3f(pos)
                if tr["progress"] >= 1.0:
                    node.getField("translation").setSFVec3f(list(en))
                    rot = node.getField("rotation")
                    if rot:
                        rot.setSFRotation([0, 0, 1, 0])
                    node.resetPhysics()
                    if tr.get("freeze"):
                        self._freeze_object(tr["obj"])
                    self._transiting_ids.discard(tr["obj"]["id"])
                    self._transits.remove(tr)
            except Exception:
                self._transiting_ids.discard(tr["obj"]["id"])
                self._transits.remove(tr)

    @property
    def held_object_id(self) -> Optional[str]:
        """ID of currently held object, or None."""
        return self._held_object["id"] if self._held_object else None

    def find_nearest_object(
        self, robot_pos: Tuple[float, float, float],
    ) -> Optional[str]:
        """Find the nearest available object to the robot.

        Placed items are skipped unless they are inside a container (i.e. still
        available for picking).
        """
        best_id = None
        best_dist = float("inf")

        for obj_info in self._object_nodes:
            obj_id = obj_info["id"]
            placed = obj_id in self._placed_ids and not self._item_in_container(obj_id)
            if placed:
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

    def find_object_in_container(
        self, container_id: str,
    ) -> Optional[str]:
        """Return the ID of an item inside a specific container."""
        container = self._containers.get(container_id)
        if not container or not container["items"]:
            return None
        return container["items"][0]["id"]

    def _item_in_container(self, obj_id: str) -> bool:
        """Return True if the item is currently inside any container."""
        for container in self._containers.values():
            if any(it["id"] == obj_id for it in container["items"]):
                return True
        return False

    def find_object_in_region(
        self, region: str,
    ) -> Optional[str]:
        """Return the ID of an available object currently in *region*."""
        for obj_info in self._object_nodes:
            obj_id = obj_info["id"]
            if obj_id in self._placed_ids and not self._item_in_container(obj_id):
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
        for container in self._containers.values():
            container["items"] = []
        for obj_info in self._object_nodes:
            try:
                node = obj_info["node"]
                node.getField("translation").setSFVec3f(list(obj_info["initial_pos"]))
                rot_field = node.getField("rotation")
                if rot_field:
                    rot_field.setSFRotation([0, 0, 1, 0])
                self._unfreeze_object(obj_info)
                node.resetPhysics()
            except Exception:
                pass
        self._initialize_container_contents()

    def reset_objects_to_initial(self, obj_ids: List[str]) -> None:
        """Teleport a specific set of objects home and rebuild container tracking."""
        ids = set(obj_ids)
        self._placed_ids -= ids
        if self._held_object and self._held_object["id"] in ids:
            self._held_object = None
        for container in self._containers.values():
            container["items"] = [it for it in container["items"] if it["id"] not in ids]
        for obj_info in self._object_nodes:
            if obj_info["id"] not in ids:
                continue
            self._unfreeze_object(obj_info)
            try:
                node = obj_info["node"]
                node.getField("translation").setSFVec3f(list(obj_info["initial_pos"]))
                rot_field = node.getField("rotation")
                if rot_field:
                    rot_field.setSFRotation([0, 0, 1, 0])
                node.resetPhysics()
            except Exception:
                pass
        self._initialize_container_contents()

    def reset_stock(self) -> None:
        """Respawn only the stock-room items, leaving customer props alone."""
        for obj_info in self._object_nodes:
            if obj_info["table"] != "stock":
                continue
            obj_id = obj_info["id"]
            if self._held_object and self._held_object["id"] == obj_id:
                continue
            # Don't steal an item that another robot is currently carrying or
            # that has already been placed into a customer tray/basket.
            if self._item_in_container(obj_id):
                continue
            try:
                pos = obj_info["node"].getField("translation").getSFVec3f()
                if pos[2] > _HELD_Z_THRESHOLD - 0.1:
                    continue
            except Exception:
                pass
            self._placed_ids.discard(obj_id)
            self._remove_from_container(obj_id)
            self._unfreeze_object(obj_info)
            try:
                node = obj_info["node"]
                node.getField("translation").setSFVec3f(list(obj_info["initial_pos"]))
                rot_field = node.getField("rotation")
                if rot_field:
                    rot_field.setSFRotation([0, 0, 1, 0])
                node.resetPhysics()
            except Exception:
                pass
