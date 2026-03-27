"""TIAGo-specific pick-and-place skill using arm + gripper driver.

When object_sensors is provided, uses Supervisor-based teleportation
for grasping/releasing objects (standard for Webots demos without IK).
"""

from __future__ import annotations

from typing import Optional

from architecture_core.skills.pick_place_skill import PickPlaceSkill

from plugins.tiago_webots.robot.driver import TiagoDriver


class TiagoPickPlaceSkill(PickPlaceSkill):
    """Pick/place with TIAGo arm and gripper hardware."""

    def __init__(
        self,
        driver: TiagoDriver,
        object_sensors=None,
        drop_off_pos: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.driver = driver
        self._object_sensors = object_sensors
        self._drop_off_pos = drop_off_pos or (0.0, 0.0, 0.74)

    def start(self, req) -> None:
        super().start(req)
        # Set manipulation target for target-specific arm proximity
        if self._object_sensors is not None and self._target_object:
            self._object_sensors.set_manipulation_target(self._target_object)

    def _on_arm_move(self, target: object) -> None:
        self.driver.extend_arm()

    def _on_grasp(self) -> None:
        self.driver.close_gripper()
        # Supervisor-based grasping: teleport object to gripper
        if self._object_sensors is not None and self._target_object:
            self._object_sensors.supervisor_grasp(self._target_object)

    def _on_release(self) -> None:
        self.driver.open_gripper()
        # Supervisor-based release: teleport object to drop-off
        if self._object_sensors is not None:
            self._object_sensors.supervisor_release(self._drop_off_pos)

    def _on_retract(self) -> None:
        self.driver.retract_arm()

    def stop(self, reason: str = "") -> None:
        # Clear manipulation target before retract
        if self._object_sensors is not None:
            self._object_sensors.set_manipulation_target(None)
        self.driver.retract_arm()
        super().stop(reason)
