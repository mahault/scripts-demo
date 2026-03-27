"""TIAGo-specific handover skill using arm + gripper driver."""

from __future__ import annotations

from architecture_core.skills.handover_skill import HandoverSkill

from plugins.tiago_webots.robot.driver import TiagoDriver


class TiagoHandoverSkill(HandoverSkill):
    """Handover with TIAGo arm and gripper hardware."""

    def __init__(self, driver: TiagoDriver) -> None:
        super().__init__()
        self.driver = driver

    def _on_extend(self) -> None:
        self.driver.extend_arm()
        self.driver.open_gripper()

    def _on_retract(self) -> None:
        self.driver.retract_arm()

    def _on_grasp(self) -> None:
        self.driver.close_gripper()

    def _on_release(self) -> None:
        self.driver.open_gripper()

    def stop(self, reason: str = "") -> None:
        self.driver.retract_arm()
        super().stop(reason)
