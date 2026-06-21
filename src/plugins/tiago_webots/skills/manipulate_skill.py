"""Minimal TIAGo manipulate skill for replaying learned restock primitives.

The learner discovers ``manipulate`` primitives (``extend_arm`` / ``retract_arm``
/ ``look_around``) by watching the worker.  When it replays the crystallised
script it needs a registered skill of that name; this provides a lightweight one
that drives the arm/head gesture for a short dwell and reports ``SUCCESS`` — so
the learned sequence executes without the worker's Supervisor-teleport
manipulation machinery (which the learner does not own).
"""

from __future__ import annotations

from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate
from architecture_core.core.status import Status
from architecture_core.skills.base import Skill
from architecture_core.cognition.tom.intent_policy import IntentPolicy

from plugins.tiago_webots.robot.driver import TiagoDriver


class TiagoManipulateSkill(Skill):
    """Replay a manipulation gesture (arm extend/retract or a head glance)."""

    name = "manipulate"
    DWELL_TICKS = 40

    def __init__(self, driver: TiagoDriver) -> None:
        self.driver = driver
        self._action = "extend_arm"
        self._remaining = 0

    def start(self, req: SkillRequest) -> None:
        self._action = req.params.get("action", "extend_arm")
        self._remaining = self.DWELL_TICKS
        self.driver.stop()
        if self._action == "extend_arm":
            self.driver.open_gripper()
            self.driver.extend_arm()
        elif self._action == "retract_arm":
            self.driver.retract_arm()
            self.driver.close_gripper()
        elif self._action == "look_around":
            self.driver.set_head_pan_tilt(0.6, 0.0)

    def tick(self, pb: PerceptBundle, update: SkillUpdate) -> Status:
        self.driver.stop()
        self._remaining -= 1
        if self._remaining <= 0:
            # Tidy up before finishing: retract arm and recentre the head.
            self.driver.retract_arm()
            self.driver.set_head_pan_tilt(0.0, 0.0)
            return "SUCCESS"
        return "RUNNING"

    def stop(self, reason: str = "") -> None:
        self.driver.stop()


class ManipulateIntentPolicy(IntentPolicy):
    """Manipulation is not socially modulated — pass the base update through."""

    def approach(self, pb, req, base):  # noqa: D102
        return base

    def avoid(self, pb, req, base):  # noqa: D102
        return base

    def yield_(self, pb, req, base):  # noqa: D102
        return base

    def wait(self, pb, req, base):  # noqa: D102
        return base
