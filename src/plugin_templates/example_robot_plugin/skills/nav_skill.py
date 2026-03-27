"""plugin_templates/example_robot_plugin/skills/nav_skill.py
TODOs:
- Wrap a robot controller here (Webots/ROS2/etc.).
- Implement start/tick/stop.
"""

from architecture_core.skills.base import Skill
from architecture_core.core.types import SkillRequest, SkillUpdate, PerceptBundle
from architecture_core.core.status import Status

class ExampleNavSkill(Skill):
    name = "navigate"

    def __init__(self) -> None:
        self._goal = None
        self._started = False

    def start(self, req: SkillRequest) -> None:
        self._goal = req.goal
        self._started = True

    def tick(self, pb: PerceptBundle, update: SkillUpdate) -> Status:
        if not self._started:
            # framework should call start; this is defensive
            return "FAILURE"

        # TODO: interpret update.params/update.constraints to drive robot
        # e.g. speed_scale, stop_distance, keepout_zones, etc.
        return "RUNNING"

    def stop(self, reason: str = "") -> None:
        # TODO: stop robot motion
        self._started = False
