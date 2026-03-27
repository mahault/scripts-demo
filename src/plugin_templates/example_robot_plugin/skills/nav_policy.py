"""plugin_templates/example_robot_plugin/skills/nav_policy.py
TODOs:
- Example IntentPolicy implementation (forced approach/avoid/yield/wait handling).
"""

from architecture_core.cognition.tom.intent_policy import IntentPolicy
from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate

class ExampleNavIntentPolicy(IntentPolicy):
    def approach(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate:
        base.params.setdefault("speed_scale", 1.0)
        base.constraints.setdefault("stop_distance", 0.8)
        return base

    def avoid(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate:
        base.params.setdefault("speed_scale", 0.6)
        # TODO: add keepout zones derived from pb.social/proxemics
        return base

    def yield_(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate:
        base.params.setdefault("speed_scale", 0.4)
        base.constraints.setdefault("stop_distance", 1.2)
        return base

    def wait(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate:
        base.params["speed_scale"] = 0.0
        return base
