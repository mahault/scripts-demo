"""main.py

Wire perception -> blackboard -> executive for headless testing.
Plugins register their skills; this file shows the wiring pattern.
"""

from __future__ import annotations

from architecture_core.core.blackboard import Blackboard
from architecture_core.core.registry import SkillRegistry
from architecture_core.core.executive import Executive
from architecture_core.cognition.norms.norm_engine import NormEngine
from architecture_core.cognition.norms.rules import PersonalSpaceRule, SpeedLimitRule
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.tom.tom_modulator import ToMModulator
from architecture_core.safety.shield import SafetyShield
from architecture_core.perception.perception_pipeline import PerceptionPipeline
from architecture_core.perception.augmentations.proxemics import ProxemicsAugmentation
from architecture_core.perception.sensors.base_sensors import SensorInterface


class MockSensors(SensorInterface):
    """Minimal sensor for headless testing (no robot)."""

    def read(self):
        return {"robot_pose": (0, 0, 0, 0), "agents": [], "timestamp": 0.0}


def main():
    bb = Blackboard()
    registry = SkillRegistry()

    perception = PerceptionPipeline(
        sensors=MockSensors(),
        augmentations=[ProxemicsAugmentation()],
    )

    norms = NormEngine(rules=[
        PersonalSpaceRule(min_distance=0.5, veto_distance=0.3),
        SpeedLimitRule(max_speed=0.5),
    ])

    # ToMModulator without an adapter returns neutral.
    # To enable ToM, pass a ToMPlannerAdapter:
    #   from architecture_core.cognition.tom.models.tom_planner_adapter import ToMPlannerAdapter
    #   adapter = ToMPlannerAdapter(agent_id=0, goal_x=1.0, goal_y=0.0,
    #                               alpha=0.5, planner_path="...")
    #   tom = ToMModulator(adapter=adapter)
    tom = ToMModulator()

    executive = Executive(
        bb=bb,
        registry=registry,
        scripts=ScriptManager(),
        norms=norms,
        tom=tom,
        shield=SafetyShield(),
    )

    # TODO: register plugin skills before entering loop
    # from plugins.tiago_webots.plugin import register
    # register(registry, driver)

    t = 0.0
    dt = 0.1
    try:
        while True:
            bb.percept = perception.tick(t)
            executive.tick(t)
            t += dt
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
