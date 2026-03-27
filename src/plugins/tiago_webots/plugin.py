"""TIAGo Webots plugin registration.

Registers all TIAGo skills and their intent policies with
the social-layer skill registry.
"""

from __future__ import annotations

from architecture_core.core.registry import SkillEntry, SkillRegistry
from architecture_core.skills.handover_skill import HandoverIntentPolicy
from architecture_core.skills.pick_place_skill import PickPlaceIntentPolicy
from architecture_core.skills.gaze_skill import GazeIntentPolicy

from plugins.tiago_webots.robot.driver import TiagoDriver
from plugins.tiago_webots.skills.nav_skill import TiagoNavSkill
from plugins.tiago_webots.skills.nav_policy import TiagoNavIntentPolicy
from plugins.tiago_webots.skills.handover_skill import TiagoHandoverSkill
from plugins.tiago_webots.skills.pick_place_skill import TiagoPickPlaceSkill
from plugins.tiago_webots.skills.gaze_skill import TiagoGazeSkill


def register(
    registry: SkillRegistry,
    driver: TiagoDriver,
    object_sensors=None,
    drop_off_pos: tuple = (0.0, 0.0, 0.74),
) -> None:
    """Wire all TIAGo skills + policies into *registry*."""
    # Navigation
    registry.register(SkillEntry(
        skill=TiagoNavSkill(driver), policy=TiagoNavIntentPolicy(),
    ))
    # Handover
    registry.register(SkillEntry(
        skill=TiagoHandoverSkill(driver), policy=HandoverIntentPolicy(),
    ))
    # Pick/place (with optional Supervisor grasping for Phase 10)
    registry.register(SkillEntry(
        skill=TiagoPickPlaceSkill(driver, object_sensors=object_sensors,
                                  drop_off_pos=drop_off_pos),
        policy=PickPlaceIntentPolicy(),
    ))
    # Gaze
    registry.register(SkillEntry(
        skill=TiagoGazeSkill(driver), policy=GazeIntentPolicy(),
    ))
