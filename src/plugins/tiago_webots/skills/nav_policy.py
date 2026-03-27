"""IntentPolicy for TIAGo navigation.

Translates the four abstract intents (approach / avoid / yield / wait)
into concrete speed-scale and stop-distance parameters that the
TiagoNavSkill understands.

Yield uses speed modulation only (no target override).  The driver's
reactive obstacle avoidance handles lateral dodging naturally.  After a
yield timeout (~3s at 5Hz) speed increases to break symmetric deadlocks.
"""

from __future__ import annotations

from architecture_core.cognition.tom.intent_policy import IntentPolicy
from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate

_YIELD_TIMEOUT_TICKS = 15  # ~3s at 5Hz deliberation


class TiagoNavIntentPolicy(IntentPolicy):

    def __init__(self) -> None:
        self._yield_ticks: int = 0

    def approach(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate:
        self._yield_ticks = 0
        base.params["speed_scale"] = 1.0
        base.params["stop_distance"] = 0.5
        return base

    def avoid(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate:
        self._yield_ticks = 0
        base.params["speed_scale"] = 0.5
        base.params["stop_distance"] = 1.2
        prox = pb.social.get("proxemics", {})
        readings = prox.get("readings", [])
        if readings:
            base.constraints["avoid_entity"] = getattr(readings[0], "entity_id", "unknown")
        return base

    def yield_(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate:
        self._yield_ticks += 1

        if self._yield_ticks > _YIELD_TIMEOUT_TICKS:
            # Timeout: increase speed to break symmetric deadlock
            base.params["speed_scale"] = 0.5
        else:
            # Normal yield: slow sidestep away from the other agent
            base.params["speed_scale"] = 0.3

        # Yield means "move aside" — stop_distance must be collision-only
        # so the sidestep waypoint actually produces displacement.
        base.params["stop_distance"] = 0.3

        return base

    def wait(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate:
        self._yield_ticks = 0
        base.params["speed_scale"] = 0.0
        base.params["stop_distance"] = 0.0
        return base
