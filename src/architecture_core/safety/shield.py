"""architecture_core/safety/shield.py

Enforce hard constraints on the SkillUpdate before it reaches the skill.

Guarantees:
- Nothing downstream can increase speed above the cap.
- Emergency stop when distance drops below threshold.
- Velocity-aware emergency stop when TTC drops below threshold.
- Keepout zones are passed through to the skill.
"""

from __future__ import annotations

from typing import Optional

from architecture_core.core.types import (
    NormativeConstraints,
    PerceptBundle,
    SkillRequest,
    SkillUpdate,
)
from architecture_core.safety.velocity_predictor import (
    VelocityPredictor,
    speed_reduction_factor,
)


class SafetyShield:
    def __init__(
        self,
        emergency_stop_distance: float = 0.3,
        predictor: Optional[VelocityPredictor] = None,
        emergency_ttc: float = 0.5,
    ) -> None:
        self.emergency_stop_distance = emergency_stop_distance
        self.predictor = predictor
        self.emergency_ttc = emergency_ttc

    def apply(
        self,
        pb: PerceptBundle,
        req: SkillRequest,
        update: SkillUpdate,
        norms: NormativeConstraints,
    ) -> SkillUpdate:
        proxemics = pb.social.get("proxemics", {})
        closest = proxemics.get("closest_distance", float("inf"))

        # 1. Distance-based emergency stop
        if closest < self.emergency_stop_distance:
            update.intent = "wait"
            update.params["speed_scale"] = 0.0
            update.constraints["emergency_stop"] = True
            update.debug += " [SHIELD:emergency-stop]"
            return update

        # 2. Velocity-based emergency stop
        if self.predictor is not None:
            prediction = self.predictor.predict(pb)

            if prediction.min_robot_ttc < self.emergency_ttc:
                update.intent = "wait"
                update.params["speed_scale"] = 0.0
                update.constraints["emergency_stop"] = True
                update.constraints["emergency_ttc"] = prediction.min_robot_ttc
                update.debug += (
                    f" [SHIELD:predictive-stop"
                    f" ttc={prediction.min_robot_ttc:.2f}s]"
                )
                return update

            # Preemptive braking between emergency and caution
            if prediction.min_robot_ttc < float("inf"):
                caution = self.emergency_ttc * 4.0
                ttc_factor = speed_reduction_factor(
                    prediction.min_robot_ttc,
                    self.emergency_ttc,
                    caution,
                )
                cur = update.params.get("speed_scale", 1.0)
                if ttc_factor < cur:
                    update.params["speed_scale"] = ttc_factor
                    update.debug += (
                        f" [SHIELD:predictive-brake"
                        f" factor={ttc_factor:.2f}]"
                    )

        # 3. Apply hard norm constraints
        if "max_linear_speed" in norms.hard:
            cap = norms.hard["max_linear_speed"]
            cur = update.params.get("speed_scale", 1.0)
            update.params["speed_scale"] = min(cur, cap)

        if "speed_cap" in norms.hard:
            # Skip speed cap when yielding — the robot needs speed
            # to get out of the other agent's way
            if update.intent != "yield":
                cap = norms.hard["speed_cap"]
                cur = update.params.get("speed_scale", 1.0)
                update.params["speed_scale"] = min(cur, cap)

        if "min_agent_distance" in norms.hard:
            update.constraints["stop_distance"] = norms.hard[
                "min_agent_distance"
            ]

        if "keepout_zones" in norms.hard:
            update.constraints["keepout_zones"] = norms.hard["keepout_zones"]

        return update
