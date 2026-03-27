"""Predictive collision norm rule.

Evaluates trajectory-based collision predictions and produces
graduated speed constraints based on time-to-collision.

Runs at deliberation rate (2 Hz default). Handles:
- Robot-agent TTC: speed reduction proportional to TTC
- Robot-obstacle TTC: speed reduction + keepout zones
- Agent-agent TTC near robot: yield/wait to stay clear
"""

from __future__ import annotations

from typing import List, Optional

from architecture_core.cognition.norms.rules import NormRule
from architecture_core.core.types import (
    NormativeConstraints,
    PerceptBundle,
    SkillRequest,
)
from architecture_core.safety.geometry import point_distance
from architecture_core.safety.velocity_predictor import (
    CollisionPrediction,
    VelocityPredictor,
    speed_reduction_factor,
)


class PredictiveCollisionRule(NormRule):
    """Graduated speed constraints from predicted collisions.

    Parameters:
        predictor: VelocityPredictor instance
        ttc_stop: TTC below which speed factor = 0 (full stop)
        ttc_caution: TTC above which no speed reduction
        veto_ttc: TTC below which a veto is issued
        agent_collision_proximity: how close robot must be to an
            agent-agent collision point to trigger yield (metres)
    """

    name = "predictive_collision"

    def __init__(
        self,
        predictor: Optional[VelocityPredictor] = None,
        ttc_stop: float = 0.5,
        ttc_caution: float = 3.0,
        veto_ttc: float = 0.3,
        agent_collision_proximity: float = 2.0,
    ) -> None:
        self.predictor = predictor or VelocityPredictor()
        self.ttc_stop = ttc_stop
        self.ttc_caution = ttc_caution
        self.veto_ttc = veto_ttc
        self.agent_collision_proximity = agent_collision_proximity
        self._last_prediction: Optional[CollisionPrediction] = None

    @property
    def last_prediction(self) -> Optional[CollisionPrediction]:
        return self._last_prediction

    def evaluate(
        self, pb: PerceptBundle, candidate: SkillRequest,
    ) -> NormativeConstraints:
        constraints = NormativeConstraints()

        prediction = self.predictor.predict(pb)
        self._last_prediction = prediction

        # --- Robot collision risks ---
        if prediction.min_robot_ttc < float("inf"):
            factor = speed_reduction_factor(
                prediction.min_robot_ttc,
                self.ttc_stop,
                self.ttc_caution,
            )
            constraints.hard["speed_cap"] = factor

            constraints.hard["predictive_collision"] = {
                "min_robot_ttc": prediction.min_robot_ttc,
                "min_agent_agent_ttc": prediction.min_agent_agent_ttc,
                "num_risks": len(prediction.risks),
                "robot_velocity": prediction.robot_velocity,
            }

            if prediction.min_robot_ttc < self.veto_ttc:
                constraints.veto = (
                    f"Imminent collision: TTC={prediction.min_robot_ttc:.2f}s"
                )

        # --- Agent-agent collision risks near robot ---
        robot_pose = pb.world.get("robot_pose", (0, 0, 0, 0))
        rx, ry = robot_pose[0], robot_pose[1]
        yield_agents: List[str] = []

        for risk in prediction.risks:
            if risk.risk_type == "agent_agent":
                cp_dist = point_distance(
                    rx, ry, risk.collision_point[0], risk.collision_point[1],
                )
                if cp_dist < self.agent_collision_proximity:
                    yield_agents.extend([risk.entity_a, risk.entity_b])

        if yield_agents:
            constraints.soft["yield_agents"] = list(set(yield_agents))
            aa_factor = speed_reduction_factor(
                prediction.min_agent_agent_ttc,
                self.ttc_stop * 1.5,
                self.ttc_caution,
            )
            existing_cap = constraints.hard.get("speed_cap", 1.0)
            constraints.hard["speed_cap"] = min(existing_cap, aa_factor)

        return constraints
