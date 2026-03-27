"""Affect-modulated norm rules.

Two rules that couple the robot's self-affect (from EmpathicModulator)
to the norm engine:

- AffectModulatedSpeedRule — reduces speed cap when valence is negative
- DistressVetoRule — hard veto when both valence and arousal cross
  extreme thresholds (companion-mode gate from mindsphere-coach)
"""

from __future__ import annotations

from typing import Optional

from architecture_core.core.types import (
    AffectState,
    NormativeConstraints,
    PerceptBundle,
    SkillRequest,
)
from architecture_core.cognition.norms.rules import NormRule


class AffectModulatedSpeedRule(NormRule):
    """Reduce speed cap proportionally to negative self-affect valence.

    When ``valence < valence_threshold``, the speed cap scales linearly
    from ``base_max_speed`` down to ``min_speed_fraction * base_max_speed``
    at valence = -1.0.
    """

    name = "affect_speed"

    def __init__(
        self,
        base_max_speed: float = 0.5,
        valence_threshold: float = -0.3,
        min_speed_fraction: float = 0.3,
    ) -> None:
        self.base_max_speed = base_max_speed
        self.valence_threshold = valence_threshold
        self.min_speed_fraction = min_speed_fraction
        self._affect: AffectState = AffectState()

    def set_affect(self, affect: AffectState) -> None:
        self._affect = affect

    def evaluate(
        self, pb: PerceptBundle, candidate: SkillRequest
    ) -> NormativeConstraints:
        constraints = NormativeConstraints()

        if self._affect.valence >= self.valence_threshold:
            # No modulation needed
            constraints.hard["max_linear_speed"] = self.base_max_speed
            return constraints

        # Linear interpolation: threshold → base, -1.0 → min
        depth = (self.valence_threshold - self._affect.valence) / (
            abs(self.valence_threshold) + 1.0
        )
        factor = max(self.min_speed_fraction, 1.0 - (1.0 - self.min_speed_fraction) * depth)
        constraints.hard["max_linear_speed"] = round(
            self.base_max_speed * factor, 3
        )
        return constraints


class DistressVetoRule(NormRule):
    """Hard veto when robot is in extreme distress.

    Vetoes ALL actions when BOTH:
        valence < valence_threshold AND arousal > arousal_threshold

    This is the companion-mode hard gate from mindsphere-coach:
    when the robot is too distressed, it should stop and de-escalate
    rather than continue the current action.
    """

    name = "distress_veto"

    def __init__(
        self,
        valence_threshold: float = -0.8,
        arousal_threshold: float = 0.8,
    ) -> None:
        self.valence_threshold = valence_threshold
        self.arousal_threshold = arousal_threshold
        self._affect: AffectState = AffectState()

    def set_affect(self, affect: AffectState) -> None:
        self._affect = affect

    def evaluate(
        self, pb: PerceptBundle, candidate: SkillRequest
    ) -> NormativeConstraints:
        constraints = NormativeConstraints()

        if (
            self._affect.valence < self.valence_threshold
            and self._affect.arousal > self.arousal_threshold
        ):
            constraints.veto = (
                f"Distress veto: valence={self._affect.valence:.2f} "
                f"arousal={self._affect.arousal:.2f}"
            )

        return constraints
