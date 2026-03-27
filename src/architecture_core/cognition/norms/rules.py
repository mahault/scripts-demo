"""architecture_core/cognition/norms/rules.py

Modular norm rules evaluated by the NormEngine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from architecture_core.core.types import (
    NormativeConstraints,
    PerceptBundle,
    SkillRequest,
)


class NormRule(ABC):
    """Base class for a single evaluable norm rule."""

    name: str = "unnamed"

    @abstractmethod
    def evaluate(
        self, pb: PerceptBundle, candidate: SkillRequest
    ) -> NormativeConstraints:
        ...


class PersonalSpaceRule(NormRule):
    """Enforce minimum distance to detected agents."""

    name = "personal_space"

    def __init__(
        self,
        min_distance: float = 0.5,
        veto_distance: float = 0.3,
    ) -> None:
        self.min_distance = min_distance
        self.veto_distance = veto_distance

    @classmethod
    def from_profile(cls, profile: object) -> "PersonalSpaceRule":
        """Create from a ProxemicPrior (or any object with distance attributes)."""
        return cls(
            min_distance=profile.personal_distance,
            veto_distance=profile.intimate_distance,
        )

    def set_norm_features(self, features: Dict[str, float]) -> None:
        """Update thresholds from discovered norm context.

        Called by Executive when the active ScriptPattern has learned
        norm_features.  Extracts ``mean_inter_human_distance`` and
        derives intimate from the Hall ratio (0.375).
        """
        pd = features.get("mean_inter_human_distance")
        if pd is not None:
            self.min_distance = pd
            self.veto_distance = pd * 0.375

    def evaluate(
        self, pb: PerceptBundle, candidate: SkillRequest
    ) -> NormativeConstraints:
        constraints = NormativeConstraints()
        proxemics = pb.social.get("proxemics", {})
        closest = proxemics.get("closest_distance", float("inf"))

        if closest < self.veto_distance:
            constraints.veto = (
                f"Too close to agent ({closest:.2f}m < "
                f"{self.veto_distance}m)"
            )
        elif closest < self.min_distance:
            constraints.hard["min_agent_distance"] = self.min_distance
            constraints.hard["speed_cap"] = round(closest / self.min_distance, 2)
        return constraints


class KeepoutZoneRule(NormRule):
    """Prevent entry into restricted zones."""

    name = "keepout_zones"

    def __init__(self, zones: Optional[List[Dict[str, Any]]] = None) -> None:
        self.zones = zones or []

    def evaluate(
        self, pb: PerceptBundle, candidate: SkillRequest
    ) -> NormativeConstraints:
        constraints = NormativeConstraints()
        if self.zones:
            constraints.hard["keepout_zones"] = self.zones
        return constraints


class SpeedLimitRule(NormRule):
    """Global speed cap."""

    name = "speed_limit"

    def __init__(self, max_speed: float = 0.5) -> None:
        self.max_speed = max_speed

    def set_norm_features(self, features: Dict[str, float]) -> None:
        """Update speed limit from discovered norm context."""
        speed = features.get("max_approach_speed")
        if speed is not None:
            self.max_speed = speed

    def evaluate(
        self, pb: PerceptBundle, candidate: SkillRequest
    ) -> NormativeConstraints:
        constraints = NormativeConstraints()
        constraints.hard["max_linear_speed"] = self.max_speed
        return constraints
