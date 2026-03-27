"""architecture_core/perception/perception_pipeline.py

Build a PerceptBundle from sensor readings plus an augmentation chain.
Robot-specific sensors belong in plugin packages.
"""

from __future__ import annotations

from typing import Any, List, Optional

from architecture_core.core.types import PerceptBundle
from architecture_core.perception.sensors.base_sensors import SensorInterface


class PerceptionPipeline:
    def __init__(
        self,
        sensors: SensorInterface,
        augmentations: Optional[List[Any]] = None,
    ) -> None:
        self.sensors = sensors
        self.augmentations = augmentations or []

    def tick(self, t: float) -> PerceptBundle:
        raw = self.sensors.read()

        social: dict = {}
        attention: dict = {}
        uncertainty: dict = {}

        for aug in self.augmentations:
            result = aug.augment(raw)
            target = getattr(aug, "section", "social")
            if target == "social":
                social.update(result)
            elif target == "attention":
                attention.update(result)
            elif target == "uncertainty":
                uncertainty.update(result)
            else:
                social.update(result)

        return PerceptBundle(
            t=t,
            world=raw,
            social=social,
            attention=attention,
            raw=raw,
            uncertainty=uncertainty,
        )
