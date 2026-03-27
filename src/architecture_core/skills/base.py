"""architecture_core/skills/base.py
TODOs:
- Public skill interface plugins implement.
- Skills should not know about Scripts/Norms/ToM directly; they only receive PerceptBundle + SkillUpdate.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from architecture_core.core.types import SkillRequest, SkillUpdate, PerceptBundle
from architecture_core.core.status import Status

class Skill(ABC):
    name: str

    @abstractmethod
    def start(self, req: SkillRequest) -> None: ...

    @abstractmethod
    def tick(self, pb: PerceptBundle, update: SkillUpdate) -> Status: ...

    @abstractmethod
    def stop(self, reason: str = "") -> None: ...
