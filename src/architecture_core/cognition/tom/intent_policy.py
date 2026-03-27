"""architecture_core/cognition/tom/intent_policy.py
TODOs:
- This is the enforced interface you liked.
- Every plugin skill must provide an IntentPolicy (or use a default one) that handles approach/avoid/yield/wait.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate

class IntentPolicy(ABC):
    """Compile the 4 ToM intents into skill-specific params/constraints.
    Return SkillUpdate fragments (params/constraints) to be merged into the current SkillUpdate.
    """

    @abstractmethod
    def approach(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate: ...

    @abstractmethod
    def avoid(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate: ...

    @abstractmethod
    def yield_(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate: ...

    @abstractmethod
    def wait(self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate) -> SkillUpdate: ...
