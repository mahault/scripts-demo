"""architecture_core/core/registry.py
TODOs:
- Skill registry stores (Skill, IntentPolicy) pairs.
- Registration should fail loudly if IntentPolicy missing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from architecture_core.skills.base import Skill
from architecture_core.cognition.tom.intent_policy import IntentPolicy

@dataclass
class SkillEntry:
    skill: Skill
    policy: IntentPolicy

class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, SkillEntry] = {}

    def register(self, entry: SkillEntry) -> None:
        name = entry.skill.name
        if not name:
            raise ValueError("Skill must have a non-empty name")
        # Hard enforcement:
        if entry.policy is None:
            raise TypeError(f"Skill '{name}' must provide an IntentPolicy")
        self._skills[name] = entry

    def get(self, name: str) -> SkillEntry:
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not registered")
        return self._skills[name]
