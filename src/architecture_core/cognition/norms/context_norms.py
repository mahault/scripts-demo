"""Task-context-dependent norms.

Different situations activate different norm rule sets. For example,
a manufacturing floor context might have stricter speed limits than
an open social area.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from architecture_core.cognition.norms.rules import NormRule
from architecture_core.core.types import NormativeConstraints, PerceptBundle, SkillRequest


@dataclass
class TaskContext:
    """A named context with its own norm rules."""
    name: str
    situations: List[str] = field(default_factory=list)
    rules: List[NormRule] = field(default_factory=list)


class ContextualNormRule(NormRule):
    """Activates different norm sets based on recognized situation."""

    name = "contextual_norms"

    def __init__(
        self,
        contexts: List[TaskContext],
        default_context: Optional[str] = None,
    ) -> None:
        self._contexts = contexts
        self._by_name: Dict[str, TaskContext] = {c.name: c for c in contexts}
        self._active: Optional[str] = default_context

    def set_situation(self, situation: str) -> None:
        """Called by Executive when situation is recognized."""
        for ctx in self._contexts:
            if situation in ctx.situations:
                self._active = ctx.name
                return

    @property
    def active_context(self) -> Optional[str]:
        return self._active

    def evaluate(
        self, pb: PerceptBundle, candidate: SkillRequest
    ) -> NormativeConstraints:
        if self._active is None or self._active not in self._by_name:
            return NormativeConstraints()

        ctx = self._by_name[self._active]
        merged = NormativeConstraints()

        for rule in ctx.rules:
            result = rule.evaluate(pb, candidate)

            # First veto wins
            if result.veto is not None and merged.veto is None:
                merged.veto = result.veto

            # Merge hard constraints (same logic as NormEngine)
            for key, value in result.hard.items():
                if key in merged.hard:
                    existing = merged.hard[key]
                    if isinstance(value, (int, float)) and isinstance(
                        existing, (int, float)
                    ):
                        if key.startswith("max"):
                            merged.hard[key] = min(existing, value)
                        elif key.startswith("min"):
                            merged.hard[key] = max(existing, value)
                        else:
                            merged.hard[key] = value
                    else:
                        merged.hard[key] = value
                else:
                    merged.hard[key] = value

            # Merge soft constraints
            merged.soft.update(result.soft)

        return merged
