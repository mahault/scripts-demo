"""architecture_core/cognition/norms/norm_engine.py

Evaluate candidate SkillRequests against a set of norm rules.
Merges constraints (most restrictive wins) and returns the first veto.
"""

from __future__ import annotations

from typing import List, Optional

from architecture_core.cognition.norms.rules import NormRule
from architecture_core.core.types import (
    NormativeConstraints,
    PerceptBundle,
    SkillRequest,
)


class NormEngine:
    def __init__(self, rules: Optional[List[NormRule]] = None) -> None:
        self.rules = rules or []

    def evaluate(
        self, pb: PerceptBundle, candidate: SkillRequest
    ) -> NormativeConstraints:
        merged = NormativeConstraints()

        for rule in self.rules:
            result = rule.evaluate(pb, candidate)

            # First veto wins
            if result.veto is not None and merged.veto is None:
                merged.veto = result.veto

            # Merge hard constraints
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
