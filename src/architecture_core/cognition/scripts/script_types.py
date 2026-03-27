"""architecture_core/cognition/scripts/script_types.py

Script-sequence types used by the ScriptManager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

from architecture_core.core.types import AffectState, DeonticMode, PerceptBundle, SkillRequest
from architecture_core.core.status import Status


@dataclass
class ScriptStep:
    """One step in a script sequence.

    Variational script enrichments (Albarracin et al. 2021):
    - gate: mandatory steps are script identity invariants; flexible steps adapt
    - deontic: policy-level prior biasing for this step
    - expected_affect: C-matrix preferred affect outcome at this gate
    - expected_situation: weak-script situation type expected here
    """
    request: SkillRequest
    transition_on: Status = "SUCCESS"
    timeout_override: Optional[float] = None
    gate: Literal["mandatory", "flexible"] = "mandatory"
    deontic: DeonticMode = "permitted"
    expected_affect: Optional[AffectState] = None
    expected_situation: Optional[str] = None
    primitive_name: Optional[str] = None


@dataclass
class SituationType:
    """A-matrix conceptual cluster — event type for weak-script recognition."""
    name: str
    feature_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScriptSequence:
    """Ordered list of steps to execute."""
    name: str
    steps: List[ScriptStep] = field(default_factory=list)
    loop: bool = False


class ScriptCondition:
    """A predicate on PerceptBundle for conditional transitions."""
    def __init__(
        self,
        predicate: Callable[[PerceptBundle], bool],
        description: str = "",
    ) -> None:
        self.predicate = predicate
        self.description = description

    def evaluate(self, pb: PerceptBundle) -> bool:
        return self.predicate(pb)
