"""architecture_core/core/blackboard.py
TODOs:
- Store latest PerceptBundle.
- Store current active SkillRequest, active constraints, and any script state.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from architecture_core.core.types import (
    AffectState,
    NormativeConstraints,
    PerceptBundle,
    ScriptViolation,
    SkillRequest,
)

@dataclass
class Blackboard:
    percept: Optional[PerceptBundle] = None
    active_req: Optional[SkillRequest] = None
    norms: NormativeConstraints = field(default_factory=NormativeConstraints)
    self_affect: AffectState = field(default_factory=AffectState)
    latest_violation: Optional[ScriptViolation] = None
    deontic_context: Dict[str, Any] = field(default_factory=dict)
    active_pattern: Optional[str] = None
    trajectory_free_energy: float = 0.0
    norm_snapshot: Dict[str, float] = field(default_factory=dict)
    active_norm_features: Dict[str, float] = field(default_factory=dict)
    collision_prediction: Optional[Any] = None
    last_social_efe: float = 0.0        # G_social from last EFE computation
    last_policy_entropy: float = 0.0    # H[q(π)] from last EFE computation
    last_num_policies: int = 5          # |Π| for entropy normalization
