"""Repertoire learning types for script acquisition and consolidation.

Active inference framing:
- ScriptPrimitive: atomic generative model (single-step, known A/B matrices)
- ScriptPattern: learned multi-step generative model (variable precision)
- ScriptTrajectory: observed data against which patterns are evaluated
- Precision accumulation: weak scripts strengthen via free energy minimisation

Theoretical sources:
- Albarracin, Constant, Friston & Ramstead (2021) — A Variational Approach to Scripts
- Pattisapu, Verbelen, Pitliya, Kiefer & Albarracin (2024) — Circumplex affect
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from architecture_core.core.types import AffectState, DeonticMode, SkillRequest
from architecture_core.core.status import Status


# =====================================================================
# Script Primitives -- atomic social action units
# =====================================================================
@dataclass
class ScriptPrimitive:
    """Atomic social action unit -- the smallest composable building block.

    Each primitive maps to a SkillRequest template plus social metadata:
    what situation it applies to, what affect it expects, and how it
    transitions (preconditions and postconditions in situation space).

    Active inference: a primitive is a single-step generative model
    with known A-matrix (expected observations) and B-matrix (expected
    transition to the next situation).
    """
    name: str
    skill_template: SkillRequest
    precondition_situations: List[str] = field(default_factory=list)
    postcondition_situation: str = ""
    expected_affect: Optional[AffectState] = None
    typical_duration_s: float = 5.0
    deontic_default: DeonticMode = "permitted"


# =====================================================================
# Trajectory Recording -- what actually happened
# =====================================================================
@dataclass
class TrajectoryStep:
    """One recorded step from actual execution.

    Captures the full observation-action-outcome tuple needed for
    free energy computation.
    """
    t: float
    primitive_name: str
    situation_belief: Dict[str, float] = field(default_factory=dict)
    most_likely_situation: str = ""
    action_skill: str = ""
    outcome: Status = "RUNNING"
    affect_before: AffectState = field(default_factory=AffectState)
    affect_after: AffectState = field(default_factory=AffectState)
    violation_kl: float = 0.0
    human_intent: str = "neutral"
    norm_snapshot: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScriptTrajectory:
    """Complete recorded execution of a script (or composed sequence).

    The trajectory is the observation against which generative models
    (ScriptPatterns) are evaluated.
    """
    script_name: str
    situation_type: str = ""
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_duration_s: float = 0.0
    final_outcome: Status = "RUNNING"
    mean_violation_kl: float = 0.0
    total_prediction_error: float = 0.0

    @property
    def success_rate(self) -> float:
        if not self.steps:
            return 0.0
        successes = sum(1 for s in self.steps if s.outcome == "SUCCESS")
        return successes / len(self.steps)

    @property
    def step_count(self) -> int:
        return len(self.steps)


# =====================================================================
# Transition model -- B-matrix entries
# =====================================================================
@dataclass
class TransitionEntry:
    """B-matrix entry: observed transition from one primitive to another.

    count: times this specific transition was observed
    total_from: total transitions from the source primitive (for normalisation)
    mean_kl: average prediction error during this transition
    """
    count: int = 0
    total_from: int = 0
    mean_kl: float = 0.0

    @property
    def probability(self) -> float:
        if self.total_from == 0:
            return 0.0
        return self.count / self.total_from


# =====================================================================
# Script Pattern -- a learned generative model of step sequences
# =====================================================================
@dataclass
class ScriptPattern:
    """A learned behavioral pattern -- trajectory-level generative model.

    Active inference (Albarracin et al. 2021):
    - Weak scripts are loosely-jointed semantic clusters: unordered sets of
      concepts whose internal topology reshapes depending on context.
    - Strong scripts are pragmatically sequenced: the same concepts tied
      into a specific temporal ordering through experience.
    - The transition from weak to strong is a structural transformation
      mediated by B-matrix crystallization, not just a precision increase.

    Fields:
    - primitive_cluster: the semantic content (unordered set of concepts)
    - context_topology: per-context directed pairwise connection weights
      (proto-B-matrix for weak scripts)
    - primitives_sequence: crystallized ordering (populated on consolidation
      for cluster-based patterns, or directly for legacy patterns)
    - transition_counts: empirical B-matrix (built from observed trajectories)
    - situation_affinity: D-matrix prior over which situations this applies to
    - precision: confidence / inverse temperature
    """
    name: str
    primitives_sequence: List[str] = field(default_factory=list)
    precision: float = 0.1
    trajectory_count: int = 0
    mean_free_energy: float = 10.0
    situation_affinity: Dict[str, float] = field(default_factory=dict)
    transition_counts: Dict[str, Dict[str, TransitionEntry]] = field(
        default_factory=dict
    )
    total_success_rate: float = 0.0
    last_used_t: float = 0.0
    is_strong: bool = False

    # Semantic cluster representation (Albarracin et al. 2021)
    primitive_cluster: Set[str] = field(default_factory=set)
    context_topology: Dict[str, Dict[Tuple[str, str], float]] = field(
        default_factory=dict
    )

    # Discovered norm parameters for this pattern's context
    norm_features: Dict[str, float] = field(default_factory=dict)

    # Compositional memory fields
    source_fragments: List[str] = field(default_factory=list)
    primitive_weights: Dict[str, float] = field(default_factory=dict)
    composition_count: int = 0
    composition_signature: str = ""

    @property
    def is_composite(self) -> bool:
        """True if this pattern was assembled from multiple fragments."""
        return len(self.source_fragments) > 1

    @property
    def is_cluster_based(self) -> bool:
        """True if this pattern uses the semantic cluster representation."""
        return len(self.primitive_cluster) > 0

    @property
    def reliability(self) -> float:
        """Sigmoid-gated confidence, analogous to IntentParticleFilter.reliability."""
        z = (self.precision - 1.0) / 0.5
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)


# =====================================================================
# Weighted pattern -- fragment with retrieval weight
# =====================================================================
@dataclass
class WeightedPattern:
    """A pattern with an associated retrieval weight."""
    pattern: ScriptPattern
    weight: float


# =====================================================================
# Repertoire configuration
# =====================================================================
@dataclass
class RepertoireConfig:
    """Configuration for the script learning system."""
    # Consolidation thresholds
    strong_precision_threshold: float = 2.0
    weak_precision_floor: float = 0.05
    min_trajectories_for_promotion: int = 5
    max_free_energy_for_promotion: float = 3.0

    # Precision dynamics
    precision_gain_on_success: float = 0.3
    precision_loss_on_violation: float = 0.2
    precision_decay_rate: float = 0.01

    # Composition
    composition_efe_weight_efficiency: float = 0.4
    composition_efe_weight_empathy: float = 0.3
    composition_efe_weight_epistemic: float = 0.3
    max_composition_length: int = 6

    # Trajectory inference
    trajectory_match_threshold: float = 0.6
    max_stored_trajectories: int = 100

    # Norm learning
    norm_learning_rate: float = 0.3   # base EMA alpha for norm features
    norm_min_learning_rate: float = 0.02  # floor for high-precision patterns

    # Particle filter
    n_particles: int = 50
    resample_threshold: float = 0.5

    # Compositional retrieval
    topk_recognition: int = 4
    graph_diffusion_alpha: float = 0.25
    graph_diffusion_steps: int = 2
    composition_confidence_threshold: float = 0.7
    composition_consolidation_count: int = 3
    graph_edge_decay: float = 0.99
