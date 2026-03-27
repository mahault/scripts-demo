"""architecture_core/core/types.py
TODOs:
- These are the main public contracts (stable API).
- Avoid importing any robot-specific modules here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

Intent = Literal["approach", "avoid", "yield", "wait", "neutral"]

@dataclass
class SkillRequest:
    """Chosen by Scripts (Layer 1)."""
    skill: str
    goal: Dict[str, Any] = field(default_factory=dict)
    success: Dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 10.0
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SkillUpdate:
    """Produced by ToM + IntentPolicy (Layer 2)."""
    intent: Intent = "neutral"
    params: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    recommend_interrupt: bool = False
    debug: str = ""

@dataclass
class NormativeConstraints:
    """Produced by Norms (Layer 1)."""
    hard: Dict[str, Any] = field(default_factory=dict)  # must never violate
    soft: Dict[str, Any] = field(default_factory=dict)  # preferences
    veto: Optional[str] = None                          # reason if vetoed

@dataclass
class PerceptBundle:
    """Produced by Perception/Augmentation (Layer 3).
    Keep this robot-agnostic: poses, detections, social features, confidence.
    """
    t: float
    world: Dict[str, Any] = field(default_factory=dict)
    social: Dict[str, Any] = field(default_factory=dict)
    attention: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Variational scripts + empathic modulator types
# =====================================================================
DeonticMode = Literal["obligatory", "permitted", "forbidden"]


@dataclass
class AffectState:
    """Point on the circumplex (Pattisapu & Albarracin 2024).

    Grounded in free energy dynamics:
      valence ∝ -ΔF   (Joffily & Coricelli 2013)
      arousal ∝ H[q(π)]  (policy posterior entropy / precision control)

    Stored arousal/valence are normalized to [-1, 1] by EmpathicModulator.
    Raw values are kept for debugging and principled scaling.

    Normalization:
      arousal = 2 * (H[q(π)] / log|Π|) - 1   (stable across policy counts)
      valence = tanh(-ΔF / τ)                 (τ = scale parameter)
    """
    arousal: float = 0.0              # [-1, 1] normalized from policy_entropy
    valence: float = 0.0              # [-1, 1] normalized from -delta_free_energy
    raw_free_energy: float = 0.0      # F(t) for computing ΔF next tick
    policy_entropy: float = 0.0       # H[q(π)] in nats (raw)
    delta_free_energy: float = 0.0    # ΔF = F(t) - F(t-1) (raw)


@dataclass
class ScriptViolation:
    """Mismatch between expected and observed situation type."""
    script_name: str = ""
    step_index: int = 0
    expected_situation: str = ""
    observed_situation: str = ""
    kl_divergence: float = 0.0
    timestamp: float = 0.0
