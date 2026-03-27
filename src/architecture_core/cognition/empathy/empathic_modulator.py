"""Empathic modulator — robot self-affect grounded in free energy dynamics.

Maintains the robot's OWN affective state on the circumplex.  Primary
signals derive from expected free energy (G) and policy entropy:

  valence = tanh(-ΔG / τ)          (Joffily & Coricelli 2013)
  arousal = 2·(H[q(π)] / log|Π|) - 1   (normalized policy entropy)

Note: strictly, valence should derive from variational free energy F
(inference/evidence bound), not expected free energy G (policy evaluation).
Currently we use G_social as the best available proxy.  When a proper VFE
estimate becomes available, substitute it here.

Empathic contagion enters as a secondary coupling term in the generative
model: observed other-affect shifts our predicted environmental risk,
which modifies G and thereby affects valence via -ΔG.  Contagion does
NOT directly mutate valence/arousal.

Script violations increase free energy (they ARE prediction errors),
accumulated via pending_fe_increment and folded into the next G by
the Executive.

Theoretical sources:
- Joffily & Coricelli (2013) — "Emotional Valence and the Free-Energy Principle"
- Pattisapu, Verbelen, Pitliya, Kiefer & Albarracin (2024)
  "Free Energy in a Circumplex Model of Emotion"
- Albarracin, Constant, Friston & Ramstead (2021)
  "A Variational Approach to Scripts"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from architecture_core.core.types import (
    AffectState,
    PerceptBundle,
    ScriptViolation,
    SkillUpdate,
)


@dataclass
class EmpathyConfig:
    """Tunable empathy parameters.

    contagion_alpha : float
        Coupling strength: how much others' affect shifts our G estimate.
        0 = selfish robot, higher = stronger empathic coupling.
    max_contagion_step : float
        Cap on per-tick G shift from contagion.
    valence_scale : float
        τ in tanh(-ΔG / τ) — controls valence sensitivity to G changes.
    distress_valence : float
        Extreme negative valence threshold for interrupt.
    distress_arousal : float
        Extreme arousal threshold for interrupt.
    """
    contagion_alpha: float = 0.15
    max_contagion_step: float = 0.2
    valence_scale: float = 1.0
    distress_valence: float = -0.6
    distress_arousal: float = 0.6


class EmpathicModulator:
    """Robot self-affect engine grounded in free energy dynamics.

    Lifecycle per tick (called from Executive):
        1. ``predict(G, H, |Π|)`` — valence from -ΔG, arousal from H[q(π)]
        2. ``observe(pb, violation)`` — store contagion coupling + violation → G
        3. ``modulate(base_update)``  — adjust SkillUpdate based on self-affect

    Contagion enters as a coupling term: others' affect shifts our G
    estimate (via ``contagion_fe_shift()``), which the Executive folds
    into the next ``predict()`` call.  Violations accumulate as pending
    G increments (via ``consume_pending_fe()``).
    """

    def __init__(self, config: Optional[EmpathyConfig] = None) -> None:
        self._config = config or EmpathyConfig()
        self._state = AffectState()
        self._last_violation: Optional[ScriptViolation] = None
        # Coupling signals (set by observe, consumed by Executive)
        self._pending_fe_increment: float = 0.0
        self._other_mean_valence: float = 0.0
        self._other_max_arousal: float = 0.0

    @property
    def state(self) -> AffectState:
        return self._state

    # ------------------------------------------------------------------
    # 1. PREDICT — valence from -ΔG, arousal from H[q(π)]
    # ------------------------------------------------------------------
    def predict(
        self,
        current_fe: float = 0.0,
        policy_entropy: float = 0.0,
        num_policies: int = 5,
    ) -> None:
        """Compute self-affect from free energy dynamics.

        Parameters
        ----------
        current_fe : float
            Effective G(t) — should include base G_social + violation
            increments + contagion shifts, as composed by the Executive.
        policy_entropy : float
            H[q(π)] in nats — entropy over policy posterior.
        num_policies : int
            |Π| — number of policies for entropy normalization.
        """
        cfg = self._config
        prev_F = self._state.raw_free_energy

        # ΔG = G(t) - G(t-1)
        delta_F = current_fe - prev_F
        self._state.delta_free_energy = delta_F
        self._state.raw_free_energy = current_fe
        self._state.policy_entropy = policy_entropy

        # Valence = tanh(-ΔG / τ): G decreasing → positive valence
        self._state.valence = math.tanh(-delta_F / max(cfg.valence_scale, 1e-6))

        # Arousal = 2 * (H[q(π)] / log|Π|) - 1: normalized to [-1, 1]
        log_n = math.log(max(num_policies, 2))
        normalized_entropy = min(1.0, max(0.0, policy_entropy / log_n))
        self._state.arousal = 2.0 * normalized_entropy - 1.0

    # ------------------------------------------------------------------
    # 2. OBSERVE — store coupling signals (no direct affect mutation)
    # ------------------------------------------------------------------
    def observe(
        self,
        pb: PerceptBundle,
        violation: Optional[ScriptViolation] = None,
    ) -> None:
        """Extract coupling signals from observations.

        Does NOT directly modify valence/arousal.  Instead:
        - Stores other-affect for contagion G coupling
        - Accumulates violation KL as pending G increment

        The Executive composes these into the next ``predict()`` call.
        """
        # --- Store other-affect for contagion coupling ---
        affect_data = pb.social.get("affect", {})
        readings = affect_data.get("readings", [])

        if readings:
            other_valences = [r.get("valence", 0.0) for r in readings]
            other_arousals = [r.get("arousal", 0.0) for r in readings]
            self._other_mean_valence = sum(other_valences) / len(other_valences)
            self._other_max_arousal = max(other_arousals)
        else:
            self._other_mean_valence = 0.0
            self._other_max_arousal = 0.0

        # --- Violation → accumulate G increment ---
        if (
            violation is not None
            and violation.kl_divergence > 0
            and violation is not self._last_violation
        ):
            self._last_violation = violation
            self._pending_fe_increment += violation.kl_divergence

    # ------------------------------------------------------------------
    # G coupling accessors (consumed by Executive)
    # ------------------------------------------------------------------
    def consume_pending_fe(self) -> float:
        """Return and reset accumulated violation G increment."""
        inc = self._pending_fe_increment
        self._pending_fe_increment = 0.0
        return inc

    def contagion_fe_shift(self) -> float:
        """G shift from empathic contagion coupling.

        Others' distress (negative valence) increases our predicted risk,
        producing a positive G shift.  Others' wellbeing (positive valence)
        decreases our predicted risk, producing a negative G shift.

        This is a coupling term in the generative model: observed
        other-affect modifies our expected environmental risk.
        """
        cfg = self._config
        # Negative other valence → positive G shift (more risk)
        shift = -cfg.contagion_alpha * self._other_mean_valence
        return max(-cfg.max_contagion_step, min(cfg.max_contagion_step, shift))

    # ------------------------------------------------------------------
    # 3. MODULATE — diagnostics + distress interrupt
    # ------------------------------------------------------------------
    def modulate(self, base_update: SkillUpdate) -> SkillUpdate:
        """Inject affect diagnostics and check distress interrupt.

        Speed modulation and epistemic boost are no longer applied here:
        - Speed: handled by EFE naturally (affect risk term makes cautious
          actions preferred → yield/wait selected → lower speed inherently)
        - Epistemic: handled by precision coupling (high arousal → lower β
          → more exploration in softmax)

        Returns a *new* SkillUpdate (does not mutate the input).
        """
        cfg = self._config
        params = dict(base_update.params)

        # Inject self-affect into params for downstream consumers
        params["self_affect_arousal"] = round(self._state.arousal, 3)
        params["self_affect_valence"] = round(self._state.valence, 3)

        # Distress interrupt (safety override — kept)
        recommend_interrupt = base_update.recommend_interrupt
        if (
            self._state.valence < cfg.distress_valence
            and self._state.arousal > cfg.distress_arousal
        ):
            recommend_interrupt = True

        return SkillUpdate(
            intent=base_update.intent,
            params=params,
            constraints=base_update.constraints,
            recommend_interrupt=recommend_interrupt,
            debug=base_update.debug,
        )
