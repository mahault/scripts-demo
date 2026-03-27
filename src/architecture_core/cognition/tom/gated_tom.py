"""Gated Theory of Mind — entropy-based trust gating.

Interpolates between a learned model (particle filter) and a static
prior prediction based on the particle filter's reliability.  When
the filter is confident (low weight entropy), trust the learned model.
When uncertain (high entropy), revert to a safe static prior.

Ported from GatedToM (empathy-prisoner-dilemma).

q_gated = reliability * q_learned + (1 - reliability) * q_prior

The sigmoid gating creates a smooth transition:
- New/unknown agents → low reliability → safe prior (yield/wait biased)
- Well-observed agents → high reliability → trust learned model
- After prediction failures → entropy rises → auto-degrades trust
"""

from __future__ import annotations

from typing import Dict, Optional

from architecture_core.core.types import AffectState, Intent, PerceptBundle

from architecture_core.cognition.tom.intent_particle_filter import (
    INTENT_LABELS,
    IntentParticleFilter,
    IntentProfile,
    ObservationContext,
)
from architecture_core.cognition.tom.social_efe import (
    SocialEFE,
    SocialEFEOutput,
    RolloutConfig,
    RolloutEFEOutput,
)


# Default prior: cautious — yield and wait biased
DEFAULT_PRIOR: Dict[Intent, float] = {
    "approach": 0.10,
    "avoid": 0.15,
    "yield": 0.30,
    "wait": 0.25,
    "neutral": 0.20,
}

# Conservative default profile for rollout when the particle filter
# hasn't converged.  Assumes a moderately responsive, cautious agent —
# better to over-predict responsiveness than to assume they'll ignore us.
_DEFAULT_PROFILE = IntentProfile(
    approach_bias=0.3,     # slight tendency to approach (not avoidant)
    responsiveness=1.0,    # responsive to robot actions (cautious assumption)
    precision=1.0,         # moderate predictability
    empathy_j=0.3,         # slight empathy
)


class GatedToM:
    """Entropy-gated intent prediction with Social EFE action selection.

    Maintains a per-agent IntentParticleFilter and gates its predictions
    based on reliability.  Uses SocialEFE for empathy-weighted action
    selection.

    Parameters
    ----------
    empathy_factor : float
        Lambda for SocialEFE: 0 = selfish, 1 = fully empathetic.
    n_particles : int
        Particles per agent filter.
    prior : dict | None
        Static prior distribution over intents for unknown agents.
    beta : float
        Action precision for EFE softmax.
    epistemic_weight : float
        Weight on information gain in EFE.
    """

    def __init__(
        self,
        empathy_factor: float = 0.3,
        n_particles: int = 100,
        prior: Optional[Dict[Intent, float]] = None,
        beta: float = 2.0,
        epistemic_weight: float = 0.5,
    ) -> None:
        self._empathy_factor = empathy_factor
        self._n_particles = n_particles
        self._prior = prior or DEFAULT_PRIOR
        self._efe = SocialEFE(
            empathy_factor=empathy_factor,
            beta=beta,
            epistemic_weight=epistemic_weight,
        )

        # Per-agent particle filters
        self._filters: Dict[str, IntentParticleFilter] = {}

    def get_or_create_filter(self, entity_id: str) -> IntentParticleFilter:
        """Get or create a particle filter for an agent."""
        if entity_id not in self._filters:
            self._filters[entity_id] = IntentParticleFilter(
                n_particles=self._n_particles,
            )
        return self._filters[entity_id]

    def update(self, entity_id: str, obs: ObservationContext) -> None:
        """Update the particle filter for one agent with new observations."""
        pf = self.get_or_create_filter(entity_id)
        pf.update(obs)

    def predict_intent(
        self, entity_id: str, obs: ObservationContext,
    ) -> Dict[Intent, float]:
        """Gated intent prediction for one agent.

        q_gated = reliability * q_learned + (1 - reliability) * q_prior
        """
        pf = self.get_or_create_filter(entity_id)

        # Learned prediction from particle filter
        q_learned = pf.predict_intent(obs)
        reliability = pf.reliability

        # Blend with prior
        q_gated: Dict[Intent, float] = {}
        for intent in INTENT_LABELS:
            q_learned_val = q_learned.get(intent, 0.0)
            q_prior_val = self._prior.get(intent, 0.2)
            q_gated[intent] = reliability * q_learned_val + (1 - reliability) * q_prior_val

        # Normalise
        total = sum(q_gated.values())
        if total > 0:
            q_gated = {k: v / total for k, v in q_gated.items()}

        return q_gated

    def select_action(
        self,
        entity_id: str,
        obs: ObservationContext,
        other_affect: Optional[AffectState] = None,
    ) -> SocialEFEOutput:
        """Full active inference action selection.

        1. Get gated intent prediction for the human
        2. Compute Social EFE for each robot action candidate
        3. Return softmax-selected action with full breakdown
        """
        pf = self.get_or_create_filter(entity_id)
        q_human = self.predict_intent(entity_id, obs)

        return self._efe.compute(
            q_human=q_human,
            obs=obs,
            particle_filter=pf,
            other_affect=other_affect,
        )

    def select_action_rollout(
        self,
        entity_id: str,
        obs: ObservationContext,
        other_affect: Optional[AffectState] = None,
        rollout_config: Optional[RolloutConfig] = None,
        initial_obstruction: float = 1.0,
        self_arousal: Optional[float] = None,
    ) -> RolloutEFEOutput:
        """Multi-step action selection via backward induction.

        Always uses the full rollout — the collision penalty and multi-step
        planning are needed from the first tick.  When the particle filter
        hasn't converged yet (low reliability), a conservative default
        profile is blended in so the rollout assumes a cautious other agent.

        Parameters
        ----------
        initial_obstruction : float
            How much the robot currently blocks the agent's path [0, 1].
            Computed from perpendicular distance to the agent's trajectory.
        """
        pf = self.get_or_create_filter(entity_id)
        q_human = self.predict_intent(entity_id, obs)
        rel = pf.reliability

        # Blend learned profile with conservative default based on reliability.
        # Low reliability → mostly default (cautious, responsive agent model).
        # High reliability → trust the learned profile.
        learned = pf.mean_profile()
        profile = IntentProfile(
            approach_bias=rel * learned.approach_bias + (1 - rel) * _DEFAULT_PROFILE.approach_bias,
            responsiveness=rel * learned.responsiveness + (1 - rel) * _DEFAULT_PROFILE.responsiveness,
            precision=rel * learned.precision + (1 - rel) * _DEFAULT_PROFILE.precision,
            empathy_j=rel * learned.empathy_j + (1 - rel) * _DEFAULT_PROFILE.empathy_j,
        )

        return self._efe.compute_rollout(
            q_human=q_human,
            obs=obs,
            mean_profile=profile,
            particle_filter=pf,
            other_affect=other_affect,
            config=rollout_config,
            initial_obstruction=initial_obstruction,
            self_arousal=self_arousal,
        )

    @property
    def empathy_factor(self) -> float:
        return self._empathy_factor

    @empathy_factor.setter
    def empathy_factor(self, value: float) -> None:
        self._empathy_factor = max(0.0, min(1.0, value))
        self._efe.empathy_factor = self._empathy_factor

    def agent_reliability(self, entity_id: str) -> float:
        """Get reliability score for a specific agent."""
        if entity_id not in self._filters:
            return 0.0
        return self._filters[entity_id].reliability

    def prune_stale(self, active_ids: set) -> None:
        """Remove filters for agents no longer visible."""
        stale = [k for k in self._filters if k not in active_ids]
        for k in stale:
            del self._filters[k]
