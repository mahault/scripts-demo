"""Social Expected Free Energy for robotics.

Computes empathy-weighted action selection via expected surprisal:

  G_social(a) = (1-λ)·G_self(a) + λ·G_other(a) + G_epistemic(a)

Each term is expected surprisal E_q(o|π)[-ln p(o|C)] under preferences:

- G_self:   binary outcome {progress, no_progress}.
            p(progress | a, h) from _P_PROGRESS matrix.
            Risk = cross_entropy(p_progress, P_PREF_PROGRESS).

- G_other:  binary outcome {comfortable, uncomfortable}.
            p(comfortable | a, h, context) from _P_COMFORT matrices.
            Risk = cross_entropy(p_comfort, P_PREF_COMFORT).
            + affect risk: cross_entropy(p_worsen, P_PREF_AFFECT).

- G_collision: binary outcome {safe, collision}.
            p(collision | gap, obstruction) from sigmoid model.
            Risk = cross_entropy(p_coll, P_PREF_COLLISION).

- G_epistemic: E_q(o|π)[KL(q(s|o,π) || q(s|π))].
            Information gain about the human's IntentProfile.

All four risk terms use the same cross-entropy formula, making
the system uniformly EFE with explicit outcomes and preferences.

Theoretical sources:
- Friston et al. (2017) — Active inference and EFE
- Pattisapu & Albarracin (2024) — Circumplex affect in free energy
- Joffily & Coricelli (2013) — Emotional valence and free energy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from architecture_core.core.types import AffectState, Intent

from architecture_core.cognition.tom.intent_particle_filter import (
    INTENT_LABELS,
    IntentParticleFilter,
    IntentProfile,
    ObservationContext,
    compute_intent_probability,
)


# =====================================================================
# Outcome observation likelihoods — generative model parameters
# =====================================================================
# p(o=progress | robot_action, human_intent) — observation likelihood
# for the self-progress outcome variable.  Values in [0, 1].
#
# Rows: robot intent, Cols: human intent
# Higher = more likely to make goal progress in this interaction
_P_PROGRESS: Dict[Intent, Dict[Intent, float]] = {
    "approach": {"approach": 0.5, "avoid": 0.8, "yield": 0.9, "wait": 0.7, "neutral": 0.6},
    "avoid":    {"approach": 0.3, "avoid": 0.2, "yield": 0.4, "wait": 0.5, "neutral": 0.4},
    "yield":    {"approach": 0.6, "avoid": 0.3, "yield": 0.1, "wait": 0.4, "neutral": 0.3},
    "wait":     {"approach": 0.4, "avoid": 0.5, "yield": 0.5, "wait": 0.01, "neutral": 0.3},
    "neutral":  {"approach": 0.5, "avoid": 0.4, "yield": 0.4, "wait": 0.3, "neutral": 0.3},
}

# p(o=comfortable | robot_action, human_intent) when robot IS obstructing.
# Observation likelihood for the other-comfort outcome variable.
_P_COMFORT_SOCIAL: Dict[Intent, Dict[Intent, float]] = {
    "approach": {"approach": 0.5, "avoid": 0.2, "yield": 0.3, "wait": 0.4, "neutral": 0.4},
    "avoid":    {"approach": 0.7, "avoid": 0.3, "yield": 0.5, "wait": 0.5, "neutral": 0.5},
    "yield":    {"approach": 0.9, "avoid": 0.4, "yield": 0.5, "wait": 0.6, "neutral": 0.6},
    "wait":     {"approach": 0.3, "avoid": 0.5, "yield": 0.5, "wait": 0.2, "neutral": 0.3},
    "neutral":  {"approach": 0.6, "avoid": 0.4, "yield": 0.5, "wait": 0.4, "neutral": 0.4},
}

# p(o=comfortable | robot_action) when robot is NOT obstructing (baseline).
_P_COMFORT_BASELINE: Dict[Intent, float] = {
    "approach": 0.5,
    "avoid": 0.3,
    "yield": 0.2,
    "wait": 0.2,
    "neutral": 0.4,
}

# Preference priors: p(o=good | C)
_P_PREF_PROGRESS = 0.9       # strong preference for making progress
_P_PREF_COMFORT = 0.9        # strong preference for other's comfort
_P_PREF_NO_COLLISION = 0.99  # very strong preference for safety
_P_PREF_NO_WORSEN = 0.95     # strong preference for not worsening affect
_EPSILON = 0.01               # floor for zero-probability cells

# Backward-compat aliases (used by tests)
REWARD_SELF = _P_PROGRESS
REWARD_OTHER = _P_COMFORT_SOCIAL


def _cross_entropy(p_outcome: float, p_pref: float) -> float:
    """E_q(o|π)[-ln p(o|C)] for binary outcome variable.

    Cross-entropy between predicted outcome distribution q(o) and
    preference distribution p(o|C).  Used for all EFE risk terms.

    Parameters
    ----------
    p_outcome : float
        p(o=good | action, context) — observation likelihood.
    p_pref : float
        p(o=good | C) — preference for the good outcome.

    Returns
    -------
    float
        Expected surprisal (positive; lower = outcome matches preferences).
    """
    p_outcome = max(_EPSILON, min(1.0 - _EPSILON, p_outcome))
    p_pref = max(_EPSILON, min(1.0 - _EPSILON, p_pref))
    return -(p_outcome * math.log(p_pref) + (1.0 - p_outcome) * math.log(1.0 - p_pref))


def _proximity_factor(obs: ObservationContext) -> float:
    """Obstruction relevance: 1.0 when head-on close, ~0 when far/diverging."""
    if obs.distance > 4.0:
        return 0.0
    base = max(0.0, 1.0 - obs.distance / 4.0)
    if not obs.approaching:
        base *= 0.2
    return base


def _p_comfort(robot_intent: Intent, human_intent: Intent, obs: ObservationContext) -> float:
    """p(o=comfortable | robot_action, human_intent, context).

    Context-dependent observation likelihood for the other agent's comfort.
    When robot is obstructing (close, approaching), uses the social matrix.
    When far or diverging, uses the baseline.
    """
    social = _P_COMFORT_SOCIAL.get(robot_intent, {}).get(human_intent, 0.4)
    baseline = _P_COMFORT_BASELINE.get(robot_intent, 0.4)
    proximity = _proximity_factor(obs)
    return proximity * social + (1.0 - proximity) * baseline


# Backward-compat alias (used by tests)
reward_other = _p_comfort


# =====================================================================
# Intent speed mapping for rollout state transitions
# =====================================================================
INTENT_SPEED: Dict[Intent, float] = {
    "approach":  1.00,   # closing at full speed
    "avoid":    -0.50,   # retreating
    "yield":     0.00,   # gap unchanged — motion is adaptive (lateral/backward)
    "wait":      0.00,   # stationary
    "neutral":   0.30,   # slow drift
}

# Precomputed speed array for fast expected-speed computation (numpy dot product)
_INTENT_SPEED_ARRAY = np.array([INTENT_SPEED[i] for i in INTENT_LABELS])

# Obstruction decay per step: how quickly each intent clears the path.
# 1.0 = no change, 0.0 = instant clearance.
OBSTRUCTION_DECAY: Dict[Intent, float] = {
    "approach":  1.00,   # stays on path
    "avoid":     0.70,   # some clearance
    "yield":     0.40,   # fast adaptive clearance
    "wait":      1.00,   # stays put
    "neutral":   0.90,   # slight drift
}


# =====================================================================
# EFE result
# =====================================================================
@dataclass
class EFEResult:
    """Result of Social EFE computation for one candidate action."""
    intent: Intent
    g_self: float
    g_other: float
    g_epistemic: float
    g_social: float
    probability: float = 0.0


@dataclass
class SocialEFEOutput:
    """Full EFE output: per-action breakdown + selected action."""
    actions: List[EFEResult]
    selected: Intent
    selected_probability: float
    empathy_factor: float
    distribution: Dict[Intent, float]


@dataclass
class RolloutConfig:
    """Configuration for multi-step EFE rollout."""
    horizon: int = 10
    dt: float = 0.4           # seconds per planning step
    gamma: float = 0.9        # discount factor
    min_distance: float = 0.3  # minimum allowed distance (body radii)


@dataclass
class RolloutEFEOutput:
    """Result of multi-step backward induction rollout."""
    selected: Intent                    # optimal first action
    full_policy: List[Intent]           # optimal action at each step
    value: float                        # V*(step 0)
    distribution: Dict[Intent, float]   # softmax over first-step Q-values
    single_step_output: SocialEFEOutput # back-compat single-step result


def _collision_probability(gap: float, obst: float) -> float:
    """p(collision | gap, obstruction) — smooth sigmoid model.

    Returns the probability of collision given the current gap distance
    and obstruction level.  The sigmoid provides smooth onset rather
    than a hard threshold.

    Parameters
    ----------
    gap : float
        Signed distance along approach axis.
    obst : float
        Obstruction level [0, 1].

    Returns
    -------
    float
        Collision probability [0, 1].
    """
    if obst <= 0.0:
        return 0.0
    return obst / (1.0 + math.exp(5.0 * (abs(gap) - 0.5)))


def _predict_affect_worsening(robot_intent: Intent, other_affect: AffectState) -> float:
    """p(affect worsens | action, current affect state).

    Models how likely the robot's action is to worsen the other agent's
    affective state, given their current distress level.  Returns 0.0
    when the other is not distressed.

    This is the observation likelihood for the affect risk term in EFE.
    """
    if other_affect is None:
        return 0.0
    distress = max(0.0, -other_affect.valence)
    if distress <= 0.0:
        return 0.0
    if robot_intent in ("approach", "neutral"):
        return min(1.0, 0.3 + 0.5 * distress)  # likely to worsen
    elif robot_intent in ("yield", "wait"):
        return max(0.0, 0.1 * distress)  # unlikely to worsen
    else:  # avoid
        return 0.2 * distress


def _immediate_cost(
    action: Intent,
    q_h,
    obs: ObservationContext,
    lam: float,
    gap: Optional[float] = None,
    obst: Optional[float] = None,
    other_affect: Optional[AffectState] = None,
) -> float:
    """Compute immediate EFE cost for one robot action.

    Shared by compute() (single-step), _backward_induction() (DP loop),
    and first-step Q computation.  All terms are expected surprisal
    E_q(o|π)[-ln p(o|C)] under explicit preferences.

    Parameters
    ----------
    action : Intent
        Robot's candidate action.
    q_h : array-like or dict
        Human intent distribution.  If dict, looked up by intent name.
        If ndarray, indexed by INTENT_LABELS order.
    obs : ObservationContext
        Current observation context (for proximity blending).
    lam : float
        Empathy factor λ ∈ [0, 1].
    gap : float | None
        Signed distance for collision risk (rollout only).
    obst : float | None
        Obstruction level for collision risk (rollout only).
    other_affect : AffectState | None
        Other agent's affect for affect risk.
    """
    g_self = 0.0
    g_other = 0.0

    for h_idx, h_intent in enumerate(INTENT_LABELS):
        if isinstance(q_h, dict):
            p_h = q_h.get(h_intent, 0.0)
        else:
            p_h = float(q_h[h_idx])

        # G_self: expected surprisal under progress preferences
        p_prog = _P_PROGRESS.get(action, {}).get(h_intent, 0.5)
        g_self += p_h * _cross_entropy(p_prog, _P_PREF_PROGRESS)

        # G_other: expected surprisal under comfort preferences
        p_comf = _p_comfort(action, h_intent, obs)
        g_other += p_h * _cross_entropy(p_comf, _P_PREF_COMFORT)

    # Affect risk: expected surprisal under affect stability preferences
    if other_affect is not None:
        p_worsen = _predict_affect_worsening(action, other_affect)
        if p_worsen > 0:
            g_other += _cross_entropy(p_worsen, 1.0 - _P_PREF_NO_WORSEN)

    immediate = (1.0 - lam) * g_self + lam * g_other

    # Collision risk: expected surprisal under safety preferences
    if gap is not None and obst is not None:
        p_coll = _collision_probability(gap, obst)
        if p_coll > 0:
            immediate += _cross_entropy(p_coll, 1.0 - _P_PREF_NO_COLLISION)

    return immediate


# =====================================================================
# SocialEFE
# =====================================================================
class SocialEFE:
    """Empathy-weighted Expected Free Energy for social navigation.

    Parameters
    ----------
    empathy_factor : float
        Lambda in [0, 1].  0 = purely selfish, 1 = fully empathetic.
    beta : float
        Action precision (inverse temperature for softmax).
    epistemic_weight : float
        Weight on information gain term.
    """

    def __init__(
        self,
        empathy_factor: float = 0.3,
        beta: float = 2.0,
        epistemic_weight: float = 0.5,
    ) -> None:
        self.empathy_factor = max(0.0, min(1.0, empathy_factor))
        self.beta = beta
        self.epistemic_weight = epistemic_weight

    def compute(
        self,
        q_human: Dict[Intent, float],
        obs: ObservationContext,
        particle_filter: Optional[IntentParticleFilter] = None,
        other_affect: Optional[AffectState] = None,
        self_arousal: Optional[float] = None,
    ) -> SocialEFEOutput:
        """Compute Social EFE for each candidate robot action.

        Parameters
        ----------
        q_human : dict[Intent, float]
            Predicted human intent distribution (from particle filter
            or GatedToM).
        obs : ObservationContext
            Current observation context (for epistemic value).
        particle_filter : IntentParticleFilter | None
            If provided, used for epistemic value computation.
        other_affect : AffectState | None
            Observed human affect state (for affect penalty).

        Returns
        -------
        SocialEFEOutput
            Per-action EFE breakdown and softmax-selected action.
        """
        results: List[EFEResult] = []
        lam = self.empathy_factor

        for robot_intent in INTENT_LABELS:
            # --- G_self + G_other via expected surprisal ---
            g_self = 0.0
            g_other = 0.0
            for human_intent in INTENT_LABELS:
                p_h = q_human.get(human_intent, 0.0)
                p_prog = _P_PROGRESS.get(robot_intent, {}).get(human_intent, 0.5)
                g_self += p_h * _cross_entropy(p_prog, _P_PREF_PROGRESS)
                p_comf = _p_comfort(robot_intent, human_intent, obs)
                g_other += p_h * _cross_entropy(p_comf, _P_PREF_COMFORT)

            # Affect risk: expected surprisal under affect stability prefs
            if other_affect is not None:
                p_worsen = _predict_affect_worsening(robot_intent, other_affect)
                if p_worsen > 0:
                    g_other += _cross_entropy(p_worsen, 1.0 - _P_PREF_NO_WORSEN)

            # --- G_epistemic: information gain ---
            g_epistemic = 0.0
            if particle_filter is not None:
                g_epistemic = (
                    self.epistemic_weight
                    * particle_filter.epistemic_value(robot_intent, obs)
                )

            # --- G_social: empathy-weighted combination ---
            g_social = (1 - lam) * g_self + lam * g_other + g_epistemic

            results.append(EFEResult(
                intent=robot_intent,
                g_self=round(g_self, 4),
                g_other=round(g_other, 4),
                g_epistemic=round(g_epistemic, 4),
                g_social=round(g_social, 4),
            ))

        # --- Action selection via softmax(-β_eff * G_social) ---
        # Precision coupling: high arousal → lower β → more exploration
        effective_beta = self.beta
        if self_arousal is not None:
            precision_mod = 1.5 - max(0.0, min(1.0, self_arousal))
            effective_beta = self.beta * precision_mod

        g_values = np.array([r.g_social for r in results])
        logits = -effective_beta * g_values  # Lower EFE = better = higher logit
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        for i, r in enumerate(results):
            r.probability = round(float(probs[i]), 4)

        distribution = {r.intent: r.probability for r in results}
        selected_idx = int(np.argmax(probs))
        selected = results[selected_idx].intent

        return SocialEFEOutput(
            actions=sorted(results, key=lambda r: r.probability, reverse=True),
            selected=selected,
            selected_probability=float(probs[selected_idx]),
            empathy_factor=lam,
            distribution=distribution,
        )

    # ------------------------------------------------------------------
    # Multi-step rollout via backward induction (Sophisticated Inference)
    # ------------------------------------------------------------------
    def compute_rollout(
        self,
        q_human: Dict[Intent, float],
        obs: ObservationContext,
        mean_profile: IntentProfile,
        particle_filter: Optional[IntentParticleFilter] = None,
        other_affect: Optional[AffectState] = None,
        config: Optional[RolloutConfig] = None,
        initial_obstruction: float = 1.0,
        self_arousal: Optional[float] = None,
    ) -> RolloutEFEOutput:
        """Multi-step EFE via backward induction.

        At each step, assumes optimal play going forward (sophisticated
        inference, Friston et al. 2021).  Uses the particle filter's
        mean_profile to predict how the other agent responds at each step.

        Parameters
        ----------
        q_human : dict[Intent, float]
            Current predicted human intent distribution.
        obs : ObservationContext
            Current observation context.
        mean_profile : IntentProfile
            Learned model of the other agent (from particle filter).
        particle_filter : IntentParticleFilter | None
            For epistemic value in single-step fallback.
        other_affect : AffectState | None
            Current human affect state.
        config : RolloutConfig | None
            Rollout parameters (defaults used if None).
        initial_obstruction : float
            How much the robot currently blocks the agent's path [0, 1].
            1.0 = fully on path, 0.0 = completely clear.  Computed from
            the robot's perpendicular distance to the agent's trajectory.
        """
        cfg = config or RolloutConfig()

        # Single-step output for back-compat
        single_step = self.compute(
            q_human, obs, particle_filter, other_affect,
            self_arousal=self_arousal,
        )

        # Backward induction
        policy, q_values = self._backward_induction(
            obs, mean_profile, cfg, initial_obstruction,
        )

        # Softmax distribution over first-step Q-values
        # Precision coupling: high arousal → lower β → more exploration
        effective_beta = self.beta
        if self_arousal is not None:
            precision_mod = 1.5 - max(0.0, min(1.0, self_arousal))
            effective_beta = self.beta * precision_mod

        logits = -effective_beta * q_values
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()
        distribution = {INTENT_LABELS[i]: float(probs[i]) for i in range(len(INTENT_LABELS))}

        selected_idx = int(np.argmin(q_values))  # lower EFE = better
        selected = INTENT_LABELS[selected_idx]

        return RolloutEFEOutput(
            selected=selected,
            full_policy=policy,
            value=float(q_values[selected_idx]),
            distribution=distribution,
            single_step_output=single_step,
        )

    def _backward_induction(
        self,
        obs: ObservationContext,
        profile: IntentProfile,
        cfg: RolloutConfig,
        initial_obstruction: float = 1.0,
    ) -> tuple:
        """Backward pass over discretized (time, gap, obstruction) state.

        Two state dimensions:
        - **gap**: signed distance along approach axis.
          Positive = haven't met, negative = passed.
        - **obstruction**: how much the robot blocks the other agent's
          path, in [0, 1].  Yield decays obstruction quickly; wait/
          approach leave it unchanged.  This is direction-agnostic —
          the actual escape direction is chosen by the IntentPolicy.

        Returns ``(full_policy, q_values_at_step_0)``.
        """
        T = cfg.horizon
        lam = self.empathy_factor

        # Discretize gap
        gap_min = -3.0
        gap_max = max(obs.distance + 1.0, 6.0)
        n_gap = 15
        gap_edges = np.linspace(gap_min, gap_max, n_gap)
        gap_width = gap_edges[1] - gap_edges[0]

        # Discretize obstruction [0, 1]
        n_obs = 4
        obs_edges = np.linspace(0.0, 1.0, n_obs)
        obs_width = obs_edges[1] - obs_edges[0]

        def gap_to_bin(g: float) -> int:
            return max(0, min(n_gap - 1, int((g - gap_min) / gap_width)))

        def obs_to_bin(o: float) -> int:
            return max(0, min(n_obs - 1, int(o / obs_width)))

        def state_to_context(gap: float, obstruction: float) -> tuple:
            """Convert (gap, obstruction) to (distance, approaching)."""
            distance = max(cfg.min_distance, abs(gap))
            # Robot is obstructing when: ahead of agent AND still on path
            obstructing = gap > 0.0 and obstruction > 0.3
            return distance, obstructing

        # V*[t][gap_bin][obs_bin]
        V = np.zeros((T + 1, n_gap, n_obs))
        pi = [[["neutral"] * n_obs for _ in range(n_gap)] for _ in range(T)]

        for t in range(T - 1, -1, -1):
            for bg in range(n_gap):
                for bo in range(n_obs):
                    gap = gap_edges[bg]
                    obst = obs_edges[bo]
                    distance, is_approaching = state_to_context(gap, obst)

                    best_q = float("inf")
                    best_action = "neutral"

                    for action in INTENT_LABELS:
                        step_obs = ObservationContext(
                            kinematic_intent=obs.kinematic_intent,
                            distance=distance,
                            velocity=obs.velocity,
                            approaching=is_approaching,
                            gaze_on_robot=obs.gaze_on_robot,
                            body_orientation=obs.body_orientation,
                            valence=obs.valence,
                            arousal=obs.arousal,
                            robot_last_intent=action,
                        )

                        q_h = compute_intent_probability(profile, step_obs)

                        immediate = _immediate_cost(
                            action, q_h, step_obs, lam,
                            gap=gap, obst=obst,
                        )

                        # Gap transition (marginalize over human intents)
                        robot_spd = INTENT_SPEED.get(action, 0.0)
                        human_spd = float(q_h @ _INTENT_SPEED_ARRAY)
                        next_gap = gap - (robot_spd + human_spd) * cfg.dt

                        # Obstruction transition
                        decay = OBSTRUCTION_DECAY.get(action, 1.0)
                        next_obst = max(0.0, min(1.0, obst * decay))

                        next_bg = gap_to_bin(next_gap)
                        next_bo = obs_to_bin(next_obst)

                        q_val = immediate + cfg.gamma * V[t + 1][next_bg][next_bo]

                        if q_val < best_q:
                            best_q = q_val
                            best_action = action

                    V[t][bg][bo] = best_q
                    pi[t][bg][bo] = best_action

        # Forward simulate to extract policy
        full_policy: List[Intent] = []
        gap = obs.distance
        obst = initial_obstruction
        for t in range(T):
            bg = gap_to_bin(gap)
            bo = obs_to_bin(obst)
            action = pi[t][bg][bo]
            full_policy.append(action)
            # Simulate forward
            distance, is_approaching = state_to_context(gap, obst)
            step_obs = ObservationContext(
                kinematic_intent=obs.kinematic_intent,
                distance=distance,
                velocity=obs.velocity,
                approaching=is_approaching,
                gaze_on_robot=obs.gaze_on_robot,
                body_orientation=obs.body_orientation,
                valence=obs.valence,
                arousal=obs.arousal,
                robot_last_intent=action,
            )
            q_h = compute_intent_probability(profile, step_obs)
            human_spd = float(q_h @ _INTENT_SPEED_ARRAY)
            gap -= (INTENT_SPEED.get(action, 0.0) + human_spd) * cfg.dt
            obst = max(0.0, min(1.0, obst * OBSTRUCTION_DECAY.get(action, 1.0)))

        # Q-values at step 0 — evaluate each action at the initial grid cell.
        # Uses grid coordinates (not exact) for consistency with full_policy.
        initial_bg = gap_to_bin(obs.distance)
        initial_bo = obs_to_bin(initial_obstruction)
        gap_0 = gap_edges[initial_bg]
        obst_0 = obs_edges[initial_bo]
        distance_0, approaching_0 = state_to_context(gap_0, obst_0)
        first_step_q = np.zeros(len(INTENT_LABELS))
        for a_idx, action in enumerate(INTENT_LABELS):
            step_obs = ObservationContext(
                kinematic_intent=obs.kinematic_intent,
                distance=distance_0,
                velocity=obs.velocity,
                approaching=approaching_0,
                gaze_on_robot=obs.gaze_on_robot,
                body_orientation=obs.body_orientation,
                valence=obs.valence,
                arousal=obs.arousal,
                robot_last_intent=action,
            )
            q_h = compute_intent_probability(profile, step_obs)
            immediate = _immediate_cost(
                action, q_h, step_obs, lam,
                gap=gap_0, obst=obst_0,
            )
            human_spd = float(q_h @ _INTENT_SPEED_ARRAY)
            next_gap = gap_0 - (INTENT_SPEED.get(action, 0.0) + human_spd) * cfg.dt
            next_obst = max(0.0, min(1.0, obst_0 * OBSTRUCTION_DECAY.get(action, 1.0)))
            next_bg = gap_to_bin(next_gap)
            next_bo = obs_to_bin(next_obst)
            first_step_q[a_idx] = immediate + cfg.gamma * V[1][next_bg][next_bo]

        return full_policy, first_step_q
