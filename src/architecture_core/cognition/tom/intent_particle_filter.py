"""Active inference intent particle filter.

Maintains a particle-based posterior over each observed agent's
IntentProfile — a parametric model of their behavioral tendencies.
This replaces the log-linear product-of-experts in IntentInference
with proper Bayesian inference.

Ported from OpponentInversion (empathy-prisoner-dilemma) and adapted
for continuous robotics observations (gaze, affect, kinematics) rather
than discrete game actions.

Theoretical sources:
- Particle filter structure from OpponentInversion (empathy-prisoner-dilemma)
- Active inference framing from Friston et al. (2017)
- Behavioral profiling from Albarracin et al. (2021) scripts paper
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from architecture_core.core.types import Intent, PerceptBundle


# All intent labels
INTENT_LABELS: List[Intent] = ["approach", "avoid", "yield", "wait", "neutral"]
_INTENT_TO_IDX = {i: idx for idx, i in enumerate(INTENT_LABELS)}
_N_INTENTS = len(INTENT_LABELS)


# =====================================================================
# IntentProfile — parametric agent model
# =====================================================================
@dataclass
class IntentProfile:
    """Behavioral parameters for one observed agent.

    Analogous to BehavioralProfile from empathy-prisoner-dilemma,
    adapted for robotics social navigation.

    approach_bias : float
        Base tendency to approach the robot (logit scale).
        Positive = cooperative/approaching, negative = avoidant.
    responsiveness : float
        Sensitivity to the robot's recent actions.
        High = tit-for-tat-like (reciprocates robot behavior).
    precision : float
        Action determinism (inverse temperature, beta).
        High = predictable, low = noisy/random.
    empathy_j : float
        Degree to which this agent considers robot wellbeing.
        0 = purely selfish, 1 = fully empathetic.
    """
    approach_bias: float = 0.0
    responsiveness: float = 0.0
    precision: float = 1.0
    empathy_j: float = 0.5


# =====================================================================
# Observation context for likelihood computation
# =====================================================================
@dataclass
class ObservationContext:
    """Multi-modal observations from one tick for one agent."""
    # Kinematic
    kinematic_intent: Intent = "neutral"
    distance: float = 5.0
    velocity: float = 0.0
    approaching: bool = False
    # Engagement
    gaze_on_robot: float = 0.5
    body_orientation: float = 0.5
    # Affect
    valence: float = 0.0
    arousal: float = 0.0
    # Robot's recent action (for responsiveness modeling)
    robot_last_intent: Intent = "neutral"


# =====================================================================
# A-matrix: generative model observation likelihood P(intent | profile, context)
# =====================================================================
#
# Maps latent IntentProfile parameters → intent logits via linear
# combination.  profile.precision acts as inverse temperature (β)
# scaling all logits before softmax.
#
# Each intent's logit is:
#   logit_i = Σ_j  A[i,j] · feature_j
#
# where features are derived from (profile, context) pairs:
#   bias      = profile.approach_bias
#   resp      = profile.responsiveness * robot_feature
#   emp_shift = 2 * profile.empathy_j - 0.5
#   dist      = max(0, 1 - distance/5)
#   bias_abs  = |profile.approach_bias|
#   emp       = profile.empathy_j
#   resp_pos  = profile.responsiveness * max(0, robot_feature)
#
# These are generative model parameters — they define how the latent
# profile produces observable intent.  profile.precision scales the
# entire logit vector (inverse temperature).
#
# Columns:  bias   resp   emp_shift  dist  bias_abs  emp  resp_pos  const
_A_MATRIX = {
    "approach": {"bias": 1.0, "resp": 1.0, "emp_shift": 1.0, "dist": 1.0},
    "avoid":    {"bias": -0.8, "resp": -0.5, "emp_shift": -0.5},
    "yield":    {"bias_abs": -0.5, "emp": 1.0, "resp_pos": 0.3},
    "wait":     {"bias_abs": -0.3, "dist": 0.2},
    "neutral":  {},  # reference category (all zeros)
}

# Robot's last intent → scalar feature for responsiveness modeling
_ROBOT_INTENT_FEATURE = {
    "approach": 1.0, "avoid": -1.0, "yield": -0.5,
    "wait": 0.0, "neutral": 0.0,
}


def compute_intent_probability(profile: IntentProfile, obs: ObservationContext) -> np.ndarray:
    """P(intent | profile, context) via softmax over A-matrix logits.

    Module-level function so the EFE rollout can reuse the same generative
    model without coupling to IntentParticleFilter.

    Logit computation follows the A-matrix structure defined above:
      logit_i = Σ_j A[i,j] · feature_j
      P(intent) = softmax(precision · logits)
    """
    robot_feature = _ROBOT_INTENT_FEATURE.get(obs.robot_last_intent, 0.0)

    # Derived features
    bias = profile.approach_bias
    resp = profile.responsiveness * robot_feature
    emp_shift = 2.0 * profile.empathy_j - 0.5
    dist = max(0.0, 1.0 - obs.distance / 5.0)
    bias_abs = abs(profile.approach_bias)
    emp = profile.empathy_j
    resp_pos = profile.responsiveness * max(0.0, robot_feature)

    features = {
        "bias": bias, "resp": resp, "emp_shift": emp_shift, "dist": dist,
        "bias_abs": bias_abs, "emp": emp, "resp_pos": resp_pos,
    }

    logits = np.zeros(_N_INTENTS)
    for i, intent in enumerate(INTENT_LABELS):
        row = _A_MATRIX[intent]
        for feat_name, weight in row.items():
            logits[i] += weight * features[feat_name]

    logits *= profile.precision
    logits -= logits.max()
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


# =====================================================================
# IntentParticleFilter
# =====================================================================
class IntentParticleFilter:
    """Particle filter over IntentProfile for one observed agent.

    Each particle represents a hypothesis about the agent's behavioral
    parameters.  On each observation, particles are weighted by how
    well they explain the observed signals, then resampled when the
    effective sample size drops.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    resample_threshold : float
        Resample when ESS < threshold * n_particles.
    jitter_std : float
        Standard deviation of Gaussian jitter added after resampling.
    """

    def __init__(
        self,
        n_particles: int = 100,
        resample_threshold: float = 0.5,
        jitter_std: float = 0.1,
    ) -> None:
        self._n = n_particles
        self._resample_threshold = resample_threshold
        self._jitter_std = jitter_std
        self._effective_jitter = jitter_std  # modulated by arousal
        self._rng = np.random.default_rng(42)

        # Initialize particles from priors
        self._profiles: List[IntentProfile] = []
        self._weights = np.ones(n_particles) / n_particles

        for _ in range(n_particles):
            self._profiles.append(IntentProfile(
                approach_bias=float(self._rng.normal(0.0, 2.0)),
                responsiveness=float(self._rng.normal(0.0, 1.5)),
                precision=float(max(0.01, self._rng.gamma(2.0, 1.0))),
                empathy_j=float(self._rng.uniform(0.0, 1.0)),
            ))

    @property
    def n_particles(self) -> int:
        return self._n

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    # ------------------------------------------------------------------
    # Core: update beliefs from observation
    # ------------------------------------------------------------------
    def update(self, obs: ObservationContext) -> None:
        """Bayesian weight update from multi-modal observation.

        Computes P(observation | profile) for each particle, multiplies
        into weights, normalises, and resamples if ESS is low.
        """
        log_likelihoods = np.zeros(self._n)

        for k, profile in enumerate(self._profiles):
            log_likelihoods[k] = self._log_likelihood(profile, obs)

        # Numerically stable weight update in log space
        log_weights = np.log(np.maximum(self._weights, 1e-30)) + log_likelihoods
        log_weights -= log_weights.max()  # shift for stability
        self._weights = np.exp(log_weights)

        w_sum = self._weights.sum()
        if w_sum > 0:
            self._weights /= w_sum
        else:
            self._weights = np.ones(self._n) / self._n

        # Resample if ESS is low
        if self.ess < self._resample_threshold * self._n:
            self._resample()

    # ------------------------------------------------------------------
    # Predict intent distribution
    # ------------------------------------------------------------------
    def predict_intent(self, obs: ObservationContext) -> Dict[Intent, float]:
        """Bayesian model averaging: q(intent) = sum_k w_k * P(intent|profile_k).

        Returns a proper probability distribution over intent labels.
        """
        dist = np.zeros(_N_INTENTS)

        for k, profile in enumerate(self._profiles):
            p_intent = self._intent_probability(profile, obs)
            dist += self._weights[k] * p_intent

        total = dist.sum()
        if total > 0:
            dist /= total

        return {INTENT_LABELS[i]: float(dist[i]) for i in range(_N_INTENTS)}

    # ------------------------------------------------------------------
    # Weighted mean profile (for interpretation)
    # ------------------------------------------------------------------
    def mean_profile(self) -> IntentProfile:
        """Weighted average of all particle profiles."""
        ab = sum(self._weights[k] * self._profiles[k].approach_bias for k in range(self._n))
        rs = sum(self._weights[k] * self._profiles[k].responsiveness for k in range(self._n))
        pr = sum(self._weights[k] * self._profiles[k].precision for k in range(self._n))
        ej = sum(self._weights[k] * self._profiles[k].empathy_j for k in range(self._n))
        return IntentProfile(
            approach_bias=float(ab),
            responsiveness=float(rs),
            precision=float(pr),
            empathy_j=float(ej),
        )

    # ------------------------------------------------------------------
    # Arousal-modulated jitter (precision feedback)
    # ------------------------------------------------------------------
    def set_arousal_modulation(self, arousal: float) -> None:
        """Modulate resampling jitter by arousal (precision feedback).

        High arousal → more jitter → faster adaptation (uncertain → explore).
        Low arousal → less jitter → conservative updates (confident → exploit).

        Modulation range: [0.5, 2.0] on base jitter_std.
        """
        mod = 0.5 + 1.5 * max(0.0, min(1.0, arousal))
        self._effective_jitter = self._jitter_std * mod

    # ------------------------------------------------------------------
    # Reliability metrics
    # ------------------------------------------------------------------
    @property
    def ess(self) -> float:
        """Effective sample size: 1 / sum(w_k^2)."""
        return 1.0 / np.sum(self._weights ** 2)

    @property
    def entropy(self) -> float:
        """Shannon entropy of particle weights."""
        h = 0.0
        for w in self._weights:
            if w > 1e-10:
                h -= w * math.log(w)
        return h

    @property
    def reliability(self) -> float:
        """Profile-concentration confidence score in [0, 1].

        Measures how much the particle *profiles* have converged relative
        to their prior spread.  This survives resampling (which resets
        weights to uniform but clusters profiles around high-likelihood
        regions), unlike the old weight-entropy metric.

        Uses sigmoid gating from GatedToM.
        """
        # Posterior standard deviations of key profile parameters
        ab = np.array([p.approach_bias for p in self._profiles])
        ej = np.array([p.empathy_j for p in self._profiles])

        # Prior standard deviations (from __init__ sampling)
        ab_prior_std = 2.0                    # N(0, 2.0)
        ej_prior_std = 1.0 / math.sqrt(12)   # U(0, 1) ≈ 0.289

        # Concentration = 1 - (posterior_std / prior_std), clamped to [0, 1]
        ab_conc = max(0.0, 1.0 - float(np.std(ab)) / ab_prior_std)
        ej_conc = max(0.0, 1.0 - float(np.std(ej)) / ej_prior_std)

        confidence = 0.5 * (ab_conc + ej_conc)
        return _sigmoid(confidence, center=0.5, scale=0.1)

    # ------------------------------------------------------------------
    # Epistemic value: full-state information gain
    # ------------------------------------------------------------------
    def epistemic_value(self, candidate_intent: Intent, obs: ObservationContext) -> float:
        """E_q(o|π)[KL(q(s|o,π) || q(s|π))] — expected information gain.

        For each possible response intent o:
          1. Compute hypothetical posterior weights q(s|o,π)
          2. KL divergence from prior q(s|π) to posterior q(s|o,π)
          3. Weight by marginal p(o|π)

        This covers ALL profile parameters (approach_bias, responsiveness,
        precision, empathy_j) simultaneously via particle weight updates,
        replacing the old discretized single-parameter entropy approach.

        Returns negative E[KL] (lower = more informative, consistent with EFE).
        """
        expected_kl = 0.0

        for r_idx, response_intent in enumerate(INTENT_LABELS):
            marginal_p = 0.0
            hyp_weights = np.zeros(self._n)

            hyp_obs = ObservationContext(
                kinematic_intent=response_intent,
                distance=obs.distance,
                velocity=obs.velocity,
                approaching=obs.approaching,
                gaze_on_robot=obs.gaze_on_robot,
                body_orientation=obs.body_orientation,
                valence=obs.valence,
                arousal=obs.arousal,
                robot_last_intent=candidate_intent,
            )

            for k, profile in enumerate(self._profiles):
                p_response = self._intent_probability(profile, hyp_obs)[r_idx]
                marginal_p += self._weights[k] * p_response
                hyp_weights[k] = self._weights[k] * p_response

            if marginal_p > 1e-10:
                hyp_weights /= hyp_weights.sum()
                kl = self._particle_kl(hyp_weights)
                expected_kl += marginal_p * kl

        return -expected_kl  # Negative: lower EFE = more informative

    # ------------------------------------------------------------------
    # Private: generative model
    # ------------------------------------------------------------------
    def _log_likelihood(self, profile: IntentProfile, obs: ObservationContext) -> float:
        """Log P(observation | profile).

        The generative model predicts what signals we'd observe from an
        agent with this profile, then scores the actual observation.
        """
        p_intent = self._intent_probability(profile, obs)
        obs_idx = _INTENT_TO_IDX.get(obs.kinematic_intent, 4)

        # Kinematic likelihood: how likely is the observed kinematic intent?
        ll = math.log(max(p_intent[obs_idx], 1e-8))

        # Engagement consistency: high approach_bias + empathy → expect gaze
        expected_gaze = _sigmoid(profile.approach_bias * 0.5 + profile.empathy_j)
        ll += _gaussian_log_prob(obs.gaze_on_robot, expected_gaze, sigma=0.3)

        # Affect consistency: empathetic agents tend positive valence
        expected_valence = 0.3 * profile.empathy_j - 0.2 * (1 - profile.empathy_j)
        ll += _gaussian_log_prob(obs.valence, expected_valence, sigma=0.5)

        # Arousal consistency: responsive agents show higher arousal
        expected_arousal = 0.2 + 0.3 * abs(profile.responsiveness)
        ll += _gaussian_log_prob(obs.arousal, expected_arousal, sigma=0.4)

        return ll

    def _intent_probability(self, profile: IntentProfile, obs: ObservationContext) -> np.ndarray:
        return compute_intent_probability(profile, obs)

    # ------------------------------------------------------------------
    # Private: resampling
    # ------------------------------------------------------------------
    def _resample(self) -> None:
        """Systematic resampling with jitter for diversity."""
        positions = (np.arange(self._n) + self._rng.uniform()) / self._n
        cumsum = np.cumsum(self._weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, self._n - 1)

        jitter = self._effective_jitter
        new_profiles = []
        for idx in indices:
            old = self._profiles[idx]
            new_profiles.append(IntentProfile(
                approach_bias=old.approach_bias + float(self._rng.normal(0, jitter)),
                responsiveness=old.responsiveness + float(self._rng.normal(0, jitter)),
                precision=max(0.01, old.precision + float(self._rng.normal(0, jitter * 0.5))),
                empathy_j=float(np.clip(
                    old.empathy_j + self._rng.normal(0, jitter * 0.5),
                    0.0, 1.0,
                )),
            ))

        self._profiles = new_profiles
        self._weights = np.ones(self._n) / self._n

    # ------------------------------------------------------------------
    # Private: full-state KL divergence over particle weights
    # ------------------------------------------------------------------
    def _particle_kl(self, posterior_weights: np.ndarray) -> float:
        """KL(q_posterior || q_prior) from particle weights.

        KL = Σ_k w_post(k) · ln(w_post(k) / w_prior(k))

        This captures information gain across ALL profile parameters
        simultaneously — the posterior weight for each particle reflects
        how well its entire IntentProfile explains the observation.
        """
        prior_w = self._weights
        kl = 0.0
        for k in range(self._n):
            if posterior_weights[k] > 1e-10 and prior_w[k] > 1e-10:
                kl += posterior_weights[k] * math.log(
                    posterior_weights[k] / prior_w[k]
                )
        return max(0.0, kl)


# =====================================================================
# Math helpers
# =====================================================================
def _sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    """Numerically stable sigmoid."""
    z = (x - center) / max(scale, 1e-8)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _gaussian_log_prob(x: float, mean: float, sigma: float = 1.0) -> float:
    """Log probability under Gaussian (up to constant)."""
    return -0.5 * ((x - mean) / sigma) ** 2
