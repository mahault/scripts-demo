"""A-matrix learner — generative likelihood learning via variational free energy.

Learns the observation likelihood P(o|s) — the A-matrix — by minimizing
variational free energy of a generative model. This replaces discriminative
logistic regression with proper active inference parameter learning.

Generative model:
    P(o, s) = P(o|s) P(s)

    where:
    - s ∈ {contexts} is the hidden state (situation type)
    - o ∈ R^F is the continuous observation (feature vector)
    - P(o|s) = N(o; mu_s, Sigma_s) — Gaussian likelihood per state
    - P(s) = Cat(d) — categorical prior (D-matrix)

The A-matrix parameters (mu_s, Sigma_s) are learned by accumulating
sufficient statistics from observations labeled by their posterior state:

    For each observation o with posterior q(s):
        n_s += q(s)
        sum_s += q(s) * o
        sq_sum_s += q(s) * o * o^T

    Then:
        mu_s = sum_s / n_s
        Sigma_s = sq_sum_s / n_s - mu_s * mu_s^T

This is variational EM: the E-step computes q(s|o) using current parameters,
the M-step updates parameters from sufficient statistics. The VFE decreases
monotonically, guaranteeing convergence.

The connection to the existing WeakScriptRecognizer:
- The existing recognizer computes logits = W @ features (linear in features)
- This corresponds to a Gaussian generative model where:
    W[s, :] = Sigma_s^{-1} @ mu_s  (precision-weighted mean)
- So we learn the same structure, but generatively rather than discriminatively
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from architecture_core.cognition.scripts.script_types import SituationType
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer


@dataclass
class LearningMetrics:
    """Metrics from one VFE evaluation."""
    episode: int
    vfe: float  # variational free energy
    accuracy: float  # classification accuracy under current model
    kl_divergence: float  # KL[q(s) || p(s)]
    expected_log_likelihood: float  # E_q[ln P(o|s)]


@dataclass
class LabeledSample:
    """A labeled feature vector for training."""
    features: Dict[str, float]
    label: str  # ground-truth context


# Feature names used by WeakScriptRecognizer._extract_features()
FEATURE_NAMES = [
    "agent_count", "min_distance", "inverse_min_distance",
    "mean_velocity", "max_arousal", "max_engagement",
    "has_hazard", "cue_queue_here", "cue_staff_only", "cue_quiet_zone",
]


class RecognizerLearner:
    """Learns A-matrix parameters via variational free energy minimization.

    Implements a Gaussian generative model for each context (hidden state):
        P(o | s=c) = N(o; mu_c, diag(sigma_c^2))

    Learning proceeds by accumulating sufficient statistics (online VB):
        1. E-step: compute q(s|o) = softmax(ln P(o|s) + ln P(s))
        2. M-step: update mu_s, sigma_s from q-weighted observations

    The resulting generative model is equivalent to the linear-softmax
    classifier in WeakScriptRecognizer, but derived from proper free
    energy minimization rather than discriminative fitting.

    Parameters
    ----------
    context_names : list[str]
        Labels for the situation types.
    feature_names : list[str] | None
        Feature names. Defaults to FEATURE_NAMES.
    prior_count : float
        Prior pseudo-count for each context (influences prior precision).
    prior_variance : float
        Prior variance assumption for each feature per context.
    """

    def __init__(
        self,
        context_names: List[str],
        feature_names: Optional[List[str]] = None,
        prior_count: float = 1.0,
        prior_variance: float = 1.0,
    ) -> None:
        self._contexts = list(context_names)
        self._features = feature_names or list(FEATURE_NAMES)
        self._n_contexts = len(self._contexts)
        self._n_features = len(self._features)

        # Sufficient statistics for each context
        # n_s: effective observation count
        # sum_s: sum of q(s)*o (for computing mean)
        # sq_sum_s: sum of q(s)*o^2 (for computing variance)
        self._n_s = np.full(self._n_contexts, prior_count)
        self._sum_s = np.zeros((self._n_contexts, self._n_features))
        self._sq_sum_s = np.full(
            (self._n_contexts, self._n_features), prior_count * prior_variance
        )

        # Prior parameters (for VFE computation)
        self._prior_count = prior_count
        self._prior_variance = prior_variance

        # D-matrix prior (uniform initially)
        self._log_prior = np.full(self._n_contexts, -math.log(self._n_contexts))

        # Derived parameters (recomputed from sufficient statistics)
        self._mu = np.zeros((self._n_contexts, self._n_features))
        self._sigma_sq = np.full(
            (self._n_contexts, self._n_features), prior_variance
        )
        self._recompute_parameters()

        self._history: List[LearningMetrics] = []
        self._total_observations = 0

    def _recompute_parameters(self) -> None:
        """Recompute mu and sigma from sufficient statistics."""
        for s in range(self._n_contexts):
            if self._n_s[s] > 0:
                self._mu[s] = self._sum_s[s] / self._n_s[s]
                # Variance = E[o^2] - E[o]^2, with prior floor
                variance = self._sq_sum_s[s] / self._n_s[s] - self._mu[s] ** 2
                self._sigma_sq[s] = np.maximum(variance, 0.01)
            else:
                self._mu[s] = 0.0
                self._sigma_sq[s] = self._prior_variance

    @property
    def history(self) -> List[LearningMetrics]:
        return list(self._history)

    @property
    def parameters(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Current generative model parameters: context -> feature -> (mu, sigma)."""
        result = {}
        for i, ctx in enumerate(self._contexts):
            result[ctx] = {}
            for j, feat in enumerate(self._features):
                result[ctx][feat] = (
                    float(self._mu[i, j]),
                    float(math.sqrt(self._sigma_sq[i, j])),
                )
        return result

    def warm_start(self, situation_types: List[SituationType]) -> None:
        """Initialize from existing hand-coded SituationTypes.

        Interprets feature_weights as precision-weighted means:
            W[s,f] = mu[s,f] / sigma_sq[s,f]

        So we set mu[s,f] = W[s,f] * sigma_sq[s,f] (with sigma_sq=1 initially).
        Non-specified weights imply mu=0 for that feature.
        """
        for st in situation_types:
            if st.name in self._contexts:
                i = self._contexts.index(st.name)
                for feat_name, weight in st.feature_weights.items():
                    if feat_name in self._features:
                        j = self._features.index(feat_name)
                        # Weight = mu / sigma^2, so mu = weight * sigma^2
                        self._mu[i, j] = weight * self._sigma_sq[i, j]
                        # Set sufficient statistics consistent with this
                        self._sum_s[i, j] = self._mu[i, j] * self._n_s[i]
                        self._sq_sum_s[i, j] = (
                            self._sigma_sq[i, j] + self._mu[i, j] ** 2
                        ) * self._n_s[i]

    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy vector."""
        return np.array([features.get(f, 0.0) for f in self._features])

    def _log_likelihood(self, o: np.ndarray) -> np.ndarray:
        """Compute log P(o|s) for all states s.

        P(o|s) = prod_f N(o_f; mu_sf, sigma_sf^2)
        ln P(o|s) = sum_f [-0.5 * (o_f - mu_sf)^2 / sigma_sf^2 - 0.5 * ln(2*pi*sigma_sf^2)]
        """
        # (n_contexts, n_features)
        diff = o[np.newaxis, :] - self._mu  # broadcast
        log_ll = -0.5 * np.sum(
            diff ** 2 / self._sigma_sq + np.log(2 * math.pi * self._sigma_sq),
            axis=1,
        )
        return log_ll  # shape: (n_contexts,)

    def _posterior(self, o: np.ndarray) -> np.ndarray:
        """Compute q(s|o) = softmax(ln P(o|s) + ln P(s)).

        This is the E-step: infer the posterior over hidden states
        given the current generative model parameters.
        """
        log_joint = self._log_likelihood(o) + self._log_prior
        # Numerically stable softmax
        log_joint -= log_joint.max()
        q = np.exp(log_joint)
        q_sum = q.sum()
        if q_sum > 0:
            q /= q_sum
        else:
            q = np.full(self._n_contexts, 1.0 / self._n_contexts)
        return q

    def observe(self, features: Dict[str, float], label: Optional[str] = None) -> float:
        """Process one observation, updating sufficient statistics.

        If label is provided (supervised), q(s) is set to one-hot on label.
        Otherwise (unsupervised), q(s) is inferred from the generative model.

        Returns the variational free energy for this observation:
            F = -E_q[ln P(o|s)] + KL[q(s) || P(s)]

        Parameters
        ----------
        features : dict
            Feature vector.
        label : str | None
            If provided, use as hard assignment (supervised).
            If None, infer q(s|o) from generative model (unsupervised).

        Returns
        -------
        Variational free energy for this observation.
        """
        o = self._features_to_vector(features)

        # E-step: compute posterior q(s|o)
        if label is not None and label in self._contexts:
            # Supervised: hard assignment
            q = np.zeros(self._n_contexts)
            q[self._contexts.index(label)] = 1.0
        else:
            # Unsupervised: infer from generative model
            q = self._posterior(o)

        # M-step: accumulate sufficient statistics
        for s in range(self._n_contexts):
            if q[s] > 1e-10:
                self._n_s[s] += q[s]
                self._sum_s[s] += q[s] * o
                self._sq_sum_s[s] += q[s] * o ** 2

        # Recompute parameters from updated statistics
        self._recompute_parameters()
        self._total_observations += 1

        # Compute VFE for this observation
        log_ll = self._log_likelihood(o)
        ell = float(np.sum(q * log_ll))  # E_q[ln P(o|s)]

        # KL[q(s) || P(s)]
        kl = 0.0
        for s in range(self._n_contexts):
            if q[s] > 1e-10:
                kl += q[s] * (math.log(q[s]) - self._log_prior[s])

        vfe = -ell + kl
        return vfe

    def train(
        self,
        train_samples: List[LabeledSample],
        val_samples: Optional[List[LabeledSample]] = None,
        epochs: int = 10,
    ) -> List[LearningMetrics]:
        """Train by processing observations (multiple passes for convergence).

        Each epoch processes all training samples, updating sufficient
        statistics incrementally. Multiple epochs allow the E-step
        posteriors to improve as parameters sharpen.

        Parameters
        ----------
        train_samples : list[LabeledSample]
            Training data.
        val_samples : list[LabeledSample] | None
            Validation data for monitoring.
        epochs : int
            Number of passes over the training data.

        Returns
        -------
        List of LearningMetrics per epoch.
        """
        metrics_list = []

        for epoch in range(epochs):
            epoch_vfe = 0.0

            for sample in train_samples:
                vfe = self.observe(sample.features, label=sample.label)
                epoch_vfe += vfe

            # Evaluate
            train_acc = self._evaluate_accuracy(train_samples)
            avg_vfe = epoch_vfe / max(len(train_samples), 1)

            val_acc = 0.0
            val_vfe = 0.0
            if val_samples:
                val_acc = self._evaluate_accuracy(val_samples)
                val_vfe = self._evaluate_vfe(val_samples)

            # KL from prior (average over contexts)
            kl = self._model_complexity()

            metrics = LearningMetrics(
                episode=epoch,
                vfe=avg_vfe,
                accuracy=train_acc,
                kl_divergence=kl,
                expected_log_likelihood=-avg_vfe + kl,
            )
            metrics_list.append(metrics)
            self._history.append(metrics)

        return metrics_list

    def _evaluate_accuracy(self, samples: List[LabeledSample]) -> float:
        """Classification accuracy under current generative model."""
        if not samples:
            return 0.0
        correct = 0
        for sample in samples:
            o = self._features_to_vector(sample.features)
            q = self._posterior(o)
            predicted = self._contexts[int(np.argmax(q))]
            if predicted == sample.label:
                correct += 1
        return correct / len(samples)

    def _evaluate_vfe(self, samples: List[LabeledSample]) -> float:
        """Average VFE on a set of samples (without updating)."""
        if not samples:
            return 0.0
        total_vfe = 0.0
        for sample in samples:
            o = self._features_to_vector(sample.features)
            q = self._posterior(o)
            log_ll = self._log_likelihood(o)
            ell = float(np.sum(q * log_ll))
            kl = sum(
                q[s] * (math.log(max(q[s], 1e-10)) - self._log_prior[s])
                for s in range(self._n_contexts)
                if q[s] > 1e-10
            )
            total_vfe += -ell + kl
        return total_vfe / len(samples)

    def _model_complexity(self) -> float:
        """KL divergence of learned parameters from prior.

        Measures how much the model has moved from the uninformative prior.
        This is the complexity term in the free energy bound.
        """
        # KL for Gaussian means: KL[N(mu, sigma^2) || N(0, prior_var)]
        kl = 0.0
        for s in range(self._n_contexts):
            for f in range(self._n_features):
                mu = self._mu[s, f]
                var = self._sigma_sq[s, f]
                prior_var = self._prior_variance
                # KL for single Gaussian
                kl += 0.5 * (
                    var / prior_var + mu ** 2 / prior_var
                    - 1.0 - math.log(max(var / prior_var, 1e-10))
                )
        return kl / (self._n_contexts * self._n_features)

    def apply_to_recognizer(self, recognizer: WeakScriptRecognizer) -> None:
        """Apply learned generative model to recognizer as equivalent weights.

        The precision-weighted mean mu_s / sigma_s^2 is equivalent to
        the feature weight W[s, f] in the linear-softmax recognizer.
        """
        for i, ctx in enumerate(self._contexts):
            weights = {}
            for j, feat in enumerate(self._features):
                # Weight = mu / sigma^2 (precision-weighted mean)
                w = float(self._mu[i, j] / max(self._sigma_sq[i, j], 1e-6))
                if abs(w) > 1e-6:
                    weights[feat] = w
            recognizer.set_weights(ctx, weights)

    def get_situation_types(self) -> List[SituationType]:
        """Convert learned parameters to SituationType objects.

        Uses the full Gaussian discriminant function:
            logit_c = sum_f (mu_cf/sigma_cf^2) * o_f + bias_c

        where the bias includes the Gaussian normalizer:
            bias_c = -0.5 * sum_f (mu_cf^2/sigma_cf^2 + ln(sigma_cf^2))

        Without the bias, a pure linear model (precision-weighted means)
        misclassifies when class variances differ — the normalizer encodes
        which classes are a priori less "spread out" and therefore more
        likely to generate observations close to their mean.

        The bias is included as the weight for a constant feature "_bias"
        (which must be added to the feature extractor as _bias=1.0).
        """
        types = []
        for i, ctx in enumerate(self._contexts):
            weights = {}
            # Linear term: mu_cf / sigma_cf^2
            for j, feat in enumerate(self._features):
                w = float(self._mu[i, j] / max(self._sigma_sq[i, j], 1e-6))
                if abs(w) > 1e-6:
                    weights[feat] = w
            # Bias term: -0.5 * sum_f (mu^2/sigma^2 + ln(sigma^2))
            bias = 0.0
            for j in range(len(self._features)):
                mu_sq_over_sig = self._mu[i, j] ** 2 / max(self._sigma_sq[i, j], 1e-6)
                log_sig = math.log(max(self._sigma_sq[i, j], 1e-6))
                bias -= 0.5 * (mu_sq_over_sig + log_sig)
            weights["_bias"] = float(bias)
            types.append(SituationType(name=ctx, feature_weights=weights))
        return types

    def confusion_matrix(
        self, samples: List[LabeledSample],
    ) -> Dict[str, Dict[str, int]]:
        """Compute confusion matrix on a set of samples."""
        matrix: Dict[str, Dict[str, int]] = {
            ctx: {c: 0 for c in self._contexts}
            for ctx in self._contexts
        }
        for sample in samples:
            o = self._features_to_vector(sample.features)
            q = self._posterior(o)
            predicted = self._contexts[int(np.argmax(q))]
            matrix[sample.label][predicted] += 1
        return matrix
