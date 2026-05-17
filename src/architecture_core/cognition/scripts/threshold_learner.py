"""D-matrix precision gating — context gating via Dirichlet precision.

In active inference, fragment gating is NOT a hard threshold. Instead,
it emerges naturally from the precision (confidence) of the D-matrix
prior. Fragments with low Dirichlet concentration for the current
context contribute negligible prior mass and are effectively "gated out."

The gating mechanism:

    P(activate f | context c) = sigma(gamma * (E[D(f,c)] - E[D(f,:)]))

    where:
    - E[D(f,c)] = d_fc / sum_k(d_fk) is the expected affinity
    - E[D(f,:)] = 1/K is the uniform baseline
    - gamma is the precision (inverse temperature) on the prior
    - sigma is the sigmoid function

The precision gamma is itself learned via variational Laplace:
    gamma* = argmin_gamma F(gamma)

    where F is the variational free energy evaluated at gamma:
    F(gamma) = -E_q[ln P(o|s, gamma)] + KL[q(s) || P(s; gamma)]

In practice, gamma increases when the gated fragment set produces
low prediction error (good models get trusted more), and decreases
when prediction error is high (back off to less selective gating).

This is the standard active inference precision update:
    d_gamma/dt = -dF/d_gamma

    which for a scalar precision on a categorical prior reduces to:
    gamma_new = gamma_old - lr * (expected_error - actual_error)

When gamma is high: only high-affinity fragments pass (sharp gating)
When gamma is low: most fragments pass (flat gating, exploratory)

The sigmoid ensures smooth, differentiable gating rather than a hard
threshold, which is critical for gradient-based precision optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PrecisionUpdate:
    """Record of a precision update."""
    episode: int
    old_gamma: float
    new_gamma: float
    prediction_error: float
    expected_error: float
    gated_fraction: float  # fraction of fragments that passed


@dataclass
class GatingResult:
    """Result of applying precision-gated fragment selection."""
    active_fragments: List[str]
    activation_probabilities: Dict[str, float]
    gamma: float
    effective_threshold: float  # the implicit threshold at P=0.5


class PrecisionGating:
    """Precision-based fragment gating via variational Laplace.

    Replaces the hard threshold with a learned precision parameter
    gamma that controls how selectively the D-matrix prior gates
    fragment activation.

    Parameters
    ----------
    initial_gamma : float
        Initial precision (inverse temperature). Higher = more selective.
        Default 1.0 corresponds to moderate selectivity.
    gamma_prior_mean : float
        Prior mean for gamma (for regularization toward moderate selectivity).
    gamma_prior_precision : float
        Prior precision on gamma (how strongly to regularize).
    learning_rate : float
        Step size for variational Laplace updates.
    min_gamma : float
        Floor on precision (prevents gamma → 0 = no gating at all).
    max_gamma : float
        Ceiling on precision (prevents gamma → ∞ = too aggressive gating).
    """

    def __init__(
        self,
        initial_gamma: float = 1.0,
        gamma_prior_mean: float = 1.0,
        gamma_prior_precision: float = 0.1,
        learning_rate: float = 0.1,
        min_gamma: float = 0.1,
        max_gamma: float = 10.0,
        warmup_observations: int = 5,
    ) -> None:
        self._gamma = initial_gamma
        self._gamma_prior_mean = gamma_prior_mean
        self._gamma_prior_precision = gamma_prior_precision
        self._lr = learning_rate
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma

        # Running estimate of expected prediction error (for gradient)
        self._expected_error = 2.0  # initial estimate
        self._error_ema_alpha = 0.3  # EMA smoothing for expected error (fast calibration)

        # Empirical Bayes warmup: buffer first N observations to initialize expected_error
        self._warmup_observations = warmup_observations
        self._warmup_buffer: List[float] = []
        self._warmup_complete = False

        self._history: List[PrecisionUpdate] = []
        self._total_episodes = 0

    @property
    def gamma(self) -> float:
        """Current precision (inverse temperature) for gating."""
        return self._gamma

    @property
    def effective_threshold(self) -> float:
        """The implicit affinity threshold where P(activate) = 0.5.

        At P=0.5, the sigmoid argument is 0:
            gamma * (affinity - baseline) = 0
            → affinity = baseline = 1/K (for K contexts)

        But in practice, since we compare to mean affinity:
            threshold = 1/K (the point where a fragment is no
            better than chance for this context)

        Higher gamma makes the sigmoid steeper around this threshold,
        not shifting it but making gating more decisive.
        """
        return 0.5  # sigmoid crosses 0.5 at x=0, i.e., affinity = baseline

    @property
    def history(self) -> List[PrecisionUpdate]:
        return list(self._history)

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    def gate_fragments(
        self,
        affinities: Dict[str, float],
        n_contexts: int = 3,
        rng: Optional[np.random.Generator] = None,
    ) -> GatingResult:
        """Apply precision-gated selection to fragment affinities.

        Parameters
        ----------
        affinities : dict[str, float]
            Fragment name -> effective affinity for current context.
            These are E[D(f,c)] from the Dirichlet posterior.
        n_contexts : int
            Number of contexts (for computing uniform baseline).
        rng : np.random.Generator | None
            When provided, sample from Bernoulli(sigmoid_prob) instead of
            hard cutoff at 0.5. This introduces principled stochasticity
            (action selection under uncertainty). When None, deterministic
            behavior is preserved (backward compatible).

        Returns
        -------
        GatingResult with active fragments and their activation probabilities.
        """
        baseline = 1.0 / n_contexts
        active = []
        probs = {}

        for frag_name, affinity in affinities.items():
            # Sigmoid gating: P(activate) = sigma(gamma * (aff - baseline))
            logit = self._gamma * (affinity - baseline)
            prob = _sigmoid(logit)
            probs[frag_name] = prob

            if rng is not None:
                # Stochastic: Bernoulli sampling from sigmoid probability
                if rng.random() < prob:
                    active.append(frag_name)
            else:
                # Deterministic: hard cutoff at 0.5
                if prob > 0.5:
                    active.append(frag_name)

        return GatingResult(
            active_fragments=active,
            activation_probabilities=probs,
            gamma=self._gamma,
            effective_threshold=baseline,
        )

    def update(
        self,
        prediction_error: float,
        n_active: int,
        n_total: int,
    ) -> PrecisionUpdate:
        """Update precision via variational Laplace on free energy gradient.

        The free energy gradient with respect to gamma:
            dF/d_gamma = (expected_PE - actual_PE) + prior_pull

        When actual PE < expected PE: model is doing well → increase gamma
        (be more selective, trust the gating more).

        When actual PE > expected PE: model is struggling → decrease gamma
        (be less selective, let more fragments through for exploration).

        The prior_pull term regularizes gamma toward the prior mean,
        preventing runaway precision.

        Uses empirical Bayes warmup: the first N observations are buffered
        to initialize expected_error from data rather than a fixed prior.
        During warmup, gamma is not updated (only observed).

        Parameters
        ----------
        prediction_error : float
            Per-transition VFE (observation-level prediction error).
        n_active : int
            Number of fragments that were active (passed gating).
        n_total : int
            Total number of available fragments.

        Returns
        -------
        PrecisionUpdate record.
        """
        old_gamma = self._gamma
        gated_fraction = n_active / max(n_total, 1)

        # Empirical Bayes warmup: buffer observations to initialize expected_error
        if not self._warmup_complete:
            self._warmup_buffer.append(prediction_error)
            if len(self._warmup_buffer) >= self._warmup_observations:
                self._expected_error = sum(self._warmup_buffer) / len(self._warmup_buffer)
                self._warmup_complete = True
            # Record but don't update gamma during warmup
            self._total_episodes += 1
            update = PrecisionUpdate(
                episode=self._total_episodes,
                old_gamma=old_gamma,
                new_gamma=self._gamma,
                prediction_error=prediction_error,
                expected_error=self._expected_error,
                gated_fraction=gated_fraction,
            )
            self._history.append(update)
            return update

        # Free energy gradient w.r.t. gamma
        # Negative gradient (we minimize F, so we move against the gradient)
        error_signal = self._expected_error - prediction_error

        # Prior regularization: pull gamma toward prior mean
        prior_pull = self._gamma_prior_precision * (
            self._gamma_prior_mean - self._gamma
        )

        # Precision update (gradient descent on F)
        d_gamma = self._lr * (error_signal + prior_pull)
        self._gamma = np.clip(
            self._gamma + d_gamma,
            self._min_gamma,
            self._max_gamma,
        )
        self._gamma = float(self._gamma)

        # Update expected error (EMA)
        self._expected_error = (
            (1 - self._error_ema_alpha) * self._expected_error
            + self._error_ema_alpha * prediction_error
        )

        self._total_episodes += 1

        update = PrecisionUpdate(
            episode=self._total_episodes,
            old_gamma=old_gamma,
            new_gamma=self._gamma,
            prediction_error=prediction_error,
            expected_error=self._expected_error,
            gated_fraction=gated_fraction,
        )
        self._history.append(update)
        return update

    def has_converged(self, window: int = 20, threshold: float = 0.05) -> bool:
        """Check if precision has stabilized.

        Requires warmup to be complete before convergence is possible.
        """
        if not self._warmup_complete:
            return False
        if len(self._history) < window:
            return False
        recent_gammas = [h.new_gamma for h in self._history[-window:]]
        return float(np.std(recent_gammas)) < threshold

    def variational_free_energy(self) -> float:
        """Current VFE contribution from the precision parameter.

        F_gamma = 0.5 * tau_prior * (gamma - mu_prior)^2

        This is the KL cost of the precision being away from its prior.
        """
        diff = self._gamma - self._gamma_prior_mean
        return 0.5 * self._gamma_prior_precision * diff ** 2

    def reset(self, initial_gamma: Optional[float] = None) -> None:
        """Reset precision to initial value and clear warmup state."""
        if initial_gamma is not None:
            self._gamma = initial_gamma
        else:
            self._gamma = self._gamma_prior_mean
        self._expected_error = 2.0
        self._warmup_buffer.clear()
        self._warmup_complete = False
        self._history.clear()
        self._total_episodes = 0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)
