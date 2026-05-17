"""D-matrix affinity learner — Dirichlet concentration updating.

Learns fragment-context affinities (the D-matrix prior) through
Bayesian accumulation of evidence, following the active inference
framework where the D-matrix is parameterized as a Dirichlet distribution.

Each fragment f maintains a Dirichlet distribution over contexts:
    D(f) = Dir(d_f1, d_f2, ..., d_fK)

The expected affinity (prior probability) is:
    P(c | f) = d_fc / sum_k(d_fk)

Evidence accumulation:
    When fragment f is active in context c and achieves low variational
    free energy F, we increment the concentration:
        d_fc += eta * exp(-F / F_0)

    where:
    - eta is the evidence strength (proportion of accurate predictions)
    - F is the trajectory's total VFE (prediction error)
    - F_0 is a normalizing constant (expected free energy under the prior)

This is the standard Bayesian update for categorical distributions:
observing that fragment f "works well" in context c is equivalent to
observing a categorical outcome, which increments the corresponding
Dirichlet concentration parameter.

No reward signal, no learning rate decay — just evidence accumulation.
The concentration parameters monotonically increase, making the posterior
sharper (more precise) with experience.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from scipy.special import digamma, gammaln  # type: ignore[import-untyped]


@dataclass
class AffinityUpdate:
    """Record of a single Dirichlet concentration update."""
    fragment: str
    context: str
    evidence: float  # increment to concentration
    old_concentration: float
    new_concentration: float
    posterior_affinity: float  # E[P(c|f)] after update
    free_energy: float  # trajectory VFE that triggered this
    episode: int


@dataclass
class LearningCurvePoint:
    """One point on the learning curve."""
    episode: int
    fragment: str
    context: str
    affinity: float  # E[P(c|f)]
    concentration: float  # raw Dirichlet parameter
    precision: float  # sum of concentrations = total evidence


class AffinityLearner:
    """Learns D-matrix affinities via Dirichlet concentration accumulation.

    The D-matrix in active inference is a categorical prior P(s_0) over
    initial states. Here, each fragment maintains a Dirichlet prior over
    contexts, encoding "in which contexts is this fragment appropriate?"

    The Dirichlet parameterization ensures:
    - Proper Bayesian updating (conjugate prior for categorical)
    - Natural precision tracking (sum of concentrations = total evidence)
    - Convergence guarantees (posterior concentrates with evidence)
    - No ad hoc learning rates or reward signals

    Parameters
    ----------
    fragment_names : list[str]
        Names of all fragments in the repertoire.
    contexts : list[str]
        Context labels (e.g., ["reception", "corridor", "hospital"]).
    prior_concentration : float
        Initial Dirichlet concentration per (fragment, context) pair.
        Low values (e.g., 1.0) = uninformative/flat prior.
        Equal values = symmetric prior (no context preference).
    free_energy_scale : float
        Normalizing constant F_0 for evidence strength.
        Evidence = exp(-F / F_0), so F_0 controls sensitivity.
    """

    def __init__(
        self,
        fragment_names: List[str],
        contexts: List[str],
        prior_concentration: float = 1.0,
        free_energy_scale: float = 5.0,
    ) -> None:
        self._fragment_names = list(fragment_names)
        self._contexts = list(contexts)
        self._fe_scale = free_energy_scale

        # Dirichlet concentration parameters: d[fragment][context]
        # Symmetric prior = uninformative (no context preference)
        self._concentrations: Dict[str, Dict[str, float]] = {
            frag: {ctx: prior_concentration for ctx in contexts}
            for frag in fragment_names
        }

        # Store initial prior for KL computation
        self._prior_concentration = prior_concentration

        # Learning history
        self._history: List[LearningCurvePoint] = []
        self._total_episodes = 0

    @property
    def affinities(self) -> Dict[str, Dict[str, float]]:
        """Current expected affinities E[P(c|f)] from Dirichlet posterior.

        This is the normalized concentration: d_fc / sum_k(d_fk).
        """
        result = {}
        for frag in self._fragment_names:
            concs = self._concentrations[frag]
            total = sum(concs.values())
            if total > 0:
                result[frag] = {ctx: c / total for ctx, c in concs.items()}
            else:
                n = len(self._contexts)
                result[frag] = {ctx: 1.0 / n for ctx in self._contexts}
        return result

    @property
    def concentrations(self) -> Dict[str, Dict[str, float]]:
        """Raw Dirichlet concentration parameters."""
        return {
            frag: dict(ctx_map)
            for frag, ctx_map in self._concentrations.items()
        }

    @property
    def learning_curve(self) -> List[LearningCurvePoint]:
        """Full learning history."""
        return list(self._history)

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    def get_affinity(self, fragment: str, context: str) -> float:
        """Get expected affinity E[P(c|f)] for one fragment-context pair."""
        concs = self._concentrations.get(fragment)
        if concs is None:
            return 1.0 / len(self._contexts)
        total = sum(concs.values())
        return concs.get(context, self._prior_concentration) / max(total, 1e-10)

    def get_precision(self, fragment: str) -> float:
        """Get precision (total concentration) for a fragment.

        Higher precision = more evidence accumulated = sharper posterior.
        This is the Dirichlet precision parameter: sum of all concentrations.
        """
        concs = self._concentrations.get(fragment)
        if concs is None:
            return self._prior_concentration * len(self._contexts)
        return sum(concs.values())

    def get_situation_affinity(self, fragment: str) -> Dict[str, float]:
        """Get the full affinity vector for a fragment (for ScriptPattern)."""
        concs = self._concentrations.get(fragment)
        if concs is None:
            n = len(self._contexts)
            return {ctx: 1.0 / n for ctx in self._contexts}
        total = sum(concs.values())
        return {ctx: c / max(total, 1e-10) for ctx, c in concs.items()}

    def update(
        self,
        fragment: str,
        context: str,
        prediction_error: float,
        accuracy: float = 1.0,
    ) -> AffinityUpdate:
        """Accumulate evidence for fragment-context pair.

        This is the core Bayesian update: observing that fragment f
        produced low free energy in context c increments d_fc.

        The evidence strength is:
            eta = accuracy * exp(-F / F_0)

        where:
        - accuracy: fraction of predictions that were correct (0 to 1)
        - F: total variational free energy of the trajectory
        - F_0: scale parameter (expected FE under prior)

        This follows from treating each successful trajectory as a
        categorical observation: "fragment f belongs to context c."
        The exponential weighting means low-FE trajectories provide
        stronger evidence (they were more accurately predicted by the
        generative model).

        Parameters
        ----------
        fragment : str
            Fragment that was active.
        context : str
            Context in which the trajectory occurred.
        prediction_error : float
            Total variational free energy of the trajectory.
        accuracy : float
            Proportion of accurate step predictions [0, 1].
        """
        if fragment not in self._concentrations:
            raise ValueError(f"Unknown fragment: {fragment}")
        if context not in self._concentrations[fragment]:
            raise ValueError(f"Unknown context: {context}")

        # Evidence strength: exp(-F/F_0) * accuracy
        # Low FE = high evidence; high FE = negligible evidence
        evidence = accuracy * math.exp(-prediction_error / self._fe_scale)
        evidence = max(0.0, evidence)  # Non-negative concentration increment

        # Dirichlet update: increment concentration
        old_conc = self._concentrations[fragment][context]
        new_conc = old_conc + evidence
        self._concentrations[fragment][context] = new_conc

        self._total_episodes += 1

        # Compute posterior affinity
        total = sum(self._concentrations[fragment].values())
        posterior_aff = new_conc / max(total, 1e-10)

        # Record
        self._history.append(LearningCurvePoint(
            episode=self._total_episodes,
            fragment=fragment,
            context=context,
            affinity=posterior_aff,
            concentration=new_conc,
            precision=total,
        ))

        return AffinityUpdate(
            fragment=fragment,
            context=context,
            evidence=evidence,
            old_concentration=old_conc,
            new_concentration=new_conc,
            posterior_affinity=posterior_aff,
            free_energy=prediction_error,
            episode=self._total_episodes,
        )

    def variational_free_energy(self, fragment: str) -> float:
        """Compute the variational free energy of the Dirichlet posterior.

        F = KL[Dir(d_posterior) || Dir(d_prior)]
          = ln B(d_prior) - ln B(d_posterior)
            + sum_k (d_post_k - d_prior_k) * (psi(d_post_k) - psi(sum_k d_post_k))

        where B is the multivariate beta function and psi is the digamma.

        This quantifies how far the posterior has moved from the prior —
        i.e., how much has been learned about this fragment's context
        preferences.
        """
        concs = self._concentrations.get(fragment)
        if concs is None:
            return 0.0

        d_post = np.array([concs[ctx] for ctx in self._contexts])
        d_prior = np.full(len(self._contexts), self._prior_concentration)

        # KL[Dir(d_post) || Dir(d_prior)]
        kl = (
            gammaln(d_post.sum()) - gammaln(d_prior.sum())
            - np.sum(gammaln(d_post)) + np.sum(gammaln(d_prior))
            + np.sum((d_post - d_prior) * (digamma(d_post) - digamma(d_post.sum())))
        )
        return float(kl)

    def expected_free_energy_reduction(
        self, fragment: str, context: str,
    ) -> float:
        """Expected information gain from activating fragment in context.

        G = E_q[ln q(c|f) - ln p(c|f)]

        This is the epistemic value of choosing to activate fragment f
        in context c — how much would we expect to learn?

        Under the Dirichlet model:
        G ≈ psi(d_fc + 1) - psi(sum_k d_fk + 1) - [psi(d_fc) - psi(sum_k d_fk)]
          = psi(d_fc + 1) - psi(d_fc) - [psi(S+1) - psi(S)]

        where S = sum of concentrations. This is larger when d_fc is small
        (we have less evidence about this pair).
        """
        concs = self._concentrations.get(fragment)
        if concs is None:
            return 0.0

        d_fc = concs.get(context, self._prior_concentration)
        S = sum(concs.values())

        # Information gain from one more observation
        ig = (digamma(d_fc + 1) - digamma(d_fc)) - (digamma(S + 1) - digamma(S))
        return float(ig)

    def correlation_with_expert(
        self,
        expert_affinities: Dict[str, Dict[str, float]],
    ) -> float:
        """Compute Pearson correlation between learned and expert affinities."""
        learned = self.affinities
        learned_vals = []
        expert_vals = []

        for frag in self._fragment_names:
            for ctx in self._contexts:
                learned_vals.append(learned.get(frag, {}).get(ctx, 0.0))
                expert_vals.append(
                    expert_affinities.get(frag, {}).get(ctx, 0.5)
                )

        if len(learned_vals) < 2:
            return 0.0

        learned_arr = np.array(learned_vals)
        expert_arr = np.array(expert_vals)

        mean_l = learned_arr.mean()
        mean_e = expert_arr.mean()
        num = np.sum((learned_arr - mean_l) * (expert_arr - mean_e))
        denom = math.sqrt(
            np.sum((learned_arr - mean_l) ** 2)
            * np.sum((expert_arr - mean_e) ** 2)
        )

        if denom < 1e-10:
            return 0.0
        return float(num / denom)

    def has_converged(self, window: int = 20, threshold: float = 0.005) -> bool:
        """Check if posterior affinities have stabilized.

        With Dirichlet updating, convergence means the posterior
        expected values are no longer changing appreciably — the
        concentration is high enough that new evidence barely shifts
        the mean.
        """
        if len(self._history) < window:
            return False

        recent = self._history[-window:]
        # Check if affinity values are stable
        affinities_by_pair: Dict[Tuple[str, str], List[float]] = {}
        for point in recent:
            key = (point.fragment, point.context)
            affinities_by_pair.setdefault(key, []).append(point.affinity)

        if not affinities_by_pair:
            return False

        max_std = 0.0
        for values in affinities_by_pair.values():
            if len(values) > 1:
                max_std = max(max_std, float(np.std(values)))

        return max_std < threshold

    def reset(self, prior_concentration: Optional[float] = None) -> None:
        """Reset all concentrations to prior."""
        pc = prior_concentration if prior_concentration is not None else self._prior_concentration
        for frag in self._fragment_names:
            for ctx in self._contexts:
                self._concentrations[frag][ctx] = pc
        self._history.clear()
        self._total_episodes = 0
