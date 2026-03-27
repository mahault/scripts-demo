"""Script particle filter for trajectory-level inference.

Maintains a distribution over (pattern_name, step_index) hypotheses.
On each step, particles advance according to their pattern's B-matrix
and are weighted by how well the observed step matches.

This is the trajectory-level analogue of IntentParticleFilter:
- IntentParticleFilter: particles over IntentProfile, per-tick
- ScriptParticleFilter: particles over ScriptPattern x step_position,
  per-step, tracking multi-step trajectory coherence

Active inference:
- Prediction: advance particles along their pattern's step sequence
- Observation: weight particles by match between predicted and observed step
- Update: normalise weights, resample if ESS low
- Free energy: -log(marginal_likelihood) = trajectory prediction error

Adapted from IntentParticleFilter (cognition/tom/intent_particle_filter.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from architecture_core.cognition.scripts.repertoire_types import (
    ScriptPattern,
    TrajectoryStep,
)


@dataclass
class TrajectoryParticle:
    """One hypothesis about which script pattern is being followed.

    For sequence-mode (strong/legacy): step_index tracks position in
    primitives_sequence, seen_primitives is None.

    For cluster-mode (weak, cluster-based): step_index tracks how many
    cluster members have been seen, seen_primitives records which ones.
    """
    pattern_name: str
    step_index: int
    weight: float = 1.0
    seen_primitives: Optional[Set[str]] = None


class ScriptParticleFilter:
    """Particle filter over script patterns for trajectory inference.

    Parameters
    ----------
    patterns : dict[str, ScriptPattern]
        Known patterns to track.
    n_particles : int
        Total particles distributed across patterns.
    resample_threshold : float
        Resample when ESS < threshold * n_particles.
    """

    def __init__(
        self,
        patterns: Dict[str, ScriptPattern],
        n_particles: int = 50,
        resample_threshold: float = 0.5,
    ) -> None:
        self._patterns = patterns
        self._n = n_particles
        self._resample_threshold = resample_threshold
        self._rng = np.random.default_rng(42)
        self._particles: List[TrajectoryParticle] = []
        self._trajectory_fe: float = 0.0
        self._initialized = False

    def initialize(self, situation: str) -> None:
        """Distribute particles across patterns, weighted by situation affinity.

        Patterns with higher affinity for the current situation get more
        particles (D-matrix prior).
        """
        self._particles = []
        self._trajectory_fe = 0.0
        self._initialized = True

        if not self._patterns:
            return

        # Compute prior weights from situation affinity + precision
        pattern_names = list(self._patterns.keys())
        raw_weights = []
        for name in pattern_names:
            p = self._patterns[name]
            affinity = p.situation_affinity.get(situation, 0.1)
            raw_weights.append(affinity * max(p.precision, 0.01))

        total = sum(raw_weights)
        if total <= 0:
            # Uniform if no affinity info
            raw_weights = [1.0] * len(pattern_names)
            total = sum(raw_weights)

        # Allocate particles proportionally
        for i, name in enumerate(pattern_names):
            n_alloc = max(1, round(self._n * raw_weights[i] / total))
            for _ in range(n_alloc):
                self._particles.append(
                    TrajectoryParticle(pattern_name=name, step_index=0)
                )

        # Trim or pad to exactly n_particles
        while len(self._particles) > self._n:
            self._particles.pop()
        while len(self._particles) < self._n:
            # Add to highest-weighted pattern
            best = pattern_names[np.argmax(raw_weights)]
            self._particles.append(
                TrajectoryParticle(pattern_name=best, step_index=0)
            )

        # Uniform initial weights
        w = 1.0 / len(self._particles)
        for p in self._particles:
            p.weight = w

    def predict(self) -> None:
        """Advance particles one step along their hypothesised pattern.

        Dual mode:
        - Cluster mode (weak, cluster-based): no positional advancement;
          weight by how many cluster members remain unseen.
        - Sequence mode (strong/legacy): advance through primitives_sequence
          using B-matrix transition probabilities.
        """
        if not self._particles:
            return

        for particle in self._particles:
            pattern = self._patterns.get(particle.pattern_name)
            if pattern is None:
                particle.weight *= 0.01
                continue

            if pattern.is_cluster_based and not pattern.is_strong:
                # --- Cluster mode ---
                seen_count = len(particle.seen_primitives) if particle.seen_primitives else 0
                remaining = len(pattern.primitive_cluster) - seen_count
                particle.weight *= 0.9 if remaining > 0 else 0.1
            else:
                # --- Sequence mode (unchanged) ---
                seq_len = len(pattern.primitives_sequence)
                if particle.step_index + 1 < seq_len:
                    from_prim = pattern.primitives_sequence[particle.step_index]
                    to_prim = pattern.primitives_sequence[particle.step_index + 1]

                    trans_prob = self._transition_prob(pattern, from_prim, to_prim)
                    particle.step_index += 1
                    particle.weight *= max(trans_prob, 0.05)
                else:
                    particle.weight *= 0.1

    def update(self, observed_step: TrajectoryStep) -> None:
        """Weight particles by observation likelihood.

        Dual mode:
        - Cluster mode: membership + topology-weighted likelihood
        - Sequence mode: positional match likelihood

        Also tracks seen primitives for cluster-mode particles.
        """
        if not self._particles:
            return

        marginal = 0.0
        for particle in self._particles:
            ll = self._observation_likelihood(particle, observed_step)
            particle.weight *= ll
            marginal += particle.weight

            # Track seen primitives for cluster-mode particles
            pattern = self._patterns.get(particle.pattern_name)
            if (
                pattern is not None
                and pattern.is_cluster_based
                and not pattern.is_strong
            ):
                if particle.seen_primitives is None:
                    particle.seen_primitives = set()
                if observed_step.primitive_name in pattern.primitive_cluster:
                    particle.seen_primitives.add(observed_step.primitive_name)
                    particle.step_index = len(particle.seen_primitives)

        # Accumulate free energy
        if marginal > 1e-30:
            self._trajectory_fe += -math.log(marginal)
        else:
            self._trajectory_fe += 10.0  # Large penalty

        # Normalise
        total = sum(p.weight for p in self._particles)
        if total > 1e-30:
            for p in self._particles:
                p.weight /= total
        else:
            # All collapsed — reset uniform
            w = 1.0 / len(self._particles)
            for p in self._particles:
                p.weight = w

        # Resample if needed
        if self.ess < self._resample_threshold * len(self._particles):
            self._resample()

    def infer(self) -> Dict[str, float]:
        """Return posterior P(pattern | trajectory_so_far).

        Marginalises over step_index.
        """
        if not self._particles:
            return {}

        dist: Dict[str, float] = {}
        for p in self._particles:
            dist[p.pattern_name] = dist.get(p.pattern_name, 0.0) + p.weight

        # Normalise
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}

        return dist

    @property
    def best_pattern(self) -> Optional[str]:
        dist = self.infer()
        if not dist:
            return None
        return max(dist, key=dist.get)

    @property
    def confidence(self) -> float:
        dist = self.infer()
        if not dist:
            return 0.0
        return max(dist.values())

    @property
    def trajectory_free_energy(self) -> float:
        return self._trajectory_fe

    @property
    def ess(self) -> float:
        """Effective sample size."""
        if not self._particles:
            return 0.0
        weights = np.array([p.weight for p in self._particles])
        sq_sum = np.sum(weights ** 2)
        return 1.0 / sq_sum if sq_sum > 0 else 0.0

    @property
    def entropy(self) -> float:
        """Shannon entropy of particle weights."""
        h = 0.0
        for p in self._particles:
            if p.weight > 1e-10:
                h -= p.weight * math.log(p.weight)
        return h

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        """Reset for a new trajectory."""
        self._particles = []
        self._trajectory_fe = 0.0
        self._initialized = False

    def update_patterns(self, patterns: Dict[str, ScriptPattern]) -> None:
        """Update the known patterns (e.g. after consolidation)."""
        self._patterns = patterns

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _transition_prob(
        self, pattern: ScriptPattern, from_prim: str, to_prim: str,
    ) -> float:
        """Look up B-matrix transition probability."""
        trans = pattern.transition_counts.get(from_prim, {})
        entry = trans.get(to_prim)
        if entry is not None and entry.probability > 0:
            return entry.probability
        # No data — use precision-weighted uniform
        # High-precision patterns penalise unknown transitions more
        return 0.5 / max(pattern.precision, 0.1)

    def _observation_likelihood(
        self, particle: TrajectoryParticle, step: TrajectoryStep,
    ) -> float:
        """Compute P(observed_step | particle hypothesis).

        Routes to cluster or sequence likelihood depending on pattern type.
        """
        pattern = self._patterns.get(particle.pattern_name)
        if pattern is None:
            return 0.01

        if pattern.is_cluster_based and not pattern.is_strong:
            return self._cluster_likelihood(particle, pattern, step)
        else:
            return self._sequence_likelihood(particle, pattern, step)

    def _sequence_likelihood(
        self, particle: TrajectoryParticle, pattern: ScriptPattern,
        step: TrajectoryStep,
    ) -> float:
        """Likelihood for sequence-mode (strong/legacy) patterns.

        Based on positional match: does the observed primitive match the
        expected primitive at this step index?
        """
        seq = pattern.primitives_sequence
        if particle.step_index >= len(seq):
            return 0.05  # Past end of pattern

        expected_prim = seq[particle.step_index]
        observed_prim = step.primitive_name

        # Primitive match likelihood
        if expected_prim == observed_prim:
            ll_prim = 0.7 + 0.3 * min(pattern.precision / 3.0, 1.0)
        else:
            ll_prim = 0.1

        # Situation match: low violation KL = good prediction
        ll_sit = math.exp(-0.5 * step.violation_kl ** 2)

        return max(ll_prim * ll_sit, 1e-10)

    def _cluster_likelihood(
        self, particle: TrajectoryParticle, pattern: ScriptPattern,
        step: TrajectoryStep,
    ) -> float:
        """Likelihood for cluster-mode (weak, cluster-based) patterns.

        Based on:
        1. Cluster membership: is the observed primitive in the cluster?
        2. Context topology: how connected is this primitive to others
           in the current context? (proto-B-matrix)
        3. Situation KL: lower violation = better prediction

        This implements the key insight from Albarracin et al. (2021):
        weak scripts are semantic clusters whose topology reshapes
        under different contexts.
        """
        observed = step.primitive_name

        # Cluster membership
        if observed not in pattern.primitive_cluster:
            ll_member = 0.1
        else:
            ll_member = 0.7

            # Topology bonus: average connection strength in current context
            context = step.most_likely_situation
            topo = pattern.context_topology.get(context, {})
            if topo:
                connections = [
                    topo.get((observed, other), 0.0)
                    for other in pattern.primitive_cluster
                    if other != observed
                ]
                if connections:
                    avg_connection = sum(connections) / len(connections)
                    ll_member += 0.2 * avg_connection

        # Situation match
        ll_sit = math.exp(-0.5 * step.violation_kl ** 2)

        return max(ll_member * ll_sit, 1e-10)

    def _resample(self) -> None:
        """Systematic resampling preserving pattern diversity."""
        n = len(self._particles)
        if n == 0:
            return

        weights = np.array([p.weight for p in self._particles])
        cumsum = np.cumsum(weights)
        positions = (np.arange(n) + self._rng.uniform()) / n
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)

        new_particles = []
        for idx in indices:
            old = self._particles[idx]
            new_particles.append(TrajectoryParticle(
                pattern_name=old.pattern_name,
                step_index=old.step_index,
                weight=1.0 / n,
                seen_primitives=set(old.seen_primitives) if old.seen_primitives else None,
            ))

        self._particles = new_particles
