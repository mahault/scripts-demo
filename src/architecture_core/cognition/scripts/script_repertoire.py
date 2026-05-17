"""Script repertoire -- master orchestrator for script learning.

Ties together trajectory tracking, particle filter inference,
precision-based consolidation, and EFE-based composition.

Active inference framing:
- The repertoire IS the agent's generative model of social behaviour
- Each pattern is a hypothesis about behavioural structure
- Precision = confidence in each hypothesis
- Learning = Bayesian precision accumulation via free energy minimisation
- Composition = hierarchical generative model sampling when no pattern fits
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from architecture_core.core.types import AffectState, SkillRequest
from architecture_core.core.status import Status

from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    ScriptTrajectory,
    TrajectoryStep,
    WeightedPattern,
)
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.trajectory_tracker import TrajectoryTracker
from architecture_core.cognition.scripts.trajectory_inference import ScriptParticleFilter
from architecture_core.cognition.scripts.script_composer import ScriptComposer


class ScriptRepertoire:
    """Manages the learned repertoire of script patterns.

    Orchestrates:
    1. Trajectory tracking (what happened)
    2. Trajectory inference (which pattern is active)
    3. Precision update (how well patterns predict)
    4. Weak-to-strong consolidation (promote reliable patterns)
    5. Composition (generate new patterns on-the-fly)
    6. Pattern-to-ScriptSequence conversion (interface with ScriptManager)
    """

    def __init__(
        self,
        library: PrimitiveLibrary,
        config: Optional[RepertoireConfig] = None,
        initial_patterns: Optional[List[ScriptPattern]] = None,
        affinity_learner: Optional[object] = None,
    ) -> None:
        self._library = library
        self._config = config or RepertoireConfig()
        self._patterns: Dict[str, ScriptPattern] = {}
        self._tracker = TrajectoryTracker(self._config)
        self._inference = ScriptParticleFilter(
            self._patterns,
            n_particles=self._config.n_particles,
            resample_threshold=self._config.resample_threshold,
        )
        self._composer = ScriptComposer(library, self._config)
        self._active_pattern_name: Optional[str] = None
        self._norm_snapshot: Dict[str, float] = {}
        # Compositional mode: None = disabled (old behaviour), dict = enabled
        self._pattern_graph: Optional[Dict[str, Dict[str, float]]] = None
        self._pattern_graph_size: int = 0
        # Optional affinity learner for D-matrix updates
        self._affinity_learner = affinity_learner

        if initial_patterns:
            for p in initial_patterns:
                self._patterns[p.name] = p

    # ------------------------------------------------------------------
    # Norm snapshot (set by Executive each tick)
    # ------------------------------------------------------------------
    def set_norm_snapshot(self, snapshot: Dict[str, float]) -> None:
        """Store latest norm observation for trajectory recording."""
        self._norm_snapshot = snapshot

    # ------------------------------------------------------------------
    # Tick-level interface
    # ------------------------------------------------------------------
    def on_situation_recognized(
        self, situation_belief: Dict[str, float], t: float,
    ) -> Optional[str]:
        """Called when WeakScriptRecognizer produces a new belief.

        If no trajectory is active, starts one and initialises inference.
        If inference indicates high free energy, triggers composition.

        Returns the name of the active/composed pattern, or None.
        """
        most_likely = max(situation_belief, key=situation_belief.get) if situation_belief else ""

        # Start new trajectory if not recording
        if not self._tracker.is_recording:
            pattern_name = self._select_or_compose(most_likely, situation_belief)
            self._active_pattern_name = pattern_name
            self._tracker.begin_trajectory(pattern_name or "unknown", most_likely)

            # Initialise particle filter
            if self._patterns:
                self._inference.update_patterns(self._patterns)
                self._inference.initialize(most_likely)

            return pattern_name

        # Check if inference suggests switching
        if self._inference.is_initialized and self._inference.trajectory_free_energy > 5.0:
            # High free energy — current pattern isn't working
            # Compose a new one
            new_pattern = self._composer.compose(
                most_likely, situation_belief,
                norm_snapshot=self._norm_snapshot,
            )
            self._patterns[new_pattern.name] = new_pattern
            self._active_pattern_name = new_pattern.name
            return new_pattern.name

        return self._active_pattern_name

    def on_step_completed(
        self,
        t: float,
        primitive_name: str,
        situation_belief: Optional[Dict[str, float]] = None,
        outcome: Status = "SUCCESS",
        affect_before: Optional[AffectState] = None,
        affect_after: Optional[AffectState] = None,
        violation_kl: float = 0.0,
        human_intent: str = "neutral",
    ) -> None:
        """Called when a script step completes.

        1. Record in trajectory tracker
        2. Update particle filter (predict + update)
        3. Update precision of matched patterns
        """
        most_likely = ""
        if situation_belief:
            most_likely = max(situation_belief, key=situation_belief.get)

        # Record step
        self._tracker.record_step(
            t=t,
            primitive_name=primitive_name,
            situation_belief=situation_belief or {},
            most_likely_situation=most_likely,
            outcome=outcome,
            affect_before=affect_before,
            affect_after=affect_after,
            violation_kl=violation_kl,
            human_intent=human_intent,
            norm_snapshot=dict(self._norm_snapshot),
        )

        # Update inference
        if self._inference.is_initialized:
            step = TrajectoryStep(
                t=t,
                primitive_name=primitive_name,
                situation_belief=situation_belief or {},
                most_likely_situation=most_likely,
                outcome=outcome,
                affect_before=affect_before or AffectState(),
                affect_after=affect_after or AffectState(),
                violation_kl=violation_kl,
                human_intent=human_intent,
                norm_snapshot=dict(self._norm_snapshot),
            )
            self._inference.predict()
            self._inference.update(step)

    def on_script_completed(self, final_outcome: Status) -> None:
        """Called when a full script finishes.

        1. Finalise trajectory
        2. Update pattern precision
        3. Check consolidation
        4. Reset inference
        """
        traj = self._tracker.end_trajectory(final_outcome)
        if traj is None:
            return

        # Compute trajectory free energy from inference
        traj.total_prediction_error = self._inference.trajectory_free_energy

        # Update precision for the best-matched pattern
        dist = {}
        if self._inference.is_initialized:
            dist = self._inference.infer()
            if dist:
                best = max(dist, key=dist.get)
                self._update_precision(best, traj)
                self._check_consolidation(best)

        # Also update the active pattern if known
        if self._active_pattern_name and self._active_pattern_name in self._patterns:
            best_inferred = max(dist, key=dist.get) if dist else None
            if self._active_pattern_name != best_inferred:
                self._update_precision(self._active_pattern_name, traj)

            # Compositional bookkeeping: increment composition_count for
            # patterns sharing the same composition_signature
            active_pat = self._patterns.get(self._active_pattern_name)
            if active_pat is not None and active_pat.composition_signature:
                sig = active_pat.composition_signature
                for pat in self._patterns.values():
                    if pat.composition_signature == sig:
                        pat.composition_count += 1

            # Strengthen sequential edges between source fragments
            if (
                active_pat is not None
                and self._pattern_graph is not None
                and active_pat.source_fragments
            ):
                frags = active_pat.source_fragments
                for i in range(len(frags) - 1):
                    if frags[i] in self._patterns and frags[i + 1] in self._patterns:
                        self._strengthen_edge(frags[i], frags[i + 1])

        # Reset for next trajectory
        self._inference.reset()
        self._active_pattern_name = None

    # ------------------------------------------------------------------
    # Pattern management
    # ------------------------------------------------------------------
    def get_strong_scripts(self) -> List[ScriptPattern]:
        """Return patterns promoted to strong scripts."""
        return [p for p in self._patterns.values() if p.is_strong]

    def get_best_pattern(self, situation: str) -> Optional[ScriptPattern]:
        """Return highest-precision pattern for a situation type."""
        best: Optional[ScriptPattern] = None
        best_score = -1.0
        for p in self._patterns.values():
            affinity = p.situation_affinity.get(situation, 0.0)
            score = affinity * p.precision
            if score > best_score:
                best_score = score
                best = p
        return best if best_score > 0 else None

    # ------------------------------------------------------------------
    # Compositional graph mode
    # ------------------------------------------------------------------
    def enable_compositional_mode(self) -> None:
        """Activate the pattern graph for compositional retrieval.

        Builds the initial graph from all current pattern pairs.
        Until this is called, ``_pattern_graph is None`` and the
        3-tier selection skips Tier 2 entirely.
        """
        self._pattern_graph = {}
        self._pattern_graph_size = 0
        self._ensure_pattern_graph()

    def _ensure_pattern_graph(self) -> None:
        """Lazily rebuild the pattern graph when the pattern set grows."""
        if self._pattern_graph is None:
            return
        if len(self._patterns) == self._pattern_graph_size:
            return

        graph = self._pattern_graph
        names = list(self._patterns.keys())

        for i, name_a in enumerate(names):
            if name_a not in graph:
                graph[name_a] = {}
            pat_a = self._patterns[name_a]
            for name_b in names[i + 1:]:
                if name_b not in graph:
                    graph[name_b] = {}
                pat_b = self._patterns[name_b]

                # Skip if edge already exists
                if name_b in graph[name_a]:
                    continue

                # Jaccard overlap of primitive sets
                prims_a = pat_a.primitive_cluster or set(pat_a.primitives_sequence)
                prims_b = pat_b.primitive_cluster or set(pat_b.primitives_sequence)
                if prims_a and prims_b:
                    jaccard = len(prims_a & prims_b) / len(prims_a | prims_b)
                else:
                    jaccard = 0.0

                # Cosine similarity of situation_affinity vectors
                all_sits = set(pat_a.situation_affinity) | set(pat_b.situation_affinity)
                if all_sits:
                    dot = sum(
                        pat_a.situation_affinity.get(s, 0.0)
                        * pat_b.situation_affinity.get(s, 0.0)
                        for s in all_sits
                    )
                    mag_a = math.sqrt(sum(
                        v * v for v in pat_a.situation_affinity.values()
                    )) or 1e-9
                    mag_b = math.sqrt(sum(
                        v * v for v in pat_b.situation_affinity.values()
                    )) or 1e-9
                    cosine = dot / (mag_a * mag_b)
                else:
                    cosine = 0.0

                edge = 0.5 * jaccard + 0.5 * cosine
                edge = max(0.0, min(1.0, edge))
                graph[name_a][name_b] = edge
                graph[name_b][name_a] = edge

        self._pattern_graph_size = len(self._patterns)

    def retrieve_composition(
        self,
        query_scores: Dict[str, float],
        alpha: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> List[WeightedPattern]:
        """Graph-diffusion retrieval over the pattern graph.

        Parameters
        ----------
        query_scores : dict[str, float]
            Initial activation score for each pattern name.
        alpha : float | None
            Diffusion mixing coefficient (default from config).
        steps : int | None
            Number of diffusion iterations (default from config).

        Returns list of ``WeightedPattern`` sorted by descending weight.
        """
        cfg = self._config
        if alpha is None:
            alpha = cfg.graph_diffusion_alpha
        if steps is None:
            steps = cfg.graph_diffusion_steps

        self._ensure_pattern_graph()
        graph = self._pattern_graph

        # Start from query scores (patterns not in query get 0)
        scores: Dict[str, float] = {}
        for name in self._patterns:
            scores[name] = query_scores.get(name, 0.0)

        # Diffusion iterations
        if graph is not None:
            for _ in range(steps):
                new_scores: Dict[str, float] = {}
                for name in scores:
                    self_score = (1.0 - alpha) * scores[name]
                    neighbors = graph.get(name, {})
                    if neighbors:
                        neighbor_sum = sum(
                            neighbors[nb] * scores.get(nb, 0.0)
                            for nb in neighbors
                        )
                        weight_sum = sum(neighbors.values())
                        if weight_sum > 0:
                            neighbor_avg = neighbor_sum / weight_sum
                        else:
                            neighbor_avg = 0.0
                    else:
                        neighbor_avg = 0.0
                    new_scores[name] = self_score + alpha * neighbor_avg
                scores = new_scores

        # Build result sorted descending
        result = []
        for name, w in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if w > 0 and name in self._patterns:
                result.append(WeightedPattern(
                    pattern=self._patterns[name], weight=w,
                ))
        return result

    def _strengthen_edge(
        self, name_a: str, name_b: str, delta: float = 0.05,
    ) -> None:
        """Symmetric edge weight update, capped at 1.0."""
        if self._pattern_graph is None:
            return
        graph = self._pattern_graph
        for a, b in [(name_a, name_b), (name_b, name_a)]:
            if a in graph:
                current = graph[a].get(b, 0.0)
                graph[a][b] = min(1.0, current + delta)

    def pattern_to_sequence(
        self, pattern: ScriptPattern, context: Optional[str] = None,
    ) -> ScriptSequence:
        """Convert a ScriptPattern to a ScriptSequence for ScriptManager.

        For strong/legacy patterns: uses the stored primitives_sequence.
        For weak cluster-based patterns: derives ordering from
        context_topology via greedy walk.  Same cluster, different
        context -> different step ordering.
        """
        if pattern.is_cluster_based and not pattern.is_strong:
            ordered = self._derive_sequence_from_topology(pattern, context)
        else:
            ordered = list(pattern.primitives_sequence)

        steps = []
        for prim_name in ordered:
            prim = self._library.get(prim_name)
            if prim is None:
                continue
            steps.append(ScriptStep(
                request=SkillRequest(
                    skill=prim.skill_template.skill,
                    goal=dict(prim.skill_template.goal),
                    params=dict(prim.skill_template.params),
                    timeout_s=prim.typical_duration_s,
                ),
                gate="flexible",
                deontic=prim.deontic_default,
                expected_affect=prim.expected_affect,
                expected_situation=prim.postcondition_situation or None,
                primitive_name=prim_name,
            ))
        return ScriptSequence(name=pattern.name, steps=steps)

    @property
    def patterns(self) -> Dict[str, ScriptPattern]:
        return dict(self._patterns)

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @property
    def strong_count(self) -> int:
        return sum(1 for p in self._patterns.values() if p.is_strong)

    @property
    def active_pattern_name(self) -> Optional[str]:
        return self._active_pattern_name

    @property
    def active_norm_features(self) -> Dict[str, float]:
        """Norm features from the currently active pattern."""
        if self._active_pattern_name and self._active_pattern_name in self._patterns:
            return dict(self._patterns[self._active_pattern_name].norm_features)
        return {}

    @property
    def trajectory_free_energy(self) -> float:
        return self._inference.trajectory_free_energy

    @property
    def tracker(self) -> TrajectoryTracker:
        return self._tracker

    @property
    def inference(self) -> ScriptParticleFilter:
        return self._inference

    # ------------------------------------------------------------------
    # Precision dynamics
    # ------------------------------------------------------------------
    def _select_or_compose(
        self, situation: str, belief: Dict[str, float],
    ) -> Optional[str]:
        """Select best existing pattern or compose a new one.

        3-tier logic:
        Tier 1: Strong confident match (reliability > threshold) -> reuse.
        Tier 2: Compositional (graph active, >=2 patterns) -> diffuse + compose.
        Tier 3: Scratch composition (fallback = exact legacy behaviour).

        When ``_pattern_graph is None`` Tier 2 is skipped entirely so
        behaviour is identical to the pre-compositional code path.
        """
        cfg = self._config

        # Tier 1 — strong confident match
        # When graph is inactive, use legacy threshold for exact backward compat
        tier1_threshold = (
            cfg.composition_confidence_threshold
            if self._pattern_graph is not None
            else cfg.trajectory_match_threshold
        )
        best = self.get_best_pattern(situation)
        if best is not None and best.reliability > tier1_threshold:
            return best.name

        # Tier 2 — compositional retrieval (only if graph is active)
        if self._pattern_graph is not None and len(self._patterns) >= 2:
            self._ensure_pattern_graph()
            # Build query from affinity * precision for each pattern
            query_scores: Dict[str, float] = {}
            for name, pat in self._patterns.items():
                aff = pat.situation_affinity.get(situation, 0.0)
                query_scores[name] = aff * pat.precision
            weighted = self.retrieve_composition(query_scores)
            if len(weighted) >= 2:
                composite = self._composer.compose_from_patterns(
                    weighted, situation,
                )
                if composite is not None:
                    self._patterns[composite.name] = composite
                    # Strengthen co-activation edges
                    frag_names = composite.source_fragments
                    for i in range(len(frag_names)):
                        for j in range(i + 1, len(frag_names)):
                            self._strengthen_edge(frag_names[i], frag_names[j])
                    return composite.name

        # Tier 3 — scratch composition (legacy fallback)
        new_pattern = self._composer.compose(
            situation, belief, norm_snapshot=self._norm_snapshot,
        )
        self._patterns[new_pattern.name] = new_pattern
        return new_pattern.name

    def _update_precision(
        self, pattern_name: str, trajectory: ScriptTrajectory,
    ) -> None:
        """Update pattern precision based on trajectory outcome."""
        pattern = self._patterns.get(pattern_name)
        if pattern is None:
            return

        cfg = self._config
        fe = trajectory.total_prediction_error
        fe_max = cfg.max_free_energy_for_promotion * 2.0

        if fe < cfg.max_free_energy_for_promotion:
            # Good trajectory — increase precision
            quality = 1.0 - fe / fe_max if fe_max > 0 else 1.0
            delta = cfg.precision_gain_on_success * quality * trajectory.success_rate
            pattern.precision += delta
        else:
            # Bad trajectory — decrease precision
            delta = cfg.precision_loss_on_violation * (fe / fe_max if fe_max > 0 else 1.0)
            pattern.precision = max(cfg.weak_precision_floor, pattern.precision - delta)

        # Update running averages
        n = pattern.trajectory_count
        pattern.trajectory_count = n + 1
        if n > 0:
            pattern.mean_free_energy = (n * pattern.mean_free_energy + fe) / (n + 1)
            pattern.total_success_rate = (
                n * pattern.total_success_rate + trajectory.success_rate
            ) / (n + 1)
        else:
            pattern.mean_free_energy = fe
            pattern.total_success_rate = trajectory.success_rate

        # Update B-matrix (transition counts)
        steps = trajectory.steps
        for i in range(len(steps) - 1):
            from_prim = steps[i].primitive_name
            to_prim = steps[i + 1].primitive_name
            if from_prim not in pattern.transition_counts:
                pattern.transition_counts[from_prim] = {}
            trans = pattern.transition_counts[from_prim]
            if to_prim not in trans:
                from architecture_core.cognition.scripts.repertoire_types import TransitionEntry
                trans[to_prim] = TransitionEntry()
            entry = trans[to_prim]
            entry.count += 1
            # Update total_from for all transitions from this source
            total = sum(e.count for e in trans.values())
            for e in trans.values():
                e.total_from = total

        # Aggregate norm features from trajectory (precision-gated EMA)
        # High-precision patterns resist change (more evidence needed),
        # low-precision patterns adapt quickly.
        norm_snapshots = [s.norm_snapshot for s in steps if s.norm_snapshot]
        if norm_snapshots:
            base_alpha = cfg.norm_learning_rate
            min_alpha = cfg.norm_min_learning_rate
            # Gate: alpha decreases with precision (1/(1+precision))
            alpha = max(min_alpha, base_alpha / (1.0 + pattern.precision))
            for snapshot in norm_snapshots:
                for key, value in snapshot.items():
                    if key in pattern.norm_features:
                        pattern.norm_features[key] = (
                            (1 - alpha) * pattern.norm_features[key]
                            + alpha * value
                        )
                    else:
                        pattern.norm_features[key] = value

        # Hook: accumulate D-matrix evidence via affinity learner if available
        # Uses Dirichlet concentration update: d_fc += evidence(FE, accuracy)
        if self._affinity_learner is not None:
            context = trajectory.situation if hasattr(trajectory, "situation") else ""
            if not context and steps:
                context = steps[0].most_likely_situation
            if context:
                self._affinity_learner.update(
                    fragment=pattern_name,
                    context=context,
                    prediction_error=fe,
                    accuracy=trajectory.success_rate,
                )
                # Apply posterior affinity back to pattern
                learned_aff = self._affinity_learner.get_situation_affinity(pattern_name)
                if learned_aff:
                    pattern.situation_affinity.update(learned_aff)

    def _check_consolidation(self, pattern_name: str) -> bool:
        """Check if a pattern should be promoted to strong.

        For cluster-based patterns, consolidation also crystallizes the
        sequence: the B-matrix's most probable path becomes the definitive
        primitives_sequence.  This is the structural transformation from
        unordered semantic cluster to ordered behavioral sequence.
        """
        pattern = self._patterns.get(pattern_name)
        if pattern is None or pattern.is_strong:
            return False

        cfg = self._config
        if (
            pattern.precision >= cfg.strong_precision_threshold
            and pattern.trajectory_count >= cfg.min_trajectories_for_promotion
            and pattern.mean_free_energy <= cfg.max_free_energy_for_promotion
        ):
            # Crystallize: extract sequence from B-matrix
            if pattern.is_cluster_based:
                crystallized = self._crystallize_sequence(pattern)
                if crystallized:
                    pattern.primitives_sequence = crystallized

            pattern.is_strong = True
            return True

        # Compositional consolidation path: composites that have been
        # repeatedly co-activated can also crystallize into strong scripts.
        if (
            pattern.is_composite
            and pattern.composition_count >= cfg.composition_consolidation_count
            and pattern.trajectory_count >= cfg.min_trajectories_for_promotion
            and pattern.mean_free_energy <= cfg.max_free_energy_for_promotion
        ):
            if pattern.is_cluster_based:
                crystallized = self._crystallize_sequence(pattern)
                if crystallized:
                    pattern.primitives_sequence = crystallized
            pattern.is_strong = True
            return True

        return False

    def _derive_sequence_from_topology(
        self, pattern: ScriptPattern, context: Optional[str] = None,
    ) -> List[str]:
        """Derive a primitive ordering from context topology via greedy walk.

        The same cluster produces different orderings in different contexts,
        reflecting how context reshapes the topology of loosely-jointed
        semantic concepts (Albarracin et al. 2021).

        Fallback chain: topology -> primitives_sequence -> sorted cluster.
        """
        if not pattern.primitive_cluster:
            return list(pattern.primitives_sequence)

        # Pick context: explicit, or highest-affinity, or first available
        if context is None and pattern.context_topology:
            if pattern.situation_affinity:
                context = max(
                    pattern.situation_affinity, key=pattern.situation_affinity.get,
                )
            else:
                context = next(iter(pattern.context_topology))

        topo = pattern.context_topology.get(context, {}) if context else {}

        if not topo:
            # No topology data — fall back
            if pattern.primitives_sequence:
                return list(pattern.primitives_sequence)
            return sorted(pattern.primitive_cluster)

        # Compute outgoing weight sums to find best starting primitive
        out_weights: Dict[str, float] = {}
        for (a, _b), w in topo.items():
            out_weights[a] = out_weights.get(a, 0.0) + w

        remaining = set(pattern.primitive_cluster)
        sequence: List[str] = []

        # Start from highest outgoing weight
        current = max(remaining, key=lambda p: out_weights.get(p, 0.0))
        sequence.append(current)
        remaining.discard(current)

        while remaining:
            # Follow highest-weighted edge from current to remaining
            best_next = None
            best_w = -1.0
            for candidate in remaining:
                w = topo.get((current, candidate), 0.0)
                if w > best_w:
                    best_w = w
                    best_next = candidate
            if best_next is None:
                break
            sequence.append(best_next)
            remaining.discard(best_next)
            current = best_next

        return sequence

    def _crystallize_sequence(self, pattern: ScriptPattern) -> List[str]:
        """Extract most-probable ordering from accumulated B-matrix transitions.

        This is the structural weak->strong transformation: an unordered
        semantic cluster becomes an ordered sequence based on which
        transition paths were observed most frequently.

        Falls back to topology-derived ordering if B-matrix is sparse.
        """
        all_prims = (
            set(pattern.primitive_cluster)
            if pattern.primitive_cluster
            else set(pattern.primitives_sequence)
        )
        if not all_prims:
            return []

        if not pattern.transition_counts:
            return self._derive_sequence_from_topology(pattern)

        # Count incoming transitions to find likely start primitive
        incoming_count: Dict[str, int] = {p: 0 for p in all_prims}
        for from_p, targets in pattern.transition_counts.items():
            for to_p, entry in targets.items():
                if to_p in incoming_count:
                    incoming_count[to_p] += entry.count

        remaining = set(all_prims)
        # Start from primitive with fewest incoming transitions
        start = min(remaining, key=lambda p: incoming_count.get(p, 0))
        sequence = [start]
        remaining.discard(start)

        current = start
        while remaining:
            # Follow highest-probability transition
            trans = pattern.transition_counts.get(current, {})
            best_next = None
            best_prob = -1.0
            for candidate in remaining:
                entry = trans.get(candidate)
                prob = entry.probability if entry else 0.0
                if prob > best_prob:
                    best_prob = prob
                    best_next = candidate

            if best_next is None:
                best_next = next(iter(remaining))

            sequence.append(best_next)
            remaining.discard(best_next)
            current = best_next

        return sequence

    def decay_precision(self, t: float) -> None:
        """Decay all pattern precisions toward floor over time."""
        cfg = self._config
        for pattern in self._patterns.values():
            decay = cfg.precision_decay_rate
            if pattern.is_strong:
                decay *= 0.1  # Strong scripts decay 10x slower
            pattern.precision = max(
                cfg.weak_precision_floor,
                pattern.precision - decay,
            )
