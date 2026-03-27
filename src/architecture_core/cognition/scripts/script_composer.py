"""Script composer -- EFE-based composition from primitives.

When no known ScriptPattern matches the current situation (high
trajectory free energy), the composer assembles a new script by
chaining primitives using Expected Free Energy scoring.

G_compose = w_eff * G_efficiency + w_emp * G_empathy + w_epi * G_epistemic

Active inference planning: each candidate primitive is scored by how
much it reduces expected free energy, balancing task efficiency,
empathic impact, and information gain.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from architecture_core.core.types import AffectState

from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    ScriptPrimitive,
    WeightedPattern,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary


class ScriptComposer:
    """Compose new scripts from primitives using EFE scoring."""

    def __init__(
        self,
        library: PrimitiveLibrary,
        config: Optional[RepertoireConfig] = None,
    ) -> None:
        self._library = library
        self._config = config or RepertoireConfig()
        # Track usage counts for epistemic scoring
        self._usage_counts: Dict[str, int] = {}

    def compose(
        self,
        current_situation: str,
        situation_belief: Optional[Dict[str, float]] = None,
        goal_situation: Optional[str] = None,
        affect_state: Optional[AffectState] = None,
        human_intent: str = "neutral",
        norm_snapshot: Optional[Dict[str, float]] = None,
    ) -> ScriptPattern:
        """Generate a new script pattern from primitives.

        Returns a weak ScriptPattern (low precision) assembled by
        greedily selecting primitives that minimise expected free energy.
        """
        cfg = self._config
        sequence: List[str] = []
        visited: set = set()
        sit = current_situation

        for depth in range(cfg.max_composition_length):
            if goal_situation and sit == goal_situation:
                break

            candidates = self._library.get_applicable(sit)
            if not candidates:
                break

            # Score each candidate
            scores: Dict[str, float] = {}
            for prim in candidates:
                if prim.name in visited:
                    continue
                score = self._score_primitive(
                    prim, sit, goal_situation, affect_state, human_intent, depth,
                )
                scores[prim.name] = score

            if not scores:
                break

            # Softmax selection
            selected = self._softmax_select(scores, beta=2.0)
            prim = self._library.get(selected)
            if prim is None:
                break

            sequence.append(selected)
            visited.add(selected)
            self._usage_counts[selected] = self._usage_counts.get(selected, 0) + 1
            sit = prim.postcondition_situation or sit

        if not sequence:
            # Fallback: single wait-acknowledge
            wait = self._library.get("wait-acknowledge")
            if wait is not None:
                sequence = ["wait-acknowledge"]
            else:
                # Use first available primitive
                all_prims = self._library.all_primitives
                if all_prims:
                    sequence = [all_prims[0].name]

        # Build semantic cluster with context-conditioned topology
        cluster = set(sequence)
        all_contexts: Set[str] = set()
        for prim_name in cluster:
            prim = self._library.get(prim_name)
            if prim:
                all_contexts.update(prim.precondition_situations)
        if current_situation:
            all_contexts.add(current_situation)

        topology = self._build_context_topology(cluster, list(all_contexts))

        # Create weak pattern (cluster-based)
        name = f"composed_{'_'.join(sequence[:3])}"
        return ScriptPattern(
            name=name,
            primitives_sequence=sequence,
            precision=0.1,
            situation_affinity={current_situation: 1.0},
            primitive_cluster=cluster,
            context_topology=topology,
            norm_features=dict(norm_snapshot) if norm_snapshot else {},
        )

    def compose_from_patterns(
        self,
        weighted_patterns: List[WeightedPattern],
        context: str,
    ) -> Optional[ScriptPattern]:
        """Compose a new pattern by merging multiple weighted fragment patterns.

        Given a list of ``WeightedPattern`` (each a fragment with a retrieval
        weight), merge their primitives, situation affinities, and norm
        features into a single composite ``ScriptPattern``.

        Returns ``None`` if the input is empty or yields no primitives.
        """
        if not weighted_patterns:
            return None

        # 1. Merge primitive weights: union across fragments, weighted by activation
        merged_prim_weights: Dict[str, float] = {}
        for wp in weighted_patterns:
            pat = wp.pattern
            w = wp.weight
            # Use pattern's own primitive_weights if available,
            # otherwise fall back to uniform over cluster/sequence
            if pat.primitive_weights:
                prim_source = pat.primitive_weights
            else:
                prims = pat.primitive_cluster or set(pat.primitives_sequence)
                if prims:
                    uniform = 1.0 / len(prims)
                    prim_source = {p: uniform for p in prims}
                else:
                    continue
            for prim_name, prim_w in prim_source.items():
                merged_prim_weights[prim_name] = (
                    merged_prim_weights.get(prim_name, 0.0) + prim_w * w
                )

        if not merged_prim_weights:
            return None

        # 2. Build cluster from merged primitives
        cluster = set(merged_prim_weights.keys())

        # 3. Order by weight for primitives_sequence
        ordered = sorted(
            merged_prim_weights.keys(),
            key=lambda p: merged_prim_weights[p],
            reverse=True,
        )

        # 4. Merge situation affinities: weighted average
        merged_affinity: Dict[str, float] = {}
        total_weight = sum(wp.weight for wp in weighted_patterns)
        if total_weight > 0:
            for wp in weighted_patterns:
                for sit, aff in wp.pattern.situation_affinity.items():
                    merged_affinity[sit] = (
                        merged_affinity.get(sit, 0.0)
                        + aff * wp.weight / total_weight
                    )

        # 5. Merge norm features: weighted average
        merged_norms: Dict[str, float] = {}
        if total_weight > 0:
            for wp in weighted_patterns:
                for key, val in wp.pattern.norm_features.items():
                    merged_norms[key] = (
                        merged_norms.get(key, 0.0)
                        + val * wp.weight / total_weight
                    )

        # 6. Build context topology
        all_contexts: Set[str] = set()
        for prim_name in cluster:
            prim = self._library.get(prim_name)
            if prim:
                all_contexts.update(prim.precondition_situations)
        if context:
            all_contexts.add(context)
        topology = self._build_context_topology(cluster, list(all_contexts))

        # 7. Track provenance
        source_fragments = [wp.pattern.name for wp in weighted_patterns]
        sig_parts = sorted(source_fragments)
        composition_signature = "|".join(sig_parts)

        name = f"composite_{'_'.join(ordered[:3])}"
        return ScriptPattern(
            name=name,
            primitives_sequence=ordered,
            precision=0.1,
            situation_affinity=merged_affinity,
            primitive_cluster=cluster,
            context_topology=topology,
            norm_features=merged_norms,
            source_fragments=source_fragments,
            primitive_weights=merged_prim_weights,
            composition_count=1,
            composition_signature=composition_signature,
        )

    def _score_primitive(
        self,
        prim: ScriptPrimitive,
        current_situation: str,
        goal_situation: Optional[str],
        affect_state: Optional[AffectState],
        human_intent: str,
        depth: int,
    ) -> float:
        """Compute EFE-like score for one candidate primitive.

        Lower score = better (consistent with EFE convention: minimise G).
        """
        cfg = self._config

        # G_efficiency: does this primitive advance toward the goal?
        g_eff = 0.0
        if goal_situation and prim.postcondition_situation:
            if prim.postcondition_situation == goal_situation:
                g_eff = -1.0  # Reaches goal directly
            elif prim.postcondition_situation == current_situation:
                g_eff = 0.5  # No progress
            else:
                g_eff = -0.3  # Some progress (different situation)
        else:
            # No goal — prefer shorter sequences
            g_eff = 0.1 * depth

        # G_empathy: prefer primitives that improve affect
        g_emp = 0.0
        if prim.expected_affect and affect_state:
            # Prefer improving valence when it's negative
            if affect_state.valence < 0 and prim.expected_affect.valence > 0:
                g_emp = -0.5  # Good: improves affect
            elif affect_state.valence >= 0 and prim.expected_affect.valence < -0.3:
                g_emp = 0.5  # Bad: worsens affect
        # Also consider human intent
        if human_intent == "avoid" and prim.name in ("approach-greet", "follow-maintain"):
            g_emp += 0.3  # Approaching someone who's avoiding = bad

        # G_epistemic: prefer less-used primitives (information gain)
        usage = self._usage_counts.get(prim.name, 0)
        g_epi = -1.0 / (1.0 + usage)

        return (
            cfg.composition_efe_weight_efficiency * g_eff
            + cfg.composition_efe_weight_empathy * g_emp
            + cfg.composition_efe_weight_epistemic * g_epi
        )

    def _softmax_select(self, scores: Dict[str, float], beta: float = 2.0) -> str:
        """Select from candidates via softmax over negative scores.

        Lower score = better, so we negate before softmax.
        """
        names = list(scores.keys())
        values = np.array([-scores[n] for n in names])
        values -= values.max()
        exp_v = np.exp(beta * values)
        probs = exp_v / exp_v.sum()

        # Deterministic: pick argmax (for reproducibility)
        return names[int(np.argmax(probs))]

    def _build_context_topology(
        self,
        cluster: Set[str],
        contexts: List[str],
    ) -> Dict[str, Dict[Tuple[str, str], float]]:
        """Compute initial context-dependent topology from primitive pre/postconditions.

        For each context, directed pairwise weights encode how naturally
        primitive A transitions to primitive B.  This is the proto-B-matrix
        for weak scripts -- the same cluster gets different topologies in
        different contexts.

        weight(A->B | C) = base * relevance(A,C) * relevance(B,C)
          base = 1.0 if A.postcondition in B.preconditions, else 0.1
          relevance(X,C) = 1.0 if C in X.preconditions, else 0.2
        """
        topology: Dict[str, Dict[Tuple[str, str], float]] = {}

        primitives = [self._library.get(n) for n in cluster]
        primitives = [p for p in primitives if p is not None]

        if len(primitives) < 2:
            return topology

        for context in contexts:
            pairs: Dict[Tuple[str, str], float] = {}
            for a in primitives:
                for b in primitives:
                    if a.name == b.name:
                        continue
                    # Natural transition: A's postcondition meets B's preconditions
                    base = (
                        1.0
                        if a.postcondition_situation in b.precondition_situations
                        else 0.1
                    )
                    # Context relevance of each primitive
                    rel_a = 1.0 if context in a.precondition_situations else 0.2
                    rel_b = 1.0 if context in b.precondition_situations else 0.2
                    pairs[(a.name, b.name)] = base * rel_a * rel_b

            # Normalize to [0, 1]
            if pairs:
                max_w = max(pairs.values())
                if max_w > 0:
                    pairs = {k: v / max_w for k, v in pairs.items()}

            topology[context] = pairs

        return topology

    def reset_usage(self) -> None:
        """Reset usage counts (e.g. at episode boundaries)."""
        self._usage_counts.clear()
