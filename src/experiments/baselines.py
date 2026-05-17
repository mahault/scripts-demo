"""Baseline implementations for comparison against the learned active inference model.

Four baselines that use the same fragment/primitive infrastructure but
with degraded decision-making:

1. Lookup table: context -> fixed behavior sequence (expert-specified)
2. No gating: threshold=0, all fragments always active
3. Random fragments: randomly select k fragments
4. Uniform affinities: all affinities=0.5, normal threshold
"""

from __future__ import annotations

import math
import json
from typing import Dict, List, Optional, Set

import numpy as np

from architecture_core.core.types import PerceptBundle
from architecture_core.cognition.scripts.script_types import SituationType
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    WeightedPattern,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_composer import ScriptComposer
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire


# Expert-specified sequences per context (for lookup table baseline)
EXPERT_SEQUENCES = {
    "reception": [
        "scan-environment",
        "position-in-queue",
        "wait-for-turn",
        "approach-counter",
        "engage-staff",
    ],
    "corridor": [
        "scan-environment",
        "yield-pass",
        "gaze-avert",
    ],
    "hospital": [
        "scan-environment",
        "position-in-queue",
        "wait-for-turn",
        "wait-acknowledge",
        "approach-counter",
    ],
}


class BaselineResult:
    """Result from running a baseline on one condition."""

    def __init__(
        self,
        baseline_name: str,
        context: str,
        active_fragments: List[str],
        sequence: List[str],
        total_vfe: float,
        total_efe: float,
        n_primitives: int,
    ) -> None:
        self.baseline_name = baseline_name
        self.context = context
        self.active_fragments = active_fragments
        self.sequence = sequence
        self.total_vfe = total_vfe
        self.total_efe = total_efe
        self.n_primitives = n_primitives


def _compute_sequence_vfe(topology: dict, sequence: List[str]) -> float:
    """Total VFE of a sequence under a topology."""
    if len(sequence) < 2:
        return 0.0
    vfe = 0.0
    for i in range(len(sequence) - 1):
        w = topology.get((sequence[i], sequence[i + 1]), 0.0)
        vfe += -math.log(max(w, 1e-6))
    return vfe


def _compute_policy_efe(
    topology: dict,
    sequence: List[str],
    lib: PrimitiveLibrary,
    context_affinities: Dict[str, float],
) -> float:
    """Expected free energy of a policy sequence."""
    if not sequence:
        return 0.0
    G = 0.0
    for prim_name in sequence:
        prim = lib.get(prim_name)
        if prim:
            post_sit = prim.postcondition_situation
            p_pref = context_affinities.get(post_sit, 0.1)
            p_pref = max(min(p_pref, 0.999), 0.001)
            risk = -math.log(p_pref)
        else:
            risk = -math.log(0.1)

        outgoing = {}
        for (a, b), w in topology.items():
            if a == prim_name:
                outgoing[b] = w
        if outgoing:
            total = sum(outgoing.values())
            if total > 0:
                probs = [w / total for w in outgoing.values()]
                ambiguity = -sum(p * math.log(max(p, 1e-6)) for p in probs)
            else:
                ambiguity = math.log(max(len(sequence), 1))
        else:
            ambiguity = math.log(max(len(sequence), 1))

        G += risk + ambiguity
    return G


class LookupTableBaseline:
    """Baseline 1: Context -> fixed expert-specified sequence.

    No inference, no gating, no composition. Just a lookup table
    mapping context to a pre-defined behavior sequence.
    """

    def __init__(self, sequences: Optional[Dict[str, List[str]]] = None) -> None:
        self._sequences = sequences or dict(EXPERT_SEQUENCES)

    def run(
        self,
        context: str,
        fragments: Dict[str, ScriptPattern],
        lib: PrimitiveLibrary,
    ) -> BaselineResult:
        """Run the lookup table baseline for one context."""
        sequence = self._sequences.get(context, [])

        # Build topology from all fragments
        topology = self._build_topology(fragments, context)

        # Compute metrics
        vfe = _compute_sequence_vfe(topology, sequence)
        efe = _compute_policy_efe(topology, sequence, lib, {})

        return BaselineResult(
            baseline_name="lookup_table",
            context=context,
            active_fragments=list(fragments.keys()),
            sequence=sequence,
            total_vfe=vfe,
            total_efe=efe,
            n_primitives=len(sequence),
        )

    def _build_topology(
        self, fragments: Dict[str, ScriptPattern], context: str
    ) -> dict:
        """Build topology from all fragment clusters."""
        topology = {}
        all_prims: Set[str] = set()
        for frag in fragments.values():
            all_prims.update(frag.primitive_cluster or set(frag.primitives_sequence))
            topo = frag.context_topology.get(context, {})
            topology.update(topo)

        # Add uniform edges for missing connections
        for a in all_prims:
            for b in all_prims:
                if a != b and (a, b) not in topology:
                    topology[(a, b)] = 0.1
        return topology


class NoGatingBaseline:
    """Baseline 2: Threshold=0, all fragments always active.

    Uses the full composition pipeline but with no context gating.
    Every fragment is included regardless of affinity.
    """

    def run(
        self,
        context: str,
        fragments: Dict[str, ScriptPattern],
        lib: PrimitiveLibrary,
        recognizer: WeakScriptRecognizer,
        pb: PerceptBundle,
    ) -> BaselineResult:
        """Run with all fragments active (no gating)."""
        # All fragments pass (threshold = 0)
        active_fragments = list(fragments.values())

        cfg = RepertoireConfig()
        rep = ScriptRepertoire(lib, cfg, initial_patterns=active_fragments)
        rep.enable_compositional_mode()

        # Use recognizer for context inference but no gating
        belief = recognizer.recognize(pb)
        inferred = belief.most_likely

        # Score and compose
        query_scores = {}
        for name, pat in rep.patterns.items():
            aff = pat.situation_affinity.get(inferred, 0.0)
            query_scores[name] = aff * pat.precision

        weighted = rep.retrieve_composition(query_scores)
        composer = ScriptComposer(lib, cfg)
        composite = composer.compose_from_patterns(weighted, inferred)

        if composite is None:
            return BaselineResult(
                baseline_name="no_gating",
                context=context,
                active_fragments=[f.name for f in active_fragments],
                sequence=[],
                total_vfe=0.0,
                total_efe=0.0,
                n_primitives=0,
            )

        topo = composite.context_topology.get(inferred, {})
        sequence = list(composite.primitives_sequence)
        vfe = _compute_sequence_vfe(topo, sequence)
        efe = _compute_policy_efe(topo, sequence, lib, {})

        return BaselineResult(
            baseline_name="no_gating",
            context=context,
            active_fragments=[f.name for f in active_fragments],
            sequence=sequence,
            total_vfe=vfe,
            total_efe=efe,
            n_primitives=len(sequence),
        )


class RandomFragmentsBaseline:
    """Baseline 3: Randomly select k fragments.

    k is matched to the learned model's average number of active fragments
    for this context.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        context: str,
        fragments: Dict[str, ScriptPattern],
        lib: PrimitiveLibrary,
        k: int = 3,
        recognizer: Optional[WeakScriptRecognizer] = None,
        pb: Optional[PerceptBundle] = None,
    ) -> BaselineResult:
        """Run with k randomly selected fragments."""
        frag_names = list(fragments.keys())
        k_actual = min(k, len(frag_names))

        selected_names = list(self._rng.choice(
            frag_names, size=k_actual, replace=False
        ))
        selected = [fragments[name] for name in selected_names]

        cfg = RepertoireConfig()
        rep = ScriptRepertoire(lib, cfg, initial_patterns=selected)
        rep.enable_compositional_mode()

        # Infer context if recognizer available
        inferred = context
        if recognizer and pb:
            belief = recognizer.recognize(pb)
            inferred = belief.most_likely

        query_scores = {}
        for name, pat in rep.patterns.items():
            aff = pat.situation_affinity.get(inferred, 0.0)
            query_scores[name] = aff * pat.precision

        weighted = rep.retrieve_composition(query_scores)
        composer = ScriptComposer(lib, cfg)
        composite = composer.compose_from_patterns(weighted, inferred)

        if composite is None:
            return BaselineResult(
                baseline_name="random_fragments",
                context=context,
                active_fragments=selected_names,
                sequence=[],
                total_vfe=0.0,
                total_efe=0.0,
                n_primitives=0,
            )

        topo = composite.context_topology.get(inferred, {})
        sequence = list(composite.primitives_sequence)
        vfe = _compute_sequence_vfe(topo, sequence)
        efe = _compute_policy_efe(topo, sequence, lib, {})

        return BaselineResult(
            baseline_name="random_fragments",
            context=context,
            active_fragments=selected_names,
            sequence=sequence,
            total_vfe=vfe,
            total_efe=efe,
            n_primitives=len(sequence),
        )


class UniformAffinitiesBaseline:
    """Baseline 4: All affinities set to 0.5, normal threshold.

    Tests whether context-specific affinities matter by removing
    all context preference from the D-matrix while keeping the
    gating mechanism intact.
    """

    def run(
        self,
        context: str,
        fragments: Dict[str, ScriptPattern],
        lib: PrimitiveLibrary,
        threshold: float = 0.4,
        recognizer: Optional[WeakScriptRecognizer] = None,
        pb: Optional[PerceptBundle] = None,
    ) -> BaselineResult:
        """Run with uniform affinities (0.5 for all fragment-context pairs)."""
        # Create fragments with uniform affinities
        uniform_fragments = []
        for frag in fragments.values():
            uniform_frag = ScriptPattern(
                name=frag.name,
                primitives_sequence=list(frag.primitives_sequence),
                primitive_cluster=set(frag.primitive_cluster) if frag.primitive_cluster else None,
                precision=frag.precision,
                situation_affinity={ctx: 0.5 for ctx in frag.situation_affinity},
            )
            uniform_fragments.append(uniform_frag)

        # Apply threshold gating with uniform affinities
        # With threshold=0.4 and affinity=0.5, all fragments pass
        active = [f for f in uniform_fragments
                  if f.situation_affinity.get(context, 0.5) >= threshold]

        if not active:
            return BaselineResult(
                baseline_name="uniform_affinities",
                context=context,
                active_fragments=[],
                sequence=[],
                total_vfe=0.0,
                total_efe=0.0,
                n_primitives=0,
            )

        cfg = RepertoireConfig()
        rep = ScriptRepertoire(lib, cfg, initial_patterns=active)
        rep.enable_compositional_mode()

        inferred = context
        if recognizer and pb:
            belief = recognizer.recognize(pb)
            inferred = belief.most_likely

        query_scores = {}
        for name, pat in rep.patterns.items():
            aff = pat.situation_affinity.get(inferred, 0.0)
            query_scores[name] = aff * pat.precision

        weighted = rep.retrieve_composition(query_scores)
        composer = ScriptComposer(lib, cfg)
        composite = composer.compose_from_patterns(weighted, inferred)

        if composite is None:
            return BaselineResult(
                baseline_name="uniform_affinities",
                context=context,
                active_fragments=[f.name for f in active],
                sequence=[],
                total_vfe=0.0,
                total_efe=0.0,
                n_primitives=0,
            )

        topo = composite.context_topology.get(inferred, {})
        sequence = list(composite.primitives_sequence)
        vfe = _compute_sequence_vfe(topo, sequence)
        efe = _compute_policy_efe(topo, sequence, lib, {})

        return BaselineResult(
            baseline_name="uniform_affinities",
            context=context,
            active_fragments=[f.name for f in active],
            sequence=sequence,
            total_vfe=vfe,
            total_efe=efe,
            n_primitives=len(sequence),
        )


def run_all_baselines(
    context: str,
    fragments: Dict[str, ScriptPattern],
    lib: PrimitiveLibrary,
    recognizer: WeakScriptRecognizer,
    pb: PerceptBundle,
    k_random: int = 3,
    threshold: float = 0.4,
    seed: Optional[int] = None,
) -> List[BaselineResult]:
    """Run all four baselines for one condition.

    Parameters
    ----------
    context : str
        Ground-truth context.
    fragments : dict
        All available fragments.
    lib : PrimitiveLibrary
        Primitive library.
    recognizer : WeakScriptRecognizer
        Situation recognizer instance.
    pb : PerceptBundle
        Perception input.
    k_random : int
        Number of fragments for random baseline.
    threshold : float
        Gating threshold for uniform baseline.
    seed : int | None
        Random seed for random baseline.

    Returns
    -------
    List of 4 BaselineResults.
    """
    results = []

    # Baseline 1: Lookup table
    lookup = LookupTableBaseline()
    results.append(lookup.run(context, fragments, lib))

    # Baseline 2: No gating
    no_gate = NoGatingBaseline()
    results.append(no_gate.run(context, fragments, lib, recognizer, pb))

    # Baseline 3: Random fragments
    random_bl = RandomFragmentsBaseline(seed=seed)
    results.append(random_bl.run(context, fragments, lib, k_random, recognizer, pb))

    # Baseline 4: Uniform affinities
    uniform = UniformAffinitiesBaseline()
    results.append(uniform.run(context, fragments, lib, threshold, recognizer, pb))

    return results
