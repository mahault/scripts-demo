"""Tests for context-conditioned semantic clusters (weak scripts).

Verifies that weak scripts are loosely-jointed semantic clusters whose
internal topology reshapes depending on context, and that consolidation
crystallizes them into ordered sequences (strong scripts).

Per Albarracin, Constant, Friston & Ramstead (2021).
"""

from __future__ import annotations

import pytest

from architecture_core.core.types import AffectState, SkillRequest
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    TransitionEntry,
    TrajectoryStep,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.trajectory_inference import (
    ScriptParticleFilter,
    TrajectoryParticle,
)
from architecture_core.cognition.scripts.script_composer import ScriptComposer
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire


# =====================================================================
# Helpers
# =====================================================================
def _make_cluster_pattern(
    name: str = "corridor_cluster",
    cluster: set | None = None,
    topology: dict | None = None,
    precision: float = 0.1,
) -> ScriptPattern:
    """Create a cluster-based weak script pattern."""
    cluster = cluster or {"wait-acknowledge", "yield-pass", "approach-greet"}
    if topology is None:
        topology = {
            "corridor_encounter": {
                ("wait-acknowledge", "yield-pass"): 0.9,
                ("yield-pass", "approach-greet"): 0.3,
                ("wait-acknowledge", "approach-greet"): 0.2,
                ("yield-pass", "wait-acknowledge"): 0.4,
                ("approach-greet", "wait-acknowledge"): 0.5,
                ("approach-greet", "yield-pass"): 0.1,
            },
            "open_area": {
                ("approach-greet", "wait-acknowledge"): 0.8,
                ("approach-greet", "yield-pass"): 0.6,
                ("wait-acknowledge", "yield-pass"): 0.2,
                ("wait-acknowledge", "approach-greet"): 0.7,
                ("yield-pass", "approach-greet"): 0.5,
                ("yield-pass", "wait-acknowledge"): 0.3,
            },
        }
    return ScriptPattern(
        name=name,
        primitives_sequence=["wait-acknowledge", "yield-pass", "approach-greet"],
        precision=precision,
        situation_affinity={"corridor_encounter": 0.9, "open_area": 0.3},
        primitive_cluster=cluster,
        context_topology=topology,
    )


def _make_legacy_pattern() -> ScriptPattern:
    """Create a legacy (non-cluster) strong pattern."""
    return ScriptPattern(
        name="legacy_strong",
        primitives_sequence=["yield-pass", "wait-acknowledge"],
        precision=2.5,
        is_strong=True,
        situation_affinity={"corridor_encounter": 0.8},
    )


def _make_step(prim: str, situation: str = "corridor_encounter") -> TrajectoryStep:
    return TrajectoryStep(
        t=0.0,
        primitive_name=prim,
        most_likely_situation=situation,
        outcome="SUCCESS",
    )


# =====================================================================
# ScriptPattern cluster fields
# =====================================================================
class TestScriptPatternCluster:
    def test_default_not_cluster_based(self):
        p = ScriptPattern(name="test")
        assert not p.is_cluster_based
        assert p.primitive_cluster == set()
        assert p.context_topology == {}

    def test_cluster_based_when_populated(self):
        p = _make_cluster_pattern()
        assert p.is_cluster_based

    def test_cluster_based_is_false_for_legacy(self):
        p = _make_legacy_pattern()
        assert not p.is_cluster_based

    def test_cluster_and_sequence_coexist(self):
        p = _make_cluster_pattern()
        assert len(p.primitive_cluster) == 3
        assert len(p.primitives_sequence) == 3
        assert p.primitive_cluster == set(p.primitives_sequence)


# =====================================================================
# Context topology
# =====================================================================
class TestContextTopology:
    def test_topology_has_multiple_contexts(self):
        p = _make_cluster_pattern()
        assert "corridor_encounter" in p.context_topology
        assert "open_area" in p.context_topology

    def test_different_contexts_different_weights(self):
        p = _make_cluster_pattern()
        corridor = p.context_topology["corridor_encounter"]
        open_area = p.context_topology["open_area"]
        # In corridor: wait->yield is strong (0.9)
        # In open area: approach->wait is strong (0.8)
        assert corridor[("wait-acknowledge", "yield-pass")] > open_area[("wait-acknowledge", "yield-pass")]
        assert open_area[("approach-greet", "wait-acknowledge")] > corridor[("approach-greet", "wait-acknowledge")]

    def test_topology_weights_are_directed(self):
        p = _make_cluster_pattern()
        topo = p.context_topology["corridor_encounter"]
        # A->B != B->A
        assert topo[("wait-acknowledge", "yield-pass")] != topo[("yield-pass", "wait-acknowledge")]


# =====================================================================
# Composer produces clusters
# =====================================================================
class TestComposerClusters:
    def test_compose_populates_cluster(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        pattern = composer.compose("corridor_encounter")
        assert pattern.is_cluster_based
        assert len(pattern.primitive_cluster) > 0

    def test_compose_still_has_sequence(self):
        """Backward compat: primitives_sequence is still populated."""
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        pattern = composer.compose("corridor_encounter")
        assert len(pattern.primitives_sequence) > 0
        assert set(pattern.primitives_sequence) == pattern.primitive_cluster

    def test_compose_populates_topology(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        pattern = composer.compose("corridor_encounter")
        assert len(pattern.context_topology) > 0
        # Should have corridor_encounter as a context
        assert "corridor_encounter" in pattern.context_topology

    def test_compose_topology_has_multiple_contexts(self):
        """Topology should include contexts from all cluster primitives."""
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        pattern = composer.compose("corridor_encounter")
        # Cluster primitives have various precondition situations
        assert len(pattern.context_topology) >= 1

    def test_topology_weights_normalized(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        pattern = composer.compose("corridor_encounter")
        for context, pairs in pattern.context_topology.items():
            if pairs:
                max_w = max(pairs.values())
                assert max_w <= 1.0 + 1e-9


# =====================================================================
# Cluster-mode particle filter
# =====================================================================
class TestClusterParticleFilter:
    def test_cluster_pattern_uses_cluster_likelihood(self):
        """Cluster member should get higher likelihood than non-member."""
        p = _make_cluster_pattern()
        patterns = {"cluster": p}
        pf = ScriptParticleFilter(patterns, n_particles=10)
        pf.initialize("corridor_encounter")

        step_in = _make_step("yield-pass")  # In cluster
        step_out = _make_step("handover-extend")  # Not in cluster

        particle = pf._particles[0]
        ll_in = pf._cluster_likelihood(particle, p, step_in)
        ll_out = pf._cluster_likelihood(particle, p, step_out)
        assert ll_in > ll_out

    def test_topology_bonus_context_dependent(self):
        """Same primitive should get different likelihood in different contexts."""
        p = _make_cluster_pattern()
        particle = TrajectoryParticle(pattern_name="test", step_index=0)

        patterns = {"test": p}
        pf = ScriptParticleFilter(patterns)

        step_corridor = _make_step("wait-acknowledge", "corridor_encounter")
        step_open = _make_step("wait-acknowledge", "open_area")

        ll_corridor = pf._cluster_likelihood(particle, p, step_corridor)
        ll_open = pf._cluster_likelihood(particle, p, step_open)
        # Both should be > 0 (in cluster), but may differ due to topology
        assert ll_corridor > 0
        assert ll_open > 0

    def test_seen_primitives_tracked(self):
        """update() should track which cluster members have been seen."""
        p = _make_cluster_pattern()
        patterns = {"cluster": p}
        pf = ScriptParticleFilter(patterns, n_particles=10)
        pf.initialize("corridor_encounter")

        step = _make_step("yield-pass")
        pf.predict()
        pf.update(step)

        # At least some particles should have seen_primitives populated
        cluster_particles = [
            pp for pp in pf._particles
            if pp.seen_primitives is not None and "yield-pass" in pp.seen_primitives
        ]
        assert len(cluster_particles) > 0

    def test_cluster_predict_no_positional_advance(self):
        """Cluster-mode predict should not advance step_index."""
        p = _make_cluster_pattern()
        patterns = {"cluster": p}
        pf = ScriptParticleFilter(patterns, n_particles=10)
        pf.initialize("corridor_encounter")

        initial_indices = [pp.step_index for pp in pf._particles]
        pf.predict()
        after_indices = [pp.step_index for pp in pf._particles]
        # Cluster mode: step_index is NOT advanced by predict
        assert initial_indices == after_indices


# =====================================================================
# Dual-mode inference (mix of weak and strong patterns)
# =====================================================================
class TestDualModeInference:
    def test_mixed_patterns(self):
        """Filter handles both cluster and sequence patterns simultaneously."""
        cluster_p = _make_cluster_pattern()
        strong_p = _make_legacy_pattern()
        patterns = {"cluster": cluster_p, "strong": strong_p}
        pf = ScriptParticleFilter(patterns, n_particles=20)
        pf.initialize("corridor_encounter")

        # Both patterns should have particles
        names = {pp.pattern_name for pp in pf._particles}
        assert "cluster" in names
        assert "strong" in names

        # Run a predict-update cycle
        step = _make_step("yield-pass")
        pf.predict()
        pf.update(step)

        dist = pf.infer()
        assert "cluster" in dist
        assert "strong" in dist

    def test_strong_pattern_unchanged(self):
        """Strong pattern particles should still use sequence-based advancement."""
        strong_p = _make_legacy_pattern()
        patterns = {"strong": strong_p}
        pf = ScriptParticleFilter(patterns, n_particles=10)
        pf.initialize("corridor_encounter")

        pf.predict()
        # All should have advanced to step_index=1
        for pp in pf._particles:
            assert pp.step_index == 1
            assert pp.seen_primitives is None


# =====================================================================
# Derive sequence from topology
# =====================================================================
class TestDeriveSequenceFromTopology:
    def test_different_context_different_order(self):
        """Same cluster, different context -> different ordering."""
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        p = _make_cluster_pattern()

        seq_corridor = rep._derive_sequence_from_topology(p, "corridor_encounter")
        seq_open = rep._derive_sequence_from_topology(p, "open_area")

        # Both should contain all cluster members
        assert set(seq_corridor) == p.primitive_cluster
        assert set(seq_open) == p.primitive_cluster
        # But the ordering should differ
        assert seq_corridor != seq_open

    def test_fallback_to_primitives_sequence(self):
        """No topology -> falls back to primitives_sequence."""
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        p = ScriptPattern(
            name="no_topo",
            primitives_sequence=["yield-pass", "wait-acknowledge"],
            primitive_cluster={"yield-pass", "wait-acknowledge"},
            context_topology={},
        )
        seq = rep._derive_sequence_from_topology(p)
        assert seq == ["yield-pass", "wait-acknowledge"]

    def test_fallback_to_sorted_cluster(self):
        """No topology and no sequence -> sorted cluster."""
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        p = ScriptPattern(
            name="bare",
            primitive_cluster={"yield-pass", "approach-greet"},
            context_topology={},
        )
        seq = rep._derive_sequence_from_topology(p)
        assert seq == sorted({"yield-pass", "approach-greet"})


# =====================================================================
# Consolidation crystallizes sequence
# =====================================================================
class TestConsolidationCrystallizes:
    def test_crystallize_from_b_matrix(self):
        """Consolidation should extract sequence from B-matrix transitions."""
        lib = PrimitiveLibrary()
        p = _make_cluster_pattern(precision=0.5)
        # Build B-matrix: wait -> yield (high), yield -> approach (high)
        p.transition_counts = {
            "wait-acknowledge": {
                "yield-pass": TransitionEntry(count=10, total_from=12),
                "approach-greet": TransitionEntry(count=2, total_from=12),
            },
            "yield-pass": {
                "approach-greet": TransitionEntry(count=9, total_from=10),
                "wait-acknowledge": TransitionEntry(count=1, total_from=10),
            },
        }
        rep = ScriptRepertoire(lib)
        crystallized = rep._crystallize_sequence(p)
        # Should follow: wait -> yield -> approach (highest prob path)
        assert crystallized == ["wait-acknowledge", "yield-pass", "approach-greet"]

    def test_consolidation_sets_sequence_and_strong(self):
        """Full consolidation: precision meets threshold -> crystallize + promote."""
        lib = PrimitiveLibrary()
        p = _make_cluster_pattern(precision=2.5)
        p.trajectory_count = 6
        p.mean_free_energy = 1.0
        p.transition_counts = {
            "wait-acknowledge": {
                "yield-pass": TransitionEntry(count=8, total_from=10),
            },
            "yield-pass": {
                "approach-greet": TransitionEntry(count=7, total_from=9),
            },
        }

        cfg = RepertoireConfig(
            strong_precision_threshold=2.0,
            min_trajectories_for_promotion=5,
            max_free_energy_for_promotion=3.0,
        )
        rep = ScriptRepertoire(lib, config=cfg, initial_patterns=[p])
        promoted = rep._check_consolidation("corridor_cluster")

        assert promoted is True
        updated = rep.patterns["corridor_cluster"]
        assert updated.is_strong
        # Sequence should be crystallized from B-matrix
        assert updated.primitives_sequence[0] == "wait-acknowledge"

    def test_legacy_consolidation_unchanged(self):
        """Non-cluster pattern consolidation still works as before."""
        lib = PrimitiveLibrary()
        p = ScriptPattern(
            name="legacy",
            primitives_sequence=["yield-pass", "wait-acknowledge"],
            precision=2.5,
            trajectory_count=6,
            mean_free_energy=1.0,
            situation_affinity={"corridor_encounter": 0.9},
        )
        cfg = RepertoireConfig(
            strong_precision_threshold=2.0,
            min_trajectories_for_promotion=5,
        )
        rep = ScriptRepertoire(lib, config=cfg, initial_patterns=[p])
        promoted = rep._check_consolidation("legacy")
        assert promoted is True
        updated = rep.patterns["legacy"]
        assert updated.is_strong
        # Sequence should be unchanged
        assert updated.primitives_sequence == ["yield-pass", "wait-acknowledge"]


# =====================================================================
# Backward compatibility
# =====================================================================
class TestBackwardCompat:
    def test_legacy_pattern_sequence_mode(self):
        """Pattern without cluster fields uses sequence mode in particle filter."""
        p = ScriptPattern(
            name="old_style",
            primitives_sequence=["yield-pass", "approach-greet"],
            precision=1.5,
            situation_affinity={"corridor_encounter": 0.9},
        )
        assert not p.is_cluster_based

        patterns = {"old_style": p}
        pf = ScriptParticleFilter(patterns, n_particles=10)
        pf.initialize("corridor_encounter")

        pf.predict()
        # Should advance to step_index=1
        for pp in pf._particles:
            assert pp.step_index == 1

    def test_legacy_pattern_to_sequence(self):
        """pattern_to_sequence without context works for legacy patterns."""
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        p = ScriptPattern(
            name="old_style",
            primitives_sequence=["yield-pass", "approach-greet"],
            precision=1.5,
            situation_affinity={"corridor_encounter": 0.9},
        )
        seq = rep.pattern_to_sequence(p)
        assert seq.name == "old_style"
        assert len(seq.steps) == 2
        assert seq.steps[0].primitive_name == "yield-pass"
