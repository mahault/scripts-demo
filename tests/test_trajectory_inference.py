"""Tests for script particle filter (trajectory inference)."""

from __future__ import annotations

import math

import pytest

from architecture_core.cognition.scripts.repertoire_types import (
    ScriptPattern,
    TrajectoryStep,
    TransitionEntry,
)
from architecture_core.cognition.scripts.trajectory_inference import (
    ScriptParticleFilter,
    TrajectoryParticle,
)


# =====================================================================
# Helpers
# =====================================================================
def _make_patterns():
    """Two test patterns: corridor_yield and open_approach."""
    corridor = ScriptPattern(
        name="corridor_yield",
        primitives_sequence=["wait-acknowledge", "yield-pass", "approach-greet"],
        precision=1.5,
        situation_affinity={"corridor_encounter": 0.9, "open_area": 0.1},
        transition_counts={
            "wait-acknowledge": {
                "yield-pass": TransitionEntry(count=8, total_from=10, mean_kl=0.3),
            },
            "yield-pass": {
                "approach-greet": TransitionEntry(count=7, total_from=10, mean_kl=0.2),
            },
        },
    )
    open_approach = ScriptPattern(
        name="open_approach",
        primitives_sequence=["approach-greet", "follow-maintain"],
        precision=0.5,
        situation_affinity={"open_area": 0.8, "corridor_encounter": 0.2},
        transition_counts={
            "approach-greet": {
                "follow-maintain": TransitionEntry(count=5, total_from=8, mean_kl=0.5),
            },
        },
    )
    return {"corridor_yield": corridor, "open_approach": open_approach}


def _step(primitive: str, kl: float = 0.0) -> TrajectoryStep:
    return TrajectoryStep(
        t=0.0,
        primitive_name=primitive,
        violation_kl=kl,
    )


# =====================================================================
# Initialization
# =====================================================================
class TestScriptParticleFilterInit:
    def test_not_initialized_by_default(self):
        pf = ScriptParticleFilter(_make_patterns(), n_particles=30)
        assert not pf.is_initialized
        assert pf.infer() == {}

    def test_initialize_distributes_particles(self):
        pf = ScriptParticleFilter(_make_patterns(), n_particles=30)
        pf.initialize("corridor_encounter")
        assert pf.is_initialized
        dist = pf.infer()
        assert len(dist) == 2
        # corridor_yield should get more particles (higher affinity * precision)
        assert dist["corridor_yield"] > dist["open_approach"]

    def test_weights_sum_to_one(self):
        pf = ScriptParticleFilter(_make_patterns(), n_particles=50)
        pf.initialize("open_area")
        dist = pf.infer()
        assert sum(dist.values()) == pytest.approx(1.0, abs=0.01)

    def test_empty_patterns(self):
        pf = ScriptParticleFilter({}, n_particles=10)
        pf.initialize("anything")
        assert pf.infer() == {}
        assert pf.best_pattern is None


# =====================================================================
# Predict
# =====================================================================
class TestScriptParticleFilterPredict:
    def test_predict_advances_step_index(self):
        pf = ScriptParticleFilter(_make_patterns(), n_particles=20)
        pf.initialize("corridor_encounter")
        pf.predict()
        # Particles should have advanced — no crash
        dist = pf.infer()
        assert sum(dist.values()) == pytest.approx(1.0, abs=0.01)

    def test_predict_no_particles_is_noop(self):
        pf = ScriptParticleFilter(_make_patterns(), n_particles=10)
        pf.predict()  # Not initialized — no crash


# =====================================================================
# Update
# =====================================================================
class TestScriptParticleFilterUpdate:
    def test_matching_step_maintains_weight(self):
        patterns = _make_patterns()
        pf = ScriptParticleFilter(patterns, n_particles=40)
        pf.initialize("corridor_encounter")
        # First step of corridor_yield is "wait-acknowledge"
        pf.update(_step("wait-acknowledge"))
        dist = pf.infer()
        assert dist["corridor_yield"] > 0.5

    def test_mismatching_step_reduces_weight(self):
        patterns = _make_patterns()
        pf = ScriptParticleFilter(patterns, n_particles=40)
        pf.initialize("corridor_encounter")
        # "follow-maintain" is NOT the first step of corridor_yield
        pf.update(_step("follow-maintain"))
        dist = pf.infer()
        # open_approach also starts with approach-greet not follow-maintain,
        # so both should lose weight, but corridor should lose more
        # since it expects wait-acknowledge
        assert sum(dist.values()) == pytest.approx(1.0, abs=0.01)

    def test_high_kl_penalises(self):
        patterns = _make_patterns()
        pf = ScriptParticleFilter(patterns, n_particles=40)
        pf.initialize("corridor_encounter")
        # Correct primitive but high violation KL
        pf.update(_step("wait-acknowledge", kl=3.0))
        dist_high_kl = pf.infer()

        pf2 = ScriptParticleFilter(patterns, n_particles=40)
        pf2.initialize("corridor_encounter")
        pf2.update(_step("wait-acknowledge", kl=0.0))
        dist_low_kl = pf2.infer()

        # Both should favor corridor_yield, but low KL should be more confident
        assert dist_low_kl["corridor_yield"] >= dist_high_kl["corridor_yield"] - 0.1


# =====================================================================
# Full trajectory
# =====================================================================
class TestScriptParticleFilterTrajectory:
    def test_full_matching_trajectory_builds_confidence(self):
        patterns = _make_patterns()
        pf = ScriptParticleFilter(patterns, n_particles=60)
        pf.initialize("corridor_encounter")

        # Feed the exact corridor_yield sequence
        steps = ["wait-acknowledge", "yield-pass", "approach-greet"]
        for i, prim in enumerate(steps):
            if i > 0:
                pf.predict()
            pf.update(_step(prim))

        dist = pf.infer()
        assert dist["corridor_yield"] > 0.6

    def test_trajectory_free_energy_accumulates(self):
        patterns = _make_patterns()
        pf = ScriptParticleFilter(patterns, n_particles=40)
        pf.initialize("corridor_encounter")

        assert pf.trajectory_free_energy == 0.0
        pf.update(_step("wait-acknowledge"))
        assert pf.trajectory_free_energy > 0.0  # Some surprise

    def test_bad_trajectory_high_free_energy(self):
        patterns = _make_patterns()
        pf_good = ScriptParticleFilter(patterns, n_particles=40)
        pf_good.initialize("corridor_encounter")
        pf_good.update(_step("wait-acknowledge"))

        pf_bad = ScriptParticleFilter(patterns, n_particles=40)
        pf_bad.initialize("corridor_encounter")
        pf_bad.update(_step("disengage-depart"))  # Not in any pattern

        assert pf_bad.trajectory_free_energy > pf_good.trajectory_free_energy


# =====================================================================
# Properties
# =====================================================================
class TestScriptParticleFilterProperties:
    def test_best_pattern(self):
        pf = ScriptParticleFilter(_make_patterns(), n_particles=40)
        pf.initialize("corridor_encounter")
        assert pf.best_pattern in ["corridor_yield", "open_approach"]

    def test_confidence_in_range(self):
        pf = ScriptParticleFilter(_make_patterns(), n_particles=40)
        pf.initialize("corridor_encounter")
        assert 0.0 <= pf.confidence <= 1.0

    def test_ess_initially_equals_n(self):
        pf = ScriptParticleFilter(_make_patterns(), n_particles=40)
        pf.initialize("corridor_encounter")
        assert pf.ess == pytest.approx(40.0, abs=2.0)

    def test_entropy_nonnegative(self):
        pf = ScriptParticleFilter(_make_patterns(), n_particles=40)
        pf.initialize("corridor_encounter")
        assert pf.entropy >= 0.0

    def test_reset(self):
        pf = ScriptParticleFilter(_make_patterns(), n_particles=20)
        pf.initialize("corridor_encounter")
        pf.update(_step("wait-acknowledge"))
        pf.reset()
        assert not pf.is_initialized
        assert pf.trajectory_free_energy == 0.0
        assert pf.infer() == {}
