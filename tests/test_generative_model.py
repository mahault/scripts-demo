"""Tests for the factored POMDP generative model."""

import numpy as np
import pytest

from architecture_core.cognition.planning.generative_model import (
    TaskPhase,
    HandState,
    TargetMode,
    TaskPolicy,
    N_STATES,
    N_POLICIES,
    N_PHASES,
    N_HAND,
    N_TARGET,
    OBS_DIMS,
    MODALITY_NAMES,
    Observation,
    state_index,
    state_factors,
    discretize_distance,
    discretize_commitment,
    build_A_matrices,
    build_B_matrices,
    build_C_vectors,
    build_D_vector,
    update_B_from_distances,
)


# ── State indexing ─────────────────────────────────────────────

class TestStateIndexing:
    def test_n_states(self):
        assert N_STATES == 30

    def test_roundtrip(self):
        for i in range(N_STATES):
            p, h, m = state_factors(i)
            assert state_index(p, h, m) == i

    def test_factors_in_range(self):
        for i in range(N_STATES):
            p, h, m = state_factors(i)
            assert 0 <= p < N_PHASES
            assert 0 <= h < N_HAND
            assert 0 <= m < N_TARGET

    def test_unique_indices(self):
        indices = set()
        for p in TaskPhase:
            for h in HandState:
                for m in TargetMode:
                    idx = state_index(p, h, m)
                    assert idx not in indices
                    indices.add(idx)
        assert len(indices) == N_STATES


# ── Discretisation ─────────────────────────────────────────────

class TestDiscretisation:
    @pytest.mark.parametrize("d,expected", [
        (0.0, 2), (0.3, 2), (0.5, 2),   # AT
        (0.6, 1), (1.0, 1), (1.5, 1),   # NEAR
        (1.6, 0), (3.0, 0), (10.0, 0),  # FAR
    ])
    def test_distance(self, d, expected):
        assert discretize_distance(d) == expected

    @pytest.mark.parametrize("belief,expected", [
        (0.0, 0), (0.1, 0), (0.19, 0),  # CLEAR
        (0.2, 1), (0.3, 1), (0.49, 1),  # LOW
        (0.5, 2), (0.7, 2), (1.0, 2),   # HIGH
    ])
    def test_commitment(self, belief, expected):
        assert discretize_commitment(belief) == expected


# ── A matrices ─────────────────────────────────────────────────

class TestAMatrices:
    @pytest.fixture
    def A(self):
        return build_A_matrices()

    def test_all_modalities_present(self, A):
        for name in MODALITY_NAMES:
            assert name in A

    def test_shapes(self, A):
        for name, mat in A.items():
            assert mat.shape == (OBS_DIMS[name], N_STATES)

    def test_columns_normalised(self, A):
        for name, mat in A.items():
            col_sums = mat.sum(axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-10,
                                       err_msg=f"A[{name}] not normalised")

    def test_no_negative_entries(self, A):
        for name, mat in A.items():
            assert np.all(mat >= 0), f"A[{name}] has negative entries"

    def test_hold_near_deterministic(self, A):
        """hold modality should be near-deterministic for hand state."""
        for h in HandState:
            for p in TaskPhase:
                for m in TargetMode:
                    s = state_index(p, h, m)
                    if h == HandState.EMPTY:
                        assert A["hold"][0, s] > 0.9  # P(NO | EMPTY) > 0.9
                    else:
                        assert A["hold"][1, s] > 0.9  # P(YES | HOLDING) > 0.9

    def test_at_object_high_proximity(self, A):
        """AT_OBJECT + EMPTY should have high arm proximity likelihood."""
        for m in TargetMode:
            s = state_index(TaskPhase.AT_OBJECT, HandState.EMPTY, m)
            assert A["arm"][1, s] > 0.8  # P(YES | AT_OBJECT, EMPTY)

    def test_social_depends_on_target_mode(self, A):
        """Social obs should differ by target_mode."""
        s_free = state_index(0, 0, TargetMode.FREE)
        s_contest = state_index(0, 0, TargetMode.CONTESTED)
        # CONTESTED should have higher P(HIGH)
        assert A["social"][2, s_contest] > A["social"][2, s_free]


# ── B matrices ─────────────────────────────────────────────────

class TestBMatrices:
    @pytest.fixture
    def B(self):
        return build_B_matrices()

    def test_all_policies_present(self, B):
        for pi in TaskPolicy:
            assert int(pi) in B

    def test_shapes(self, B):
        for pi, mat in B.items():
            assert mat.shape == (N_STATES, N_STATES)

    def test_columns_normalised(self, B):
        for pi, mat in B.items():
            col_sums = mat.sum(axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-10,
                                       err_msg=f"B[{pi}] not normalised")

    def test_no_negative_entries(self, B):
        for pi, mat in B.items():
            assert np.all(mat >= 0), f"B[{pi}] has negative entries"

    def test_pickup_transitions(self, B):
        """PICKUP from AT_OBJECT/EMPTY should go to TRANSPORT/HOLDING."""
        B_pickup = B[int(TaskPolicy.PICKUP)]
        for m in TargetMode:
            s_from = state_index(TaskPhase.AT_OBJECT, HandState.EMPTY, m)
            s_to = state_index(TaskPhase.TRANSPORT, HandState.HOLDING, m)
            # B[s', s] = P(s' | s, PICKUP)
            assert B_pickup[s_to, s_from] > 0.5

    def test_place_transitions(self, B):
        """PLACE from AT_DROPOFF/HOLDING should go to DONE/EMPTY."""
        B_place = B[int(TaskPolicy.PLACE)]
        for m in TargetMode:
            s_from = state_index(TaskPhase.AT_DROPOFF, HandState.HOLDING, m)
            s_to = state_index(TaskPhase.DONE, HandState.EMPTY, m)
            assert B_place[s_to, s_from] > 0.5

    def test_yield_identity_phase(self, B):
        """YIELD should preserve phase/hand (only target_mode may change)."""
        B_yield = B[int(TaskPolicy.YIELD)]
        for p in TaskPhase:
            for h in HandState:
                # Sum probability over all target modes staying in same phase/hand
                for m in TargetMode:
                    s = state_index(p, h, m)
                    same_ph_mass = sum(
                        B_yield[state_index(p, h, m2), s]
                        for m2 in TargetMode
                    )
                    assert same_ph_mass > 0.9


# ── Distance-conditioned B update ─────────────────────────────

class TestDistanceUpdate:
    def test_close_increases_advance(self):
        B = build_B_matrices()
        B_close = update_B_from_distances(B, d_obj=0.2, d_drop=5.0)
        B_far = update_B_from_distances(B, d_obj=5.0, d_drop=5.0)

        s_from = state_index(TaskPhase.APPROACH, HandState.EMPTY, TargetMode.FREE)
        s_to = state_index(TaskPhase.AT_OBJECT, HandState.EMPTY, TargetMode.FREE)

        p_close = B_close[int(TaskPolicy.NAV_OBJ)][s_to, s_from]
        p_far = B_far[int(TaskPolicy.NAV_OBJ)][s_to, s_from]
        assert p_close > p_far

    def test_columns_still_normalised(self):
        B = build_B_matrices()
        B_updated = update_B_from_distances(B, d_obj=0.3, d_drop=0.3)
        for pi, mat in B_updated.items():
            col_sums = mat.sum(axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_does_not_modify_base(self):
        B = build_B_matrices()
        B_orig = {k: v.copy() for k, v in B.items()}
        _ = update_B_from_distances(B, d_obj=0.1, d_drop=0.1)
        for k in B:
            np.testing.assert_array_equal(B[k], B_orig[k])


# ── C vectors ──────────────────────────────────────────────────

class TestCVectors:
    def test_shapes(self):
        C = build_C_vectors(empathy_factor=1.0)
        for name, vec in C.items():
            assert vec.shape == (OBS_DIMS[name],)

    def test_placed_most_preferred(self):
        C = build_C_vectors()
        assert C["obj_z"][2] > C["obj_z"][0]  # PLACED > TABLE
        assert C["obj_z"][2] > C["obj_z"][1]  # PLACED > HELD

    def test_empathy_scales_social(self):
        C_selfish = build_C_vectors(empathy_factor=0.0)
        C_empathic = build_C_vectors(empathy_factor=1.0)
        # Selfish robot doesn't disprefer HIGH social contention
        assert C_selfish["social"][2] == 0.0
        # Empathic robot does
        assert C_empathic["social"][2] < 0.0


# ── D vector ───────────────────────────────────────────────────

class TestDVector:
    def test_shape(self):
        D = build_D_vector()
        assert D.shape == (N_STATES,)

    def test_normalised(self):
        D = build_D_vector()
        assert np.isclose(D.sum(), 1.0)

    def test_approach_free_dominant(self):
        D = build_D_vector()
        s = state_index(TaskPhase.APPROACH, HandState.EMPTY, TargetMode.FREE)
        assert D[s] > 0.8

    def test_all_positive(self):
        D = build_D_vector()
        assert np.all(D > 0)


# ── Observation dataclass ──────────────────────────────────────

class TestObservation:
    def test_as_dict(self):
        obs = Observation(d_obj=0, d_drop=2, arm=1, hold=0, obj_z=0, social=0, target=0)
        d = obs.as_dict()
        assert len(d) == 8
        assert d["d_obj"] == 0
        assert d["arm"] == 1
