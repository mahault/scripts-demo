"""Tests for the variational inference engine."""

import numpy as np
import pytest

from architecture_core.cognition.planning.generative_model import (
    TaskPhase,
    HandState,
    TargetMode,
    TaskPolicy,
    N_STATES,
    N_POLICIES,
    Observation,
    state_index,
    build_A_matrices,
    build_B_matrices,
    build_C_vectors,
    build_D_vector,
    update_B_from_distances,
)
from architecture_core.cognition.planning.variational_engine import (
    belief_update,
    evaluate_policies,
    select_policy,
)


@pytest.fixture
def model():
    """Full generative model."""
    return {
        "A": build_A_matrices(),
        "B": build_B_matrices(),
        "C": build_C_vectors(empathy_factor=1.0),
        "D": build_D_vector(),
    }


# ── Belief update ─────────────────────────────────────────────

class TestBeliefUpdate:
    def test_posterior_normalised(self, model):
        obs = Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=0, target=0)
        posterior, _ = belief_update(model["D"], obs, model["A"])
        assert np.isclose(posterior.sum(), 1.0)

    def test_posterior_shape(self, model):
        obs = Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=0, target=0)
        posterior, _ = belief_update(model["D"], obs, model["A"])
        assert posterior.shape == (N_STATES,)

    def test_vfe_is_scalar(self, model):
        obs = Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=0, target=0)
        _, vfe = belief_update(model["D"], obs, model["A"])
        assert isinstance(vfe, float)

    def test_consistent_obs_concentrate_belief(self, model):
        """Approach observations should concentrate belief on APPROACH."""
        obs_approach = Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=0, target=0)
        posterior, _ = belief_update(model["D"], obs_approach, model["A"])
        s = state_index(TaskPhase.APPROACH, HandState.EMPTY, TargetMode.FREE)
        assert posterior[s] > 0.5

    def test_sequential_convergence(self, model):
        """After several consistent ticks, beliefs converge strongly."""
        obs = Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=0, target=0)
        q = model["D"].copy()
        for _ in range(5):
            prior = model["B"][int(TaskPolicy.NAV_OBJ)] @ q
            q, _ = belief_update(prior, obs, model["A"])

        s = state_index(TaskPhase.APPROACH, HandState.EMPTY, TargetMode.FREE)
        assert q[s] > 0.9

    def test_at_object_obs_shift_beliefs(self, model):
        """Observing AT + arm proximity shifts beliefs to AT_OBJECT."""
        # First build up APPROACH beliefs
        obs_approach = Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=0, target=0)
        q = model["D"].copy()
        for _ in range(3):
            prior = model["B"][int(TaskPolicy.NAV_OBJ)] @ q
            q, _ = belief_update(prior, obs_approach, model["A"])

        # Now observe AT + arm proximity
        obs_at = Observation(d_obj=2, d_drop=0, arm=1, hold=0, obj_z=0, social=0, target=0)
        prior = model["B"][int(TaskPolicy.NAV_OBJ)] @ q
        q_at, _ = belief_update(prior, obs_at, model["A"])

        s_at = state_index(TaskPhase.AT_OBJECT, HandState.EMPTY, TargetMode.FREE)
        assert q_at[s_at] > 0.3

    def test_holding_obs_shift_to_transport(self, model):
        """Observing hold=YES + obj_z=HELD shifts to TRANSPORT."""
        # Start near AT_OBJECT
        obs_at = Observation(d_obj=2, d_drop=0, arm=1, hold=0, obj_z=0, social=0, target=0)
        q = model["D"].copy()
        for _ in range(5):
            prior = model["B"][int(TaskPolicy.NAV_OBJ)] @ q
            q, _ = belief_update(prior, obs_at, model["A"])

        # Now observe holding
        obs_hold = Observation(d_obj=2, d_drop=0, arm=0, hold=1, obj_z=1, social=0, target=0)
        prior = model["B"][int(TaskPolicy.PICKUP)] @ q
        q_hold, _ = belief_update(prior, obs_hold, model["A"])

        s_transport = state_index(TaskPhase.TRANSPORT, HandState.HOLDING, TargetMode.FREE)
        assert q_hold[s_transport] > 0.3

    def test_normalisation_invariant_random(self, model):
        """Posterior sums to 1 for any random observation."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            obs = Observation(
                d_obj=rng.randint(3), d_drop=rng.randint(3),
                arm=rng.randint(2), hold=rng.randint(2),
                obj_z=rng.randint(3), social=rng.randint(3),
                target=rng.randint(3),
            )
            p, _ = belief_update(model["D"], obs, model["A"])
            assert np.isclose(p.sum(), 1.0, atol=1e-10)

    def test_contested_social_obs(self, model):
        """HIGH social observation shifts belief toward CONTESTED."""
        obs = Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=2, target=0)
        posterior, _ = belief_update(model["D"], obs, model["A"])
        p_contested = sum(
            posterior[state_index(p, h, TargetMode.CONTESTED)]
            for p in TaskPhase for h in HandState
        )
        p_free = sum(
            posterior[state_index(p, h, TargetMode.FREE)]
            for p in TaskPhase for h in HandState
        )
        # CONTESTED should gain relative to its prior (very low initially)
        assert p_contested > 0.1  # rose from ~0.001


# ── EFE evaluation ────────────────────────────────────────────

class TestEFEEvaluation:
    def test_output_shape(self, model):
        G = evaluate_policies(model["D"], model["B"], model["C"], model["A"])
        assert G.shape == (N_POLICIES,)

    def test_nav_obj_preferred_when_approaching(self, model):
        """NAV_OBJ should have lowest G when beliefs are in APPROACH."""
        obs = Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=0, target=0)
        posterior, _ = belief_update(model["D"], obs, model["A"])
        G = evaluate_policies(posterior, model["B"], model["C"], model["A"])
        assert np.argmin(G) == int(TaskPolicy.NAV_OBJ)

    def test_pickup_preferred_at_object(self, model):
        """PICKUP should have lowest G when beliefs are in AT_OBJECT."""
        # Build AT_OBJECT beliefs
        obs_at = Observation(d_obj=2, d_drop=0, arm=1, hold=0, obj_z=0, social=0, target=0)
        q = model["D"].copy()
        for _ in range(5):
            prior = model["B"][int(TaskPolicy.NAV_OBJ)] @ q
            q, _ = belief_update(prior, obs_at, model["A"])

        B_close = update_B_from_distances(model["B"], d_obj=0.3, d_drop=5.0)
        G = evaluate_policies(q, B_close, model["C"], model["A"])
        assert np.argmin(G) == int(TaskPolicy.PICKUP)

    def test_nav_drop_preferred_when_transporting(self, model):
        """NAV_DROP should be preferred when TRANSPORT/HOLDING."""
        obs_hold = Observation(d_obj=2, d_drop=0, arm=0, hold=1, obj_z=1, social=0, target=0)
        # Create transport-concentrated belief
        q = np.full(N_STATES, 1e-6)
        for m in TargetMode:
            q[state_index(TaskPhase.TRANSPORT, HandState.HOLDING, m)] = 0.3
        q /= q.sum()

        prior = model["B"][int(TaskPolicy.NAV_DROP)] @ q
        q_tr, _ = belief_update(prior, obs_hold, model["A"])
        G = evaluate_policies(q_tr, model["B"], model["C"], model["A"])
        assert np.argmin(G) == int(TaskPolicy.NAV_DROP)

    def test_social_G_affects_result(self, model):
        """External social_G should shift EFE values."""
        posterior, _ = belief_update(model["D"],
            Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=0, target=0),
            model["A"])
        G_base = evaluate_policies(posterior, model["B"], model["C"], model["A"])
        social_G = {int(TaskPolicy.NAV_OBJ): 100.0}  # heavily penalise NAV_OBJ
        G_penalised = evaluate_policies(posterior, model["B"], model["C"], model["A"], social_G)
        assert G_penalised[int(TaskPolicy.NAV_OBJ)] > G_base[int(TaskPolicy.NAV_OBJ)]


# ── Policy selection ──────────────────────────────────────────

class TestPolicySelection:
    def test_returns_valid_index(self, model):
        G = evaluate_policies(model["D"], model["B"], model["C"], model["A"])
        selected, q_pi, H_pi = select_policy(G, beta=4.0, arousal=0.5)
        assert 0 <= selected < N_POLICIES

    def test_q_pi_normalised(self, model):
        G = evaluate_policies(model["D"], model["B"], model["C"], model["A"])
        _, q_pi, _ = select_policy(G, beta=4.0, arousal=0.5)
        assert np.isclose(q_pi.sum(), 1.0)

    def test_precision_coupling(self, model):
        """High arousal → higher entropy (more exploration)."""
        G = evaluate_policies(model["D"], model["B"], model["C"], model["A"])
        _, _, H_low = select_policy(G, beta=4.0, arousal=0.1)
        _, _, H_high = select_policy(G, beta=4.0, arousal=0.9)
        assert H_high > H_low

    def test_masking(self, model):
        """Masked policies should get ~0 probability."""
        G = evaluate_policies(model["D"], model["B"], model["C"], model["A"])
        mask = np.ones(N_POLICIES, dtype=bool)
        mask[int(TaskPolicy.NAV_OBJ)] = False
        _, q_pi, _ = select_policy(G, beta=4.0, arousal=0.5, mask=mask)
        assert q_pi[int(TaskPolicy.NAV_OBJ)] < 1e-6

    def test_selected_is_argmax(self, model):
        """Selected policy should be argmax of q(pi)."""
        G = evaluate_policies(model["D"], model["B"], model["C"], model["A"])
        selected, q_pi, _ = select_policy(G, beta=4.0, arousal=0.5)
        assert selected == np.argmax(q_pi)

    def test_uniform_G_gives_high_entropy(self):
        """When all G values are equal, entropy should be maximal."""
        G_uniform = np.zeros(N_POLICIES)
        _, _, H = select_policy(G_uniform, beta=4.0, arousal=0.5)
        H_max = np.log(N_POLICIES)
        assert H > 0.95 * H_max  # near-maximal


# ── Empathy effect ─────────────────────────────────────────────

class TestEmpathyEffect:
    def test_selfish_ignores_social(self):
        A = build_A_matrices()
        B = build_B_matrices()
        C_selfish = build_C_vectors(empathy_factor=0.0)
        D = build_D_vector()

        obs = Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=2, target=0)
        posterior, _ = belief_update(D, obs, A)

        G = evaluate_policies(posterior, B, C_selfish, A)
        # Social HIGH shouldn't penalise any policy
        assert C_selfish["social"][2] == 0.0

    def test_empathic_penalises_contention(self):
        A = build_A_matrices()
        B = build_B_matrices()
        D = build_D_vector()

        obs = Observation(d_obj=0, d_drop=0, arm=0, hold=0, obj_z=0, social=2, target=0)
        posterior, _ = belief_update(D, obs, A)

        C_selfish = build_C_vectors(empathy_factor=0.0)
        C_empathic = build_C_vectors(empathy_factor=1.0)

        G_selfish = evaluate_policies(posterior, B, C_selfish, A)
        G_empathic = evaluate_policies(posterior, B, C_empathic, A)

        # Empathic robot should have different G values for social-sensitive policies
        diff = np.abs(G_empathic - G_selfish)
        assert np.any(diff > 0.01)
