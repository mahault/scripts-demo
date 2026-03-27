"""Variational inference engine for the task POMDP.

Pure-math module: belief update (VFE minimisation via exact Bayes) and
policy evaluation (EFE computation).  No side effects, no mutable state.

Policy evaluation:
  G(pi) = G_pragmatic + G_epistemic + G_social

Policy posterior (Da Costa 2020):
  q(pi) = sigma(-gamma * G(pi) + ln E)
where E is the habit prior (log-prior over policies).

All functions operate on numpy arrays produced by generative_model.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np

from architecture_core.cognition.planning.generative_model import (
    MODALITY_NAMES,
    N_POLICIES,
    N_STATES,
    Observation,
)

_EPS = 1e-16


# ── Belief update (VFE minimisation) ──────────────────────────

def belief_update(
    prior: np.ndarray,
    observations: Union[Observation, Dict[str, int]],
    A: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, float]:
    """Exact Bayesian belief update with VFE decomposition.

    Parameters
    ----------
    prior : (N_STATES,) predicted state distribution P(s_t | s_{t-1}, pi)
    observations : discretised observation vector (8 modalities)
    A : observation likelihood matrices {modality: (n_obs, N_STATES)}

    Returns
    -------
    posterior : (N_STATES,) normalised posterior q(s)
    vfe : scalar variational free energy F = F_accuracy + F_complexity
    """
    obs = observations.as_dict() if isinstance(observations, Observation) else observations
    prior_safe = np.clip(prior, _EPS, None)
    prior_safe /= prior_safe.sum()

    # Accumulate log-likelihood across modalities: ln P(o | s) = Σ_m ln A[m][o_m, s]
    log_lik = np.zeros(N_STATES)
    for m_name in MODALITY_NAMES:
        o_val = obs[m_name]
        log_lik += np.log(np.clip(A[m_name][o_val, :], _EPS, None))

    # Log-space posterior for numerical stability
    log_joint = np.log(prior_safe) + log_lik
    log_joint -= log_joint.max()
    joint = np.exp(log_joint)
    evidence = joint.sum()
    posterior = joint / max(evidence, _EPS)

    # VFE = F_accuracy + F_complexity
    #   F_accuracy = -E_q[ln P(o|s)]  (reconstruction error)
    #   F_complexity = KL[q(s) || prior(s)]
    posterior_safe = np.clip(posterior, _EPS, None)
    F_accuracy = -np.dot(posterior_safe, log_lik)
    F_complexity = np.dot(
        posterior_safe,
        np.log(posterior_safe) - np.log(prior_safe),
    )
    vfe = float(F_accuracy + F_complexity)

    return posterior, vfe


# ── KL divergence helper ──────────────────────────────────────

def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL[p || q] with numerical safety."""
    p_safe = np.clip(p, _EPS, None)
    p_safe /= p_safe.sum()
    q_safe = np.clip(q, _EPS, None)
    q_safe /= q_safe.sum()
    return float(np.dot(p_safe, np.log(p_safe) - np.log(q_safe)))


# ── Single-step EFE helper ────────────────────────────────────

def _single_step_efe(
    q_next: np.ndarray,
    C: Dict[str, np.ndarray],
    A: Dict[str, np.ndarray],
    social_G: Optional[Dict[int, float]],
    pi: int,
) -> float:
    """Compute G_pi (pragmatic + epistemic + social) for one policy.

    Does NOT include F_pi — caller adds it separately.
    """
    g_prag = 0.0
    g_epist = 0.0

    for m_name in MODALITY_NAMES:
        A_m = A[m_name]
        C_m = C[m_name]

        # Predicted observation distribution: P(o | pi) = A @ q_next
        q_o = A_m @ q_next
        q_o = np.clip(q_o, _EPS, None)
        q_o /= q_o.sum()

        # Pragmatic: -E[ln P(o|C)] = -Σ_o q(o|pi) * C[o]
        g_prag += -np.dot(q_o, C_m)

        # Epistemic: H_cond - H_marg = -I(o; s | pi)
        # H_marg = H[P(o|pi)]
        H_marg = -np.dot(q_o, np.log(q_o))

        # H_cond = E_s'[H[P(o|s')]] = Σ_s' q_next[s'] * H[A[:,s']]
        A_safe = np.clip(A_m, _EPS, None)
        H_per_state = -np.sum(A_safe * np.log(A_safe), axis=0)  # (N_STATES,)
        H_cond = np.dot(q_next, H_per_state)

        g_epist += H_cond - H_marg  # negative MI

    # Social EFE (from SocialEFE for task-relevant policies)
    g_social = 0.0
    if social_G is not None and pi in social_G:
        g_social = social_G[pi]

    return g_prag + g_epist + g_social


# ── EFE policy evaluation ─────────────────────────────────────

def evaluate_policies(
    posterior: np.ndarray,
    B: Dict[int, np.ndarray],
    C: Dict[str, np.ndarray],
    A: Dict[str, np.ndarray],
    social_G: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Compute expected free energy G(pi) for each policy.

    G(pi) = G_pragmatic + G_epistemic + G_social

    Parameters
    ----------
    posterior : (N_STATES,) current state belief q(s)
    B : transition matrices {policy_int: (N_STATES, N_STATES)}
    C : log-preference vectors {modality: (n_obs,)}
    A : observation likelihood matrices {modality: (n_obs, N_STATES)}
    social_G : optional {policy_int: float} social EFE terms

    Returns
    -------
    G : (N_POLICIES,) G(pi) per policy (lower = preferred)
    """
    G = np.zeros(N_POLICIES)

    for pi in range(N_POLICIES):
        if pi not in B:
            G[pi] = 1e6
            continue

        q_next = B[pi] @ posterior  # predicted next state (N_STATES,)
        q_next = np.clip(q_next, _EPS, None)
        q_next /= q_next.sum()

        G[pi] = _single_step_efe(q_next, C, A, social_G, pi)

    return G


# ── Multi-step EFE (T=2 lookahead) ────────────────────────────

def evaluate_policies_multistep(
    posterior: np.ndarray,
    B: Dict[int, np.ndarray],
    C: Dict[str, np.ndarray],
    A: Dict[str, np.ndarray],
    social_G: Optional[Dict[int, float]] = None,
    T: int = 2,
    gamma: float = 0.9,
) -> np.ndarray:
    """T-step EFE: E_pi = F_pi + G_pi + gamma * min_{pi2} E_pi2.

    At T=2, each policy's value includes the best continuation.
    NAV_OBJ benefits because its continuation (PICKUP at AT_OBJECT)
    has high pragmatic value.

    Parameters
    ----------
    posterior : (N_STATES,) current state belief q(s)
    B : transition matrices {policy_int: (N_STATES, N_STATES)}
    C : log-preference vectors {modality: (n_obs,)}
    A : observation likelihood matrices {modality: (n_obs, N_STATES)}
    social_G : optional {policy_int: float} social EFE terms
    T : planning horizon (default 2)
    gamma : discount factor (default 0.9)

    Returns
    -------
    E : (N_POLICIES,) E_pi per policy (lower = preferred)
    """
    G = np.zeros(N_POLICIES)

    for pi in range(N_POLICIES):
        if pi not in B:
            G[pi] = 1e6
            continue

        q_t = B[pi] @ posterior
        q_t = np.clip(q_t, _EPS, None)
        q_t /= q_t.sum()

        # Step 1: G_pi (pragmatic + epistemic + social)
        G[pi] = _single_step_efe(q_t, C, A, social_G, pi)

        # Step 2..T: discounted optimal continuation
        if T >= 2:
            G_cont = float("inf")
            for pi2 in range(N_POLICIES):
                if pi2 not in B:
                    continue
                q_t2 = B[pi2] @ q_t
                q_t2 = np.clip(q_t2, _EPS, None)
                q_t2 /= q_t2.sum()
                G_cont = min(G_cont, _single_step_efe(q_t2, C, A, social_G, pi2))
            G[pi] += gamma * G_cont

    return G


# ── Policy selection with precision coupling ───────────────────

def select_policy(
    G: np.ndarray,
    beta: float,
    arousal: float,
    mask: Optional[np.ndarray] = None,
    E: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray, float]:
    """Select policy via softmax with precision coupling and habit prior.

    Da Costa (2020): q(pi) = sigma(-gamma * E_pi + ln E)
    where E is the habit prior (log-prior over policies).

    Parameters
    ----------
    G : (N_POLICIES,) expected free energy per policy (E_pi from evaluate_*)
    beta : base softmax precision (inverse temperature)
    arousal : self-arousal [0, 1] — high arousal reduces precision
    mask : (N_POLICIES,) boolean — True = feasible, False = masked out
    E : (N_POLICIES,) habit prior over policies (optional, default uniform)

    Returns
    -------
    selected : int, MAP policy index
    q_pi : (N_POLICIES,) policy posterior (from EFE, not hand-coded)
    H_pi : float, policy entropy H[q(pi)] for arousal computation
    """
    # Precision coupling: high arousal → lower beta → more exploration
    effective_beta = beta * (1.5 - np.clip(arousal, 0.0, 1.0))
    effective_beta = max(effective_beta, 0.01)  # floor to prevent division issues

    G_eff = G.copy()
    if mask is not None:
        G_eff[~mask] = 1e10  # infeasible policies get infinite G

    # Da Costa 2020: q(pi) = sigma(-gamma * E_pi + ln E)
    logits = -effective_beta * G_eff
    if E is not None:
        logits += np.log(np.clip(E, _EPS, None))  # habit prior
    logits -= logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    q_pi = exp_logits / exp_logits.sum()

    # Policy entropy
    q_pi_safe = np.clip(q_pi, _EPS, None)
    H_pi = float(-np.dot(q_pi_safe, np.log(q_pi_safe)))

    selected = int(np.argmax(q_pi))

    return selected, q_pi, H_pi
