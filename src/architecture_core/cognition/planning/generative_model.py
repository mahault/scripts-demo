"""Factored POMDP generative model for unified task control.

30-state discrete POMDP: phase(5) x hand(2) x target_mode(3).
7 observation modalities, 7 policies.

Matrices encode the generative model for a table-clearing task where
task phases emerge from posterior beliefs and actions are selected by
EFE minimisation rather than explicit state machine transitions.

NumPy only — no JAX or pymdp dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple

import numpy as np


# ── State factors ──────────────────────────────────────────────

class TaskPhase(IntEnum):
    APPROACH = 0
    AT_OBJECT = 1
    TRANSPORT = 2
    AT_DROPOFF = 3
    DONE = 4


class HandState(IntEnum):
    EMPTY = 0
    HOLDING = 1


class TargetMode(IntEnum):
    FREE = 0
    CONTESTED = 1
    LOST = 2


N_PHASES = len(TaskPhase)
N_HAND = len(HandState)
N_TARGET = len(TargetMode)
N_STATES = N_PHASES * N_HAND * N_TARGET  # 30


def state_index(phase: int, hand: int, target_mode: int) -> int:
    """Flat index into the 30-state vector: phase * 6 + hand * 3 + target_mode."""
    return phase * (N_HAND * N_TARGET) + hand * N_TARGET + target_mode


def state_factors(index: int) -> Tuple[int, int, int]:
    """Recover (phase, hand, target_mode) from flat index."""
    target_mode = index % N_TARGET
    remainder = index // N_TARGET
    hand = remainder % N_HAND
    phase = remainder // N_HAND
    return phase, hand, target_mode


# ── Observation modalities ─────────────────────────────────────

class DistObs(IntEnum):
    FAR = 0      # > 1.5 m
    NEAR = 1     # 0.5–1.5 m
    AT = 2       # <= 0.5 m


class BinaryObs(IntEnum):
    NO = 0
    YES = 1


class ObjZObs(IntEnum):
    TABLE = 0
    HELD = 1
    PLACED = 2


class SocialObs(IntEnum):
    CLEAR = 0
    LOW = 1
    HIGH = 2


class TargetObs(IntEnum):
    AVAILABLE = 0
    HELD_OTHER = 1
    GONE = 2


class SkillOutcomeObs(IntEnum):
    NONE = 0      # no outcome this tick
    SUCCESS = 1   # skill completed successfully
    FAIL = 2      # skill failed
    TIMEOUT = 3   # skill timed out


# Dimensionality per modality
OBS_DIMS: Dict[str, int] = {
    "d_obj": 3,
    "d_drop": 3,
    "arm": 2,
    "hold": 2,
    "obj_z": 3,
    "social": 3,
    "target": 3,
    "skill_outcome": 4,
}

MODALITY_NAMES = list(OBS_DIMS.keys())


@dataclass
class Observation:
    """Discretised observation vector (8 modalities)."""
    d_obj: int
    d_drop: int
    arm: int
    hold: int
    obj_z: int
    social: int
    target: int
    skill_outcome: int = 0  # SkillOutcomeObs.NONE

    def as_dict(self) -> Dict[str, int]:
        return {
            "d_obj": self.d_obj,
            "d_drop": self.d_drop,
            "arm": self.arm,
            "hold": self.hold,
            "obj_z": self.obj_z,
            "social": self.social,
            "target": self.target,
            "skill_outcome": self.skill_outcome,
        }


# ── Policies ───────────────────────────────────────────────────

class TaskPolicy(IntEnum):
    NAV_OBJ = 0
    PICKUP = 1
    NAV_DROP = 2
    PLACE = 3
    YIELD = 4
    WAIT = 5
    RESELECT = 6


N_POLICIES = len(TaskPolicy)


# ── Discretisation helpers ─────────────────────────────────────

def discretize_distance(d: float) -> int:
    """Map continuous distance to DistObs enum value."""
    if d <= 0.5:
        return int(DistObs.AT)
    elif d <= 1.5:
        return int(DistObs.NEAR)
    return int(DistObs.FAR)


def discretize_commitment(belief: float) -> int:
    """Map commitment belief [0,1] to SocialObs enum value."""
    if belief < 0.2:
        return int(SocialObs.CLEAR)
    elif belief < 0.5:
        return int(SocialObs.LOW)
    return int(SocialObs.HIGH)


# ── Internal helpers ───────────────────────────────────────────

_EPS = 1e-16


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _normalize_columns(M: np.ndarray) -> np.ndarray:
    """Ensure each column of M sums to 1."""
    col_sums = M.sum(axis=0)
    col_sums = np.where(col_sums < _EPS, 1.0, col_sums)
    return M / col_sums[np.newaxis, :]


# ── Base target-mode transition matrices ───────────────────────

def _target_mode_transitions(policy: int) -> np.ndarray:
    """P(target_mode' | target_mode) for a given policy.

    Shape (3, 3): T[m', m] = probability of transitioning from m to m'.
    Columns sum to 1.
    """
    if policy == TaskPolicy.YIELD:
        # Yielding can resolve contention
        return np.array([
            # from: FREE    CONTESTED  LOST
            [0.88,   0.18,      0.03],   # → FREE
            [0.10,   0.79,      0.02],   # → CONTESTED
            [0.02,   0.03,      0.95],   # → LOST
        ])
    elif policy == TaskPolicy.WAIT:
        # Waiting slightly resolves contention
        return np.array([
            [0.90,   0.14,      0.03],
            [0.08,   0.83,      0.02],
            [0.02,   0.03,      0.95],
        ])
    elif policy == TaskPolicy.RESELECT:
        # Reselect forces mostly FREE
        return np.array([
            [0.92,   0.85,      0.80],
            [0.06,   0.10,      0.10],
            [0.02,   0.05,      0.10],
        ])
    else:
        # Base target-mode dynamics (policy-independent)
        return np.array([
            [0.90,   0.12,      0.03],
            [0.08,   0.85,      0.02],
            [0.02,   0.03,      0.95],
        ])


# ── A matrices (observation likelihoods) ───────────────────────

def build_A_matrices() -> Dict[str, np.ndarray]:
    """Build observation likelihood matrices P(o | s) for all 7 modalities.

    Returns dict mapping modality name to array of shape (n_obs_m, 30).
    Each column is a probability distribution over observations given state.
    """
    A: Dict[str, np.ndarray] = {}

    # -- d_obj: distance to target object (3, 30) --
    A_d_obj = np.zeros((3, N_STATES))
    for s in range(N_STATES):
        p, h, m = state_factors(s)
        if p == TaskPhase.APPROACH:
            #               FAR   NEAR  AT
            A_d_obj[:, s] = [0.55, 0.35, 0.10]
        elif p == TaskPhase.AT_OBJECT:
            A_d_obj[:, s] = [0.05, 0.15, 0.80]
        elif p in (TaskPhase.TRANSPORT, TaskPhase.AT_DROPOFF):
            if h == HandState.HOLDING:
                # Held object is at gripper → distance ≈ 0
                A_d_obj[:, s] = [0.05, 0.10, 0.85]
            else:
                A_d_obj[:, s] = [0.50, 0.30, 0.20]
        elif p == TaskPhase.DONE:
            A_d_obj[:, s] = [0.60, 0.25, 0.15]
    A["d_obj"] = A_d_obj

    # -- d_drop: distance to dropoff (3, 30) --
    A_d_drop = np.zeros((3, N_STATES))
    for s in range(N_STATES):
        p, h, m = state_factors(s)
        if p in (TaskPhase.APPROACH, TaskPhase.AT_OBJECT):
            A_d_drop[:, s] = [0.65, 0.25, 0.10]
        elif p == TaskPhase.TRANSPORT:
            A_d_drop[:, s] = [0.45, 0.40, 0.15]
        elif p == TaskPhase.AT_DROPOFF:
            A_d_drop[:, s] = [0.05, 0.15, 0.80]
        elif p == TaskPhase.DONE:
            A_d_drop[:, s] = [0.20, 0.35, 0.45]
    A["d_drop"] = A_d_drop

    # -- arm: arm proximity to any on-table object (2, 30) --
    A_arm = np.zeros((2, N_STATES))
    for s in range(N_STATES):
        p, h, m = state_factors(s)
        if p == TaskPhase.AT_OBJECT and h == HandState.EMPTY:
            #              NO    YES
            A_arm[:, s] = [0.10, 0.90]
        elif p == TaskPhase.APPROACH:
            A_arm[:, s] = [0.75, 0.25]
        else:
            A_arm[:, s] = [0.85, 0.15]
    A["arm"] = A_arm

    # -- hold: holding object (2, 30) --
    # Near-deterministic: directly observed from sensor
    A_hold = np.zeros((2, N_STATES))
    for s in range(N_STATES):
        _, h, _ = state_factors(s)
        if h == HandState.EMPTY:
            A_hold[:, s] = [0.99, 0.01]
        else:
            A_hold[:, s] = [0.01, 0.99]
    A["hold"] = A_hold

    # -- obj_z: object z-height inference (3, 30) --
    A_obj_z = np.zeros((3, N_STATES))
    for s in range(N_STATES):
        p, h, m = state_factors(s)
        if m == TargetMode.LOST:
            # Object may be held by other or gone — ambiguous z
            A_obj_z[:, s] = [0.20, 0.55, 0.25]
        elif p in (TaskPhase.APPROACH, TaskPhase.AT_OBJECT):
            #                  TABLE HELD  PLACED
            A_obj_z[:, s] = [0.90, 0.05, 0.05]
        elif p in (TaskPhase.TRANSPORT, TaskPhase.AT_DROPOFF):
            if h == HandState.HOLDING:
                A_obj_z[:, s] = [0.05, 0.90, 0.05]
            else:
                A_obj_z[:, s] = [0.70, 0.10, 0.20]
        elif p == TaskPhase.DONE:
            A_obj_z[:, s] = [0.05, 0.05, 0.90]
    A["obj_z"] = A_obj_z

    # -- social: commitment belief discretised (3, 30) --
    # Depends primarily on target_mode
    A_social = np.zeros((3, N_STATES))
    for s in range(N_STATES):
        _, _, m = state_factors(s)
        if m == TargetMode.FREE:
            #                   CLEAR LOW   HIGH
            A_social[:, s] = [0.80, 0.15, 0.05]
        elif m == TargetMode.CONTESTED:
            A_social[:, s] = [0.10, 0.30, 0.60]
        elif m == TargetMode.LOST:
            A_social[:, s] = [0.40, 0.30, 0.30]
    A["social"] = A_social

    # -- target: object availability (3, 30) --
    # Depends primarily on target_mode
    A_target = np.zeros((3, N_STATES))
    for s in range(N_STATES):
        _, _, m = state_factors(s)
        if m == TargetMode.FREE:
            #                    AVAILABLE  HELD_OTHER  GONE
            A_target[:, s] = [0.90,      0.05,       0.05]
        elif m == TargetMode.CONTESTED:
            A_target[:, s] = [0.65,      0.20,       0.15]
        elif m == TargetMode.LOST:
            A_target[:, s] = [0.10,      0.60,       0.30]
    A["target"] = A_target

    # -- skill_outcome: last skill result (4, 30) --
    # NONE=0, SUCCESS=1, FAIL=2, TIMEOUT=3
    # Most ticks produce NONE.  SUCCESS is evidence for phase transitions.
    # FAIL is evidence for CONTESTED.  TIMEOUT is strong CONTESTED/LOST evidence.
    A_skill = np.zeros((4, N_STATES))
    for s in range(N_STATES):
        p, h, m = state_factors(s)
        #                    NONE  SUCCESS FAIL  TIMEOUT
        # Default: NONE is overwhelmingly likely (most ticks have no outcome)
        A_skill[:, s] = [0.85, 0.05, 0.05, 0.05]

        # Phase transitions where SUCCESS is expected
        if p == TaskPhase.TRANSPORT and h == HandState.HOLDING and m == TargetMode.FREE:
            # Just picked up → SUCCESS likely
            A_skill[:, s] = [0.40, 0.45, 0.10, 0.05]
        elif p == TaskPhase.DONE and h == HandState.EMPTY:
            # Just placed → SUCCESS likely
            A_skill[:, s] = [0.40, 0.50, 0.05, 0.05]

        # CONTESTED states → FAIL/TIMEOUT more likely
        if m == TargetMode.CONTESTED:
            A_skill[:, s] = [0.60, 0.05, 0.25, 0.10]
        elif m == TargetMode.LOST:
            A_skill[:, s] = [0.50, 0.02, 0.18, 0.30]
    A["skill_outcome"] = A_skill

    return A


# ── B matrices (transition models) ────────────────────────────

def build_B_matrices() -> Dict[int, np.ndarray]:
    """Build transition matrices P(s' | s, pi) for all 7 policies.

    Returns dict mapping TaskPolicy int to array of shape (30, 30).
    B[pi][s', s] = P(s' | s, pi).  Columns sum to 1.

    NAV_OBJ and NAV_DROP advance probabilities are set to low defaults;
    call update_B_from_distances() each tick with actual distances.
    """
    B: Dict[int, np.ndarray] = {}

    for pi in TaskPolicy:
        B_pi = np.zeros((N_STATES, N_STATES))
        T_target = _target_mode_transitions(pi)

        for s in range(N_STATES):
            p, h, m = state_factors(s)

            # Phase x hand transitions for this policy
            # List of ((phase', hand'), probability)
            ph_transitions = _phase_hand_transitions(pi, p, h)

            for (p_next, h_next), ph_prob in ph_transitions:
                for m_next in range(N_TARGET):
                    s_next = state_index(p_next, h_next, m_next)
                    B_pi[s_next, s] += ph_prob * T_target[m_next, m]

        B[int(pi)] = _normalize_columns(B_pi)

    return B


def _phase_hand_transitions(
    policy: int, phase: int, hand: int,
) -> list:
    """Return phase x hand transition rules for a policy applied to (phase, hand).

    Returns list of ((phase', hand'), probability).
    """
    stay = [((phase, hand), 1.0)]

    if policy == TaskPolicy.NAV_OBJ:
        if phase == TaskPhase.APPROACH and hand == HandState.EMPTY:
            # Default low advance (updated by update_B_from_distances)
            return [
                ((TaskPhase.AT_OBJECT, HandState.EMPTY), 0.10),
                ((TaskPhase.APPROACH, HandState.EMPTY), 0.90),
            ]
        return stay

    elif policy == TaskPolicy.PICKUP:
        if phase == TaskPhase.AT_OBJECT and hand == HandState.EMPTY:
            return [
                ((TaskPhase.TRANSPORT, HandState.HOLDING), 0.95),
                ((TaskPhase.AT_OBJECT, HandState.EMPTY), 0.05),
            ]
        return stay

    elif policy == TaskPolicy.NAV_DROP:
        if phase == TaskPhase.TRANSPORT and hand == HandState.HOLDING:
            return [
                ((TaskPhase.AT_DROPOFF, HandState.HOLDING), 0.10),
                ((TaskPhase.TRANSPORT, HandState.HOLDING), 0.90),
            ]
        return stay

    elif policy == TaskPolicy.PLACE:
        if phase == TaskPhase.AT_DROPOFF and hand == HandState.HOLDING:
            return [
                ((TaskPhase.DONE, HandState.EMPTY), 0.95),
                ((TaskPhase.AT_DROPOFF, HandState.HOLDING), 0.05),
            ]
        return stay

    elif policy in (TaskPolicy.YIELD, TaskPolicy.WAIT):
        return stay

    elif policy == TaskPolicy.RESELECT:
        return [
            ((TaskPhase.APPROACH, HandState.EMPTY), 0.90),
            ((phase, hand), 0.10),
        ]

    return stay


# ── Distance-conditioned B update ─────────────────────────────

def update_B_from_distances(
    B: Dict[int, np.ndarray],
    d_obj: float,
    d_drop: float,
) -> Dict[int, np.ndarray]:
    """Return updated B matrices with distance-conditioned advance probabilities.

    p_advance = sigmoid((threshold - d) / scale) — smooth transition that
    replaces discrete NAV SUCCESS events.  When close, advance probability
    is high; when far, it is low.
    """
    B_updated = {k: v.copy() for k, v in B.items()}

    p_adv_obj = float(np.clip(_sigmoid((0.5 - d_obj) / 0.2), 0.01, 0.99))
    p_adv_drop = float(np.clip(_sigmoid((0.5 - d_drop) / 0.2), 0.01, 0.99))

    T_target_base = _target_mode_transitions(TaskPolicy.NAV_OBJ)

    # NAV_OBJ: rebuild columns for (APPROACH, EMPTY, *) states
    B_nav = B_updated[int(TaskPolicy.NAV_OBJ)]
    for m in range(N_TARGET):
        s = state_index(TaskPhase.APPROACH, HandState.EMPTY, m)
        B_nav[:, s] = 0.0
        for m_next in range(N_TARGET):
            t_prob = T_target_base[m_next, m]
            s_advance = state_index(TaskPhase.AT_OBJECT, HandState.EMPTY, m_next)
            s_stay = state_index(TaskPhase.APPROACH, HandState.EMPTY, m_next)
            B_nav[s_advance, s] += p_adv_obj * t_prob
            B_nav[s_stay, s] += (1.0 - p_adv_obj) * t_prob

    # NAV_DROP: rebuild columns for (TRANSPORT, HOLDING, *) states
    B_drop = B_updated[int(TaskPolicy.NAV_DROP)]
    for m in range(N_TARGET):
        s = state_index(TaskPhase.TRANSPORT, HandState.HOLDING, m)
        B_drop[:, s] = 0.0
        for m_next in range(N_TARGET):
            t_prob = T_target_base[m_next, m]
            s_advance = state_index(TaskPhase.AT_DROPOFF, HandState.HOLDING, m_next)
            s_stay = state_index(TaskPhase.TRANSPORT, HandState.HOLDING, m_next)
            B_drop[s_advance, s] += p_adv_drop * t_prob
            B_drop[s_stay, s] += (1.0 - p_adv_drop) * t_prob

    return B_updated


# ── C vectors (log-preferences) ───────────────────────────────

def build_C_vectors(empathy_factor: float = 0.0) -> Dict[str, np.ndarray]:
    """Build preferred observation vectors ln P(o | C) for each modality.

    Positive = preferred, negative = dispreferred.
    Social dispreference for contention is scaled by empathy_factor.
    """
    return {
        #                    FAR   NEAR  AT
        "d_obj":   np.array([0.0,  0.0,  0.5]),
        "d_drop":  np.array([0.0,  0.0,  2.0]),
        #                    NO    YES
        "arm":     np.array([0.0,  0.0]),
        "hold":    np.array([0.0,  1.0]),
        #                    TABLE HELD  PLACED
        "obj_z":   np.array([0.0,  0.0,  3.0]),
        #                    CLEAR LOW   HIGH
        "social":  np.array([1.0,  0.0,  -1.5 * empathy_factor]),
        #                    AVAIL HELD_O GONE
        "target":  np.array([0.5,  -2.0, -2.0]),
        #                    NONE  SUCC  FAIL  TIMEOUT
        "skill_outcome": np.array([0.0, 2.0, -1.0, -2.0]),
    }


# ── D vector (initial prior) ──────────────────────────────────

def build_D_vector() -> np.ndarray:
    """Initial state prior: mostly APPROACH / EMPTY / FREE."""
    D = np.full(N_STATES, 1e-4)
    D[state_index(TaskPhase.APPROACH, HandState.EMPTY, TargetMode.FREE)] = 0.85
    D[state_index(TaskPhase.APPROACH, HandState.EMPTY, TargetMode.CONTESTED)] = 0.10
    D /= D.sum()
    return D
