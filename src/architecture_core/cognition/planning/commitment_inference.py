"""Commitment inference from observable behavioral cues.

Infers per-object commitment beliefs for each observed agent based on
publicly observable cues: distance trend, heading alignment, proximity.
No communication — all evidence comes from world-state observations.

Pipeline per (agent, object) pair:
  evidence = w_closing * I[dist decreasing]
           + w_heading * max(0, cos(heading - bearing))
           + w_proximity * I[dist < radius]
  activation = sigmoid(beta * (evidence - threshold))
  belief(t) = ema_lambda * activation + (1 - ema_lambda) * belief(t-1)

Robot-agnostic: imports only from stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class CommitmentBelief:
    """Belief that an agent is committed to a specific object."""
    agent_id: str
    object_id: str
    evidence: float = 0.0   # raw weighted evidence [0, 1]
    belief: float = 0.05    # smoothed belief [0, 1]


@dataclass
class CommitmentConfig:
    """Tunable parameters for commitment inference."""
    w_closing: float = 0.4         # weight: distance decreasing
    w_heading: float = 0.35        # weight: heading aligned with object
    w_proximity: float = 0.25      # weight: within proximity zone
    proximity_radius: float = 2.0  # metres
    beta: float = 6.0              # sigmoid steepness
    threshold: float = 0.4         # sigmoid center
    ema_lambda: float = 0.3        # EMA smoothing (higher = faster update)
    min_belief: float = 0.05       # floor on belief


class CommitmentInference:
    """Infers per-object commitment from observable behavioral cues.

    Parameters
    ----------
    config : CommitmentConfig, optional
        Tunable parameters.  Defaults to reasonable values.
    """

    def __init__(self, config: Optional[CommitmentConfig] = None) -> None:
        self._cfg = config or CommitmentConfig()
        # {agent_id: {object_id: CommitmentBelief}}
        self._beliefs: Dict[str, Dict[str, CommitmentBelief]] = {}
        # {agent_id: (prev_x, prev_y)} for distance-trend detection
        self._prev_pos: Dict[str, Tuple[float, float]] = {}
        # {agent_id: {object_id: prev_dist}} for closing detection
        self._prev_dist: Dict[str, Dict[str, float]] = {}

    def update(
        self,
        agent_id: str,
        agent_pos: Tuple[float, float],
        agent_heading: float,
        objects: Dict[str, Tuple[float, float]],
    ) -> Dict[str, CommitmentBelief]:
        """Update commitment beliefs for one agent given current observations.

        Parameters
        ----------
        agent_id : str
            Identifier for the observed agent.
        agent_pos : (x, y)
            Agent's current 2D position.
        agent_heading : float
            Agent's heading in radians (0 = +x, pi/2 = +y).
        objects : dict
            {object_id: (x, y)} positions of all candidate objects.

        Returns
        -------
        dict
            {object_id: CommitmentBelief} updated beliefs.
        """
        cfg = self._cfg

        if agent_id not in self._beliefs:
            self._beliefs[agent_id] = {}
        if agent_id not in self._prev_dist:
            self._prev_dist[agent_id] = {}

        agent_beliefs = self._beliefs[agent_id]
        prev_dists = self._prev_dist[agent_id]

        for obj_id, obj_pos in objects.items():
            # Current distance
            dx = obj_pos[0] - agent_pos[0]
            dy = obj_pos[1] - agent_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            # --- Evidence 1: distance decreasing ---
            closing = 0.0
            if obj_id in prev_dists:
                if dist < prev_dists[obj_id] - 0.01:
                    closing = 1.0
            prev_dists[obj_id] = dist

            # --- Evidence 2: heading aligned with object ---
            heading_aligned = 0.0
            if dist > 0.01:
                bearing = math.atan2(dy, dx)
                angle_diff = agent_heading - bearing
                # Normalize to [-pi, pi]
                angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                heading_aligned = max(0.0, math.cos(angle_diff))

            # --- Evidence 3: within proximity zone ---
            in_proximity = 1.0 if dist < cfg.proximity_radius else 0.0

            # --- Weighted evidence ---
            evidence = (
                cfg.w_closing * closing
                + cfg.w_heading * heading_aligned
                + cfg.w_proximity * in_proximity
            )

            # --- Sigmoid activation ---
            z = cfg.beta * (evidence - cfg.threshold)
            activation = 1.0 / (1.0 + math.exp(-z))

            # --- EMA smoothing ---
            if obj_id in agent_beliefs:
                prev_belief = agent_beliefs[obj_id].belief
            else:
                prev_belief = cfg.min_belief

            new_belief = cfg.ema_lambda * activation + (1.0 - cfg.ema_lambda) * prev_belief
            new_belief = max(cfg.min_belief, new_belief)

            agent_beliefs[obj_id] = CommitmentBelief(
                agent_id=agent_id,
                object_id=obj_id,
                evidence=evidence,
                belief=new_belief,
            )

        # Store position for next call
        self._prev_pos[agent_id] = agent_pos

        return dict(agent_beliefs)

    def get_belief(self, agent_id: str, object_id: str) -> float:
        """Return commitment belief for a specific (agent, object) pair.

        Returns min_belief if the pair has never been observed.
        """
        agent_beliefs = self._beliefs.get(agent_id)
        if agent_beliefs is None:
            return self._cfg.min_belief
        belief = agent_beliefs.get(object_id)
        if belief is None:
            return self._cfg.min_belief
        return belief.belief

    def reset(self, agent_id: Optional[str] = None) -> None:
        """Reset beliefs.  If agent_id given, reset only that agent."""
        if agent_id is not None:
            self._beliefs.pop(agent_id, None)
            self._prev_pos.pop(agent_id, None)
            self._prev_dist.pop(agent_id, None)
        else:
            self._beliefs.clear()
            self._prev_pos.clear()
            self._prev_dist.clear()
