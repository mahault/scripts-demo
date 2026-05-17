"""Synthetic episode generator for training learners without Webots.

Generates PerceptBundle sequences parameterized by context (reception,
corridor, hospital) with controlled noise. Each episode represents a
full interaction trajectory in a given environment.

The generator encodes the statistical regularities of each context:
- Reception: many agents, close proximity, high engagement, queue cues
- Corridor: few agents, high velocity, medium distance
- Hospital: many agents, quiet zone cues, low arousal, moderate engagement
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from architecture_core.core.types import PerceptBundle


@dataclass
class EpisodeConfig:
    """Configuration for synthetic episode generation."""
    n_steps: int = 5
    noise_std: float = 0.1
    seed: Optional[int] = None


@dataclass
class Episode:
    """A synthetic interaction episode."""
    context: str
    percepts: List[PerceptBundle]
    label: str  # ground-truth context label
    success: bool = True
    prediction_error: float = 0.0


# Context-specific generation parameters (means and std devs)
_CONTEXT_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "reception": {
        "agent_count": (3.0, 1.0),
        "min_distance": (2.0, 0.5),
        "mean_velocity": (0.1, 0.05),
        "max_arousal": (0.3, 0.1),
        "max_engagement": (0.7, 0.15),
        "has_hazard": (0.0, 0.0),
        "cue_queue_here": (1.0, 0.0),
        "cue_staff_only": (0.0, 0.0),
        "cue_quiet_zone": (0.0, 0.0),
    },
    "corridor": {
        "agent_count": (1.0, 0.5),
        "min_distance": (5.0, 1.5),
        "mean_velocity": (0.8, 0.2),
        "max_arousal": (0.1, 0.05),
        "max_engagement": (0.2, 0.1),
        "has_hazard": (0.0, 0.0),
        "cue_queue_here": (0.0, 0.0),
        "cue_staff_only": (0.0, 0.0),
        "cue_quiet_zone": (0.0, 0.0),
    },
    "hospital": {
        "agent_count": (4.0, 1.0),
        "min_distance": (2.5, 0.8),
        "mean_velocity": (0.05, 0.03),
        "max_arousal": (0.1, 0.05),
        "max_engagement": (0.3, 0.1),
        "has_hazard": (0.0, 0.0),
        "cue_queue_here": (0.0, 0.0),
        "cue_staff_only": (0.0, 0.0),
        "cue_quiet_zone": (1.0, 0.0),
    },
}


class EpisodeGenerator:
    """Generates synthetic episodes for training and evaluation.

    Each episode is a sequence of PerceptBundles that statistically
    resembles observations from a particular context. The generator
    does not require Webots — it samples from parameterized distributions
    learned from the context profiles.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility. None for random.
    noise_std : float
        Base noise standard deviation added to continuous features.
    """

    def __init__(self, seed: Optional[int] = None, noise_std: float = 0.1) -> None:
        self._rng = np.random.default_rng(seed)
        self._noise_std = noise_std

    def generate_episode(
        self,
        context: str,
        n_steps: int = 5,
        noise_std: Optional[float] = None,
    ) -> Episode:
        """Generate one synthetic episode for the given context.

        Parameters
        ----------
        context : str
            Target context ("reception", "corridor", "hospital").
        n_steps : int
            Number of percept steps in the episode.
        noise_std : float | None
            Override noise level (uses instance default if None).

        Returns
        -------
        Episode with context-appropriate PerceptBundles.
        """
        if context not in _CONTEXT_PARAMS:
            raise ValueError(f"Unknown context: {context}. "
                             f"Available: {list(_CONTEXT_PARAMS.keys())}")

        sigma = noise_std if noise_std is not None else self._noise_std
        params = _CONTEXT_PARAMS[context]
        percepts = []

        for step in range(n_steps):
            pb = self._generate_percept(context, params, sigma, t=float(step))
            percepts.append(pb)

        return Episode(
            context=context,
            percepts=percepts,
            label=context,
            success=True,
            prediction_error=0.0,
        )

    def generate_batch(
        self,
        context: str,
        n_episodes: int = 50,
        n_steps: int = 5,
        noise_std: Optional[float] = None,
    ) -> List[Episode]:
        """Generate a batch of episodes for one context."""
        return [
            self.generate_episode(context, n_steps, noise_std)
            for _ in range(n_episodes)
        ]

    def generate_training_set(
        self,
        n_per_context: int = 200,
        n_steps: int = 5,
        noise_std: Optional[float] = None,
        train_ratio: float = 0.8,
    ) -> Tuple[List[Episode], List[Episode]]:
        """Generate a full training set with train/test split.

        Parameters
        ----------
        n_per_context : int
            Episodes per context for the full set.
        n_steps : int
            Steps per episode.
        noise_std : float | None
            Noise level override.
        train_ratio : float
            Fraction for training (remainder is test).

        Returns
        -------
        (train_episodes, test_episodes) tuple.
        """
        all_episodes: List[Episode] = []
        for context in _CONTEXT_PARAMS:
            episodes = self.generate_batch(context, n_per_context, n_steps, noise_std)
            all_episodes.extend(episodes)

        # Shuffle
        indices = self._rng.permutation(len(all_episodes))
        shuffled = [all_episodes[i] for i in indices]

        split_idx = int(len(shuffled) * train_ratio)
        return shuffled[:split_idx], shuffled[split_idx:]

    def _generate_percept(
        self,
        context: str,
        params: Dict[str, Tuple[float, float]],
        noise_std: float,
        t: float,
    ) -> PerceptBundle:
        """Generate a single PerceptBundle from context parameters."""
        # Sample feature values
        features = {}
        for feat_name, (mean, std) in params.items():
            # Binary features (cues, hazard) use Bernoulli
            if feat_name.startswith("cue_") or feat_name == "has_hazard":
                features[feat_name] = float(self._rng.random() < mean)
            else:
                # Continuous: sample from Gaussian with context std + noise
                total_std = math.sqrt(std**2 + noise_std**2)
                value = float(self._rng.normal(mean, total_std))
                features[feat_name] = max(0.0, value)

        # Convert features to PerceptBundle structure
        agent_count = max(1, int(round(features.get("agent_count", 1.0))))
        min_dist = max(0.5, features.get("min_distance", 5.0))

        # Generate agent poses based on min_distance
        agents = []
        for i in range(agent_count):
            dist = min_dist + float(self._rng.exponential(1.0))
            angle = float(self._rng.uniform(0, 2 * math.pi))
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            agents.append({"pose": (round(x, 2), round(y, 2))})

        # Build deontic cues
        deontic_cues = []
        if features.get("cue_queue_here", 0.0) > 0.5:
            deontic_cues.append({"type": "queue_here", "active": True})
        if features.get("cue_staff_only", 0.0) > 0.5:
            deontic_cues.append({"type": "staff_only", "active": True})
        if features.get("cue_quiet_zone", 0.0) > 0.5:
            deontic_cues.append({"type": "quiet_zone", "active": True})

        # Build hazards
        hazards = []
        if features.get("has_hazard", 0.0) > 0.5:
            hazards.append({"type": "obstacle", "position": (1.0, 1.0)})

        velocity = max(0.0, features.get("mean_velocity", 0.0))
        arousal = max(0.0, min(1.0, features.get("max_arousal", 0.0)))
        engagement = max(0.0, min(1.0, features.get("max_engagement", 0.0)))

        return PerceptBundle(
            t=t,
            world={
                "agents": agents,
                "robot_pose": (0, 0, 0, 0),
                "hazards": hazards,
                "deontic_cues": deontic_cues,
            },
            social={
                "affect": {"readings": [{"arousal": arousal}]},
                "engagement": {"readings": [{"score": engagement}]},
            },
            attention={
                "saliency": {"targets": [{"velocity": velocity}]},
            },
        )
