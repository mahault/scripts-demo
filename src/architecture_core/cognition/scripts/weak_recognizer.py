"""Weak script recognizer — A-matrix event-type classification.

Maps PerceptBundle observations to a probability distribution over
situation types (corridor encounter, open-area crossing, handover, etc.).

Implements the A-matrix conceptual clusters from:
    Albarracin, Constant, Friston & Ramstead (2021)
    "A Variational Approach to Scripts"

Each situation type is a distribution over sub-concept features
(proximity pattern, agent count, motion direction).  The recognizer
computes feature evidence for the current percept and produces a
posterior over situation types via softmax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from architecture_core.core.types import PerceptBundle
from architecture_core.cognition.scripts.script_types import SituationType


@dataclass
class SituationBelief:
    """Posterior distribution over situation types."""
    distribution: Dict[str, float]
    most_likely: str
    confidence: float
    entropy: float


class WeakScriptRecognizer:
    """Recognize the current situation type from perceptual features.

    The A-matrix maps observations to latent event types.  Each
    ``SituationType`` carries feature weights; evidence is the
    dot product of weights with extracted features, passed through
    softmax to obtain a proper distribution.

    Parameters
    ----------
    situation_types : list[SituationType]
        Known event types with their feature weight vectors.
    prior : dict[str, float] | None
        Prior belief over situation types (D-matrix).  If None,
        uniform prior is used.
    temperature : float
        Softmax temperature — lower = sharper discrimination.
    """

    def __init__(
        self,
        situation_types: List[SituationType],
        prior: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
    ) -> None:
        if not situation_types:
            raise ValueError("At least one SituationType required")
        self._types = {st.name: st for st in situation_types}
        self._temperature = max(temperature, 1e-6)

        # Uniform prior if not provided
        if prior is not None:
            self._prior = prior
        else:
            n = len(situation_types)
            self._prior = {st.name: 1.0 / n for st in situation_types}

    def recognize(self, pb: PerceptBundle) -> SituationBelief:
        """Compute posterior over situation types from perceptual features.

        Feature extraction from PerceptBundle:
        - ``agent_count``: number of agents in world
        - ``min_distance``: closest agent distance
        - ``mean_velocity``: average agent velocity (from saliency)
        - ``max_arousal``: highest arousal in affect readings
        - ``max_engagement``: highest engagement score
        - ``has_hazard``: 1.0 if any hazard present, else 0.0

        The recognizer dot-products these features with each situation
        type's weight vector (missing weights default to 0).
        """
        features = self._extract_features(pb)

        # Compute log-evidence for each type: dot(weights, features) + log(prior)
        logits: Dict[str, float] = {}
        for name, st in self._types.items():
            score = sum(
                st.feature_weights.get(f, 0.0) * v
                for f, v in features.items()
            )
            log_prior = math.log(max(self._prior.get(name, 1e-8), 1e-8))
            logits[name] = score / self._temperature + log_prior

        # Softmax
        distribution = _softmax(logits)

        # Derived quantities
        most_likely = max(distribution, key=distribution.get)  # type: ignore[arg-type]
        confidence = distribution[most_likely]
        entropy = _entropy(distribution)

        return SituationBelief(
            distribution=distribution,
            most_likely=most_likely,
            confidence=confidence,
            entropy=entropy,
        )

    def recognize_topk(
        self, pb: PerceptBundle, k: int = 4,
    ) -> List[Tuple[str, float]]:
        """Return the top-k situation types by posterior probability.

        Delegates to ``recognize()`` for the full posterior, then sorts
        descending and returns the top-k ``(situation_name, weight)`` pairs.
        """
        belief = self.recognize(pb)
        sorted_items = sorted(
            belief.distribution.items(), key=lambda x: x[1], reverse=True,
        )
        return sorted_items[:k]

    def kl_from_expected(self, belief: SituationBelief, expected: str) -> float:
        """Surprise of observing ``belief`` when ``expected`` was predicted.

        Uses the negative log probability of the expected type under
        the current posterior as a KL-like divergence signal.  This is
        the information-theoretic surprise:  -log q(expected).

        Returns 0.0 if expected type is not in the distribution.
        """
        q = belief.distribution.get(expected, 0.0)
        if q <= 0:
            return 10.0  # cap at high surprise
        return -math.log(q)

    # ------------------------------------------------------------------
    def _extract_features(self, pb: PerceptBundle) -> Dict[str, float]:
        """Pull numeric features from PerceptBundle sections."""
        features: Dict[str, float] = {}

        # World-level features
        agents = pb.world.get("agents", [])
        features["agent_count"] = float(len(agents))

        if agents:
            robot_pose = pb.world.get("robot_pose", (0, 0, 0, 0))
            rx, ry = robot_pose[0], robot_pose[1]
            dists = []
            for a in agents:
                pose = a.get("pose", (0, 0))
                d = math.sqrt((pose[0] - rx) ** 2 + (pose[1] - ry) ** 2)
                dists.append(d)
            features["min_distance"] = min(dists)
            features["inverse_min_distance"] = 1.0 / max(features["min_distance"], 0.1)
        else:
            features["min_distance"] = 10.0  # far default
            features["inverse_min_distance"] = 0.1

        # Saliency features
        sal = pb.attention.get("saliency", {})
        targets = sal.get("targets", [])
        if targets:
            vels = [t.get("velocity", 0.0) for t in targets]
            features["mean_velocity"] = sum(vels) / len(vels)
        else:
            features["mean_velocity"] = 0.0

        # Affect features
        aff = pb.social.get("affect", {})
        readings = aff.get("readings", [])
        if readings:
            features["max_arousal"] = max(r.get("arousal", 0.0) for r in readings)
        else:
            features["max_arousal"] = 0.0

        # Engagement features
        eng = pb.social.get("engagement", {})
        eng_readings = eng.get("readings", [])
        if eng_readings:
            features["max_engagement"] = max(r.get("score", 0.0) for r in eng_readings)
        else:
            features["max_engagement"] = 0.0

        # Hazard
        hazards = pb.world.get("hazards", [])
        features["has_hazard"] = 1.0 if hazards else 0.0

        # Deontic cues (artifact-script signals)
        # World may provide: [{type: 'queue_here', active: bool, ...}, ...]
        cues = pb.world.get("deontic_cues", [])
        cue_types = set()
        for c in cues:
            try:
                if c.get("active", False):
                    cue_types.add(str(c.get("type", "")).lower())
            except Exception:
                continue

        # Binary features for the most common demo cues.
        features["cue_queue_here"] = 1.0 if "queue_here" in cue_types else 0.0
        features["cue_staff_only"] = 1.0 if "staff_only" in cue_types else 0.0
        features["cue_quiet_zone"] = 1.0 if "quiet_zone" in cue_types else 0.0

        return features


# =====================================================================
# Math helpers
# =====================================================================
def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    """Numerically stable softmax over a dict of logits."""
    max_val = max(logits.values())
    exps = {k: math.exp(v - max_val) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def _entropy(dist: Dict[str, float]) -> float:
    """Shannon entropy of a distribution."""
    h = 0.0
    for p in dist.values():
        if p > 0:
            h -= p * math.log(p)
    return h
