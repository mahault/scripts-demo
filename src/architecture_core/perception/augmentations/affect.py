"""Circumplex model of emotion augmentation.

Implements the Active Inference formulation from:
    Pattisapu, Verbelen, Pitliya, Kiefer & Albarracin (2024)
    "Free Energy in a Circumplex Model of Emotion"

Two-dimensional emotional space (Russell's Circumplex):
    Arousal = H[Q(s|o)] = entropy of posterior beliefs
        High entropy -> high uncertainty -> high arousal
        Low entropy  -> high certainty  -> low arousal
    Valence = Utility - Expected Utility
        Positive -> "better than expected"
        Negative -> "worse than expected"

In the robotics context, the observation source is sensor-provided
affect cues (facial AU codes, body posture, physiological signals,
voice prosody) rather than LLM text classification.  The POMDP
belief-update machinery is identical to the mindsphere-coach
implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# =====================================================================
# Circumplex mapping — 8 emotion sectors (45 deg each)
# =====================================================================
EMOTION_SECTORS = {
    "happy":      (337.5,  22.5),
    "excited":    ( 22.5,  67.5),
    "alert":      ( 67.5, 112.5),
    "angry":      (112.5, 157.5),
    "sad":        (157.5, 202.5),
    "depressed":  (202.5, 247.5),
    "calm":       (247.5, 292.5),
    "relaxed":    (292.5, 337.5),
}

# Observation levels for sensor-classified signals
VALENCE_OBS_LEVELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]
AROUSAL_OBS_LEVELS = ["very_low", "low", "moderate", "high", "very_high"]

# Continuous values mapped from the 5 discrete levels
_VALENCE_VALUES = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
_AROUSAL_VALUES = np.array([ 0.1,  0.3, 0.5, 0.7, 0.9])


# =====================================================================
# Data classes
# =====================================================================
@dataclass
class EmotionalState:
    """Single point on the circumplex."""
    arousal: float          # [-1, 1]
    valence: float          # [-1, 1]
    intensity: float = 0.0  # sqrt(a^2 + v^2)
    angle: float = 0.0      # degrees on the circumplex
    emotion: str = ""

    def __post_init__(self) -> None:
        self.intensity = math.sqrt(self.arousal ** 2 + self.valence ** 2)
        self.angle = math.degrees(math.atan2(self.arousal, self.valence))
        if self.angle < 0:
            self.angle += 360.0
        self.emotion = _angle_to_emotion(self.angle)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arousal": round(self.arousal, 3),
            "valence": round(self.valence, 3),
            "intensity": round(self.intensity, 3),
            "angle": round(self.angle, 1),
            "emotion": self.emotion,
        }


@dataclass
class AffectReading:
    """Per-agent affect estimate."""
    entity_id: str
    state: EmotionalState
    confidence: float = 0.5
    observation_source: str = "unknown"  # "face", "body", "voice", "physio", …


# =====================================================================
# POMDP matrices — from Pattisapu et al. (2024) / mindsphere-coach
# =====================================================================
def _build_A(n: int = 5) -> np.ndarray:
    """Observation model p(obs | state).  Diagonal-dominant with noise."""
    A = np.array([
        [0.60, 0.20, 0.05, 0.02, 0.01],
        [0.25, 0.50, 0.15, 0.05, 0.02],
        [0.10, 0.20, 0.55, 0.20, 0.10],
        [0.03, 0.07, 0.15, 0.50, 0.25],
        [0.02, 0.03, 0.10, 0.23, 0.62],
    ], dtype=np.float64)
    for c in range(n):
        A[:, c] /= A[:, c].sum()
    return A


def _build_B(n: int = 5) -> np.ndarray:
    """Transition model p(s' | s).  70 % inertia, 20 % drift to neutral."""
    B = np.zeros((n, n), dtype=np.float64)
    mid = n // 2
    for s in range(n):
        B[s, s] = 0.70
        if s < mid:
            B[s + 1, s] = 0.20
        elif s > mid:
            B[s - 1, s] = 0.20
        else:
            B[s, s] += 0.20
        B[:, s] += 0.10 / n
    for c in range(n):
        B[:, c] /= B[:, c].sum()
    return B


def _build_D(n: int = 5) -> np.ndarray:
    """Prior — mild assumption of neutral."""
    D = np.array([0.05, 0.15, 0.60, 0.15, 0.05], dtype=np.float64)
    return D / D.sum()


# =====================================================================
# Per-agent emotion tracker (Bayesian filter)
# =====================================================================
class _AgentEmotionTracker:
    """Maintains POMDP beliefs over one agent's emotional state."""

    def __init__(self) -> None:
        self.A_v = _build_A()
        self.A_a = _build_A()
        self.B_v = _build_B()
        self.B_a = _build_B()
        self.bel_v = _build_D()
        self.bel_a = _build_D()
        self.history: List[EmotionalState] = []

    def update(self, v_idx: int, a_idx: int) -> EmotionalState:
        """Bayesian belief update + dynamics."""
        # Likelihood * prior
        self.bel_v = self.A_v[v_idx, :] * self.bel_v
        s = self.bel_v.sum()
        self.bel_v = self.bel_v / s if s > 0 else _build_D()

        self.bel_a = self.A_a[a_idx, :] * self.bel_a
        s = self.bel_a.sum()
        self.bel_a = self.bel_a / s if s > 0 else _build_D()

        # Transition dynamics (inertia + drift to neutral)
        self.bel_v = self.B_v @ self.bel_v
        self.bel_a = self.B_a @ self.bel_a

        # Smoothed continuous state from beliefs
        val = float(np.dot(self.bel_v, _VALENCE_VALUES))
        aro_raw = float(np.dot(self.bel_a, _AROUSAL_VALUES))
        aro = (aro_raw - 0.5) * 2  # centre to [-0.8, 0.8]

        state = EmotionalState(arousal=aro, valence=val)
        self.history.append(state)
        return state

    def predict_trend(self) -> Dict[str, float]:
        """Forecast next-step emotional change via the B-matrix."""
        cur_v = float(np.dot(self.bel_v, _VALENCE_VALUES))
        cur_a = float(np.dot(self.bel_a, _AROUSAL_VALUES))
        nxt_v = float(np.dot(self.B_v @ self.bel_v, _VALENCE_VALUES))
        nxt_a = float(np.dot(self.B_a @ self.bel_a, _AROUSAL_VALUES))
        return {
            "valence_trend": round(nxt_v - cur_v, 4),
            "arousal_trend": round(nxt_a - cur_a, 4),
        }


# =====================================================================
# Public augmentation
# =====================================================================
class AffectAugmentation:
    """Compute per-agent emotional state and inject into ``pb.social``.

    Sensor contract — each agent dict in ``world["agents"]`` may carry::

        "affect_cues": {
            "valence_idx": int,   # 0-4 index into VALENCE_OBS_LEVELS
            "arousal_idx": int,   # 0-4 index into AROUSAL_OBS_LEVELS
            "source": str,        # "face" | "body" | "voice" | "physio"
            "confidence": float,  # [0, 1]
        }

    If ``affect_cues`` is absent the tracker still applies transition
    dynamics (drift to neutral) so that stale readings decay gracefully.
    """

    section = "social"

    def __init__(self) -> None:
        self._trackers: Dict[str, _AgentEmotionTracker] = {}

    def augment(self, world: Dict[str, Any]) -> Dict[str, Any]:
        agents = world.get("agents", [])
        readings: List[AffectReading] = []
        dominant_emotion = "neutral"
        max_intensity = 0.0

        for agent in agents:
            eid = agent.get("id", "unknown")
            if eid not in self._trackers:
                self._trackers[eid] = _AgentEmotionTracker()
            tracker = self._trackers[eid]

            cues = agent.get("affect_cues")
            if cues:
                v_idx = int(cues.get("valence_idx", 2))
                a_idx = int(cues.get("arousal_idx", 2))
                source = cues.get("source", "unknown")
                conf = float(cues.get("confidence", 0.5))
            else:
                # No new observation — run dynamics only (neutral obs)
                v_idx, a_idx, source, conf = 2, 2, "inferred", 0.2

            state = tracker.update(v_idx, a_idx)
            readings.append(AffectReading(
                entity_id=eid,
                state=state,
                confidence=conf,
                observation_source=source,
            ))

            if state.intensity > max_intensity:
                max_intensity = state.intensity
                dominant_emotion = state.emotion

        # Aggregate: highest-arousal agent determines the "social mood"
        highest_arousal = max(
            (r.state.arousal for r in readings), default=0.0
        )
        lowest_valence = min(
            (r.state.valence for r in readings), default=0.0
        )

        return {
            "affect": {
                "readings": [
                    {
                        "entity_id": r.entity_id,
                        **r.state.to_dict(),
                        "confidence": round(r.confidence, 2),
                        "source": r.observation_source,
                    }
                    for r in readings
                ],
                "dominant_emotion": dominant_emotion,
                "highest_arousal": round(highest_arousal, 3),
                "lowest_valence": round(lowest_valence, 3),
                "num_agents_tracked": len(readings),
            }
        }


# =====================================================================
# Helpers
# =====================================================================
def _angle_to_emotion(angle: float) -> str:
    a = angle % 360
    if 337.5 <= a or a < 22.5:
        return "happy"
    if a < 67.5:
        return "excited"
    if a < 112.5:
        return "alert"
    if a < 157.5:
        return "angry"
    if a < 202.5:
        return "sad"
    if a < 247.5:
        return "depressed"
    if a < 292.5:
        return "calm"
    return "relaxed"


def obs_index_from_continuous(value: float, levels: np.ndarray = _VALENCE_VALUES) -> int:
    """Map a continuous value to the nearest discrete observation index."""
    return int(np.argmin(np.abs(levels - value)))
