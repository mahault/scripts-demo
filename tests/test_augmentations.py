"""Tests for affect, engagement, and saliency augmentations."""

from __future__ import annotations

import math
import time

import pytest

from architecture_core.perception.augmentations.affect import (
    AffectAugmentation,
    EmotionalState,
    _angle_to_emotion,
    obs_index_from_continuous,
)
from architecture_core.perception.augmentations.engagement import (
    EngagementAugmentation,
)
from architecture_core.perception.augmentations.saliency import (
    SaliencyAugmentation,
)


# =====================================================================
# Affect — Circumplex model
# =====================================================================
class TestEmotionalState:
    def test_happy_quadrant(self):
        s = EmotionalState(arousal=0.0, valence=0.8)
        assert s.emotion == "happy"
        assert s.intensity == pytest.approx(0.8)

    def test_angry_quadrant(self):
        s = EmotionalState(arousal=0.6, valence=-0.6)
        assert s.emotion == "angry"

    def test_sad_quadrant(self):
        s = EmotionalState(arousal=0.0, valence=-0.8)
        assert s.emotion == "sad"

    def test_calm_quadrant(self):
        s = EmotionalState(arousal=-0.8, valence=0.0)
        assert s.emotion == "calm"

    def test_excited_quadrant(self):
        s = EmotionalState(arousal=0.8, valence=0.8)
        assert s.emotion == "excited"

    def test_to_dict(self):
        s = EmotionalState(arousal=0.5, valence=0.5)
        d = s.to_dict()
        assert "arousal" in d
        assert "valence" in d
        assert "emotion" in d
        assert "intensity" in d
        assert "angle" in d


class TestAngleToEmotion:
    def test_all_sectors(self):
        assert _angle_to_emotion(0) == "happy"
        assert _angle_to_emotion(45) == "excited"
        assert _angle_to_emotion(90) == "alert"
        assert _angle_to_emotion(135) == "angry"
        assert _angle_to_emotion(180) == "sad"
        assert _angle_to_emotion(225) == "depressed"
        assert _angle_to_emotion(270) == "calm"
        assert _angle_to_emotion(315) == "relaxed"
        assert _angle_to_emotion(350) == "happy"


class TestObsIndex:
    def test_maps_correctly(self):
        assert obs_index_from_continuous(-0.8) == 0
        assert obs_index_from_continuous(0.0) == 2
        assert obs_index_from_continuous(0.8) == 4


class TestAffectAugmentation:
    def test_no_agents(self):
        aug = AffectAugmentation()
        result = aug.augment({"robot_pose": (0, 0, 0, 0), "agents": []})
        aff = result["affect"]
        assert aff["num_agents_tracked"] == 0
        assert aff["dominant_emotion"] == "neutral"

    def test_single_agent_neutral(self):
        aug = AffectAugmentation()
        result = aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [{"id": "human_1", "pose": (1, 0)}],
        })
        aff = result["affect"]
        assert aff["num_agents_tracked"] == 1
        # No affect_cues → defaults to neutral obs → state should be near neutral
        r = aff["readings"][0]
        assert r["entity_id"] == "human_1"
        assert "emotion" in r

    def test_positive_affect_cues(self):
        aug = AffectAugmentation()
        # Repeatedly observe positive+calm to shift belief
        for _ in range(10):
            result = aug.augment({
                "robot_pose": (0, 0, 0, 0),
                "agents": [{
                    "id": "h1",
                    "pose": (1, 0),
                    "affect_cues": {
                        "valence_idx": 4,  # very_positive
                        "arousal_idx": 1,  # low
                        "source": "face",
                        "confidence": 0.9,
                    },
                }],
            })
        aff = result["affect"]
        r = aff["readings"][0]
        assert r["valence"] > 0.0
        assert r["emotion"] in ("happy", "relaxed", "calm")

    def test_negative_affect_cues(self):
        aug = AffectAugmentation()
        for _ in range(10):
            result = aug.augment({
                "robot_pose": (0, 0, 0, 0),
                "agents": [{
                    "id": "h1",
                    "pose": (1, 0),
                    "affect_cues": {
                        "valence_idx": 0,  # very_negative
                        "arousal_idx": 4,  # very_high
                        "source": "voice",
                        "confidence": 0.8,
                    },
                }],
            })
        aff = result["affect"]
        r = aff["readings"][0]
        assert r["valence"] < 0.0
        assert r["arousal"] > 0.0
        assert r["emotion"] in ("angry", "alert")

    def test_belief_drift_to_neutral(self):
        aug = AffectAugmentation()
        # First: strong negative signal
        for _ in range(5):
            aug.augment({
                "robot_pose": (0, 0, 0, 0),
                "agents": [{
                    "id": "h1",
                    "pose": (1, 0),
                    "affect_cues": {"valence_idx": 0, "arousal_idx": 4},
                }],
            })
        # Then: no cues (should drift toward neutral)
        for _ in range(20):
            result = aug.augment({
                "robot_pose": (0, 0, 0, 0),
                "agents": [{"id": "h1", "pose": (1, 0)}],
            })
        r = result["affect"]["readings"][0]
        # Should be closer to neutral than the initial strong negative
        assert abs(r["valence"]) < 0.5

    def test_section_is_social(self):
        assert AffectAugmentation.section == "social"


# =====================================================================
# Engagement
# =====================================================================
class TestEngagementAugmentation:
    def test_no_agents(self):
        aug = EngagementAugmentation()
        result = aug.augment({"robot_pose": (0, 0, 0, 0), "agents": []})
        eng = result["engagement"]
        assert eng["num_engaged"] == 0
        assert eng["most_engaged_entity"] is None

    def test_close_gazing_agent_is_engaged(self):
        aug = EngagementAugmentation()
        result = aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [{
                "id": "h1",
                "pose": (0.8, 0),
                "engagement_cues": {
                    "gaze_on_robot": 0.9,
                    "body_orientation": 0.85,
                },
            }],
        })
        eng = result["engagement"]
        assert eng["num_engaged"] >= 1
        r = eng["readings"][0]
        assert r["level"] in ("high", "medium")
        assert r["gaze_on_robot"] == 0.9

    def test_distant_agent_low_engagement(self):
        aug = EngagementAugmentation()
        result = aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [{
                "id": "h1",
                "pose": (8, 0),
                "engagement_cues": {
                    "gaze_on_robot": 0.1,
                    "body_orientation": 0.1,
                },
            }],
        })
        r = result["engagement"]["readings"][0]
        assert r["level"] in ("low", "none")

    def test_stale_tracker_pruned(self):
        aug = EngagementAugmentation()
        # Tick with agent present
        aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [{"id": "h1", "pose": (1, 0)}],
        })
        assert "h1" in aug._trackers

        # Tick without agent
        aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [],
        })
        assert "h1" not in aug._trackers

    def test_section_is_social(self):
        assert EngagementAugmentation.section == "social"


# =====================================================================
# Saliency
# =====================================================================
class TestSaliencyAugmentation:
    def test_no_targets(self):
        aug = SaliencyAugmentation()
        result = aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [],
            "timestamp": 0.0,
        })
        sal = result["saliency"]
        assert sal["num_targets"] == 0
        assert sal["most_salient"] is None

    def test_close_agent_high_salience(self):
        aug = SaliencyAugmentation()
        result = aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [
                {"id": "close", "pose": (0.5, 0)},
                {"id": "far", "pose": (4.0, 0)},
            ],
            "timestamp": 0.0,
        })
        sal = result["saliency"]
        # Close agent should be most salient
        assert sal["most_salient"] == "close"
        # Scores should be ordered
        scores = {t["entity_id"]: t["score"] for t in sal["targets"]}
        assert scores["close"] > scores["far"]

    def test_hazard_high_salience(self):
        aug = SaliencyAugmentation()
        result = aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [],
            "hazards": [(1.0, 0.0, 0.5, 0.5)],
            "timestamp": 0.0,
        })
        sal = result["saliency"]
        assert sal["num_targets"] == 1
        t = sal["targets"][0]
        assert t["category"] == "hazard"
        assert t["score"] > 0.3  # threat factor + proximity

    def test_novelty_decays(self):
        aug = SaliencyAugmentation(novelty_boost_ticks=3)
        # First tick: novel
        r1 = aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [{"id": "a1", "pose": (2, 0)}],
            "timestamp": 0.0,
        })
        assert r1["saliency"]["targets"][0]["is_novel"] is True

        # After novelty_boost_ticks: no longer novel
        for i in range(1, 5):
            r = aug.augment({
                "robot_pose": (0, 0, 0, 0),
                "agents": [{"id": "a1", "pose": (2, 0)}],
                "timestamp": i * 0.1,
            })
        assert r["saliency"]["targets"][0]["is_novel"] is False

    def test_fast_agent_high_salience(self):
        aug = SaliencyAugmentation()
        # First tick: establish position
        aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [
                {"id": "fast", "pose": (3, 0)},
                {"id": "slow", "pose": (3, 1)},
            ],
            "timestamp": 0.0,
        })
        # Second tick: fast agent moved significantly
        result = aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [
                {"id": "fast", "pose": (2, 0)},   # moved 1m in 0.1s = 10 m/s
                {"id": "slow", "pose": (3, 1)},    # didn't move
            ],
            "timestamp": 0.1,
        })
        scores = {t["entity_id"]: t["score"] for t in result["saliency"]["targets"]}
        assert scores["fast"] > scores["slow"]

    def test_section_is_attention(self):
        assert SaliencyAugmentation.section == "attention"

    def test_sorted_by_salience(self):
        aug = SaliencyAugmentation()
        result = aug.augment({
            "robot_pose": (0, 0, 0, 0),
            "agents": [
                {"id": "far", "pose": (5, 0)},
                {"id": "close", "pose": (0.5, 0)},
                {"id": "mid", "pose": (2, 0)},
            ],
            "timestamp": 0.0,
        })
        targets = result["saliency"]["targets"]
        scores = [t["score"] for t in targets]
        assert scores == sorted(scores, reverse=True)
