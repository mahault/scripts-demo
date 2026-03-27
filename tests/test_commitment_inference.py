"""Tests for CommitmentInference — observable commitment beliefs."""

import math
import pytest

from architecture_core.cognition.planning.commitment_inference import (
    CommitmentConfig,
    CommitmentInference,
)


def _make_ci(**kwargs) -> CommitmentInference:
    return CommitmentInference(CommitmentConfig(**kwargs))


class TestApproachingBuildsCommitment:
    def test_approaching_builds_commitment(self):
        """Agent moving toward an object should build commitment belief."""
        ci = _make_ci(ema_lambda=0.5)
        objects = {"cup_1": (3.0, 0.0)}

        # First call establishes baseline
        ci.update("agent_A", (5.0, 0.0), 0.0, objects)
        # Agent moves closer, heading toward object (heading=pi, pointing left)
        ci.update("agent_A", (4.0, 0.0), math.pi, objects)
        b1 = ci.get_belief("agent_A", "cup_1")

        # Move even closer
        ci.update("agent_A", (3.5, 0.0), math.pi, objects)
        b2 = ci.get_belief("agent_A", "cup_1")

        assert b2 > b1, "Belief should increase as agent approaches"
        assert b2 > 0.2, "Should be meaningfully above min_belief"


class TestStationaryLowCommitment:
    def test_stationary_low_commitment(self):
        """Agent not moving should have low commitment."""
        ci = _make_ci(ema_lambda=0.5)
        objects = {"cup_1": (3.0, 0.0)}

        # Same position twice — no closing evidence
        ci.update("agent_A", (5.0, 0.0), 0.0, objects)
        ci.update("agent_A", (5.0, 0.0), 0.0, objects)

        b = ci.get_belief("agent_A", "cup_1")
        assert b < 0.3, f"Stationary agent belief should be low, got {b}"


class TestEmaIsSticky:
    def test_ema_is_sticky(self):
        """Belief should persist after evidence drops (EMA inertia)."""
        ci = _make_ci(ema_lambda=0.3)
        objects = {"cup_1": (3.0, 0.0)}

        # Build up belief
        for x in [5.0, 4.5, 4.0, 3.5]:
            ci.update("agent_A", (x, 0.0), math.pi, objects)
        high_belief = ci.get_belief("agent_A", "cup_1")

        # Now agent stops — evidence drops but belief should persist
        ci.update("agent_A", (3.5, 0.0), math.pi, objects)
        after_stop = ci.get_belief("agent_A", "cup_1")

        # Should still be substantial (not instantly dropping to min)
        assert after_stop > high_belief * 0.5, (
            f"Belief should persist via EMA: was {high_belief:.3f}, "
            f"after stop {after_stop:.3f}"
        )


class TestMultipleObjectsDifferentiated:
    def test_multiple_objects_differentiated(self):
        """Heading toward one object should give it higher belief than others."""
        ci = _make_ci(ema_lambda=0.5)
        objects = {
            "cup_east": (5.0, 0.0),    # heading 0
            "cup_north": (0.0, 5.0),   # heading pi/2
        }

        # Agent at origin, heading east (0 radians), approaching cup_east
        ci.update("agent_A", (2.0, 0.0), 0.0, objects)
        ci.update("agent_A", (3.0, 0.0), 0.0, objects)

        b_east = ci.get_belief("agent_A", "cup_east")
        b_north = ci.get_belief("agent_A", "cup_north")

        assert b_east > b_north, (
            f"Heading-aligned object should have higher belief: "
            f"east={b_east:.3f} vs north={b_north:.3f}"
        )


class TestProximityZone:
    def test_proximity_zone(self):
        """Agent within proximity radius should get bonus evidence."""
        ci = _make_ci(proximity_radius=2.0, ema_lambda=0.5)
        objects = {"cup_1": (3.0, 0.0)}

        # Far away — no proximity bonus
        ci.update("agent_A", (10.0, 0.0), math.pi, objects)
        ci.update("agent_A", (9.0, 0.0), math.pi, objects)
        b_far = ci.get_belief("agent_A", "cup_1")

        # Reset and try close
        ci.reset()
        ci.update("agent_A", (4.5, 0.0), math.pi, objects)
        ci.update("agent_A", (4.0, 0.0), math.pi, objects)
        b_close = ci.get_belief("agent_A", "cup_1")

        assert b_close > b_far, (
            f"Proximity bonus should increase belief: "
            f"close={b_close:.3f} vs far={b_far:.3f}"
        )


class TestHeadingMisaligned:
    def test_heading_misaligned(self):
        """Agent heading 180 degrees away from object should have low heading evidence."""
        ci = _make_ci(ema_lambda=0.5)
        objects = {"cup_1": (5.0, 0.0)}  # object is to the east

        # Agent heading west (pi) — away from object
        ci.update("agent_A", (3.0, 0.0), math.pi, objects)
        ci.update("agent_A", (3.0, 0.0), math.pi, objects)
        b_away = ci.get_belief("agent_A", "cup_1")

        ci.reset()

        # Agent heading east (0) — toward object
        ci.update("agent_A", (3.0, 0.0), 0.0, objects)
        ci.update("agent_A", (3.0, 0.0), 0.0, objects)
        b_toward = ci.get_belief("agent_A", "cup_1")

        assert b_toward > b_away, (
            f"Heading toward should beat heading away: "
            f"toward={b_toward:.3f} vs away={b_away:.3f}"
        )


class TestResetClears:
    def test_reset_clears(self):
        """Reset should revert beliefs to min_belief."""
        ci = _make_ci(ema_lambda=0.5)
        objects = {"cup_1": (3.0, 0.0)}

        ci.update("agent_A", (5.0, 0.0), math.pi, objects)
        ci.update("agent_A", (4.0, 0.0), math.pi, objects)
        assert ci.get_belief("agent_A", "cup_1") > 0.05

        ci.reset()
        assert ci.get_belief("agent_A", "cup_1") == 0.05

    def test_reset_single_agent(self):
        """Reset with agent_id should only clear that agent."""
        ci = _make_ci(ema_lambda=0.5)
        objects = {"cup_1": (3.0, 0.0)}

        ci.update("agent_A", (5.0, 0.0), math.pi, objects)
        ci.update("agent_A", (4.0, 0.0), math.pi, objects)
        ci.update("agent_B", (5.0, 0.0), math.pi, objects)
        ci.update("agent_B", (4.0, 0.0), math.pi, objects)

        ci.reset("agent_A")
        assert ci.get_belief("agent_A", "cup_1") == 0.05
        assert ci.get_belief("agent_B", "cup_1") > 0.05


class TestUnknownPairReturnsMin:
    def test_unknown_pair_returns_min(self):
        """Never-observed (agent, object) should return min_belief."""
        ci = _make_ci(min_belief=0.05)
        assert ci.get_belief("unknown_agent", "unknown_obj") == 0.05
