"""Tests for TiagoNavIntentPolicy yield fix (Phase 10c).

Yield now uses speed modulation only — no target_x/target_y override.
After a timeout, speed increases to break symmetric deadlocks.
"""

import pytest

from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate
from plugins.tiago_webots.skills.nav_policy import TiagoNavIntentPolicy, _YIELD_TIMEOUT_TICKS


def _make_pb(robot_pose=(0, 0, 0, 0), agents=None):
    return PerceptBundle(
        t=0.0,
        world={
            "robot_pose": robot_pose,
            "agents": agents or [],
        },
        social={},
    )


def _make_req(x=5.0, y=5.0):
    return SkillRequest(skill="navigate", goal={"x": x, "y": y})


def _make_update():
    return SkillUpdate(params={}, constraints={})


class TestYieldNoTargetOverride:
    def test_yield_no_target_override(self):
        """Yield should NOT set target_x/target_y (no escape position)."""
        policy = TiagoNavIntentPolicy()
        pb = _make_pb(agents=[{"pose": (1, 0)}])
        result = policy.yield_(pb, _make_req(), _make_update())

        assert "target_x" not in result.params
        assert "target_y" not in result.params


class TestYieldSlowSpeed:
    def test_yield_slow_speed(self):
        """Normal yield should set speed_scale to 0.3."""
        policy = TiagoNavIntentPolicy()
        pb = _make_pb(agents=[{"pose": (1, 0)}])
        result = policy.yield_(pb, _make_req(), _make_update())

        assert result.params["speed_scale"] == pytest.approx(0.3)


class TestYieldStopDistance:
    def test_yield_stop_distance(self):
        """Yield should set stop_distance to collision-only (0.3)."""
        policy = TiagoNavIntentPolicy()
        pb = _make_pb(agents=[{"pose": (1, 0)}])
        result = policy.yield_(pb, _make_req(), _make_update())

        assert result.params["stop_distance"] == pytest.approx(0.3)


class TestYieldTimeoutFaster:
    def test_yield_timeout_faster(self):
        """After timeout ticks, yield should increase speed to 0.5."""
        policy = TiagoNavIntentPolicy()
        pb = _make_pb(agents=[{"pose": (1, 0)}])
        req = _make_req()

        # Tick up to timeout
        for _ in range(_YIELD_TIMEOUT_TICKS):
            result = policy.yield_(pb, req, _make_update())

        # At this point, _yield_ticks == _YIELD_TIMEOUT_TICKS (not > yet)
        assert result.params["speed_scale"] == pytest.approx(0.3)

        # One more tick → timeout exceeded
        result = policy.yield_(pb, req, _make_update())
        assert result.params["speed_scale"] == pytest.approx(0.5)
        assert result.params["stop_distance"] == pytest.approx(0.3)


class TestApproachResetsYield:
    def test_approach_resets_yield(self):
        """Approach intent should reset yield tick counter."""
        policy = TiagoNavIntentPolicy()
        pb = _make_pb(agents=[{"pose": (1, 0)}])
        req = _make_req()

        # Build up yield ticks
        for _ in range(10):
            policy.yield_(pb, req, _make_update())

        # Approach resets
        policy.approach(pb, req, _make_update())

        # Now yield again — should be back to slow speed
        result = policy.yield_(pb, req, _make_update())
        assert result.params["speed_scale"] == pytest.approx(0.3)

    def test_avoid_resets_yield(self):
        """Avoid intent should also reset yield tick counter."""
        policy = TiagoNavIntentPolicy()
        pb = _make_pb(agents=[{"pose": (1, 0)}])
        req = _make_req()

        for _ in range(10):
            policy.yield_(pb, req, _make_update())

        policy.avoid(pb, req, _make_update())

        result = policy.yield_(pb, req, _make_update())
        assert result.params["speed_scale"] == pytest.approx(0.3)
