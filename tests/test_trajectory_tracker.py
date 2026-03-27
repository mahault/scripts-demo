"""Tests for trajectory tracker."""

from __future__ import annotations

import pytest

from architecture_core.core.types import AffectState
from architecture_core.cognition.scripts.repertoire_types import RepertoireConfig
from architecture_core.cognition.scripts.trajectory_tracker import TrajectoryTracker


class TestTrajectoryTrackerBasic:
    def test_initial_state(self):
        tt = TrajectoryTracker()
        assert tt.current is None
        assert not tt.is_recording
        assert tt.completed_trajectories == []

    def test_begin_trajectory(self):
        tt = TrajectoryTracker()
        tt.begin_trajectory("test_script", "corridor_encounter")
        assert tt.is_recording
        assert tt.current is not None
        assert tt.current.script_name == "test_script"
        assert tt.current.situation_type == "corridor_encounter"

    def test_record_step_without_begin_is_noop(self):
        tt = TrajectoryTracker()
        tt.record_step(t=1.0, primitive_name="yield-pass")
        assert tt.current is None

    def test_record_step(self):
        tt = TrajectoryTracker()
        tt.begin_trajectory("test", "corridor")
        tt.record_step(
            t=1.0,
            primitive_name="yield-pass",
            outcome="SUCCESS",
            situation_belief={"corridor": 0.8, "open": 0.2},
            most_likely_situation="corridor",
            action_skill="navigate",
        )
        assert tt.current.step_count == 1
        assert tt.current.steps[0].primitive_name == "yield-pass"

    def test_multiple_steps(self):
        tt = TrajectoryTracker()
        tt.begin_trajectory("test", "corridor")
        for i in range(5):
            tt.record_step(t=float(i), primitive_name=f"step_{i}", outcome="SUCCESS")
        assert tt.current.step_count == 5


class TestTrajectoryTrackerEnd:
    def test_end_trajectory(self):
        tt = TrajectoryTracker()
        tt.begin_trajectory("test", "corridor")
        tt.record_step(t=0.0, primitive_name="a", outcome="SUCCESS")
        tt.record_step(t=2.0, primitive_name="b", outcome="SUCCESS")
        traj = tt.end_trajectory("SUCCESS")
        assert traj is not None
        assert traj.final_outcome == "SUCCESS"
        assert traj.total_duration_s == pytest.approx(2.0)
        assert not tt.is_recording
        assert len(tt.completed_trajectories) == 1

    def test_end_without_begin(self):
        tt = TrajectoryTracker()
        result = tt.end_trajectory("SUCCESS")
        assert result is None

    def test_mean_violation_kl(self):
        tt = TrajectoryTracker()
        tt.begin_trajectory("test", "corridor")
        tt.record_step(t=0, primitive_name="a", violation_kl=2.0)
        tt.record_step(t=1, primitive_name="b", violation_kl=0.0)
        tt.record_step(t=2, primitive_name="c", violation_kl=4.0)
        traj = tt.end_trajectory("SUCCESS")
        # Only non-zero KLs: (2.0 + 4.0) / 2 = 3.0
        assert traj.mean_violation_kl == pytest.approx(3.0)

    def test_no_violations_zero_mean(self):
        tt = TrajectoryTracker()
        tt.begin_trajectory("test", "corridor")
        tt.record_step(t=0, primitive_name="a")
        traj = tt.end_trajectory("SUCCESS")
        assert traj.mean_violation_kl == 0.0


class TestTrajectoryTrackerMemory:
    def test_bounded_storage(self):
        cfg = RepertoireConfig(max_stored_trajectories=3)
        tt = TrajectoryTracker(config=cfg)
        for i in range(5):
            tt.begin_trajectory(f"script_{i}", "test")
            tt.record_step(t=0, primitive_name="a")
            tt.end_trajectory("SUCCESS")
        assert len(tt.completed_trajectories) == 3
        # Should keep the most recent
        names = [t.script_name for t in tt.completed_trajectories]
        assert names == ["script_2", "script_3", "script_4"]

    def test_filter_by_situation(self):
        tt = TrajectoryTracker()
        tt.begin_trajectory("s1", "corridor")
        tt.record_step(t=0, primitive_name="a")
        tt.end_trajectory("SUCCESS")
        tt.begin_trajectory("s2", "open_area")
        tt.record_step(t=0, primitive_name="b")
        tt.end_trajectory("SUCCESS")
        tt.begin_trajectory("s3", "corridor")
        tt.record_step(t=0, primitive_name="c")
        tt.end_trajectory("SUCCESS")

        corridor = tt.get_trajectories_for_situation("corridor")
        assert len(corridor) == 2
        open_area = tt.get_trajectories_for_situation("open_area")
        assert len(open_area) == 1

    def test_clear(self):
        tt = TrajectoryTracker()
        tt.begin_trajectory("test", "corridor")
        tt.record_step(t=0, primitive_name="a")
        tt.end_trajectory("SUCCESS")
        assert len(tt.completed_trajectories) == 1
        tt.clear()
        assert len(tt.completed_trajectories) == 0
        assert not tt.is_recording
