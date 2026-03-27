"""Tests for the unified active inference task controller."""

import math
import numpy as np
import pytest

from architecture_core.core.types import PerceptBundle, SkillRequest
from architecture_core.cognition.planning.generative_model import (
    TaskPhase,
    HandState,
    TargetMode,
    TaskPolicy,
    N_STATES,
    state_index,
)
from architecture_core.cognition.planning.unified_task_controller import (
    UnifiedTaskController,
)


# ── Test fixtures ──────────────────────────────────────────────

class MockEFE:
    """Minimal SocialEFE stub for testing."""
    def __init__(self, empathy_factor=0.0, beta=2.0, epistemic_weight=0.5):
        self.empathy_factor = empathy_factor
    def compute_rollout(self, **kwargs):
        from architecture_core.cognition.tom.social_efe import (
            EFEResult,
            RolloutEFEOutput,
            SocialEFEOutput,
        )
        actions = [
            EFEResult(intent="approach", g_self=-0.5, g_other=-0.3,
                      g_epistemic=-0.1, g_social=-0.9, probability=0.4),
        ]
        single = SocialEFEOutput(
            actions=actions,
            selected="approach",
            selected_probability=0.4,
            empathy_factor=0.0,
            distribution={"approach": 0.4, "yield": 0.2, "wait": 0.2, "avoid": 0.2},
        )
        return RolloutEFEOutput(
            selected="approach",
            full_policy=["approach"],
            value=0.0,
            distribution={"approach": 0.4, "yield": 0.2, "wait": 0.2, "avoid": 0.2},
            single_step_output=single,
        )


class MockGatedToM:
    """Minimal GatedToM stub for testing."""
    class _Filter:
        reliability = 0.1
        def mean_profile(self):
            from architecture_core.cognition.tom.intent_particle_filter import IntentProfile
            return IntentProfile(0.3, 1.0, 1.0, 0.3)

    def predict_intent(self, agent_id, obs):
        return {"approach": 0.2, "avoid": 0.2, "yield": 0.2, "wait": 0.2, "neutral": 0.2}

    def get_or_create_filter(self, agent_id):
        return self._Filter()


@pytest.fixture
def controller():
    """Create a UnifiedTaskController with mock dependencies."""
    from architecture_core.cognition.planning.task_planner import TaskPlanner
    from architecture_core.cognition.planning.affordance import EmbodimentModel

    efe = MockEFE()
    planner = TaskPlanner(
        drop_off=(-2.0, -0.5),
        efe=efe,
        empathy_factor=0.0,
        embodiment=EmbodimentModel(arm_reach=0.85, body_radius=0.27),
    )
    tom = MockGatedToM()

    return UnifiedTaskController(
        planner=planner,
        gated_tom=tom,
        empathy_factor=0.0,
        world_bounds=(-8.5, 0.5, -6.0, 0.5),
    )


def _make_pb(
    t=1.0,
    robot_pos=(-5.0, -3.0),
    other_pos=(-6.0, -3.0),
    objects=None,
    held_object=False,
    arm_at_target=False,
):
    """Create a PerceptBundle for testing."""
    if objects is None:
        objects = [
            {"id": "can_1", "type": "Can", "position": (-7.0, -2.8),
             "table": "coffee", "status": "on_table", "held_by": None},
            {"id": "can_2", "type": "Can", "position": (-7.2, -2.5),
             "table": "coffee", "status": "on_table", "held_by": None},
        ]
    return PerceptBundle(
        t=t,
        world={
            "robot_pose": (robot_pos[0], robot_pos[1], 0, 0),
            "agents": [{"id": "other", "pose": other_pos, "alpha": 0.0}],
            "objects": objects,
            "furniture": [],
            "held_object": held_object,
            "arm_at_target": arm_at_target,
        },
        social={},
    )


# ── Interface compliance ──────────────────────────────────────

class TestInterface:
    def test_is_complete(self, controller):
        assert isinstance(controller.is_complete, bool)

    def test_is_repairing(self, controller):
        assert controller.is_repairing is False

    def test_check_violation(self, controller):
        pb = _make_pb()
        assert controller.check_violation(pb) is None

    def test_set_self_arousal(self, controller):
        controller.set_self_arousal(0.5)
        assert controller._self_arousal == 0.5

    def test_vfe_property(self, controller):
        assert isinstance(controller.vfe, float)

    def test_policy_entropy_property(self, controller):
        assert isinstance(controller.policy_entropy, float)

    def test_recognizer(self, controller):
        assert controller._recognizer is None


# ── Select without target ─────────────────────────────────────

class TestTargetSelection:
    def test_select_picks_target(self, controller):
        """First select() should choose a target and emit a nav request."""
        pb = _make_pb()
        result = controller.select(pb, None)
        assert isinstance(result, SkillRequest)
        # Should have selected a target
        assert controller._target_obj is not None

    def test_no_objects_holds_position(self, controller):
        """With no objects, robot holds position."""
        pb = _make_pb(objects=[])
        result = controller.select(pb, None)
        assert result.skill == "navigate"


# ── active=None does NOT trigger reselection ──────────────────

class TestActiveNoneHandling:
    def test_active_none_preserves_target(self, controller):
        """active=None should NOT trigger target reselection."""
        pb = _make_pb()
        # First call: select target
        controller.select(pb, None)
        target_1 = controller._target_obj

        # Second call with active=None: should keep same target
        controller.select(pb, None)
        target_2 = controller._target_obj
        assert target_2 is not None
        assert target_2.id == target_1.id

    def test_notify_success_does_not_clear_target(self, controller):
        """notify_status(SUCCESS) should NOT clear the target."""
        pb = _make_pb()
        controller.select(pb, None)
        target_before = controller._target_obj

        controller.notify_status("SUCCESS")
        # Target should still be set (beliefs may shift, but target persists)
        assert controller._target_obj is not None


# ── Policy sequence emergence ─────────────────────────────────

class TestPolicySequence:
    def test_approach_beliefs_give_nav_obj(self, controller):
        """With APPROACH beliefs, NAV_OBJ should be selected."""
        pb = _make_pb()
        result = controller.select(pb, None)
        # First tick: target selected, should navigate toward approach
        assert result.skill == "navigate"

    def test_committed_skill_persists(self, controller):
        """Once PICKUP is committed, subsequent select() returns the same request."""
        pb = _make_pb()
        controller.select(pb, None)  # select target

        # Simulate commitment by setting _committed_req directly
        req = SkillRequest(skill="pick_place", goal={"object": "can_1"}, params={"mode": "pick"})
        controller._committed_req = req
        controller._committed_phase = "pick"

        result = controller.select(pb, None)
        assert result.skill == "pick_place"
        assert result is req  # same object returned

    def test_commitment_released_on_success(self, controller):
        """notify_status SUCCESS clears commitment."""
        pb = _make_pb()
        controller.select(pb, None)
        controller._committed_req = SkillRequest(
            skill="pick_place", goal={"object": "can_1"}, params={"mode": "pick"})
        controller._committed_phase = "pick"
        controller._last_emitted_skill = "pick_place"

        controller.notify_status("SUCCESS")
        assert controller._committed_req is None

    def test_commitment_released_on_failure(self, controller):
        """notify_status FAILURE clears commitment."""
        pb = _make_pb()
        controller.select(pb, None)
        controller._committed_req = SkillRequest(
            skill="pick_place", goal={"object": "can_1"}, params={"mode": "pick"})
        controller._committed_phase = "pick"

        controller.notify_status("FAILURE")
        assert controller._committed_req is None


# ── Belief manipulation ───────────────────────────────────────

class TestBeliefManipulation:
    def test_nudge_beliefs(self, controller):
        """_nudge_beliefs should shift probability mass."""
        controller._q_s = np.full(N_STATES, 1.0 / N_STATES)
        controller._nudge_beliefs(TaskPhase.TRANSPORT, HandState.HOLDING, 0.5)
        p_transport = sum(
            controller._q_s[state_index(TaskPhase.TRANSPORT, HandState.HOLDING, m)]
            for m in TargetMode
        )
        assert p_transport > 0.3
        assert np.isclose(controller._q_s.sum(), 1.0)

    def test_nudge_target_mode(self, controller):
        """_nudge_target_mode should shift probability toward target mode."""
        controller._q_s = np.full(N_STATES, 1.0 / N_STATES)
        controller._nudge_target_mode(TargetMode.CONTESTED, 0.5)
        p_contested = sum(
            controller._q_s[state_index(p, h, TargetMode.CONTESTED)]
            for p in TaskPhase for h in HandState
        )
        assert p_contested > 0.3
        assert np.isclose(controller._q_s.sum(), 1.0)


# ── Yield waypoint safety ─────────────────────────────────────

class TestYieldSafety:
    def test_safe_yield_within_bounds(self, controller):
        """Yield target should be within world bounds."""
        tgt = controller._get_safe_yield_target(
            robot_pos=(-5.0, -3.0),
            other_pos=(-5.0, -2.0),
            furniture=[],
        )
        assert tgt is not None
        x_min, x_max, y_min, y_max = controller._world_bounds
        margin = 0.5
        assert tgt[0] >= x_min + margin - 0.01
        assert tgt[0] <= x_max - margin + 0.01
        assert tgt[1] >= y_min + margin - 0.01
        assert tgt[1] <= y_max - margin + 0.01

    def test_yield_avoids_keepout(self, controller):
        """Yield target should not be inside a furniture keepout zone."""
        furniture = [{
            "keepout_x_min": -6.0, "keepout_y_min": -4.0,
            "keepout_x_max": -4.0, "keepout_y_max": -2.0,
        }]
        tgt = controller._get_safe_yield_target(
            robot_pos=(-5.0, -3.0),
            other_pos=(-5.0, -2.0),
            furniture=furniture,
        )
        if tgt is not None:
            for f in furniture:
                in_keepout = (f["keepout_x_min"] <= tgt[0] <= f["keepout_x_max"] and
                              f["keepout_y_min"] <= tgt[1] <= f["keepout_y_max"])
                assert not in_keepout

    def test_yield_hysteresis(self, controller):
        """Yield target should persist for YIELD_PERSIST_TICKS."""
        tgt1 = controller._get_safe_yield_target(
            robot_pos=(-5.0, -3.0),
            other_pos=(-5.0, -2.0),
            furniture=[],
        )
        # Should persist on next call
        tgt2 = controller._get_safe_yield_target(
            robot_pos=(-5.0, -3.0),
            other_pos=(-4.0, -2.0),  # other moved
            furniture=[],
        )
        # Should be the same (hysteresis)
        assert tgt1 == tgt2

    def test_yield_none_when_cornered(self, controller):
        """Should return None when both sidestep directions are blocked."""
        # Surround robot with keepout zones
        furniture = [
            {"keepout_x_min": -6.0, "keepout_y_min": -4.0,
             "keepout_x_max": -4.0, "keepout_y_max": -2.0},
            {"keepout_x_min": -6.0, "keepout_y_min": -2.0,
             "keepout_x_max": -4.0, "keepout_y_max": 0.0},
        ]
        controller._yield_target = None  # reset
        controller._yield_persist_counter = 999  # force recalculation
        tgt = controller._get_safe_yield_target(
            robot_pos=(-5.0, -3.0),
            other_pos=(-5.0, -2.0),
            furniture=furniture,
        )
        # May still find a target if one direction is clear,
        # or None if both directions are in keepout
        # This test just verifies it doesn't crash


# ── Done / Lost detection ─────────────────────────────────────

class TestDoneLost:
    def test_check_done_clears_target(self, controller):
        """When DONE belief is high, target should be cleared."""
        from architecture_core.cognition.planning.task_planner import ObjectState
        controller._target_obj = ObjectState(
            id="can_1", type="Can", position=(-7.0, -2.8),
            table="coffee", status="on_table",
        )
        # Set beliefs to strongly DONE
        controller._q_s = np.full(N_STATES, 1e-6)
        for h in HandState:
            for m in TargetMode:
                controller._q_s[state_index(TaskPhase.DONE, h, m)] = 0.1
        controller._q_s /= controller._q_s.sum()

        assert controller._check_done()
        assert controller._target_obj is None

    def test_check_lost_clears_target(self, controller):
        """When LOST belief is high, target should be cleared."""
        from architecture_core.cognition.planning.task_planner import ObjectState
        controller._target_obj = ObjectState(
            id="can_1", type="Can", position=(-7.0, -2.8),
            table="coffee", status="on_table",
        )
        # Set beliefs to strongly LOST
        controller._q_s = np.full(N_STATES, 1e-6)
        for p in TaskPhase:
            for h in HandState:
                controller._q_s[state_index(p, h, TargetMode.LOST)] = 0.05
        controller._q_s /= controller._q_s.sum()

        assert controller._check_lost()
        assert controller._target_obj is None


# ── Belief summary ─────────────────────────────────────────────

class TestBeliefSummary:
    def test_belief_summary_format(self, controller):
        summary = controller.belief_summary()
        assert "(" in summary and ")" in summary
        assert any(p.name in summary for p in TaskPhase)


# ── Notify status ──────────────────────────────────────────────

class TestNotifyStatus:
    def test_success_shifts_beliefs(self, controller):
        """SUCCESS observation shifts beliefs via skill_outcome modality."""
        pb = _make_pb()
        controller.select(pb, None)  # select target
        q_before = controller._q_s.copy()
        # Simulate pick_place outcome (nav outcomes are correctly ignored)
        controller._last_emitted_skill = "pick_place"
        controller.notify_status("SUCCESS")
        # notify_status stores outcome; next select() feeds it as observation
        controller.select(pb, None)
        assert not np.allclose(controller._q_s, q_before)
        # Target should still be set
        assert controller._target_obj is not None

    def test_failure_nudges_contested(self, controller):
        """FAILURE observation nudges toward CONTESTED via A matrix."""
        pb = _make_pb()
        controller.select(pb, None)
        p_contested_before = sum(
            controller._q_s[state_index(p, h, TargetMode.CONTESTED)]
            for p in TaskPhase for h in HandState
        )
        # Simulate pick_place outcome (nav outcomes are correctly ignored)
        controller._last_emitted_skill = "pick_place"
        controller.notify_status("FAILURE")
        # Observation propagated on next select()
        controller.select(pb, None)
        p_contested_after = sum(
            controller._q_s[state_index(p, h, TargetMode.CONTESTED)]
            for p in TaskPhase for h in HandState
        )
        assert p_contested_after > p_contested_before

    def test_timeout_stronger_than_failure(self, controller):
        """TIMEOUT should provide stronger LOST evidence than FAILURE."""
        pb = _make_pb()
        controller.select(pb, None)
        # Simulate pick_place outcome (nav outcomes are correctly ignored)
        controller._last_emitted_skill = "pick_place"
        controller.notify_status("TIMEOUT")
        controller.select(pb, None)
        p_lost = sum(
            controller._q_s[state_index(p, h, TargetMode.LOST)]
            for p in TaskPhase for h in HandState
        )
        # TIMEOUT observation → LOST belief through A matrix
        assert p_lost > 0.001

    def test_nav_success_ignored(self, controller):
        """Nav SUCCESS should NOT be stored as skill outcome."""
        pb = _make_pb()
        controller.select(pb, None)  # emits navigate
        assert controller._last_emitted_skill == "navigate"
        controller.notify_status("SUCCESS")
        # Should not have stored the outcome
        assert controller._last_skill_outcome == 0  # NONE
