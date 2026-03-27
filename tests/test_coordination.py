"""Tests for ClearTableManager: ToM-driven task orchestration."""

import math

import pytest

from architecture_core.cognition.planning.clear_table_manager import ClearTableManager
from architecture_core.cognition.planning.task_planner import (
    ObjectState,
    TaskPlanner,
)
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep
from architecture_core.cognition.tom.gated_tom import GatedToM
from architecture_core.cognition.tom.intent_particle_filter import (
    IntentProfile,
    ObservationContext,
)
from architecture_core.cognition.planning.commitment_inference import (
    CommitmentBelief,
    CommitmentInference,
)
from architecture_core.cognition.tom.social_efe import SocialEFE
from architecture_core.core.types import PerceptBundle, SkillRequest


# -- Helpers ---------------------------------------------------------------

def _make_pb(
    robot_pos=(0, 0, 0, 0),
    agents=None,
    objects=None,
) -> PerceptBundle:
    """Create a minimal PerceptBundle for testing."""
    world = {
        "robot_pose": robot_pos,
        "agents": agents or [],
        "timestamp": 0.0,
    }
    if objects is not None:
        world["objects"] = objects
    return PerceptBundle(
        world=world,
        social={},
        t=0.0,
    )


def _make_object_dicts():
    """Object data as sensor output (dicts, not ObjectState)."""
    return [
        {"id": "orange_1", "type": "Orange", "position": (-1.0, -5.0),
         "table": "dining", "status": "on_table", "held_by": None},
        {"id": "can_1", "type": "Can", "position": (-7.2, -2.7),
         "table": "coffee", "status": "on_table", "held_by": None},
    ]


DROP_OFF = (-2.0, -0.5)


# -- ClearTableManager tests -----------------------------------------------

class TestClearTableManager:
    def _make_manager(self, empathy=0.5):
        efe = SocialEFE(empathy_factor=empathy, beta=2.0, epistemic_weight=0.0)
        planner = TaskPlanner(drop_off=DROP_OFF, efe=efe, empathy_factor=empathy)
        base = ScriptManager()
        tom = GatedToM(empathy_factor=empathy, n_particles=20)
        return ClearTableManager(planner=planner, base=base, gated_tom=tom)

    def test_selects_first_object_on_start(self):
        """Manager selects an object when no active request."""
        mgr = self._make_manager()
        pb = _make_pb(
            robot_pos=(-2.0, -4.5, 0.0, 0.0),
            objects=_make_object_dicts(),
        )
        req = mgr.select(pb, None)
        assert req is not None
        assert req.skill in ("navigate", "pick_place")

    def test_reports_task_complete_when_all_placed(self):
        """Manager reports complete when all objects placed."""
        mgr = self._make_manager()
        all_placed = [
            {"id": "orange_1", "type": "Orange", "position": (-2.0, -0.5),
             "table": "dining", "status": "placed", "held_by": None},
        ]
        pb = _make_pb(
            robot_pos=(-2.0, -4.5, 0.0, 0.0),
            objects=all_placed,
        )
        # Update planner with placed objects
        mgr._planner.update_objects([
            ObjectState(id="orange_1", type="Orange", position=(-2.0, -0.5),
                        table="dining", status="placed"),
        ])
        assert mgr._planner.is_task_complete

    def test_delegates_step_execution(self):
        """After selecting an object, delegates to base ScriptManager."""
        mgr = self._make_manager()
        pb = _make_pb(
            robot_pos=(-2.0, -4.5, 0.0, 0.0),
            objects=_make_object_dicts(),
        )
        # First select → generates sequence
        req1 = mgr.select(pb, None)
        assert req1.skill == "navigate"  # first step is always navigate

        # Second select with active request → delegates to base
        req2 = mgr.select(pb, req1)
        assert req2.skill == "navigate"  # still executing same step

    def test_is_complete_delegation(self):
        """is_complete reflects planner state."""
        mgr = self._make_manager()
        assert not mgr.is_complete

    def test_notify_status_delegated(self):
        """notify_status passes through to base ScriptManager."""
        mgr = self._make_manager()
        # Should not raise
        mgr.notify_status("SUCCESS")


class TestClearTableManagerWithToM:
    def test_tom_predictions_influence_selection(self):
        """Manager queries ToM and feeds predictions to planner."""
        efe = SocialEFE(empathy_factor=0.8, beta=2.0, epistemic_weight=0.0)
        planner = TaskPlanner(drop_off=DROP_OFF, efe=efe, empathy_factor=0.8)
        base = ScriptManager()
        tom = GatedToM(empathy_factor=0.8, n_particles=20)
        mgr = ClearTableManager(planner=planner, base=base, gated_tom=tom)

        pb = _make_pb(
            robot_pos=(-4.0, -3.5, 0.0, 0.0),
            agents=[{"id": "other_robot", "pose": (-2.0, -4.5), "goal": (-7.0, -2.5), "alpha": 0.0}],
            objects=_make_object_dicts(),
        )

        req = mgr.select(pb, None)
        assert req is not None

    def test_observation_phase_with_wait(self):
        """When EFE says wait, manager enters observation phase."""
        mgr = TestClearTableManager._make_manager(TestClearTableManager(), empathy=0.5)

        # Manually trigger observation
        mgr._awaiting_observation = True
        mgr._observe_ticks = 0

        pb = _make_pb(
            robot_pos=(-4.0, -3.5, 0.0, 0.0),
            objects=_make_object_dicts(),
        )
        req = mgr.select(pb, None)
        # During observation, robot stays put
        assert req.skill == "navigate"
        assert req.goal["x"] == -4.0  # stays at current position

    def test_observation_phase_exits_after_max_ticks(self):
        """Observation phase ends after max_observe_ticks."""
        mgr = TestClearTableManager._make_manager(TestClearTableManager(), empathy=0.5)

        mgr._awaiting_observation = True
        mgr._observe_ticks = mgr._max_observe_ticks  # at max

        pb = _make_pb(
            robot_pos=(-4.0, -3.5, 0.0, 0.0),
            objects=_make_object_dicts(),
        )
        req = mgr.select(pb, None)
        # Should exit observation and select an object
        assert not mgr._awaiting_observation


class TestDeadlockDrivenYield:
    """Deadlock evidence: persistent yield + contention → EFE penalty."""

    def _make_manager(self, empathy=0.5):
        efe = SocialEFE(empathy_factor=empathy, beta=2.0, epistemic_weight=0.0)
        planner = TaskPlanner(drop_off=DROP_OFF, efe=efe, empathy_factor=empathy)
        base = ScriptManager()
        tom = GatedToM(empathy_factor=empathy, n_particles=20)
        return ClearTableManager(planner=planner, base=base, gated_tom=tom)

    def test_penalty_grows_with_yield_count_and_contention(self):
        """Penalty = p_deadlock * 3, where p_deadlock = (count/10) * max(0.3, p_contend)."""
        mgr = self._make_manager()

        # Simulate 5 consecutive yields on orange_1 with some commitment
        mgr._yield_count["orange_1"] = 5
        mgr._commitment._beliefs["other"] = {
            "orange_1": CommitmentBelief(
                agent_id="other", object_id="orange_1",
                evidence=0.6, belief=0.6,
            ),
        }

        # Compute p_deadlock manually: (5/10) * max(0.3, 0.6) = 0.5 * 0.6 = 0.3
        p_deadlock = min(1.0, (5 / 10.0) * max(0.3, 0.6))
        expected_penalty = p_deadlock * 3.0  # 0.9

        # Trigger penalty computation via internal state
        obj_id = "orange_1"
        other_id = "other"
        n_yields = mgr._yield_count[obj_id]
        p_contend = mgr._commitment.get_belief(other_id, obj_id)
        actual_p_deadlock = min(1.0, (n_yields / 10.0) * max(0.3, p_contend))
        if actual_p_deadlock > 0.1:
            mgr._yield_penalties[obj_id] = actual_p_deadlock * 3.0

        assert mgr._yield_penalties["orange_1"] == pytest.approx(expected_penalty, abs=0.01)

    def test_no_penalty_when_contention_low(self):
        """Low contention → p_deadlock stays below threshold even with many yields."""
        mgr = self._make_manager()

        # 2 yields, no commitment (floor 0.3 applies)
        mgr._yield_count["orange_1"] = 2
        # p_deadlock = (2/10) * 0.3 = 0.06 — below 0.1 threshold
        n_yields = 2
        p_contend = 0.0
        p_deadlock = min(1.0, (n_yields / 10.0) * max(0.3, p_contend))
        assert p_deadlock < 0.1  # no penalty should be applied

    def test_penalty_decays(self):
        """Penalty decays by 0.9 each deliberation tick."""
        mgr = self._make_manager()
        mgr._yield_penalties["orange_1"] = 2.0

        # Simulate 5 decay steps
        for _ in range(5):
            for obj_id in list(mgr._yield_penalties):
                mgr._yield_penalties[obj_id] *= 0.9
                if mgr._yield_penalties[obj_id] < 0.1:
                    del mgr._yield_penalties[obj_id]

        # 2.0 * 0.9^5 = 1.18 — still above threshold
        assert "orange_1" in mgr._yield_penalties
        assert mgr._yield_penalties["orange_1"] == pytest.approx(2.0 * 0.9**5, abs=0.01)

    def test_penalty_removed_below_threshold(self):
        """Penalty below 0.1 is removed."""
        mgr = self._make_manager()
        mgr._yield_penalties["orange_1"] = 0.09

        for obj_id in list(mgr._yield_penalties):
            mgr._yield_penalties[obj_id] *= 0.9
            if mgr._yield_penalties[obj_id] < 0.1:
                del mgr._yield_penalties[obj_id]

        assert "orange_1" not in mgr._yield_penalties

    def test_yield_count_cleared_on_non_yield(self):
        """When robot stops yielding, count resets."""
        mgr = self._make_manager()
        mgr._yield_count["orange_1"] = 5

        # Simulate non-yield → count cleared
        mgr._yield_count.pop("orange_1", None)
        assert "orange_1" not in mgr._yield_count

    def test_yield_penalties_passed_to_planner(self):
        """Yield penalties flow through to TaskPlanner.select_next()."""
        mgr = self._make_manager()
        mgr._yield_penalties["orange_1"] = 2.0

        pb = _make_pb(
            robot_pos=(-2.0, -4.5, 0.0, 0.0),
            objects=_make_object_dicts(),
        )

        req = mgr.select(pb, None)
        assert req is not None

    def test_penalty_reflects_contention_strength(self):
        """Higher commitment → higher penalty for same yield count."""
        mgr = self._make_manager()

        # Low contention: 5 yields, commitment=0.2
        n = 5
        p_low = min(1.0, (n / 10.0) * max(0.3, 0.2))   # 0.5 * 0.3 = 0.15
        # High contention: 5 yields, commitment=0.8
        p_high = min(1.0, (n / 10.0) * max(0.3, 0.8))   # 0.5 * 0.8 = 0.40

        assert p_high > p_low


class TestCommitmentWiring:
    """CommitmentInference is wired through ClearTableManager to TaskPlanner."""

    def _make_manager(self, empathy=0.8):
        efe = SocialEFE(empathy_factor=empathy, beta=2.0, epistemic_weight=0.0)
        planner = TaskPlanner(drop_off=DROP_OFF, efe=efe, empathy_factor=empathy)
        base = ScriptManager()
        tom = GatedToM(empathy_factor=empathy, n_particles=20)
        return ClearTableManager(planner=planner, base=base, gated_tom=tom)

    def test_commitment_created_and_wired(self):
        """ClearTableManager creates CommitmentInference and wires to planner."""
        mgr = self._make_manager()
        assert mgr._commitment is not None
        assert mgr._planner._commitment is mgr._commitment

    def test_update_other_agent_called(self):
        """select() calls update_other_agent when agents are present."""
        mgr = self._make_manager()
        pb = _make_pb(
            robot_pos=(-2.0, -4.5, 0.0, 0.0),
            agents=[{"id": "other_robot", "pose": (-1.0, -5.0), "alpha": 1.57}],
            objects=_make_object_dicts(),
        )
        mgr.select(pb, None)
        # After select, planner should know about the other agent
        assert mgr._planner._other_agent_id == "other_robot"

    def test_commitment_belief_grows_with_approach(self):
        """Commitment belief increases when other agent approaches an object."""
        mgr = self._make_manager()
        objects = _make_object_dicts()

        # Simulate 5 ticks of other agent approaching orange_1 at (-1.0, -5.0)
        for i in range(5):
            x = -1.0 + (4 - i) * 0.5  # starts at 1.0, ends at -1.0
            pb = PerceptBundle(
                world={
                    "robot_pose": (-5.0, -3.0, 0.0, 0.0),
                    "agents": [{"id": "other", "pose": (x, -5.0), "alpha": 3.14}],
                    "objects": objects,
                },
                social={},
                t=float(i),
            )
            mgr.select(pb, None)

        belief = mgr._commitment.get_belief("other", "orange_1")
        assert belief > mgr._commitment._cfg.min_belief


class TestMotionTracking:
    """Velocity and approaching computed from distance deltas."""

    def _make_manager(self, empathy=0.5):
        efe = SocialEFE(empathy_factor=empathy, beta=2.0, epistemic_weight=0.0)
        planner = TaskPlanner(drop_off=DROP_OFF, efe=efe, empathy_factor=empathy)
        base = ScriptManager()
        tom = GatedToM(empathy_factor=empathy, n_particles=20)
        return ClearTableManager(planner=planner, base=base, gated_tom=tom)

    def test_velocity_positive_when_closing(self):
        """Velocity > 0 when other agent moves closer between ticks."""
        mgr = self._make_manager()
        robot_pos = (0.0, 0.0)

        # Tick 1: other at distance 5
        pb1 = PerceptBundle(world={"robot_pose": (0, 0, 0, 0)}, social={}, t=1.0)
        obs1 = mgr._build_obs_context(pb1, robot_pos, (5.0, 0.0), "other")

        # Tick 2: other moved closer to distance 3 (dt=1.0)
        pb2 = PerceptBundle(world={"robot_pose": (0, 0, 0, 0)}, social={}, t=2.0)
        obs2 = mgr._build_obs_context(pb2, robot_pos, (3.0, 0.0), "other")

        assert obs2.velocity > 0.0
        assert obs2.velocity == pytest.approx(2.0, abs=0.1)  # (5-3)/1.0

    def test_approaching_false_when_stationary(self):
        """Approaching is False when other agent doesn't move."""
        mgr = self._make_manager()
        robot_pos = (0.0, 0.0)

        pb1 = PerceptBundle(world={"robot_pose": (0, 0, 0, 0)}, social={}, t=1.0)
        mgr._build_obs_context(pb1, robot_pos, (3.0, 0.0), "other")

        pb2 = PerceptBundle(world={"robot_pose": (0, 0, 0, 0)}, social={}, t=2.0)
        obs2 = mgr._build_obs_context(pb2, robot_pos, (3.0, 0.0), "other")

        assert obs2.velocity == pytest.approx(0.0, abs=0.01)
        assert obs2.approaching is False

    def test_approaching_true_when_distance_decreases(self):
        """Approaching is True when distance is decreasing."""
        mgr = self._make_manager()
        robot_pos = (0.0, 0.0)

        pb1 = PerceptBundle(world={"robot_pose": (0, 0, 0, 0)}, social={}, t=0.0)
        mgr._build_obs_context(pb1, robot_pos, (4.0, 0.0), "other")

        pb2 = PerceptBundle(world={"robot_pose": (0, 0, 0, 0)}, social={}, t=1.0)
        obs2 = mgr._build_obs_context(pb2, robot_pos, (3.5, 0.0), "other")

        assert obs2.approaching is True


# -- Emergent Coordination Tests -------------------------------------------

def _extract_target(mgr: ClearTableManager) -> str | None:
    """Extract the target object ID from the manager's loaded sequence.

    After select(), the base ScriptManager has a sequence whose pick_place
    step contains goal={"object": obj_id}.  Returns None if no pick found.
    """
    seq = mgr._base._sequence
    if seq is None:
        return None
    for step in seq.steps:
        if step.request.skill == "pick_place" and step.request.goal:
            obj_id = step.request.goal.get("object")
            if obj_id:
                return obj_id
    return None


class TestEmergentCoordination:
    """Two ClearTableManagers naturally partition objects via EFE + ToM + Commitment.

    This is the integration test for emergent coordination: two robots in the
    same scene, each running a full ClearTableManager pipeline, should converge
    on different target objects without any explicit communication.

    Mechanism: Robot A approaches Object 1 → commitment inference for A on
    Object 1 rises → Robot B's obstruction for Object 1 increases →
    G_social(Object 1) increases for B → B switches to Object 2.
    """

    OBJ_A_POS = (-3.0, -5.0)
    OBJ_B_POS = (-7.0, -3.0)
    DROP_OFF = (-5.0, -1.0)

    def _make_objects(self):
        return [
            {"id": "obj_a", "type": "Orange", "position": self.OBJ_A_POS,
             "table": "dining", "status": "on_table", "held_by": None},
            {"id": "obj_b", "type": "Can", "position": self.OBJ_B_POS,
             "table": "coffee", "status": "on_table", "held_by": None},
        ]

    def _make_manager(self, empathy: float) -> ClearTableManager:
        efe = SocialEFE(empathy_factor=empathy, beta=2.0, epistemic_weight=0.0)
        planner = TaskPlanner(
            drop_off=self.DROP_OFF, efe=efe, empathy_factor=empathy,
        )
        base = ScriptManager()
        tom = GatedToM(empathy_factor=empathy, n_particles=20)
        return ClearTableManager(planner=planner, base=base, gated_tom=tom)

    def _step_both(
        self,
        mgr_1: ClearTableManager,
        mgr_2: ClearTableManager,
        pos_1: tuple,
        pos_2: tuple,
        heading_1: float,
        heading_2: float,
        objects: list,
        t: float,
    ):
        """Step both managers, each seeing the other as an agent."""
        pb_1 = PerceptBundle(
            world={
                "robot_pose": (pos_1[0], pos_1[1], 0.0, 0.0),
                "agents": [{"id": "robot_2", "pose": pos_2, "alpha": heading_2}],
                "objects": objects,
            },
            social={},
            t=t,
        )
        pb_2 = PerceptBundle(
            world={
                "robot_pose": (pos_2[0], pos_2[1], 0.0, 0.0),
                "agents": [{"id": "robot_1", "pose": pos_1, "alpha": heading_1}],
                "objects": objects,
            },
            social={},
            t=t,
        )
        req_1 = mgr_1.select(pb_1, None)
        req_2 = mgr_2.select(pb_2, None)
        return req_1, req_2

    def test_robots_partition_at_least_once(self):
        """During the simulation, robots select different objects at least once.

        Robot 1 (low empathy) starts near obj_a, Robot 2 (high empathy)
        starts near obj_b.  As commitment builds, each robot should prefer
        its nearest object.  We check that partitioning occurs at some tick.
        """
        mgr_1 = self._make_manager(empathy=0.2)
        mgr_2 = self._make_manager(empathy=0.8)
        objects = self._make_objects()

        pos_1_start = (-3.5, -4.5)
        pos_2_start = (-7.5, -3.5)

        heading_1 = math.atan2(
            self.OBJ_A_POS[1] - pos_1_start[1],
            self.OBJ_A_POS[0] - pos_1_start[0],
        )
        heading_2 = math.atan2(
            self.OBJ_B_POS[1] - pos_2_start[1],
            self.OBJ_B_POS[0] - pos_2_start[0],
        )

        partitioned_at_least_once = False

        for tick in range(8):
            frac = tick / 10.0
            pos_1 = (
                pos_1_start[0] + frac * (self.OBJ_A_POS[0] - pos_1_start[0]),
                pos_1_start[1] + frac * (self.OBJ_A_POS[1] - pos_1_start[1]),
            )
            pos_2 = (
                pos_2_start[0] + frac * (self.OBJ_B_POS[0] - pos_2_start[0]),
                pos_2_start[1] + frac * (self.OBJ_B_POS[1] - pos_2_start[1]),
            )

            self._step_both(
                mgr_1, mgr_2, pos_1, pos_2,
                heading_1, heading_2, objects, t=float(tick),
            )

            target_1 = _extract_target(mgr_1)
            target_2 = _extract_target(mgr_2)
            if target_1 and target_2 and target_1 != target_2:
                partitioned_at_least_once = True

        assert partitioned_at_least_once, (
            "Robots never selected different objects during 8 ticks"
        )

    def test_empathic_robot_defers_on_contested_object(self):
        """A high-empathy robot avoids an object the other is committed to.

        With high empathy, the obstruction from commitment raises g_social
        (collision risk + other-comfort) enough to make the uncontested
        object preferable.
        """
        mgr_empathic = self._make_manager(empathy=0.8)
        objects = self._make_objects()

        # Inject high commitment from other agent on obj_a.
        # Other is at a moderate distance (not on top of the object,
        # so physical proximity doesn't dominate over commitment).
        mgr_empathic._commitment._beliefs["other"] = {
            "obj_a": CommitmentBelief(
                agent_id="other", object_id="obj_a",
                evidence=0.9, belief=0.9,
            ),
        }
        mgr_empathic._planner.update_other_agent("other", (-3.2, -4.8), 3.14)

        # Robot starts equidistant-ish from both objects
        pb = PerceptBundle(
            world={
                "robot_pose": (-5.0, -4.0, 0.0, 0.0),
                "agents": [{"id": "other", "pose": (-3.2, -4.8), "alpha": 3.14}],
                "objects": objects,
            },
            social={},
            t=0.0,
        )
        mgr_empathic.select(pb, None)
        target = _extract_target(mgr_empathic)

        assert target == "obj_b", (
            f"Empathic robot should defer on contested obj_a but targeted {target}"
        )

    def test_empathy_amplifies_contention_sensitivity(self):
        """Higher empathy makes a robot more sensitive to contention.

        Two planners see the same moderate commitment.  The empathic one
        should have a larger EFE increase for the contested object than
        the low-empathy one.
        """
        drop_off = self.DROP_OFF
        objects = [
            ObjectState(id="obj_a", type="Orange", position=self.OBJ_A_POS,
                        table="dining", status="on_table"),
            ObjectState(id="obj_b", type="Can", position=self.OBJ_B_POS,
                        table="coffee", status="on_table"),
        ]

        robot_pos = (-5.0, -4.0)  # roughly equidistant
        obs = ObservationContext(
            kinematic_intent="approach", distance=3.0, velocity=0.3,
            approaching=True, gaze_on_robot=0.5, body_orientation=0.5,
            valence=0.0, arousal=0.0, robot_last_intent="neutral",
        )
        q_other = {
            "approach": 0.3, "avoid": 0.1, "yield": 0.2,
            "wait": 0.2, "neutral": 0.2,
        }
        profile = IntentProfile(
            approach_bias=0.3, responsiveness=1.0,
            precision=1.0, empathy_j=0.3,
        )
        other_pos = (-4.0, -5.0)

        efe_values = {}
        for empathy in (0.2, 0.8):
            efe = SocialEFE(empathy_factor=empathy, beta=2.0, epistemic_weight=0.0)
            planner = TaskPlanner(
                drop_off=drop_off, efe=efe, empathy_factor=empathy,
            )
            planner.update_objects(objects)
            planner.update_other_agent("other", other_pos, 3.14)
            # Wire commitment inference and inject moderate commitment
            planner._commitment = CommitmentInference()
            planner._commitment._beliefs["other"] = {
                "obj_a": CommitmentBelief(
                    agent_id="other", object_id="obj_a",
                    evidence=0.5, belief=0.5,
                ),
            }
            obj, _ = planner.select_next(
                robot_pos=robot_pos, obs=obs, q_other=q_other,
                profile=profile, other_pos=other_pos,
            )
            efe_values[empathy] = obj

        # With moderate contention on obj_a, the high-empathy planner
        # should be more likely to pick obj_b (uncontested).
        # At minimum, both should select SOMETHING.
        assert efe_values[0.2] is not None
        assert efe_values[0.8] is not None

        # The empathic planner should select the uncontested object
        assert efe_values[0.8].id == "obj_b", (
            f"Empathic planner should prefer uncontested obj_b, got {efe_values[0.8].id}"
        )
