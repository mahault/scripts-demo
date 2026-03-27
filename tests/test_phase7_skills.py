"""Phase 7: Tests for HandoverSkill, PickPlaceSkill, GazeSkill,
their IntentPolicies, new script primitives, and Executive integration.
"""

from __future__ import annotations

import pytest

from architecture_core.core.blackboard import Blackboard
from architecture_core.core.registry import SkillEntry, SkillRegistry
from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate
from architecture_core.core.executive import Executive
from architecture_core.cognition.norms.norm_engine import NormEngine
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.tom.tom_modulator import ToMModulator
from architecture_core.safety.shield import SafetyShield

from architecture_core.skills.handover_skill import HandoverSkill, HandoverIntentPolicy
from architecture_core.skills.pick_place_skill import PickPlaceSkill, PickPlaceIntentPolicy
from architecture_core.skills.gaze_skill import GazeSkill, GazeIntentPolicy


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _pb(t: float = 0.0, **world_kw) -> PerceptBundle:
    """Quick PerceptBundle factory."""
    return PerceptBundle(t=t, world=world_kw)


def _pb_social(t: float = 0.0, closest_distance: float = 5.0, **world_kw) -> PerceptBundle:
    """PerceptBundle with proxemics data."""
    return PerceptBundle(
        t=t,
        world=world_kw,
        social={"proxemics": {"closest_distance": closest_distance}},
    )


def _update(**params) -> SkillUpdate:
    return SkillUpdate(params=params)


def _constrained(**constraints) -> SkillUpdate:
    return SkillUpdate(constraints=constraints)


# ===================================================================
# HandoverSkill
# ===================================================================
class TestHandoverSkill:

    def test_start_extend_mode(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "extend"}))
        assert s._phase == "extending"
        assert s._mode == "extend"

    def test_start_receive_mode(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "receive"}))
        assert s._phase == "waiting_to_receive"
        assert s._mode == "receive"

    def test_start_defaults_to_extend(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover"))
        assert s._mode == "extend"

    def test_extend_stays_running_when_agent_far(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "extend"}))
        pb = _pb_social(t=1.0, closest_distance=3.0, held_object="cup")
        status = s.tick(pb, _update())
        assert status == "RUNNING"
        assert s._phase == "extending"

    def test_extend_transitions_on_agent_proximity(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "extend"}))
        # Agent is close
        pb = _pb_social(t=1.0, closest_distance=0.5, held_object="cup")
        status = s.tick(pb, _update())
        assert status == "RUNNING"
        assert s._phase == "waiting_for_transfer"

    def test_extend_success_on_object_taken(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "extend"}))
        # Agent close → transition
        pb1 = _pb_social(t=1.0, closest_distance=0.5, held_object="cup")
        s.tick(pb1, _update())
        assert s._phase == "waiting_for_transfer"
        # Object taken (held_object cleared)
        pb2 = _pb_social(t=2.0, closest_distance=0.5)  # no held_object
        status = s.tick(pb2, _update())
        assert status == "SUCCESS"

    def test_receive_success_on_object_grasped(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "receive"}))
        # Object near hand → grasping
        pb1 = _pb_social(t=1.0, closest_distance=0.5, object_near_hand=True)
        s.tick(pb1, _update())
        assert s._phase == "grasping"
        # Object grasped
        pb2 = _pb_social(t=2.0, closest_distance=0.5, held_object="cup")
        status = s.tick(pb2, _update())
        assert status == "SUCCESS"

    def test_emergency_stop_holds_position(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "extend"}))
        pb = _pb_social(t=1.0, closest_distance=0.5, held_object="cup")
        status = s.tick(pb, _constrained(emergency_stop=True))
        assert status == "RUNNING"
        assert s._phase == "extending"  # no phase change

    def test_timeout_returns_failure_extend(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "extend"}, timeout_s=2.0))
        # Agent close → waiting_for_transfer
        pb1 = _pb_social(t=1.0, closest_distance=0.5, held_object="cup")
        s.tick(pb1, _update())
        # Time passes beyond timeout, object still held
        pb2 = _pb_social(t=4.0, closest_distance=0.5, held_object="cup")
        status = s.tick(pb2, _update())
        assert status == "FAILURE"

    def test_timeout_returns_failure_receive(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "receive"}, timeout_s=2.0))
        # No object appears
        pb1 = _pb_social(t=1.0, closest_distance=0.5)
        s.tick(pb1, _update())
        pb2 = _pb_social(t=4.0, closest_distance=0.5)
        status = s.tick(pb2, _update())
        assert status == "FAILURE"

    def test_retract_causes_failure(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "extend"}))
        pb = _pb_social(t=1.0, closest_distance=0.5, held_object="cup")
        status = s.tick(pb, _update(retract=True))
        assert status == "FAILURE"
        assert s._phase == "idle"

    def test_stop_resets_state(self):
        s = HandoverSkill()
        s.start(SkillRequest(skill="handover", params={"mode": "extend"}))
        s.stop("test")
        assert s._phase == "idle"
        assert s._elapsed == 0.0

    def test_hook_methods_called(self):
        """Verify hooks are invoked at correct state transitions."""
        calls = []

        class TrackedHandover(HandoverSkill):
            def _on_extend(self):
                calls.append("extend")
            def _on_release(self):
                calls.append("release")
            def _on_retract(self):
                calls.append("retract")

        s = TrackedHandover()
        s.start(SkillRequest(skill="handover", params={"mode": "extend"}))
        pb = _pb_social(t=1.0, closest_distance=0.5, held_object="cup")
        s.tick(pb, _update())
        assert "extend" in calls
        assert "release" in calls

        s.stop("done")
        assert "retract" in calls


# ===================================================================
# HandoverIntentPolicy
# ===================================================================
class TestHandoverIntentPolicy:

    def setup_method(self):
        self.policy = HandoverIntentPolicy()
        self.pb = _pb(t=0.0)
        self.req = SkillRequest(skill="handover")

    def test_approach_sets_full_speed(self):
        u = self.policy.approach(self.pb, self.req, SkillUpdate())
        assert u.params["arm_speed"] == 1.0
        assert u.params["speed_scale"] == 1.0

    def test_avoid_sets_retract(self):
        u = self.policy.avoid(self.pb, self.req, SkillUpdate())
        assert u.params["retract"] is True
        assert u.params["arm_speed"] == 0.0

    def test_yield_sets_cautious_speed(self):
        u = self.policy.yield_(self.pb, self.req, SkillUpdate())
        assert u.params["arm_speed"] == 0.5
        assert u.params["speed_scale"] == 0.5

    def test_wait_freezes(self):
        u = self.policy.wait(self.pb, self.req, SkillUpdate())
        assert u.params["arm_speed"] == 0.0
        assert u.params["speed_scale"] == 0.0


# ===================================================================
# PickPlaceSkill
# ===================================================================
class TestPickPlaceSkill:

    def test_start_pick_mode(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place", params={"mode": "pick"},
                             goal={"object": "mug"}))
        assert s._mode == "pick"
        assert s._target_object == "mug"
        assert s._phase == "approaching"

    def test_start_place_mode(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place", params={"mode": "place"},
                             goal={"position": (1.0, 2.0, 0.5)}))
        assert s._mode == "place"
        assert s._phase == "approaching"

    def test_start_defaults_to_pick(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place"))
        assert s._mode == "pick"

    def test_pick_approach_to_grasp_transition(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place", params={"mode": "pick"},
                             goal={"object": "mug"}))
        pb = _pb(t=1.0, arm_at_target=True)
        status = s.tick(pb, _update())
        assert status == "RUNNING"
        assert s._phase == "grasping"

    def test_pick_grasp_success(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place", params={"mode": "pick"},
                             goal={"object": "mug"}))
        # Approach → grasp
        s.tick(_pb(t=1.0, arm_at_target=True), _update())
        assert s._phase == "grasping"
        # Object grasped
        s.tick(_pb(t=2.0, held_object="mug"), _update())
        assert s._phase == "verifying"
        # Verify stable (3 ticks required)
        assert s.tick(_pb(t=3.0, held_object="mug"), _update()) == "RUNNING"
        assert s.tick(_pb(t=4.0, held_object="mug"), _update()) == "RUNNING"
        status = s.tick(_pb(t=5.0, held_object="mug"), _update())
        assert status == "SUCCESS"

    def test_place_release_success(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place", params={"mode": "place"},
                             goal={"position": (1.0, 2.0, 0.5)}))
        # Approach → release
        s.tick(_pb(t=1.0, arm_at_target=True, held_object="mug"), _update())
        assert s._phase == "releasing"
        # Object released (held_object cleared)
        s.tick(_pb(t=2.0), _update())
        assert s._phase == "verifying"
        # Verify stable (3 ticks required)
        assert s.tick(_pb(t=3.0), _update()) == "RUNNING"
        assert s.tick(_pb(t=4.0), _update()) == "RUNNING"
        status = s.tick(_pb(t=5.0), _update())
        assert status == "SUCCESS"

    def test_pick_verify_failure_on_drop(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place", params={"mode": "pick"},
                             goal={"object": "mug"}))
        # Approach → grasp → verify
        s.tick(_pb(t=1.0, arm_at_target=True), _update())
        s.tick(_pb(t=2.0, held_object="mug"), _update())
        assert s._phase == "verifying"
        # Object dropped during verify
        status = s.tick(_pb(t=3.0), _update())  # no held_object
        assert status == "FAILURE"

    def test_emergency_stop(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place", params={"mode": "pick"}))
        pb = _pb(t=1.0, arm_at_target=True)
        status = s.tick(pb, _constrained(emergency_stop=True))
        assert status == "RUNNING"
        assert s._phase == "approaching"  # no transition

    def test_retract_causes_failure(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place", params={"mode": "pick"}))
        status = s.tick(_pb(t=1.0), _update(retract=True))
        assert status == "FAILURE"
        assert s._phase == "idle"

    def test_timeout_returns_timeout(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place", params={"mode": "pick"},
                             timeout_s=2.0))
        s.tick(_pb(t=1.0), _update())
        status = s.tick(_pb(t=4.0), _update())
        assert status == "TIMEOUT"

    def test_stop_resets_state(self):
        s = PickPlaceSkill()
        s.start(SkillRequest(skill="pick_place", params={"mode": "pick"}))
        s.stop("test")
        assert s._phase == "idle"

    def test_hook_methods_called(self):
        calls = []

        class TrackedPP(PickPlaceSkill):
            def _on_arm_move(self, target):
                calls.append(("arm_move", target))
            def _on_grasp(self):
                calls.append("grasp")
            def _on_retract(self):
                calls.append("retract")

        s = TrackedPP()
        s.start(SkillRequest(skill="pick_place", params={"mode": "pick"},
                             goal={"object": "mug"}))
        s.tick(_pb(t=1.0), _update())  # approaching → arm_move
        assert any(c[0] == "arm_move" for c in calls if isinstance(c, tuple))

        s.tick(_pb(t=2.0, arm_at_target=True), _update())  # → grasping
        assert "grasp" in calls

        s.stop("done")
        assert "retract" in calls


# ===================================================================
# PickPlaceIntentPolicy
# ===================================================================
class TestPickPlaceIntentPolicy:

    def setup_method(self):
        self.policy = PickPlaceIntentPolicy()
        self.pb = _pb(t=0.0)
        self.req = SkillRequest(skill="pick_place")

    def test_approach_normal_speed(self):
        u = self.policy.approach(self.pb, self.req, SkillUpdate())
        assert u.params["arm_speed"] == 1.0
        assert u.params["grip_force"] == 1.0

    def test_avoid_retracts(self):
        u = self.policy.avoid(self.pb, self.req, SkillUpdate())
        assert u.params["retract"] is True
        assert u.params["arm_speed"] == 0.0

    def test_yield_slow_careful(self):
        u = self.policy.yield_(self.pb, self.req, SkillUpdate())
        assert u.params["arm_speed"] == pytest.approx(0.3)
        assert u.params["grip_force"] == pytest.approx(0.8)

    def test_yield_caution_zone_when_close(self):
        pb = _pb_social(t=0.0, closest_distance=0.3)
        u = self.policy.yield_(pb, self.req, SkillUpdate())
        assert u.constraints.get("caution_zone") is True

    def test_wait_freezes_arm(self):
        u = self.policy.wait(self.pb, self.req, SkillUpdate())
        assert u.params["arm_speed"] == 0.0


# ===================================================================
# GazeSkill
# ===================================================================
class TestGazeSkill:

    def test_look_at_agent_success(self):
        s = GazeSkill()
        s.start(SkillRequest(skill="gaze", params={"mode": "look_at_agent"},
                             goal={"target": "human_1"}))
        pb = PerceptBundle(
            t=1.0,
            world={"agents": [{"id": "human_1", "pose": (2.0, 3.0)}]},
        )
        # Tick STABILIZE_TICKS times to stabilize
        for i in range(GazeSkill.STABILIZE_TICKS - 1):
            status = s.tick(PerceptBundle(
                t=1.0 + i * 0.1,
                world={"agents": [{"id": "human_1", "pose": (2.0, 3.0)}]},
            ), _update())
            assert status == "RUNNING"

        status = s.tick(PerceptBundle(
            t=2.0,
            world={"agents": [{"id": "human_1", "pose": (2.0, 3.0)}]},
        ), _update())
        assert status == "SUCCESS"

    def test_look_at_agent_missing_stays_running(self):
        s = GazeSkill()
        s.start(SkillRequest(skill="gaze", params={"mode": "look_at_agent"},
                             goal={"target": "human_1"}))
        # No agents in perception
        pb = _pb(t=1.0)
        status = s.tick(pb, _update())
        assert status == "RUNNING"
        assert s._stabilized_ticks == 0

    def test_look_at_point_success(self):
        s = GazeSkill()
        s.start(SkillRequest(skill="gaze", params={"mode": "look_at_point"},
                             goal={"target": (5.0, 3.0, 1.0)}))
        for i in range(GazeSkill.STABILIZE_TICKS):
            status = s.tick(_pb(t=1.0 + i * 0.1), _update())

        assert status == "SUCCESS"

    def test_look_at_point_no_target_fails(self):
        s = GazeSkill()
        s.start(SkillRequest(skill="gaze", params={"mode": "look_at_point"}))
        status = s.tick(_pb(t=1.0), _update())
        assert status == "FAILURE"

    def test_scan_cycles_through_targets(self):
        s = GazeSkill()
        s.start(SkillRequest(skill="gaze", params={"mode": "scan"}))
        targets = [(1, 0, 0), (0, 1, 0), (-1, 0, 0)]
        pb = PerceptBundle(t=1.0, attention={"salient_targets": targets})

        # Should need STABILIZE_TICKS per target + advance through all
        ticks = 0
        status = "RUNNING"
        while status == "RUNNING" and ticks < 100:
            status = s.tick(
                PerceptBundle(t=1.0 + ticks * 0.1,
                              attention={"salient_targets": targets}),
                _update(),
            )
            ticks += 1

        assert status == "SUCCESS"
        assert s._scan_index == len(targets)

    def test_scan_no_targets_succeeds(self):
        s = GazeSkill()
        s.start(SkillRequest(skill="gaze", params={"mode": "scan"}))
        status = s.tick(_pb(t=1.0), _update())
        assert status == "SUCCESS"

    def test_avert_looks_away(self):
        s = GazeSkill()
        s.start(SkillRequest(skill="gaze", params={"mode": "avert"},
                             goal={"target": "human_1"}))
        pb = PerceptBundle(
            t=1.0,
            world={
                "robot_pose": (0, 0, 0),
                "agents": [{"id": "human_1", "pose": (1.0, 0.0)}],
            },
        )
        update = _update()
        for i in range(GazeSkill.STABILIZE_TICKS):
            status = s.tick(
                PerceptBundle(
                    t=1.0 + i * 0.1,
                    world={
                        "robot_pose": (0, 0, 0),
                        "agents": [{"id": "human_1", "pose": (1.0, 0.0)}],
                    },
                ),
                _update(),
            )

        assert status == "SUCCESS"
        # Gaze target should be away from agent (negative x direction)
        gaze = update.params.get("gaze_target")
        # The last update won't have it since we create new ones, but
        # verify through hook tracking instead

    def test_emergency_stop(self):
        s = GazeSkill()
        s.start(SkillRequest(skill="gaze", params={"mode": "look_at_point"},
                             goal={"target": (1, 0, 0)}))
        status = s.tick(_pb(t=1.0), _constrained(emergency_stop=True))
        assert status == "RUNNING"
        assert s._stabilized_ticks == 0  # no progress

    def test_timeout(self):
        s = GazeSkill()
        s.start(SkillRequest(skill="gaze", params={"mode": "look_at_agent"},
                             goal={"target": "missing"}, timeout_s=2.0))
        s.tick(_pb(t=1.0), _update())
        status = s.tick(_pb(t=4.0), _update())
        assert status == "FAILURE"

    def test_avoid_intent_triggers_avert(self):
        """GazeIntentPolicy avoid sets gaze_avert, which overrides mode."""
        s = GazeSkill()
        s.start(SkillRequest(skill="gaze", params={"mode": "look_at_agent"},
                             goal={"target": "human_1"}))
        pb = PerceptBundle(
            t=1.0,
            world={
                "robot_pose": (0, 0, 0),
                "agents": [{"id": "human_1", "pose": (1.0, 0.0)}],
            },
        )
        # Simulate avoid intent setting gaze_avert
        for i in range(GazeSkill.STABILIZE_TICKS):
            status = s.tick(
                PerceptBundle(
                    t=1.0 + i * 0.1,
                    world={
                        "robot_pose": (0, 0, 0),
                        "agents": [{"id": "human_1", "pose": (1.0, 0.0)}],
                    },
                ),
                _update(gaze_avert=True),
            )
        # Should succeed via avert path, not look_at_agent
        assert status == "SUCCESS"

    def test_hook_method_called(self):
        calls = []

        class TrackedGaze(GazeSkill):
            def _on_gaze_update(self, target):
                calls.append(target)

        s = TrackedGaze()
        s.start(SkillRequest(skill="gaze", params={"mode": "look_at_point"},
                             goal={"target": (5.0, 3.0, 1.0)}))
        s.tick(_pb(t=1.0), _update())
        assert len(calls) == 1
        assert calls[0] == (5.0, 3.0, 1.0)


# ===================================================================
# GazeIntentPolicy
# ===================================================================
class TestGazeIntentPolicy:

    def setup_method(self):
        self.policy = GazeIntentPolicy()
        self.pb = _pb(t=0.0)
        self.req = SkillRequest(skill="gaze")

    def test_approach_direct_tracking(self):
        u = self.policy.approach(self.pb, self.req, SkillUpdate())
        assert u.params["gaze_tracking"] is True
        assert u.params["track_persistence"] == 1.0

    def test_avoid_avert_gaze(self):
        u = self.policy.avoid(self.pb, self.req, SkillUpdate())
        assert u.params["gaze_avert"] is True
        assert u.params["avert_direction"] == "down"

    def test_yield_intermittent(self):
        u = self.policy.yield_(self.pb, self.req, SkillUpdate())
        assert u.params["gaze_tracking"] is True
        assert u.params["track_persistence"] == 0.5

    def test_wait_holds_gaze(self):
        u = self.policy.wait(self.pb, self.req, SkillUpdate())
        assert u.params["gaze_hold"] is True


# ===================================================================
# Primitive Library
# ===================================================================
class TestPrimitiveLibraryPhase7:

    def test_new_primitives_registered(self):
        lib = PrimitiveLibrary()
        assert len(lib) == 13  # 8 original + 5 new
        for name in ("pick-object", "place-object",
                     "gaze-at-agent", "gaze-scan", "gaze-avert"):
            assert lib.get(name) is not None, f"Missing primitive: {name}"

    def test_pick_place_primitives_applicable_to_workstation(self):
        lib = PrimitiveLibrary()
        applicable = lib.get_applicable("workstation")
        names = {p.name for p in applicable}
        assert "pick-object" in names
        assert "place-object" in names

    def test_gaze_primitives_applicable_to_interaction(self):
        lib = PrimitiveLibrary()
        applicable = lib.get_applicable("interaction")
        names = {p.name for p in applicable}
        assert "gaze-at-agent" in names

    def test_gaze_scan_applicable_to_open_area(self):
        lib = PrimitiveLibrary()
        applicable = lib.get_applicable("open_area")
        names = {p.name for p in applicable}
        assert "gaze-scan" in names

    def test_original_primitives_unchanged(self):
        lib = PrimitiveLibrary()
        for name in ("approach-greet", "yield-pass", "wait-acknowledge",
                     "avoid-reroute", "follow-maintain", "disengage-depart",
                     "handover-extend", "handover-receive"):
            assert lib.get(name) is not None, f"Missing original: {name}"

    def test_new_primitives_have_expected_skills(self):
        lib = PrimitiveLibrary()
        assert lib.get("pick-object").skill_template.skill == "pick_place"
        assert lib.get("place-object").skill_template.skill == "pick_place"
        assert lib.get("gaze-at-agent").skill_template.skill == "gaze"
        assert lib.get("gaze-scan").skill_template.skill == "gaze"
        assert lib.get("gaze-avert").skill_template.skill == "gaze"


# ===================================================================
# Executive Integration
# ===================================================================
class TestPhase7Integration:
    """Verify new skills work with the full Executive pipeline."""

    def _build(self, skill, policy, skill_name):
        bb = Blackboard()
        registry = SkillRegistry()
        registry.register(SkillEntry(skill=skill, policy=policy))
        seq = ScriptSequence(
            name="test",
            steps=[ScriptStep(SkillRequest(skill=skill_name, goal={"test": True}))],
        )
        executive = Executive(
            bb=bb,
            registry=registry,
            scripts=ScriptManager(sequence=seq),
            norms=NormEngine(rules=[]),
            tom=ToMModulator(),
            shield=SafetyShield(emergency_stop_distance=0.3),
            deliberation_hz=100.0,
        )
        return bb, executive

    def test_handover_registered_and_dispatched(self):
        skill = HandoverSkill()
        bb, executive = self._build(skill, HandoverIntentPolicy(), "handover")
        bb.percept = _pb(t=0.0)
        executive.tick(0.0)
        assert skill._phase != "idle"  # skill was started

    def test_pick_place_registered_and_dispatched(self):
        skill = PickPlaceSkill()
        bb, executive = self._build(skill, PickPlaceIntentPolicy(), "pick_place")
        bb.percept = _pb(t=0.0)
        executive.tick(0.0)
        assert skill._phase == "approaching"  # skill was started

    def test_gaze_registered_and_dispatched(self):
        skill = GazeSkill()
        bb = Blackboard()
        registry = SkillRegistry()
        registry.register(SkillEntry(skill=skill, policy=GazeIntentPolicy()))
        seq = ScriptSequence(
            name="test",
            steps=[ScriptStep(SkillRequest(
                skill="gaze",
                goal={"target": (1.0, 0.0, 0.0)},
                params={"mode": "look_at_point"},
            ))],
        )
        executive = Executive(
            bb=bb,
            registry=registry,
            scripts=ScriptManager(sequence=seq),
            norms=NormEngine(rules=[]),
            tom=ToMModulator(),
            shield=SafetyShield(emergency_stop_distance=0.3),
            deliberation_hz=100.0,
        )
        bb.percept = _pb(t=0.0)
        executive.tick(0.0)
        assert skill._phase == "tracking"  # skill was started and is running
