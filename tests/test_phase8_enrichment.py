"""Phase 8: Tests for behavior tree, adaptive norms, dynamic keepout,
and context-dependent norms.
"""

from __future__ import annotations

import pytest

from architecture_core.core.blackboard import Blackboard
from architecture_core.core.registry import SkillEntry, SkillRegistry
from architecture_core.core.types import (
    NormativeConstraints,
    PerceptBundle,
    SkillRequest,
    SkillUpdate,
)
from architecture_core.core.executive import Executive
from architecture_core.cognition.norms.norm_engine import NormEngine
from architecture_core.cognition.norms.rules import (
    NormRule,
    PersonalSpaceRule,
    SpeedLimitRule,
)
from architecture_core.cognition.norms.adaptive_norms import (
    DEFAULT_PRIOR,
    ProxemicPrior,
    extract_norm_features,
)
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    ScriptTrajectory,
    TrajectoryStep,
)
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.script_composer import ScriptComposer
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.norms.dynamic_keepout import DynamicKeepoutRule
from architecture_core.cognition.norms.context_norms import (
    ContextualNormRule,
    TaskContext,
)
from architecture_core.cognition.scripts.behavior_tree import (
    ActionNode,
    ConditionNode,
    InverterNode,
    ParallelNode,
    RepeatNode,
    SelectorNode,
    SequenceNode,
    SucceederNode,
)
from architecture_core.cognition.scripts.bt_manager import (
    BehaviorTreeManager,
    tree_from_sequence,
)
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.script_types import (
    ScriptCondition,
    ScriptSequence,
    ScriptStep,
)
from architecture_core.cognition.tom.tom_modulator import ToMModulator
from architecture_core.perception.augmentations.proxemics import (
    ProxemicsAugmentation,
    classify_zone,
)
from architecture_core.safety.shield import SafetyShield
from architecture_core.skills.base import Skill
from architecture_core.cognition.tom.intent_policy import IntentPolicy


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _pb(t: float = 0.0, **world_kw) -> PerceptBundle:
    return PerceptBundle(t=t, world=world_kw)


class MockSkill(Skill):
    name = "navigate"

    def __init__(self):
        self.started = False
        self._return_status = "RUNNING"

    def start(self, req):
        self.started = True

    def tick(self, pb, update):
        return self._return_status

    def stop(self, reason=""):
        self.started = False


class MockIntentPolicy(IntentPolicy):
    def approach(self, pb, req, base):
        return base

    def avoid(self, pb, req, base):
        return base

    def yield_(self, pb, req, base):
        return base

    def wait(self, pb, req, base):
        return base


# ===================================================================
# Behavior Tree Nodes
# ===================================================================
class TestActionNode:

    def test_returns_running_initially(self):
        node = ActionNode(SkillRequest(skill="navigate"))
        assert node.tick(_pb()) == "RUNNING"

    def test_completes_on_set_result_success(self):
        node = ActionNode(SkillRequest(skill="navigate"))
        node.tick(_pb())
        node.set_result("SUCCESS")
        assert node.tick(_pb()) == "SUCCESS"

    def test_completes_on_set_result_failure(self):
        node = ActionNode(SkillRequest(skill="navigate"))
        node.tick(_pb())
        node.set_result("FAILURE")
        assert node.tick(_pb()) == "FAILURE"

    def test_reset_clears_result(self):
        node = ActionNode(SkillRequest(skill="navigate"))
        node.tick(_pb())
        node.set_result("SUCCESS")
        assert node.tick(_pb()) == "SUCCESS"
        node.reset()
        assert node.tick(_pb()) == "RUNNING"

    def test_on_activate_callback(self):
        calls = []
        node = ActionNode(
            SkillRequest(skill="navigate"),
            on_activate=lambda n: calls.append(n),
        )
        node.tick(_pb())
        assert len(calls) == 1
        # Second tick doesn't re-trigger
        node.tick(_pb())
        assert len(calls) == 1


class TestConditionNode:

    def test_true_returns_success(self):
        cond = ScriptCondition(lambda pb: True, "always true")
        node = ConditionNode(cond)
        assert node.tick(_pb()) == "SUCCESS"

    def test_false_returns_failure(self):
        cond = ScriptCondition(lambda pb: False, "always false")
        node = ConditionNode(cond)
        assert node.tick(_pb()) == "FAILURE"

    def test_reads_perceptbundle(self):
        cond = ScriptCondition(
            lambda pb: pb.world.get("door_open", False),
            "door check",
        )
        node = ConditionNode(cond)
        assert node.tick(_pb(door_open=False)) == "FAILURE"
        assert node.tick(_pb(door_open=True)) == "SUCCESS"


class TestSequenceNode:

    def test_succeeds_when_all_children_succeed(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        seq = SequenceNode([a, b])

        assert seq.tick(_pb()) == "RUNNING"  # a is running
        a.set_result("SUCCESS")
        assert seq.tick(_pb()) == "RUNNING"  # b is running
        b.set_result("SUCCESS")
        assert seq.tick(_pb()) == "SUCCESS"

    def test_fails_on_first_failure(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        seq = SequenceNode([a, b])

        seq.tick(_pb())
        a.set_result("FAILURE")
        assert seq.tick(_pb()) == "FAILURE"

    def test_resumes_running_child(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        seq = SequenceNode([a, b])

        seq.tick(_pb())  # starts a
        assert seq.tick(_pb()) == "RUNNING"  # a still running
        a.set_result("SUCCESS")
        seq.tick(_pb())  # now b starts
        assert seq.tick(_pb()) == "RUNNING"  # b still running

    def test_reset(self):
        a = ActionNode(SkillRequest(skill="a"))
        seq = SequenceNode([a])
        seq.tick(_pb())
        a.set_result("SUCCESS")
        assert seq.tick(_pb()) == "SUCCESS"
        seq.reset()
        assert seq.tick(_pb()) == "RUNNING"


class TestSelectorNode:

    def test_succeeds_on_first_success(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        sel = SelectorNode([a, b])

        sel.tick(_pb())
        a.set_result("SUCCESS")
        assert sel.tick(_pb()) == "SUCCESS"

    def test_fails_when_all_fail(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        sel = SelectorNode([a, b])

        sel.tick(_pb())
        a.set_result("FAILURE")
        sel.tick(_pb())
        b.set_result("FAILURE")
        assert sel.tick(_pb()) == "FAILURE"

    def test_resumes_running_child(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        sel = SelectorNode([a, b])

        sel.tick(_pb())
        assert sel.tick(_pb()) == "RUNNING"  # a still running

    def test_tries_next_on_failure(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        sel = SelectorNode([a, b])

        sel.tick(_pb())
        a.set_result("FAILURE")
        sel.tick(_pb())  # now tries b
        assert sel.tick(_pb()) == "RUNNING"  # b is running


class TestParallelNode:

    def test_succeed_on_all(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        par = ParallelNode([a, b], policy="succeed_on_all")

        par.tick(_pb())
        a.set_result("SUCCESS")
        assert par.tick(_pb()) == "RUNNING"  # b still running
        b.set_result("SUCCESS")
        assert par.tick(_pb()) == "SUCCESS"

    def test_succeed_on_one(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        par = ParallelNode([a, b], policy="succeed_on_one")

        par.tick(_pb())
        a.set_result("SUCCESS")
        assert par.tick(_pb()) == "SUCCESS"  # one succeeded

    def test_fail_on_any_for_succeed_on_all(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        par = ParallelNode([a, b], policy="succeed_on_all")

        par.tick(_pb())
        a.set_result("FAILURE")
        assert par.tick(_pb()) == "FAILURE"


class TestDecoratorNodes:

    def test_inverter_flips_success_to_failure(self):
        a = ActionNode(SkillRequest(skill="a"))
        inv = InverterNode(a)
        inv.tick(_pb())
        a.set_result("SUCCESS")
        assert inv.tick(_pb()) == "FAILURE"

    def test_inverter_flips_failure_to_success(self):
        a = ActionNode(SkillRequest(skill="a"))
        inv = InverterNode(a)
        inv.tick(_pb())
        a.set_result("FAILURE")
        assert inv.tick(_pb()) == "SUCCESS"

    def test_inverter_running_passes_through(self):
        a = ActionNode(SkillRequest(skill="a"))
        inv = InverterNode(a)
        assert inv.tick(_pb()) == "RUNNING"

    def test_repeat_n_times(self):
        a = ActionNode(SkillRequest(skill="a"))
        rep = RepeatNode(a, count=3)

        for i in range(2):
            rep.tick(_pb())
            a.set_result("SUCCESS")
            assert rep.tick(_pb()) == "RUNNING"  # still repeating

        rep.tick(_pb())
        a.set_result("SUCCESS")
        assert rep.tick(_pb()) == "SUCCESS"  # done after 3

    def test_repeat_fails_on_child_failure(self):
        a = ActionNode(SkillRequest(skill="a"))
        rep = RepeatNode(a, count=3)

        rep.tick(_pb())
        a.set_result("FAILURE")
        assert rep.tick(_pb()) == "FAILURE"

    def test_succeeder_always_succeeds(self):
        a = ActionNode(SkillRequest(skill="a"))
        suc = SucceederNode(a)
        suc.tick(_pb())
        a.set_result("FAILURE")
        assert suc.tick(_pb()) == "SUCCESS"

    def test_succeeder_running_passes_through(self):
        a = ActionNode(SkillRequest(skill="a"))
        suc = SucceederNode(a)
        assert suc.tick(_pb()) == "RUNNING"


# ===================================================================
# BehaviorTreeManager
# ===================================================================
class TestBehaviorTreeManager:

    def test_select_returns_active_action_request(self):
        a = ActionNode(SkillRequest(skill="navigate", goal={"x": 5}))
        mgr = BehaviorTreeManager(root=SequenceNode([a]))
        req = mgr.select(_pb())
        assert req.skill == "navigate"
        assert req.goal["x"] == 5

    def test_notify_status_advances_tree(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        mgr = BehaviorTreeManager(root=SequenceNode([a, b]))

        req = mgr.select(_pb())
        assert req.skill == "a"

        mgr.notify_status("SUCCESS")
        req = mgr.select(_pb())
        assert req.skill == "b"

    def test_sequence_full_lifecycle(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        mgr = BehaviorTreeManager(root=SequenceNode([a, b]))

        assert not mgr.is_complete
        mgr.select(_pb())
        mgr.notify_status("SUCCESS")
        mgr.select(_pb())
        mgr.notify_status("SUCCESS")
        mgr.select(_pb())
        assert mgr.is_complete

    def test_selector_with_conditions(self):
        cond_fail = ConditionNode(ScriptCondition(lambda pb: False))
        action = ActionNode(SkillRequest(skill="fallback"))
        mgr = BehaviorTreeManager(root=SelectorNode([cond_fail, action]))

        req = mgr.select(_pb())
        assert req.skill == "fallback"

    def test_tree_from_sequence_compat(self):
        seq = ScriptSequence(
            name="test",
            steps=[
                ScriptStep(SkillRequest(skill="a")),
                ScriptStep(SkillRequest(skill="b")),
            ],
        )
        root = tree_from_sequence(seq)
        mgr = BehaviorTreeManager(root=root)

        req = mgr.select(_pb())
        assert req.skill == "a"
        mgr.notify_status("SUCCESS")
        req = mgr.select(_pb())
        assert req.skill == "b"

    def test_is_complete_false_initially(self):
        a = ActionNode(SkillRequest(skill="a"))
        mgr = BehaviorTreeManager(root=SequenceNode([a]))
        assert not mgr.is_complete

    def test_is_repairing_always_false(self):
        a = ActionNode(SkillRequest(skill="a"))
        mgr = BehaviorTreeManager(root=SequenceNode([a]))
        assert not mgr.is_repairing

    def test_parallel_actions(self):
        a = ActionNode(SkillRequest(skill="a"))
        b = ActionNode(SkillRequest(skill="b"))
        mgr = BehaviorTreeManager(
            root=ParallelNode([a, b], policy="succeed_on_all")
        )
        # Both activate — manager tracks last activated
        mgr.select(_pb())
        assert not mgr.is_complete


# ===================================================================
# Norm Feature Extraction
# ===================================================================
class TestNormFeatureExtraction:

    def test_extract_interhuman_distances(self):
        pb = PerceptBundle(t=0.0, world={
            "agents": [
                {"id": "h1", "pose": (0, 0)},
                {"id": "h2", "pose": (0.8, 0)},
            ],
        })
        feats = extract_norm_features(pb)
        assert feats["mean_inter_human_distance"] == pytest.approx(0.8, abs=0.01)

    def test_extract_approach_speed(self):
        pb = PerceptBundle(t=0.0, world={
            "agents": [
                {"id": "h1", "pose": (0, 0), "velocity": (0.4, 0.3)},
            ],
        })
        feats = extract_norm_features(pb)
        assert "max_approach_speed" in feats
        assert feats["max_approach_speed"] == pytest.approx(0.5, abs=0.01)

    def test_extract_gaze_engagement(self):
        pb = PerceptBundle(t=0.0, world={}, social={
            "engagement": {
                "readings": [
                    {"entity_id": "h1", "gaze_score": 0.8},
                    {"entity_id": "h2", "gaze_score": 0.4},
                ],
            },
        })
        feats = extract_norm_features(pb)
        assert feats["mean_gaze_engagement"] == pytest.approx(0.6, abs=0.01)

    def test_extract_affect(self):
        pb = PerceptBundle(t=0.0, world={}, social={
            "affect": {
                "readings": [
                    {"entity_id": "h1", "valence": 0.5},
                    {"entity_id": "h2", "valence": -0.1},
                ],
            },
        })
        feats = extract_norm_features(pb)
        assert feats["mean_human_valence"] == pytest.approx(0.2, abs=0.01)

    def test_extract_interaction_duration(self):
        pb = PerceptBundle(t=0.0, world={}, social={
            "engagement": {
                "readings": [
                    {"entity_id": "h1", "gaze_score": 0.5, "duration": 3.5},
                    {"entity_id": "h2", "gaze_score": 0.3, "duration": 1.2},
                ],
            },
        })
        feats = extract_norm_features(pb)
        assert feats["max_interaction_duration"] == pytest.approx(3.5)

    def test_extract_no_agents_empty(self):
        pb = PerceptBundle(t=0.0, world={"agents": []})
        feats = extract_norm_features(pb)
        assert "mean_inter_human_distance" not in feats
        assert "max_approach_speed" not in feats


# ===================================================================
# Norm Discovery via ScriptRepertoire
# ===================================================================
class TestNormDiscovery:

    def test_pattern_norm_features_default_empty(self):
        p = ScriptPattern(name="test")
        assert p.norm_features == {}

    def test_trajectory_step_norm_snapshot(self):
        step = TrajectoryStep(
            t=0.0, primitive_name="approach-greet",
            norm_snapshot={"mean_inter_human_distance": 0.9},
        )
        assert step.norm_snapshot["mean_inter_human_distance"] == pytest.approx(0.9)

    def test_set_norm_snapshot_stored(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        rep.set_norm_snapshot({"mean_inter_human_distance": 1.0})
        assert rep._norm_snapshot["mean_inter_human_distance"] == pytest.approx(1.0)

    def test_norm_features_accumulate_on_completion(self):
        """After script completes, pattern.norm_features updated from trajectory."""
        lib = PrimitiveLibrary()
        cfg = RepertoireConfig(
            norm_learning_rate=1.0,  # full adoption for test clarity
            norm_min_learning_rate=1.0,
        )
        rep = ScriptRepertoire(lib, config=cfg)

        # Create a pattern manually
        pattern = ScriptPattern(
            name="test_pattern",
            primitives_sequence=["approach-greet"],
            precision=0.1,
            situation_affinity={"open_area": 1.0},
        )
        rep._patterns["test_pattern"] = pattern
        rep._active_pattern_name = "test_pattern"

        # Set norm snapshot and record a step
        rep.set_norm_snapshot({"mean_inter_human_distance": 0.7})
        rep._tracker.begin_trajectory("test_pattern", "open_area")
        rep.on_step_completed(
            t=0.0, primitive_name="approach-greet",
            situation_belief={"open_area": 1.0}, outcome="SUCCESS",
        )
        rep.on_script_completed("SUCCESS")

        # Pattern should have acquired norm_features
        assert "mean_inter_human_distance" in pattern.norm_features

    def test_norm_features_ema_update(self):
        """Repeated completions refine norm_features — precision gates the rate."""
        lib = PrimitiveLibrary()
        cfg = RepertoireConfig(
            norm_learning_rate=0.5,
            norm_min_learning_rate=0.02,
        )
        rep = ScriptRepertoire(lib, config=cfg)

        pattern = ScriptPattern(
            name="test",
            primitives_sequence=["approach-greet"],
            precision=0.1,
            situation_affinity={"open_area": 1.0},
            norm_features={"mean_inter_human_distance": 1.2},
        )
        rep._patterns["test"] = pattern

        # First trajectory: observe 0.6m
        rep._active_pattern_name = "test"
        rep.set_norm_snapshot({"mean_inter_human_distance": 0.6})
        rep._tracker.begin_trajectory("test", "open_area")
        rep.on_step_completed(
            t=0.0, primitive_name="approach-greet",
            situation_belief={"open_area": 1.0}, outcome="SUCCESS",
        )
        rep.on_script_completed("SUCCESS")

        # Should have moved toward 0.6 but not jumped there
        d = pattern.norm_features["mean_inter_human_distance"]
        assert d < 1.2
        assert d > 0.6

    def test_precision_gates_norm_learning_rate(self):
        """High-precision patterns resist norm change."""
        lib = PrimitiveLibrary()
        cfg = RepertoireConfig(
            norm_learning_rate=0.5,
            norm_min_learning_rate=0.02,
        )

        # Low precision pattern
        low_p = ScriptPattern(
            name="low", primitives_sequence=["approach-greet"],
            precision=0.1, situation_affinity={"open_area": 1.0},
            norm_features={"mean_inter_human_distance": 1.2},
        )
        # High precision pattern
        high_p = ScriptPattern(
            name="high", primitives_sequence=["approach-greet"],
            precision=5.0, situation_affinity={"open_area": 1.0},
            norm_features={"mean_inter_human_distance": 1.2},
        )

        for pattern in [low_p, high_p]:
            rep = ScriptRepertoire(lib, config=cfg)
            rep._patterns[pattern.name] = pattern
            rep._active_pattern_name = pattern.name
            rep.set_norm_snapshot({"mean_inter_human_distance": 0.6})
            rep._tracker.begin_trajectory(pattern.name, "open_area")
            rep.on_step_completed(
                t=0.0, primitive_name="approach-greet",
                situation_belief={"open_area": 1.0}, outcome="SUCCESS",
            )
            rep.on_script_completed("SUCCESS")

        # Low precision moved more
        low_delta = abs(1.2 - low_p.norm_features["mean_inter_human_distance"])
        high_delta = abs(1.2 - high_p.norm_features["mean_inter_human_distance"])
        assert low_delta > high_delta

    def test_composition_seeds_norm_features(self):
        lib = PrimitiveLibrary()
        composer = ScriptComposer(lib)
        snapshot = {"mean_inter_human_distance": 0.8, "max_approach_speed": 0.4}
        pattern = composer.compose(
            "open_area", {"open_area": 1.0}, norm_snapshot=snapshot,
        )
        assert pattern.norm_features["mean_inter_human_distance"] == pytest.approx(0.8)
        assert pattern.norm_features["max_approach_speed"] == pytest.approx(0.4)

    def test_active_norm_features_property(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        pattern = ScriptPattern(
            name="test", primitives_sequence=["approach-greet"],
            norm_features={"mean_inter_human_distance": 0.9},
        )
        rep._patterns["test"] = pattern
        rep._active_pattern_name = "test"
        assert rep.active_norm_features["mean_inter_human_distance"] == pytest.approx(0.9)

    def test_active_norm_features_empty_when_no_pattern(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        assert rep.active_norm_features == {}


# ===================================================================
# Adaptive Rules
# ===================================================================
class TestAdaptiveRules:

    def test_personal_space_set_norm_features(self):
        rule = PersonalSpaceRule(min_distance=1.2, veto_distance=0.45)
        rule.set_norm_features({"mean_inter_human_distance": 0.8})
        assert rule.min_distance == pytest.approx(0.8)
        assert rule.veto_distance == pytest.approx(0.8 * 0.375)

    def test_speed_limit_set_norm_features(self):
        rule = SpeedLimitRule(max_speed=0.5)
        rule.set_norm_features({"max_approach_speed": 0.3})
        assert rule.max_speed == pytest.approx(0.3)

    def test_from_profile_with_proxemic_prior(self):
        prior = ProxemicPrior(personal_distance=1.5, intimate_distance=0.6,
                              social_distance=3.0)
        aug = ProxemicsAugmentation.from_profile(prior)
        assert aug.thresholds["personal"] == pytest.approx(1.5)
        assert aug.thresholds["intimate"] == pytest.approx(0.6)

        rule = PersonalSpaceRule.from_profile(prior)
        assert rule.min_distance == pytest.approx(1.5)
        assert rule.veto_distance == pytest.approx(0.6)

    def test_set_norm_features_affects_evaluation(self):
        rule = PersonalSpaceRule(min_distance=1.0, veto_distance=0.4)
        pb = PerceptBundle(t=0.0, social={"proxemics": {"closest_distance": 0.7}})
        # 0.7 < 1.0 → speed cap
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert "speed_cap" in result.hard

        # After norm discovery: comfortable distance is 0.5m
        rule.set_norm_features({"mean_inter_human_distance": 0.5})
        result2 = rule.evaluate(pb, SkillRequest(skill="navigate"))
        # 0.7 > 0.5 (new min_distance) → no constraint
        assert result2.veto is None
        assert "speed_cap" not in result2.hard

    def test_backward_compat_without_norm_features(self):
        """Rules work normally without set_norm_features ever called."""
        rule = PersonalSpaceRule(min_distance=1.0, veto_distance=0.4)
        pb = PerceptBundle(t=0.0, social={"proxemics": {"closest_distance": 0.6}})
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert "speed_cap" in result.hard

    def test_zone_classification_with_prior(self):
        prior = ProxemicPrior(intimate_distance=0.15, personal_distance=0.5,
                              social_distance=1.5)
        thresholds = {
            "intimate": prior.intimate_distance,
            "personal": prior.personal_distance,
            "social": prior.social_distance,
        }
        assert classify_zone(0.3, thresholds) == "personal"
        assert classify_zone(0.3) == "intimate"  # default thresholds


# ===================================================================
# Dynamic Keepout Zones
# ===================================================================
class TestDynamicKeepout:

    def test_obstacles_create_zones(self):
        rule = DynamicKeepoutRule(obstacle_margin=0.5)
        pb = _pb(
            robot_pose=(10, 10, 0),
            obstacles=[{"pose": (3, 3), "radius": 0.3}],
        )
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert "keepout_zones" in result.hard
        assert len(result.hard["keepout_zones"]) == 1
        assert result.hard["keepout_zones"][0]["radius"] == pytest.approx(0.8)

    def test_hazards_create_zones(self):
        rule = DynamicKeepoutRule(hazard_margin=1.0)
        pb = _pb(
            robot_pose=(10, 10, 0),
            hazards=[{"pose": (5, 5), "radius": 0.5}],
        )
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        zones = result.hard["keepout_zones"]
        assert len(zones) == 1
        assert zones[0]["radius"] == pytest.approx(1.5)
        assert zones[0]["source"] == "hazard"

    def test_margin_added_to_radius(self):
        rule = DynamicKeepoutRule(obstacle_margin=0.3)
        pb = _pb(
            robot_pose=(10, 10, 0),
            obstacles=[{"pose": (5, 5), "radius": 1.0}],
        )
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert result.hard["keepout_zones"][0]["radius"] == pytest.approx(1.3)

    def test_veto_when_inside_zone(self):
        rule = DynamicKeepoutRule(obstacle_margin=0.5)
        pb = _pb(
            robot_pose=(3, 3, 0),  # inside the obstacle zone
            obstacles=[{"pose": (3, 3), "radius": 0.3}],
        )
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert result.veto is not None
        assert "keepout" in result.veto.lower()

    def test_no_zones_no_constraints(self):
        rule = DynamicKeepoutRule()
        pb = _pb(robot_pose=(0, 0, 0))
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert result.veto is None
        assert "keepout_zones" not in result.hard

    def test_multiple_obstacles(self):
        rule = DynamicKeepoutRule(obstacle_margin=0.5)
        pb = _pb(
            robot_pose=(10, 10, 0),
            obstacles=[
                {"pose": (1, 1), "radius": 0.2},
                {"pose": (5, 5), "radius": 0.4},
            ],
        )
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert len(result.hard["keepout_zones"]) == 2


# ===================================================================
# Contextual Norms
# ===================================================================
class TestContextualNorms:

    def _make_rule(self):
        factory_ctx = TaskContext(
            name="factory",
            situations=["workstation", "hazard"],
            rules=[SpeedLimitRule(max_speed=0.2)],
        )
        social_ctx = TaskContext(
            name="social_area",
            situations=["interaction", "meeting_point"],
            rules=[SpeedLimitRule(max_speed=0.8)],
        )
        return ContextualNormRule(
            contexts=[factory_ctx, social_ctx],
            default_context=None,
        )

    def test_default_no_constraints(self):
        rule = self._make_rule()
        result = rule.evaluate(_pb(), SkillRequest(skill="navigate"))
        assert result.veto is None
        assert not result.hard

    def test_situation_activates_context(self):
        rule = self._make_rule()
        rule.set_situation("workstation")
        assert rule.active_context == "factory"

    def test_context_rules_applied(self):
        rule = self._make_rule()
        rule.set_situation("workstation")
        result = rule.evaluate(_pb(), SkillRequest(skill="navigate"))
        assert result.hard["max_linear_speed"] == pytest.approx(0.2)

    def test_different_situations_different_rules(self):
        rule = self._make_rule()

        rule.set_situation("workstation")
        r1 = rule.evaluate(_pb(), SkillRequest(skill="navigate"))

        rule.set_situation("interaction")
        r2 = rule.evaluate(_pb(), SkillRequest(skill="navigate"))

        assert r1.hard["max_linear_speed"] < r2.hard["max_linear_speed"]

    def test_unknown_situation_keeps_current(self):
        rule = self._make_rule()
        rule.set_situation("workstation")
        rule.set_situation("unknown_place")
        # Unknown doesn't match any context, so active stays "factory"
        assert rule.active_context == "factory"

    def test_context_with_veto_rule(self):
        veto_rule = _VetoRule()
        ctx = TaskContext(
            name="danger",
            situations=["hazard"],
            rules=[veto_rule],
        )
        rule = ContextualNormRule(contexts=[ctx])
        rule.set_situation("hazard")
        result = rule.evaluate(_pb(), SkillRequest(skill="navigate"))
        assert result.veto is not None


class _VetoRule(NormRule):
    name = "test_veto"

    def evaluate(self, pb, candidate):
        return NormativeConstraints(veto="test veto")


# ===================================================================
# Integration
# ===================================================================
class TestPhase8Integration:

    def _build_executive(self, scripts, norms_rules=None):
        bb = Blackboard()
        registry = SkillRegistry()
        skill = MockSkill()
        registry.register(SkillEntry(skill=skill, policy=MockIntentPolicy()))
        executive = Executive(
            bb=bb,
            registry=registry,
            scripts=scripts,
            norms=NormEngine(rules=norms_rules or []),
            tom=ToMModulator(),
            shield=SafetyShield(emergency_stop_distance=0.3),
            deliberation_hz=100.0,
        )
        return bb, executive, skill

    def test_bt_manager_with_executive(self):
        a = ActionNode(SkillRequest(skill="navigate", goal={"x": 1}))
        mgr = BehaviorTreeManager(root=SequenceNode([a]))
        bb, executive, skill = self._build_executive(mgr)
        bb.percept = _pb(t=0.0)
        executive.tick(0.0)
        assert skill.started

    def test_norm_discovery_end_to_end(self):
        """Pattern norm_features → PersonalSpaceRule → adapted constraints."""
        rule = PersonalSpaceRule(min_distance=1.2, veto_distance=0.45)
        # Simulate pattern that learned norm_features from observations
        rule.set_norm_features({"mean_inter_human_distance": 0.5})

        # Agent at 0.4m — inside new min_distance (0.5) but outside veto
        pb = PerceptBundle(
            t=0.0,
            social={"proxemics": {"closest_distance": 0.4}},
        )
        result = rule.evaluate(pb, SkillRequest(skill="navigate"))
        assert "speed_cap" in result.hard
        # 0.4 > veto (0.5*0.375=0.1875) → no veto
        assert result.veto is None

    def test_executive_pushes_norm_features(self):
        """Executive extracts norm snapshot and pushes to repertoire."""
        seq = ScriptSequence(
            name="test",
            steps=[ScriptStep(SkillRequest(skill="navigate", goal={"x": 1}))],
        )
        rule = PersonalSpaceRule(min_distance=1.2, veto_distance=0.45)
        bb, executive, skill = self._build_executive(
            ScriptManager(sequence=seq),
            norms_rules=[rule],
        )
        # Scene: two humans at 0.8m apart
        bb.percept = PerceptBundle(
            t=0.0,
            world={
                "agents": [
                    {"id": "h1", "pose": (0, 0)},
                    {"id": "h2", "pose": (0.8, 0)},
                ],
            },
            social={"proxemics": {"closest_distance": 2.0}},
        )
        executive.tick(0.0)
        # Norm snapshot should have been extracted
        assert "mean_inter_human_distance" in bb.norm_snapshot
        assert bb.norm_snapshot["mean_inter_human_distance"] == pytest.approx(0.8, abs=0.01)

    def test_dynamic_keepout_with_executive(self):
        seq = ScriptSequence(
            name="test",
            steps=[ScriptStep(SkillRequest(skill="navigate", goal={"x": 1}))],
        )
        rule = DynamicKeepoutRule(obstacle_margin=0.5)
        bb, executive, skill = self._build_executive(
            ScriptManager(sequence=seq),
            norms_rules=[rule],
        )
        # Robot inside obstacle zone → veto
        bb.percept = _pb(t=0.0, robot_pose=(1, 1, 0),
                         obstacles=[{"pose": (1, 1), "radius": 0.5}])
        executive.tick(0.0)
        assert bb.norms.veto is not None

    def test_backward_compat_script_manager_unchanged(self):
        """Existing ScriptManager still works with Executive."""
        seq = ScriptSequence(
            name="test",
            steps=[ScriptStep(SkillRequest(skill="navigate", goal={"x": 1}))],
        )
        bb, executive, skill = self._build_executive(ScriptManager(sequence=seq))
        bb.percept = _pb(t=0.0)
        executive.tick(0.0)
        assert skill.started

    def test_bt_selector_with_condition_and_executive(self):
        """BT with condition node used for conditional branching."""
        # If door is open → go through, else → go around
        door_check = ConditionNode(
            ScriptCondition(lambda pb: pb.world.get("door_open", False))
        )
        go_through = ActionNode(
            SkillRequest(skill="navigate", goal={"x": 5, "y": 0})
        )
        go_around = ActionNode(
            SkillRequest(skill="navigate", goal={"x": 5, "y": 3})
        )

        tree = SelectorNode([
            SequenceNode([door_check, go_through]),
            go_around,
        ])
        mgr = BehaviorTreeManager(root=tree)

        # Door closed → condition fails → selector tries go_around
        req = mgr.select(_pb(door_open=False))
        assert req.goal.get("y") == 3  # go_around path

    def test_bt_selector_condition_true_path(self):
        door_check = ConditionNode(
            ScriptCondition(lambda pb: pb.world.get("door_open", False))
        )
        go_through = ActionNode(
            SkillRequest(skill="navigate", goal={"x": 5, "y": 0})
        )
        go_around = ActionNode(
            SkillRequest(skill="navigate", goal={"x": 5, "y": 3})
        )

        tree = SelectorNode([
            SequenceNode([door_check, go_through]),
            go_around,
        ])
        mgr = BehaviorTreeManager(root=tree)

        # Door open → condition passes → go_through
        req = mgr.select(_pb(door_open=True))
        assert req.goal.get("y") == 0  # go_through path
