"""Phase 9: Predictive Safety — velocity prediction + multi-agent collision.

Tests for:
- compute_ttc: quadratic TTC computation
- speed_reduction_factor: linear speed ramp
- VelocityPredictor: multi-pair prediction engine
- PredictiveCollisionRule: NormRule producing graduated constraints
- SafetyShield with VelocityPredictor: velocity-aware emergency stop
- Integration: full pipeline with Executive
"""

import math

import pytest

from architecture_core.core.types import (
    NormativeConstraints,
    PerceptBundle,
    SkillRequest,
    SkillUpdate,
)
from architecture_core.safety.velocity_predictor import (
    CollisionPrediction,
    CollisionRisk,
    VelocityPredictor,
    compute_ttc,
    speed_reduction_factor,
)
from architecture_core.cognition.norms.predictive_collision import (
    PredictiveCollisionRule,
)
from architecture_core.safety.shield import SafetyShield


# ===================================================================
# Helpers
# ===================================================================

def _pb(t=0.0, robot_pose=(0, 0, 0, 0), agents=None, obstacles=None,
        closest_distance=float("inf")):
    world = {"robot_pose": robot_pose}
    if agents is not None:
        world["agents"] = agents
    if obstacles is not None:
        world["obstacles"] = obstacles
    social = {"proxemics": {"closest_distance": closest_distance}}
    return PerceptBundle(t=t, world=world, social=social)


# ===================================================================
# TestComputeTTC
# ===================================================================

class TestComputeTTC:
    def test_head_on_collision(self):
        # Two entities 2m apart, approaching each other at 1m/s each
        ttc, closing, cp = compute_ttc(
            0, 0, 1, 0, 0.25,   # entity A at origin, moving right
            2, 0, -1, 0, 0.25,  # entity B at (2,0), moving left
        )
        # They close at 2m/s, need to cover 2-0.5=1.5m → TTC ≈ 0.75s
        assert ttc == pytest.approx(0.75, abs=0.01)
        assert closing > 0  # they are closing

    def test_parallel_paths_no_collision(self):
        # Moving in same direction, same speed, 2m apart
        ttc, closing, cp = compute_ttc(
            0, 0, 1, 0, 0.25,
            0, 2, 1, 0, 0.25,
        )
        assert ttc == float("inf")

    def test_diverging_entities(self):
        # Moving apart
        ttc, closing, cp = compute_ttc(
            0, 0, -1, 0, 0.25,
            2, 0, 1, 0, 0.25,
        )
        assert ttc == float("inf")
        assert closing < 0  # diverging

    def test_stationary_separated(self):
        ttc, closing, cp = compute_ttc(
            0, 0, 0, 0, 0.25,
            5, 0, 0, 0, 0.25,
        )
        assert ttc == float("inf")

    def test_stationary_overlapping(self):
        ttc, closing, cp = compute_ttc(
            0, 0, 0, 0, 0.25,
            0.3, 0, 0, 0, 0.25,
        )
        assert ttc == 0.0

    def test_one_stationary_one_approaching(self):
        # Agent at (3,0) moving left at 1m/s toward stationary robot
        ttc, closing, cp = compute_ttc(
            0, 0, 0, 0, 0.25,
            3, 0, -1, 0, 0.25,
        )
        # Need to cover 3-0.5=2.5m at 1m/s → TTC = 2.5s
        assert ttc == pytest.approx(2.5, abs=0.01)
        assert closing > 0

    def test_perpendicular_miss(self):
        # A at (0,0) moving right, B at (10, 0.6) moving left
        # They pass but miss (offset > combined radius)
        ttc, closing, cp = compute_ttc(
            0, 0, 1, 0, 0.25,
            10, 0.6, -1, 0, 0.25,
        )
        assert ttc == float("inf")

    def test_perpendicular_hit(self):
        # Cross paths and collide
        ttc, closing, cp = compute_ttc(
            0, 0, 1, 0, 0.25,
            2, 0.3, -1, 0, 0.25,
        )
        assert ttc < float("inf")
        assert ttc > 0


# ===================================================================
# TestSpeedReductionFactor
# ===================================================================

class TestSpeedReductionFactor:
    def test_at_stop_threshold(self):
        assert speed_reduction_factor(0.5, 0.5, 3.0) == pytest.approx(0.0)

    def test_at_caution_threshold(self):
        assert speed_reduction_factor(3.0, 0.5, 3.0) == pytest.approx(1.0)

    def test_between_thresholds(self):
        # Midpoint: (0.5+3.0)/2 = 1.75 → factor = (1.75-0.5)/(3.0-0.5) = 0.5
        f = speed_reduction_factor(1.75, 0.5, 3.0)
        assert f == pytest.approx(0.5, abs=0.01)

    def test_below_stop(self):
        assert speed_reduction_factor(0.1, 0.5, 3.0) == pytest.approx(0.0)

    def test_above_caution(self):
        assert speed_reduction_factor(10.0, 0.5, 3.0) == pytest.approx(1.0)


# ===================================================================
# TestVelocityPredictor
# ===================================================================

class TestVelocityPredictor:
    def test_no_agents_no_risks(self):
        pred = VelocityPredictor()
        pb = _pb(t=1.0)
        result = pred.predict(pb)
        assert len(result.risks) == 0
        assert result.min_robot_ttc == float("inf")

    def test_robot_velocity_estimation(self):
        pred = VelocityPredictor()
        # First tick: establish position
        pb1 = _pb(t=0.0, robot_pose=(0, 0, 0, 0))
        pred.predict(pb1)
        # Second tick: robot moved 1m in x over 1s
        pb2 = _pb(t=1.0, robot_pose=(1, 0, 0, 0))
        result = pred.predict(pb2)
        assert result.robot_velocity[0] == pytest.approx(1.0, abs=0.01)
        assert result.robot_velocity[1] == pytest.approx(0.0, abs=0.01)

    def test_robot_agent_collision(self):
        pred = VelocityPredictor()
        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (3, 0), "velocity": (-1, 0)},
        ])
        result = pred.predict(pb)
        assert len(result.risks) == 1
        assert result.risks[0].risk_type == "robot_agent"
        assert result.risks[0].entity_b == "h1"
        assert result.min_robot_ttc < float("inf")

    def test_robot_agent_diverging(self):
        pred = VelocityPredictor()
        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (3, 0), "velocity": (1, 0)},
        ])
        result = pred.predict(pb)
        # Agent moving away from stationary robot — no risk within horizon
        assert result.min_robot_ttc == float("inf")

    def test_robot_obstacle_collision(self):
        pred = VelocityPredictor()
        # Establish velocity: robot moving right at 1m/s
        pred.predict(_pb(t=0.0, robot_pose=(0, 0, 0, 0)))
        pb = _pb(t=1.0, robot_pose=(1, 0, 0, 0), obstacles=[
            {"id": "wall", "pose": (4, 0), "radius": 0.3},
        ])
        result = pred.predict(pb)
        robot_obs_risks = [r for r in result.risks if r.risk_type == "robot_obstacle"]
        assert len(robot_obs_risks) == 1

    def test_agent_agent_collision(self):
        pred = VelocityPredictor()
        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (0, 3), "velocity": (0, -1)},
            {"id": "h2", "pose": (0, -3), "velocity": (0, 1)},
        ])
        result = pred.predict(pb)
        aa_risks = [r for r in result.risks if r.risk_type == "agent_agent"]
        assert len(aa_risks) == 1
        assert result.min_agent_agent_ttc < float("inf")

    def test_agent_velocity_from_bundle(self):
        pred = VelocityPredictor()
        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (3, 0), "velocity": (-2, 0)},
        ])
        result = pred.predict(pb)
        # Agent approaching at 2m/s, distance 3m, combined radius 0.5m
        # TTC = (3-0.5)/2 = 1.25s
        assert result.min_robot_ttc == pytest.approx(1.25, abs=0.05)

    def test_agent_no_velocity_stationary(self):
        pred = VelocityPredictor()
        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (3, 0)},  # no velocity key
        ])
        result = pred.predict(pb)
        # Both stationary, separated → no collision
        assert result.min_robot_ttc == float("inf")

    def test_risks_sorted_by_ttc(self):
        pred = VelocityPredictor()
        pb = _pb(t=0.0, agents=[
            {"id": "far", "pose": (5, 0), "velocity": (-1, 0)},
            {"id": "near", "pose": (2, 0), "velocity": (-1, 0)},
        ])
        result = pred.predict(pb)
        assert len(result.risks) >= 2
        # Should be sorted ascending by TTC
        for i in range(len(result.risks) - 1):
            assert result.risks[i].ttc <= result.risks[i + 1].ttc

    def test_max_horizon_filtering(self):
        pred = VelocityPredictor(max_horizon=2.0)
        pb = _pb(t=0.0, agents=[
            # Agent far away, slow approach → TTC > 2s
            {"id": "h1", "pose": (10, 0), "velocity": (-0.5, 0)},
        ])
        result = pred.predict(pb)
        # TTC = (10-0.5)/0.5 = 19s → beyond horizon
        assert result.min_robot_ttc == float("inf")


# ===================================================================
# TestPredictiveCollisionRule
# ===================================================================

class TestPredictiveCollisionRule:
    def test_no_agents_no_constraints(self):
        rule = PredictiveCollisionRule()
        pb = _pb(t=0.0)
        result = rule.evaluate(pb, SkillRequest(skill="nav"))
        assert result.veto is None
        assert "speed_cap" not in result.hard

    def test_approaching_agent_speed_cap(self):
        rule = PredictiveCollisionRule(ttc_stop=0.5, ttc_caution=3.0)
        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (2, 0), "velocity": (-1, 0)},
        ])
        result = rule.evaluate(pb, SkillRequest(skill="nav"))
        assert "speed_cap" in result.hard
        assert result.hard["speed_cap"] < 1.0

    def test_imminent_collision_veto(self):
        rule = PredictiveCollisionRule(veto_ttc=0.3)
        # Agent very close and fast → TTC < 0.3s
        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (0.8, 0), "velocity": (-5, 0)},
        ])
        result = rule.evaluate(pb, SkillRequest(skill="nav"))
        assert result.veto is not None
        assert "Imminent collision" in result.veto

    def test_caution_zone_partial_speed(self):
        rule = PredictiveCollisionRule(ttc_stop=0.5, ttc_caution=3.0)
        # Agent approaching → TTC between 0.5 and 3.0
        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (3, 0), "velocity": (-1, 0)},
        ])
        result = rule.evaluate(pb, SkillRequest(skill="nav"))
        cap = result.hard.get("speed_cap", 1.0)
        assert 0.0 < cap < 1.0

    def test_agent_agent_near_robot_yield(self):
        rule = PredictiveCollisionRule(agent_collision_proximity=3.0)
        # Two agents about to collide near robot
        pb = _pb(t=0.0, robot_pose=(0, 0, 0, 0), agents=[
            {"id": "h1", "pose": (1, 1), "velocity": (0, -1)},
            {"id": "h2", "pose": (1, -1), "velocity": (0, 1)},
        ])
        result = rule.evaluate(pb, SkillRequest(skill="nav"))
        assert "yield_agents" in result.soft
        yield_set = set(result.soft["yield_agents"])
        assert "h1" in yield_set
        assert "h2" in yield_set

    def test_agent_agent_far_no_yield(self):
        rule = PredictiveCollisionRule(agent_collision_proximity=2.0)
        # Two agents about to collide far from robot
        pb = _pb(t=0.0, robot_pose=(0, 0, 0, 0), agents=[
            {"id": "h1", "pose": (20, 1), "velocity": (0, -1)},
            {"id": "h2", "pose": (20, -1), "velocity": (0, 1)},
        ])
        result = rule.evaluate(pb, SkillRequest(skill="nav"))
        assert "yield_agents" not in result.soft

    def test_prediction_data_in_hard(self):
        rule = PredictiveCollisionRule()
        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (2, 0), "velocity": (-1, 0)},
        ])
        result = rule.evaluate(pb, SkillRequest(skill="nav"))
        assert "predictive_collision" in result.hard
        data = result.hard["predictive_collision"]
        assert "min_robot_ttc" in data
        assert data["num_risks"] >= 1

    def test_merges_with_norm_engine(self):
        from architecture_core.cognition.norms.norm_engine import NormEngine
        from architecture_core.cognition.norms.rules import SpeedLimitRule

        rule = PredictiveCollisionRule(ttc_stop=0.5, ttc_caution=3.0)
        speed_rule = SpeedLimitRule(max_speed=0.8)
        engine = NormEngine(rules=[speed_rule, rule])

        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (2, 0), "velocity": (-1, 0)},
        ])
        result = engine.evaluate(pb, SkillRequest(skill="nav"))
        # Both rules should have contributed — speed_cap should be <= 0.8
        # and prediction should have reduced it further
        cap = result.hard.get("speed_cap", result.hard.get("max_linear_speed", 1.0))
        assert cap <= 0.8


# ===================================================================
# TestShieldWithPredictor
# ===================================================================

class TestShieldWithPredictor:
    def test_distance_emergency_still_works(self):
        shield = SafetyShield(
            emergency_stop_distance=0.3,
            predictor=VelocityPredictor(),
        )
        pb = _pb(t=0.0, closest_distance=0.2)
        update = SkillUpdate(intent="approach", params={"speed_scale": 1.0})
        result = shield.apply(pb, SkillRequest(skill="nav"), update,
                              NormativeConstraints())
        assert result.intent == "wait"
        assert result.params["speed_scale"] == 0.0
        assert result.constraints.get("emergency_stop") is True

    def test_velocity_emergency_stop(self):
        predictor = VelocityPredictor()
        shield = SafetyShield(
            predictor=predictor,
            emergency_ttc=0.5,
        )
        # Agent approaching very fast → TTC < 0.5s
        pb = _pb(t=0.0, closest_distance=5.0, agents=[
            {"id": "h1", "pose": (1, 0), "velocity": (-5, 0)},
        ])
        update = SkillUpdate(intent="approach", params={"speed_scale": 1.0})
        result = shield.apply(pb, SkillRequest(skill="nav"), update,
                              NormativeConstraints())
        assert result.intent == "wait"
        assert result.params["speed_scale"] == 0.0
        assert result.constraints.get("emergency_stop") is True
        assert "predictive-stop" in result.debug

    def test_predictive_brake_reduces_speed(self):
        predictor = VelocityPredictor()
        shield = SafetyShield(
            predictor=predictor,
            emergency_ttc=0.5,
        )
        # Agent approaching moderately → TTC between 0.5 and 2.0
        pb = _pb(t=0.0, closest_distance=5.0, agents=[
            {"id": "h1", "pose": (2, 0), "velocity": (-1, 0)},
        ])
        update = SkillUpdate(intent="approach", params={"speed_scale": 1.0})
        result = shield.apply(pb, SkillRequest(skill="nav"), update,
                              NormativeConstraints())
        assert result.params["speed_scale"] < 1.0
        assert "predictive-brake" in result.debug

    def test_no_predictor_unchanged(self):
        shield = SafetyShield(emergency_stop_distance=0.3)
        pb = _pb(t=0.0, closest_distance=5.0, agents=[
            {"id": "h1", "pose": (2, 0), "velocity": (-1, 0)},
        ])
        update = SkillUpdate(intent="approach", params={"speed_scale": 0.8})
        result = shield.apply(pb, SkillRequest(skill="nav"), update,
                              NormativeConstraints())
        # Without predictor, agent velocity doesn't matter
        assert result.params["speed_scale"] == pytest.approx(0.8)

    def test_distance_takes_precedence(self):
        predictor = VelocityPredictor()
        shield = SafetyShield(
            emergency_stop_distance=0.3,
            predictor=predictor,
            emergency_ttc=0.5,
        )
        # Both distance AND velocity should trigger stop —
        # distance check fires first
        pb = _pb(t=0.0, closest_distance=0.2, agents=[
            {"id": "h1", "pose": (0.2, 0), "velocity": (-5, 0)},
        ])
        update = SkillUpdate(intent="approach", params={"speed_scale": 1.0})
        result = shield.apply(pb, SkillRequest(skill="nav"), update,
                              NormativeConstraints())
        assert result.constraints.get("emergency_stop") is True
        assert "emergency-stop" in result.debug
        # Should have distance-based, not predictive
        assert "predictive-stop" not in result.debug

    def test_norm_speed_cap_still_enforced(self):
        predictor = VelocityPredictor()
        shield = SafetyShield(predictor=predictor)
        pb = _pb(t=0.0, closest_distance=10.0)
        update = SkillUpdate(intent="approach", params={"speed_scale": 1.0})
        norms = NormativeConstraints(hard={"max_linear_speed": 0.4})
        result = shield.apply(pb, SkillRequest(skill="nav"), update, norms)
        assert result.params["speed_scale"] == pytest.approx(0.4)

    def test_debug_annotations(self):
        predictor = VelocityPredictor()
        shield = SafetyShield(predictor=predictor, emergency_ttc=0.5)
        # Agent approaching → preemptive brake zone
        pb = _pb(t=0.0, closest_distance=5.0, agents=[
            {"id": "h1", "pose": (2, 0), "velocity": (-1, 0)},
        ])
        update = SkillUpdate(intent="approach", params={"speed_scale": 1.0})
        result = shield.apply(pb, SkillRequest(skill="nav"), update,
                              NormativeConstraints())
        assert "SHIELD" in result.debug


# ===================================================================
# TestIntegration
# ===================================================================

class TestIntegration:
    def test_collision_prediction_on_blackboard(self):
        from architecture_core.core.blackboard import Blackboard
        from architecture_core.core.registry import SkillRegistry, SkillEntry
        from architecture_core.core.executive import Executive
        from architecture_core.cognition.norms.norm_engine import NormEngine
        from architecture_core.cognition.scripts.script_manager import ScriptManager
        from architecture_core.cognition.tom.tom_modulator import ToMModulator
        from architecture_core.skills.base import Skill
        from architecture_core.cognition.tom.intent_policy import IntentPolicy

        class _Skill(Skill):
            name = "navigate"
            def start(self, req): pass
            def tick(self, pb, update): return "RUNNING"
            def stop(self, reason=""): pass

        class _Policy(IntentPolicy):
            def approach(self, pb, req, base): return base
            def avoid(self, pb, req, base): return base
            def yield_(self, pb, req, base): return base
            def wait(self, pb, req, base): return base

        bb = Blackboard()
        registry = SkillRegistry()
        registry.register(SkillEntry(skill=_Skill(), policy=_Policy()))

        rule = PredictiveCollisionRule()
        executive = Executive(
            bb=bb, registry=registry,
            scripts=ScriptManager(),
            norms=NormEngine(rules=[rule]),
            tom=ToMModulator(),
            shield=SafetyShield(),
            deliberation_hz=100.0,
        )

        bb.percept = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (2, 0), "velocity": (-1, 0)},
        ])
        executive.tick(0.0)
        assert bb.collision_prediction is not None
        assert bb.collision_prediction.min_robot_ttc < float("inf")

    def test_no_rule_blackboard_none(self):
        from architecture_core.core.blackboard import Blackboard
        from architecture_core.core.registry import SkillRegistry, SkillEntry
        from architecture_core.core.executive import Executive
        from architecture_core.cognition.norms.norm_engine import NormEngine
        from architecture_core.cognition.scripts.script_manager import ScriptManager
        from architecture_core.cognition.tom.tom_modulator import ToMModulator
        from architecture_core.skills.base import Skill
        from architecture_core.cognition.tom.intent_policy import IntentPolicy

        class _Skill(Skill):
            name = "navigate"
            def start(self, req): pass
            def tick(self, pb, update): return "RUNNING"
            def stop(self, reason=""): pass

        class _Policy(IntentPolicy):
            def approach(self, pb, req, base): return base
            def avoid(self, pb, req, base): return base
            def yield_(self, pb, req, base): return base
            def wait(self, pb, req, base): return base

        bb = Blackboard()
        registry = SkillRegistry()
        registry.register(SkillEntry(skill=_Skill(), policy=_Policy()))

        executive = Executive(
            bb=bb, registry=registry,
            scripts=ScriptManager(),
            norms=NormEngine(rules=[]),
            tom=ToMModulator(),
            shield=SafetyShield(),
            deliberation_hz=100.0,
        )

        bb.percept = _pb(t=0.0)
        executive.tick(0.0)
        assert bb.collision_prediction is None

    def test_full_pipeline_speed_reduction(self):
        from architecture_core.core.blackboard import Blackboard
        from architecture_core.core.registry import SkillRegistry, SkillEntry
        from architecture_core.core.executive import Executive
        from architecture_core.cognition.norms.norm_engine import NormEngine
        from architecture_core.cognition.scripts.script_manager import ScriptManager
        from architecture_core.cognition.tom.tom_modulator import ToMModulator
        from architecture_core.skills.base import Skill
        from architecture_core.cognition.tom.intent_policy import IntentPolicy

        class _Skill(Skill):
            name = "navigate"
            def start(self, req): pass
            def tick(self, pb, update):
                self.last_update = update
                return "RUNNING"
            def stop(self, reason=""): pass

        class _Policy(IntentPolicy):
            def approach(self, pb, req, base):
                base.params["speed_scale"] = 1.0
                return base
            def avoid(self, pb, req, base): return base
            def yield_(self, pb, req, base): return base
            def wait(self, pb, req, base): return base

        bb = Blackboard()
        registry = SkillRegistry()
        skill = _Skill()
        registry.register(SkillEntry(skill=skill, policy=_Policy()))

        predictor = VelocityPredictor()
        rule = PredictiveCollisionRule(
            predictor=predictor, ttc_stop=0.5, ttc_caution=3.0,
        )
        shield = SafetyShield(predictor=predictor)

        executive = Executive(
            bb=bb, registry=registry,
            scripts=ScriptManager(),
            norms=NormEngine(rules=[rule]),
            tom=ToMModulator(),
            shield=shield,
            deliberation_hz=100.0,
        )

        bb.percept = _pb(t=0.0, closest_distance=5.0, agents=[
            {"id": "h1", "pose": (2, 0), "velocity": (-1, 0)},
        ])
        executive.tick(0.0)

        # The skill should have received a reduced speed
        assert hasattr(skill, "last_update")
        assert skill.last_update.params["speed_scale"] < 1.0

    def test_full_pipeline_emergency_stop(self):
        from architecture_core.core.blackboard import Blackboard
        from architecture_core.core.registry import SkillRegistry, SkillEntry
        from architecture_core.core.executive import Executive
        from architecture_core.cognition.norms.norm_engine import NormEngine
        from architecture_core.cognition.scripts.script_manager import ScriptManager
        from architecture_core.cognition.tom.tom_modulator import ToMModulator
        from architecture_core.skills.base import Skill
        from architecture_core.cognition.tom.intent_policy import IntentPolicy

        class _Skill(Skill):
            name = "navigate"
            def start(self, req): pass
            def tick(self, pb, update):
                self.last_update = update
                return "RUNNING"
            def stop(self, reason=""): pass

        class _Policy(IntentPolicy):
            def approach(self, pb, req, base): return base
            def avoid(self, pb, req, base): return base
            def yield_(self, pb, req, base): return base
            def wait(self, pb, req, base): return base

        bb = Blackboard()
        registry = SkillRegistry()
        skill = _Skill()
        registry.register(SkillEntry(skill=skill, policy=_Policy()))

        predictor = VelocityPredictor()
        shield = SafetyShield(
            predictor=predictor, emergency_ttc=0.5,
        )
        rule = PredictiveCollisionRule(predictor=predictor, veto_ttc=0.3)

        executive = Executive(
            bb=bb, registry=registry,
            scripts=ScriptManager(),
            norms=NormEngine(rules=[rule]),
            tom=ToMModulator(),
            shield=shield,
            deliberation_hz=100.0,
        )

        # Very fast agent → TTC < 0.3s → veto from rule
        bb.percept = _pb(t=0.0, closest_distance=5.0, agents=[
            {"id": "h1", "pose": (0.8, 0), "velocity": (-5, 0)},
        ])
        executive.tick(0.0)
        # Veto should have stopped action
        assert bb.active_req is None

    def test_agent_agent_causes_robot_slowdown(self):
        rule = PredictiveCollisionRule(agent_collision_proximity=5.0)
        pb = _pb(t=0.0, robot_pose=(0, 0, 0, 0), agents=[
            {"id": "h1", "pose": (2, 1), "velocity": (0, -1)},
            {"id": "h2", "pose": (2, -1), "velocity": (0, 1)},
        ])
        result = rule.evaluate(pb, SkillRequest(skill="nav"))
        assert "yield_agents" in result.soft
        # Speed should be reduced due to nearby agent-agent collision
        assert result.hard.get("speed_cap", 1.0) < 1.0

    def test_no_false_positives_stationary(self):
        pred = VelocityPredictor()
        rule = PredictiveCollisionRule(predictor=pred)
        # Agents near robot but all stationary
        pb = _pb(t=0.0, agents=[
            {"id": "h1", "pose": (1, 0)},
            {"id": "h2", "pose": (0, 1)},
        ])
        result = rule.evaluate(pb, SkillRequest(skill="nav"))
        assert result.veto is None
        assert "speed_cap" not in result.hard
