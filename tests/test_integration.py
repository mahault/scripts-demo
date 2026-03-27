"""Integration tests: full pipeline without Webots."""

from __future__ import annotations

import pytest

from architecture_core.core.blackboard import Blackboard
from architecture_core.core.registry import SkillEntry, SkillRegistry
from architecture_core.core.types import (
    PerceptBundle,
    SkillRequest,
    SkillUpdate,
)
from architecture_core.core.executive import Executive
from architecture_core.cognition.norms.norm_engine import NormEngine
from architecture_core.cognition.norms.rules import PersonalSpaceRule, SpeedLimitRule
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep
from architecture_core.cognition.tom.tom_modulator import ToMModulator
from architecture_core.cognition.empathy.empathic_modulator import (
    EmpathicModulator,
    EmpathyConfig,
)
from architecture_core.cognition.norms.affect_rules import (
    AffectModulatedSpeedRule,
    DistressVetoRule,
)
from architecture_core.safety.shield import SafetyShield
from architecture_core.perception.perception_pipeline import PerceptionPipeline
from architecture_core.perception.augmentations.proxemics import ProxemicsAugmentation
from architecture_core.perception.augmentations.affect import AffectAugmentation
from architecture_core.perception.sensors.base_sensors import SensorInterface
from architecture_core.skills.base import Skill
from architecture_core.cognition.tom.intent_policy import IntentPolicy


# -------------------------------------------------------------------
# Mock skill + policy for testing
# -------------------------------------------------------------------
class MockSkill(Skill):
    name = "navigate"

    def __init__(self):
        self.started = False
        self.stopped = False
        self.tick_count = 0
        self.last_update = None
        self._return_status = "RUNNING"

    def start(self, req):
        self.started = True
        self.stopped = False

    def tick(self, pb, update):
        self.tick_count += 1
        self.last_update = update
        return self._return_status

    def stop(self, reason=""):
        self.stopped = True
        self.started = False


class MockIntentPolicy(IntentPolicy):
    def approach(self, pb, req, base):
        base.params["speed_scale"] = 1.0
        return base

    def avoid(self, pb, req, base):
        base.params["speed_scale"] = 0.5
        return base

    def yield_(self, pb, req, base):
        base.params["speed_scale"] = 0.3
        return base

    def wait(self, pb, req, base):
        base.params["speed_scale"] = 0.0
        return base


class ConfigurableSensors(SensorInterface):
    """Sensor whose readings can be set from the test."""

    def __init__(self):
        self.data = {"robot_pose": (0, 0, 0, 0), "agents": [], "timestamp": 0}

    def read(self):
        return dict(self.data)


# -------------------------------------------------------------------
# Integration tests
# -------------------------------------------------------------------
class TestFullPipeline:
    def _build(self, sensors=None):
        sensors = sensors or ConfigurableSensors()
        bb = Blackboard()
        registry = SkillRegistry()
        skill = MockSkill()
        registry.register(SkillEntry(skill=skill, policy=MockIntentPolicy()))

        perception = PerceptionPipeline(
            sensors=sensors,
            augmentations=[ProxemicsAugmentation()],
        )
        seq = ScriptSequence(
            name="test",
            steps=[ScriptStep(SkillRequest(skill="navigate", goal={"x": 1}))],
        )
        executive = Executive(
            bb=bb,
            registry=registry,
            scripts=ScriptManager(sequence=seq),
            norms=NormEngine(rules=[
                PersonalSpaceRule(min_distance=0.5, veto_distance=0.3),
                SpeedLimitRule(max_speed=0.5),
            ]),
            tom=ToMModulator(),
            shield=SafetyShield(emergency_stop_distance=0.3),
            deliberation_hz=100.0,  # deliberate every tick for testing
        )
        return bb, executive, perception, skill, sensors

    def test_basic_tick(self):
        bb, executive, perception, skill, sensors = self._build()
        bb.percept = perception.tick(0.0)
        executive.tick(0.0)
        assert skill.started is True
        assert skill.tick_count == 1

    def test_speed_capped_by_shield(self):
        bb, executive, perception, skill, sensors = self._build()
        bb.percept = perception.tick(0.0)
        executive.tick(0.0)
        # Neutral intent -> speed_scale not set by policy, shield caps at 0.5
        assert skill.last_update.params.get("speed_scale", 1.0) <= 0.5

    def test_emergency_stop_on_close_agent(self):
        sensors = ConfigurableSensors()
        sensors.data["agents"] = [{"id": "human", "pose": (0.2, 0)}]
        bb, executive, perception, skill, _ = self._build(sensors)

        bb.percept = perception.tick(0.0)
        executive.tick(0.0)

        # Should trigger emergency stop via shield
        # Veto is also possible if personal space rule fires first
        # Either way the skill should not be running at full speed
        if skill.last_update is not None:
            assert skill.last_update.params.get("speed_scale", 1.0) == 0.0

    def test_skill_lifecycle_on_success(self):
        bb, executive, perception, skill, sensors = self._build()
        skill._return_status = "SUCCESS"

        bb.percept = perception.tick(0.0)
        executive.tick(0.0)

        # After SUCCESS, skill should be stopped
        assert skill.stopped is True
        assert bb.active_req is None

    def test_proxemics_reach_perception(self):
        sensors = ConfigurableSensors()
        sensors.data["agents"] = [
            {"id": "robot_b", "pose": (1.0, 0)},
        ]
        bb, executive, perception, skill, _ = self._build(sensors)

        pb = perception.tick(0.0)
        prox = pb.social.get("proxemics", {})
        assert prox["closest_distance"] == pytest.approx(1.0)
        assert prox["closest_zone"] == "personal"


class TestFullPipelineWithEmpathy:
    """Integration tests with the empathic modulator wired in."""

    def _build_empathic(self, sensors=None, alpha=0.3):
        sensors = sensors or ConfigurableSensors()
        bb = Blackboard()
        registry = SkillRegistry()
        skill = MockSkill()
        registry.register(SkillEntry(skill=skill, policy=MockIntentPolicy()))

        perception = PerceptionPipeline(
            sensors=sensors,
            augmentations=[
                ProxemicsAugmentation(),
                AffectAugmentation(),
            ],
        )
        seq = ScriptSequence(
            name="test",
            steps=[ScriptStep(SkillRequest(skill="navigate", goal={"x": 1}))],
        )
        empathy = EmpathicModulator(EmpathyConfig(contagion_alpha=alpha))
        affect_speed = AffectModulatedSpeedRule(base_max_speed=0.5)

        executive = Executive(
            bb=bb,
            registry=registry,
            scripts=ScriptManager(sequence=seq),
            norms=NormEngine(rules=[
                PersonalSpaceRule(min_distance=0.5, veto_distance=0.3),
                SpeedLimitRule(max_speed=0.5),
                affect_speed,
            ]),
            tom=ToMModulator(),
            shield=SafetyShield(emergency_stop_distance=0.3),
            deliberation_hz=100.0,
            empathy=empathy,
        )
        return bb, executive, perception, skill, sensors, empathy

    def test_empathy_tick_without_agents(self):
        bb, executive, perception, skill, sensors, empathy = self._build_empathic()
        bb.percept = perception.tick(0.0)
        executive.tick(0.0)
        assert skill.started is True
        # Valence neutral (ΔG = 0 on first tick), arousal = -1 (zero entropy)
        assert abs(bb.self_affect.valence) < 0.1
        assert bb.self_affect.arousal <= 0.0  # no policy entropy → low arousal

    def test_negative_affect_contagion_shifts_fe(self):
        sensors = ConfigurableSensors()
        sensors.data["agents"] = [{
            "id": "h1",
            "pose": (1.0, 0),
            "affect_cues": {
                "valence_idx": 0,
                "arousal_idx": 4,
                "source": "face",
                "confidence": 0.9,
            },
        }]
        bb, executive, perception, skill, _, empathy = self._build_empathic(
            sensors=sensors, alpha=0.5,
        )

        # Run several ticks to let contagion couple through G
        for i in range(10):
            bb.percept = perception.tick(float(i) * 0.1)
            executive.tick(float(i) * 0.1)

        # Contagion from distressed other should have pushed G upward.
        # valence = tanh(-ΔG/τ) tracks *changes* — once stable, valence ≈ 0
        # (habituation), but raw_free_energy reflects the contagion coupling.
        assert bb.self_affect.raw_free_energy > 0.0

    def test_empathy_does_not_break_existing_behavior(self):
        bb, executive, perception, skill, sensors, _ = self._build_empathic()
        bb.percept = perception.tick(0.0)
        executive.tick(0.0)
        assert skill.tick_count == 1
        assert skill.started is True
