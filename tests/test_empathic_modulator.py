"""Tests for the EmpathicModulator (FE-grounded affect dynamics)."""

from __future__ import annotations

import math
import pytest

from architecture_core.core.types import (
    AffectState,
    PerceptBundle,
    ScriptViolation,
    SkillUpdate,
)
from architecture_core.cognition.empathy.empathic_modulator import (
    EmpathicModulator,
    EmpathyConfig,
)


# =====================================================================
# Helpers
# =====================================================================
def _make_pb(affect_readings=None, engagement_readings=None):
    social = {}
    if affect_readings is not None:
        social["affect"] = {"readings": affect_readings}
    if engagement_readings is not None:
        social["engagement"] = {"readings": engagement_readings}
    return PerceptBundle(
        t=0.0,
        world={"robot_pose": (0, 0, 0, 0), "agents": []},
        social=social,
    )


def _default_update(**kwargs):
    return SkillUpdate(
        intent=kwargs.get("intent", "approach"),
        params=kwargs.get("params", {"speed_scale": 1.0}),
    )


# =====================================================================
# Predict — valence from -ΔG, arousal from H[q(π)]
# =====================================================================
class TestPredict:
    def test_decreasing_fe_positive_valence(self):
        """G decreasing → ΔG < 0 → valence = tanh(-ΔG/τ) > 0."""
        mod = EmpathicModulator()
        mod.predict(current_fe=2.0)  # establish baseline
        mod.predict(current_fe=1.0)  # G decreased
        assert mod.state.valence > 0.0

    def test_increasing_fe_negative_valence(self):
        """G increasing → ΔG > 0 → valence = tanh(-ΔG/τ) < 0."""
        mod = EmpathicModulator()
        mod.predict(current_fe=1.0)  # establish baseline
        mod.predict(current_fe=2.0)  # G increased
        assert mod.state.valence < 0.0

    def test_stable_fe_neutral_valence(self):
        """ΔG = 0 → valence = tanh(0) = 0."""
        mod = EmpathicModulator()
        mod.predict(current_fe=1.0)
        mod.predict(current_fe=1.0)
        assert mod.state.valence == pytest.approx(0.0)

    def test_high_entropy_high_arousal(self):
        """H = log|Π| → normalized = 1.0 → arousal = 1.0."""
        mod = EmpathicModulator()
        H_max = math.log(5)
        mod.predict(policy_entropy=H_max, num_policies=5)
        assert mod.state.arousal == pytest.approx(1.0)

    def test_zero_entropy_low_arousal(self):
        """H = 0 → normalized = 0 → arousal = -1.0."""
        mod = EmpathicModulator()
        mod.predict(policy_entropy=0.0, num_policies=5)
        assert mod.state.arousal == pytest.approx(-1.0)

    def test_mid_entropy_neutral_arousal(self):
        """H = log|Π|/2 → normalized = 0.5 → arousal = 0.0."""
        mod = EmpathicModulator()
        H_half = math.log(5) / 2
        mod.predict(policy_entropy=H_half, num_policies=5)
        assert mod.state.arousal == pytest.approx(0.0)

    def test_raw_values_stored(self):
        """Raw G, ΔG, H should be stored on state."""
        mod = EmpathicModulator()
        mod.predict(current_fe=1.0)
        mod.predict(current_fe=3.0, policy_entropy=0.5)
        assert mod.state.raw_free_energy == pytest.approx(3.0)
        assert mod.state.delta_free_energy == pytest.approx(2.0)
        assert mod.state.policy_entropy == pytest.approx(0.5)

    def test_valence_bounded(self):
        """Valence should be in (-1, 1) via tanh."""
        mod = EmpathicModulator()
        mod.predict(current_fe=0.0)
        mod.predict(current_fe=100.0)  # huge ΔG
        assert -1.0 <= mod.state.valence <= 1.0

    def test_valence_scale_controls_sensitivity(self):
        """Higher τ → smaller valence magnitude for same ΔG."""
        mod_sensitive = EmpathicModulator(EmpathyConfig(valence_scale=0.5))
        mod_gentle = EmpathicModulator(EmpathyConfig(valence_scale=5.0))
        # Both: G goes from 0 to 1 (ΔG = 1)
        mod_sensitive.predict(current_fe=0.0)
        mod_sensitive.predict(current_fe=1.0)
        mod_gentle.predict(current_fe=0.0)
        mod_gentle.predict(current_fe=1.0)
        assert abs(mod_sensitive.state.valence) > abs(mod_gentle.state.valence)

    def test_num_policies_affects_normalization(self):
        """Different |Π| should change arousal for same entropy."""
        mod_small = EmpathicModulator()
        mod_large = EmpathicModulator()
        H = 1.0  # fixed entropy
        mod_small.predict(policy_entropy=H, num_policies=3)
        mod_large.predict(policy_entropy=H, num_policies=10)
        # Same H, larger policy space → lower normalized entropy → lower arousal
        assert mod_small.state.arousal > mod_large.state.arousal


# =====================================================================
# Observe — coupling signals (no direct affect mutation)
# =====================================================================
class TestObserve:
    def test_no_readings_no_coupling(self):
        """No affect readings → contagion G shift is zero."""
        mod = EmpathicModulator()
        pb = _make_pb(affect_readings=[])
        mod.observe(pb)
        assert mod.contagion_fe_shift() == pytest.approx(0.0)

    def test_distressed_other_positive_fe_shift(self):
        """Others with negative valence → positive G shift (more risk)."""
        mod = EmpathicModulator(EmpathyConfig(contagion_alpha=0.5))
        pb = _make_pb(affect_readings=[{"valence": -0.8, "arousal": 0.5}])
        mod.observe(pb)
        assert mod.contagion_fe_shift() > 0.0

    def test_happy_other_negative_fe_shift(self):
        """Others with positive valence → negative G shift (less risk)."""
        mod = EmpathicModulator(EmpathyConfig(contagion_alpha=0.5))
        pb = _make_pb(affect_readings=[{"valence": 0.8, "arousal": 0.2}])
        mod.observe(pb)
        assert mod.contagion_fe_shift() < 0.0

    def test_contagion_fe_shift_capped(self):
        """G shift should not exceed max_contagion_step."""
        cfg = EmpathyConfig(contagion_alpha=1.0, max_contagion_step=0.2)
        mod = EmpathicModulator(cfg)
        pb = _make_pb(affect_readings=[{"valence": -1.0, "arousal": 1.0}])
        mod.observe(pb)
        assert mod.contagion_fe_shift() <= 0.2

    def test_selfish_robot_no_coupling(self):
        """Zero contagion_alpha → no G shift."""
        mod = EmpathicModulator(EmpathyConfig(contagion_alpha=0.0))
        pb = _make_pb(affect_readings=[{"valence": -0.9, "arousal": 0.9}])
        mod.observe(pb)
        assert mod.contagion_fe_shift() == pytest.approx(0.0)

    def test_observe_does_not_change_affect(self):
        """Observe should NOT directly modify valence/arousal."""
        mod = EmpathicModulator(EmpathyConfig(contagion_alpha=0.5))
        mod._state = AffectState(arousal=0.3, valence=0.2)
        pb = _make_pb(affect_readings=[{"valence": -0.8, "arousal": 0.9}])
        mod.observe(pb)
        # Affect unchanged — coupling only flows through G on next predict()
        assert mod.state.arousal == pytest.approx(0.3)
        assert mod.state.valence == pytest.approx(0.2)

    def test_multiple_agents_averaged_for_coupling(self):
        """Mean valence drives contagion G shift."""
        cfg = EmpathyConfig(contagion_alpha=0.5)
        mod = EmpathicModulator(cfg)
        pb = _make_pb(affect_readings=[
            {"valence": 0.8, "arousal": 0.2},
            {"valence": -0.8, "arousal": 0.6},
        ])
        mod.observe(pb)
        # Mean valence = 0.0 → no G shift
        assert abs(mod.contagion_fe_shift()) < 0.01


# =====================================================================
# Observe — violation G accumulation
# =====================================================================
class TestObserveViolation:
    def test_violation_accumulates_fe(self):
        """Violation KL should accumulate as pending G."""
        mod = EmpathicModulator()
        pb = _make_pb()
        mod.observe(pb, ScriptViolation(kl_divergence=3.0))
        assert mod.consume_pending_fe() == pytest.approx(3.0)

    def test_multiple_violations_accumulate(self):
        """Multiple distinct violations should sum their KL."""
        mod = EmpathicModulator()
        pb = _make_pb()
        v1 = ScriptViolation(kl_divergence=1.0)
        v2 = ScriptViolation(kl_divergence=2.0)
        mod.observe(pb, v1)
        mod.observe(pb, v2)
        assert mod.consume_pending_fe() == pytest.approx(3.0)

    def test_same_violation_not_double_counted(self):
        """Same violation object should only be counted once."""
        mod = EmpathicModulator()
        pb = _make_pb()
        v = ScriptViolation(kl_divergence=3.0)
        mod.observe(pb, v)
        mod.observe(pb, v)  # same object
        assert mod.consume_pending_fe() == pytest.approx(3.0)

    def test_consume_resets_accumulator(self):
        """consume_pending_fe should return and reset."""
        mod = EmpathicModulator()
        pb = _make_pb()
        mod.observe(pb, ScriptViolation(kl_divergence=2.0))
        first = mod.consume_pending_fe()
        second = mod.consume_pending_fe()
        assert first == pytest.approx(2.0)
        assert second == pytest.approx(0.0)

    def test_no_violation_no_fe(self):
        """No violation → no pending G."""
        mod = EmpathicModulator()
        pb = _make_pb()
        mod.observe(pb)
        assert mod.consume_pending_fe() == pytest.approx(0.0)


# =====================================================================
# Integration: violation / contagion → predict → affect
# =====================================================================
class TestIntegration:
    def test_violation_produces_negative_valence(self):
        """Violation KL → G increase → ΔG > 0 → negative valence."""
        mod = EmpathicModulator()
        pb = _make_pb()

        # Establish baseline
        mod.predict(current_fe=1.0)

        # Observe violation
        mod.observe(pb, ScriptViolation(kl_divergence=2.0))

        # Executive composes: effective_G = base + violation + contagion
        violation_fe = mod.consume_pending_fe()
        contagion_fe = mod.contagion_fe_shift()
        mod.predict(current_fe=1.0 + violation_fe + contagion_fe)

        assert mod.state.valence < 0.0, "Violation should produce negative valence"

    def test_contagion_shifts_valence_via_fe(self):
        """Distressed other → G increase → negative valence."""
        mod = EmpathicModulator(EmpathyConfig(contagion_alpha=0.5))
        pb = _make_pb(affect_readings=[{"valence": -0.8, "arousal": 0.5}])

        # Establish baseline
        mod.predict(current_fe=1.0)

        # Observe distressed other
        mod.observe(pb)

        # Executive composes effective G
        contagion_fe = mod.contagion_fe_shift()
        assert contagion_fe > 0.0, "Distress should increase G"
        mod.predict(current_fe=1.0 + contagion_fe)

        assert mod.state.valence < 0.0, "Contagion should produce negative valence"

    def test_happy_other_improves_valence(self):
        """Happy other → G decrease → positive valence."""
        mod = EmpathicModulator(EmpathyConfig(contagion_alpha=0.5))
        pb = _make_pb(affect_readings=[{"valence": 0.8, "arousal": 0.2}])

        # Establish baseline
        mod.predict(current_fe=1.0)

        # Observe happy other
        mod.observe(pb)

        contagion_fe = mod.contagion_fe_shift()
        assert contagion_fe < 0.0, "Wellbeing should decrease G"
        mod.predict(current_fe=1.0 + contagion_fe)

        assert mod.state.valence > 0.0, "Happy other should produce positive valence"


# =====================================================================
# Modulate — EFE adjustment (unchanged, to be simplified in Step 5)
# =====================================================================
class TestModulate:
    def test_neutral_affect_no_modulation(self):
        mod = EmpathicModulator()
        update = _default_update()
        result = mod.modulate(update)
        assert result.params["speed_scale"] == 1.0
        assert result.recommend_interrupt is False

    def test_negative_valence_no_speed_change(self):
        """Speed modulation removed — handled by EFE affect risk term."""
        mod = EmpathicModulator()
        mod._state = AffectState(arousal=0.0, valence=-0.6)
        update = _default_update()
        result = mod.modulate(update)
        assert result.params["speed_scale"] == 1.0  # unchanged

    def test_high_arousal_no_epistemic_boost(self):
        """Epistemic boost removed — handled by precision coupling."""
        mod = EmpathicModulator()
        mod._state = AffectState(arousal=0.8, valence=0.0)
        update = _default_update()
        result = mod.modulate(update)
        assert "epistemic_boost" not in result.params

    def test_distress_triggers_interrupt(self):
        mod = EmpathicModulator()
        mod._state = AffectState(arousal=0.9, valence=-0.8)
        update = _default_update()
        result = mod.modulate(update)
        assert result.recommend_interrupt is True

    def test_distress_requires_both_conditions(self):
        mod = EmpathicModulator()
        # Only low valence, low arousal → no interrupt
        mod._state = AffectState(arousal=0.1, valence=-0.9)
        result = mod.modulate(_default_update())
        assert result.recommend_interrupt is False

        # Only high arousal, positive valence → no interrupt
        mod._state = AffectState(arousal=0.9, valence=0.1)
        result = mod.modulate(_default_update())
        assert result.recommend_interrupt is False

    def test_self_affect_in_params(self):
        mod = EmpathicModulator()
        mod._state = AffectState(arousal=0.3, valence=-0.2)
        result = mod.modulate(_default_update())
        assert result.params["self_affect_arousal"] == pytest.approx(0.3)
        assert result.params["self_affect_valence"] == pytest.approx(-0.2)

    def test_does_not_mutate_input(self):
        mod = EmpathicModulator()
        mod._state = AffectState(arousal=0.8, valence=-0.6)
        update = _default_update()
        original_params = dict(update.params)
        mod.modulate(update)
        assert update.params == original_params
