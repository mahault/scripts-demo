"""architecture_core/core/executive.py

Owns the main control loop.

Two rates:
- Deliberation (scripts + norms)  -- configurable, default 2 Hz
- Execution   (skill tick + ToM)  -- every call to ``tick()``

Merges ToM intent via IntentPolicy, applies SafetyShield, and
manages the full skill lifecycle (start / tick / stop / preempt).

Optional empathy integration:
When an ``EmpathicModulator`` is provided, the loop adds:
  predict -> check_violation -> observe -> modulate
All empathy paths are gated on ``self._empathy is not None``.

Optional repertoire integration:
When a ``ScriptRepertoire`` is provided (via LearningScriptManager),
trajectory recording hooks feed execution data back for learning.
All repertoire paths are gated on ``self._repertoire is not None``.
"""

from __future__ import annotations

from typing import Optional

from architecture_core.core.blackboard import Blackboard
from architecture_core.core.registry import SkillRegistry
from architecture_core.core.types import SkillRequest, SkillUpdate
from architecture_core.cognition.norms.norm_engine import NormEngine
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.tom.tom_modulator import ToMModulator
from architecture_core.cognition.norms.adaptive_norms import extract_norm_features
from architecture_core.safety.shield import SafetyShield


class Executive:
    def __init__(
        self,
        bb: Blackboard,
        registry: SkillRegistry,
        scripts: ScriptManager,
        norms: NormEngine,
        tom: ToMModulator,
        shield: SafetyShield,
        deliberation_hz: float = 2.0,
        empathy: Optional[object] = None,
    ) -> None:
        self.bb = bb
        self.registry = registry
        self.scripts = scripts
        self.norms = norms
        self.tom = tom
        self.shield = shield
        self._delib_period = 1.0 / deliberation_hz
        self._last_delib_t = -1e9

        self._active_skill = None
        self._active_entry = None
        self._active_req_id: int | None = None

        # Cached ToM update (reused between deliberation ticks)
        self._cached_tom_update = None

        # Optional empathy integration
        self._empathy = empathy

        # Optional repertoire integration (via LearningScriptManager)
        self._repertoire = None
        try:
            from architecture_core.cognition.scripts.learning_script_manager import LearningScriptManager
            if isinstance(scripts, LearningScriptManager):
                self._repertoire = scripts.repertoire
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def tick(self, t: float) -> None:
        pb = self.bb.percept
        if pb is None:
            return

        # ---- empathy: predict (affect from free energy dynamics) ----
        if self._empathy is not None:
            # Use actual VFE if available from unified controller
            if hasattr(self.scripts, 'vfe'):
                base_fe = self.scripts.vfe
            else:
                base_fe = self.bb.last_social_efe
            violation_fe = self._empathy.consume_pending_fe()
            contagion_fe = self._empathy.contagion_fe_shift()
            # Use policy entropy from unified controller if available
            policy_ent = (
                self.scripts.policy_entropy
                if hasattr(self.scripts, 'policy_entropy')
                else self.bb.last_policy_entropy
            )
            num_pol = 7 if hasattr(self.scripts, 'policy_entropy') else self.bb.last_num_policies
            self._empathy.predict(
                base_fe + violation_fe + contagion_fe,
                policy_ent,
                num_policies=num_pol,
            )

        # ---- deliberation: scripts + norms ----
        if (t - self._last_delib_t) >= self._delib_period:

            # Check for script violations and update situation context
            if self._empathy is not None:
                violation = self.scripts.check_violation(pb, timestamp=t)
                if violation is not None:
                    self.bb.latest_violation = violation
                else:
                    self.bb.latest_violation = None
            else:
                self.bb.latest_violation = None

            # Feed recognized situation type to ToM for intent fusion
            _recognizer = getattr(self.scripts, '_recognizer', None)
            if _recognizer is None and hasattr(self.scripts, 'base'):
                _recognizer = getattr(self.scripts.base, '_recognizer', None)
            if _recognizer is not None:
                belief = _recognizer.recognize(pb)
                self.tom.set_situation_type(belief.most_likely)

                # Feed situation to context-dependent norm rules
                for rule in self.norms.rules:
                    if hasattr(rule, "set_situation"):
                        rule.set_situation(belief.most_likely)

                # Repertoire: feed situation belief for trajectory inference
                if self._repertoire is not None:
                    pattern_name = self._repertoire.on_situation_recognized(
                        belief.distribution, t,
                    )
                    self.bb.active_pattern = pattern_name
                    self.bb.trajectory_free_energy = self._repertoire.trajectory_free_energy

                    # Push discovered norm features to adaptive rules
                    norm_feats = self._repertoire.active_norm_features
                    if norm_feats:
                        self.bb.active_norm_features = norm_feats
                        for rule in self.norms.rules:
                            if hasattr(rule, "set_norm_features"):
                                rule.set_norm_features(norm_feats)
            else:
                self.tom.set_situation_type(None)

            # Inject self-arousal for precision coupling in task-level EFE
            if self._empathy is not None and hasattr(self.scripts, 'set_self_arousal'):
                self.scripts.set_self_arousal(self._empathy.state.arousal)

            candidate = self.scripts.select(pb, self.bb.active_req)
            norms_out = self.norms.evaluate(pb, candidate)

            # Expose collision prediction on blackboard
            for rule in self.norms.rules:
                if hasattr(rule, 'last_prediction') and rule.last_prediction is not None:
                    self.bb.collision_prediction = rule.last_prediction
                    break

            if norms_out.veto is not None:
                self._stop_active_skill(f"Norm veto: {norms_out.veto}")
                self.bb.active_req = None
                self.bb.norms = norms_out
            else:
                if self._request_changed(candidate):
                    self._stop_active_skill("Request changed")
                    self._start_skill(candidate)
                self.bb.active_req = candidate
                self.bb.norms = norms_out

            self._last_delib_t = t

        # ---- empathy: observe (contagion + violation coupling) ----
        if self._empathy is not None:
            self._empathy.observe(pb, self.bb.latest_violation)
            self.bb.self_affect = self._empathy.state

            # Push affect to affect-aware norm rules
            for rule in self.norms.rules:
                if hasattr(rule, "set_affect"):
                    rule.set_affect(self._empathy.state)

        # ---- norm features: extract + store ----
        snapshot = extract_norm_features(pb)
        self.bb.norm_snapshot = snapshot
        if self._repertoire is not None:
            self._repertoire.set_norm_snapshot(snapshot)

        # ---- execution: active skill tick ----
        if self.bb.active_req is None:
            return

        # Ensure skill is started
        if self._active_skill is None:
            self._start_skill(self.bb.active_req)

        entry = self._active_entry

        # Inject self-arousal for precision coupling
        if self._empathy is not None:
            self.tom.set_self_arousal(self._empathy.state.arousal)

        # ToM intent — computed at deliberation rate, cached between ticks.
        # This prevents intent flickering that breaks yield target persistence.
        if self._cached_tom_update is None or (t - self._last_delib_t) < 0.001:
            self._cached_tom_update = self.tom.modulate(pb, self.bb.active_req)
            # Extract EFE results for next tick's affect computation
            if self._empathy is not None:
                self.bb.last_social_efe = self._cached_tom_update.params.get(
                    "g_social", 0.0,
                )
                self.bb.last_policy_entropy = self._cached_tom_update.params.get(
                    "policy_entropy", 0.0,
                )
                self.bb.last_num_policies = self._cached_tom_update.params.get(
                    "num_policies", 5,
                )
        # Fresh copy each tick — yield_(), shield, and empathy mutate params/constraints
        cached = self._cached_tom_update
        update = SkillUpdate(
            intent=cached.intent,
            params=dict(cached.params),
            constraints=dict(cached.constraints),
            recommend_interrupt=cached.recommend_interrupt,
            debug=cached.debug,
        )

        # When UTC controls navigation, it decides WHERE (via SkillRequest.goal).
        # ToM decides HOW (intent → speed/caution via IntentPolicy).
        # Clear ToM's target_x/target_y so nav skill uses UTC's goal.
        if hasattr(self.scripts, 'vfe'):
            update.params.pop("target_x", None)
            update.params.pop("target_y", None)

        # Protect committed manipulation from avoid/retract intents.
        # HAIF pattern: once manipulation starts, execute open-loop.
        # Safety shield emergency_stop is NOT suppressed (hard constraint).
        if (hasattr(self.scripts, '_committed_req')
                and self.scripts._committed_req is not None):
            if update.intent == "avoid":
                update.intent = "neutral"
                update.params.pop("retract", None)

        # Empathy modulation (affect -> EFE adjustment)
        if self._empathy is not None:
            update = self._empathy.modulate(update)

        # IntentPolicy dispatch
        if update.intent == "approach":
            compiled = entry.policy.approach(pb, self.bb.active_req, update)
        elif update.intent == "avoid":
            compiled = entry.policy.avoid(pb, self.bb.active_req, update)
        elif update.intent == "yield":
            compiled = entry.policy.yield_(pb, self.bb.active_req, update)
        elif update.intent == "wait":
            compiled = entry.policy.wait(pb, self.bb.active_req, update)
        else:
            compiled = update  # neutral

        # Safety shield (hard constraints)
        compiled = self.shield.apply(
            pb, self.bb.active_req, compiled, self.bb.norms
        )

        # Diagnostics: expose final compiled update on blackboard
        self.bb._compiled_intent = compiled.intent
        self.bb._compiled_speed_scale = compiled.params.get("speed_scale", 1.0)
        self.bb._rollout_policy = compiled.params.get("rollout_policy")
        self.bb._compiled_params = compiled.params

        # Tick skill
        status = entry.skill.tick(pb, compiled)

        # Handle terminal statuses
        if status in ("SUCCESS", "FAILURE", "TIMEOUT", "PREEMPTED"):
            self._stop_active_skill(f"Status: {status}")
            self.scripts.notify_status(status)
            self.bb.active_req = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _start_skill(self, req: SkillRequest) -> None:
        entry = self.registry.get(req.skill)
        entry.skill.start(req)
        self._active_skill = entry.skill
        self._active_entry = entry
        self._active_req_id = id(req)

    def _stop_active_skill(self, reason: str) -> None:
        if self._active_skill is not None:
            self._active_skill.stop(reason)
            self._active_skill = None
            self._active_entry = None
            self._active_req_id = None

    def _request_changed(self, candidate: SkillRequest) -> bool:
        if self.bb.active_req is None:
            return True
        if candidate.skill != self.bb.active_req.skill:
            return True
        if candidate.goal != self.bb.active_req.goal:
            return True
        return False
