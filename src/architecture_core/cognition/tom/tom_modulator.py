"""architecture_core/cognition/tom/tom_modulator.py

Apply Theory of Mind modulation during active skill execution.

Three levels of sophistication (highest available is used):
1. GatedToM — full active inference with particle filter + Social EFE
2. IntentInference — log-linear product-of-experts (fallback)
3. Raw kinematic intent from ToMPlannerAdapter (baseline)

Each level gracefully degrades to the next on error.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

from architecture_core.core.types import AffectState, PerceptBundle, SkillRequest, SkillUpdate


class ToMModulator:
    def __init__(
        self,
        adapter: Optional[object] = None,
        intent_inference: Optional[object] = None,
        gated_tom: Optional[object] = None,
    ) -> None:
        self._adapter = adapter
        self._intent_inference = intent_inference
        self._gated_tom = gated_tom
        self._situation_type: Optional[str] = None
        self._robot_last_intent: str = "neutral"
        self._last_agent_dist: Dict[Any, float] = {}  # for velocity estimation
        self._self_arousal: Optional[float] = None  # for precision coupling
        self._adapter_log_count: int = 0

    def set_situation_type(self, situation_type: Optional[str]) -> None:
        """Update the current situation context from WeakScriptRecognizer."""
        self._situation_type = situation_type

    def set_self_arousal(self, arousal: Optional[float]) -> None:
        """Set robot's own arousal for precision coupling in EFE softmax."""
        self._self_arousal = arousal

    def modulate(
        self, pb: PerceptBundle, req: SkillRequest
    ) -> SkillUpdate:
        # Step 1: Get kinematic intent from ToM planner
        if self._adapter is None:
            kinematic_update = SkillUpdate(intent="neutral", debug="no ToM adapter")
        else:
            try:
                t0 = time.time()
                kinematic_update = self._adapter.infer_intent(pb, req)  # type: ignore[union-attr]
                dt_ms = 1000 * (time.time() - t0)
                tx = kinematic_update.params.get("target_x")
                ty = kinematic_update.params.get("target_y")
                if self._adapter_log_count % 30 == 0:
                    print(f"  [ToM] adapter ok {dt_ms:.0f}ms "
                          f"intent={kinematic_update.intent} "
                          f"target=({tx},{ty}) "
                          f"keys={list(kinematic_update.params.keys())}")
                self._adapter_log_count += 1
            except Exception as exc:
                import traceback
                print(f"  [ToM] adapter ERROR: {exc}")
                print(traceback.format_exc())
                kinematic_update = SkillUpdate(
                    intent="neutral", debug=f"ToM error: {exc}"
                )

        # Step 2: Active inference via GatedToM (highest priority)
        if self._gated_tom is not None:
            try:
                result = self._modulate_via_gated_tom(pb, kinematic_update)
                if result is not None:
                    self._robot_last_intent = result.intent
                    return result
            except Exception as exc:
                kinematic_update.debug += f" | gated_tom error: {exc}"

        # Step 3: Log-linear IntentInference (fallback)
        if self._intent_inference is not None:
            try:
                result = self._modulate_via_inference(pb, kinematic_update)
                if result is not None:
                    self._robot_last_intent = result.intent
                    return result
            except Exception as exc:
                kinematic_update.debug += f" | intent_inference error: {exc}"

        # Step 4: Raw kinematic (baseline)
        self._robot_last_intent = kinematic_update.intent
        return kinematic_update

    # ------------------------------------------------------------------
    # Active inference path
    # ------------------------------------------------------------------
    def _modulate_via_gated_tom(
        self, pb: PerceptBundle, kinematic_update: SkillUpdate,
    ) -> Optional[SkillUpdate]:
        """Use GatedToM for full active inference intent prediction."""
        from architecture_core.cognition.tom.intent_particle_filter import ObservationContext
        from architecture_core.cognition.tom.gated_tom import GatedToM

        gated: GatedToM = self._gated_tom  # type: ignore[assignment]
        agents = pb.world.get("agents", [])
        if not agents:
            return None

        # Build observation context for primary agent
        agent = agents[0]
        eid = agent.get("id", "other")
        pose = agent.get("pose", (0, 0))
        robot_pose = pb.world.get("robot_pose", (0, 0, 0, 0))
        rx, ry = robot_pose[0], robot_pose[1]
        dist = math.sqrt((pose[0] - rx) ** 2 + (pose[1] - ry) ** 2)

        # Gather engagement signals
        eng_data = pb.social.get("engagement", {})
        eng_readings = eng_data.get("readings", [])
        gaze, body_orient, trend = 0.5, 0.5, 0.0
        for r in eng_readings:
            if r.get("entity_id") == eid:
                gaze = r.get("gaze_on_robot", 0.5)
                body_orient = r.get("body_orientation", 0.5)
                trend = r.get("proximity_trend", 0.0)
                break

        # Gather affect signals
        aff_data = pb.social.get("affect", {})
        aff_readings = aff_data.get("readings", [])
        valence, arousal = 0.0, 0.0
        for r in aff_readings:
            if r.get("entity_id") == eid:
                valence = r.get("valence", 0.0)
                arousal = r.get("arousal", 0.0)
                break

        # Estimate velocity from distance delta
        prev_dist = self._last_agent_dist.get(eid, dist)
        velocity = max(0.0, prev_dist - dist)  # positive = closing
        self._last_agent_dist[eid] = dist

        # Compute initial obstruction from perpendicular distance
        # to the agent's intended path (how much robot blocks their way)
        agent_goal = agent.get("goal", pose)
        initial_obstruction = self._compute_obstruction(
            rx, ry, pose, agent_goal, dist,
        )

        obs = ObservationContext(
            kinematic_intent=kinematic_update.intent,
            distance=dist,
            velocity=velocity,
            approaching=velocity > 0.02,
            gaze_on_robot=gaze,
            body_orientation=body_orient,
            valence=valence,
            arousal=arousal,
            robot_last_intent=self._robot_last_intent,
        )

        # Update particle filter with observation
        gated.update(eid, obs)

        # Robot's own arousal for precision coupling
        self_arousal = self._self_arousal

        # Arousal → jitter: modulate particle filter adaptation rate
        if self_arousal is not None:
            pf = gated.get_or_create_filter(eid)
            pf.set_arousal_modulation(self_arousal)

        # Active inference action selection via rollout EFE
        other_affect = AffectState(arousal=arousal, valence=valence)

        rollout_output = gated.select_action_rollout(
            eid, obs, other_affect=other_affect,
            initial_obstruction=initial_obstruction,
            self_arousal=self_arousal,
        )

        # Use single-step output for back-compat EFE breakdown
        efe_output = rollout_output.single_step_output

        # Build SkillUpdate from rollout result
        params = dict(kinematic_update.params)
        params["intent_confidence"] = rollout_output.distribution.get(
            rollout_output.selected, 0.0,
        )
        params["intent_distribution"] = rollout_output.distribution
        params["empathy_factor"] = efe_output.empathy_factor
        params["agent_reliability"] = gated.agent_reliability(eid)
        params["intent_sources"] = ["particle_filter", "social_efe", "gated_tom"]
        params["rollout_policy"] = rollout_output.full_policy
        params["initial_obstruction"] = initial_obstruction

        # EFE breakdown for the selected action (from single-step)
        for a in efe_output.actions:
            if a.intent == rollout_output.selected:
                params["g_self"] = a.g_self
                params["g_other"] = a.g_other
                params["g_epistemic"] = a.g_epistemic
                params["g_social"] = a.g_social
                break

        # Policy posterior entropy H[q(π)] for affect / precision coupling
        dist_values = [p for p in rollout_output.distribution.values() if p > 0]
        if dist_values:
            params["policy_entropy"] = -sum(
                p * math.log(p) for p in dist_values
            )
        else:
            params["policy_entropy"] = 0.0
        params["num_policies"] = len(rollout_output.distribution)

        # Prune stale filters
        active_ids = {a.get("id", "other") for a in agents}
        gated.prune_stale(active_ids)

        # Fallback: if intent is yield/wait but adapter didn't provide a
        # waypoint, synthesize a sidestep perpendicular to the other agent's
        # bearing.  Without this the nav skill has no intermediate target and
        # the robot freezes in place.
        intent = rollout_output.selected
        if intent in ("yield", "wait") and "target_x" not in params:
            dx = pose[0] - rx
            dy = pose[1] - ry
            d = math.sqrt(dx * dx + dy * dy) or 1.0
            # Perpendicular direction (rotate bearing 90°)
            px, py = -dy / d, dx / d
            step = 0.6  # sidestep distance
            params["target_x"] = rx + px * step
            params["target_y"] = ry + py * step

        return SkillUpdate(
            intent=intent,
            params=params,
            constraints=kinematic_update.constraints,
            recommend_interrupt=kinematic_update.recommend_interrupt,
            debug=f"active_inference: {intent} "
                  f"(p={rollout_output.distribution.get(intent, 0.0):.2f}, "
                  f"rel={gated.agent_reliability(eid):.2f}, "
                  f"policy={rollout_output.full_policy[:3]}...)",
        )

    @staticmethod
    def _compute_obstruction(
        rx: float, ry: float,
        agent_pose: tuple, agent_goal: tuple,
        dist: float,
    ) -> float:
        """Compute how much the robot obstructs the agent's path [0, 1].

        Uses perpendicular distance from robot position to the line
        defined by agent_pose → agent_goal.  Returns 1.0 when directly
        on path, 0.0 when >= CLEARANCE away.
        """
        CLEARANCE = 0.8
        dx = agent_goal[0] - agent_pose[0]
        dy = agent_goal[1] - agent_pose[1]
        path_len = math.sqrt(dx * dx + dy * dy)

        if path_len < 0.01:
            # Agent stationary or goal unknown — use distance
            return 1.0 if dist < 1.5 else 0.0

        ux, uy = dx / path_len, dy / path_len
        # Vector from agent to robot
        crx = rx - agent_pose[0]
        cry = ry - agent_pose[1]
        # Perpendicular distance from robot to agent's path
        perp_dist = abs(ux * cry - uy * crx)

        return max(0.0, 1.0 - perp_dist / CLEARANCE)

    # ------------------------------------------------------------------
    # Log-linear path (fallback)
    # ------------------------------------------------------------------
    def _modulate_via_inference(
        self, pb: PerceptBundle, kinematic_update: SkillUpdate,
    ) -> Optional[SkillUpdate]:
        """Use IntentInference for log-linear product-of-experts."""
        beliefs = self._intent_inference.infer(  # type: ignore[union-attr]
            pb,
            kinematic_intent=kinematic_update.intent,
            situation_type=self._situation_type,
        )
        if not beliefs:
            return None

        best = beliefs[0]
        params = dict(kinematic_update.params)
        params["intent_confidence"] = best.confidence
        params["intent_entropy"] = best.entropy
        params["intent_sources"] = best.evidence_sources
        params["intent_distribution"] = {
            h.intent: h.probability for h in best.hypotheses
        }
        return SkillUpdate(
            intent=best.most_likely,
            params=params,
            constraints=kinematic_update.constraints,
            recommend_interrupt=kinematic_update.recommend_interrupt,
            debug=f"fused intent: {best.most_likely} "
                  f"(conf={best.confidence:.2f}, "
                  f"sources={best.evidence_sources})",
        )
