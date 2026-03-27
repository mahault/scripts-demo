"""Robot-agnostic handover skill.

Manages the state machine for object handover between robot and human.
Two modes: extend (robot offers object) and receive (robot accepts object).

Actions are expressed through SkillUpdate.params; actual motor control
is delegated to hook methods that TIAGo (or other plugins) can override.
"""

from __future__ import annotations

from architecture_core.cognition.tom.intent_policy import IntentPolicy
from architecture_core.core.status import Status
from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate
from architecture_core.skills.base import Skill


class HandoverSkill(Skill):
    """Two-phase handover: extend (offer) or receive (accept)."""

    name = "handover"

    def __init__(self) -> None:
        self._mode: str = "extend"
        self._phase: str = "idle"
        self._target_agent: str = ""
        self._elapsed: float = 0.0
        self._timeout: float = 10.0
        self._last_t: float = 0.0
        self._held_at_start: bool = False

    # ------------------------------------------------------------------
    # Skill interface
    # ------------------------------------------------------------------
    def start(self, req: SkillRequest) -> None:
        self._mode = req.params.get("mode", "extend")
        self._target_agent = req.goal.get("agent", "")
        self._timeout = req.timeout_s
        self._elapsed = 0.0
        self._last_t = 0.0

        if self._mode == "receive":
            self._phase = "waiting_to_receive"
        else:
            self._phase = "extending"
            self._held_at_start = True

    def tick(self, pb: PerceptBundle, update: SkillUpdate) -> Status:
        # Track elapsed time
        if self._last_t > 0.0:
            self._elapsed += pb.t - self._last_t
        self._last_t = pb.t

        # Emergency stop
        if update.constraints.get("emergency_stop"):
            return "RUNNING"

        # Retract on avoid intent
        if update.params.get("retract"):
            self._on_retract()
            self._phase = "idle"
            return "FAILURE"

        # Read world state
        held_object = pb.world.get("held_object")
        object_near_hand = pb.world.get("object_near_hand", False)

        # Check agent proximity
        agent_close = self._is_agent_close(pb)

        if self._mode == "extend":
            return self._tick_extend(held_object, object_near_hand, agent_close)
        else:
            return self._tick_receive(held_object, object_near_hand)

    def stop(self, reason: str = "") -> None:
        self._on_retract()
        self._phase = "idle"
        self._elapsed = 0.0

    # ------------------------------------------------------------------
    # Extend mode state machine
    # ------------------------------------------------------------------
    def _tick_extend(
        self, held_object: object, object_near_hand: bool, agent_close: bool
    ) -> Status:
        if self._phase == "extending":
            self._on_extend()
            if agent_close:
                self._phase = "waiting_for_transfer"
                self._on_release()
            return "RUNNING"

        if self._phase == "waiting_for_transfer":
            # Object taken = held_object cleared
            if not held_object:
                return "SUCCESS"
            if self._elapsed > self._timeout:
                return "FAILURE"
            return "RUNNING"

        return "RUNNING"

    # ------------------------------------------------------------------
    # Receive mode state machine
    # ------------------------------------------------------------------
    def _tick_receive(self, held_object: object, object_near_hand: bool) -> Status:
        if self._phase == "waiting_to_receive":
            self._on_extend()
            self._on_release()  # open gripper to receive
            if object_near_hand:
                self._phase = "grasping"
                self._on_grasp()
            elif self._elapsed > self._timeout:
                return "FAILURE"
            return "RUNNING"

        if self._phase == "grasping":
            if held_object:
                return "SUCCESS"
            if self._elapsed > self._timeout:
                return "FAILURE"
            return "RUNNING"

        return "RUNNING"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_agent_close(self, pb: PerceptBundle) -> bool:
        prox = pb.social.get("proxemics", {})
        closest = prox.get("closest_distance", float("inf"))
        return closest < 1.0  # personal zone

    # ------------------------------------------------------------------
    # Hook methods (no-op here, overridden by plugin subclasses)
    # ------------------------------------------------------------------
    def _on_extend(self) -> None:
        """Called when arm should extend for handover."""

    def _on_retract(self) -> None:
        """Called when arm should retract."""

    def _on_grasp(self) -> None:
        """Called when gripper should close."""

    def _on_release(self) -> None:
        """Called when gripper should open."""


class HandoverIntentPolicy(IntentPolicy):
    """Maps social intents to handover-specific parameters."""

    def approach(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["arm_speed"] = 1.0
        base.params["speed_scale"] = 1.0
        return base

    def avoid(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["arm_speed"] = 0.0
        base.params["speed_scale"] = 0.0
        base.params["retract"] = True
        return base

    def yield_(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["arm_speed"] = 0.5
        base.params["speed_scale"] = 0.5
        return base

    def wait(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["arm_speed"] = 0.0
        base.params["speed_scale"] = 0.0
        return base
