"""Robot-agnostic pick-and-place skill.

Manages the state machine for picking up or placing down objects.
Two modes: pick (grasp object) and place (release object).

Actions are expressed through SkillUpdate.params; actual motor control
is delegated to hook methods that TIAGo (or other plugins) can override.
"""

from __future__ import annotations

from architecture_core.cognition.tom.intent_policy import IntentPolicy
from architecture_core.core.status import Status
from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate
from architecture_core.skills.base import Skill


class PickPlaceSkill(Skill):
    """Pick or place objects via arm + gripper control."""

    name = "pick_place"

    def __init__(self) -> None:
        self._mode: str = "pick"
        self._phase: str = "idle"
        self._target_object: str = ""
        self._target_position: tuple = ()
        self._elapsed: float = 0.0
        self._timeout: float = 10.0
        self._last_t: float = 0.0
        self._verify_ticks: int = 0

    # ------------------------------------------------------------------
    # Skill interface
    # ------------------------------------------------------------------
    def start(self, req: SkillRequest) -> None:
        self._mode = req.params.get("mode", "pick")
        self._target_object = req.goal.get("object", "")
        self._target_position = req.goal.get("position", ())
        self._timeout = req.timeout_s
        self._elapsed = 0.0
        self._last_t = 0.0
        self._phase = "approaching"
        self._verify_ticks = 0
        print(f"  [PICK] start mode={self._mode} obj={self._target_object}"
              f" timeout={self._timeout:.1f}s")

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

        # Timeout check
        if self._elapsed > self._timeout:
            print(f"  [PICK] TIMEOUT after {self._elapsed:.1f}s phase={self._phase}")
            return "TIMEOUT"

        # Read world state
        held_object = pb.world.get("held_object")
        arm_at_target = pb.world.get("arm_at_target", False)

        if self._mode == "pick":
            return self._tick_pick(held_object, arm_at_target)
        else:
            return self._tick_place(held_object, arm_at_target)

    def stop(self, reason: str = "") -> None:
        self._on_retract()
        self._phase = "idle"
        self._elapsed = 0.0

    # ------------------------------------------------------------------
    # Pick mode state machine
    # ------------------------------------------------------------------
    def _tick_pick(self, held_object: object, arm_at_target: bool) -> Status:
        if self._phase == "approaching":
            target = self._target_position or self._target_object
            self._on_arm_move(target)
            if arm_at_target:
                self._phase = "grasping"
                self._on_grasp()
            return "RUNNING"

        if self._phase == "grasping":
            if held_object:
                self._phase = "verifying"
                self._verify_ticks = 0
            return "RUNNING"

        if self._phase == "verifying":
            self._verify_ticks += 1
            if not held_object:
                return "FAILURE"
            if self._verify_ticks >= 3:
                return "SUCCESS"
            return "RUNNING"

        return "RUNNING"

    # ------------------------------------------------------------------
    # Place mode state machine
    # ------------------------------------------------------------------
    def _tick_place(self, held_object: object, arm_at_target: bool) -> Status:
        if self._phase == "approaching":
            target = self._target_position or self._target_object
            self._on_arm_move(target)
            if arm_at_target:
                self._phase = "releasing"
                self._on_release()
            return "RUNNING"

        if self._phase == "releasing":
            if not held_object:
                self._phase = "verifying"
                self._verify_ticks = 0
            return "RUNNING"

        if self._phase == "verifying":
            self._verify_ticks += 1
            if held_object:
                return "FAILURE"
            if self._verify_ticks >= 3:
                return "SUCCESS"
            return "RUNNING"

        return "RUNNING"

    # ------------------------------------------------------------------
    # Hook methods (no-op here, overridden by plugin subclasses)
    # ------------------------------------------------------------------
    def _on_arm_move(self, target: object) -> None:
        """Called when arm should move toward target."""

    def _on_grasp(self) -> None:
        """Called when gripper should close."""

    def _on_release(self) -> None:
        """Called when gripper should open."""

    def _on_retract(self) -> None:
        """Called when arm should retract to safe position."""


class PickPlaceIntentPolicy(IntentPolicy):
    """Maps social intents to pick/place-specific parameters."""

    def approach(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["arm_speed"] = 1.0
        base.params["grip_force"] = 1.0
        return base

    def avoid(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["arm_speed"] = 0.0
        base.params["retract"] = True
        return base

    def yield_(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["arm_speed"] = 0.3
        base.params["grip_force"] = 0.8
        # Check if someone is very close
        prox = pb.social.get("proxemics", {})
        closest = prox.get("closest_distance", float("inf"))
        if closest < 0.5:
            base.constraints["caution_zone"] = True
        return base

    def wait(
        self, pb: PerceptBundle, req: SkillRequest, base: SkillUpdate
    ) -> SkillUpdate:
        base.params["arm_speed"] = 0.0
        return base
