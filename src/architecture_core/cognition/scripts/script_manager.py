"""architecture_core/cognition/scripts/script_manager.py

Select the next SkillRequest (Layer 1 deliberation).
Supports simple ordered sequences with status-based transitions,
weak-script violation detection, and repair injection.
"""

from __future__ import annotations

from typing import Dict, Optional

from architecture_core.core.types import PerceptBundle, ScriptViolation, SkillRequest
from architecture_core.core.status import Status
from architecture_core.cognition.scripts.script_types import ScriptSequence
from architecture_core.cognition.scripts.weak_recognizer import (
    WeakScriptRecognizer,
)


class ScriptManager:
    def __init__(
        self,
        sequence: Optional[ScriptSequence] = None,
        recognizer: Optional[WeakScriptRecognizer] = None,
        violation_threshold: float = 2.0,
        repair_scripts: Optional[Dict[str, ScriptSequence]] = None,
    ) -> None:
        self._sequence = sequence
        self._current_idx = 0
        self._last_status: Status = "IDLE"

        # Variational script extensions
        self._recognizer = recognizer
        self._violation_threshold = violation_threshold
        self._repair_scripts = repair_scripts or {}

        # Repair state
        self._repair_active: Optional[ScriptSequence] = None
        self._repair_idx = 0
        self._saved_idx: Optional[int] = None  # resume point after repair

    def select(
        self, pb: PerceptBundle, active: Optional[SkillRequest]
    ) -> SkillRequest:
        # If repair script is active, serve its steps
        if self._repair_active is not None:
            if self._repair_idx < len(self._repair_active.steps):
                return self._repair_active.steps[self._repair_idx].request
            # Repair complete — resume normal script
            self._repair_active = None
            self._repair_idx = 0

        if self._sequence is None or len(self._sequence.steps) == 0:
            return SkillRequest(skill="navigate", goal={"x": 0.0, "y": 0.0})

        step = self._sequence.steps[self._current_idx]
        return step.request

    def notify_status(self, status: Status) -> None:
        """Called by the Executive when the active skill terminates."""
        # Advance repair script if active
        if self._repair_active is not None:
            step = self._repair_active.steps[self._repair_idx]
            if status == step.transition_on:
                self._repair_idx += 1
                if self._repair_idx >= len(self._repair_active.steps):
                    # Repair done
                    self._repair_active = None
                    self._repair_idx = 0
            self._last_status = status
            return

        if self._sequence is None:
            return

        step = self._sequence.steps[self._current_idx]
        if status == step.transition_on:
            self._current_idx += 1
            if self._current_idx >= len(self._sequence.steps):
                if self._sequence.loop:
                    self._current_idx = 0
                else:
                    self._current_idx = len(self._sequence.steps) - 1
        self._last_status = status

    def check_violation(
        self, pb: PerceptBundle, timestamp: float = 0.0
    ) -> Optional[ScriptViolation]:
        """Compare expected situation against recognized observation.

        Returns ScriptViolation if KL divergence exceeds threshold,
        None otherwise.  Returns None if no recognizer is configured
        or the current step has no expected_situation.
        """
        if self._recognizer is None:
            return None
        if self._sequence is None or len(self._sequence.steps) == 0:
            return None

        step = self._sequence.steps[self._current_idx]
        expected = step.expected_situation
        if expected is None:
            return None

        belief = self._recognizer.recognize(pb)
        kl = self._recognizer.kl_from_expected(belief, expected)

        if kl >= self._violation_threshold:
            violation = ScriptViolation(
                script_name=self._sequence.name,
                step_index=self._current_idx,
                expected_situation=expected,
                observed_situation=belief.most_likely,
                kl_divergence=round(kl, 4),
                timestamp=timestamp,
            )
            # Attempt repair injection
            self._try_inject_repair(belief.most_likely)
            return violation

        return None

    def _try_inject_repair(self, observed_situation: str) -> None:
        """Activate a repair script if one matches the observed situation."""
        if observed_situation in self._repair_scripts:
            self._saved_idx = self._current_idx
            self._repair_active = self._repair_scripts[observed_situation]
            self._repair_idx = 0

    @property
    def is_complete(self) -> bool:
        if self._sequence is None:
            return False
        return (
            self._current_idx >= len(self._sequence.steps) - 1
            and self._last_status == "SUCCESS"
            and not self._sequence.loop
        )

    @property
    def is_repairing(self) -> bool:
        return self._repair_active is not None

    def set_sequence(self, sequence: ScriptSequence) -> None:
        """Replace the active script sequence (used by LearningScriptManager)."""
        self._sequence = sequence
        self._current_idx = 0
        self._last_status = "IDLE"
