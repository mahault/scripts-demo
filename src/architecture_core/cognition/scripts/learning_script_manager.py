"""Learning script manager -- wraps ScriptManager with learning.

Decorator pattern: same interface as ScriptManager but adds learning
hooks via ScriptRepertoire. When no repertoire is configured, delegates
everything to the base manager unchanged (full backward compatibility).
"""

from __future__ import annotations

from typing import Dict, Optional

from architecture_core.core.types import AffectState, PerceptBundle, SkillRequest
from architecture_core.core.status import Status

from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.repertoire_types import ScriptTrajectory


class LearningScriptManager:
    """ScriptManager with learning -- wraps existing ScriptManager.

    When a ScriptRepertoire is configured:
    1. Before select(), consult the repertoire for the best pattern
    2. Convert the pattern to a ScriptSequence if one matches
    3. If no pattern matches, compose a new one
    4. Delegate actual step execution to the wrapped ScriptManager
    5. After each step, feed results back to the repertoire

    When no repertoire is configured:
    - Delegates everything unchanged (backward compatible)
    """

    def __init__(
        self,
        base_manager: ScriptManager,
        repertoire: Optional[ScriptRepertoire] = None,
    ) -> None:
        self._base = base_manager
        self._repertoire = repertoire
        self._current_pattern_name: Optional[str] = None
        self._last_primitive: str = ""
        self._last_affect: Optional[AffectState] = None

    def select(
        self, pb: PerceptBundle, active: Optional[SkillRequest] = None,
    ) -> SkillRequest:
        """Select next step, possibly from a learned or composed script."""
        if self._repertoire is None:
            return self._base.select(pb, active)

        # Consult repertoire for situation-based pattern selection
        situation_belief = self._extract_situation_belief(pb)
        if situation_belief:
            pattern_name = self._repertoire.on_situation_recognized(
                situation_belief, t=pb.t,
            )
            if pattern_name != self._current_pattern_name and pattern_name is not None:
                self._current_pattern_name = pattern_name
                # Convert to sequence and install in base manager
                # Pass situation context so cluster-based patterns get
                # context-dependent ordering
                pattern = self._repertoire.patterns.get(pattern_name)
                if pattern is not None:
                    most_likely_sit = (
                        max(situation_belief, key=situation_belief.get)
                        if situation_belief else None
                    )
                    seq = self._repertoire.pattern_to_sequence(
                        pattern, context=most_likely_sit,
                    )
                    self._base.set_sequence(seq)

        return self._base.select(pb, active)

    def check_violation(
        self, pb: PerceptBundle, timestamp: float = 0.0,
    ):
        """Check violation and feed to repertoire."""
        violation = self._base.check_violation(pb, timestamp)
        return violation

    def notify_status(self, status: Status) -> None:
        """Notify step completion and feed to repertoire."""
        self._base.notify_status(status)

        if self._repertoire is not None and status in ("SUCCESS", "FAILURE", "TIMEOUT"):
            # Feed step result to repertoire
            self._repertoire.on_step_completed(
                t=0.0,
                primitive_name=self._last_primitive or "unknown",
                outcome=status,
                affect_before=self._last_affect,
            )

            # If the script completed (base advanced past end)
            if self._base.is_complete:
                self._repertoire.on_script_completed(status)
                self._current_pattern_name = None

    def set_last_step_info(
        self,
        primitive_name: str = "",
        affect: Optional[AffectState] = None,
    ) -> None:
        """Called by Executive to provide context for learning."""
        self._last_primitive = primitive_name
        self._last_affect = affect

    @property
    def is_complete(self) -> bool:
        return self._base.is_complete

    @property
    def is_repairing(self) -> bool:
        return self._base.is_repairing

    @property
    def current_pattern_name(self) -> Optional[str]:
        return self._current_pattern_name

    @property
    def repertoire(self) -> Optional[ScriptRepertoire]:
        return self._repertoire

    @property
    def base(self) -> ScriptManager:
        return self._base

    def _extract_situation_belief(self, pb: PerceptBundle) -> Dict[str, float]:
        """Try to get situation belief from the base manager's recognizer."""
        if (
            hasattr(self._base, '_recognizer')
            and self._base._recognizer is not None
        ):
            belief = self._base._recognizer.recognize(pb)
            return belief.distribution
        return {}
