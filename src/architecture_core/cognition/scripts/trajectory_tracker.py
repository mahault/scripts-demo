"""Trajectory tracker -- records execution for learning.

Sits between the Executive and ScriptManager, observing the stream of
(situation_belief, action, outcome, affect) tuples and assembling them
into ScriptTrajectory objects.

Active inference: this is the data collection that generates observations
against which generative models (ScriptPatterns) are evaluated.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from architecture_core.core.types import AffectState
from architecture_core.core.status import Status

from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptTrajectory,
    TrajectoryStep,
)


class TrajectoryTracker:
    """Records execution trajectories for learning."""

    def __init__(self, config: Optional[RepertoireConfig] = None) -> None:
        self._config = config or RepertoireConfig()
        self._current: Optional[ScriptTrajectory] = None
        self._completed: List[ScriptTrajectory] = []

    def begin_trajectory(self, script_name: str, situation_type: str) -> None:
        """Start recording a new trajectory."""
        self._current = ScriptTrajectory(
            script_name=script_name,
            situation_type=situation_type,
        )

    def record_step(
        self,
        t: float,
        primitive_name: str,
        situation_belief: Optional[Dict[str, float]] = None,
        most_likely_situation: str = "",
        action_skill: str = "",
        outcome: Status = "RUNNING",
        affect_before: Optional[AffectState] = None,
        affect_after: Optional[AffectState] = None,
        violation_kl: float = 0.0,
        human_intent: str = "neutral",
        norm_snapshot: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record one step of the current trajectory."""
        if self._current is None:
            return

        step = TrajectoryStep(
            t=t,
            primitive_name=primitive_name,
            situation_belief=situation_belief or {},
            most_likely_situation=most_likely_situation,
            action_skill=action_skill,
            outcome=outcome,
            affect_before=affect_before or AffectState(),
            affect_after=affect_after or AffectState(),
            violation_kl=violation_kl,
            human_intent=human_intent,
            norm_snapshot=norm_snapshot or {},
        )
        self._current.steps.append(step)

    def end_trajectory(self, final_outcome: Status) -> Optional[ScriptTrajectory]:
        """Finalize the current trajectory and compute summary stats."""
        if self._current is None:
            return None

        traj = self._current
        traj.final_outcome = final_outcome

        # Compute duration
        if traj.steps:
            traj.total_duration_s = traj.steps[-1].t - traj.steps[0].t

        # Compute mean violation KL
        kls = [s.violation_kl for s in traj.steps if s.violation_kl > 0]
        traj.mean_violation_kl = sum(kls) / len(kls) if kls else 0.0

        # Store (bounded)
        self._completed.append(traj)
        if len(self._completed) > self._config.max_stored_trajectories:
            self._completed = self._completed[-self._config.max_stored_trajectories:]

        self._current = None
        return traj

    @property
    def current(self) -> Optional[ScriptTrajectory]:
        return self._current

    @property
    def is_recording(self) -> bool:
        return self._current is not None

    @property
    def completed_trajectories(self) -> List[ScriptTrajectory]:
        return list(self._completed)

    def get_trajectories_for_situation(self, situation: str) -> List[ScriptTrajectory]:
        """Return completed trajectories that started in the given situation."""
        return [t for t in self._completed if t.situation_type == situation]

    def clear(self) -> None:
        """Clear all stored trajectories."""
        self._completed.clear()
        self._current = None
