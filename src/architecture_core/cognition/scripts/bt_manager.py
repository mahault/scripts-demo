"""Behavior tree manager — drop-in alternative to ScriptManager.

Implements the same interface (select, notify_status, check_violation)
so the Executive can use either linear ScriptSequences or full BTs
without any changes.
"""

from __future__ import annotations

from typing import Optional

from architecture_core.cognition.scripts.behavior_tree import (
    ActionNode,
    BTNode,
    BTStatus,
    SequenceNode,
)
from architecture_core.cognition.scripts.script_types import ScriptCondition, ScriptSequence
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer
from architecture_core.core.status import Status
from architecture_core.core.types import PerceptBundle, ScriptViolation, SkillRequest


class BehaviorTreeManager:
    """Manages a behavior tree, exposing the ScriptManager interface."""

    def __init__(
        self,
        root: BTNode,
        recognizer: Optional[WeakScriptRecognizer] = None,
        violation_threshold: float = 2.0,
    ) -> None:
        self._root = root
        self._recognizer = recognizer
        self._violation_threshold = violation_threshold

        self._active_action: Optional[ActionNode] = None
        self._tree_status: BTStatus = "RUNNING"

        # Walk tree and wire up action callbacks
        self._wire_callbacks(root)

    def select(
        self, pb: PerceptBundle, active: Optional[SkillRequest] = None
    ) -> SkillRequest:
        """Tick the tree and return the active action's SkillRequest."""
        if self._tree_status in ("SUCCESS", "FAILURE"):
            # Tree already completed — return last action or default
            if self._active_action:
                return self._active_action.request
            return SkillRequest(skill="navigate", goal={"x": 0.0, "y": 0.0})

        self._tree_status = self._root.tick(pb)

        if self._active_action:
            return self._active_action.request
        return SkillRequest(skill="navigate", goal={"x": 0.0, "y": 0.0})

    def notify_status(self, status: Status) -> None:
        """Feed skill completion status to the active ActionNode."""
        if self._active_action:
            self._active_action.set_result(status)

    def check_violation(
        self, pb: PerceptBundle, timestamp: float = 0.0
    ) -> Optional[ScriptViolation]:
        """Delegate to recognizer if present."""
        if self._recognizer is None:
            return None
        # BT doesn't have per-step expected_situation by default
        return None

    @property
    def is_complete(self) -> bool:
        return self._tree_status in ("SUCCESS", "FAILURE")

    @property
    def is_repairing(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _on_action_activated(self, action: ActionNode) -> None:
        """Callback fired when an ActionNode starts executing."""
        self._active_action = action

    def _wire_callbacks(self, node: BTNode) -> None:
        """Recursively set on_activate callbacks on all ActionNodes."""
        if isinstance(node, ActionNode):
            node._on_activate = self._on_action_activated
        # Check for composite/decorator children
        if hasattr(node, "children"):
            for child in node.children:
                self._wire_callbacks(child)
        if hasattr(node, "child"):
            self._wire_callbacks(node.child)


def tree_from_sequence(seq: ScriptSequence) -> BTNode:
    """Convert a ScriptSequence into a SequenceNode of ActionNodes.

    This bridges existing linear scripts with the BT engine.
    """
    actions = [
        ActionNode(step.request, name=step.primitive_name or f"step_{i}")
        for i, step in enumerate(seq.steps)
    ]
    return SequenceNode(actions, name=seq.name)
