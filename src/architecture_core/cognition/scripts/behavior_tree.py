"""Behavior tree engine for script execution.

Node types:
- ActionNode: leaf wrapping a SkillRequest (RUNNING until Executive reports)
- ConditionNode: leaf evaluating a ScriptCondition predicate
- SequenceNode: ticks children L→R, fails on first FAILURE
- SelectorNode: ticks children L→R, succeeds on first SUCCESS
- ParallelNode: ticks all children, policy-based completion
- InverterNode: decorator inverting SUCCESS↔FAILURE
- RepeatNode: decorator repeating child N times
- SucceederNode: decorator always returning SUCCESS
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional

from architecture_core.cognition.scripts.script_types import ScriptCondition
from architecture_core.core.status import Status
from architecture_core.core.types import PerceptBundle, SkillRequest

BTStatus = Literal["RUNNING", "SUCCESS", "FAILURE"]


class BTNode(ABC):
    """Abstract base for all behavior tree nodes."""

    @abstractmethod
    def tick(self, pb: PerceptBundle) -> BTStatus:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


# =====================================================================
# Leaf nodes
# =====================================================================
class ActionNode(BTNode):
    """Wraps a SkillRequest. Returns RUNNING until set_result() is called."""

    def __init__(
        self,
        request: SkillRequest,
        name: str = "",
        on_activate: Optional[Callable] = None,
    ) -> None:
        self.request = request
        self.name = name
        self._on_activate = on_activate
        self._result: Optional[BTStatus] = None
        self._started = False

    def tick(self, pb: PerceptBundle) -> BTStatus:
        if self._result is not None:
            return self._result

        if not self._started:
            self._started = True
            if self._on_activate:
                self._on_activate(self)

        return "RUNNING"

    def set_result(self, status: Status) -> None:
        """Called by BehaviorTreeManager when the Executive reports completion."""
        if status == "SUCCESS":
            self._result = "SUCCESS"
        else:
            self._result = "FAILURE"

    def reset(self) -> None:
        self._result = None
        self._started = False


class ConditionNode(BTNode):
    """Evaluates a ScriptCondition predicate. No RUNNING state."""

    def __init__(self, condition: ScriptCondition, name: str = "") -> None:
        self.condition = condition
        self.name = name

    def tick(self, pb: PerceptBundle) -> BTStatus:
        return "SUCCESS" if self.condition.evaluate(pb) else "FAILURE"

    def reset(self) -> None:
        pass


# =====================================================================
# Composite nodes
# =====================================================================
class SequenceNode(BTNode):
    """Ticks children L→R. Fails on first FAILURE. Succeeds when all succeed."""

    def __init__(self, children: List[BTNode], name: str = "") -> None:
        self.children = children
        self.name = name
        self._running_idx = 0

    def tick(self, pb: PerceptBundle) -> BTStatus:
        while self._running_idx < len(self.children):
            status = self.children[self._running_idx].tick(pb)
            if status == "RUNNING":
                return "RUNNING"
            if status == "FAILURE":
                return "FAILURE"
            # SUCCESS → advance to next child
            self._running_idx += 1

        return "SUCCESS"

    def reset(self) -> None:
        self._running_idx = 0
        for child in self.children:
            child.reset()


class SelectorNode(BTNode):
    """Ticks children L→R. Succeeds on first SUCCESS. Fails when all fail."""

    def __init__(self, children: List[BTNode], name: str = "") -> None:
        self.children = children
        self.name = name
        self._running_idx = 0

    def tick(self, pb: PerceptBundle) -> BTStatus:
        while self._running_idx < len(self.children):
            status = self.children[self._running_idx].tick(pb)
            if status == "RUNNING":
                return "RUNNING"
            if status == "SUCCESS":
                return "SUCCESS"
            # FAILURE → try next child
            self._running_idx += 1

        return "FAILURE"

    def reset(self) -> None:
        self._running_idx = 0
        for child in self.children:
            child.reset()


class ParallelNode(BTNode):
    """Ticks all children every tick. Policy determines completion."""

    def __init__(
        self,
        children: List[BTNode],
        policy: Literal["succeed_on_all", "succeed_on_one"] = "succeed_on_all",
        name: str = "",
    ) -> None:
        self.children = children
        self.policy = policy
        self.name = name

    def tick(self, pb: PerceptBundle) -> BTStatus:
        results = [child.tick(pb) for child in self.children]

        if self.policy == "succeed_on_one":
            if any(r == "SUCCESS" for r in results):
                return "SUCCESS"
            if all(r == "FAILURE" for r in results):
                return "FAILURE"
            return "RUNNING"
        else:  # succeed_on_all
            if all(r == "SUCCESS" for r in results):
                return "SUCCESS"
            if any(r == "FAILURE" for r in results):
                return "FAILURE"
            return "RUNNING"

    def reset(self) -> None:
        for child in self.children:
            child.reset()


# =====================================================================
# Decorator nodes
# =====================================================================
class InverterNode(BTNode):
    """Inverts child result: SUCCESS↔FAILURE, RUNNING unchanged."""

    def __init__(self, child: BTNode) -> None:
        self.child = child

    def tick(self, pb: PerceptBundle) -> BTStatus:
        status = self.child.tick(pb)
        if status == "SUCCESS":
            return "FAILURE"
        if status == "FAILURE":
            return "SUCCESS"
        return "RUNNING"

    def reset(self) -> None:
        self.child.reset()


class RepeatNode(BTNode):
    """Repeats child N times. count=0 means repeat forever."""

    def __init__(self, child: BTNode, count: int = 0) -> None:
        self.child = child
        self.count = count
        self._completed = 0

    def tick(self, pb: PerceptBundle) -> BTStatus:
        status = self.child.tick(pb)
        if status == "RUNNING":
            return "RUNNING"

        if status == "FAILURE":
            return "FAILURE"

        # Child succeeded
        self._completed += 1
        if self.count > 0 and self._completed >= self.count:
            return "SUCCESS"

        # Reset child for next iteration
        self.child.reset()
        return "RUNNING"

    def reset(self) -> None:
        self._completed = 0
        self.child.reset()


class SucceederNode(BTNode):
    """Always returns SUCCESS regardless of child result (RUNNING passes through)."""

    def __init__(self, child: BTNode) -> None:
        self.child = child

    def tick(self, pb: PerceptBundle) -> BTStatus:
        status = self.child.tick(pb)
        if status == "RUNNING":
            return "RUNNING"
        return "SUCCESS"

    def reset(self) -> None:
        self.child.reset()
