"""Primitive library -- registry of atomic social action units.

Provides the building blocks for script composition. Each primitive
is a single-step generative model with known pre/post conditions.

Default primitives cover common social navigation patterns.
Plugins can register additional robot-specific primitives.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from architecture_core.core.types import AffectState, SkillRequest

from architecture_core.cognition.scripts.repertoire_types import ScriptPrimitive


class PrimitiveLibrary:
    """Registry of atomic social action primitives."""

    def __init__(self, register_defaults: bool = True) -> None:
        self._primitives: Dict[str, ScriptPrimitive] = {}
        if register_defaults:
            self._register_defaults()

    def register(self, primitive: ScriptPrimitive) -> None:
        """Register a primitive (overwrites if name exists)."""
        self._primitives[primitive.name] = primitive

    def get(self, name: str) -> Optional[ScriptPrimitive]:
        """Get a primitive by name."""
        return self._primitives.get(name)

    def get_applicable(self, situation: str) -> List[ScriptPrimitive]:
        """Return primitives whose preconditions include the given situation."""
        result = []
        for p in self._primitives.values():
            if not p.precondition_situations or situation in p.precondition_situations:
                result.append(p)
        return result

    @property
    def all_primitives(self) -> List[ScriptPrimitive]:
        return list(self._primitives.values())

    @property
    def names(self) -> List[str]:
        return list(self._primitives.keys())

    def __len__(self) -> int:
        return len(self._primitives)

    def _register_defaults(self) -> None:
        """Register built-in social navigation primitives."""
        defaults = [
            ScriptPrimitive(
                name="approach-greet",
                skill_template=SkillRequest(
                    skill="navigate",
                    params={"intent": "approach", "speed_scale": 0.6},
                ),
                precondition_situations=["open_area", "corridor_encounter", "meeting_point"],
                postcondition_situation="interaction",
                expected_affect=AffectState(valence=0.2, arousal=0.1),
                typical_duration_s=5.0,
                deontic_default="permitted",
            ),
            ScriptPrimitive(
                name="yield-pass",
                skill_template=SkillRequest(
                    skill="navigate",
                    params={"intent": "yield", "speed_scale": 0.3, "lateral_offset": 0.5},
                ),
                precondition_situations=["corridor_encounter", "doorway", "narrow_passage"],
                postcondition_situation="open_area",
                expected_affect=AffectState(valence=0.1, arousal=-0.1),
                typical_duration_s=3.0,
                deontic_default="obligatory",
            ),
            ScriptPrimitive(
                name="wait-acknowledge",
                skill_template=SkillRequest(
                    skill="navigate",
                    params={"intent": "wait", "speed_scale": 0.0},
                ),
                precondition_situations=["corridor_encounter", "doorway", "interaction", "handover"],
                postcondition_situation="corridor_encounter",
                expected_affect=AffectState(valence=0.0, arousal=-0.1),
                typical_duration_s=4.0,
                deontic_default="permitted",
            ),
            ScriptPrimitive(
                name="avoid-reroute",
                skill_template=SkillRequest(
                    skill="navigate",
                    params={"intent": "avoid", "speed_scale": 0.7, "reroute": True},
                ),
                precondition_situations=["corridor_encounter", "open_area", "hazard"],
                postcondition_situation="open_area",
                expected_affect=AffectState(valence=-0.1, arousal=0.2),
                typical_duration_s=6.0,
                deontic_default="permitted",
            ),
            ScriptPrimitive(
                name="follow-maintain",
                skill_template=SkillRequest(
                    skill="navigate",
                    params={"intent": "approach", "speed_scale": 0.5, "follow": True},
                ),
                precondition_situations=["open_area", "corridor_encounter", "interaction"],
                postcondition_situation="interaction",
                expected_affect=AffectState(valence=0.1, arousal=0.0),
                typical_duration_s=10.0,
                deontic_default="permitted",
            ),
            ScriptPrimitive(
                name="disengage-depart",
                skill_template=SkillRequest(
                    skill="navigate",
                    params={"intent": "avoid", "speed_scale": 0.5, "disengage": True},
                ),
                precondition_situations=["interaction", "handover", "meeting_point"],
                postcondition_situation="open_area",
                expected_affect=AffectState(valence=0.0, arousal=-0.2),
                typical_duration_s=4.0,
                deontic_default="permitted",
            ),
            ScriptPrimitive(
                name="handover-extend",
                skill_template=SkillRequest(
                    skill="handover",
                    params={"intent": "approach", "extend": True},
                ),
                precondition_situations=["interaction", "handover", "meeting_point"],
                postcondition_situation="handover",
                expected_affect=AffectState(valence=0.3, arousal=0.1),
                typical_duration_s=5.0,
                deontic_default="obligatory",
            ),
            ScriptPrimitive(
                name="handover-receive",
                skill_template=SkillRequest(
                    skill="handover",
                    params={"intent": "wait", "receive": True},
                ),
                precondition_situations=["handover", "interaction"],
                postcondition_situation="interaction",
                expected_affect=AffectState(valence=0.3, arousal=0.0),
                typical_duration_s=5.0,
                deontic_default="obligatory",
            ),
            # -- Pick/place primitives --
            ScriptPrimitive(
                name="pick-object",
                skill_template=SkillRequest(
                    skill="pick_place",
                    params={"mode": "pick"},
                ),
                precondition_situations=["interaction", "open_area", "workstation"],
                postcondition_situation="interaction",
                expected_affect=AffectState(valence=0.1, arousal=0.1),
                typical_duration_s=8.0,
                deontic_default="permitted",
            ),
            ScriptPrimitive(
                name="place-object",
                skill_template=SkillRequest(
                    skill="pick_place",
                    params={"mode": "place"},
                ),
                precondition_situations=["interaction", "handover", "workstation"],
                postcondition_situation="interaction",
                expected_affect=AffectState(valence=0.1, arousal=0.0),
                typical_duration_s=6.0,
                deontic_default="permitted",
            ),
            # -- Gaze primitives --
            ScriptPrimitive(
                name="gaze-at-agent",
                skill_template=SkillRequest(
                    skill="gaze",
                    params={"mode": "look_at_agent"},
                ),
                precondition_situations=[
                    "interaction", "corridor_encounter", "meeting_point", "handover",
                ],
                postcondition_situation="interaction",
                expected_affect=AffectState(valence=0.2, arousal=0.1),
                typical_duration_s=3.0,
                deontic_default="permitted",
            ),
            ScriptPrimitive(
                name="gaze-scan",
                skill_template=SkillRequest(
                    skill="gaze",
                    params={"mode": "scan"},
                ),
                precondition_situations=["open_area", "corridor_encounter", "hazard"],
                postcondition_situation="open_area",
                expected_affect=AffectState(valence=0.0, arousal=0.1),
                typical_duration_s=4.0,
                deontic_default="permitted",
            ),
            ScriptPrimitive(
                name="gaze-avert",
                skill_template=SkillRequest(
                    skill="gaze",
                    params={"mode": "avert"},
                ),
                precondition_situations=[
                    "interaction", "corridor_encounter", "narrow_passage",
                ],
                postcondition_situation="corridor_encounter",
                expected_affect=AffectState(valence=-0.1, arousal=-0.1),
                typical_duration_s=2.0,
                deontic_default="permitted",
            ),
        ]
        for p in defaults:
            self.register(p)
