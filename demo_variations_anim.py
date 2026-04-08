"""Animated Demo Variations — one full-pipeline GIF per experiment variant.

Each variant gets the complete 4-panel animated layout:
  - Scene panel: top-down view with robot, gaze, people, trail
  - Situation State Machine: FSM with backbone / shortcut G labels
  - Fragment Composition: weights + primitive blocks
  - Execution Sequence: step-by-step with progress markers
  - Status bar: current step, G value, progress dots

Each scene_type draws a visually distinct physical environment:
  - reception_full:     desk + queue stanchions + wall
  - corridor:           hallway with walls, people passing, no desk
  - reception_no_queue: desk but stanchions removed (dashed ghost outlines)
  - reception_sign:     desk + queue + "COME TO COUNTER" sign

Generates 8 GIFs in demo_figures/.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import networkx as nx
import numpy as np

from architecture_core.core.types import AffectState, SkillRequest
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    ScriptPrimitive,
    WeightedPattern,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_composer import ScriptComposer
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire


# ================================================================
# Domain setup
# ================================================================
def _register_reception_primitives(lib: PrimitiveLibrary) -> None:
    domain_prims = [
        ScriptPrimitive(
            name="scan-environment",
            skill_template=SkillRequest(skill="gaze", params={"mode": "scan_area"}),
            precondition_situations=["open_area", "corridor_encounter"],
            postcondition_situation="scene_assessed",
            expected_affect=AffectState(valence=0.0, arousal=0.1),
            typical_duration_s=3.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="position-in-queue",
            skill_template=SkillRequest(
                skill="navigate", params={"intent": "queue", "speed_scale": 0.4},
            ),
            precondition_situations=["scene_assessed", "open_area"],
            postcondition_situation="in_queue",
            expected_affect=AffectState(valence=0.0, arousal=-0.1),
            typical_duration_s=4.0,
            deontic_default="obligatory",
        ),
        ScriptPrimitive(
            name="wait-for-turn",
            skill_template=SkillRequest(
                skill="navigate", params={"intent": "wait", "speed_scale": 0.0},
            ),
            precondition_situations=["in_queue"],
            postcondition_situation="ready_for_service",
            expected_affect=AffectState(valence=0.0, arousal=-0.2),
            typical_duration_s=10.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="approach-counter",
            skill_template=SkillRequest(
                skill="navigate", params={"intent": "approach", "speed_scale": 0.5},
            ),
            precondition_situations=["ready_for_service", "open_area"],
            postcondition_situation="at_counter",
            expected_affect=AffectState(valence=0.2, arousal=0.1),
            typical_duration_s=4.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="engage-staff",
            skill_template=SkillRequest(
                skill="gaze", params={"mode": "look_at_staff"},
            ),
            precondition_situations=["at_counter", "interaction"],
            postcondition_situation="interaction",
            expected_affect=AffectState(valence=0.3, arousal=0.1),
            typical_duration_s=3.0,
            deontic_default="permitted",
        ),
    ]
    for p in domain_prims:
        lib.register(p)


def _make_fragment(name, primitives, affinity):
    return ScriptPattern(
        name=name,
        primitives_sequence=sorted(primitives),
        primitive_cluster=set(primitives),
        precision=0.1,
        situation_affinity=affinity,
    )


def _make_all_fragments() -> Dict[str, ScriptPattern]:
    return {
        "observe_scene": _make_fragment(
            "observe_scene",
            {"scan-environment", "gaze-scan"},
            {"reception": 0.9, "corridor": 0.4},
        ),
        "queue_position": _make_fragment(
            "queue_position",
            {"position-in-queue", "yield-pass"},
            {"reception": 0.8, "corridor": 0.3},
        ),
        "wait_patiently": _make_fragment(
            "wait_patiently",
            {"wait-for-turn", "wait-acknowledge"},
            {"reception": 0.7, "corridor": 0.5},
        ),
        "approach_service": _make_fragment(
            "approach_service",
            {"approach-counter", "gaze-at-agent"},
            {"reception": 0.9, "open_area": 0.4},
        ),
        "courtesy_space": _make_fragment(
            "courtesy_space",
            {"yield-pass", "gaze-avert"},
            {"reception": 0.3, "corridor": 0.8},
        ),
        "direct_approach": _make_fragment(
            "direct_approach",
            {"approach-counter", "engage-staff"},
            {"reception": 0.95, "corridor": 0.2},
        ),
    }


# ================================================================
# Variant configurations
# ================================================================
@dataclass
class VariantConfig:
    name: str
    context: str
    fragment_names: List[str]
    annotation: str
    file_tag: str
    scene_type: str = "reception_full"


VARIANTS = [
    # Set A: Environment changes
    VariantConfig(
        name="A1: Reception (baseline)",
        context="reception",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space"],
        annotation="Full queue norm from environment structure",
        file_tag="A1_reception_baseline",
        scene_type="reception_full",
    ),
    VariantConfig(
        name="A2: Corridor",
        context="corridor",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space"],
        annotation="Change the space, change the cognition (Lefebvre)",
        file_tag="A2_corridor",
        scene_type="corridor",
    ),
    VariantConfig(
        name="A3: Degraded reception",
        context="reception",
        fragment_names=["observe_scene", "queue_position",
                        "approach_service", "courtesy_space"],
        annotation="Norm was in the environment, not the robot (Akrich)",
        file_tag="A3_degraded_reception",
        scene_type="reception_no_queue",
    ),
    VariantConfig(
        name="A4: Enriched reception",
        context="reception",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space", "direct_approach"],
        annotation="Added cues reshape the affordance landscape (Latour)",
        file_tag="A4_enriched_reception",
        scene_type="reception_sign",
    ),
    VariantConfig(
        name="A5: Corridor (both sides)",
        context="corridor",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space"],
        annotation="Robot threads between pedestrians on both sides",
        file_tag="A5_corridor_both_sides",
        scene_type="corridor_both",
    ),
    VariantConfig(
        name="A6: Enriched + no queue cues",
        context="reception",
        fragment_names=["observe_scene", "approach_service", "courtesy_space",
                        "direct_approach"],
        annotation="Sign displaces queue norms -- does direct approach override?",
        file_tag="A6_enriched_no_queue",
        scene_type="reception_sign_direct",
    ),
    # Set B: System changes — same environment (reception_full), system differs
    VariantConfig(
        name="B5: Baseline (all 5)",
        context="reception",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space"],
        annotation="Full fragment repertoire",
        file_tag="B5_baseline_all5",
        scene_type="reception_full",
    ),
    VariantConfig(
        name="B6: Missing fragment",
        context="reception",
        fragment_names=["observe_scene", "queue_position",
                        "approach_service", "courtesy_space"],
        annotation="System lacks wait knowledge",
        file_tag="B6_missing_fragment",
        scene_type="reception_full",
    ),
    VariantConfig(
        name="B7: Extra fragment",
        context="reception",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space", "direct_approach"],
        annotation="Conflicting prior knowledge",
        file_tag="B7_extra_fragment",
        scene_type="reception_full",
    ),
    VariantConfig(
        name="B8: Minimal fragments",
        context="reception",
        fragment_names=["observe_scene", "approach_service"],
        annotation="Sparse knowledge -- can it still compose?",
        file_tag="B8_minimal_fragments",
        scene_type="reception_full",
    ),
]


# ================================================================
# Pipeline result
# ================================================================
@dataclass
class PipelineResult:
    library: PrimitiveLibrary
    fragments: List[ScriptPattern]
    composite: ScriptPattern
    weighted: List[WeightedPattern]
    graph_edges: Dict[Tuple[str, str], float]
    sequence: List[str]
    situation_chain: list
    topology: dict
    context: str
    variant_name: str
    annotation: str
    prim_to_frag: Dict[str, List[str]] = field(default_factory=dict)


# ================================================================
# Free energy
# ================================================================
def compute_transition_fe(topology, from_prim, to_prim):
    w = topology.get((from_prim, to_prim), 0.0)
    eps = 0.01
    return -math.log(w + eps)


def compute_step_fe(topology, sequence, step_idx):
    if step_idx <= 0 or step_idx >= len(sequence):
        return 0.0
    return compute_transition_fe(topology, sequence[step_idx - 1],
                                 sequence[step_idx])


# ================================================================
# Run pipeline for one variant
# ================================================================
DOMAIN_SITUATIONS = {
    "scene_assessed", "in_queue", "ready_for_service",
    "at_counter", "interaction",
}


def run_pipeline(config: VariantConfig) -> PipelineResult:
    all_frags = _make_all_fragments()
    lib = PrimitiveLibrary()
    _register_reception_primitives(lib)

    fragments = [all_frags[name] for name in config.fragment_names]

    cfg = RepertoireConfig()
    rep = ScriptRepertoire(lib, cfg, initial_patterns=fragments)
    rep.enable_compositional_mode()

    query_scores = {}
    for name, pat in rep.patterns.items():
        aff = pat.situation_affinity.get(config.context, 0.0)
        query_scores[name] = aff * pat.precision

    weighted = rep.retrieve_composition(query_scores)

    composer = ScriptComposer(lib, cfg)
    composite = composer.compose_from_patterns(weighted, config.context)
    if composite is None:
        composite = ScriptPattern(name="empty_composite")

    # Graph edges
    graph = rep._pattern_graph or {}
    graph_edges = {}
    for a, neighbors in graph.items():
        for b, w in neighbors.items():
            if (b, a) not in graph_edges:
                graph_edges[(a, b)] = w

    # Topology
    topo = composite.context_topology.get(config.context, {})
    if not topo:
        topo = next(iter(composite.context_topology.values()), {})

    # Situation chain
    sequence = list(composite.primitives_sequence)
    situation_chain = []
    domain_sit = "open_area"
    for prim_name in sequence:
        prim = lib.get(prim_name)
        post_sit = prim.postcondition_situation if prim else domain_sit
        if post_sit in DOMAIN_SITUATIONS:
            new_domain = post_sit
        else:
            new_domain = domain_sit
        situation_chain.append((prim_name, domain_sit, new_domain, post_sit))
        domain_sit = new_domain

    # Primitive -> fragment mapping
    prim_to_frag = {}
    for frag in fragments:
        for p in frag.primitive_cluster:
            prim_to_frag.setdefault(p, []).append(frag.name)

    print(f"  [{config.file_tag}] Composed {len(sequence)} primitives: "
          f"{' -> '.join(sequence)}")

    return PipelineResult(
        library=lib,
        fragments=fragments,
        composite=composite,
        weighted=weighted,
        graph_edges=graph_edges,
        sequence=sequence,
        situation_chain=situation_chain,
        topology=topo,
        context=config.context,
        variant_name=config.name,
        annotation=config.annotation,
        prim_to_frag=prim_to_frag,
    )


# ================================================================
# Scene constants — per scene type
# ================================================================
XMIN, XMAX = -0.5, 10.5
YMIN, YMAX = -0.5, 10.5

# --- Reception layout ---
DESK_X, DESK_Y = 5.0, 8.5
DESK_W, DESK_H = 3.0, 0.6

SITUATION_POS_RECEPTION = {
    "open_area":          (2.0, 2.0),
    "corridor_encounter": (2.0, 2.0),
    "scene_assessed":     (2.5, 3.0),
    "in_queue":           (4.0, 4.0),
    "ready_for_service":  (4.0, 5.5),
    "at_counter":         (5.2, 7.0),
    "interaction":        (5.5, 7.5),
}

PERSON_AT_DESK = (5.5, 7.5)
PERSON_IN_QUEUE = (4.0, 5.5)
EXIT_RIGHT = (10.5, 7.5)

PERSON_A_TARGETS_RECEPTION = {
    "open_area": PERSON_AT_DESK, "corridor_encounter": PERSON_AT_DESK,
    "scene_assessed": PERSON_AT_DESK, "in_queue": PERSON_AT_DESK,
    "ready_for_service": EXIT_RIGHT, "at_counter": EXIT_RIGHT,
    "interaction": EXIT_RIGHT,
}
PERSON_B_TARGETS_RECEPTION = {
    "open_area": PERSON_IN_QUEUE, "corridor_encounter": PERSON_IN_QUEUE,
    "scene_assessed": PERSON_IN_QUEUE, "in_queue": PERSON_IN_QUEUE,
    "ready_for_service": PERSON_AT_DESK, "at_counter": EXIT_RIGHT,
    "interaction": EXIT_RIGHT,
}

# --- Corridor layout ---
# Robot walks centrally, then yields LEFT when encountering oncoming
# pedestrians who walk on the RIGHT.  Clear spatial separation.
CORRIDOR_LEFT_WALL = 2.5
CORRIDOR_RIGHT_WALL = 7.5
CORRIDOR_DOOR_Y = 9.0

SITUATION_POS_CORRIDOR = {
    "open_area":          (5.0, 1.5),
    "corridor_encounter": (5.0, 1.5),
    "scene_assessed":     (5.0, 3.0),
    "in_queue":           (3.3, 5.0),   # yields LEFT away from people
    "ready_for_service":  (4.5, 6.5),   # back toward center
    "at_counter":         (5.0, 8.0),
    "interaction":        (5.0, 8.5),
}

# Corridor people: oncoming pedestrians on the RIGHT side.
# Robot yields LEFT — clear spatial separation shows courtesy behaviour.
CORRIDOR_PERSON_A_START = (6.5, 8.0)
CORRIDOR_PERSON_A_EXIT = (6.5, -1.0)
CORRIDOR_PERSON_B_START = (6.0, 7.0)
CORRIDOR_PERSON_B_EXIT = (6.0, -1.0)

PERSON_A_TARGETS_CORRIDOR = {
    "open_area": CORRIDOR_PERSON_A_START,
    "corridor_encounter": CORRIDOR_PERSON_A_START,
    "scene_assessed": (6.5, 6.0),
    "in_queue": (6.5, 3.5),        # passes robot on the RIGHT
    "ready_for_service": (6.5, 1.5),
    "at_counter": CORRIDOR_PERSON_A_EXIT,
    "interaction": CORRIDOR_PERSON_A_EXIT,
}
PERSON_B_TARGETS_CORRIDOR = {
    "open_area": CORRIDOR_PERSON_B_START,
    "corridor_encounter": CORRIDOR_PERSON_B_START,
    "scene_assessed": (6.0, 5.5),
    "in_queue": (6.0, 4.0),        # passes robot on the RIGHT
    "ready_for_service": (6.0, 2.0),
    "at_counter": CORRIDOR_PERSON_B_EXIT,
    "interaction": CORRIDOR_PERSON_B_EXIT,
}


# --- Corridor (both sides) layout ---
# P1 slightly right of center (x=5.6), starts far (y=8.5) — encountered SECOND.
# P2 slightly left of center (x=4.2), starts close (y=6.0) — encountered FIRST.
# The stagger creates sequential encounters: robot first avoids P2 (shifts right),
# then as P2 passes and P1 arrives, robot avoids P1 (shifts left) — a zigzag.
CORRIDOR_BOTH_PERSON_A_START = (5.6, 8.5)
CORRIDOR_BOTH_PERSON_A_EXIT  = (5.6, -1.0)
CORRIDOR_BOTH_PERSON_B_START = (4.2, 6.0)
CORRIDOR_BOTH_PERSON_B_EXIT  = (4.2, -1.0)

PERSON_A_TARGETS_CORRIDOR_BOTH = {
    "open_area":          CORRIDOR_BOTH_PERSON_A_START,
    "corridor_encounter": CORRIDOR_BOTH_PERSON_A_START,
    "scene_assessed":     (5.6, 7.0),
    "in_queue":           (5.6, 4.5),
    "ready_for_service":  (5.6, 1.5),
    "at_counter":         CORRIDOR_BOTH_PERSON_A_EXIT,
    "interaction":        CORRIDOR_BOTH_PERSON_A_EXIT,
}
PERSON_B_TARGETS_CORRIDOR_BOTH = {
    "open_area":          CORRIDOR_BOTH_PERSON_B_START,
    "corridor_encounter": CORRIDOR_BOTH_PERSON_B_START,
    "scene_assessed":     (4.2, 4.5),
    "in_queue":           (4.2, 2.5),
    "ready_for_service":  (4.2, 0.5),
    "at_counter":         CORRIDOR_BOTH_PERSON_B_EXIT,
    "interaction":        CORRIDOR_BOTH_PERSON_B_EXIT,
}


# --- Sign-direct layout ---
# "Come directly to counter" environment — no queue, people walk straight up.
# Person A is already at the desk being served.
# Person B walks directly from entrance toward the desk (no queue detour).
PERSON_DIRECT_AT_DESK = (6.0, 7.8)
PERSON_DIRECT_APPROACHING = (3.0, 3.0)

# Sign is on the LEFT side — robot approaches the left end of the desk,
# a completely different path from the standard center-right approach.
# scene_assessed is near the desk (post-interaction scan), not back at entrance.
SITUATION_POS_SIGN_DIRECT = {
    "open_area":          (1.5, 1.0),
    "corridor_encounter": (1.5, 1.0),
    "scene_assessed":     (3.2, 8.0),     # stays at desk — scans from counter
    "in_queue":           (2.0, 5.5),     # walks up the left side
    "ready_for_service":  (2.5, 7.0),     # approaching desk from left
    "at_counter":         (3.5, 8.0),     # left edge of desk
    "interaction":        (3.5, 8.2),     # stays at left edge
}

# A6 situation chain: open_area → at_counter → interaction → scene_assessed
# Person A: at desk being served, leaves when robot engages staff
# Person B: in queue, moves to desk when A leaves, stays there
PERSON_A_TARGETS_SIGN_DIRECT = {
    "open_area": PERSON_AT_DESK, "corridor_encounter": PERSON_AT_DESK,
    "scene_assessed": EXIT_RIGHT, "in_queue": PERSON_AT_DESK,
    "ready_for_service": EXIT_RIGHT,
    "at_counter": PERSON_AT_DESK,      # still at desk when robot arrives
    "interaction": EXIT_RIGHT,          # leaves as robot engages staff
}
PERSON_B_TARGETS_SIGN_DIRECT = {
    "open_area": PERSON_IN_QUEUE, "corridor_encounter": PERSON_IN_QUEUE,
    "scene_assessed": PERSON_AT_DESK, "in_queue": PERSON_IN_QUEUE,
    "ready_for_service": PERSON_AT_DESK,
    "at_counter": PERSON_IN_QUEUE,     # still waiting when robot arrives
    "interaction": PERSON_AT_DESK,     # moves to desk when A leaves
}


def get_situation_pos(scene_type):
    if scene_type in ("corridor", "corridor_both"):
        return SITUATION_POS_CORRIDOR
    if scene_type == "reception_sign_direct":
        return SITUATION_POS_SIGN_DIRECT
    return SITUATION_POS_RECEPTION


def get_person_target_maps(scene_type):
    if scene_type == "corridor_both":
        return PERSON_A_TARGETS_CORRIDOR_BOTH, PERSON_B_TARGETS_CORRIDOR_BOTH
    if scene_type == "corridor":
        return PERSON_A_TARGETS_CORRIDOR, PERSON_B_TARGETS_CORRIDOR
    if scene_type == "reception_sign_direct":
        return PERSON_A_TARGETS_SIGN_DIRECT, PERSON_B_TARGETS_SIGN_DIRECT
    return PERSON_A_TARGETS_RECEPTION, PERSON_B_TARGETS_RECEPTION


# ================================================================
# Colors
# ================================================================
COLOR_ROBOT = "#2E86C1"
COLOR_ROBOT_BODY = "#3498DB"
COLOR_PERSON = "#E67E22"
COLOR_PERSON2 = "#E74C3C"
COLOR_DESK = "#7F8C8D"
COLOR_FLOOR = "#F5F5DC"
COLOR_CORRIDOR_FLOOR = "#E8E0D0"
COLOR_GAZE = "#F1C40F"
COLOR_GAZE_SCAN = "#27AE60"
COLOR_WALL = "#D5DBDB"
COLOR_SIGN = "#27AE60"

FRAG_PALETTE = ["#4A90D9", "#50C878", "#2ECC71", "#E8734A", "#9B59B6",
                "#F39C12", "#E74C3C", "#1ABC9C"]


def get_frag_colors(fragments):
    return {f.name: FRAG_PALETTE[i % len(FRAG_PALETTE)]
            for i, f in enumerate(fragments)}


# ================================================================
# Position helpers
# ================================================================
def _lerp_pos(p1, p2, t):
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)


def get_person_positions(pre_sit, post_sit, t_frac, scene_type):
    a_targets, b_targets = get_person_target_maps(scene_type)
    a1 = a_targets.get(pre_sit, EXIT_RIGHT)
    a2 = a_targets.get(post_sit, EXIT_RIGHT)
    a_pos = _lerp_pos(a1, a2, t_frac)
    b1 = b_targets.get(pre_sit, EXIT_RIGHT)
    b2 = b_targets.get(post_sit, EXIT_RIGHT)
    b_pos = _lerp_pos(b1, b2, t_frac)
    return a_pos, b_pos


def _is_visible(pos):
    return XMIN < pos[0] < XMAX and YMIN < pos[1] < YMAX


def _apply_person_avoidance(robot_pos, person_a_pos, person_b_pos,
                            min_dist=1.2):
    """General spatial norm: maintain personal space.

    If the robot's interpolated position is within min_dist of a visible
    person, it gets pushed away proportionally.  This isn't hardcoded
    path-planning — it's the same proximity repulsion that drives the
    corridor yielding, applied as a universal spatial norm.
    """
    rx, ry = robot_pos
    for px, py in (person_a_pos, person_b_pos):
        if not _is_visible((px, py)):
            continue
        dx = rx - px
        dy = ry - py
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < min_dist and dist > 0.01:
            overlap = (min_dist - dist) / min_dist
            nudge = overlap * 1.5
            rx += (dx / dist) * nudge
            ry += (dy / dist) * nudge
    # Keep within scene bounds
    rx = max(XMIN + 0.5, min(XMAX - 0.5, rx))
    ry = max(YMIN + 0.5, min(YMAX - 0.5, ry))
    return (rx, ry)


def _compute_reactive_corridor_pos(situation, person_a_pos, person_b_pos):
    """Compute robot position in corridor that reacts to pedestrian positions.

    Uses a proximity-weighted repulsion model: each visible person exerts
    a lateral force that pushes the robot away.  Closer people (in y) exert
    stronger force, so staggered encounters produce a natural zigzag as the
    robot avoids each person in turn.

    The norm-compliant path emerges from the spatial situation.
    """
    center_x = (CORRIDOR_LEFT_WALL + CORRIDOR_RIGHT_WALL) / 2  # 5.0
    margin = 0.7  # min clearance from wall
    left_edge = CORRIDOR_LEFT_WALL + margin
    right_edge = CORRIDOR_RIGHT_WALL - margin

    # Y-progress along corridor is fixed per situation
    SIT_Y = {
        "open_area": 1.5, "corridor_encounter": 1.5,
        "scene_assessed": 3.0, "in_queue": 5.0,
        "ready_for_service": 6.5, "at_counter": 8.0,
        "interaction": 8.5,
    }
    y = SIT_Y.get(situation, 5.0)

    # For situations where the robot must share space, compute reactive x
    if situation in ("in_queue", "ready_for_service", "scene_assessed"):
        # Proximity-weighted repulsion: each person pushes the robot away,
        # with strength proportional to how close they are in y.
        force = 0.0
        any_visible = False
        for person_pos in (person_a_pos, person_b_pos):
            if _is_visible(person_pos):
                any_visible = True
                dy = abs(y - person_pos[1])
                # Sharp proximity decay — nearby people dominate
                proximity = 1.0 / (dy / 2.0 + 0.5) ** 2
                # Push away from person: if person is right of center,
                # force is negative (go left) and vice versa
                force += (center_x - person_pos[0]) * proximity

        if any_visible:
            x = center_x + force * 2.0
            x = max(left_edge, min(right_edge, x))
        else:
            x = center_x

        # Blend toward center as the encounter resolves
        if situation == "ready_for_service":
            x = (x + center_x) / 2
        if situation == "scene_assessed":
            x = center_x + (x - center_x) * 0.3
    else:
        # open_area, at_counter, interaction — no yielding needed
        x = center_x

    return (x, y)


# ================================================================
# Gaze derivation from primitive skill template
# ================================================================
def get_gaze_from_primitive(lib, prim_name, robot_pos, scene_type):
    prim = lib.get(prim_name)
    if prim is None:
        return None, None
    skill = prim.skill_template.skill if prim.skill_template else ""
    params = prim.skill_template.params if prim.skill_template else {}
    mode = params.get("mode", "")
    intent = params.get("intent", "")
    rx, ry = robot_pos

    # In corridor, gaze targets are different — no desk
    if scene_type in ("corridor", "corridor_both"):
        if skill == "gaze":
            if mode == "scan_area" or "scan" in prim_name:
                return None, "scan"
            elif "avert" in prim_name:
                return (rx - 2.0, ry - 1.0), "avert"
            else:
                return (rx, ry + 3.0), "acknowledge"
        elif skill == "navigate":
            if intent == "wait":
                return (rx + 1.0, ry), "acknowledge"
            elif intent == "approach":
                return (5.0, CORRIDOR_DOOR_Y), "focus"
            else:
                return (rx, ry + 2.0), "acknowledge"
        if "yield" in prim_name:
            return None, None
        if "wait" in prim_name:
            return (rx, ry + 2.0), "acknowledge"
        return None, None

    # Reception variants
    if skill == "gaze":
        if mode == "scan_area" or "scan" in prim_name:
            return None, "scan"
        elif mode == "look_at_staff" or "at-agent" in prim_name:
            return (DESK_X + DESK_W / 2, DESK_Y), "focus"
        elif "avert" in prim_name:
            return (rx - 2.0, ry - 1.0), "avert"
        else:
            return (DESK_X, DESK_Y), "acknowledge"
    elif skill == "navigate":
        if intent == "queue":
            return PERSON_IN_QUEUE, "acknowledge"
        elif intent == "wait":
            return PERSON_IN_QUEUE, "acknowledge"
        elif intent == "approach":
            return (DESK_X + DESK_W / 2, DESK_Y), "focus"
        else:
            return None, None
    if "yield" in prim_name:
        return None, None
    if "wait" in prim_name:
        return (DESK_X, DESK_Y), "acknowledge"
    return None, None


# ================================================================
# Drawing: Scene environment elements (per scene_type)
# ================================================================
def _draw_env_reception_full(ax):
    """Full reception: wall + desk + queue stanchions + arrows."""
    # Wall behind desk
    ax.fill_between([XMIN, XMAX], [YMAX, YMAX], [DESK_Y + 0.8, DESK_Y + 0.8],
                    color=COLOR_WALL, zorder=1)
    ax.plot([XMIN, XMAX], [DESK_Y + 0.8, DESK_Y + 0.8], color="#7F8C8D",
            linewidth=2, zorder=2)
    # Desk
    desk = mpatches.FancyBboxPatch(
        (DESK_X - DESK_W / 2, DESK_Y - DESK_H / 2), DESK_W, DESK_H,
        boxstyle="round,pad=0.1", facecolor=COLOR_DESK, edgecolor="#2C3E50",
        linewidth=2, zorder=5)
    ax.add_patch(desk)
    ax.text(DESK_X + DESK_W / 2, DESK_Y, "RECEPTION", fontsize=6,
            fontweight="bold", ha="center", va="center", color="white", zorder=6)
    # Queue stanchions with arrows
    for i in range(4):
        qy = 2.0 + i * 1.5
        ax.plot(4.0, qy, "s", color="#BDC3C7", markersize=6, zorder=2)
        if i < 3:
            ax.annotate("", xy=(4.0, qy + 1.2), xytext=(4.0, qy + 0.3),
                        arrowprops=dict(arrowstyle="->", color="#BDC3C7", lw=1),
                        zorder=2)
    ax.text(4.0, 1.2, "queue", fontsize=6, ha="center", color="#95A5A6", zorder=2)
    # Entrance
    ax.text(1.0, -0.1, "ENTRANCE", fontsize=6, ha="center", color="#95A5A6")
    ax.plot([0.0, 2.0], [-0.2, -0.2], linewidth=3, color="#27AE60", zorder=2)


def _draw_env_corridor(ax):
    """Corridor: two parallel walls forming a hallway, door at far end."""
    ax.set_facecolor(COLOR_CORRIDOR_FLOOR)
    # Left wall
    ax.fill_between([XMIN, CORRIDOR_LEFT_WALL], [YMAX, YMAX], [YMIN, YMIN],
                    color="#C4B8A8", zorder=1)
    ax.plot([CORRIDOR_LEFT_WALL, CORRIDOR_LEFT_WALL], [YMIN, YMAX],
            color="#8B7D6B", linewidth=3, zorder=2)
    # Right wall
    ax.fill_between([CORRIDOR_RIGHT_WALL, XMAX], [YMAX, YMAX], [YMIN, YMIN],
                    color="#C4B8A8", zorder=1)
    ax.plot([CORRIDOR_RIGHT_WALL, CORRIDOR_RIGHT_WALL], [YMIN, YMAX],
            color="#8B7D6B", linewidth=3, zorder=2)
    # Floor tiles (dashed lines along corridor)
    for y in range(0, 11, 2):
        ax.plot([CORRIDOR_LEFT_WALL + 0.3, CORRIDOR_RIGHT_WALL - 0.3],
                [y, y], color="#D5CFC5", linewidth=0.5, linestyle=":",
                zorder=1, alpha=0.5)
    # Center line
    ax.plot([5.0, 5.0], [YMIN, YMAX], color="#D5CFC5", linewidth=1,
            linestyle="--", zorder=1, alpha=0.3)
    # Door at far end
    door_w = 2.0
    door = mpatches.FancyBboxPatch(
        (5.0 - door_w / 2, CORRIDOR_DOOR_Y - 0.2), door_w, 0.4,
        boxstyle="round,pad=0.05", facecolor="#8B6914", edgecolor="#5C4A0E",
        linewidth=2, zorder=5)
    ax.add_patch(door)
    ax.text(5.0, CORRIDOR_DOOR_Y, "EXIT", fontsize=6, fontweight="bold",
            ha="center", va="center", color="white", zorder=6)
    # Corridor label
    ax.text(5.0, YMAX - 0.3, "CORRIDOR", fontsize=7, fontweight="bold",
            ha="center", va="top", color="#8B7D6B", zorder=6)
    # Entrance at bottom
    ax.text(5.0, 0.2, "ENTRANCE", fontsize=6, ha="center", color="#95A5A6")
    ax.plot([4.0, 6.0], [YMIN + 0.1, YMIN + 0.1], linewidth=3,
            color="#27AE60", zorder=2)


def _draw_env_reception_no_queue(ax):
    """Degraded reception: desk + wall, but queue stanchions removed.
    Ghost outlines show where they used to be."""
    # Wall
    ax.fill_between([XMIN, XMAX], [YMAX, YMAX], [DESK_Y + 0.8, DESK_Y + 0.8],
                    color=COLOR_WALL, zorder=1)
    ax.plot([XMIN, XMAX], [DESK_Y + 0.8, DESK_Y + 0.8], color="#7F8C8D",
            linewidth=2, zorder=2)
    # Desk
    desk = mpatches.FancyBboxPatch(
        (DESK_X - DESK_W / 2, DESK_Y - DESK_H / 2), DESK_W, DESK_H,
        boxstyle="round,pad=0.1", facecolor=COLOR_DESK, edgecolor="#2C3E50",
        linewidth=2, zorder=5)
    ax.add_patch(desk)
    ax.text(DESK_X + DESK_W / 2, DESK_Y, "RECEPTION", fontsize=6,
            fontweight="bold", ha="center", va="center", color="white", zorder=6)
    # Ghost queue stanchions (dashed outlines where they were)
    for i in range(4):
        qy = 2.0 + i * 1.5
        ax.plot(4.0, qy, "s", color="#E0E0E0", markersize=6, zorder=2,
                markeredgecolor="#CCCCCC", markeredgewidth=1,
                markerfacecolor="none", linestyle="--")
        # Dashed ghost squares
        ghost = mpatches.FancyBboxPatch(
            (3.85, qy - 0.15), 0.3, 0.3,
            boxstyle="round,pad=0.02", facecolor="none",
            edgecolor="#CCCCCC", linewidth=1, linestyle="dashed",
            zorder=2)
        ax.add_patch(ghost)
    ax.text(4.0, 1.2, "queue removed", fontsize=5, ha="center",
            color="#CCCCCC", style="italic", zorder=2)
    # Entrance
    ax.text(1.0, -0.1, "ENTRANCE", fontsize=6, ha="center", color="#95A5A6")
    ax.plot([0.0, 2.0], [-0.2, -0.2], linewidth=3, color="#27AE60", zorder=2)


def _draw_env_reception_sign(ax):
    """Enriched reception: full reception + prominent directional sign +
    floor affordance path showing the direct route the sign creates."""
    # Draw full reception first
    _draw_env_reception_full(ax)

    # --- Affordance path on floor: dashed green line from entrance to desk ---
    # This is the material cue — the sign creates a visible direct route
    path_xs = [1.5, 3.0, 5.5, DESK_X + DESK_W / 2]
    path_ys = [0.5, 3.0, 6.0, DESK_Y - 0.5]
    ax.plot(path_xs, path_ys, color=COLOR_SIGN, linewidth=2.5,
            linestyle=(0, (5, 3)), alpha=0.35, zorder=3)
    # Floor arrows along the path
    for i in range(len(path_xs) - 1):
        mx = (path_xs[i] + path_xs[i + 1]) / 2
        my = (path_ys[i] + path_ys[i + 1]) / 2
        ax.annotate("", xy=(path_xs[i + 1], path_ys[i + 1]),
                    xytext=(path_xs[i], path_ys[i]),
                    arrowprops=dict(arrowstyle="-|>", color=COLOR_SIGN,
                                    lw=1.5, alpha=0.25),
                    zorder=3)

    # --- Large sign on the wall ---
    sign_x, sign_y = 8.0, 7.2
    sign = mpatches.FancyBboxPatch(
        (sign_x - 1.4, sign_y - 0.6), 2.8, 1.2,
        boxstyle="round,pad=0.12", facecolor=COLOR_SIGN,
        edgecolor="#1A8A3E", linewidth=2.5, zorder=8)
    ax.add_patch(sign)
    ax.text(sign_x, sign_y + 0.15, "PLEASE COME", fontsize=6,
            fontweight="bold", ha="center", va="center", color="white", zorder=9)
    ax.text(sign_x, sign_y - 0.15, "DIRECTLY TO", fontsize=6,
            fontweight="bold", ha="center", va="center", color="white", zorder=9)
    ax.text(sign_x, sign_y - 0.42, "COUNTER", fontsize=6,
            fontweight="bold", ha="center", va="center", color="white", zorder=9)
    # Arrow from sign pointing to desk
    ax.annotate("", xy=(DESK_X + DESK_W / 2 + 0.2, DESK_Y - 0.3),
                xytext=(sign_x - 0.8, sign_y + 0.7),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_SIGN,
                                lw=3, alpha=0.7),
                zorder=7)


def _draw_env_reception_sign_direct(ax):
    """Sign-direct reception: desk + sign on LEFT side + affordance path
    leading to the left end of the desk.  No queue stanchions."""
    # Wall behind desk
    ax.fill_between([XMIN, XMAX], [YMAX, YMAX], [DESK_Y + 0.8, DESK_Y + 0.8],
                    color=COLOR_WALL, zorder=1)
    ax.plot([XMIN, XMAX], [DESK_Y + 0.8, DESK_Y + 0.8], color="#7F8C8D",
            linewidth=2, zorder=2)
    # Desk
    desk = mpatches.FancyBboxPatch(
        (DESK_X - DESK_W / 2, DESK_Y - DESK_H / 2), DESK_W, DESK_H,
        boxstyle="round,pad=0.1", facecolor=COLOR_DESK, edgecolor="#2C3E50",
        linewidth=2, zorder=5)
    ax.add_patch(desk)
    ax.text(DESK_X + DESK_W / 2, DESK_Y, "RECEPTION", fontsize=6,
            fontweight="bold", ha="center", va="center", color="white", zorder=6)
    # Entrance
    ax.text(1.0, -0.1, "ENTRANCE", fontsize=6, ha="center", color="#95A5A6")
    ax.plot([0.0, 2.0], [-0.2, -0.2], linewidth=3, color="#27AE60", zorder=2)
    # Affordance path on floor — direct route up the LEFT side to left end of desk
    left_desk_x = DESK_X - DESK_W / 2  # 3.5
    path_xs = [1.5, 1.8, 2.2, 3.0, left_desk_x]
    path_ys = [0.5, 2.5, 4.5, 6.5, DESK_Y - 0.3]
    ax.plot(path_xs, path_ys, color=COLOR_SIGN, linewidth=2.5,
            linestyle=(0, (5, 3)), alpha=0.35, zorder=3)
    for i in range(len(path_xs) - 1):
        ax.annotate("", xy=(path_xs[i + 1], path_ys[i + 1]),
                    xytext=(path_xs[i], path_ys[i]),
                    arrowprops=dict(arrowstyle="-|>", color=COLOR_SIGN,
                                    lw=1.5, alpha=0.25),
                    zorder=3)
    # Large sign on the LEFT wall
    sign_x, sign_y = 1.8, 7.2
    sign = mpatches.FancyBboxPatch(
        (sign_x - 1.4, sign_y - 0.6), 2.8, 1.2,
        boxstyle="round,pad=0.12", facecolor=COLOR_SIGN,
        edgecolor="#1A8A3E", linewidth=2.5, zorder=8)
    ax.add_patch(sign)
    ax.text(sign_x, sign_y + 0.15, "PLEASE COME", fontsize=6,
            fontweight="bold", ha="center", va="center", color="white", zorder=9)
    ax.text(sign_x, sign_y - 0.15, "DIRECTLY TO", fontsize=6,
            fontweight="bold", ha="center", va="center", color="white", zorder=9)
    ax.text(sign_x, sign_y - 0.42, "COUNTER", fontsize=6,
            fontweight="bold", ha="center", va="center", color="white", zorder=9)
    # Arrow from sign pointing to left end of desk
    ax.annotate("", xy=(left_desk_x, DESK_Y - 0.3),
                xytext=(sign_x + 0.5, sign_y + 0.7),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_SIGN,
                                lw=3, alpha=0.7),
                zorder=7)


DRAW_ENV = {
    "reception_full": _draw_env_reception_full,
    "corridor": _draw_env_corridor,
    "corridor_both": _draw_env_corridor,   # same hallway, different people
    "reception_no_queue": _draw_env_reception_no_queue,
    "reception_sign": _draw_env_reception_sign,
    "reception_sign_direct": _draw_env_reception_sign_direct,
}


# ================================================================
# Drawing: Scene panel (rich version with gaze)
# ================================================================
def draw_scene(ax, robot_pos, pre_sit, post_sit, t_frac, prim_name,
               trail_x, trail_y, lib, step_idx, n_steps, current_sit,
               scene_type):
    ax.clear()
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(COLOR_FLOOR)

    if prim_name:
        ax.set_title(f"Scene  |  {prim_name}  [{current_sit}]",
                     fontsize=8, fontweight="bold", pad=2)
    else:
        ax.set_title(f"Scene  |  [{current_sit}]", fontsize=8, pad=2)

    # Draw environment (per scene_type)
    draw_env_fn = DRAW_ENV.get(scene_type, _draw_env_reception_full)
    draw_env_fn(ax)

    # People
    a_pos, b_pos = get_person_positions(pre_sit, post_sit, t_frac, scene_type)

    if scene_type in ("corridor", "corridor_both"):
        a_label, b_label = "P1", "P2"
    else:
        a_label, b_label = "A", "B"

    if _is_visible(a_pos):
        edge_dist = min(XMAX - a_pos[0], a_pos[0] - XMIN,
                        YMAX - a_pos[1], a_pos[1] - YMIN) / 2.0
        alpha = min(1.0, max(0.1, edge_dist))
        c = plt.Circle(a_pos, 0.3, facecolor=COLOR_PERSON, edgecolor="#2C3E50",
                        linewidth=1.5, zorder=10, alpha=alpha)
        ax.add_patch(c)
        ax.text(a_pos[0], a_pos[1] - 0.45, a_label, fontsize=6, ha="center",
                color=COLOR_PERSON, fontweight="bold", zorder=11, alpha=alpha)
    if _is_visible(b_pos):
        edge_dist = min(XMAX - b_pos[0], b_pos[0] - XMIN,
                        YMAX - b_pos[1], b_pos[1] - YMIN) / 2.0
        alpha = min(1.0, max(0.1, edge_dist))
        c = plt.Circle(b_pos, 0.3, facecolor=COLOR_PERSON2, edgecolor="#2C3E50",
                        linewidth=1.5, zorder=10, alpha=alpha)
        ax.add_patch(c)
        ax.text(b_pos[0], b_pos[1] - 0.45, b_label, fontsize=6, ha="center",
                color=COLOR_PERSON2, fontweight="bold", zorder=11, alpha=alpha)

    # Trail
    if len(trail_x) > 1:
        ax.plot(trail_x, trail_y, color=COLOR_ROBOT, alpha=0.15,
                linewidth=2, zorder=3)

    # Robot with gaze
    rx, ry = robot_pos
    if len(trail_x) >= 2:
        ddx = trail_x[-1] - trail_x[-2]
        ddy = trail_y[-1] - trail_y[-2]
        if abs(ddx) > 0.01 or abs(ddy) > 0.01:
            heading = np.degrees(np.arctan2(ddy, ddx))
        else:
            heading = 90
    else:
        heading = 90

    gaze_target, gaze_type = None, None
    if prim_name:
        gaze_target, gaze_type = get_gaze_from_primitive(
            lib, prim_name, robot_pos, scene_type)
        if gaze_type == "scan":
            angle = -60 + 120 * t_frac
            gaze_dist = 4.0
            gx = rx + gaze_dist * np.cos(np.radians(angle + 60))
            gy = ry + gaze_dist * np.sin(np.radians(angle + 60))
            gaze_target = (gx, gy)

    # Draw gaze
    if gaze_target is not None:
        gx, gy = gaze_target
        if gaze_type == "focus":
            ax.annotate("", xy=(gx, gy), xytext=(rx, ry),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR_GAZE,
                                        lw=2, alpha=0.7), zorder=14)
        elif gaze_type == "scan":
            ax.plot([rx, gx], [ry, gy], color=COLOR_GAZE_SCAN, linewidth=1,
                    alpha=0.5, linestyle="--", zorder=14)
            angle_to = np.degrees(np.arctan2(gy - ry, gx - rx))
            cone = mpatches.Wedge((rx, ry), 2.5, angle_to - 15, angle_to + 15,
                                  facecolor=COLOR_GAZE_SCAN, alpha=0.1, zorder=13)
            ax.add_patch(cone)
        elif gaze_type == "acknowledge":
            ax.plot([rx, gx], [ry, gy], color=COLOR_GAZE, linewidth=1,
                    alpha=0.4, linestyle=":", zorder=14)
        elif gaze_type == "avert":
            ax.plot([rx, gx], [ry, gy], color="#95A5A6", linewidth=1,
                    alpha=0.3, linestyle=":", zorder=14)

    # Robot body
    body = plt.Circle((rx, ry), 0.35, facecolor=COLOR_ROBOT_BODY,
                       edgecolor="#1A5276", linewidth=2, zorder=15)
    ax.add_patch(body)
    dx = 0.35 * np.cos(np.radians(heading))
    dy = 0.35 * np.sin(np.radians(heading))
    ax.plot([rx, rx + dx], [ry, ry + dy], color="#1A5276", linewidth=2.5, zorder=16)
    ax.text(rx, ry, "R", fontsize=8, fontweight="bold", ha="center",
            va="center", color="white", zorder=17)


# ================================================================
# Drawing: Situation FSM (animated highlighting)
# ================================================================
def draw_situation_fsm(ax, lib, situation_chain, topology, sequence,
                       current_step, t_frac):
    ax.clear()
    ax.axis("off")
    ax.set_title("Situation State Machine  (G = -log B-matrix)",
                 fontsize=8, fontweight="bold", pad=2)

    if not situation_chain:
        ax.text(0.5, 0.5, "(no sequence)", fontsize=8, ha="center",
                va="center", transform=ax.transAxes, color="#999")
        return

    all_sits = []
    seen = set()
    for _, pre, post, _ in situation_chain:
        for s in (pre, post):
            if s not in seen:
                all_sits.append(s)
                seen.add(s)

    G = nx.DiGraph()
    for s in all_sits:
        G.add_node(s)

    backbone_edges = {}
    for idx, (prim_name, pre, post, _) in enumerate(situation_chain):
        if pre != post:
            g_val = compute_transition_fe(topology, sequence[idx - 1], prim_name) if idx > 0 else 0.0
            G.add_edge(pre, post, label=prim_name, fe=g_val,
                       backbone=True, step=idx)
            backbone_edges[(pre, post)] = (prim_name, g_val, idx)

    sit_list = list(dict.fromkeys(
        s for _, _, s, _ in situation_chain if s != situation_chain[0][1]))
    if situation_chain:
        sit_list.insert(0, situation_chain[0][1])

    for i in range(len(sit_list)):
        for j in range(i + 2, min(i + 4, len(sit_list))):
            a, b = sit_list[i], sit_list[j]
            if (a, b) not in backbone_edges and a != b:
                best_w = 0.0
                for (pa, pb), w in topology.items():
                    p_a = lib.get(pa)
                    p_b = lib.get(pb)
                    if p_a and p_b:
                        a_match = (p_a.postcondition_situation == a or
                                   a in p_a.precondition_situations)
                        b_match = (p_b.postcondition_situation == b or
                                   b in p_b.precondition_situations)
                        if a_match and b_match and w > best_w:
                            best_w = w
                skip_fe = -math.log(best_w + 0.01)
                G.add_edge(a, b, label="skip", fe=skip_fe, backbone=False,
                           step=-1)

    n = len(all_sits)
    pos = {}
    for i, s in enumerate(all_sits):
        angle = math.pi * 0.8 - i * (math.pi * 1.2 / max(n - 1, 1))
        pos[s] = (math.cos(angle), math.sin(angle))

    if current_step < 0:
        current_sit = situation_chain[0][1] if situation_chain else ""
    elif current_step < len(situation_chain):
        _, pre, post, _ = situation_chain[current_step]
        current_sit = pre if t_frac < 0.5 else post
    else:
        current_sit = situation_chain[-1][2] if situation_chain else ""

    shortcut_edges = [(a, b) for a, b in G.edges()
                      if not G[a][b].get("backbone")]
    if shortcut_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=shortcut_edges,
                               edge_color="#E74C3C", style="dashed",
                               width=1, alpha=0.3, arrows=True,
                               arrowsize=8, connectionstyle="arc3,rad=0.25")
        for a, b in shortcut_edges:
            mx = (pos[a][0] + pos[b][0]) / 2
            my = (pos[a][1] + pos[b][1]) / 2 + 0.08
            fe = G[a][b]["fe"]
            ax.text(mx, my, f"G={fe:.1f}", fontsize=5, color="#E74C3C",
                    ha="center", alpha=0.5, style="italic")

    for a, b in [(a, b) for a, b in G.edges() if G[a][b].get("backbone")]:
        data = G[a][b]
        edge_step = data["step"]
        if edge_step < current_step:
            color, width, alpha = "#27AE60", 2.5, 0.9
        elif edge_step == current_step:
            color, width = "#F39C12", 3.0
            alpha = 0.6 + 0.4 * t_frac
        else:
            color, width, alpha = "#BDC3C7", 1.5, 0.4
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(a, b)],
                               edge_color=color, width=width, alpha=alpha,
                               arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.1")
        mx = (pos[a][0] + pos[b][0]) / 2
        my = (pos[a][1] + pos[b][1]) / 2 - 0.1
        fe = data["fe"]
        ax.text(mx, my, f"{data['label']}\nG={fe:.2f}", fontsize=5,
                ha="center", color=color, fontweight="bold")

    for s in all_sits:
        x, y = pos[s]
        if s == current_sit:
            nc, ns, lw = "#F39C12", 0.14, 3
        else:
            si = all_sits.index(s)
            ci = all_sits.index(current_sit) if current_sit in all_sits else 0
            nc = "#27AE60" if si < ci else "#D5DBDB"
            ns, lw = 0.11, 1.5
        circle = plt.Circle((x, y), ns, facecolor=nc,
                            edgecolor="#2C3E50", linewidth=lw, zorder=10)
        ax.add_patch(circle)
        short = s.replace("_", "\n")
        ax.text(x, y - ns - 0.06, short, fontsize=5, ha="center",
                va="top", fontweight="bold", color="#2C3E50")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)


# ================================================================
# Drawing: Fragment Composition panel
# ================================================================
def draw_fragments(ax, fragments, composite, frag_colors, weighted):
    ax.clear()
    ax.axis("off")
    ax.set_title("Fragment Composition", fontsize=8, fontweight="bold", pad=2)

    n_frags = len(fragments)
    row_h = 0.12
    start_y = 0.92

    weight_map = {wp.pattern.name: wp.weight for wp in weighted}

    for i, frag in enumerate(fragments):
        y = start_y - i * (row_h + 0.04)
        color = frag_colors.get(frag.name, "#CCCCCC")
        w = weight_map.get(frag.name, 0.0)

        ax.text(0.02, y, f"{frag.name}", fontsize=5, fontweight="bold",
                color=color, transform=ax.transAxes, va="center")
        ax.text(0.35, y, f"w={w:.3f}", fontsize=4, color="#7F8C8D",
                transform=ax.transAxes, va="center")

        prims = sorted(frag.primitive_cluster)
        block_x = 0.45
        for p in prims:
            in_composite = p in composite.primitive_cluster
            alpha = 1.0 if in_composite else 0.3
            rect = mpatches.FancyBboxPatch(
                (block_x, y - 0.025), 0.18, 0.05,
                boxstyle="round,pad=0.01", facecolor=color, alpha=alpha,
                edgecolor="#2C3E50" if in_composite else "none",
                linewidth=0.5, transform=ax.transAxes)
            ax.add_patch(rect)
            short_p = p[:12] if len(p) > 12 else p
            ax.text(block_x + 0.09, y, short_p, fontsize=3.5, ha="center",
                    va="center", color="white" if in_composite else "#AAAAAA",
                    transform=ax.transAxes)
            block_x += 0.20

    y = start_y - n_frags * (row_h + 0.04) - 0.06
    ax.plot([0.02, 0.95], [y + 0.04, y + 0.04], color="#2C3E50",
            linewidth=0.5, transform=ax.transAxes, alpha=0.3)
    ax.text(0.02, y, "COMPOSITE", fontsize=5, fontweight="bold",
            color="#2C3E50", transform=ax.transAxes, va="center")
    ax.text(0.35, y, f"{len(composite.primitive_cluster)} primitives",
            fontsize=4, color="#7F8C8D", transform=ax.transAxes, va="center")
    ax.text(0.02, y - 0.05,
            f"from {len(composite.source_fragments)} fragments",
            fontsize=4, color="#95A5A6", transform=ax.transAxes,
            va="center", style="italic")


# ================================================================
# Drawing: Execution Sequence panel
# ================================================================
def draw_execution(ax, sequence, lib, current_step, t_frac, frag_colors,
                   prim_to_frag):
    ax.clear()
    ax.axis("off")
    ax.set_title("Execution Sequence", fontsize=8, fontweight="bold", pad=2)

    n = len(sequence)
    for i, prim_name in enumerate(sequence):
        y = 0.95 - i * (0.85 / max(n, 1))
        prim = lib.get(prim_name)
        post = prim.postcondition_situation if prim else "?"

        if i < current_step:
            color = "#27AE60"
            marker = "\u2713"
            alpha = 1.0
        elif i == current_step:
            color = "#F39C12"
            marker = "\u25B6"
            alpha = 0.6 + 0.4 * t_frac
        else:
            color = "#BDC3C7"
            marker = str(i + 1)
            alpha = 0.4

        ax.text(0.04, y, marker, fontsize=7, fontweight="bold", color=color,
                transform=ax.transAxes, va="center", ha="center")
        ax.text(0.10, y, prim_name, fontsize=5, fontweight="bold",
                color=color, transform=ax.transAxes, va="center",
                alpha=alpha)
        ax.text(0.62, y, f"\u2192 {post}", fontsize=4, color="#7F8C8D",
                transform=ax.transAxes, va="center", alpha=alpha)

        frags = prim_to_frag.get(prim_name, [])
        if frags:
            fc = frag_colors.get(frags[0], "#CCCCCC") if len(frags) == 1 else "#FFD700"
            ax.plot(0.92, y, "s", color=fc, markersize=4,
                    transform=ax.transAxes, alpha=alpha)


# ================================================================
# Build animation for one variant
# ================================================================
def make_variant_animation(config: VariantConfig, out_dir: str) -> str:
    print(f"  Running pipeline for {config.name}...")
    result = run_pipeline(config)

    sequence = result.sequence
    chain = result.situation_chain
    lib = result.library
    composite = result.composite
    fragments = result.fragments
    weighted = result.weighted
    topology = result.topology
    prim_to_frag = result.prim_to_frag
    frag_colors = get_frag_colors(fragments)
    scene_type = config.scene_type
    sit_pos = get_situation_pos(scene_type)

    n_prims = len(sequence)

    FRAMES_PER_STEP = 20
    INTRO_FRAMES = 12
    OUTRO_FRAMES = 15
    TOTAL_FRAMES = INTRO_FRAMES + max(n_prims, 1) * FRAMES_PER_STEP + OUTRO_FRAMES

    fig = plt.figure(figsize=(16, 10))
    ax_scene  = fig.add_axes([0.02, 0.20, 0.38, 0.68])
    ax_fsm    = fig.add_axes([0.42, 0.48, 0.56, 0.42])
    ax_frag   = fig.add_axes([0.42, 0.20, 0.28, 0.26])
    ax_exec   = fig.add_axes([0.72, 0.20, 0.26, 0.26])
    ax_status = fig.add_axes([0.02, 0.02, 0.96, 0.16])

    fig.text(0.5, 0.96, config.name,
             fontsize=14, fontweight="bold", ha="center", color="#2C3E50")
    fig.text(0.5, 0.93, config.annotation,
             fontsize=10, ha="center", color="#8E44AD")
    fig.text(0.5, 0.905,
             f"context={config.context}  |  "
             f"{len(fragments)} fragments  |  "
             f"{len(composite.primitive_cluster)} primitives  |  "
             f"{n_prims} steps  |  "
             f"scene={scene_type}",
             fontsize=8, ha="center", color="#7F8C8D")

    trail_x, trail_y = [], []

    def update(frame):
        if frame < INTRO_FRAMES:
            step_idx = -1
            t_frac = frame / INTRO_FRAMES
        elif frame >= INTRO_FRAMES + n_prims * FRAMES_PER_STEP:
            step_idx = n_prims
            t_frac = 1.0
        else:
            f = frame - INTRO_FRAMES
            step_idx = f // FRAMES_PER_STEP
            t_frac = (f % FRAMES_PER_STEP) / FRAMES_PER_STEP

        if step_idx < 0:
            pre_sit = chain[0][1] if chain else "open_area"
            post_sit = pre_sit
            current_sit = pre_sit
            prim_name = ""
            start = sit_pos.get(current_sit, (2.0, 2.0))
            robot_pos = (start[0], start[1] * t_frac)
        elif not chain or step_idx >= n_prims:
            pre_sit = chain[-1][2] if chain else "open_area"
            post_sit = pre_sit
            current_sit = pre_sit
            prim_name = ""
            robot_pos = sit_pos.get(current_sit, (5.0, 7.0))
        else:
            prim_name, pre_sit, post_sit, _ = chain[step_idx]
            current_sit = pre_sit if t_frac < 0.5 else post_sit

            if scene_type in ("corridor", "corridor_both"):
                # Reactive positioning: robot yields based on where people are
                a_pos, b_pos = get_person_positions(
                    pre_sit, post_sit, t_frac, scene_type)
                p1 = _compute_reactive_corridor_pos(pre_sit, a_pos, b_pos)
                p2 = _compute_reactive_corridor_pos(post_sit, a_pos, b_pos)
            else:
                p1 = sit_pos.get(pre_sit, (2.0, 2.0))
                p2 = sit_pos.get(post_sit, (2.0, 2.0))

            robot_pos = (
                p1[0] + (p2[0] - p1[0]) * t_frac,
                p1[1] + (p2[1] - p1[1]) * t_frac,
            )

        # Universal spatial norm: avoid people regardless of scene type
        if step_idx >= 0 and step_idx < n_prims:
            a_chk, b_chk = get_person_positions(
                pre_sit, post_sit, t_frac, scene_type)
            robot_pos = _apply_person_avoidance(robot_pos, a_chk, b_chk)

        trail_x.append(robot_pos[0])
        trail_y.append(robot_pos[1])

        draw_scene(ax_scene, robot_pos, pre_sit, post_sit, t_frac,
                   prim_name, trail_x, trail_y, lib, step_idx, n_prims,
                   current_sit, scene_type)

        draw_situation_fsm(ax_fsm, lib, chain, topology, sequence,
                           step_idx, t_frac)

        draw_fragments(ax_frag, fragments, composite, frag_colors, weighted)

        draw_execution(ax_exec, sequence, lib, step_idx, t_frac,
                       frag_colors, prim_to_frag)

        # Status bar
        ax_status.clear()
        ax_status.set_xlim(0, 1)
        ax_status.set_ylim(0, 1)
        ax_status.axis("off")

        if step_idx < 0:
            ax_status.text(0.5, 0.6, "Robot entering scene...",
                           fontsize=11, ha="center", color="#7F8C8D")
        elif step_idx >= n_prims:
            ax_status.text(0.5, 0.6,
                           f"Script complete -- {config.name}",
                           fontsize=10, ha="center", color="#27AE60",
                           fontweight="bold")
            ax_status.text(0.5, 0.25,
                           f"Composed from {len(fragments)} fragments via "
                           f"causal backbone extraction",
                           fontsize=8, ha="center", color="#7F8C8D")
        else:
            pn = sequence[step_idx]
            src_frags = prim_to_frag.get(pn, ["?"])
            fe = compute_step_fe(topology, sequence, step_idx)

            ax_status.text(0.02, 0.7,
                           f"Step {step_idx+1}/{n_prims}:  {pn}",
                           fontsize=10, fontweight="bold", color="#2C3E50")
            ax_status.text(0.02, 0.35,
                           f"Situation: {chain[step_idx][1]} \u2192 "
                           f"{chain[step_idx][2]}   |   "
                           f"G = {fe:.2f}   |   "
                           f"from: {', '.join(src_frags)}",
                           fontsize=8, color="#7F8C8D")

        if n_prims > 0:
            dot_spacing = min(0.09, 0.9 / n_prims)
            total_w = n_prims * dot_spacing
            start_x = 0.5 - total_w / 2
            for i in range(n_prims):
                px = start_x + i * dot_spacing + 0.03
                if i < step_idx:
                    ax_status.plot(px, 0.08, "o", color="#27AE60",
                                   markersize=7)
                elif i == step_idx:
                    ax_status.plot(px, 0.08, "o", color="#F39C12",
                                   markersize=9, markeredgecolor="#2C3E50",
                                   markeredgewidth=1.5)
                else:
                    ax_status.plot(px, 0.08, "o", color="#D5DBDB",
                                   markersize=6)

        return []

    anim = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                                   interval=100, blit=False)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{config.file_tag}.gif")
    print(f"  Rendering {TOTAL_FRAMES} frames for {config.name}...")
    anim.save(out_path, writer="pillow", fps=10, dpi=90)
    print(f"  Saved: {out_path}")
    plt.close(fig)
    return out_path


# ================================================================
# Main
# ================================================================
def main():
    out_dir = os.path.join(os.path.dirname(__file__), "demo_figures")
    paths = []
    for config in VARIANTS:
        path = make_variant_animation(config, out_dir)
        paths.append(path)

    print(f"\nGenerated {len(paths)} animation GIFs:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
