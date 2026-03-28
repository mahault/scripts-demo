"""Pipeline-driven animation — single-panel scene view.

Runs the real composition pipeline and derives ALL behaviour from the output.
Robot position comes from situation postconditions.  Gaze type comes from
primitive skill templates.  Person positions come from the current domain
situation.  Nothing is hardcoded per step.

Theoretical basis:
  - Albarracin, Constant, Friston & Ramstead (2021) — A Variational
    Approach to Scripts.
  - Nair, Austin, Watson & Banaei-Kashani (2026) — Thinking Machines:
    A Dual-System Framework for Metacognitive Control and Learning.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
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
# Domain setup (identical to demo_full_pipeline_anim.py / tests)
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


# ================================================================
# Run the real pipeline
# ================================================================
def run_pipeline():
    lib = PrimitiveLibrary()
    _register_reception_primitives(lib)

    fragments = [
        _make_fragment("observe_scene",
                       {"scan-environment", "gaze-scan"},
                       {"reception": 0.9, "corridor": 0.4}),
        _make_fragment("queue_position",
                       {"position-in-queue", "yield-pass"},
                       {"reception": 0.8, "corridor": 0.3}),
        _make_fragment("wait_patiently",
                       {"wait-for-turn", "wait-acknowledge"},
                       {"reception": 0.7, "corridor": 0.5}),
        _make_fragment("approach_service",
                       {"approach-counter", "gaze-at-agent"},
                       {"reception": 0.9, "open_area": 0.4}),
        _make_fragment("courtesy_space",
                       {"yield-pass", "gaze-avert"},
                       {"reception": 0.3, "corridor": 0.8}),
    ]

    cfg = RepertoireConfig()
    rep = ScriptRepertoire(lib, cfg, initial_patterns=fragments)
    rep.enable_compositional_mode()

    query_scores = {}
    for name, pat in rep.patterns.items():
        aff = pat.situation_affinity.get("reception", 0.0)
        query_scores[name] = aff * pat.precision

    weighted = rep.retrieve_composition(query_scores)
    composer = ScriptComposer(lib, cfg)
    composite = composer.compose_from_patterns(weighted, "reception")

    # Domain situation tracking
    DOMAIN_SITUATIONS = {
        "scene_assessed", "in_queue", "ready_for_service",
        "at_counter", "interaction",
    }
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

    # Topology for free energy values
    topo = composite.context_topology.get("reception", {})
    if not topo:
        topo = next(iter(composite.context_topology.values()), {})

    # Primitive to fragment mapping
    prim_to_frag = {}
    for frag in fragments:
        for p in frag.primitive_cluster:
            prim_to_frag.setdefault(p, []).append(frag.name)

    print(f"Composed sequence ({len(sequence)} primitives):")
    for prim_name, pre, post, raw in situation_chain:
        print(f"  {pre:20s} --[{prim_name}]--> {post}  (raw: {raw})")

    return lib, sequence, situation_chain, topo, prim_to_frag, fragments


# ================================================================
# Situation -> scene position mapping (visualisation only)
# ================================================================
SITUATION_POS = {
    "open_area":          (2.0, 2.0),
    "corridor_encounter": (2.0, 2.0),
    "scene_assessed":     (2.5, 3.0),
    "in_queue":           (4.0, 4.0),
    "ready_for_service":  (4.0, 5.5),
    "at_counter":         (5.2, 7.0),
    "interaction":        (5.5, 7.5),
}

# Scene constants
DESK_X, DESK_Y = 5.0, 8.5
DESK_W, DESK_H = 3.0, 0.6
PERSON_AT_DESK = (5.5, 7.5)
PERSON_IN_QUEUE = (4.0, 5.5)
EXIT_RIGHT = (10.5, 7.5)
XMIN, XMAX = -0.5, 10.5
YMIN, YMAX = -0.5, 10.5

# Person target positions per domain situation
PERSON_A_TARGETS = {
    "open_area":         PERSON_AT_DESK,
    "corridor_encounter": PERSON_AT_DESK,
    "scene_assessed":    PERSON_AT_DESK,
    "in_queue":          PERSON_AT_DESK,
    "ready_for_service": EXIT_RIGHT,
    "at_counter":        EXIT_RIGHT,
    "interaction":       EXIT_RIGHT,
}

PERSON_B_TARGETS = {
    "open_area":         PERSON_IN_QUEUE,
    "corridor_encounter": PERSON_IN_QUEUE,
    "scene_assessed":    PERSON_IN_QUEUE,
    "in_queue":          PERSON_IN_QUEUE,
    "ready_for_service": PERSON_AT_DESK,
    "at_counter":        EXIT_RIGHT,
    "interaction":       EXIT_RIGHT,
}

COLOR_ROBOT = "#2E86C1"
COLOR_ROBOT_BODY = "#3498DB"
COLOR_PERSON = "#E67E22"
COLOR_PERSON2 = "#E74C3C"
COLOR_DESK = "#7F8C8D"
COLOR_FLOOR = "#F5F5DC"
COLOR_GAZE = "#F1C40F"
COLOR_GAZE_SCAN = "#27AE60"


# ================================================================
# Derive gaze behaviour from primitive skill template
# ================================================================
def get_gaze_from_primitive(lib, prim_name, robot_pos):
    """Derive gaze target and type from the primitive's skill template.

    Returns (gaze_target, gaze_type) where gaze_type is one of:
      "scan", "focus", "acknowledge", "avert", None
    """
    prim = lib.get(prim_name)
    if prim is None:
        return None, None

    skill = prim.skill_template.skill if prim.skill_template else ""
    params = prim.skill_template.params if prim.skill_template else {}
    mode = params.get("mode", "")
    intent = params.get("intent", "")

    rx, ry = robot_pos

    if skill == "gaze":
        if mode == "scan_area" or "scan" in prim_name:
            # Scanning gaze — sweep the room
            return None, "scan"
        elif mode == "look_at_staff" or "at-agent" in prim_name:
            # Focused gaze at the desk
            return (DESK_X + DESK_W / 2, DESK_Y), "focus"
        elif "avert" in prim_name:
            # Avert gaze — look away
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
    # Default hub primitives
    if "yield" in prim_name:
        return None, None
    if "wait" in prim_name:
        return (DESK_X, DESK_Y), "acknowledge"
    return None, None


# ================================================================
# Smooth person position interpolation
# ================================================================
def _lerp_pos(p1, p2, t):
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)


def get_person_positions(pre_sit, post_sit, t_frac):
    """Return (person_a_pos, person_b_pos) with smooth transitions.

    People lerp between their target positions as the domain situation
    transitions — exactly like the robot does.
    """
    a1 = PERSON_A_TARGETS.get(pre_sit, EXIT_RIGHT)
    a2 = PERSON_A_TARGETS.get(post_sit, EXIT_RIGHT)
    a_pos = _lerp_pos(a1, a2, t_frac)

    b1 = PERSON_B_TARGETS.get(pre_sit, EXIT_RIGHT)
    b2 = PERSON_B_TARGETS.get(post_sit, EXIT_RIGHT)
    b_pos = _lerp_pos(b1, b2, t_frac)

    return a_pos, b_pos


def _is_visible(pos):
    return XMIN < pos[0] < XMAX and YMIN < pos[1] < YMAX


# ================================================================
# Free energy from topology
# ================================================================
def compute_transition_fe(topology, from_prim, to_prim):
    w = topology.get((from_prim, to_prim), 0.0)
    eps = 0.01
    return -math.log(w + eps)


# ================================================================
# Drawing functions
# ================================================================
def draw_scene_static(ax):
    ax.set_facecolor(COLOR_FLOOR)

    # Desk
    desk = mpatches.FancyBboxPatch(
        (DESK_X - DESK_W / 2, DESK_Y - DESK_H / 2), DESK_W, DESK_H,
        boxstyle="round,pad=0.1", facecolor=COLOR_DESK, edgecolor="#2C3E50",
        linewidth=2, zorder=5)
    ax.add_patch(desk)
    ax.text(DESK_X + DESK_W / 2, DESK_Y, "RECEPTION", fontsize=9,
            fontweight="bold", ha="center", va="center", color="white", zorder=6)

    # Wall behind desk
    ax.fill_between([XMIN, XMAX], [YMAX, YMAX], [DESK_Y + 0.8, DESK_Y + 0.8],
                    color="#D5DBDB", zorder=1)
    ax.plot([XMIN, XMAX], [DESK_Y + 0.8, DESK_Y + 0.8], color="#7F8C8D",
            linewidth=2, zorder=2)

    # Queue markers
    for i in range(4):
        qy = 2.0 + i * 1.5
        ax.plot(4.0, qy, "s", color="#BDC3C7", markersize=10, zorder=2)
        if i < 3:
            ax.annotate("", xy=(4.0, qy + 1.2), xytext=(4.0, qy + 0.3),
                        arrowprops=dict(arrowstyle="->", color="#BDC3C7", lw=1.5),
                        zorder=2)

    ax.text(4.0, 1.2, "queue", fontsize=8, ha="center", color="#95A5A6", zorder=2)

    # Entrance
    ax.annotate("ENTRANCE", xy=(1.0, -0.1), fontsize=8, ha="center",
                color="#95A5A6", zorder=2)
    ax.plot([0.0, 2.0], [-0.2, -0.2], linewidth=4, color="#27AE60", zorder=2)


def draw_person(ax, pos, label, color, facing_deg=270):
    x, y = pos
    circle = plt.Circle((x, y), 0.35, facecolor=color, edgecolor="#2C3E50",
                         linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    dx = 0.3 * np.cos(np.radians(facing_deg))
    dy = 0.3 * np.sin(np.radians(facing_deg))
    ax.plot([x, x + dx], [y, y + dy], color="#2C3E50", linewidth=2, zorder=11)
    ax.text(x, y - 0.55, label, fontsize=7, ha="center", color=color,
            fontweight="bold", zorder=11)


def draw_robot(ax, x, y, heading_deg, gaze_target=None, gaze_type=None):
    body = plt.Circle((x, y), 0.4, facecolor=COLOR_ROBOT_BODY,
                       edgecolor="#1A5276", linewidth=2, zorder=15)
    ax.add_patch(body)

    dx = 0.4 * np.cos(np.radians(heading_deg))
    dy = 0.4 * np.sin(np.radians(heading_deg))
    ax.plot([x, x + dx], [y, y + dy], color="#1A5276", linewidth=3, zorder=16)

    ax.text(x, y, "R", fontsize=10, fontweight="bold", ha="center",
            va="center", color="white", zorder=17)

    if gaze_target is not None:
        gx, gy = gaze_target
        if gaze_type == "focus":
            ax.annotate("", xy=(gx, gy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR_GAZE,
                                        lw=2.5, alpha=0.7),
                        zorder=14)
        elif gaze_type == "scan":
            ax.plot([x, gx], [y, gy], color=COLOR_GAZE_SCAN, linewidth=1.5,
                    alpha=0.5, linestyle="--", zorder=14)
            angle_to = np.degrees(np.arctan2(gy - y, gx - x))
            cone = mpatches.Wedge((x, y), 2.5, angle_to - 15, angle_to + 15,
                                  facecolor=COLOR_GAZE_SCAN, alpha=0.1, zorder=13)
            ax.add_patch(cone)
        elif gaze_type == "acknowledge":
            ax.plot([x, gx], [y, gy], color=COLOR_GAZE, linewidth=1.5,
                    alpha=0.4, linestyle=":", zorder=14)
        elif gaze_type == "avert":
            ax.plot([x, gx], [y, gy], color="#95A5A6", linewidth=1.5,
                    alpha=0.3, linestyle=":", zorder=14)


def draw_status_bar(ax_status, step_idx, t_frac, sequence, situation_chain,
                    topology, prim_to_frag):
    ax_status.clear()
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.axis("off")

    n_prims = len(sequence)

    if step_idx < 0:
        ax_status.text(0.5, 0.7, "Robot entering scene...",
                       fontsize=12, ha="center", va="center", fontweight="bold")
        return

    if step_idx >= n_prims:
        ax_status.text(0.5, 0.7, "Script complete!",
                       fontsize=12, ha="center", va="center", fontweight="bold",
                       color="#27AE60")
        ax_status.text(0.5, 0.3,
                       "Composed from 5 weak fragments via causal backbone "
                       "extraction \u2014 norm compliance from EFE minimisation",
                       fontsize=9, ha="center", va="center", color="#7F8C8D")
        return

    prim_name = sequence[step_idx]
    _, pre_sit, post_sit, _ = situation_chain[step_idx]
    sources = prim_to_frag.get(prim_name, ["?"])

    # Free energy for this transition
    fe = 0.0
    if step_idx > 0:
        fe = compute_transition_fe(topology, sequence[step_idx - 1], prim_name)

    ax_status.text(0.02, 0.75,
                   f"Step {step_idx + 1}/{n_prims}:  {prim_name}",
                   fontsize=11, ha="left", va="center", fontweight="bold",
                   color="#2C3E50")
    ax_status.text(0.02, 0.40,
                   f"{pre_sit} \u2192 {post_sit}   |   "
                   f"G = {fe:.2f}   |   from: {', '.join(sources)}",
                   fontsize=9, ha="left", va="center", color="#7F8C8D")

    # Progress dots
    if n_prims > 0:
        dot_spacing = min(0.09, 0.9 / n_prims)
        total_w = n_prims * dot_spacing
        start_x = 0.5 - total_w / 2
        for i in range(n_prims):
            px = start_x + i * dot_spacing + 0.03
            if i < step_idx:
                ax_status.plot(px, 0.08, "o", color="#27AE60", markersize=7)
            elif i == step_idx:
                ax_status.plot(px, 0.08, "o", color="#F39C12", markersize=9,
                               markeredgecolor="#2C3E50", markeredgewidth=1.5)
            else:
                ax_status.plot(px, 0.08, "o", color="#D5DBDB", markersize=6)


# ================================================================
# Main animation
# ================================================================
def make_animation():
    print("Running composition pipeline...")
    lib, sequence, situation_chain, topology, prim_to_frag, fragments = \
        run_pipeline()

    n_prims = len(sequence)

    # Animation timing
    FRAMES_PER_STEP = 30
    INTRO_FRAMES = 20
    OUTRO_FRAMES = 25
    TOTAL_FRAMES = INTRO_FRAMES + n_prims * FRAMES_PER_STEP + OUTRO_FRAMES

    fig = plt.figure(figsize=(10, 10))
    ax_scene = fig.add_axes([0.05, 0.18, 0.9, 0.75])
    ax_scene.set_xlim(XMIN, XMAX)
    ax_scene.set_ylim(YMIN, YMAX)
    ax_scene.set_aspect("equal")
    ax_scene.axis("off")

    ax_status = fig.add_axes([0.05, 0.02, 0.9, 0.14])
    ax_status.axis("off")

    fig.text(0.5, 0.96,
             "Reception Desk: Queue-Joining via Compositional Assembly",
             fontsize=14, fontweight="bold", ha="center")
    fig.text(0.5, 0.93,
             "Ordering from causal backbone \u2014 "
             "norm compliance via EFE minimisation (Albarracin et al. 2021)",
             fontsize=10, ha="center", color="#7F8C8D")

    trail_x, trail_y = [], []

    def update(frame):
        ax_scene.clear()
        ax_scene.set_xlim(XMIN, XMAX)
        ax_scene.set_ylim(YMIN, YMAX)
        ax_scene.set_aspect("equal")
        ax_scene.axis("off")

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

        # --- Robot position from situation chain (lerp between situations) ---
        if step_idx < 0:
            pre_sit = situation_chain[0][1] if situation_chain else "open_area"
            post_sit = pre_sit
            current_sit = pre_sit
            prim_name = ""
            start = SITUATION_POS.get(current_sit, (2.0, 2.0))
            robot_pos = (start[0], start[1] * t_frac)
        elif step_idx >= n_prims:
            pre_sit = situation_chain[-1][2] if situation_chain else "open_area"
            post_sit = pre_sit
            current_sit = pre_sit
            prim_name = ""
            robot_pos = SITUATION_POS.get(current_sit, (5.0, 7.0))
        else:
            prim_name, pre_sit, post_sit, _ = situation_chain[step_idx]
            current_sit = pre_sit if t_frac < 0.5 else post_sit
            p1 = SITUATION_POS.get(pre_sit, (2.0, 2.0))
            p2 = SITUATION_POS.get(post_sit, (2.0, 2.0))
            robot_pos = (
                p1[0] + (p2[0] - p1[0]) * t_frac,
                p1[1] + (p2[1] - p1[1]) * t_frac,
            )

        rx, ry = robot_pos
        trail_x.append(rx)
        trail_y.append(ry)

        # --- Derive gaze from primitive skill template ---
        gaze_target, gaze_type = None, None
        if prim_name:
            gaze_target, gaze_type = get_gaze_from_primitive(
                lib, prim_name, robot_pos)
            # For scan type, create a sweeping gaze target
            if gaze_type == "scan":
                angle = -60 + 120 * t_frac
                gaze_dist = 4.0
                gx = rx + gaze_dist * np.cos(np.radians(angle + 60))
                gy = ry + gaze_dist * np.sin(np.radians(angle + 60))
                gaze_target = (gx, gy)

        # --- Heading from movement direction ---
        if len(trail_x) >= 2:
            ddx = trail_x[-1] - trail_x[-2]
            ddy = trail_y[-1] - trail_y[-2]
            if abs(ddx) > 0.01 or abs(ddy) > 0.01:
                heading = np.degrees(np.arctan2(ddy, ddx))
            else:
                heading = 90  # default facing up
        else:
            heading = 90

        # --- Draw ---
        draw_scene_static(ax_scene)

        # Person positions (smooth lerp between situation targets)
        a_pos, b_pos = get_person_positions(pre_sit, post_sit, t_frac)
        if _is_visible(a_pos):
            edge_dist = min(XMAX - a_pos[0], a_pos[0] - XMIN) / 2.0
            alpha = min(1.0, edge_dist)
            draw_person(ax_scene, a_pos, "Person A\n(being served)",
                        COLOR_PERSON, facing_deg=270)
        if _is_visible(b_pos):
            # Derive facing from movement direction
            b1 = PERSON_B_TARGETS.get(pre_sit, EXIT_RIGHT)
            b2 = PERSON_B_TARGETS.get(post_sit, EXIT_RIGHT)
            bdx, bdy = b2[0] - b1[0], b2[1] - b1[1]
            if abs(bdx) > 0.01 or abs(bdy) > 0.01:
                b_facing = np.degrees(np.arctan2(bdy, bdx))
            else:
                b_facing = 90
            draw_person(ax_scene, b_pos, "Person B\n(in queue)",
                        COLOR_PERSON2, facing_deg=b_facing)

        # Trail
        if len(trail_x) > 1:
            ax_scene.plot(trail_x, trail_y, color=COLOR_ROBOT, alpha=0.2,
                          linewidth=2, zorder=3)

        draw_robot(ax_scene, rx, ry, heading, gaze_target, gaze_type)
        draw_status_bar(ax_status, step_idx, t_frac, sequence,
                        situation_chain, topology, prim_to_frag)

        return []

    anim = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                                   interval=80, blit=False)

    out_dir = os.path.join(os.path.dirname(__file__), "demo_figures")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "reception_desk_animation.gif")
    print(f"Rendering {TOTAL_FRAMES} frames...")
    anim.save(out_path, writer="pillow", fps=12, dpi=100)
    print(f"Saved animation to {out_path}")

    plt.close(fig)
    return out_path


if __name__ == "__main__":
    make_animation()
