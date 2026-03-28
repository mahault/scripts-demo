"""Pipeline-driven animation: runs the real composition system, visualises output.

Nothing is hard-coded.  The animation:
  1. Runs the actual composition pipeline (library -> fragments -> graph ->
     diffusion -> compose_from_patterns -> backbone extraction).
  2. Derives ALL visualisation data from the pipeline output.
  3. Shows the situation state machine with free-energy gradients -- norm
     compliance emerges from EFE minimisation, not hard constraints.
  4. Shows fragment composition (weak -> composite).
  5. Situation-driven robot positions: the robot's location is derived from
     the postcondition of the current primitive, not from hand-coded paths.

Theoretical basis: Albarracin, Constant, Friston & Ramstead (2021) --
A Variational Approach to Scripts.
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
# Domain setup (identical to tests/test_compositional_assembly.py)
# ================================================================
def _register_reception_primitives(lib: PrimitiveLibrary) -> None:
    """Register domain primitives encoding the queue-joining causal chain."""
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
@dataclass
class PipelineResult:
    library: PrimitiveLibrary
    fragments: List[ScriptPattern]
    repertoire: ScriptRepertoire
    composite: ScriptPattern
    before_diffusion: Dict[str, float]
    after_diffusion: Dict[str, float]
    weighted: List[WeightedPattern]
    graph_edges: Dict[Tuple[str, str], float]
    sequence: List[str]
    # (primitive, domain_pre, domain_post, raw_post)
    situation_chain: list
    # Topology from composite (for real G values)
    topology: dict


def run_pipeline() -> PipelineResult:
    """Execute the real composition pipeline. No simulation."""
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

    # Build query from situation belief (reception context)
    query_scores = {}
    for name, pat in rep.patterns.items():
        aff = pat.situation_affinity.get("reception", 0.0)
        query_scores[name] = aff * pat.precision

    before_diffusion = dict(query_scores)

    # Real graph diffusion retrieval
    weighted = rep.retrieve_composition(query_scores)
    after_diffusion = {wp.pattern.name: wp.weight for wp in weighted}

    # Real composition
    composer = ScriptComposer(lib, cfg)
    composite = composer.compose_from_patterns(weighted, "reception")

    # Extract graph edges
    graph = rep._pattern_graph or {}
    graph_edges = {}
    for a, neighbors in graph.items():
        for b, w in neighbors.items():
            if (b, a) not in graph_edges:  # avoid duplicates
                graph_edges[(a, b)] = w

    # Build situation chain from the composed sequence.
    # Track "domain situation" — the most specific situation in the
    # backbone chain (scene_assessed, in_queue, ready_for_service,
    # at_counter, interaction).  Hub primitives (gaze-scan, yield-pass
    # etc.) have generic postconditions that don't advance the domain
    # state.  This mirrors active inference: the agent maintains a
    # belief over situation types, and generic postconditions don't
    # override the domain belief.
    DOMAIN_SITUATIONS = {
        "scene_assessed", "in_queue", "ready_for_service",
        "at_counter", "interaction",
    }

    sequence = list(composite.primitives_sequence)
    situation_chain = []
    domain_sit = "open_area"  # starting domain situation
    for prim_name in sequence:
        prim = lib.get(prim_name)
        post_sit = prim.postcondition_situation if prim else domain_sit
        # Only advance domain situation if the postcondition is
        # a domain-specific situation (not a generic reset)
        if post_sit in DOMAIN_SITUATIONS:
            new_domain = post_sit
        else:
            new_domain = domain_sit
        situation_chain.append((prim_name, domain_sit, new_domain, post_sit))
        domain_sit = new_domain

    # Extract topology for computing real G values
    topo = composite.context_topology.get("reception", {})
    if not topo:
        # Fall back to first available context
        topo = next(iter(composite.context_topology.values()), {})

    print(f"Composed sequence ({len(sequence)} primitives):")
    for prim_name, pre, post, raw_post in situation_chain:
        print(f"  {pre:20s} --[{prim_name}]--> {post}"
              f"  (raw: {raw_post})")
    print(f"Source fragments: {composite.source_fragments}")
    print(f"Is composite: {composite.is_composite}")
    print(f"Topology contexts: {list(composite.context_topology.keys())}")
    print(f"Topology edges: {len(topo)}")

    return PipelineResult(
        library=lib,
        fragments=fragments,
        repertoire=rep,
        composite=composite,
        before_diffusion=before_diffusion,
        after_diffusion=after_diffusion,
        weighted=weighted,
        graph_edges=graph_edges,
        sequence=sequence,
        situation_chain=situation_chain,
        topology=topo,
    )


# ================================================================
# Situation -> scene position mapping (visualisation only)
# ================================================================
SITUATION_POS = {
    "open_area":         (2.0, 2.0),
    "corridor_encounter": (2.0, 2.0),
    "scene_assessed":    (2.5, 3.0),
    "in_queue":          (4.0, 4.0),
    "ready_for_service": (4.0, 5.5),
    "at_counter":        (5.2, 7.0),
    "interaction":       (5.5, 7.5),
}

# Scene constants
DESK_X, DESK_Y = 5.0, 8.5
DESK_W, DESK_H = 3.0, 0.6
PERSON_AT_DESK = (5.5, 7.5)
PERSON_IN_QUEUE = (4.0, 5.5)
EXIT_RIGHT = (10.5, 7.5)    # off-screen exit
XMIN, XMAX = -0.5, 10.5
YMIN, YMAX = -0.5, 10.5

# Person target positions per domain situation.
# These define the physical world state implied by each situation.
# As the situation transitions, people lerp smoothly between targets.
PERSON_A_TARGETS = {
    "open_area":         PERSON_AT_DESK,
    "corridor_encounter": PERSON_AT_DESK,
    "scene_assessed":    PERSON_AT_DESK,
    "in_queue":          PERSON_AT_DESK,
    "ready_for_service": EXIT_RIGHT,       # A finishes, walks off
    "at_counter":        EXIT_RIGHT,
    "interaction":       EXIT_RIGHT,
}

PERSON_B_TARGETS = {
    "open_area":         PERSON_IN_QUEUE,
    "corridor_encounter": PERSON_IN_QUEUE,
    "scene_assessed":    PERSON_IN_QUEUE,
    "in_queue":          PERSON_IN_QUEUE,
    "ready_for_service": PERSON_AT_DESK,   # B advances to desk
    "at_counter":        EXIT_RIGHT,       # B finishes, walks off
    "interaction":       EXIT_RIGHT,
}

COLOR_ROBOT = "#3498DB"
COLOR_PERSON = "#E67E22"
COLOR_PERSON2 = "#E74C3C"
COLOR_DESK = "#7F8C8D"
COLOR_FLOOR = "#F5F5DC"

# Fragment colors (assigned dynamically from pipeline output)
FRAG_PALETTE = ["#4A90D9", "#50C878", "#2ECC71", "#E8734A", "#9B59B6",
                "#F39C12", "#E74C3C", "#1ABC9C"]


def get_frag_colors(fragments):
    return {f.name: FRAG_PALETTE[i % len(FRAG_PALETTE)]
            for i, f in enumerate(fragments)}


# ================================================================
# Free energy from topology (principled active inference)
# ================================================================
def compute_transition_fe(topology, from_prim, to_prim):
    """Compute expected free energy for a primitive transition.

    G(A->B) = -log(topology_weight(A,B) + epsilon)

    The topology weight encodes how well A's postcondition matches B's
    preconditions in context.  High weight = natural transition = low G.
    Low weight = precondition mismatch = high prediction error = high G.

    This IS the active inference mechanism: the agent's generative model
    predicts situation transitions via the B-matrix (topology).  Actions
    that violate the predicted transition produce prediction error, which
    the agent minimises by preferring the low-G (backbone) path.
    """
    w = topology.get((from_prim, to_prim), 0.0)
    eps = 0.01
    return -math.log(w + eps)


def compute_step_fe(topology, sequence, step_idx):
    """Compute G for the transition into step step_idx."""
    if step_idx <= 0 or step_idx >= len(sequence):
        return 0.0
    return compute_transition_fe(topology, sequence[step_idx - 1],
                                 sequence[step_idx])


# ================================================================
# Smooth person position interpolation
# ================================================================
def _lerp_pos(p1, p2, t):
    """Linear interpolation between two (x, y) positions."""
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)


def get_person_positions(pre_sit, post_sit, t_frac):
    """Return (person_a_pos, person_b_pos) with smooth transitions.

    People lerp between their target positions as the domain situation
    transitions — exactly like the robot does.  When a person moves
    past the visible area they are not drawn.
    """
    a1 = PERSON_A_TARGETS.get(pre_sit, EXIT_RIGHT)
    a2 = PERSON_A_TARGETS.get(post_sit, EXIT_RIGHT)
    a_pos = _lerp_pos(a1, a2, t_frac)

    b1 = PERSON_B_TARGETS.get(pre_sit, EXIT_RIGHT)
    b2 = PERSON_B_TARGETS.get(post_sit, EXIT_RIGHT)
    b_pos = _lerp_pos(b1, b2, t_frac)

    return a_pos, b_pos


def _is_visible(pos):
    """Return True if position is within the visible scene bounds."""
    return XMIN < pos[0] < XMAX and YMIN < pos[1] < YMAX


# ================================================================
# Drawing functions
# ================================================================
def draw_scene(ax, robot_pos, situation, step_idx, n_steps, prim_name,
               a_pos, b_pos):
    """Draw the top-down scene panel. All positions derived from situation."""
    ax.clear()
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(COLOR_FLOOR)

    if prim_name:
        ax.set_title(f"Scene  |  {prim_name}  [{situation}]",
                     fontsize=9, fontweight="bold", pad=2)
    else:
        ax.set_title(f"Scene  |  [{situation}]", fontsize=9, pad=2)

    # Wall
    ax.fill_between([XMIN, XMAX], [YMAX, YMAX], [DESK_Y + 0.8, DESK_Y + 0.8],
                    color="#D5DBDB", zorder=1)

    # Desk
    desk = mpatches.FancyBboxPatch(
        (DESK_X - DESK_W / 2, DESK_Y - DESK_H / 2), DESK_W, DESK_H,
        boxstyle="round,pad=0.1", facecolor=COLOR_DESK, edgecolor="#2C3E50",
        linewidth=2, zorder=5)
    ax.add_patch(desk)
    ax.text(DESK_X + DESK_W / 2, DESK_Y, "RECEPTION", fontsize=6,
            fontweight="bold", ha="center", va="center", color="white", zorder=6)

    # Queue markers
    for i in range(4):
        qy = 2.0 + i * 1.5
        ax.plot(4.0, qy, "s", color="#BDC3C7", markersize=5, zorder=2)

    # People (smoothly interpolated positions)
    if _is_visible(a_pos):
        # Fade out as person approaches edge
        edge_dist = min(XMAX - a_pos[0], a_pos[0] - XMIN) / 2.0
        alpha = min(1.0, edge_dist)
        c = plt.Circle(a_pos, 0.3, facecolor=COLOR_PERSON, edgecolor="#2C3E50",
                        linewidth=1.5, zorder=10, alpha=alpha)
        ax.add_patch(c)
        ax.text(a_pos[0], a_pos[1] - 0.45, "A", fontsize=6, ha="center",
                color=COLOR_PERSON, fontweight="bold", zorder=11, alpha=alpha)
    if _is_visible(b_pos):
        edge_dist = min(XMAX - b_pos[0], b_pos[0] - XMIN) / 2.0
        alpha = min(1.0, edge_dist)
        c = plt.Circle(b_pos, 0.3, facecolor=COLOR_PERSON2, edgecolor="#2C3E50",
                        linewidth=1.5, zorder=10, alpha=alpha)
        ax.add_patch(c)
        ax.text(b_pos[0], b_pos[1] - 0.45, "B", fontsize=6, ha="center",
                color=COLOR_PERSON2, fontweight="bold", zorder=11, alpha=alpha)

    # Robot
    rx, ry = robot_pos
    body = plt.Circle((rx, ry), 0.35, facecolor=COLOR_ROBOT,
                       edgecolor="#1A5276", linewidth=2, zorder=15)
    ax.add_patch(body)
    ax.text(rx, ry, "R", fontsize=8, fontweight="bold", ha="center",
            va="center", color="white", zorder=17)


def draw_situation_fsm(ax, lib, situation_chain, topology, sequence,
                       current_step, t_frac):
    """Draw the situation state machine with free-energy gradients.

    The topology (B-matrix) from the composite encodes transition
    probabilities between primitives.  Free energy G = -log(w + eps).

    Backbone edges: high topology weight -> low G -> thick, colored.
    Shortcut edges: low topology weight -> high G -> thin, red dashed.
    """
    ax.clear()
    ax.axis("off")
    ax.set_title("Situation State Machine  (G = -log B-matrix)",
                 fontsize=9, fontweight="bold", pad=2)

    # Collect domain situations from the chain (4-tuple: prim, pre, post, raw)
    all_sits = []
    seen = set()
    for _, pre, post, _ in situation_chain:
        for s in (pre, post):
            if s not in seen:
                all_sits.append(s)
                seen.add(s)

    # Build the graph
    G = nx.DiGraph()
    for s in all_sits:
        G.add_node(s)

    # Backbone edges (from the domain situation chain)
    backbone_edges = {}
    for idx, (prim_name, pre, post, _) in enumerate(situation_chain):
        if pre != post:
            # Compute G from real topology
            if idx > 0:
                g_val = compute_transition_fe(topology, sequence[idx - 1],
                                              prim_name)
            else:
                g_val = 0.0
            G.add_edge(pre, post, label=prim_name, fe=g_val,
                       backbone=True, step=idx)
            backbone_edges[(pre, post)] = (prim_name, g_val, idx)

    # Shortcut edges (norm-violating: skip steps in domain chain)
    sit_list = list(dict.fromkeys(
        s for _, _, s, _ in situation_chain if s != situation_chain[0][1]))
    if situation_chain:
        sit_list.insert(0, situation_chain[0][1])

    for i in range(len(sit_list)):
        for j in range(i + 2, min(i + 4, len(sit_list))):
            a, b = sit_list[i], sit_list[j]
            if (a, b) not in backbone_edges and a != b:
                # Compute G from topology: find the best (highest-weight)
                # direct transition between any primitives in these
                # situations.  This is always much lower than backbone
                # weight, producing high G.
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

    # Circular layout
    n = len(all_sits)
    pos = {}
    for i, s in enumerate(all_sits):
        angle = math.pi * 0.8 - i * (math.pi * 1.2 / max(n - 1, 1))
        pos[s] = (math.cos(angle), math.sin(angle))

    # Determine current domain situation
    if current_step < 0:
        current_sit = situation_chain[0][1] if situation_chain else ""
    elif current_step < len(situation_chain):
        _, pre, post, _ = situation_chain[current_step]
        current_sit = pre if t_frac < 0.5 else post
    else:
        current_sit = situation_chain[-1][2] if situation_chain else ""

    # Draw shortcut edges (norm-violating: high G)
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

    # Draw backbone edges
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

    # Draw nodes
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


def draw_fragments(ax, fragments, composite, frag_colors, weighted):
    """Draw the fragment composition view."""
    ax.clear()
    ax.axis("off")
    ax.set_title("Fragment Composition", fontsize=9, fontweight="bold", pad=2)

    n_frags = len(fragments)
    row_h = 0.12
    start_y = 0.92

    # Weight lookup
    weight_map = {wp.pattern.name: wp.weight for wp in weighted}

    for i, frag in enumerate(fragments):
        y = start_y - i * (row_h + 0.04)
        color = frag_colors.get(frag.name, "#CCCCCC")
        w = weight_map.get(frag.name, 0.0)

        # Fragment name + weight
        ax.text(0.02, y, f"{frag.name}", fontsize=6, fontweight="bold",
                color=color, transform=ax.transAxes, va="center")
        ax.text(0.35, y, f"w={w:.3f}", fontsize=5, color="#7F8C8D",
                transform=ax.transAxes, va="center")

        # Primitive blocks
        prims = sorted(frag.primitive_cluster)
        block_x = 0.45
        for p in prims:
            # Check if this primitive made it into the composite
            in_composite = p in composite.primitive_cluster
            alpha = 1.0 if in_composite else 0.3
            rect = mpatches.FancyBboxPatch(
                (block_x, y - 0.025), 0.18, 0.05,
                boxstyle="round,pad=0.01", facecolor=color, alpha=alpha,
                edgecolor="#2C3E50" if in_composite else "none",
                linewidth=0.5, transform=ax.transAxes)
            ax.add_patch(rect)
            short_p = p[:12] if len(p) > 12 else p
            ax.text(block_x + 0.09, y, short_p, fontsize=4, ha="center",
                    va="center", color="white" if in_composite else "#AAAAAA",
                    transform=ax.transAxes)
            block_x += 0.20

    # Composite row
    y = start_y - n_frags * (row_h + 0.04) - 0.06
    ax.plot([0.02, 0.95], [y + 0.04, y + 0.04], color="#2C3E50",
            linewidth=0.5, transform=ax.transAxes, alpha=0.3)
    ax.text(0.02, y, "COMPOSITE", fontsize=6, fontweight="bold",
            color="#2C3E50", transform=ax.transAxes, va="center")
    ax.text(0.35, y, f"{len(composite.primitive_cluster)} primitives",
            fontsize=5, color="#7F8C8D", transform=ax.transAxes, va="center")
    ax.text(0.02, y - 0.05,
            f"from {len(composite.source_fragments)} fragments",
            fontsize=5, color="#95A5A6", transform=ax.transAxes,
            va="center", style="italic")


def draw_execution(ax, sequence, lib, current_step, t_frac, frag_colors,
                   prim_to_frag):
    """Draw the execution sequence panel."""
    ax.clear()
    ax.axis("off")
    ax.set_title("Execution Sequence", fontsize=9, fontweight="bold", pad=2)

    n = len(sequence)
    for i, prim_name in enumerate(sequence):
        y = 0.95 - i * (0.85 / max(n, 1))
        prim = lib.get(prim_name)
        post = prim.postcondition_situation if prim else "?"

        # Determine state
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

        # Step marker
        ax.text(0.04, y, marker, fontsize=8, fontweight="bold", color=color,
                transform=ax.transAxes, va="center", ha="center")

        # Primitive name
        ax.text(0.10, y, prim_name, fontsize=6, fontweight="bold",
                color=color, transform=ax.transAxes, va="center",
                alpha=alpha)

        # Postcondition situation
        ax.text(0.62, y, f"\u2192 {post}", fontsize=5, color="#7F8C8D",
                transform=ax.transAxes, va="center", alpha=alpha)

        # Source fragment indicator
        frags = prim_to_frag.get(prim_name, [])
        if frags:
            frag_str = ", ".join(frags)
            fc = frag_colors.get(frags[0], "#CCCCCC") if len(frags) == 1 else "#FFD700"
            ax.plot(0.92, y, "s", color=fc, markersize=5,
                    transform=ax.transAxes, alpha=alpha)


# ================================================================
# Main animation
# ================================================================
def make_animation():
    print("Running composition pipeline...")
    result = run_pipeline()

    sequence = result.sequence
    chain = result.situation_chain
    lib = result.library
    composite = result.composite
    fragments = result.fragments
    weighted = result.weighted
    topology = result.topology
    frag_colors = get_frag_colors(fragments)

    # Build primitive -> source fragments mapping
    prim_to_frag = {}
    for frag in fragments:
        for p in frag.primitive_cluster:
            prim_to_frag.setdefault(p, []).append(frag.name)

    # Animation timing
    FRAMES_PER_STEP = 25
    INTRO_FRAMES = 15
    OUTRO_FRAMES = 20
    TOTAL_FRAMES = INTRO_FRAMES + len(sequence) * FRAMES_PER_STEP + OUTRO_FRAMES

    # Figure layout
    fig = plt.figure(figsize=(16, 10))
    ax_scene = fig.add_axes([0.02, 0.20, 0.38, 0.68])
    ax_fsm   = fig.add_axes([0.42, 0.48, 0.56, 0.42])
    ax_frag  = fig.add_axes([0.42, 0.20, 0.28, 0.26])
    ax_exec  = fig.add_axes([0.72, 0.20, 0.26, 0.26])
    ax_status = fig.add_axes([0.02, 0.02, 0.96, 0.16])

    fig.text(0.5, 0.95,
             "Compositional Assembly \u2014 Norm Compliance via Free Energy",
             fontsize=14, fontweight="bold", ha="center")
    fig.text(0.5, 0.92,
             "Queue norm emerges from EFE minimisation, not hard constraints "
             "(Albarracin et al. 2021)",
             fontsize=9, ha="center", color="#7F8C8D")

    trail_x, trail_y = [], []

    def update(frame):
        if frame < INTRO_FRAMES:
            step_idx = -1
            t_frac = frame / INTRO_FRAMES
        elif frame >= INTRO_FRAMES + len(sequence) * FRAMES_PER_STEP:
            step_idx = len(sequence)
            t_frac = 1.0
        else:
            f = frame - INTRO_FRAMES
            step_idx = f // FRAMES_PER_STEP
            t_frac = (f % FRAMES_PER_STEP) / FRAMES_PER_STEP

        # Derive current domain situation and robot position from the chain
        # (4-tuple: prim_name, domain_pre, domain_post, raw_post)
        if step_idx < 0:
            pre_sit = chain[0][1] if chain else "open_area"
            post_sit = pre_sit
            current_sit = pre_sit
            prim_name = ""
            start = SITUATION_POS.get(current_sit, (2.0, 2.0))
            robot_pos = (start[0], start[1] * t_frac)
        elif step_idx >= len(sequence):
            pre_sit = chain[-1][2] if chain else "open_area"
            post_sit = pre_sit
            current_sit = pre_sit
            prim_name = ""
            robot_pos = SITUATION_POS.get(current_sit, (5.0, 7.0))
        else:
            prim_name, pre_sit, post_sit, _ = chain[step_idx]
            current_sit = pre_sit if t_frac < 0.5 else post_sit
            # Lerp robot between domain situation positions
            p1 = SITUATION_POS.get(pre_sit, (2.0, 2.0))
            p2 = SITUATION_POS.get(post_sit, (2.0, 2.0))
            robot_pos = (
                p1[0] + (p2[0] - p1[0]) * t_frac,
                p1[1] + (p2[1] - p1[1]) * t_frac,
            )

        # Smooth person positions (lerp between situation targets)
        a_pos, b_pos = get_person_positions(pre_sit, post_sit, t_frac)

        trail_x.append(robot_pos[0])
        trail_y.append(robot_pos[1])

        # Draw all panels
        draw_scene(ax_scene, robot_pos, current_sit, step_idx,
                   len(sequence), prim_name, a_pos, b_pos)
        # Trail on scene
        if len(trail_x) > 1:
            ax_scene.plot(trail_x, trail_y, color=COLOR_ROBOT, alpha=0.15,
                          linewidth=2, zorder=3)

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
        elif step_idx >= len(sequence):
            ax_status.text(0.5, 0.6,
                           "Composite script complete \u2014 norm-compliant "
                           "queue-joining from compositional assembly",
                           fontsize=10, ha="center", color="#27AE60",
                           fontweight="bold")
        else:
            pn = sequence[step_idx]
            src_frags = prim_to_frag.get(pn, ["?"])
            fe = compute_step_fe(topology, sequence, step_idx)

            ax_status.text(0.02, 0.7,
                           f"Step {step_idx+1}/{len(sequence)}:  {pn}",
                           fontsize=10, fontweight="bold", color="#2C3E50")
            ax_status.text(0.02, 0.35,
                           f"Situation: {chain[step_idx][1]} \u2192 "
                           f"{chain[step_idx][2]}   |   "
                           f"G = {fe:.2f}   |   "
                           f"from: {', '.join(src_frags)}",
                           fontsize=8, color="#7F8C8D")

        # Progress dots
        n_prims = len(sequence)
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

    out_dir = os.path.join(os.path.dirname(__file__), "demo_figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "full_pipeline_animation.gif")
    print(f"Rendering {TOTAL_FRAMES} frames...")
    anim.save(out_path, writer="pillow", fps=10, dpi=90)
    print(f"Saved to {out_path}")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    make_animation()
