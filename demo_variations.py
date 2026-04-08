"""Demo Variations — Environment Shapes Cognition.

Runs the composition pipeline with 8 variant configurations grouped into:
  - Set A (Environment-side): same robot, environment changes
  - Set B (System-side): same environment, fragment set changes

Produces two figures:
  - demo_figures/environment_variations.png  (2x2, variants A1–A4)
  - demo_figures/system_variations.png       (2x2, variants B5–B8)

Theoretical basis:
  - Lefebvre: change the space, change the cognition
  - Akrich/Latour: norms are inscribed in material arrangements
  - Albarracin et al. (2021): variational approach to scripts
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
# Domain setup (reused from demo_full_pipeline_anim.py)
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
# Fragment definitions
# ================================================================
def _make_all_fragments() -> Dict[str, ScriptPattern]:
    """Return dict of all 6 fragment configs (5 base + direct_approach)."""
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
    annotation: str  # which talk claim this validates


VARIANTS_ENV = [
    VariantConfig(
        name="Reception (baseline)",
        context="reception",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space"],
        annotation="Full queue norm from environment structure",
    ),
    VariantConfig(
        name="Corridor",
        context="corridor",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space"],
        annotation="Change the space, change the cognition (Lefebvre)",
    ),
    VariantConfig(
        name="Degraded reception",
        context="reception",
        fragment_names=["observe_scene", "queue_position",
                        "approach_service", "courtesy_space"],
        annotation="Norm was in the environment, not the robot (Akrich)",
    ),
    VariantConfig(
        name="Enriched reception",
        context="reception",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space", "direct_approach"],
        annotation="Added cues reshape the affordance landscape (Latour)",
    ),
]

VARIANTS_SYS = [
    VariantConfig(
        name="Baseline (all 5)",
        context="reception",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space"],
        annotation="Full fragment repertoire",
    ),
    VariantConfig(
        name="Missing fragment",
        context="reception",
        fragment_names=["observe_scene", "queue_position",
                        "approach_service", "courtesy_space"],
        annotation="System lacks wait knowledge",
    ),
    VariantConfig(
        name="Extra fragment",
        context="reception",
        fragment_names=["observe_scene", "queue_position", "wait_patiently",
                        "approach_service", "courtesy_space", "direct_approach"],
        annotation="Conflicting prior knowledge",
    ),
    VariantConfig(
        name="Minimal fragments",
        context="reception",
        fragment_names=["observe_scene", "approach_service"],
        annotation="Sparse knowledge — can it still compose?",
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
    sequence: List[str]
    situation_chain: list  # (prim, domain_pre, domain_post, raw_post)
    topology: dict
    context: str
    variant_name: str
    annotation: str
    # Summary stats
    backbone_len: int = 0
    total_g: float = 0.0
    shortcut_g: float = 0.0


# ================================================================
# Free energy computation (reused from demo_full_pipeline_anim.py)
# ================================================================
def compute_transition_fe(topology, from_prim, to_prim):
    """G(A->B) = -log(topology_weight(A,B) + epsilon)."""
    w = topology.get((from_prim, to_prim), 0.0)
    eps = 0.01
    return -math.log(w + eps)


# ================================================================
# Run one variant
# ================================================================
DOMAIN_SITUATIONS = {
    "scene_assessed", "in_queue", "ready_for_service",
    "at_counter", "interaction",
}


def run_variant(config: VariantConfig) -> PipelineResult:
    """Execute the composition pipeline for one variant configuration."""
    all_frags = _make_all_fragments()
    lib = PrimitiveLibrary()
    _register_reception_primitives(lib)

    fragments = [all_frags[name] for name in config.fragment_names]

    cfg = RepertoireConfig()
    rep = ScriptRepertoire(lib, cfg, initial_patterns=fragments)
    rep.enable_compositional_mode()

    # Build query from situation belief
    query_scores = {}
    for name, pat in rep.patterns.items():
        aff = pat.situation_affinity.get(config.context, 0.0)
        query_scores[name] = aff * pat.precision

    # Graph diffusion retrieval
    weighted = rep.retrieve_composition(query_scores)

    # Composition
    composer = ScriptComposer(lib, cfg)
    composite = composer.compose_from_patterns(weighted, config.context)

    if composite is None:
        # Fallback: empty composite
        composite = ScriptPattern(name="empty_composite")

    # Extract topology
    topo = composite.context_topology.get(config.context, {})
    if not topo:
        topo = next(iter(composite.context_topology.values()), {})

    # Build situation chain
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

    # Compute backbone length (unique domain transitions)
    backbone_transitions = set()
    for _, pre, post, _ in situation_chain:
        if pre != post:
            backbone_transitions.add((pre, post))
    backbone_len = len(backbone_transitions)

    # Compute total G along backbone and shortcut G
    total_g = 0.0
    for idx in range(1, len(sequence)):
        g = compute_transition_fe(topo, sequence[idx - 1], sequence[idx])
        total_g += g

    # Compute shortcut G: transitions that skip domain situations
    sit_list = list(dict.fromkeys(
        s for _, _, s, _ in situation_chain if s != (situation_chain[0][1] if situation_chain else "")))
    if situation_chain:
        sit_list.insert(0, situation_chain[0][1])

    backbone_edges = set()
    for _, pre, post, _ in situation_chain:
        if pre != post:
            backbone_edges.add((pre, post))

    shortcut_g = 0.0
    for i in range(len(sit_list)):
        for j in range(i + 2, min(i + 4, len(sit_list))):
            a, b = sit_list[i], sit_list[j]
            if (a, b) not in backbone_edges and a != b:
                best_w = 0.0
                for (pa, pb), w in topo.items():
                    p_a = lib.get(pa)
                    p_b = lib.get(pb)
                    if p_a and p_b:
                        a_match = (p_a.postcondition_situation == a or
                                   a in p_a.precondition_situations)
                        b_match = (p_b.postcondition_situation == b or
                                   b in p_b.precondition_situations)
                        if a_match and b_match and w > best_w:
                            best_w = w
                shortcut_g += -math.log(best_w + 0.01)

    return PipelineResult(
        library=lib,
        fragments=fragments,
        composite=composite,
        weighted=weighted,
        sequence=sequence,
        situation_chain=situation_chain,
        topology=topo,
        context=config.context,
        variant_name=config.name,
        annotation=config.annotation,
        backbone_len=backbone_len,
        total_g=total_g,
        shortcut_g=shortcut_g,
    )


# ================================================================
# Drawing: static FSM panel
# ================================================================
def draw_variant_panel(ax, result: PipelineResult) -> None:
    """Draw a static situation FSM for one variant in a single axes."""
    lib = result.library
    chain = result.situation_chain
    topo = result.topology
    sequence = result.sequence

    # Title and subtitle
    n_frags = len(result.fragments)
    n_prims = len(result.composite.primitive_cluster)
    ax.set_title(
        f"{result.variant_name}\n"
        f"{n_frags} frags, {n_prims} prims, backbone={result.backbone_len}",
        fontsize=8, fontweight="bold", pad=4,
    )

    ax.axis("off")

    if not chain:
        ax.text(0.5, 0.5, "(no composed sequence)", fontsize=8,
                ha="center", va="center", transform=ax.transAxes,
                color="#999999")
        return

    # Collect domain situations
    all_sits = []
    seen = set()
    for _, pre, post, _ in chain:
        for s in (pre, post):
            if s not in seen:
                all_sits.append(s)
                seen.add(s)

    # Build networkx graph
    G = nx.DiGraph()
    for s in all_sits:
        G.add_node(s)

    backbone_edges = {}
    for idx, (prim_name, pre, post, _) in enumerate(chain):
        if pre != post:
            if idx > 0:
                g_val = compute_transition_fe(topo, sequence[idx - 1], prim_name)
            else:
                g_val = 0.0
            G.add_edge(pre, post, label=prim_name, fe=g_val,
                       backbone=True, step=idx)
            backbone_edges[(pre, post)] = (prim_name, g_val, idx)

    # Shortcut edges
    sit_list = list(dict.fromkeys(
        s for _, _, s, _ in chain if s != chain[0][1]))
    sit_list.insert(0, chain[0][1])

    for i in range(len(sit_list)):
        for j in range(i + 2, min(i + 4, len(sit_list))):
            a, b = sit_list[i], sit_list[j]
            if (a, b) not in backbone_edges and a != b:
                best_w = 0.0
                for (pa, pb), w in topo.items():
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

    # Draw shortcut edges (red dashed)
    shortcut_edges = [(a, b) for a, b in G.edges()
                      if not G[a][b].get("backbone")]
    if shortcut_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=shortcut_edges,
                               edge_color="#E74C3C", style="dashed",
                               width=1, alpha=0.4, arrows=True,
                               arrowsize=8, connectionstyle="arc3,rad=0.25")
        for a, b in shortcut_edges:
            mx = (pos[a][0] + pos[b][0]) / 2
            my = (pos[a][1] + pos[b][1]) / 2 + 0.08
            fe = G[a][b]["fe"]
            ax.text(mx, my, f"G={fe:.1f}", fontsize=5, color="#E74C3C",
                    ha="center", alpha=0.6, style="italic")

    # Draw backbone edges (green)
    for a, b in [(a, b) for a, b in G.edges() if G[a][b].get("backbone")]:
        data = G[a][b]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(a, b)],
                               edge_color="#27AE60", width=2.5, alpha=0.9,
                               arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.1")
        mx = (pos[a][0] + pos[b][0]) / 2
        my = (pos[a][1] + pos[b][1]) / 2 - 0.1
        fe = data["fe"]
        ax.text(mx, my, f"{data['label']}\nG={fe:.2f}", fontsize=5,
                ha="center", color="#27AE60", fontweight="bold")

    # Draw nodes
    for s in all_sits:
        x, y = pos[s]
        circle = plt.Circle((x, y), 0.12, facecolor="#27AE60",
                            edgecolor="#2C3E50", linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        short = s.replace("_", "\n")
        ax.text(x, y - 0.18, short, fontsize=5, ha="center",
                va="top", fontweight="bold", color="#2C3E50")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)

    # Composed sequence text below the graph
    seq_str = " -> ".join(sequence)
    # Wrap long sequences
    if len(seq_str) > 60:
        parts = sequence
        mid = len(parts) // 2
        line1 = " -> ".join(parts[:mid])
        line2 = " -> ".join(parts[mid:])
        seq_str = line1 + "\n-> " + line2
    ax.text(0.5, -0.05, seq_str, fontsize=5, ha="center", va="top",
            transform=ax.transAxes, color="#555555", style="italic",
            wrap=True)

    # Annotation
    ax.text(0.5, -0.12, result.annotation, fontsize=5, ha="center",
            va="top", transform=ax.transAxes, color="#8E44AD",
            fontweight="bold")


# ================================================================
# Draw a 2x2 comparison figure
# ================================================================
def draw_comparison(results: List[PipelineResult], output_path: str,
                    suptitle: str) -> None:
    """Create a 2x2 figure from 4 pipeline results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)

    for idx, (ax, result) in enumerate(zip(axes.flat, results)):
        draw_variant_panel(ax, result)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ================================================================
# Summary table
# ================================================================
def print_summary_table(results: List[PipelineResult]) -> None:
    """Print comparison table to console."""
    header = (
        f"{'Variant':<28s} | {'Context':<10s} | {'Frags':>5s} | "
        f"{'Prims':>5s} | {'Backbone':>8s} | {'Total G':>8s} | {'Shortcut G':>10s}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        n_prims = len(r.composite.primitive_cluster)
        print(
            f"{r.variant_name:<28s} | {r.context:<10s} | {len(r.fragments):>5d} | "
            f"{n_prims:>5d} | {r.backbone_len:>8d} | {r.total_g:>8.1f} | "
            f"{r.shortcut_g:>10.1f}"
        )
    print()


# ================================================================
# Main
# ================================================================
def main():
    out_dir = os.path.join(os.path.dirname(__file__), "demo_figures")

    # --- Set A: Environment variations ---
    print("=== Set A: Environment Changes ===")
    env_results = []
    for vc in VARIANTS_ENV:
        print(f"  Running: {vc.name} (context={vc.context}, "
              f"fragments={len(vc.fragment_names)})...")
        result = run_variant(vc)
        env_results.append(result)
        print(f"    Composed {len(result.sequence)} primitives, "
              f"backbone={result.backbone_len}")

    draw_comparison(
        env_results,
        os.path.join(out_dir, "environment_variations.png"),
        "Environment Shapes Cognition — Same Robot, Different Spaces",
    )

    # --- Set B: System variations ---
    print("\n=== Set B: System Changes ===")
    sys_results = []
    for vc in VARIANTS_SYS:
        print(f"  Running: {vc.name} (context={vc.context}, "
              f"fragments={len(vc.fragment_names)})...")
        result = run_variant(vc)
        sys_results.append(result)
        print(f"    Composed {len(result.sequence)} primitives, "
              f"backbone={result.backbone_len}")

    draw_comparison(
        sys_results,
        os.path.join(out_dir, "system_variations.png"),
        "Compositional Robustness — Same Environment, Different Fragments",
    )

    # --- Summary ---
    print("\n=== Summary: All 8 Variants ===")
    print_summary_table(env_results + sys_results)


if __name__ == "__main__":
    main()
