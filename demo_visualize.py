"""Generate visualization figures for the compositional assembly demo."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import os

from architecture_core.core.types import AffectState, SkillRequest
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig, ScriptPattern, ScriptPrimitive,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_composer import ScriptComposer
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire


def frag(name, prims, affinity):
    return ScriptPattern(
        name=name,
        primitives_sequence=sorted(prims),
        primitive_cluster=set(prims),
        precision=0.1,
        situation_affinity=affinity,
    )


def _register_reception_primitives(lib):
    """Register domain primitives for the reception desk scenario."""
    domain_prims = [
        ScriptPrimitive(
            name="scan-environment",
            skill_template=SkillRequest(skill="gaze", params={"mode": "scan_area"}),
            precondition_situations=["open_area", "corridor_encounter"],
            postcondition_situation="scene_assessed",
            expected_affect=AffectState(valence=0.0, arousal=0.1),
            typical_duration_s=3.0, deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="position-in-queue",
            skill_template=SkillRequest(skill="navigate", params={"intent": "queue", "speed_scale": 0.4}),
            precondition_situations=["scene_assessed", "open_area"],
            postcondition_situation="in_queue",
            expected_affect=AffectState(valence=0.0, arousal=-0.1),
            typical_duration_s=4.0, deontic_default="obligatory",
        ),
        ScriptPrimitive(
            name="wait-for-turn",
            skill_template=SkillRequest(skill="navigate", params={"intent": "wait", "speed_scale": 0.0}),
            precondition_situations=["in_queue"],
            postcondition_situation="ready_for_service",
            expected_affect=AffectState(valence=0.0, arousal=-0.2),
            typical_duration_s=10.0, deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="approach-counter",
            skill_template=SkillRequest(skill="navigate", params={"intent": "approach", "speed_scale": 0.5}),
            precondition_situations=["ready_for_service", "open_area"],
            postcondition_situation="at_counter",
            expected_affect=AffectState(valence=0.2, arousal=0.1),
            typical_duration_s=4.0, deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="engage-staff",
            skill_template=SkillRequest(skill="gaze", params={"mode": "look_at_staff"}),
            precondition_situations=["at_counter", "interaction"],
            postcondition_situation="interaction",
            expected_affect=AffectState(valence=0.3, arousal=0.1),
            typical_duration_s=3.0, deontic_default="permitted",
        ),
    ]
    for p in domain_prims:
        lib.register(p)


# --- Setup ---
fragments = [
    frag("observe_scene",    {"scan-environment", "gaze-scan"},      {"reception": 0.9, "corridor": 0.4}),
    frag("queue_position",   {"position-in-queue", "yield-pass"},    {"reception": 0.8, "corridor": 0.3}),
    frag("wait_patiently",   {"wait-for-turn", "wait-acknowledge"},  {"reception": 0.7, "corridor": 0.5}),
    frag("approach_service", {"approach-counter", "gaze-at-agent"},  {"reception": 0.9, "open_area": 0.4}),
    frag("courtesy_space",   {"yield-pass", "gaze-avert"},           {"reception": 0.3, "corridor": 0.8}),
]

FRAG_COLORS = {
    "observe_scene":    "#4A90D9",
    "queue_position":   "#50C878",
    "wait_patiently":   "#2ECC71",
    "approach_service": "#E8734A",
    "courtesy_space":   "#9B59B6",
}

FRAG_PRIMS = {
    "observe_scene":    {"scan-environment", "gaze-scan"},
    "queue_position":   {"position-in-queue", "yield-pass"},
    "wait_patiently":   {"wait-for-turn", "wait-acknowledge"},
    "approach_service": {"approach-counter", "gaze-at-agent"},
    "courtesy_space":   {"yield-pass", "gaze-avert"},
}

lib = PrimitiveLibrary()
_register_reception_primitives(lib)
cfg = RepertoireConfig()
rep = ScriptRepertoire(lib, config=cfg, initial_patterns=fragments)
rep.enable_compositional_mode()

out_dir = os.path.join(os.path.dirname(__file__), "demo_figures")
os.makedirs(out_dir, exist_ok=True)


def get_prim_color(pname):
    sources = [fn for fn, ps in FRAG_PRIMS.items() if pname in ps]
    if len(sources) == 1:
        return FRAG_COLORS[sources[0]]
    elif len(sources) > 1:
        return "#FFD700"
    return "#CCCCCC"


# ================================================================
# Shared data
# ================================================================
query_scores = {}
for name, pat in rep.patterns.items():
    aff = pat.situation_affinity.get("reception", 0.0)
    query_scores[name] = aff * pat.precision

before = dict(query_scores)
weighted = rep.retrieve_composition(query_scores)
after = {wp.pattern.name: wp.weight for wp in weighted}

composer = ScriptComposer(lib, cfg)
result = composer.compose_from_patterns(weighted, "reception")

prim_names = sorted(result.primitive_weights.keys(),
                    key=lambda p: result.primitive_weights[p], reverse=True)
prim_vals = [result.primitive_weights[p] for p in prim_names]


# ================================================================
# FIGURE 1: Pattern Graph
# ================================================================
def fig1_pattern_graph():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    G = nx.Graph()
    for name in rep._pattern_graph:
        G.add_node(name)
    for a, neighbors in rep._pattern_graph.items():
        for b, w in neighbors.items():
            if a < b:
                G.add_edge(a, b, weight=w)

    pos = nx.spring_layout(G, seed=42, k=2)
    edges = G.edges(data=True)
    widths = [d["weight"] * 4 for _, _, d in edges]
    edge_colors = [plt.cm.YlOrRd(d["weight"]) for _, _, d in edges]

    node_colors = []
    for n in G.nodes():
        aff = rep.patterns[n].situation_affinity.get("reception", 0.0)
        node_colors.append(plt.cm.Blues(0.3 + 0.7 * aff))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000,
                           node_color=node_colors, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=edge_colors, alpha=0.8)

    edge_labels = {(a, b): f"{d['weight']:.2f}" for a, b, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

    ax.set_title("Pattern Graph\n(edge = Jaccard + cosine, node blue intensity = reception affinity)",
                 fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_pattern_graph.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 1_pattern_graph.png")


# ================================================================
# FIGURE 2: Diffusion before/after
# ================================================================
def fig2_diffusion():
    names = sorted(before.keys())
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, [before[n] for n in names], w,
           label="Before diffusion (affinity x precision)", color="#4A90D9", edgecolor="black")
    ax.bar(x + w / 2, [after.get(n, 0) for n in names], w,
           label="After graph diffusion", color="#E8734A", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Activation Score")
    ax.set_title("Graph Diffusion: Reception Query\n"
                 "(scores spread to related fragments via graph edges)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_diffusion_before_after.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 2_diffusion_before_after.png")


# ================================================================
# FIGURE 3: Composed primitive weights
# ================================================================
def fig3_primitive_weights():
    colors = [get_prim_color(p) for p in prim_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(prim_names)), prim_vals, color=colors, edgecolor="black")
    ax.set_yticks(range(len(prim_names)))
    ax.set_yticklabels(prim_names, fontsize=11)
    ax.set_xlabel("Merged Weight")
    ax.set_title("Composed Script: Primitive Weights\n(gold = shared across fragments)")
    ax.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color=c, label=n) for n, c in FRAG_COLORS.items()
    ]
    legend_patches.append(mpatches.Patch(color="#FFD700", label="shared (multi-fragment)"))
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "3_composed_primitive_weights.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 3_composed_primitive_weights.png")


# ================================================================
# FIGURE 4: Confident vs Uncertain
# ================================================================
def fig4_confident_vs_uncertain():
    confident_scores = {}
    for name, pat in rep.patterns.items():
        confident_scores[name] = pat.situation_affinity.get("reception", 0.0) * 0.9

    uncertain_scores = {}
    for name, pat in rep.patterns.items():
        all_aff = list(pat.situation_affinity.values())
        avg_aff = sum(all_aff) / len(all_aff) if all_aff else 0.0
        uncertain_scores[name] = avg_aff * 0.3

    conf_result = rep.retrieve_composition(confident_scores)
    unc_result = rep.retrieve_composition(uncertain_scores)

    conf_dict = {wp.pattern.name: wp.weight for wp in conf_result}
    unc_dict = {wp.pattern.name: wp.weight for wp in unc_result}

    conf_total = sum(conf_dict.values()) or 1
    unc_total = sum(unc_dict.values()) or 1

    names_sorted = sorted(conf_dict.keys())
    colors_list = [FRAG_COLORS.get(n, "#CCCCCC") for n in names_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    conf_vals = [conf_dict.get(n, 0) / conf_total for n in names_sorted]
    ax1.pie(conf_vals, labels=names_sorted, autopct="%1.1f%%",
            colors=colors_list, startangle=90)
    ax1.set_title("Confident Query\n(reception=0.9)", fontsize=12, fontweight="bold")

    unc_vals = [unc_dict.get(n, 0) / unc_total for n in names_sorted]
    ax2.pie(unc_vals, labels=names_sorted, autopct="%1.1f%%",
            colors=colors_list, startangle=90)
    ax2.set_title("Uncertain Query\n(flat/ambiguous)", fontsize=12, fontweight="bold")

    fig.suptitle("Uncertainty Broadens Composition\n"
                 "(more even distribution = more fragments recruited equally)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_confident_vs_uncertain.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 4_confident_vs_uncertain.png")


# ================================================================
# FIGURE 5: Full pipeline overview
# ================================================================
def fig5_full_pipeline():
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    # Panel 1: Fragments
    ax = axes[0]
    ax.set_title("1. Fragment Patterns", fontsize=11, fontweight="bold")
    ax.axis("off")
    y = 0.95
    for i, f in enumerate(fragments):
        color = FRAG_COLORS.get(f.name, list(FRAG_COLORS.values())[i % len(FRAG_COLORS)])
        prims = sorted(f.primitive_cluster)
        ax.text(0.05, y, f.name, fontsize=9, fontweight="bold", color=color,
                transform=ax.transAxes)
        y -= 0.05
        prim_str = ", ".join(prims)
        ax.text(0.08, y, prim_str, fontsize=6, transform=ax.transAxes, family="monospace")
        y -= 0.04
        aff_str = ", ".join(f"{k}:{v}" for k, v in f.situation_affinity.items())
        ax.text(0.08, y, aff_str, fontsize=6, transform=ax.transAxes, family="monospace",
                color="gray")
        y -= 0.08

    # Panel 2: Graph
    ax = axes[1]
    ax.set_title("2. Pattern Graph", fontsize=11, fontweight="bold")
    G2 = nx.Graph()
    for name in rep._pattern_graph:
        G2.add_node(name)
    for a, neighbors in rep._pattern_graph.items():
        for b, w in neighbors.items():
            if a < b:
                G2.add_edge(a, b, weight=w)
    pos2 = nx.spring_layout(G2, seed=42, k=2.5)
    edges2 = G2.edges(data=True)
    ws = [d["weight"] * 5 for _, _, d in edges2]
    node_c = [FRAG_COLORS[n] for n in G2.nodes()]
    nx.draw_networkx(G2, pos2, ax=ax, node_size=1200, font_size=7, width=ws,
                     node_color=node_c, font_weight="bold", edgecolors="black", linewidths=1)
    ax.axis("off")

    # Panel 3: Diffusion
    ax = axes[2]
    ax.set_title("3. Graph Diffusion", fontsize=11, fontweight="bold")
    names_list = sorted(before.keys())
    before_vals = [before.get(n, 0) for n in names_list]
    after_vals = [after.get(n, 0) for n in names_list]
    x = np.arange(len(names_list))
    ax.bar(x - 0.2, before_vals, 0.35, label="Before", color="#AAAAAA", edgecolor="black")
    ax.bar(x + 0.2, after_vals, 0.35, label="After", color="#E8734A", edgecolor="black")
    ax.set_xticks(x)
    short_names = [n.replace("_", "\n") for n in names_list]
    ax.set_xticklabels(short_names, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylabel("Score")

    # Panel 4: Composed result
    ax = axes[3]
    ax.set_title("4. Composed Script", fontsize=11, fontweight="bold")
    ax.axis("off")
    ax.text(0.05, 0.92, result.name, fontsize=8, fontweight="bold",
            transform=ax.transAxes, family="monospace")
    ax.text(0.05, 0.83, f"is_composite: {result.is_composite}", fontsize=9,
            transform=ax.transAxes)
    ax.text(0.05, 0.75, f"source fragments: {len(result.source_fragments)}", fontsize=9,
            transform=ax.transAxes)

    y = 0.62
    ax.text(0.05, y, "Primitives (by weight):", fontsize=9, fontweight="bold",
            transform=ax.transAxes)
    y -= 0.08
    max_w = max(prim_vals) if prim_vals else 1
    for p, v in zip(prim_names, prim_vals):
        bar_len = v / max_w * 0.5
        color = get_prim_color(p)
        ax.barh(y, bar_len, height=0.045, left=0.38, color=color, edgecolor="black",
                transform=ax.transAxes, clip_on=False)
        ax.text(0.05, y, p, fontsize=8, transform=ax.transAxes, va="center")
        ax.text(0.38 + bar_len + 0.02, y, f"{v:.3f}", fontsize=7,
                transform=ax.transAxes, va="center")
        y -= 0.07

    fig.suptitle("Compositional Assembly Pipeline: Reception Desk Demo",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "5_full_pipeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 5_full_pipeline.png")


# ================================================================
if __name__ == "__main__":
    fig1_pattern_graph()
    fig2_diffusion()
    fig3_primitive_weights()
    fig4_confident_vs_uncertain()
    fig5_full_pipeline()
    print(f"\nAll figures saved to {out_dir}")
