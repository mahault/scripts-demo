"""Statistical Analysis + Figure Generation for Environment Shapes Cognition.

Loads experiment CSV data and generates 14 PDF figures with proper
active inference axis labels and statistical annotations.

Figures:
  Exp1 (5): cue-primitive heatmap, n_primitives by cues, VFE landscape,
            backbone by context, ANOVA interaction
  Exp2 (4): degradation curves, subset violations, VFE vs fragments,
            topology density
  Exp3 (3): behavioral divergence, sequence comparison, weight profiles
  Cross (2): context effect sizes, pipeline overview
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

# ================================================================
# Configuration
# ================================================================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiment_results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

CONTEXT_COLORS = {
    "reception": "#2196F3",
    "corridor": "#FF9800",
    "hospital": "#4CAF50",
}
CONTEXT_ORDER = ["reception", "corridor", "hospital"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ================================================================
# Data loading
# ================================================================
def load_exp1() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "exp1_material_cues.csv"))


def load_exp2() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "exp2_fragment_scaling.csv"))


def load_exp3() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "exp3_context_transfer.csv"))


def load_all() -> pd.DataFrame:
    return pd.read_json(os.path.join(RESULTS_DIR, "all_experiments.json"))


# ================================================================
# Statistical helpers
# ================================================================
def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled < 1e-10:
        return 0.0
    return (m1 - m2) / pooled


def chi_square_test(observed):
    """Simple chi-square test for independence (no scipy dependency)."""
    observed = np.array(observed, dtype=float)
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    if total == 0:
        return 0.0, 1.0
    expected = row_sum * col_sum / total
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((observed - expected) ** 2 / np.where(expected > 0, expected, 1))
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    # Approximate p-value using chi-square survival function (rough)
    if df <= 0:
        return chi2, 1.0
    # Use simple normal approximation for large chi2
    p_approx = max(1e-10, math.exp(-chi2 / (2 * df))) if chi2 > df else 0.5
    return chi2, p_approx


def spearman_rank(x, y):
    """Compute Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    rho = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))
    return rho


def jaccard_distance(set1, set2):
    """Jaccard distance between two sets."""
    if not set1 and not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return 1.0 - inter / union if union > 0 else 0.0


# ================================================================
# Figure 1: Cue-Primitive Heatmap (Exp1)
# ================================================================
def fig_exp1_cue_primitive_heatmap(df: pd.DataFrame):
    """Binary heatmap: which cues produce which primitives, per context.

    Shows A-matrix -> policy mapping.
    """
    cue_cols = ["cue_stanchions", "cue_waiting_area", "cue_service_sign", "cue_social_density"]
    cue_labels = ["Stanchions", "Waiting area", "Service sign", "Social density"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax_idx, context in enumerate(CONTEXT_ORDER):
        ctx_df = df[df["context"] == context]

        # Get all unique primitives across this context
        all_prims = set()
        for ps in ctx_df["primitive_set"]:
            all_prims.update(ps.split(","))
        all_prims = sorted(all_prims)

        # Build matrix: cue × primitive → fraction of conditions where primitive appears
        matrix = np.zeros((len(cue_labels), len(all_prims)))
        for ci, cue_col in enumerate(cue_cols):
            cue_on = ctx_df[ctx_df[cue_col] == 1]
            cue_off = ctx_df[ctx_df[cue_col] == 0]
            for pi, prim in enumerate(all_prims):
                # Rate when cue is present
                rate_on = cue_on["primitive_set"].apply(lambda x: prim in x.split(",")).mean() if len(cue_on) > 0 else 0
                rate_off = cue_off["primitive_set"].apply(lambda x: prim in x.split(",")).mean() if len(cue_off) > 0 else 0
                matrix[ci, pi] = rate_on - rate_off  # Differential effect

        im = axes[ax_idx].imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[ax_idx].set_xticks(range(len(all_prims)))
        axes[ax_idx].set_xticklabels(all_prims, rotation=45, ha="right", fontsize=7)
        axes[ax_idx].set_yticks(range(len(cue_labels)))
        axes[ax_idx].set_yticklabels(cue_labels)
        axes[ax_idx].set_title(f"{context.capitalize()}", color=CONTEXT_COLORS[context])

        # Annotate with values
        for ci in range(len(cue_labels)):
            for pi in range(len(all_prims)):
                val = matrix[ci, pi]
                if abs(val) > 0.01:
                    axes[ax_idx].text(pi, ci, f"{val:.2f}", ha="center", va="center",
                                      fontsize=6, color="black" if abs(val) < 0.3 else "white")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Differential primitive inclusion rate")
    fig.suptitle("Material Cue Effect on Primitive Inclusion\n(A-matrix $\\rightarrow$ Policy Mapping)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "fig_exp1_cue_primitive_heatmap.pdf"))
    plt.close(fig)
    print("  [1/14] fig_exp1_cue_primitive_heatmap.pdf")


# ================================================================
# Figure 2: N Primitives by Cues (Exp1)
# ================================================================
def fig_exp1_n_primitives_by_cues(df: pd.DataFrame):
    """Grouped bar: n_cues × context -> n_primitives.

    Shows model complexity -> policy complexity.
    """
    df = df.copy()
    df["n_cues"] = df["cue_stanchions"] + df["cue_waiting_area"] + df["cue_service_sign"] + df["cue_social_density"]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(5)  # 0..4 cues

    for i, context in enumerate(CONTEXT_ORDER):
        ctx = df[df["context"] == context]
        means = []
        stds = []
        for nc in range(5):
            vals = ctx[ctx["n_cues"] == nc]["n_primitives"]
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)
        ax.bar(x + i * width, means, width, yerr=stds,
               label=context.capitalize(), color=CONTEXT_COLORS[context],
               alpha=0.85, capsize=3)

    ax.set_xlabel("Number of material cues present")
    ax.set_ylabel("Composed policy length $|\\pi|$")
    ax.set_title("Model Complexity $\\rightarrow$ Policy Complexity")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["0", "1", "2", "3", "4"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp1_n_primitives_by_cues.pdf"))
    plt.close(fig)
    print("  [2/14] fig_exp1_n_primitives_by_cues.pdf")


# ================================================================
# Figure 3: Free Energy Landscape (Exp1)
# ================================================================
def fig_exp1_free_energy_landscape(df: pd.DataFrame):
    """Heatmap: cue combo × context -> total_VFE.

    Shows how environment structure reduces variational free energy.
    """
    cue_cols = ["cue_stanchions", "cue_waiting_area", "cue_service_sign", "cue_social_density"]
    cue_short = ["St", "Wa", "Si", "Sd"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    for ax_idx, context in enumerate(CONTEXT_ORDER):
        ctx = df[df["context"] == context].copy()
        # Create cue combo label
        ctx["cue_combo"] = ctx.apply(
            lambda r: "".join([cue_short[i] if r[cue_cols[i]] == 1 else "-"
                               for i in range(4)]),
            axis=1
        )
        ctx = ctx.sort_values(["cue_stanchions", "cue_waiting_area", "cue_service_sign", "cue_social_density"])

        combos = ctx["cue_combo"].values
        vfe = ctx["mean_VFE"].values

        ax = axes[ax_idx]
        # Reshape into 4x4 grid (stanchions × waiting_area on axes, service_sign × social_density as cells)
        grid = np.zeros((4, 4))
        labels_y = []
        labels_x = []

        idx = 0
        for sd in range(2):
            for si in range(2):
                col = sd * 2 + si
                if col < 4:
                    labels_x.append(f"Si={si},Sd={sd}")
                for wa in range(2):
                    for st in range(2):
                        row = wa * 2 + st
                        if col == 0:
                            labels_y.append(f"St={st},Wa={wa}")
                        if row < 4 and col < 4 and idx < len(vfe):
                            grid[row, col] = vfe[idx]
                        idx += 1

        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels_x[:4], fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels_y[:4], fontsize=7)
        ax.set_title(f"{context.capitalize()}", color=CONTEXT_COLORS[context])

        for r in range(min(4, grid.shape[0])):
            for c in range(min(4, grid.shape[1])):
                ax.text(c, r, f"{grid[r,c]:.1f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Mean VFE per transition $\\bar{F}$")
    fig.suptitle("Variational Free Energy Landscape\n($-\\ln \\mathbf{B}[s_{\\tau+1}|s_\\tau, \\pi]$ averaged over transitions)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(FIG_DIR, "fig_exp1_free_energy_landscape.pdf"))
    plt.close(fig)
    print("  [3/14] fig_exp1_free_energy_landscape.pdf")


# ================================================================
# Figure 4: Backbone by Context (Exp1)
# ================================================================
def fig_exp1_backbone_by_context(df: pd.DataFrame):
    """Strip plot: cue config -> backbone_length, per context.

    Shows B-matrix causal chain structure.
    """
    df = df.copy()
    df["n_cues"] = df["cue_stanchions"] + df["cue_waiting_area"] + df["cue_service_sign"] + df["cue_social_density"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, context in enumerate(CONTEXT_ORDER):
        ctx = df[df["context"] == context]
        jitter = (np.random.RandomState(42).uniform(-0.15, 0.15, len(ctx)))
        ax.scatter(ctx["n_cues"] + i * 0.25 - 0.25 + jitter, ctx["backbone_length"],
                   alpha=0.7, s=30, color=CONTEXT_COLORS[context],
                   label=context.capitalize())

    ax.set_xlabel("Number of material cues")
    ax.set_ylabel("Backbone length (B-matrix causal chain)")
    ax.set_title("B-matrix Causal Chain Structure by Environment")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp1_backbone_by_context.pdf"))
    plt.close(fig)
    print("  [4/14] fig_exp1_backbone_by_context.pdf")


# ================================================================
# Figure 5: ANOVA Interaction (Exp1)
# ================================================================
def fig_exp1_anova_interaction(df: pd.DataFrame):
    """Interaction plot: context × n_cues -> n_primitives.

    Shows C-matrix × model interaction.
    """
    df = df.copy()
    df["n_cues"] = df["cue_stanchions"] + df["cue_waiting_area"] + df["cue_service_sign"] + df["cue_social_density"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for context in CONTEXT_ORDER:
        ctx = df[df["context"] == context]
        means = ctx.groupby("n_cues")["n_primitives"].mean()
        stds = ctx.groupby("n_cues")["n_primitives"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    marker="o", label=context.capitalize(),
                    color=CONTEXT_COLORS[context], capsize=3, linewidth=2)

    ax.set_xlabel("Number of material cues (model complexity)")
    ax.set_ylabel("Mean composed policy length $|\\pi|$")
    ax.set_title("Context $\\times$ Model Complexity Interaction\n"
                 "(C-matrix modulates response to model structure)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp1_anova_interaction.pdf"))
    plt.close(fig)
    print("  [5/14] fig_exp1_anova_interaction.pdf")


# ================================================================
# Figure 6: Degradation Curves (Exp2)
# ================================================================
def fig_exp2_degradation_curves(df: pd.DataFrame):
    """Line: k vs mean n_primitives, per context, with min/max shading.

    Shows graceful degradation as property of variational inference.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for context in CONTEXT_ORDER:
        ctx = df[df["context"] == context]
        grouped = ctx.groupby("n_fragments")["n_primitives"]
        means = grouped.mean()
        mins = grouped.min()
        maxs = grouped.max()

        color = CONTEXT_COLORS[context]
        ax.plot(means.index, means.values, marker="o",
                label=context.capitalize(), color=color, linewidth=2)
        ax.fill_between(means.index, mins.values, maxs.values,
                        alpha=0.15, color=color)

    ax.set_xlabel("Number of available fragments $k$ (generative model richness)")
    ax.set_ylabel("Composed policy length $|\\pi|$")
    ax.set_title("Graceful Degradation Under Model Sparsity\n"
                 "(Variational inference minimizes $F$ with available structure)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 7))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp2_degradation_curves.pdf"))
    plt.close(fig)
    print("  [6/14] fig_exp2_degradation_curves.pdf")


# ================================================================
# Figure 7: Subset Violations (Exp2)
# ================================================================
def fig_exp2_subset_violations(df: pd.DataFrame):
    """Bar: subset property holding rate per transition k -> k+1, per context.

    Tests monotonicity of policy under model growth.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25

    for ci, context in enumerate(CONTEXT_ORDER):
        ctx = df[df["context"] == context]
        rates = []

        for k in range(1, 6):
            k_rows = ctx[ctx["n_fragments"] == k]
            k1_rows = ctx[ctx["n_fragments"] == k + 1]

            holds = 0
            total_pairs = 0

            for _, row_k in k_rows.iterrows():
                ps_k = row_k["primitive_set"]
                fn_k = row_k["fragment_names"]
                if not isinstance(ps_k, str) or not isinstance(fn_k, str):
                    continue
                prims_k = set(ps_k.split(","))
                frags_k = set(fn_k.split(","))

                for _, row_k1 in k1_rows.iterrows():
                    fn_k1 = row_k1["fragment_names"]
                    ps_k1 = row_k1["primitive_set"]
                    if not isinstance(fn_k1, str) or not isinstance(ps_k1, str):
                        continue
                    frags_k1 = set(fn_k1.split(","))
                    if frags_k.issubset(frags_k1):
                        prims_k1 = set(ps_k1.split(","))
                        total_pairs += 1
                        if prims_k.issubset(prims_k1):
                            holds += 1

            rate = holds / total_pairs if total_pairs > 0 else 0.0
            rates.append(rate)

        x = np.arange(1, 6)
        ax.bar(x + ci * width - width, rates, width,
               label=context.capitalize(), color=CONTEXT_COLORS[context], alpha=0.85)

    ax.set_xlabel("Fragment count transition ($k \\rightarrow k+1$)")
    ax.set_ylabel("Subset property holding rate")
    ax.set_title("Policy Monotonicity Under Model Growth\n"
                 "(Higher = more predictable composition)")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"{k}→{k+1}" for k in range(1, 6)])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp2_subset_violations.pdf"))
    plt.close(fig)
    print("  [7/14] fig_exp2_subset_violations.pdf")


# ================================================================
# Figure 8: VFE vs Fragments (Exp2)
# ================================================================
def fig_exp2_free_energy_vs_fragments(df: pd.DataFrame):
    """Line: k vs mean_VFE per primitive, per context.

    Shows per-transition surprisal as function of model richness.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for context in CONTEXT_ORDER:
        ctx = df[df["context"] == context]
        grouped = ctx.groupby("n_fragments")["mean_VFE"]
        means = grouped.mean()
        stds = grouped.std()

        color = CONTEXT_COLORS[context]
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    marker="s", label=context.capitalize(),
                    color=color, capsize=3, linewidth=2)

    ax.set_xlabel("Number of available fragments $k$")
    ax.set_ylabel("Mean VFE per transition $\\bar{F} = -\\frac{1}{|\\pi|}\\sum_\\tau \\ln B[s_{\\tau+1}|s_\\tau]$")
    ax.set_title("Per-Transition Surprisal vs. Model Richness")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 7))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp2_free_energy_vs_fragments.pdf"))
    plt.close(fig)
    print("  [8/14] fig_exp2_free_energy_vs_fragments.pdf")


# ================================================================
# Figure 9: Topology Density (Exp2)
# ================================================================
def fig_exp2_topology_density(df: pd.DataFrame):
    """Line: k vs topology_density, per context.

    Shows B-matrix sparsity vs fragment count.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for context in CONTEXT_ORDER:
        ctx = df[df["context"] == context]
        grouped = ctx.groupby("n_fragments")["topology_density"]
        means = grouped.mean()
        stds = grouped.std()

        color = CONTEXT_COLORS[context]
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    marker="^", label=context.capitalize(),
                    color=color, capsize=3, linewidth=2)

    ax.set_xlabel("Number of available fragments $k$")
    ax.set_ylabel("B-matrix density ($|$edges$|$ / $|$possible edges$|$)")
    ax.set_title("Transition Model Density vs. Fragment Count")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 7))
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp2_topology_density.pdf"))
    plt.close(fig)
    print("  [9/14] fig_exp2_topology_density.pdf")


# ================================================================
# Figure 10: Behavioral Divergence (Exp3)
# ================================================================
def fig_exp3_behavioral_divergence(df: pd.DataFrame):
    """UpSet-style visualization: primitive set overlap across contexts.

    Shows same A/B, different C -> different pi.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Jaccard distance matrix
    prim_sets = {}
    for _, row in df.iterrows():
        prims = set(row["primitive_set"].split(","))
        prim_sets[row["context"]] = prims

    contexts = CONTEXT_ORDER
    n = len(contexts)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = jaccard_distance(prim_sets.get(contexts[i], set()),
                                                   prim_sets.get(contexts[j], set()))

    ax = axes[0]
    im = ax.imshow(dist_matrix, cmap="Blues", vmin=0, vmax=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([c.capitalize() for c in contexts])
    ax.set_yticks(range(n))
    ax.set_yticklabels([c.capitalize() for c in contexts])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist_matrix[i,j]:.3f}", ha="center", va="center", fontsize=10)
    ax.set_title("Jaccard Distance\n(Policy Set Divergence)")
    fig.colorbar(im, ax=ax, shrink=0.7)

    # Right: Sequence ordering differences
    ax = axes[1]
    sequences = {}
    for _, row in df.iterrows():
        sequences[row["context"]] = row["sequence"].split(" -> ")

    y_positions = {c: i for i, c in enumerate(contexts)}
    all_prims = sorted(set(p for seq in sequences.values() for p in seq))
    prim_to_x = {p: i for i, p in enumerate(all_prims)}

    for context, seq in sequences.items():
        y = y_positions[context]
        color = CONTEXT_COLORS[context]
        for pos, prim in enumerate(seq):
            ax.scatter(pos, y, s=60, color=color, zorder=5)
            ax.text(pos, y + 0.15, prim.replace("-", "\n"), fontsize=5,
                    ha="center", va="bottom", rotation=0)

    ax.set_yticks(range(len(contexts)))
    ax.set_yticklabels([c.capitalize() for c in contexts])
    ax.set_xlabel("Position in composed sequence")
    ax.set_title("Sequence Ordering\n(Same primitives, different order)")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Behavioral Divergence: Same Knowledge, Different Preferences\n"
                 "(Identical A/B-matrices, different C-matrix $\\rightarrow$ different $\\pi$)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(FIG_DIR, "fig_exp3_behavioral_divergence.pdf"))
    plt.close(fig)
    print("  [10/14] fig_exp3_behavioral_divergence.pdf")


# ================================================================
# Figure 11: Sequence Comparison (Exp3)
# ================================================================
def fig_exp3_sequence_comparison(df: pd.DataFrame):
    """Side-by-side sequences, color-coded by source fragment.

    Shows how C-matrix reweights the same fragments.
    """
    # Fragment -> primitive mapping
    frag_prims = {
        "observe_scene": {"scan-environment", "gaze-scan"},
        "queue_position": {"position-in-queue", "yield-pass"},
        "wait_patiently": {"wait-for-turn", "wait-acknowledge"},
        "approach_service": {"approach-counter", "gaze-at-agent"},
        "courtesy_space": {"yield-pass", "gaze-avert"},
        "direct_approach": {"approach-counter", "engage-staff"},
    }

    frag_colors = {
        "observe_scene": "#E3F2FD",
        "queue_position": "#FFF3E0",
        "wait_patiently": "#E8F5E9",
        "approach_service": "#FCE4EC",
        "courtesy_space": "#F3E5F5",
        "direct_approach": "#FFFDE7",
    }

    def prim_to_frag(prim_name):
        """Map primitive to its primary fragment."""
        for fname, prims in frag_prims.items():
            if prim_name in prims:
                return fname
        return "unknown"

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

    for ax_idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[ax_idx]
        context = row["context"]
        sequence = row["sequence"].split(" -> ")

        for i, prim in enumerate(sequence):
            frag = prim_to_frag(prim)
            color = frag_colors.get(frag, "#EEEEEE")
            rect = FancyBboxPatch((i - 0.4, -0.3), 0.8, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor="gray", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(i, 0.0, prim.replace("-", "\n"), fontsize=6,
                    ha="center", va="center")
            ax.text(i, -0.45, frag[:8], fontsize=5, ha="center", va="top",
                    color="gray", style="italic")

        ax.set_xlim(-0.6, len(sequence) - 0.4)
        ax.set_ylim(-0.7, 0.5)
        ax.set_ylabel(f"{context.capitalize()}", fontsize=11,
                       color=CONTEXT_COLORS[context], fontweight="bold")
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.2)

    axes[-1].set_xlabel("Sequence position")
    fig.suptitle("Composed Sequences Across Contexts\n"
                 "(Color = source fragment, same knowledge base)",
                 fontsize=13)

    # Legend
    handles = [mpatches.Patch(facecolor=frag_colors[f], edgecolor="gray", label=f)
               for f in frag_prims]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, "fig_exp3_sequence_comparison.pdf"))
    plt.close(fig)
    print("  [11/14] fig_exp3_sequence_comparison.pdf")


# ================================================================
# Figure 12: Weight Profiles (Exp3)
# ================================================================
def fig_exp3_weight_profiles(df: pd.DataFrame):
    """Radar/spider chart: Q(f|c) posterior fragment weights per context.

    Shows belief propagation converges differently per context.
    """
    frag_names = ["observe_scene", "queue_position", "wait_patiently",
                  "approach_service", "courtesy_space", "direct_approach"]
    n_frags = len(frag_names)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, n_frags, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    for _, row in df.iterrows():
        context = row["context"]
        ws = json.loads(row["weighted_scores"])
        values = [ws.get(f, 0.0) for f in frag_names]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2,
                color=CONTEXT_COLORS[context],
                label=context.capitalize())
        ax.fill(angles, values, alpha=0.1, color=CONTEXT_COLORS[context])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(frag_names, fontsize=8)
    ax.set_title("Posterior Fragment Weights $Q(f|c)$\n"
                 "(Belief propagation on factor graph)",
                 fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp3_weight_profiles.pdf"))
    plt.close(fig)
    print("  [12/14] fig_exp3_weight_profiles.pdf")


# ================================================================
# Figure 13: Context Effect Sizes (Cross-experiment)
# ================================================================
def fig_context_effect_sizes(df_all: pd.DataFrame):
    """Forest plot: Cohen's d for each metric between context pairs.

    Quantifies C-matrix effect magnitude.
    """
    metrics = ["n_primitives", "total_VFE", "mean_VFE", "total_EFE",
               "backbone_length", "topology_density",
               "n_advancing", "n_returning"]
    metric_labels = [
        "$|\\pi|$ (policy length)", "Total VFE", "Mean VFE", "Total EFE",
        "Backbone length", "B-matrix density",
        "$n$ advancing", "$n$ returning"
    ]

    pairs = [("reception", "corridor"), ("reception", "hospital"), ("corridor", "hospital")]
    pair_labels = ["Rec. vs. Corr.", "Rec. vs. Hosp.", "Corr. vs. Hosp."]

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = 0
    y_ticks = []
    y_labels = []

    for mi, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        for pi, ((c1, c2), plabel) in enumerate(zip(pairs, pair_labels)):
            g1 = df_all[df_all["context"] == c1][metric].values
            g2 = df_all[df_all["context"] == c2][metric].values
            d = cohens_d(g1, g2)

            color = "#2196F3" if pi == 0 else "#FF9800" if pi == 1 else "#4CAF50"
            ax.barh(y_pos, d, height=0.6, color=color, alpha=0.7)
            ax.text(d + 0.05 * np.sign(d), y_pos, f"{d:.2f}",
                    va="center", fontsize=7)

            y_ticks.append(y_pos)
            y_labels.append(f"{mlabel}\n({plabel})" if pi == 0 else plabel)
            y_pos += 1
        y_pos += 0.5  # Gap between metrics

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Cohen's $d$ effect size")
    ax.set_title("Context Effect Sizes (C-matrix Impact)\n"
                 "across all experimental conditions")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(-0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0.8, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axvline(-0.8, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

    # Effect size reference lines
    ax.text(0.2, y_pos + 0.5, "small", fontsize=7, color="gray", ha="center")
    ax.text(0.8, y_pos + 0.5, "large", fontsize=7, color="gray", ha="center")

    handles = [mpatches.Patch(color="#2196F3", alpha=0.7, label="Rec. vs. Corr."),
               mpatches.Patch(color="#FF9800", alpha=0.7, label="Rec. vs. Hosp."),
               mpatches.Patch(color="#4CAF50", alpha=0.7, label="Corr. vs. Hosp.")]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_context_effect_sizes.pdf"))
    plt.close(fig)
    print("  [13/14] fig_context_effect_sizes.pdf")


# ================================================================
# Figure 14: Pipeline Overview (Methods)
# ================================================================
def fig_pipeline_overview():
    """Schematic of generative model + composition pipeline.

    Shows A, B, C, D matrices and their role in composition.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(7, 7.6, "Compositional Assembly Pipeline", fontsize=14,
            ha="center", fontweight="bold")

    # --- Generative Model (left) ---
    gm_x, gm_y = 1.5, 5.5
    gm = FancyBboxPatch((gm_x - 1.3, gm_y - 1.0), 2.6, 2.0,
                         boxstyle="round,pad=0.1",
                         facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2)
    ax.add_patch(gm)
    ax.text(gm_x, gm_y + 0.7, "Generative Model", fontsize=10,
            ha="center", fontweight="bold", color="#1565C0")
    ax.text(gm_x, gm_y + 0.2, "$P(\\tilde{o}, \\tilde{s}, \\pi)$", fontsize=11,
            ha="center", style="italic")
    ax.text(gm_x, gm_y - 0.3,
            "$= P(s_1) \\prod B(\\pi) \\prod A$",
            fontsize=8, ha="center", color="#333")
    ax.text(gm_x, gm_y - 0.7,
            "D · B · A matrices",
            fontsize=8, ha="center", color="#666")

    # --- Matrix boxes ---
    mat_y = 3.0
    matrices = [
        ("A", "P(o|s)", "Obs.\nlikelihood", "#E8F5E9", "#2E7D32", 0.5),
        ("B", "P(s'|s,π)", "Transition\nmodel", "#FFF3E0", "#E65100", 3.0),
        ("C", "ln P(o)", "Preferences\n(context)", "#FCE4EC", "#C62828", 5.5),
        ("D", "P(s₁)", "Initial\nstate prior", "#F3E5F5", "#6A1B9A", 8.0),
    ]

    for label, formula, desc, bg, fg, mx in matrices:
        box = FancyBboxPatch((mx - 0.8, mat_y - 0.6), 1.6, 1.2,
                              boxstyle="round,pad=0.05",
                              facecolor=bg, edgecolor=fg, linewidth=1.5)
        ax.add_patch(box)
        ax.text(mx, mat_y + 0.3, f"{label}-matrix", fontsize=9,
                ha="center", fontweight="bold", color=fg)
        ax.text(mx, mat_y - 0.0, formula, fontsize=8, ha="center")
        ax.text(mx, mat_y - 0.35, desc, fontsize=6, ha="center", color="#666")

    # --- Pipeline stages (right) ---
    stages = [
        ("1. D-matrix\nScoring", "w_f = aff · prec", 10.0, 6.5),
        ("2. Belief\nPropagation", "Graph diffusion", 10.0, 5.5),
        ("3. B-matrix\nConstruction", "compose_from_patterns()", 10.0, 4.5),
        ("4. Backbone\nExtraction", "Causal sequence", 10.0, 3.5),
        ("5. Policy\nSelection", "σ(-γ · G(π))", 10.0, 2.5),
    ]

    for label, formula, sx, sy in stages:
        box = FancyBboxPatch((sx - 1.2, sy - 0.35), 2.4, 0.7,
                              boxstyle="round,pad=0.05",
                              facecolor="#FFFDE7", edgecolor="#F57F17", linewidth=1)
        ax.add_patch(box)
        ax.text(sx - 0.5, sy, label, fontsize=7, ha="center", fontweight="bold")
        ax.text(sx + 0.7, sy, formula, fontsize=6, ha="center", color="#666", style="italic")

    # Arrows between stages
    for i in range(len(stages) - 1):
        ax.annotate("", xy=(10.0, stages[i + 1][3] + 0.35),
                     xytext=(10.0, stages[i][3] - 0.35),
                     arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    # --- Context boxes at bottom ---
    ctx_y = 1.0
    for i, (ctx, color) in enumerate(CONTEXT_COLORS.items()):
        cx = 3.0 + i * 4.0
        box = FancyBboxPatch((cx - 1.5, ctx_y - 0.3), 3.0, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor=color, linewidth=1, alpha=0.2)
        ax.add_patch(box)
        ax.text(cx, ctx_y, f"{ctx.capitalize()} context\n(different C-matrix)",
                fontsize=8, ha="center", color=color, fontweight="bold")

    # Arrow from C-matrix to contexts
    ax.annotate("", xy=(7, 1.4), xytext=(5.5, mat_y - 0.6),
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5, ls="--"))

    # Arrow from generative model to pipeline
    ax.annotate("", xy=(8.8, 6.5), xytext=(2.8, 5.5),
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2))

    fig.savefig(os.path.join(FIG_DIR, "fig_pipeline_overview.pdf"))
    plt.close(fig)
    print("  [14/14] fig_pipeline_overview.pdf")


# ================================================================
# Statistical summary
# ================================================================
def print_statistics(df1, df2, df3, df_all):
    """Print key statistical results."""
    print("\n" + "=" * 60)
    print("Statistical Summary")
    print("=" * 60)

    # Exp1: Chi-square for each cue
    print("\n--- Exp1: Cue-Primitive Independence (Chi-square) ---")
    cue_cols = ["cue_stanchions", "cue_waiting_area", "cue_service_sign", "cue_social_density"]
    cue_labels = ["Stanchions", "Waiting area", "Service sign", "Social density"]
    for cue_col, cue_label in zip(cue_cols, cue_labels):
        # Build contingency table: cue present/absent × low/high primitive count
        median_prims = df1["n_primitives"].median()
        observed = np.zeros((2, 2))
        for _, row in df1.iterrows():
            ci = row[cue_col]
            pi = 1 if row["n_primitives"] > median_prims else 0
            observed[ci, pi] += 1
        chi2, p = chi_square_test(observed)
        print(f"  {cue_label}: chi2={chi2:.2f}, p~{p:.4f}")

    # Exp1: two-way descriptive stats
    print("\n--- Exp1: Context x n_cues (Descriptive) ---")
    df1_copy = df1.copy()
    df1_copy["n_cues"] = df1_copy[cue_cols].sum(axis=1)
    for context in CONTEXT_ORDER:
        ctx = df1_copy[df1_copy["context"] == context]
        rho = spearman_rank(ctx["n_cues"].values, ctx["n_primitives"].values)
        print(f"  {context}: Spearman rho(n_cues, n_prims) = {rho:.3f}")

    # Exp2: Spearman monotonicity
    print("\n--- Exp2: Fragment Scaling Monotonicity ---")
    for context in CONTEXT_ORDER:
        ctx = df2[df2["context"] == context]
        rho = spearman_rank(ctx["n_fragments"].values, ctx["n_primitives"].values)
        print(f"  {context}: Spearman rho(k, |pi|) = {rho:.3f}")

    # Exp3: KL divergences
    print("\n--- Exp3: Context Transfer KL Divergences ---")
    for _, row in df3.iterrows():
        print(f"  {row['context']}: D_KL from reception baseline = {row['D_KL_from_baseline']:.4f}")

    # Exp3: Jaccard distances
    print("\n--- Exp3: Behavioral Divergence (Jaccard) ---")
    prim_sets = {}
    for _, row in df3.iterrows():
        prim_sets[row["context"]] = set(row["primitive_set"].split(","))
    for c1, c2 in combinations(CONTEXT_ORDER, 2):
        jd = jaccard_distance(prim_sets.get(c1, set()), prim_sets.get(c2, set()))
        print(f"  {c1} vs {c2}: Jaccard distance = {jd:.3f}")

    # Cross-experiment: effect sizes
    print("\n--- Cross-Experiment: Cohen's d (n_primitives) ---")
    for c1, c2 in combinations(CONTEXT_ORDER, 2):
        g1 = df_all[df_all["context"] == c1]["n_primitives"].values
        g2 = df_all[df_all["context"] == c2]["n_primitives"].values
        d = cohens_d(g1, g2)
        print(f"  {c1} vs {c2}: d = {d:.3f}")

    print("\n--- Cross-Experiment: Cohen's d (total_EFE) ---")
    for c1, c2 in combinations(CONTEXT_ORDER, 2):
        g1 = df_all[df_all["context"] == c1]["total_EFE"].values
        g2 = df_all[df_all["context"] == c2]["total_EFE"].values
        d = cohens_d(g1, g2)
        print(f"  {c1} vs {c2}: d = {d:.3f}")


# ================================================================
# Main
# ================================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading experiment data...")
    df1 = load_exp1()
    df2 = load_exp2()
    df3 = load_exp3()
    df_all = load_all()

    print(f"  Exp1: {len(df1)} rows")
    print(f"  Exp2: {len(df2)} rows")
    print(f"  Exp3: {len(df3)} rows")
    print(f"  All:  {len(df_all)} rows")

    print("\nGenerating figures...")

    # Exp1 figures
    fig_exp1_cue_primitive_heatmap(df1)
    fig_exp1_n_primitives_by_cues(df1)
    fig_exp1_free_energy_landscape(df1)
    fig_exp1_backbone_by_context(df1)
    fig_exp1_anova_interaction(df1)

    # Exp2 figures
    fig_exp2_degradation_curves(df2)
    fig_exp2_subset_violations(df2)
    fig_exp2_free_energy_vs_fragments(df2)
    fig_exp2_topology_density(df2)

    # Exp3 figures
    fig_exp3_behavioral_divergence(df3)
    fig_exp3_sequence_comparison(df3)
    fig_exp3_weight_profiles(df3)

    # Cross-experiment figures
    fig_context_effect_sizes(df_all)
    fig_pipeline_overview()

    # Statistics
    print_statistics(df1, df2, df3, df_all)

    print(f"\nAll 14 figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
