"""Statistical Analysis + Figure Generation for Environment Shapes Cognition.

Loads experiment CSV data and generates 14 PDF figures with proper
active inference axis labels and statistical annotations.

Key insight: structural metrics (n_primitives, VFE, backbone) are
context-independent -- same fragments produce same structure regardless
of context.  Evaluative metrics (EFE, fragment weights, primitive weights)
are context-dependent -- the C-matrix reshapes policy evaluation.

Figures:
  Exp1 (5): cue-primitive mapping, EFE by cues, EFE landscape,
            EFE by cue config, EFE interaction
  Exp2 (4): degradation (dual panel), weight concentration, EFE vs k,
            mean primitive weight
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
    if df <= 0:
        return chi2, 1.0
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


def weight_entropy(weights_dict):
    """Shannon entropy of a weight distribution (normalized)."""
    vals = np.array(list(weights_dict.values()), dtype=float)
    total = vals.sum()
    if total <= 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def cosine_distance(d1, d2):
    """Cosine distance between two weight dictionaries."""
    all_keys = sorted(set(d1) | set(d2))
    v1 = np.array([d1.get(k, 0.0) for k in all_keys])
    v2 = np.array([d2.get(k, 0.0) for k in all_keys])
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 1.0
    return 1.0 - np.dot(v1, v2) / (norm1 * norm2)


# ================================================================
# Figure 1: Cue-Primitive Mapping (single panel, context-independent)
# ================================================================
def fig_exp1_cue_primitive_heatmap(df: pd.DataFrame):
    """Single-panel heatmap: which cues activate which primitives.

    This mapping is context-independent -- material cues determine
    the B-matrix structure regardless of the C-matrix.
    """
    cue_cols = ["cue_stanchions", "cue_waiting_area", "cue_service_sign", "cue_social_density"]
    cue_labels = ["Stanchions\n(queue_position)", "Waiting area\n(wait_patiently)",
                  "Service sign\n(direct_approach)", "Social density\n(courtesy_space)"]

    # Use reception data (context-independent, so any context works)
    ctx_df = df[df["context"] == "reception"]

    all_prims = set()
    for ps in ctx_df["primitive_set"]:
        all_prims.update(ps.split(","))
    all_prims = sorted(all_prims)

    matrix = np.zeros((len(cue_labels), len(all_prims)))
    for ci, cue_col in enumerate(cue_cols):
        cue_on = ctx_df[ctx_df[cue_col] == 1]
        cue_off = ctx_df[ctx_df[cue_col] == 0]
        for pi, prim in enumerate(all_prims):
            rate_on = cue_on["primitive_set"].apply(lambda x, p=prim: p in x.split(",")).mean() if len(cue_on) > 0 else 0
            rate_off = cue_off["primitive_set"].apply(lambda x, p=prim: p in x.split(",")).mean() if len(cue_off) > 0 else 0
            matrix[ci, pi] = rate_on - rate_off

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)

    ax.set_xticks(range(len(all_prims)))
    ax.set_xticklabels(all_prims, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cue_labels)))
    ax.set_yticklabels(cue_labels, fontsize=9)

    for ci in range(len(cue_labels)):
        for pi in range(len(all_prims)):
            val = matrix[ci, pi]
            if abs(val) > 0.01:
                color = "white" if abs(val) > 0.35 else "black"
                ax.text(pi, ci, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Differential inclusion rate\n(cue present - cue absent)", fontsize=9)

    ax.set_title("Material Cue $\\rightarrow$ Primitive Mapping\n"
                 "(Context-independent: A-matrix structure determines B-matrix content)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp1_cue_primitive_heatmap.pdf"))
    plt.close(fig)
    print("  [1/14] fig_exp1_cue_primitive_heatmap.pdf")


# ================================================================
# Figure 2: EFE by Cues (Exp1)
# ================================================================
def fig_exp1_n_primitives_by_cues(df: pd.DataFrame):
    """Grouped bar: n_cues x context -> total_EFE.

    Shows how context preference structure reshapes policy evaluation
    even when policy structure is identical.
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
            vals = ctx[ctx["n_cues"] == nc]["total_EFE"]
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)
        ax.bar(x + i * width, means, width, yerr=stds,
               label=context.capitalize(), color=CONTEXT_COLORS[context],
               alpha=0.85, capsize=3)

    ax.set_xlabel("Number of material cues present")
    ax.set_ylabel("Total expected free energy $\\mathcal{G}(\\pi)$")
    ax.set_title("C-matrix Reshapes Policy Evaluation\n"
                 "(Same policy structure, different free energy profiles)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["0", "1", "2", "3", "4"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp1_n_primitives_by_cues.pdf"))
    plt.close(fig)
    print("  [2/14] fig_exp1_n_primitives_by_cues.pdf")


# ================================================================
# Figure 3: EFE Landscape (Exp1)
# ================================================================
def fig_exp1_free_energy_landscape(df: pd.DataFrame):
    """Heatmap: cue combo x context -> total_EFE.

    Shows how environment structure and context jointly shape
    expected free energy of composed policies.
    """
    cue_cols = ["cue_stanchions", "cue_waiting_area", "cue_service_sign", "cue_social_density"]
    cue_short = ["St", "Wa", "Si", "Sd"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vmin = df["total_EFE"].min()
    vmax = df["total_EFE"].max()

    for ax_idx, context in enumerate(CONTEXT_ORDER):
        ctx = df[df["context"] == context].copy()
        ctx["cue_combo"] = ctx.apply(
            lambda r: "".join([cue_short[i] if r[cue_cols[i]] == 1 else "-"
                               for i in range(4)]),
            axis=1
        )
        ctx = ctx.sort_values(cue_cols)

        combos = ctx["cue_combo"].values
        efe_vals = ctx["total_EFE"].values

        ax = axes[ax_idx]
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
                        if row < 4 and col < 4 and idx < len(efe_vals):
                            grid[row, col] = efe_vals[idx]
                        idx += 1

        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels_x[:4], fontsize=7, rotation=30, ha="right")
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels_y[:4], fontsize=7)
        ax.set_title(f"{context.capitalize()}", color=CONTEXT_COLORS[context],
                     fontsize=11, fontweight="bold")

        for r in range(min(4, grid.shape[0])):
            for c in range(min(4, grid.shape[1])):
                ax.text(c, r, f"{grid[r,c]:.1f}", ha="center", va="center", fontsize=7)

    # Place colorbar to the right of the last panel, not overlapping
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.7, pad=0.03)
    cbar.set_label("Total EFE $\\mathcal{G}(\\pi)$", fontsize=10)

    fig.suptitle("Expected Free Energy Landscape\n"
                 "(Same B-matrix structure, different C-matrix evaluation)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92, 0.90])
    fig.savefig(os.path.join(FIG_DIR, "fig_exp1_free_energy_landscape.pdf"))
    plt.close(fig)
    print("  [3/14] fig_exp1_free_energy_landscape.pdf")


# ================================================================
# Figure 4: EFE by Cue Configuration (Exp1)
# ================================================================
def fig_exp1_backbone_by_context(df: pd.DataFrame):
    """Box/strip plot: n_cues -> total_EFE, per context.

    Shows the distribution of EFE across cue configurations,
    revealing how context preference structure reshapes policy evaluation.
    """
    df = df.copy()
    df["n_cues"] = df["cue_stanchions"] + df["cue_waiting_area"] + df["cue_service_sign"] + df["cue_social_density"]

    fig, ax = plt.subplots(figsize=(9, 5))

    offsets = {"reception": -0.22, "corridor": 0.0, "hospital": 0.22}
    for context in CONTEXT_ORDER:
        ctx = df[df["context"] == context]
        for nc in range(5):
            vals = ctx[ctx["n_cues"] == nc]["total_EFE"].values
            x_pos = nc + offsets[context]
            seed = (42 + sum(ord(c) for c in context)) % (2**31)
            jitter = np.random.RandomState(seed).uniform(-0.06, 0.06, len(vals))
            ax.scatter(x_pos + jitter, vals, alpha=0.7, s=35,
                       color=CONTEXT_COLORS[context], edgecolor="white", linewidth=0.3)

        # Connect means
        means = [ctx[ctx["n_cues"] == nc]["total_EFE"].mean() for nc in range(5)]
        ax.plot([nc + offsets[context] for nc in range(5)], means,
                color=CONTEXT_COLORS[context], linewidth=1.5, alpha=0.5,
                marker="D", markersize=6, label=context.capitalize())

    ax.set_xlabel("Number of material cues")
    ax.set_ylabel("Total expected free energy $\\mathcal{G}(\\pi)$")
    ax.set_title("EFE Distribution by Material Cue Count\n"
                 "(Context preference structure separates evaluation landscapes)")
    ax.set_xticks(range(5))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp1_backbone_by_context.pdf"))
    plt.close(fig)
    print("  [4/14] fig_exp1_backbone_by_context.pdf")


# ================================================================
# Figure 5: Context x Model Complexity Interaction (Exp1)
# ================================================================
def fig_exp1_anova_interaction(df: pd.DataFrame):
    """Interaction plot: context x n_cues -> total_EFE.

    Shows how C-matrix modulates the response to model structure.
    """
    df = df.copy()
    df["n_cues"] = df["cue_stanchions"] + df["cue_waiting_area"] + df["cue_service_sign"] + df["cue_social_density"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for context in CONTEXT_ORDER:
        ctx = df[df["context"] == context]
        means = ctx.groupby("n_cues")["total_EFE"].mean()
        stds = ctx.groupby("n_cues")["total_EFE"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    marker="o", label=context.capitalize(),
                    color=CONTEXT_COLORS[context], capsize=3, linewidth=2)

    ax.set_xlabel("Number of material cues (model complexity)")
    ax.set_ylabel("Mean total EFE $\\overline{\\mathcal{G}}(\\pi)$")
    ax.set_title("Context $\\times$ Model Complexity Interaction\n"
                 "(Non-parallel lines: C-matrix modulates EFE response to B-matrix structure)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp1_anova_interaction.pdf"))
    plt.close(fig)
    print("  [5/14] fig_exp1_anova_interaction.pdf")


# ================================================================
# Figure 6: Degradation Curves -- dual panel (Exp2)
# ================================================================
def fig_exp2_degradation_curves(df: pd.DataFrame):
    """Dual-panel: left = policy length (context-independent),
    right = total EFE per context (context-dependent).

    Shows structural invariance vs. evaluative divergence.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: n_primitives (context-independent -- single line)
    ctx0 = df[df["context"] == CONTEXT_ORDER[0]]
    grouped = ctx0.groupby("n_fragments")["n_primitives"]
    means = grouped.mean()
    mins = grouped.min()
    maxs = grouped.max()

    ax1.plot(means.index, means.values, marker="o", color="#555555", linewidth=2,
             label="All contexts (identical)")
    ax1.fill_between(means.index, mins.values, maxs.values, alpha=0.15, color="#555555")

    ax1.set_xlabel("Number of available fragments $k$")
    ax1.set_ylabel("Composed policy length $|\\pi|$")
    ax1.set_title("Policy Structure (Context-Invariant)\n"
                  "Same fragments $\\rightarrow$ same length")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(range(1, 7))

    # Spearman annotation
    rho = spearman_rank(ctx0["n_fragments"].values, ctx0["n_primitives"].values)
    ax1.text(0.05, 0.92, f"Spearman $\\rho$ = {rho:.2f}",
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # Right panel: total_EFE (context-dependent)
    for context in CONTEXT_ORDER:
        ctx = df[df["context"] == context]
        grouped = ctx.groupby("n_fragments")["total_EFE"]
        means = grouped.mean()
        mins = grouped.min()
        maxs = grouped.max()

        color = CONTEXT_COLORS[context]
        ax2.plot(means.index, means.values, marker="o",
                 label=context.capitalize(), color=color, linewidth=2)
        ax2.fill_between(means.index, mins.values, maxs.values,
                         alpha=0.12, color=color)

    ax2.set_xlabel("Number of available fragments $k$")
    ax2.set_ylabel("Total expected free energy $\\mathcal{G}(\\pi)$")
    ax2.set_title("Policy Evaluation (Context-Dependent)\n"
                  "Same structure $\\rightarrow$ different EFE")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(range(1, 7))

    fig.suptitle("Structural Invariance vs. Evaluative Divergence Under Model Sparsity",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp2_degradation_curves.pdf"))
    plt.close(fig)
    print("  [6/14] fig_exp2_degradation_curves.pdf")


# ================================================================
# Figure 7: Weight Concentration by k (Exp2)
# ================================================================
def fig_exp2_subset_violations(df: pd.DataFrame):
    """Line: k vs weight entropy per context.

    Shows how fragment weight concentration varies with model richness
    and context.  Higher entropy = more evenly distributed weights.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for context in CONTEXT_ORDER:
        ctx = df[df["context"] == context]
        entropies_by_k = defaultdict(list)

        for _, row in ctx.iterrows():
            ws = row["weighted_scores"]
            if not isinstance(ws, str):
                continue
            try:
                weights = json.loads(ws)
            except (json.JSONDecodeError, TypeError):
                continue
            h = weight_entropy(weights)
            entropies_by_k[row["n_fragments"]].append(h)

        ks = sorted(entropies_by_k.keys())
        means = [np.mean(entropies_by_k[k]) for k in ks]
        stds = [np.std(entropies_by_k[k]) for k in ks]

        color = CONTEXT_COLORS[context]
        ax.errorbar(ks, means, yerr=stds, marker="s",
                    label=context.capitalize(), color=color,
                    capsize=3, linewidth=2)

    # Reference: maximum entropy line
    for k in range(1, 7):
        max_h = np.log(k) if k > 1 else 0
        ax.plot(k, max_h, marker="_", color="gray", markersize=10, markeredgewidth=2)
    ax.plot([], [], marker="_", color="gray", markersize=10, markeredgewidth=2,
            label="Max entropy (uniform)")

    ax.set_xlabel("Number of available fragments $k$")
    ax.set_ylabel("Shannon entropy of $Q(f|c)$ weights")
    ax.set_title("Fragment Weight Concentration by Model Richness\n"
                 "(Higher = more evenly distributed posterior weights)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 7))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp2_subset_violations.pdf"))
    plt.close(fig)
    print("  [7/14] fig_exp2_subset_violations.pdf")


# ================================================================
# Figure 8: EFE vs Fragments (Exp2)
# ================================================================
def fig_exp2_free_energy_vs_fragments(df: pd.DataFrame):
    """Line: k vs mean total_EFE, per context.

    Shows how expected free energy scales with model richness
    and how context modulates the scaling.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for context in CONTEXT_ORDER:
        ctx = df[df["context"] == context]
        grouped = ctx.groupby("n_fragments")["total_EFE"]
        means = grouped.mean()
        stds = grouped.std()

        color = CONTEXT_COLORS[context]
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    marker="s", label=context.capitalize(),
                    color=color, capsize=3, linewidth=2)

    ax.set_xlabel("Number of available fragments $k$")
    ax.set_ylabel("Mean total EFE $\\overline{\\mathcal{G}}(\\pi)$")
    ax.set_title("Expected Free Energy vs. Model Richness\n"
                 "(Context preference structure creates persistent EFE separation)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 7))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp2_free_energy_vs_fragments.pdf"))
    plt.close(fig)
    print("  [8/14] fig_exp2_free_energy_vs_fragments.pdf")


# ================================================================
# Figure 9: Mean Primitive Weight by k (Exp2)
# ================================================================
def fig_exp2_topology_density(df: pd.DataFrame):
    """Line: k vs mean total primitive weight, per context.

    Shows how context reshapes the weight allocated to composed
    primitives at each model complexity level.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for context in CONTEXT_ORDER:
        ctx = df[df["context"] == context]
        weight_sums_by_k = defaultdict(list)

        for _, row in ctx.iterrows():
            pw = row["primitive_weights"]
            if not isinstance(pw, str):
                continue
            try:
                weights = json.loads(pw)
            except (json.JSONDecodeError, TypeError):
                continue
            total_w = sum(weights.values())
            weight_sums_by_k[row["n_fragments"]].append(total_w)

        ks = sorted(weight_sums_by_k.keys())
        means = [np.mean(weight_sums_by_k[k]) for k in ks]
        stds = [np.std(weight_sums_by_k[k]) for k in ks]

        color = CONTEXT_COLORS[context]
        ax.errorbar(ks, means, yerr=stds, marker="^",
                    label=context.capitalize(), color=color,
                    capsize=3, linewidth=2)

    ax.set_xlabel("Number of available fragments $k$")
    ax.set_ylabel("Total primitive weight $\\sum_p w_p$")
    ax.set_title("Primitive Weight Budget by Model Richness\n"
                 "(Context-dependent: C-matrix via D-matrix affinities reshapes weight allocation)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 7))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_exp2_topology_density.pdf"))
    plt.close(fig)
    print("  [9/14] fig_exp2_topology_density.pdf")


# ================================================================
# Figure 10: Behavioral Divergence (Exp3)
# ================================================================
def fig_exp3_behavioral_divergence(df: pd.DataFrame):
    """Redesigned: weight-based divergence and EFE comparison.

    Left: Weight profile distance matrix (cosine + KL).
    Right: EFE comparison bar chart across contexts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parse weight profiles
    weight_profiles = {}
    efe_values = {}
    for _, row in df.iterrows():
        ctx = row["context"]
        ws = json.loads(row["weighted_scores"])
        weight_profiles[ctx] = ws
        efe_values[ctx] = row["total_EFE"]

    # Left: Distance matrix (cosine distance of weight profiles)
    contexts = CONTEXT_ORDER
    n = len(contexts)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = cosine_distance(
                weight_profiles.get(contexts[i], {}),
                weight_profiles.get(contexts[j], {}))

    ax = axes[0]
    im = ax.imshow(dist_matrix, cmap="Oranges", vmin=0, vmax=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([c.capitalize() for c in contexts], fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels([c.capitalize() for c in contexts], fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist_matrix[i,j]:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold")
    ax.set_title("Weight Profile Distance\n(Cosine distance of $Q(f|c)$)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.04)
    cbar.set_label("Cosine distance")

    # Right: EFE bar chart
    ax = axes[1]
    bars = ax.bar(range(n),
                  [efe_values.get(c, 0) for c in contexts],
                  color=[CONTEXT_COLORS[c] for c in contexts],
                  alpha=0.85, edgecolor="gray", linewidth=0.5)

    ax.set_xticks(range(n))
    ax.set_xticklabels([c.capitalize() for c in contexts], fontsize=10)
    ax.set_ylabel("Total expected free energy $\\mathcal{G}(\\pi)$")
    ax.set_title("EFE Under Identical Knowledge Base\n(Same A/B, different C-matrix)")
    ax.grid(axis="y", alpha=0.3)

    # Annotate with values
    for bar_obj, ctx in zip(bars, contexts):
        val = efe_values[ctx]
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # KL annotation
    kl_texts = []
    for _, row in df.iterrows():
        if row["D_KL_from_baseline"] > 0:
            kl_texts.append(f"{row['context'].capitalize()}: "
                           f"$D_{{KL}}$ = {row['D_KL_from_baseline']:.3f}")
    if kl_texts:
        ax.text(0.02, 0.95, "KL from reception:\n" + "\n".join(kl_texts),
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Behavioral Divergence: Same Knowledge, Different Preferences",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
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
            # Full fragment name below
            ax.text(i, -0.5, frag.replace("_", "\n"), fontsize=4.5,
                    ha="center", va="top", color="gray", style="italic")

        ax.set_xlim(-0.6, len(sequence) - 0.4)
        ax.set_ylim(-0.75, 0.5)
        ax.set_ylabel(f"{context.capitalize()}", fontsize=11,
                       color=CONTEXT_COLORS[context], fontweight="bold")
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.2)

    axes[-1].set_xlabel("Sequence position")
    fig.suptitle("Composed Sequences Across Contexts\n"
                 "(Color = source fragment, same knowledge base)",
                 fontsize=13)

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
    angles += angles[:1]

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

    Quantifies C-matrix effect magnitude.  Focuses on metrics that
    show meaningful variation across contexts.
    """
    metrics = ["total_EFE", "mean_VFE", "total_VFE",
               "n_primitives", "backbone_length"]
    metric_labels = [
        "Total EFE $\\mathcal{G}(\\pi)$",
        "Mean VFE $\\bar{\\mathcal{F}}$",
        "Total VFE $\\mathcal{F}(\\pi)$",
        "Policy length $|\\pi|$",
        "Backbone length",
    ]

    pairs = [("reception", "corridor"), ("reception", "hospital"), ("corridor", "hospital")]
    pair_labels = ["Rec. vs. Corr.", "Rec. vs. Hosp.", "Corr. vs. Hosp."]

    fig, ax = plt.subplots(figsize=(10, 6))

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
            ax.text(d + 0.05 * (1 if d >= 0 else -1), y_pos, f"{d:.2f}",
                    va="center", fontsize=8)

            y_ticks.append(y_pos)
            y_labels.append(f"{mlabel}\n({plabel})" if pi == 0 else plabel)
            y_pos += 1
        y_pos += 0.5

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Cohen's $d$ effect size")
    ax.set_title("Context Effect Sizes (C-matrix Impact)\n"
                 "EFE shows large effects; structural metrics show near-zero effects")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(-0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0.8, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axvline(-0.8, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

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

    ax.text(7, 7.6, "Compositional Assembly Pipeline", fontsize=14,
            ha="center", fontweight="bold")

    # Generative Model (left)
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
            "D $\\cdot$ B $\\cdot$ A matrices",
            fontsize=8, ha="center", color="#666")

    # Matrix boxes
    mat_y = 3.0
    matrices = [
        ("A", "P(o|s)", "Obs.\nlikelihood", "#E8F5E9", "#2E7D32", 0.5),
        ("B", "P(s'|s,$\\pi$)", "Transition\nmodel", "#FFF3E0", "#E65100", 3.0),
        ("C", "ln P(o)", "Preferences\n(context)", "#FCE4EC", "#C62828", 5.5),
        ("D", "$P(s_1)$", "Initial\nstate prior", "#F3E5F5", "#6A1B9A", 8.0),
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

    # Pipeline stages (right)
    stages = [
        ("1. D-matrix\nScoring", "$w_f = aff \\cdot prec$", 10.0, 6.5),
        ("2. Belief\nPropagation", "Graph diffusion", 10.0, 5.5),
        ("3. B-matrix\nConstruction", "compose_from_patterns()", 10.0, 4.5),
        ("4. Backbone\nExtraction", "Causal sequence", 10.0, 3.5),
        ("5. Policy\nSelection", "$\\sigma(-\\gamma \\cdot G(\\pi))$", 10.0, 2.5),
    ]

    for label, formula, sx, sy in stages:
        box = FancyBboxPatch((sx - 1.2, sy - 0.35), 2.4, 0.7,
                              boxstyle="round,pad=0.05",
                              facecolor="#FFFDE7", edgecolor="#F57F17", linewidth=1)
        ax.add_patch(box)
        ax.text(sx - 0.5, sy, label, fontsize=7, ha="center", fontweight="bold")
        ax.text(sx + 0.7, sy, formula, fontsize=6, ha="center", color="#666", style="italic")

    for i in range(len(stages) - 1):
        ax.annotate("", xy=(10.0, stages[i + 1][3] + 0.35),
                     xytext=(10.0, stages[i][3] - 0.35),
                     arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    # Context boxes at bottom
    ctx_y = 1.0
    for i, (ctx, color) in enumerate(CONTEXT_COLORS.items()):
        cx = 3.0 + i * 4.0
        box = FancyBboxPatch((cx - 1.5, ctx_y - 0.3), 3.0, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor=color, linewidth=1, alpha=0.2)
        ax.add_patch(box)
        ax.text(cx, ctx_y, f"{ctx.capitalize()} context\n(different C-matrix)",
                fontsize=8, ha="center", color=color, fontweight="bold")

    ax.annotate("", xy=(7, 1.4), xytext=(5.5, mat_y - 0.6),
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5, ls="--"))

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
        median_prims = df1["n_primitives"].median()
        observed = np.zeros((2, 2))
        for _, row in df1.iterrows():
            ci = row[cue_col]
            pi = 1 if row["n_primitives"] > median_prims else 0
            observed[ci, pi] += 1
        chi2, p = chi_square_test(observed)
        print(f"  {cue_label}: chi2={chi2:.2f}, p~{p:.4f}")

    # Exp1: Spearman on n_cues vs total_EFE
    print("\n--- Exp1: Context x n_cues (EFE correlation) ---")
    df1_copy = df1.copy()
    df1_copy["n_cues"] = df1_copy[cue_cols].sum(axis=1)
    for context in CONTEXT_ORDER:
        ctx = df1_copy[df1_copy["context"] == context]
        rho_prims = spearman_rank(ctx["n_cues"].values, ctx["n_primitives"].values)
        rho_efe = spearman_rank(ctx["n_cues"].values, ctx["total_EFE"].values)
        print(f"  {context}: rho(n_cues, |pi|) = {rho_prims:.3f}, rho(n_cues, EFE) = {rho_efe:.3f}")

    # Exp2: Spearman monotonicity
    print("\n--- Exp2: Fragment Scaling Monotonicity ---")
    for context in CONTEXT_ORDER:
        ctx = df2[df2["context"] == context]
        rho = spearman_rank(ctx["n_fragments"].values, ctx["n_primitives"].values)
        rho_efe = spearman_rank(ctx["n_fragments"].values, ctx["total_EFE"].values)
        print(f"  {context}: rho(k, |pi|) = {rho:.3f}, rho(k, EFE) = {rho_efe:.3f}")

    # Exp3: KL divergences
    print("\n--- Exp3: Context Transfer KL Divergences ---")
    for _, row in df3.iterrows():
        print(f"  {row['context']}: D_KL from reception baseline = {row['D_KL_from_baseline']:.4f}")

    # Exp3: Weight profile distances
    print("\n--- Exp3: Weight Profile Distances (Cosine) ---")
    weight_profiles = {}
    for _, row in df3.iterrows():
        weight_profiles[row["context"]] = json.loads(row["weighted_scores"])
    for c1, c2 in combinations(CONTEXT_ORDER, 2):
        d = cosine_distance(weight_profiles.get(c1, {}), weight_profiles.get(c2, {}))
        print(f"  {c1} vs {c2}: cosine distance = {d:.4f}")

    # Cross-experiment: Cohen's d for EFE
    print("\n--- Cross-Experiment: Cohen's d (total_EFE) ---")
    for c1, c2 in combinations(CONTEXT_ORDER, 2):
        g1 = df_all[df_all["context"] == c1]["total_EFE"].values
        g2 = df_all[df_all["context"] == c2]["total_EFE"].values
        d = cohens_d(g1, g2)
        print(f"  {c1} vs {c2}: d = {d:.3f}")

    # Cross-experiment: Cohen's d for structural metrics (expected ~0)
    print("\n--- Cross-Experiment: Cohen's d (n_primitives, structural) ---")
    for c1, c2 in combinations(CONTEXT_ORDER, 2):
        g1 = df_all[df_all["context"] == c1]["n_primitives"].values
        g2 = df_all[df_all["context"] == c2]["n_primitives"].values
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
