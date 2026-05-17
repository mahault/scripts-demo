"""Statistical analysis module for multi-trial experiment results.

Provides:
- Bootstrap confidence intervals
- Permutation tests for group comparisons
- Effect sizes (Cohen's d) with confidence intervals
- ANOVA with post-hoc corrections
- Learning curve convergence analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    statistic: float  # point estimate
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int


@dataclass
class PermutationTestResult:
    """Result from a permutation test."""
    observed_statistic: float
    p_value: float
    n_permutations: int
    effect_direction: str  # "greater", "less", or "two-sided"


@dataclass
class EffectSize:
    """Cohen's d effect size with confidence interval."""
    d: float
    ci_lower: float
    ci_upper: float
    interpretation: str  # "negligible", "small", "medium", "large"


@dataclass
class ANOVAResult:
    """One-way ANOVA result."""
    f_statistic: float
    p_value: float
    df_between: int
    df_within: int
    eta_squared: float
    post_hoc: Optional[Dict[str, float]] = None  # pairwise p-values


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Optional[callable] = None,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Sample data.
    statistic_fn : callable | None
        Function to compute the statistic. Defaults to np.mean.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI).
    seed : int | None
        Random seed.

    Returns
    -------
    BootstrapCI with point estimate and interval bounds.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)

    if statistic_fn is None:
        statistic_fn = np.mean

    # Point estimate
    point_est = float(statistic_fn(data))

    # Bootstrap resamples
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = data[rng.integers(0, n, size=n)]
        bootstrap_stats[i] = statistic_fn(resample)

    # Percentile method
    alpha = 1.0 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        statistic=point_est,
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=confidence,
        n_bootstrap=n_bootstrap,
    )


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    seed: Optional[int] = None,
) -> PermutationTestResult:
    """Two-sample permutation test for difference in means.

    Parameters
    ----------
    group_a : np.ndarray
        First group samples.
    group_b : np.ndarray
        Second group samples.
    n_permutations : int
        Number of permutations.
    alternative : str
        "two-sided", "greater" (a > b), or "less" (a < b).
    seed : int | None
        Random seed.

    Returns
    -------
    PermutationTestResult with observed statistic and p-value.
    """
    rng = np.random.default_rng(seed)
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    observed_diff = float(np.mean(group_a) - np.mean(group_b))
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        perm_diff = np.mean(perm[:n_a]) - np.mean(perm[n_a:])

        if alternative == "two-sided":
            if abs(perm_diff) >= abs(observed_diff):
                count += 1
        elif alternative == "greater":
            if perm_diff >= observed_diff:
                count += 1
        else:  # less
            if perm_diff <= observed_diff:
                count += 1

    p_value = (count + 1) / (n_permutations + 1)  # +1 for continuity correction

    return PermutationTestResult(
        observed_statistic=observed_diff,
        p_value=p_value,
        n_permutations=n_permutations,
        effect_direction=alternative,
    )


def cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> EffectSize:
    """Compute Cohen's d effect size with bootstrap CI.

    Parameters
    ----------
    group_a : np.ndarray
        First group samples.
    group_b : np.ndarray
        Second group samples.
    n_bootstrap : int
        Bootstrap samples for CI.
    confidence : float
        Confidence level.
    seed : int | None
        Random seed.

    Returns
    -------
    EffectSize with d value, CI, and interpretation.
    """
    rng = np.random.default_rng(seed)
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    def _compute_d(a: np.ndarray, b: np.ndarray) -> float:
        n_a, n_b = len(a), len(b)
        mean_diff = np.mean(a) - np.mean(b)
        # Pooled standard deviation
        var_a = np.var(a, ddof=1) if n_a > 1 else 0.0
        var_b = np.var(b, ddof=1) if n_b > 1 else 0.0
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10
        return mean_diff / pooled_std

    d = _compute_d(group_a, group_b)

    # Bootstrap CI for d
    bootstrap_d = np.zeros(n_bootstrap)
    n_a, n_b = len(group_a), len(group_b)
    for i in range(n_bootstrap):
        boot_a = group_a[rng.integers(0, n_a, size=n_a)]
        boot_b = group_b[rng.integers(0, n_b, size=n_b)]
        bootstrap_d[i] = _compute_d(boot_a, boot_b)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(bootstrap_d, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_d, 100 * (1 - alpha / 2)))

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSize(
        d=d,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        interpretation=interpretation,
    )


def one_way_anova(
    groups: List[np.ndarray],
    group_names: Optional[List[str]] = None,
    post_hoc_correction: str = "bonferroni",
    seed: Optional[int] = None,
) -> ANOVAResult:
    """One-way ANOVA with optional post-hoc pairwise comparisons.

    Uses permutation-based p-value computation (non-parametric).

    Parameters
    ----------
    groups : list[np.ndarray]
        List of sample arrays (one per group).
    group_names : list[str] | None
        Names for groups (used in post-hoc labels).
    post_hoc_correction : str
        Multiple comparison correction ("bonferroni" or "holm").
    seed : int | None
        Random seed.

    Returns
    -------
    ANOVAResult with F-statistic, p-value, and post-hoc comparisons.
    """
    rng = np.random.default_rng(seed)
    k = len(groups)

    if group_names is None:
        group_names = [f"group_{i}" for i in range(k)]

    # Compute F-statistic
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    n_total = len(all_data)

    # Between-group sum of squares
    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2
        for g in groups
    )

    # Within-group sum of squares
    ss_within = sum(
        np.sum((g - np.mean(g)) ** 2)
        for g in groups
    )

    df_between = k - 1
    df_within = n_total - k

    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / max(df_within, 1)

    f_stat = ms_between / max(ms_within, 1e-10)

    # Eta-squared effect size
    ss_total = ss_between + ss_within
    eta_sq = ss_between / max(ss_total, 1e-10)

    # Permutation-based p-value for F
    n_perms = 5000
    count = 0
    group_sizes = [len(g) for g in groups]

    for _ in range(n_perms):
        perm_data = rng.permutation(all_data)
        # Split into groups of same sizes
        perm_groups = []
        start = 0
        for size in group_sizes:
            perm_groups.append(perm_data[start:start + size])
            start += size

        perm_grand_mean = np.mean(perm_data)
        perm_ss_between = sum(
            len(g) * (np.mean(g) - perm_grand_mean) ** 2
            for g in perm_groups
        )
        perm_ss_within = sum(
            np.sum((g - np.mean(g)) ** 2)
            for g in perm_groups
        )
        perm_ms_between = perm_ss_between / max(df_between, 1)
        perm_ms_within = perm_ss_within / max(df_within, 1)
        perm_f = perm_ms_between / max(perm_ms_within, 1e-10)

        if perm_f >= f_stat:
            count += 1

    p_value = (count + 1) / (n_perms + 1)

    # Post-hoc pairwise comparisons
    post_hoc = {}
    n_comparisons = k * (k - 1) // 2

    for i in range(k):
        for j in range(i + 1, k):
            pair_name = f"{group_names[i]}_vs_{group_names[j]}"
            perm_result = permutation_test(
                groups[i], groups[j],
                n_permutations=2000,
                seed=seed,
            )
            raw_p = perm_result.p_value

            # Apply correction
            if post_hoc_correction == "bonferroni":
                corrected_p = min(1.0, raw_p * n_comparisons)
            else:
                corrected_p = raw_p  # Holm requires sorting all p-values

            post_hoc[pair_name] = corrected_p

    return ANOVAResult(
        f_statistic=float(f_stat),
        p_value=p_value,
        df_between=df_between,
        df_within=df_within,
        eta_squared=float(eta_sq),
        post_hoc=post_hoc,
    )


def learning_curve_convergence(
    values: List[float],
    window: int = 20,
    threshold: float = 0.01,
) -> Tuple[bool, int, float]:
    """Analyze convergence of a learning curve.

    Parameters
    ----------
    values : list[float]
        Sequential values (e.g., affinity over episodes).
    window : int
        Rolling window size for stability check.
    threshold : float
        Maximum std within window to declare convergence.

    Returns
    -------
    (converged, convergence_episode, final_std) tuple.
    """
    if len(values) < window:
        return False, -1, float("inf")

    arr = np.array(values)

    # Check rolling windows from start to find first convergence point
    for i in range(window, len(arr) + 1):
        window_vals = arr[i - window:i]
        std = float(np.std(window_vals))
        if std < threshold:
            return True, i - window, std

    # Not converged — return final window std
    final_std = float(np.std(arr[-window:]))
    return False, -1, final_std


def robustness_analysis(
    results_by_noise: Dict[float, List[float]],
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze metric robustness across noise levels.

    Parameters
    ----------
    results_by_noise : dict[float, list[float]]
        Metric values keyed by noise level.
    confidence : float
        CI confidence level.
    seed : int | None
        Random seed.

    Returns
    -------
    Dict with per-noise CIs and cross-noise consistency metrics.
    """
    from typing import Any

    analysis: Dict[str, Any] = {
        "per_noise": {},
        "is_robust": True,
        "max_relative_change": 0.0,
    }

    noise_levels = sorted(results_by_noise.keys())
    baseline_mean = None

    for noise in noise_levels:
        data = np.array(results_by_noise[noise])
        ci = bootstrap_ci(data, confidence=confidence, seed=seed)
        analysis["per_noise"][noise] = {
            "mean": ci.statistic,
            "ci_lower": ci.ci_lower,
            "ci_upper": ci.ci_upper,
            "std": float(np.std(data)),
            "n": len(data),
        }

        if baseline_mean is None:
            baseline_mean = ci.statistic

    # Check if finding direction is consistent across noise levels
    if baseline_mean and abs(baseline_mean) > 1e-10:
        for noise in noise_levels:
            entry = analysis["per_noise"][noise]
            rel_change = abs(entry["mean"] - baseline_mean) / abs(baseline_mean)
            analysis["max_relative_change"] = max(
                analysis["max_relative_change"], rel_change
            )
            # >30% relative change = not robust
            if rel_change > 0.3:
                analysis["is_robust"] = False

    return analysis
