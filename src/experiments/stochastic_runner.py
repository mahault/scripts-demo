"""Stochastic multi-trial experiment runner.

Runs N trials per condition with different random seeds and noise levels,
producing genuinely variable samples for meaningful statistical analysis.

Supports:
- Multiple noise levels (sigma=0.05, 0.1, 0.2)
- Configurable number of trials per condition
- Parallel-ready structure (conditions are independent)
- Integration with baselines for comparison
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class TrialResult:
    """Result from one trial of one condition."""
    condition_id: str
    trial_idx: int
    seed: int
    noise_std: float
    context: str
    # Core metrics
    n_active_fragments: int
    n_primitives: int
    total_vfe: float
    mean_vfe: float
    total_efe: float
    # Perception
    inferred_context: str
    belief_confidence: float
    belief_entropy: float
    # Topology
    topology_density: float
    # Sequence info
    sequence: str
    primitive_set: str


@dataclass
class MultiTrialResult:
    """Aggregated results from multiple trials of one condition."""
    condition_id: str
    context: str
    noise_std: float
    n_trials: int
    # Aggregated metrics (mean +/- std)
    vfe_mean: float
    vfe_std: float
    efe_mean: float
    efe_std: float
    n_primitives_mean: float
    n_primitives_std: float
    confidence_mean: float
    confidence_std: float
    # Recognition accuracy across trials
    recognition_accuracy: float
    # Individual trial results
    trials: List[TrialResult] = field(default_factory=list)


@dataclass
class StochasticExperimentConfig:
    """Configuration for stochastic experiment runs."""
    n_trials: int = 50
    noise_levels: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    base_seed: int = 12345
    include_baselines: bool = True
    output_dir: str = "experiment_results/stochastic"


class StochasticRunner:
    """Runs multi-trial experiments with stochastic variation.

    Parameters
    ----------
    run_fn : callable
        Function that runs one condition and returns metrics.
        Signature: run_fn(condition, noise_std, seed) -> dict
    config : StochasticExperimentConfig
        Experiment configuration.
    """

    def __init__(
        self,
        run_fn: Callable[..., Dict[str, Any]],
        config: Optional[StochasticExperimentConfig] = None,
    ) -> None:
        self._run_fn = run_fn
        self._config = config or StochasticExperimentConfig()
        self._rng = np.random.default_rng(self._config.base_seed)
        self._results: List[MultiTrialResult] = []

    @property
    def results(self) -> List[MultiTrialResult]:
        return list(self._results)

    def run_condition(
        self,
        condition: Any,
        condition_id: str,
        context: str,
        noise_std: float,
        n_trials: Optional[int] = None,
    ) -> MultiTrialResult:
        """Run multiple trials for one condition at one noise level.

        Parameters
        ----------
        condition : Any
            Condition object to pass to run_fn.
        condition_id : str
            Unique identifier for this condition.
        context : str
            Ground-truth context label.
        noise_std : float
            Noise standard deviation for this set of trials.
        n_trials : int | None
            Override number of trials (uses config default if None).

        Returns
        -------
        MultiTrialResult with all trial outcomes.
        """
        trials_n = n_trials or self._config.n_trials
        trials: List[TrialResult] = []

        # Generate unique seeds for each trial
        seeds = [int(self._rng.integers(0, 2**31)) for _ in range(trials_n)]

        for trial_idx, seed in enumerate(seeds):
            try:
                result_dict = self._run_fn(
                    condition,
                    noise_std=noise_std,
                    seed=seed,
                )

                trial = TrialResult(
                    condition_id=condition_id,
                    trial_idx=trial_idx,
                    seed=seed,
                    noise_std=noise_std,
                    context=context,
                    n_active_fragments=result_dict.get("n_active_fragments", 0),
                    n_primitives=result_dict.get("n_primitives", 0),
                    total_vfe=result_dict.get("total_VFE", 0.0),
                    mean_vfe=result_dict.get("mean_VFE", 0.0),
                    total_efe=result_dict.get("total_EFE", 0.0),
                    inferred_context=result_dict.get("inferred_context", ""),
                    belief_confidence=result_dict.get("belief_confidence", 0.0),
                    belief_entropy=result_dict.get("belief_entropy", 0.0),
                    topology_density=result_dict.get("topology_density", 0.0),
                    sequence=result_dict.get("sequence", ""),
                    primitive_set=result_dict.get("primitive_set", ""),
                )
                trials.append(trial)
            except Exception as e:
                # Record failed trial with zeros
                trials.append(TrialResult(
                    condition_id=condition_id,
                    trial_idx=trial_idx,
                    seed=seed,
                    noise_std=noise_std,
                    context=context,
                    n_active_fragments=0,
                    n_primitives=0,
                    total_vfe=0.0,
                    mean_vfe=0.0,
                    total_efe=0.0,
                    inferred_context="",
                    belief_confidence=0.0,
                    belief_entropy=0.0,
                    topology_density=0.0,
                    sequence="",
                    primitive_set="",
                ))

        # Aggregate
        multi = self._aggregate_trials(condition_id, context, noise_std, trials)
        self._results.append(multi)
        return multi

    def run_all_conditions(
        self,
        conditions: List[Tuple[Any, str, str]],
        noise_levels: Optional[List[float]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[MultiTrialResult]:
        """Run all conditions across all noise levels.

        Parameters
        ----------
        conditions : list[(condition, condition_id, context)]
            List of conditions to run.
        noise_levels : list[float] | None
            Noise levels (uses config default if None).
        progress_callback : callable | None
            Called with (completed, total) for progress tracking.

        Returns
        -------
        List of MultiTrialResult for all condition x noise combinations.
        """
        levels = noise_levels or self._config.noise_levels
        total = len(conditions) * len(levels)
        completed = 0

        all_results: List[MultiTrialResult] = []

        for noise_std in levels:
            for condition, cond_id, context in conditions:
                result = self.run_condition(
                    condition, cond_id, context, noise_std
                )
                all_results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return all_results

    def _aggregate_trials(
        self,
        condition_id: str,
        context: str,
        noise_std: float,
        trials: List[TrialResult],
    ) -> MultiTrialResult:
        """Compute summary statistics over trials."""
        n = len(trials)
        if n == 0:
            return MultiTrialResult(
                condition_id=condition_id,
                context=context,
                noise_std=noise_std,
                n_trials=0,
                vfe_mean=0.0, vfe_std=0.0,
                efe_mean=0.0, efe_std=0.0,
                n_primitives_mean=0.0, n_primitives_std=0.0,
                confidence_mean=0.0, confidence_std=0.0,
                recognition_accuracy=0.0,
                trials=[],
            )

        vfe_vals = np.array([t.total_vfe for t in trials])
        efe_vals = np.array([t.total_efe for t in trials])
        n_prim_vals = np.array([float(t.n_primitives) for t in trials])
        conf_vals = np.array([t.belief_confidence for t in trials])

        # Recognition accuracy
        correct = sum(1 for t in trials if t.inferred_context == context)
        rec_acc = correct / n

        return MultiTrialResult(
            condition_id=condition_id,
            context=context,
            noise_std=noise_std,
            n_trials=n,
            vfe_mean=float(np.mean(vfe_vals)),
            vfe_std=float(np.std(vfe_vals)),
            efe_mean=float(np.mean(efe_vals)),
            efe_std=float(np.std(efe_vals)),
            n_primitives_mean=float(np.mean(n_prim_vals)),
            n_primitives_std=float(np.std(n_prim_vals)),
            confidence_mean=float(np.mean(conf_vals)),
            confidence_std=float(np.std(conf_vals)),
            recognition_accuracy=rec_acc,
            trials=trials,
        )

    def save_results(self, filepath: Optional[str] = None) -> str:
        """Save all results to JSON.

        Returns the filepath where results were saved.
        """
        if filepath is None:
            os.makedirs(self._config.output_dir, exist_ok=True)
            filepath = os.path.join(
                self._config.output_dir,
                f"stochastic_results_{int(time.time())}.json",
            )

        data = []
        for multi in self._results:
            entry = {
                "condition_id": multi.condition_id,
                "context": multi.context,
                "noise_std": multi.noise_std,
                "n_trials": multi.n_trials,
                "vfe_mean": multi.vfe_mean,
                "vfe_std": multi.vfe_std,
                "efe_mean": multi.efe_mean,
                "efe_std": multi.efe_std,
                "n_primitives_mean": multi.n_primitives_mean,
                "n_primitives_std": multi.n_primitives_std,
                "confidence_mean": multi.confidence_mean,
                "confidence_std": multi.confidence_std,
                "recognition_accuracy": multi.recognition_accuracy,
                "trials": [
                    {
                        "trial_idx": t.trial_idx,
                        "seed": t.seed,
                        "total_vfe": t.total_vfe,
                        "total_efe": t.total_efe,
                        "n_primitives": t.n_primitives,
                        "inferred_context": t.inferred_context,
                        "belief_confidence": t.belief_confidence,
                    }
                    for t in multi.trials
                ],
            }
            data.append(entry)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return filepath
