"""Stress test: sweep alpha combinations and report outcomes.

Runs headless Webots simulations with different empathy (alpha) values
for each robot and records time-to-goal, minimum inter-agent distance,
and collision count.

Usage:
    python scripts/stress_test.py --world worlds/tiago_warehouse.wbt
    python scripts/stress_test.py --sweep          # full alpha grid
    python scripts/stress_test.py --quick           # subset for CI
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
ALPHA_QUICK = [0.0, 2.0, 6.0]
DEFAULT_TIMEOUT = 120  # seconds per simulation


@dataclass
class RunResult:
    alpha_1: float
    alpha_2: float
    time_to_goal_1: float = -1.0
    time_to_goal_2: float = -1.0
    min_distance: float = float("inf")
    collisions: int = 0
    both_reached: bool = False
    timed_out: bool = False


def build_custom_data(goal_x: float, goal_y: float, alpha: float, agent_id: int) -> str:
    return f"{goal_x},{goal_y},{alpha},{agent_id}"


def run_simulation(
    world_path: str,
    alpha_1: float,
    alpha_2: float,
    webots_bin: str = "webots",
    timeout: int = DEFAULT_TIMEOUT,
) -> RunResult:
    """Run a single Webots simulation and parse the console output."""
    result = RunResult(alpha_1=alpha_1, alpha_2=alpha_2)

    env = os.environ.copy()
    env["TIAGO_1_CUSTOM"] = build_custom_data(3.0, 0.0, alpha_1, 0)
    env["TIAGO_2_CUSTOM"] = build_custom_data(-3.0, 0.0, alpha_2, 1)

    cmd = [
        webots_bin,
        "--mode=fast",
        "--no-rendering",
        "--minimize",
        f"--stdout",
        f"--stderr",
        world_path,
    ]

    print(f"  Running alpha=({alpha_1:.1f}, {alpha_2:.1f}) ...", end=" ", flush=True)
    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed = time.time() - start
        output = proc.stdout + proc.stderr

        # Parse output for results
        for line in output.splitlines():
            if "SUCCESS - Goal reached" in line:
                if "TIAGo_1" in line:
                    result.time_to_goal_1 = elapsed
                elif "TIAGo_2" in line:
                    result.time_to_goal_2 = elapsed

        result.both_reached = result.time_to_goal_1 > 0 and result.time_to_goal_2 > 0
        print(f"done ({elapsed:.1f}s)")

    except subprocess.TimeoutExpired:
        result.timed_out = True
        print(f"TIMEOUT ({timeout}s)")
    except FileNotFoundError:
        print(f"ERROR: '{webots_bin}' not found. Set --webots-bin or add to PATH.")
        sys.exit(1)

    return result


def run_sweep(
    world_path: str,
    alphas: List[float],
    webots_bin: str,
    timeout: int,
) -> List[RunResult]:
    """Run all alpha combinations."""
    results: List[RunResult] = []
    pairs = list(itertools.product(alphas, repeat=2))

    print(f"Running {len(pairs)} alpha combinations...")
    for a1, a2 in pairs:
        r = run_simulation(world_path, a1, a2, webots_bin, timeout)
        results.append(r)

    return results


def print_summary(results: List[RunResult]) -> None:
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    print(f"{'alpha_1':>8} {'alpha_2':>8} {'t_goal_1':>10} {'t_goal_2':>10} "
          f"{'min_dist':>10} {'collisions':>10} {'both':>6} {'timeout':>8}")
    print("-" * 70)
    for r in results:
        t1 = f"{r.time_to_goal_1:.1f}" if r.time_to_goal_1 > 0 else "N/A"
        t2 = f"{r.time_to_goal_2:.1f}" if r.time_to_goal_2 > 0 else "N/A"
        md = f"{r.min_distance:.2f}" if r.min_distance < float("inf") else "N/A"
        print(f"{r.alpha_1:>8.1f} {r.alpha_2:>8.1f} {t1:>10} {t2:>10} "
              f"{md:>10} {r.collisions:>10} {'Y' if r.both_reached else 'N':>6} "
              f"{'Y' if r.timed_out else 'N':>8}")
    print("=" * 70)


def save_csv(results: List[RunResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "alpha_1", "alpha_2", "time_to_goal_1", "time_to_goal_2",
            "min_distance", "collisions", "both_reached", "timed_out",
        ])
        for r in results:
            writer.writerow([
                r.alpha_1, r.alpha_2, r.time_to_goal_1, r.time_to_goal_2,
                r.min_distance, r.collisions, r.both_reached, r.timed_out,
            ])
    print(f"Results saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Social-layer stress test")
    parser.add_argument("--world", default="worlds/tiago_warehouse.wbt",
                        help="Path to Webots world file")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full alpha grid sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick subset of alphas")
    parser.add_argument("--alpha1", type=float, default=0.0,
                        help="Alpha for TIAGo 1 (single run)")
    parser.add_argument("--alpha2", type=float, default=6.0,
                        help="Alpha for TIAGo 2 (single run)")
    parser.add_argument("--webots-bin", default="webots",
                        help="Path to Webots executable")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Timeout per simulation in seconds")
    parser.add_argument("--csv", default="stress_test_results.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    world_path = str(Path(args.world).resolve())

    if args.sweep:
        results = run_sweep(world_path, ALPHA_VALUES, args.webots_bin, args.timeout)
    elif args.quick:
        results = run_sweep(world_path, ALPHA_QUICK, args.webots_bin, args.timeout)
    else:
        results = [run_simulation(world_path, args.alpha1, args.alpha2,
                                  args.webots_bin, args.timeout)]

    print_summary(results)
    save_csv(results, args.csv)


if __name__ == "__main__":
    main()
