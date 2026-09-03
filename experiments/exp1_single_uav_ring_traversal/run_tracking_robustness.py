#!/usr/bin/env python3
"""Stress-test planned trajectories under smooth bounded tracking errors."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safe_game_flow.evaluation import evaluate_ring_trajectory  # noqa: E402


def _smooth_bounded_error(
    rng: np.random.Generator, horizon: int, maximum_amplitude: float
) -> np.ndarray:
    if maximum_amplitude == 0:
        return np.zeros((horizon, 3))
    raw = rng.normal(size=(horizon, 3))
    radius = max(3, horizon // 20)
    grid = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (grid / max(radius / 2, 1)) ** 2)
    kernel /= kernel.sum()
    smooth = np.stack(
        [np.convolve(raw[:, axis], kernel, mode="same") for axis in range(3)], axis=1
    )
    phase = np.linspace(0.0, 1.0, horizon)
    smooth *= np.sin(np.pi * phase)[:, None] ** 2
    maximum_norm = np.max(np.linalg.norm(smooth, axis=1))
    if maximum_norm < 1e-12:
        return np.zeros_like(smooth)
    return maximum_amplitude * smooth / maximum_norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--error-levels", type=str, default="0,0.005,0.01,0.02,0.04,0.06,0.08"
    )
    parser.add_argument("--planner-seeds", type=str, default="401,402,403")
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    sweep_dir = args.sweep_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = [float(value) for value in args.error_levels.split(",")]
    planner_seeds = [int(value) for value in args.planner_seeds.split(",")]

    archive = np.load(benchmark_dir / "data" / "test" / "conditions.npz")
    try:
        conditions = np.asarray(archive["values"])
        schema = tuple(str(item) for item in archive["schema"])
    finally:
        archive.close()

    method_paths = {
        "quintic_skeleton": lambda seed: benchmark_dir
        / "generated"
        / "quintic_skeleton"
        / "trajectories.npy",
        "projected_velocity_steps8": lambda seed: sweep_dir
        / "runs"
        / "projected_velocity"
        / "steps_8"
        / f"seed_{seed}"
        / "trajectories.npy",
        "proposed_steps1": lambda seed: sweep_dir
        / "runs"
        / "proposed_trajectory"
        / "steps_1"
        / f"seed_{seed}"
        / "trajectories.npy",
        "proposed_steps8": lambda seed: sweep_dir
        / "runs"
        / "proposed_trajectory"
        / "steps_8"
        / f"seed_{seed}"
        / "trajectories.npy",
        "proposed_robust_steps1": lambda seed: output_dir
        / "robust_plans"
        / "proposed_steps1"
        / f"seed_{seed}"
        / "trajectories.npy",
        "proposed_robust_steps8": lambda seed: output_dir
        / "robust_plans"
        / "proposed_steps8"
        / f"seed_{seed}"
        / "trajectories.npy",
    }

    run_rows = []
    for method_name, path_function in method_paths.items():
        for planner_seed in planner_seeds:
            path = path_function(planner_seed)
            trajectories = np.transpose(np.load(path), (0, 2, 1))
            if len(trajectories) != len(conditions):
                raise ValueError(f"Trajectory/condition mismatch for {path}")
            for level in levels:
                rng = np.random.default_rng(10_000 + planner_seed + int(level * 1e6))
                metrics = []
                for trajectory, condition in zip(trajectories, conditions, strict=True):
                    disturbed = trajectory + _smooth_bounded_error(
                        rng, len(trajectory), level
                    )
                    metrics.append(
                        evaluate_ring_trajectory(disturbed, condition, schema)
                    )
                run_rows.append(
                    {
                        "method": method_name,
                        "planner_seed": planner_seed,
                        "maximum_tracking_error_m": level,
                        "ring_pass_rate": float(np.mean([item.passed_ring for item in metrics])),
                        "collision_free_rate": float(
                            np.mean([item.collision_free for item in metrics])
                        ),
                        "dynamic_feasibility_rate": float(
                            np.mean([item.dynamically_feasible for item in metrics])
                        ),
                        "mean_minimum_barrier": float(
                            np.mean([item.minimum_barrier for item in metrics])
                        ),
                        "minimum_barrier": float(
                            np.min([item.minimum_barrier for item in metrics])
                        ),
                    }
                )

    with (output_dir / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["method"], row["maximum_tracking_error_m"])].append(row)
    aggregate = []
    for (method, level), group in grouped.items():
        row = {"method": method, "maximum_tracking_error_m": level}
        for metric in (
            "ring_pass_rate",
            "collision_free_rate",
            "dynamic_feasibility_rate",
            "mean_minimum_barrier",
            "minimum_barrier",
        ):
            values = np.asarray([item[metric] for item in group])
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        aggregate.append(row)
    aggregate.sort(key=lambda item: (item["method"], item["maximum_tracking_error_m"]))
    with (output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = (
        ("ring_pass_rate", "Ring pass rate"),
        ("collision_free_rate", "Collision-free rate"),
        ("mean_minimum_barrier", "Mean minimum barrier"),
    )
    for method in method_paths:
        method_rows = [row for row in aggregate if row["method"] == method]
        x = [100.0 * row["maximum_tracking_error_m"] for row in method_rows]
        for axis, (metric, label) in zip(axes, specs, strict=True):
            y = [row[f"{metric}_mean"] for row in method_rows]
            error = [row[f"{metric}_std"] for row in method_rows]
            axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=method)
            axis.set_xlabel("Maximum smooth tracking error (cm)")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("Trajectory-level robustness to bounded smooth tracking error")
    figure.tight_layout(rect=(0, 0.15, 1, 0.95))
    figure.savefig(output_dir / "tracking_error_robustness.png", dpi=220)
    plt.close(figure)
    print(f"Tracking robustness results saved to: {output_dir}")


if __name__ == "__main__":
    main()
