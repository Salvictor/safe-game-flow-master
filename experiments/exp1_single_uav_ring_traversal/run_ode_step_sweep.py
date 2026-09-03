#!/usr/bin/env python3
"""Measure trajectory/velocity planning quality and latency versus ODE steps."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        completed.check_returncode()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--seeds", type=str, default="401,402,403")
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(item) for item in args.steps.split(",")]
    seeds = [int(item) for item in args.seeds.split(",")]
    if any(item < 1 for item in steps):
        raise ValueError("ODE steps must be positive")

    models = benchmark_dir / "models"
    conditions = benchmark_dir / "data" / "test" / "conditions.npz"
    expert_trajectories = np.transpose(
        np.load(benchmark_dir / "data" / "test" / "trajectories.npy"), (0, 2, 1)
    )
    methods = {
        "raw_trajectory": (
            models / "raw_position_cfm" / "checkpoint.pt",
            ["--disable-cbf", "--unsafe-gaussian-prior"],
        ),
        "boundary_trajectory": (
            models / "boundary_residual_cfm" / "checkpoint.pt",
            ["--disable-cbf", "--unsafe-gaussian-prior"],
        ),
        "raw_velocity": (
            models / "velocity_cfm" / "checkpoint.pt",
            ["--disable-cbf", "--unsafe-gaussian-prior"],
        ),
        "projected_velocity": (
            models / "velocity_cfm" / "checkpoint.pt",
            ["--disable-cbf", "--unsafe-gaussian-prior", "--project-velocity-boundary"],
        ),
        "proposed_trajectory": (
            models / "boundary_residual_cfm" / "checkpoint.pt",
            ["--kinodynamic-project"],
        ),
    }
    sampler = str(EXPERIMENT_DIR / "sample_conditional_fm.py")
    evaluator = str(EXPERIMENT_DIR / "evaluate_trajectories.py")
    python = sys.executable
    rows: list[dict] = []

    for method_name, (checkpoint, flags) in methods.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        for ode_steps in steps:
            for seed in seeds:
                print(
                    f"Running method={method_name}, ode_steps={ode_steps}, seed={seed}",
                    flush=True,
                )
                run_dir = output_dir / "runs" / method_name / f"steps_{ode_steps}" / f"seed_{seed}"
                eval_dir = output_dir / "evaluation" / method_name / f"steps_{ode_steps}" / f"seed_{seed}"
                command = [
                    python,
                    sampler,
                    "--checkpoint",
                    str(checkpoint),
                    "--conditions",
                    str(conditions),
                    "--output-dir",
                    str(run_dir),
                    "--num-conditions",
                    str(args.test_size),
                    "--ode-steps",
                    str(ode_steps),
                    "--seed",
                    str(seed),
                    *flags,
                ]
                if args.cpu:
                    command.append("--cpu")
                if not (run_dir / "sampling_summary.json").is_file():
                    _run(command)
                if not (eval_dir / "summary.json").is_file():
                    _run(
                        [
                            python,
                            evaluator,
                            "--trajectories",
                            str(run_dir / "trajectories.npy"),
                            "--conditions",
                            str(run_dir / "conditions_used.npz"),
                            "--output-dir",
                            str(eval_dir),
                        ]
                    )
                metrics = _load(eval_dir / "summary.json")
                timing = _load(run_dir / "sampling_summary.json")
                generated = np.transpose(np.load(run_dir / "trajectories.npy"), (0, 2, 1))
                rows.append(
                    {
                        "method": method_name,
                        "ode_steps": ode_steps,
                        "seed": seed,
                        **metrics,
                        "trajectory_rmse_to_expert": float(
                            np.sqrt(np.mean((generated - expert_trajectories) ** 2))
                        ),
                        "ode_ms_per_trajectory": timing["ode_milliseconds_per_trajectory"],
                        "end_to_end_ms_per_trajectory": timing[
                            "end_to_end_milliseconds_per_trajectory"
                        ],
                        "mean_residual_scale": timing["mean_residual_scale"],
                        "cbf_intervention_rate": (
                            timing["cbf_projection"]["intervention_rate"]
                            if timing["cbf_projection"] is not None
                            else 0.0
                        ),
                    }
                )

    with (output_dir / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["ode_steps"])].append(row)
    metric_names = [
        "success_rate",
        "ring_pass_rate",
        "collision_free_rate",
        "dynamic_feasibility_rate",
        "endpoint_feasibility_rate",
        "mean_maximum_speed",
        "mean_maximum_acceleration",
        "mean_path_length",
        "trajectory_rmse_to_expert",
        "end_to_end_ms_per_trajectory",
        "ode_ms_per_trajectory",
        "mean_residual_scale",
        "cbf_intervention_rate",
    ]
    aggregate = []
    for (method, ode_steps), group in grouped.items():
        result = {"method": method, "ode_steps": ode_steps, "num_seeds": len(group)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group])
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(result)
    aggregate.sort(key=lambda item: (item["method"], item["ode_steps"]))
    with (output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_specs = (
        ("end_to_end_ms_per_trajectory_mean", "End-to-end planning time (ms)"),
        ("success_rate_mean", "Task success rate"),
        ("dynamic_feasibility_rate_mean", "Dynamic feasibility rate"),
        ("mean_maximum_acceleration_mean", "Mean maximum acceleration (m/s²)"),
    )
    for method in methods:
        method_rows = [item for item in aggregate if item["method"] == method]
        x = [item["ode_steps"] for item in method_rows]
        for axis, (metric, label) in zip(axes.flat, plot_specs, strict=True):
            y = [item[metric] for item in method_rows]
            error = [item[metric.replace("_mean", "_std")] for item in method_rows]
            axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=method)
            axis.set_xscale("log", base=2)
            axis.set_xlabel("ODE integration steps")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[1, 0].set_ylim(-0.02, 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle("Trajectory and velocity planning versus ODE integration steps")
    figure.tight_layout(rect=(0, 0.08, 1, 0.97))
    figure.savefig(output_dir / "ode_step_tradeoffs.png", dpi=220)
    plt.close(figure)
    print(f"ODE-step sweep saved to: {output_dir}")


if __name__ == "__main__":
    main()
