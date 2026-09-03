#!/usr/bin/env python3
"""Run matched trajectory-level component ablations for Experiment 2."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
EXP1_DIR = PROJECT_ROOT / "experiments" / "exp1_single_uav_ring_traversal"
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safe_game_flow.trajectories import quintic_boundary_skeleton  # noqa: E402


def _run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    benchmark = args.benchmark_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated_root = output / "generated"
    evaluation_root = output / "evaluation"
    conditions_path = benchmark / "data" / "test" / "conditions.npz"
    boundary_checkpoint = benchmark / "models" / "boundary_residual_cfm" / "checkpoint.pt"
    velocity_checkpoint = benchmark / "models" / "velocity_cfm" / "checkpoint.pt"
    sampler = EXP1_DIR / "sample_conditional_fm.py"
    evaluator = EXP1_DIR / "evaluate_trajectories.py"
    python = sys.executable

    methods = {
        "boundary_only": {
            "checkpoint": boundary_checkpoint,
            "flags": ["--disable-cbf", "--unsafe-gaussian-prior"],
        },
        "plus_safe_prior": {
            "checkpoint": boundary_checkpoint,
            "flags": ["--disable-cbf"],
        },
        "plus_cbf": {
            "checkpoint": boundary_checkpoint,
            "flags": [
                "--post-step-safety-correction",
                "--cbf-collocation-factor", "10",
            ],
        },
        "plus_kinodynamic": {
            "checkpoint": boundary_checkpoint,
            "flags": [
                "--kinodynamic-project",
                "--post-step-safety-correction",
                "--cbf-collocation-factor", "10",
            ],
        },
        "plus_execution_margin": {
            "checkpoint": boundary_checkpoint,
            "flags": [
                "--kinodynamic-project",
                "--post-step-safety-correction",
                "--cbf-collocation-factor", "10",
                "--maximum-speed", "1.95",
                "--maximum-acceleration", "2.0",
            ],
        },
        "projected_velocity_cfm": {
            "checkpoint": velocity_checkpoint,
            "flags": [
                "--disable-cbf",
                "--unsafe-gaussian-prior",
                "--project-velocity-boundary",
            ],
        },
    }

    archive = np.load(conditions_path)
    try:
        conditions = np.asarray(archive["values"], dtype=np.float32)[: args.test_size]
        schema = tuple(str(item) for item in archive["schema"])
    finally:
        archive.close()
    expert = np.transpose(
        np.load(benchmark / "data" / "test" / "trajectories.npy")[: args.test_size],
        (0, 2, 1),
    )
    rows: list[dict] = []

    skeleton_dir = generated_root / "quintic_skeleton"
    skeleton_eval = evaluation_root / "quintic_skeleton"
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    skeletons = quintic_boundary_skeleton(conditions, expert.shape[1], schema)
    np.save(skeleton_dir / "trajectories.npy", np.transpose(skeletons, (0, 2, 1)))
    np.savez_compressed(
        skeleton_dir / "conditions_used.npz", values=conditions, schema=np.asarray(schema)
    )
    _run(
        [
            python, str(evaluator),
            "--trajectories", str(skeleton_dir / "trajectories.npy"),
            "--conditions", str(skeleton_dir / "conditions_used.npz"),
            "--output-dir", str(skeleton_eval),
        ]
    )
    rows.append(
        {
            "method": "quintic_skeleton",
            **_load_json(skeleton_eval / "summary.json"),
            "trajectory_rmse_to_expert": float(np.sqrt(np.mean((skeletons - expert) ** 2))),
            "batch_end_to_end_ms_per_trajectory": 0.0,
            "cbf_intervention_rate": 0.0,
            "mean_residual_scale": 0.0,
        }
    )

    for method, configuration in methods.items():
        generated = generated_root / method
        evaluation = evaluation_root / method
        command = [
            python, str(sampler),
            "--checkpoint", str(configuration["checkpoint"]),
            "--conditions", str(conditions_path),
            "--output-dir", str(generated),
            "--num-conditions", str(args.test_size),
            "--ode-steps", str(args.ode_steps),
            "--seed", str(args.seed),
            *configuration["flags"],
        ]
        if args.cpu:
            command.append("--cpu")
        _run(command)
        _run(
            [
                python, str(evaluator),
                "--trajectories", str(generated / "trajectories.npy"),
                "--conditions", str(generated / "conditions_used.npz"),
                "--output-dir", str(evaluation),
            ]
        )
        summary = _load_json(evaluation / "summary.json")
        timing = _load_json(generated / "sampling_summary.json")
        trajectories = np.transpose(np.load(generated / "trajectories.npy"), (0, 2, 1))
        rows.append(
            {
                "method": method,
                **summary,
                "trajectory_rmse_to_expert": float(
                    np.sqrt(np.mean((trajectories - expert) ** 2))
                ),
                "batch_end_to_end_ms_per_trajectory": timing[
                    "end_to_end_milliseconds_per_trajectory"
                ],
                "cbf_intervention_rate": (
                    timing["cbf_projection"]["intervention_rate"]
                    if timing["cbf_projection"] is not None
                    else 0.0
                ),
                "mean_residual_scale": timing["mean_residual_scale"],
            }
        )

    order = [
        "quintic_skeleton",
        "projected_velocity_cfm",
        "boundary_only",
        "plus_safe_prior",
        "plus_cbf",
        "plus_kinodynamic",
        "plus_execution_margin",
    ]
    rows.sort(key=lambda item: order.index(item["method"]))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "configuration.json").write_text(
        json.dumps(
            {
                "benchmark_dir": str(benchmark),
                "ode_steps": args.ode_steps,
                "test_size": args.test_size,
                "seed": args.seed,
                "shared_boundary_checkpoint": str(boundary_checkpoint),
                "execution_margin_limits": {"speed": 1.95, "acceleration": 2.0},
                "physical_evaluation_limits": {"speed": 2.0, "acceleration": 3.0},
                "discrete_safety_correction": True,
                "cbf_collocation_factor": 10,
                "kinodynamic_projection": "analytic with nonconvex fallback",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    labels = [item["method"].replace("_", "\n") for item in rows]
    x = np.arange(len(rows))
    specifications = (
        ("success_rate", "Planning success rate", (0.0, 1.05)),
        ("dynamic_feasibility_rate", "Dynamic feasibility rate", (0.0, 1.05)),
        ("mean_maximum_acceleration", "Mean peak acceleration (m/s²)", None),
        ("batch_end_to_end_ms_per_trajectory", "Batched planning time (ms/trajectory)", None),
    )
    for axis, (metric, title, limits) in zip(axes.flat, specifications, strict=True):
        axis.bar(x, [item[metric] for item in rows])
        axis.set_xticks(x, labels, fontsize=7)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if limits is not None:
            axis.set_ylim(*limits)
    figure.suptitle("Experiment 2: matched component ablation at 8 ODE steps")
    figure.tight_layout()
    figure.savefig(output / "planning_ablation.png", dpi=220)
    plt.close(figure)
    print(f"Planning ablation saved to: {output}")


if __name__ == "__main__":
    main()
