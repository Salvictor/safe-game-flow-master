#!/usr/bin/env python3
"""Run controlled baselines and the proposed single-UAV ring method end to end."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safe_game_flow.trajectories import quintic_boundary_skeleton  # noqa: E402


def _run(arguments: list[str]) -> None:
    print("\n$", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=EXPERIMENT_DIR.parents[1], check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "baseline_benchmark",
    )
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--time-emb-dim", type=int, default=64)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--train-seed", type=int, default=101)
    parser.add_argument("--test-seed", type=int, default=202)
    parser.add_argument("--model-seed", type=int, default=303)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--endpoint-offset-fraction",
        type=float,
        default=0.45,
        help="task difficulty: start/goal lateral offset relative to clear aperture",
    )
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    train_data = work_dir / "data" / "train"
    test_data = work_dir / "data" / "test"
    models_dir = work_dir / "models"
    generated_dir = work_dir / "generated"
    evaluation_dir = work_dir / "evaluation"
    for path in (train_data, test_data, models_dir, generated_dir, evaluation_dir):
        path.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    generator_script = str(EXPERIMENT_DIR / "generate_dataset.py")
    trainer_script = str(EXPERIMENT_DIR / "train_conditional_fm.py")
    sampler_script = str(EXPERIMENT_DIR / "sample_conditional_fm.py")
    evaluator_script = str(EXPERIMENT_DIR / "evaluate_trajectories.py")

    if not args.skip_training:
        _run(
            [
                python,
                generator_script,
                "--num-trajectories",
                str(args.train_size),
                "--seed",
                str(args.train_seed),
                "--output-dir",
                str(train_data),
                "--no-preview",
                "--endpoint-offset-fraction",
                str(args.endpoint_offset_fraction),
            ]
        )
        _run(
            [
                python,
                generator_script,
                "--num-trajectories",
                str(args.test_size),
                "--seed",
                str(args.test_seed),
                "--output-dir",
                str(test_data),
                "--no-preview",
                "--endpoint-offset-fraction",
                str(args.endpoint_offset_fraction),
            ]
        )
        for representation, model_name in (
            ("position", "raw_position_cfm"),
            ("boundary_residual", "boundary_residual_cfm"),
            ("velocity", "velocity_cfm"),
        ):
            command = [
                python,
                trainer_script,
                "--data-dir",
                str(train_data),
                "--save-dir",
                str(models_dir / model_name),
                "--representation",
                representation,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                "0",
                "--hidden",
                str(args.hidden),
                "--blocks",
                str(args.blocks),
                "--time-emb-dim",
                str(args.time_emb_dim),
                "--ode-steps",
                str(args.ode_steps),
                "--preview-interval",
                "0",
                "--seed",
                str(args.model_seed),
            ]
            if args.cpu:
                command.append("--cpu")
            _run(command)

    methods = {
        "raw_cfm": {
            "checkpoint": models_dir / "raw_position_cfm" / "checkpoint.pt",
            "flags": ["--disable-cbf", "--unsafe-gaussian-prior"],
        },
        "boundary_cfm": {
            "checkpoint": models_dir / "boundary_residual_cfm" / "checkpoint.pt",
            "flags": ["--disable-cbf", "--unsafe-gaussian-prior"],
        },
        "velocity_cfm": {
            "checkpoint": models_dir / "velocity_cfm" / "checkpoint.pt",
            "flags": ["--disable-cbf", "--unsafe-gaussian-prior"],
        },
        "velocity_boundary_projected": {
            "checkpoint": models_dir / "velocity_cfm" / "checkpoint.pt",
            "flags": [
                "--disable-cbf",
                "--unsafe-gaussian-prior",
                "--project-velocity-boundary",
            ],
        },
        "boundary_safe_prior": {
            "checkpoint": models_dir / "boundary_residual_cfm" / "checkpoint.pt",
            "flags": ["--disable-cbf"],
        },
        "boundary_safe_prior_cbf": {
            "checkpoint": models_dir / "boundary_residual_cfm" / "checkpoint.pt",
            "flags": [],
        },
        "proposed_full": {
            "checkpoint": models_dir / "boundary_residual_cfm" / "checkpoint.pt",
            "flags": ["--kinodynamic-project"],
        },
    }

    test_archive = np.load(test_data / "conditions.npz")
    try:
        test_conditions = np.asarray(test_archive["values"])
        test_schema = tuple(str(item) for item in test_archive["schema"])
    finally:
        test_archive.close()
    expert_trajectories = np.transpose(np.load(test_data / "trajectories.npy"), (0, 2, 1))

    rows = []
    skeleton_output = generated_dir / "quintic_skeleton"
    skeleton_evaluation = evaluation_dir / "quintic_skeleton"
    skeleton_output.mkdir(parents=True, exist_ok=True)
    skeleton_start = time.perf_counter()
    skeletons = quintic_boundary_skeleton(
        test_conditions, expert_trajectories.shape[1], test_schema
    )
    skeleton_seconds = time.perf_counter() - skeleton_start
    np.save(skeleton_output / "trajectories.npy", np.transpose(skeletons, (0, 2, 1)))
    np.savez_compressed(
        skeleton_output / "conditions_used.npz",
        values=test_conditions,
        schema=np.asarray(test_schema),
    )
    _run(
        [
            python,
            evaluator_script,
            "--trajectories",
            str(skeleton_output / "trajectories.npy"),
            "--conditions",
            str(skeleton_output / "conditions_used.npz"),
            "--output-dir",
            str(skeleton_evaluation),
        ]
    )
    skeleton_metrics = _load_json(skeleton_evaluation / "summary.json")
    rows.append(
        {
            "method": "quintic_skeleton",
            **skeleton_metrics,
            "trajectory_rmse_to_expert": float(
                np.sqrt(np.mean((skeletons - expert_trajectories) ** 2))
            ),
            "ode_ms_per_trajectory": 1000.0 * skeleton_seconds / len(skeletons),
            "kinodynamic_projection_seconds": 0.0,
            "cbf_intervention_rate": 0.0,
            "mean_residual_scale": 0.0,
        }
    )
    for method_name, method in methods.items():
        output = generated_dir / method_name
        evaluation = evaluation_dir / method_name
        command = [
            python,
            sampler_script,
            "--checkpoint",
            str(method["checkpoint"]),
            "--conditions",
            str(test_data / "conditions.npz"),
            "--output-dir",
            str(output),
            "--num-conditions",
            str(args.test_size),
            "--ode-steps",
            str(args.ode_steps),
            "--seed",
            str(args.model_seed + 1),
            *method["flags"],
        ]
        if args.cpu:
            command.append("--cpu")
        _run(command)
        _run(
            [
                python,
                evaluator_script,
                "--trajectories",
                str(output / "trajectories.npy"),
                "--conditions",
                str(output / "conditions_used.npz"),
                "--output-dir",
                str(evaluation),
            ]
        )
        metrics = _load_json(evaluation / "summary.json")
        sampling = _load_json(output / "sampling_summary.json")
        generated_trajectories = np.transpose(
            np.load(output / "trajectories.npy"), (0, 2, 1)
        )
        row = {
            "method": method_name,
            **metrics,
            "trajectory_rmse_to_expert": float(
                np.sqrt(np.mean((generated_trajectories - expert_trajectories) ** 2))
            ),
            "ode_ms_per_trajectory": sampling["ode_milliseconds_per_trajectory"],
            "kinodynamic_projection_seconds": sampling["kinodynamic_projection_seconds"],
            "cbf_intervention_rate": (
                sampling["cbf_projection"]["intervention_rate"]
                if sampling["cbf_projection"] is not None
                else 0.0
            ),
            "mean_residual_scale": sampling["mean_residual_scale"],
        }
        rows.append(row)

    comparison_path = work_dir / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (work_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
        "cpu_count": os.cpu_count(),
        "configuration": vars(args) | {"work_dir": str(work_dir)},
    }
    (work_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, default=str), encoding="utf-8"
    )

    rate_names = (
        "success_rate",
        "ring_pass_rate",
        "collision_free_rate",
        "dynamic_feasibility_rate",
        "endpoint_feasibility_rate",
    )
    x = range(len(rows))
    width = 0.16
    fig, ax = plt.subplots(figsize=(13, 6))
    for metric_index, metric in enumerate(rate_names):
        ax.bar(
            [value + (metric_index - 2) * width for value in x],
            [row[metric] for row in rows],
            width,
            label=metric.replace("_rate", ""),
        )
    ax.set_xticks(list(x), [row["method"] for row in rows], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Single-UAV ring traversal: controlled method comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(work_dir / "comparison_rates.png", dpi=200)
    plt.close(fig)

    print("\nBenchmark complete:", comparison_path)
    for row in rows:
        print(
            f"{row['method']:24s} success={row['success_rate']:.3f} "
            f"safe={row['collision_free_rate']:.3f} "
            f"dynamic={row['dynamic_feasibility_rate']:.3f} "
            f"endpoint={row['endpoint_feasibility_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
