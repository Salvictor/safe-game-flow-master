#!/usr/bin/env python3
"""Measure batch-size-one latency for every Experiment 2 ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
EXP1_DIR = PROJECT_ROOT / "experiments" / "exp1_single_uav_ring_traversal"
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, EXP1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from safe_game_flow.evaluation import evaluate_ring_trajectory  # noqa: E402
from safe_game_flow.flow_matching.model import FlowMatching1D, model_config_from_checkpoint  # noqa: E402
from safe_game_flow.flow_matching.normalization import denormalize_trajectory  # noqa: E402
from safe_game_flow.flow_matching.train import sample_ode  # noqa: E402
from safe_game_flow.safety import RingResidualCBFProjector, project_boundary_residual_to_feasibility  # noqa: E402
from safe_game_flow.trajectories import (  # noqa: E402
    decode_boundary_residual,
    integrate_velocity_profile,
    project_velocity_boundary_constraints,
    quintic_boundary_skeleton,
)
from sample_conditional_fm import _sample_safe_passage_prior  # noqa: E402


def _load_model(path: Path, device: torch.device) -> tuple[FlowMatching1D, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = FlowMatching1D(**model_config_from_checkpoint(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _condition_tensor(condition: np.ndarray, checkpoint: dict, device: torch.device) -> torch.Tensor:
    mean = torch.as_tensor(checkpoint["condition_mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["condition_std"], dtype=torch.float32)
    return ((torch.from_numpy(condition[None].astype(np.float32)) - mean[None]) / std[None]).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    benchmark = args.benchmark_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = np.load(benchmark / "data" / "test" / "conditions.npz")
    try:
        conditions = np.asarray(archive["values"], dtype=np.float32)[: args.queries]
        schema = tuple(str(item) for item in archive["schema"])
    finally:
        archive.close()
    boundary_model, boundary_checkpoint = _load_model(
        benchmark / "models" / "boundary_residual_cfm" / "checkpoint.pt", device
    )
    velocity_model, velocity_checkpoint = _load_model(
        benchmark / "models" / "velocity_cfm" / "checkpoint.pt", device
    )

    methods = (
        "quintic_skeleton",
        "projected_velocity_cfm",
        "boundary_only",
        "plus_safe_prior",
        "plus_cbf",
        "plus_kinodynamic",
        "plus_execution_margin",
    )

    def plan_boundary(
        condition: np.ndarray,
        generator: torch.Generator,
        *,
        safe_prior: bool,
        cbf: bool,
        kinodynamic: bool,
        post_step: bool,
        maximum_speed: float,
        maximum_acceleration: float,
    ) -> np.ndarray:
        mean = torch.as_tensor(boundary_checkpoint["mean"], device=device, dtype=torch.float32)
        std = torch.as_tensor(boundary_checkpoint["std"], device=device, dtype=torch.float32)
        projector = None
        initial_state = None
        if safe_prior or cbf:
            projector = RingResidualCBFProjector(
                condition[None], schema, int(boundary_checkpoint["H"]), mean, std,
                alpha=5.0, collocation_factor=10, projection_iterations=2,
            )
        if safe_prior:
            initial_state, _ = _sample_safe_passage_prior(
                projector, condition[None], mean, std, generator, device, maximum_attempts=20
            )
        normalised = sample_ode(
            boundary_model,
            1,
            int(boundary_checkpoint["H"]),
            args.ode_steps,
            device,
            generator=generator,
            condition=_condition_tensor(condition, boundary_checkpoint, device),
            velocity_projector=projector if cbf else None,
            state_projector=projector.project_state if post_step else None,
            initial_state=initial_state,
        )
        residual = np.transpose(
            denormalize_trajectory(normalised, mean, std).cpu().numpy(), (0, 2, 1)
        )[0]
        if kinodynamic:
            return project_boundary_residual_to_feasibility(
                residual,
                condition,
                schema,
                maximum_speed=maximum_speed,
                maximum_acceleration=maximum_acceleration,
            ).trajectory
        return decode_boundary_residual(residual, condition, schema)

    def plan(method: str, condition: np.ndarray, seed: int) -> np.ndarray:
        generator = torch.Generator(device=device).manual_seed(seed)
        if method == "quintic_skeleton":
            return quintic_boundary_skeleton(condition, int(boundary_checkpoint["H"]), schema)
        if method == "projected_velocity_cfm":
            mean = torch.as_tensor(velocity_checkpoint["mean"], device=device, dtype=torch.float32)
            std = torch.as_tensor(velocity_checkpoint["std"], device=device, dtype=torch.float32)
            normalised = sample_ode(
                velocity_model, 1, int(velocity_checkpoint["H"]), args.ode_steps, device,
                generator=generator,
                condition=_condition_tensor(condition, velocity_checkpoint, device),
            )
            velocity = np.transpose(
                denormalize_trajectory(normalised, mean, std).cpu().numpy(), (0, 2, 1)
            )[0]
            velocity = project_velocity_boundary_constraints(velocity, condition, schema)
            return integrate_velocity_profile(velocity, condition, schema)
        configuration = {
            "boundary_only": (False, False, False, False, 2.0, 3.0),
            "plus_safe_prior": (True, False, False, False, 2.0, 3.0),
            "plus_cbf": (True, True, False, True, 2.0, 3.0),
            "plus_kinodynamic": (True, True, True, True, 2.0, 3.0),
            "plus_execution_margin": (True, True, True, True, 1.95, 2.0),
        }[method]
        return plan_boundary(
            condition,
            generator,
            safe_prior=configuration[0],
            cbf=configuration[1],
            kinodynamic=configuration[2],
            post_step=configuration[3],
            maximum_speed=configuration[4],
            maximum_acceleration=configuration[5],
        )

    rows: list[dict] = []
    for method in methods:
        print(f"Timing {method}", flush=True)
        for index in range(args.warmup):
            plan(method, conditions[index % len(conditions)], args.seed + index)
        for index, condition in enumerate(conditions):
            start = time.perf_counter_ns()
            trajectory = plan(method, condition, args.seed + 10_000 + index)
            latency = (time.perf_counter_ns() - start) / 1e6
            metrics = evaluate_ring_trajectory(trajectory, condition, schema)
            rows.append(
                {
                    "method": method,
                    "query_index": index,
                    "latency_ms": latency,
                    "success": int(metrics.successful),
                    "collision_free": int(metrics.collision_free),
                    "dynamic_feasible": int(metrics.dynamically_feasible),
                }
            )

    with (output / "all_queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    aggregate = []
    for method in methods:
        group = grouped[method]
        latency = np.asarray([item["latency_ms"] for item in group])
        aggregate.append(
            {
                "method": method,
                "queries": len(group),
                "median_ms": float(np.median(latency)),
                "p95_ms": float(np.percentile(latency, 95)),
                "p99_ms": float(np.percentile(latency, 99)),
                "mean_ms": float(np.mean(latency)),
                "success_rate": float(np.mean([item["success"] for item in group])),
                "collision_free_rate": float(np.mean([item["collision_free"] for item in group])),
                "dynamic_feasibility_rate": float(np.mean([item["dynamic_feasible"] for item in group])),
            }
        )
    with (output / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (output / "latency.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (output / "environment.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "ode_steps": args.ode_steps,
                "model_loading_excluded": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(13, 6))
    x = np.arange(len(aggregate))
    axis.bar(x - 0.18, [item["median_ms"] for item in aggregate], 0.36, label="Median")
    axis.bar(x + 0.18, [item["p99_ms"] for item in aggregate], 0.36, label="P99")
    axis.set_yscale("log")
    axis.set_xticks(x, [item["method"].replace("_", "\n") for item in aggregate], fontsize=8)
    axis.set_ylabel("Batch-size-one online latency (ms)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "ablation_latency.png", dpi=220)
    plt.close(figure)
    print(f"Ablation latency saved to: {output}")


if __name__ == "__main__":
    main()
