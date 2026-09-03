#!/usr/bin/env python3
"""Benchmark genuine batch-size-one ring-planning latency.

Model/checkpoint loading and warm-up are excluded.  The measured interval starts
with online condition normalization and includes prior construction, ODE
integration, decoding, and all method-specific projection stages.  Trajectory
evaluation is deliberately performed after the timer stops.
"""

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
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safe_game_flow.evaluation import evaluate_ring_trajectory  # noqa: E402
from safe_game_flow.flow_matching.model import (  # noqa: E402
    FlowMatching1D,
    model_config_from_checkpoint,
)
from safe_game_flow.flow_matching.normalization import denormalize_trajectory  # noqa: E402
from safe_game_flow.flow_matching.train import sample_ode  # noqa: E402
from safe_game_flow.safety import (  # noqa: E402
    RingResidualCBFProjector,
    project_boundary_residual_to_feasibility,
)
from safe_game_flow.trajectories import (  # noqa: E402
    decode_boundary_residual,
    integrate_velocity_profile,
    project_velocity_boundary_constraints,
    quintic_boundary_skeleton,
)

from sample_conditional_fm import _sample_safe_passage_prior  # noqa: E402


def _load_conditions(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    archive = np.load(path)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("conditions must be an .npz archive")
    try:
        return (
            np.asarray(archive["values"], dtype=np.float32),
            tuple(str(item) for item in archive["schema"]),
        )
    finally:
        archive.close()


def _load_model(path: Path, device: torch.device) -> tuple[FlowMatching1D, dict]:
    checkpoint = torch.load(path, map_location=device)
    model = FlowMatching1D(**model_config_from_checkpoint(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _normalised_condition(condition: np.ndarray, checkpoint: dict, device: torch.device) -> torch.Tensor:
    mean = torch.as_tensor(checkpoint["condition_mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["condition_std"], dtype=torch.float32)
    result = (torch.from_numpy(condition[None, :]) - mean[None, :]) / std[None, :]
    return result.to(device)


def _sample_boundary(
    condition: np.ndarray,
    schema: tuple[str, ...],
    model: FlowMatching1D,
    checkpoint: dict,
    device: torch.device,
    ode_steps: int,
    generator: torch.Generator,
    *,
    safe: bool,
    maximum_speed: float,
    maximum_acceleration: float,
) -> np.ndarray:
    normalised_condition = _normalised_condition(condition, checkpoint, device)
    mean = torch.as_tensor(checkpoint["mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(checkpoint["std"], device=device, dtype=torch.float32)
    projector = RingResidualCBFProjector(
        condition[None, :],
        schema,
        int(checkpoint["H"]),
        mean,
        std,
        alpha=5.0,
        collocation_factor=2,
        projection_iterations=2,
    )
    initial_state = None
    velocity_projector = None
    if safe:
        initial_state, _ = _sample_safe_passage_prior(
            projector,
            condition[None, :],
            mean,
            std,
            generator,
            device,
            maximum_attempts=20,
        )
        velocity_projector = projector
    normalised = sample_ode(
        model,
        num_samples=1,
        H=int(checkpoint["H"]),
        ode_steps=ode_steps,
        device=device,
        generator=generator,
        condition=normalised_condition,
        velocity_projector=velocity_projector,
        initial_state=initial_state,
    )
    residual = np.transpose(
        denormalize_trajectory(normalised, mean, std).cpu().numpy(), (0, 2, 1)
    )[0]
    if safe:
        return project_boundary_residual_to_feasibility(
            residual,
            condition,
            schema,
            maximum_speed=maximum_speed,
            maximum_acceleration=maximum_acceleration,
        ).trajectory
    return decode_boundary_residual(residual, condition, schema)


def _sample_velocity(
    condition: np.ndarray,
    schema: tuple[str, ...],
    model: FlowMatching1D,
    checkpoint: dict,
    device: torch.device,
    ode_steps: int,
    generator: torch.Generator,
) -> np.ndarray:
    normalised = sample_ode(
        model,
        num_samples=1,
        H=int(checkpoint["H"]),
        ode_steps=ode_steps,
        device=device,
        generator=generator,
        condition=_normalised_condition(condition, checkpoint, device),
    )
    mean = torch.as_tensor(checkpoint["mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(checkpoint["std"], device=device, dtype=torch.float32)
    velocity = np.transpose(
        denormalize_trajectory(normalised, mean, std).cpu().numpy(), (0, 2, 1)
    )[0]
    velocity = project_velocity_boundary_constraints(velocity, condition, schema)
    return integrate_velocity_profile(velocity, condition, schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", default="1,4,8,16")
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=921)
    parser.add_argument("--maximum-speed", type=float, default=1.95)
    parser.add_argument("--maximum-acceleration", type=float, default=2.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.queries < 1 or args.warmup < 0:
        raise ValueError("queries must be positive and warmup non-negative")
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    benchmark = args.benchmark_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    conditions, schema = _load_conditions(benchmark / "data" / "test" / "conditions.npz")
    if args.queries > len(conditions):
        raise ValueError(f"requested {args.queries} queries but only {len(conditions)} exist")
    models = benchmark / "models"
    boundary_model, boundary_checkpoint = _load_model(
        models / "boundary_residual_cfm" / "checkpoint.pt", device
    )
    velocity_model, velocity_checkpoint = _load_model(
        models / "velocity_cfm" / "checkpoint.pt", device
    )
    if tuple(boundary_checkpoint.get("condition_schema", ())) != schema:
        raise ValueError("boundary checkpoint and condition schemas differ")
    if tuple(velocity_checkpoint.get("condition_schema", ())) != schema:
        raise ValueError("velocity checkpoint and condition schemas differ")

    steps = [int(item) for item in args.steps.split(",")]
    methods = ("quintic_skeleton", "boundary_trajectory", "projected_velocity", "proposed_safe")
    rows: list[dict] = []

    def plan(method: str, condition: np.ndarray, ode_steps: int, seed: int) -> np.ndarray:
        generator = torch.Generator(device=device).manual_seed(seed)
        if method == "quintic_skeleton":
            return quintic_boundary_skeleton(condition, int(boundary_checkpoint["H"]), schema)
        if method == "boundary_trajectory":
            return _sample_boundary(
                condition, schema, boundary_model, boundary_checkpoint, device,
                ode_steps, generator, safe=False,
                maximum_speed=args.maximum_speed,
                maximum_acceleration=args.maximum_acceleration,
            )
        if method == "projected_velocity":
            return _sample_velocity(
                condition, schema, velocity_model, velocity_checkpoint, device,
                ode_steps, generator,
            )
        if method == "proposed_safe":
            return _sample_boundary(
                condition, schema, boundary_model, boundary_checkpoint, device,
                ode_steps, generator, safe=True,
                maximum_speed=args.maximum_speed,
                maximum_acceleration=args.maximum_acceleration,
            )
        raise KeyError(method)

    for method in methods:
        method_steps = [0] if method == "quintic_skeleton" else steps
        for ode_steps in method_steps:
            print(f"Benchmarking method={method}, steps={ode_steps}", flush=True)
            for index in range(args.warmup):
                plan(method, conditions[index % len(conditions)], max(ode_steps, 1), args.seed + index)
            for index, condition in enumerate(conditions[: args.queries]):
                start = time.perf_counter_ns()
                trajectory = plan(method, condition, max(ode_steps, 1), args.seed + 10_000 + index)
                latency_ms = (time.perf_counter_ns() - start) / 1e6
                metrics = evaluate_ring_trajectory(
                    trajectory,
                    condition,
                    schema,
                    maximum_speed=2.0,
                    maximum_acceleration=3.0,
                )
                rows.append(
                    {
                        "method": method,
                        "ode_steps": ode_steps,
                        "query_index": index,
                        "latency_ms": latency_ms,
                        "success": int(
                            metrics.passed_ring
                            and metrics.collision_free
                            and metrics.dynamically_feasible
                            and metrics.endpoint_feasible
                        ),
                        "maximum_speed": metrics.maximum_speed,
                        "maximum_acceleration": metrics.maximum_acceleration,
                    }
                )

    with (output / "all_queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["ode_steps"])].append(row)
    aggregate = []
    for (method, ode_steps), group in grouped.items():
        latency = np.asarray([item["latency_ms"] for item in group])
        aggregate.append(
            {
                "method": method,
                "ode_steps": ode_steps,
                "queries": len(group),
                "median_ms": float(np.median(latency)),
                "p95_ms": float(np.percentile(latency, 95)),
                "p99_ms": float(np.percentile(latency, 99)),
                "mean_ms": float(np.mean(latency)),
                "std_ms": float(np.std(latency, ddof=1)),
                "success_rate": float(np.mean([item["success"] for item in group])),
            }
        )
    aggregate.sort(key=lambda item: (item["method"], item["ode_steps"]))
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (output / "environment.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "torch_version": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "model_loading_excluded": True,
                "warmup_queries_per_configuration": args.warmup,
                "timed_pipeline": "condition normalization through decoded/projected trajectory",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in methods[1:]:
        group = [item for item in aggregate if item["method"] == method]
        x = [item["ode_steps"] for item in group]
        axes[0].plot(x, [item["median_ms"] for item in group], marker="o", label=method)
        axes[0].plot(x, [item["p99_ms"] for item in group], marker="x", linestyle="--")
        axes[1].plot(x, [item["success_rate"] for item in group], marker="o", label=method)
    axes[0].axhline(
        next(item["median_ms"] for item in aggregate if item["method"] == "quintic_skeleton"),
        color="black", linestyle=":", label="quintic median",
    )
    axes[0].set_ylabel("Batch-size-one latency (ms)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Planning success rate")
    axes[1].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("ODE integration steps")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle("Online single-query latency: solid=median, dashed=P99")
    figure.tight_layout()
    figure.savefig(output / "single_query_latency.png", dpi=220)
    plt.close(figure)
    print(f"Single-query latency benchmark saved to: {output}")


if __name__ == "__main__":
    main()
