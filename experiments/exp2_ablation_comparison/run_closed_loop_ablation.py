#!/usr/bin/env python3
"""Run paired severe-uncertainty 6-DoF closed-loop ablations."""

from __future__ import annotations

import argparse
import csv
import json
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

from safe_game_flow.evaluation import evaluate_ring_trajectory, ring_from_condition  # noqa: E402
from safe_game_flow.simulation import QuadrotorParameters, simulate_quadrotor_tracking  # noqa: E402


METHODS = (
    "quintic_skeleton",
    "projected_velocity_cfm",
    "boundary_only",
    "plus_safe_prior",
    "plus_cbf",
    "plus_kinodynamic",
    "plus_execution_margin",
)
SEVERE = {
    "wind_acceleration": 1.2,
    "position_noise_std": 0.015,
    "velocity_noise_std": 0.05,
    "mass_scale_range": (0.90, 1.10),
    "motor_tau_scale_range": (0.7, 1.4),
    "command_delay": 0.02,
}


def _unit_random_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    vector[2] *= 0.35
    return vector / np.linalg.norm(vector)


def _crossing_time(trajectory: np.ndarray, duration: float, ring) -> float:
    crossing = ring.detect_ring_crossing(trajectory, direction=1)
    if not crossing.passed or crossing.segment_index is None:
        return float("nan")
    index = crossing.segment_index
    signed = np.asarray(ring.signed_plane_distance(trajectory))
    denominator = signed[index + 1] - signed[index]
    fraction = 0.0 if abs(denominator) < 1e-12 else -signed[index] / denominator
    return float((index + np.clip(fraction, 0.0, 1.0)) / (len(trajectory) - 1) * duration)


def _bootstrap_comparisons(rows: list[dict], samples: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    result = []
    metrics = (
        "tracking_rmse_m",
        "maximum_tracking_error_m",
        "crossing_time_error_s",
        "maximum_tilt_degrees",
    )
    keyed = {
        (int(row["seed"]), int(row["trajectory_index"]), row["method"]): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    indices = sorted({int(row["trajectory_index"]) for row in rows})
    proposed = "plus_execution_margin"
    for baseline in METHODS[:-1]:
        for metric in metrics:
            clusters = []
            for index in indices:
                difference = []
                for run_seed in seeds:
                    base = float(keyed[(run_seed, index, baseline)][metric])
                    prop = float(keyed[(run_seed, index, proposed)][metric])
                    if np.isfinite(base) and np.isfinite(prop):
                        difference.append(base - prop)
                clusters.append(np.asarray(difference))
            observed = np.concatenate([cluster for cluster in clusters if len(cluster)])
            draws = np.empty(samples)
            for draw in range(samples):
                selected = rng.integers(0, len(clusters), len(clusters))
                draw_values = [clusters[item] for item in selected if len(clusters[item])]
                draws[draw] = np.mean(np.concatenate(draw_values))
            lower, upper = np.percentile(draws, (2.5, 97.5))
            result.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "paired_runs": len(observed),
                    "baseline_minus_full_mean": float(np.mean(observed)),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "conclusion": (
                        "full_better" if lower > 0 else "full_worse" if upper < 0 else "inconclusive"
                    ),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-trajectories", type=int, default=24)
    parser.add_argument("--seeds", default="901,902,903")
    parser.add_argument("--integration-dt", type=float, default=0.004)
    parser.add_argument("--control-dt", type=float, default=0.004)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    args = parser.parse_args()
    planning = args.planning_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = planning / "generated"
    archive = np.load(generated / "quintic_skeleton" / "conditions_used.npz")
    try:
        conditions = np.asarray(archive["values"], dtype=float)[: args.num_trajectories]
        schema = tuple(str(item) for item in archive["schema"])
    finally:
        archive.close()
    trajectories = {
        method: np.transpose(np.load(generated / method / "trajectories.npy"), (0, 2, 1))[
            : args.num_trajectories
        ]
        for method in METHODS
    }
    seeds = [int(item) for item in args.seeds.split(",")]
    parameters = QuadrotorParameters()
    rows: list[dict] = []
    for run_seed in seeds:
        for method in METHODS:
            print(f"seed={run_seed}, method={method}", flush=True)
            for index, (trajectory, condition) in enumerate(
                zip(trajectories[method], conditions, strict=True)
            ):
                rng = np.random.default_rng(run_seed * 10_000 + index)
                wind = SEVERE["wind_acceleration"] * _unit_random_vector(rng)
                mass_scale = rng.uniform(*SEVERE["mass_scale_range"])
                tau_scale = rng.uniform(*SEVERE["motor_tau_scale_range"])
                duration = float(condition[schema.index("flight_time")])
                simulation = simulate_quadrotor_tracking(
                    trajectory,
                    duration,
                    parameters=parameters,
                    integration_dt=args.integration_dt,
                    control_dt=args.control_dt,
                    command_delay=SEVERE["command_delay"],
                    disturbance_acceleration=wind,
                    position_noise_std=SEVERE["position_noise_std"],
                    velocity_noise_std=SEVERE["velocity_noise_std"],
                    mass_scale=mass_scale,
                    motor_time_constant_scale=tau_scale,
                    seed=run_seed * 10_000 + index,
                )
                metrics = evaluate_ring_trajectory(
                    simulation.positions,
                    condition,
                    schema,
                    maximum_speed=2.0,
                    maximum_acceleration=3.0,
                    dense_factor=1,
                )
                ring = ring_from_condition(condition, schema)
                planned_crossing = _crossing_time(trajectory, duration, ring)
                actual_crossing = _crossing_time(simulation.positions, duration, ring)
                rows.append(
                    {
                        "seed": run_seed,
                        "method": method,
                        "trajectory_index": index,
                        "tracking_rmse_m": simulation.tracking_rmse,
                        "maximum_tracking_error_m": simulation.maximum_tracking_error,
                        "crossing_time_error_s": abs(actual_crossing - planned_crossing),
                        "maximum_tilt_degrees": simulation.maximum_tilt_degrees,
                        "motor_saturation_fraction": simulation.motor_saturation_fraction,
                        "ring_passed": int(metrics.passed_ring),
                        "collision_free": int(metrics.collision_free),
                        "endpoint_feasible": int(metrics.endpoint_feasible),
                        "minimum_barrier": metrics.minimum_barrier,
                        "mass_scale": mass_scale,
                        "motor_tau_scale": tau_scale,
                        "wind_x": wind[0],
                        "wind_y": wind[1],
                        "wind_z": wind[2],
                    }
                )

    with (output / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    aggregate = []
    for method in METHODS:
        group = grouped[method]
        def values(name: str) -> np.ndarray:
            return np.asarray([float(item[name]) for item in group])
        crossing = values("crossing_time_error_s")
        aggregate.append(
            {
                "method": method,
                "runs": len(group),
                "tracking_rmse_mean_m": float(np.mean(values("tracking_rmse_m"))),
                "tracking_rmse_p95_m": float(np.percentile(values("tracking_rmse_m"), 95)),
                "maximum_tracking_error_p95_m": float(np.percentile(values("maximum_tracking_error_m"), 95)),
                "crossing_time_error_mean_s": float(np.mean(crossing[np.isfinite(crossing)])),
                "maximum_tilt_p95_degrees": float(np.percentile(values("maximum_tilt_degrees"), 95)),
                "motor_saturation_mean": float(np.mean(values("motor_saturation_fraction"))),
                "ring_pass_rate": float(np.mean(values("ring_passed"))),
                "collision_free_rate": float(np.mean(values("collision_free"))),
                "endpoint_feasibility_rate": float(np.mean(values("endpoint_feasible"))),
                "minimum_barrier": float(np.min(values("minimum_barrier"))),
            }
        )
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    bootstrap = _bootstrap_comparisons(rows, args.bootstrap_samples, seed=1901)
    with (output / "paired_bootstrap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bootstrap[0]))
        writer.writeheader()
        writer.writerows(bootstrap)
    (output / "paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    (output / "configuration.json").write_text(
        json.dumps(
            {
                "scenario": SEVERE,
                "num_trajectories": len(conditions),
                "seeds": seeds,
                "integration_dt": args.integration_dt,
                "control_dt": args.control_dt,
                "paired_uncertainty": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(aggregate))
    labels = [item["method"].replace("_", "\n") for item in aggregate]
    plots = (
        ("tracking_rmse_mean_m", "Mean tracking RMSE (m)"),
        ("maximum_tilt_p95_degrees", "P95 maximum tilt (deg)"),
        ("ring_pass_rate", "Closed-loop ring pass rate"),
    )
    for axis, (metric, title) in zip(axes, plots, strict=True):
        axis.bar(x, [item[metric] for item in aggregate])
        axis.set_xticks(x, labels, fontsize=7)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    axes[2].set_ylim(0.0, 1.05)
    figure.suptitle("Experiment 2: paired 6-DoF ablation under severe uncertainty")
    figure.tight_layout()
    figure.savefig(output / "closed_loop_ablation.png", dpi=220)
    plt.close(figure)
    print(f"Closed-loop ablation saved to: {output}")


if __name__ == "__main__":
    main()
