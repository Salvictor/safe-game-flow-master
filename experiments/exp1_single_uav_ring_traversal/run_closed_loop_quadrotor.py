#!/usr/bin/env python3
"""Evaluate planned ring trajectories with a common 6-DoF quadrotor plant."""

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

from safe_game_flow.evaluation import (  # noqa: E402
    evaluate_ring_trajectory,
    ring_from_condition,
)
from safe_game_flow.simulation import (  # noqa: E402
    QuadrotorParameters,
    simulate_quadrotor_tracking,
)


SCENARIOS = {
    "nominal": {
        "wind_acceleration": 0.0,
        "position_noise_std": 0.0,
        "velocity_noise_std": 0.0,
        "mass_scale_range": (1.0, 1.0),
        "motor_tau_scale_range": (1.0, 1.0),
        "command_delay": 0.0,
    },
    "moderate_uncertainty": {
        "wind_acceleration": 0.6,
        "position_noise_std": 0.005,
        "velocity_noise_std": 0.02,
        "mass_scale_range": (0.95, 1.05),
        "motor_tau_scale_range": (0.8, 1.2),
        "command_delay": 0.01,
    },
    "severe_uncertainty": {
        "wind_acceleration": 1.2,
        "position_noise_std": 0.015,
        "velocity_noise_std": 0.05,
        "mass_scale_range": (0.90, 1.10),
        "motor_tau_scale_range": (0.7, 1.4),
        "command_delay": 0.02,
    },
}


def _load_conditions(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    archive = np.load(path)
    try:
        return np.asarray(archive["values"], dtype=float), tuple(str(x) for x in archive["schema"])
    finally:
        archive.close()


def _condition_value(condition: np.ndarray, schema: tuple[str, ...], name: str) -> float:
    return float(condition[schema.index(name)])


def _crossing_time(trajectory: np.ndarray, duration: float, ring) -> float:
    crossing = ring.detect_ring_crossing(trajectory, direction=1)
    if not crossing.passed or crossing.segment_index is None:
        return float("nan")
    index = crossing.segment_index
    signed = np.asarray(ring.signed_plane_distance(trajectory))
    denominator = signed[index + 1] - signed[index]
    fraction = 0.0 if abs(denominator) < 1e-12 else -signed[index] / denominator
    return float((index + np.clip(fraction, 0.0, 1.0)) / (len(trajectory) - 1) * duration)


def _unit_random_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    vector[2] *= 0.35
    return vector / np.linalg.norm(vector)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--ode-sweep-dir", type=Path, required=True)
    parser.add_argument("--robustness-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-trajectories", type=int, default=64)
    parser.add_argument("--integration-dt", type=float, default=0.002)
    parser.add_argument("--control-dt", type=float, default=0.002)
    parser.add_argument("--seeds", default="701,702,703")
    args = parser.parse_args()

    benchmark = args.benchmark_dir.resolve()
    ode_sweep = args.ode_sweep_dir.resolve()
    robustness = args.robustness_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    conditions, schema = _load_conditions(benchmark / "data" / "test" / "conditions.npz")
    count = min(args.num_trajectories, len(conditions))
    conditions = conditions[:count]
    plan_paths = {
        "quintic_skeleton": benchmark / "generated" / "quintic_skeleton" / "trajectories.npy",
        "boundary_cfm_steps8": ode_sweep / "runs" / "boundary_trajectory" / "steps_8" / "seed_401" / "trajectories.npy",
        "projected_velocity_steps8": ode_sweep / "runs" / "projected_velocity" / "steps_8" / "seed_401" / "trajectories.npy",
        "proposed_robust_steps1": robustness / "robust_plans" / "proposed_steps1" / "seed_401" / "trajectories.npy",
        "proposed_robust_steps8": robustness / "robust_plans" / "proposed_steps8" / "seed_401" / "trajectories.npy",
    }
    plans = {}
    for name, path in plan_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        plans[name] = np.transpose(np.load(path), (0, 2, 1))[:count]

    seeds = [int(item) for item in args.seeds.split(",")]
    parameters = QuadrotorParameters()
    rows: list[dict] = []
    example_saved = False
    for scenario_name, scenario in SCENARIOS.items():
        for repetition_seed in seeds:
            for method, trajectories in plans.items():
                print(
                    f"scenario={scenario_name}, seed={repetition_seed}, method={method}",
                    flush=True,
                )
                for index, (trajectory, condition) in enumerate(
                    zip(trajectories, conditions, strict=True)
                ):
                    # Method-independent seed gives paired plant/sensor
                    # uncertainty for every planner on the same task instance.
                    rng = np.random.default_rng(repetition_seed * 10_000 + index)
                    duration = _condition_value(condition, schema, "flight_time")
                    wind = (
                        scenario["wind_acceleration"] * _unit_random_vector(rng)
                        if scenario["wind_acceleration"] > 0
                        else np.zeros(3)
                    )
                    mass_scale = rng.uniform(*scenario["mass_scale_range"])
                    tau_scale = rng.uniform(*scenario["motor_tau_scale_range"])
                    result = simulate_quadrotor_tracking(
                        trajectory,
                        duration,
                        parameters=parameters,
                        integration_dt=args.integration_dt,
                        control_dt=args.control_dt,
                        command_delay=scenario["command_delay"],
                        disturbance_acceleration=wind,
                        position_noise_std=scenario["position_noise_std"],
                        velocity_noise_std=scenario["velocity_noise_std"],
                        mass_scale=mass_scale,
                        motor_time_constant_scale=tau_scale,
                        seed=repetition_seed * 10_000 + index,
                    )
                    metrics = evaluate_ring_trajectory(
                        result.positions,
                        condition,
                        schema,
                        maximum_speed=2.0,
                        maximum_acceleration=3.0,
                        dense_factor=1,
                    )
                    ring = ring_from_condition(condition, schema)
                    reference_crossing = _crossing_time(trajectory, duration, ring)
                    actual_crossing = _crossing_time(result.positions, duration, ring)
                    rows.append(
                        {
                            "scenario": scenario_name,
                            "seed": repetition_seed,
                            "method": method,
                            "trajectory_index": index,
                            "tracking_rmse_m": result.tracking_rmse,
                            "maximum_tracking_error_m": result.maximum_tracking_error,
                            "ring_passed": int(metrics.passed_ring),
                            "collision_free": int(metrics.collision_free),
                            "endpoint_feasible": int(metrics.endpoint_feasible),
                            "minimum_barrier": metrics.minimum_barrier,
                            "crossing_clearance_m": metrics.crossing_clearance,
                            "crossing_time_error_s": abs(actual_crossing - reference_crossing),
                            "motor_saturation_fraction": result.motor_saturation_fraction,
                            "maximum_tilt_degrees": result.maximum_tilt_degrees,
                            "wind_acceleration_mps2": np.linalg.norm(wind),
                            "mass_scale": mass_scale,
                            "motor_tau_scale": tau_scale,
                            "command_delay_s": scenario["command_delay"],
                        }
                    )
                    if not example_saved and scenario_name == "moderate_uncertainty" and method == "proposed_robust_steps8":
                        np.savez_compressed(
                            output / "example_closed_loop.npz",
                            times=result.times,
                            reference=result.reference_positions,
                            actual=result.positions,
                            motor_thrusts=result.motor_thrusts,
                            commanded_motor_thrusts=result.commanded_motor_thrusts,
                            ring_center=ring.center,
                            ring_normal=ring.normal,
                        )
                        example_saved = True

    with (output / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["method"])].append(row)
    aggregate = []
    for (scenario, method), group in grouped.items():
        def values(name: str) -> np.ndarray:
            return np.asarray([float(item[name]) for item in group])

        crossing_error = values("crossing_time_error_s")
        finite_crossing_error = crossing_error[np.isfinite(crossing_error)]
        aggregate.append(
            {
                "scenario": scenario,
                "method": method,
                "runs": len(group),
                "ring_pass_rate": float(np.mean(values("ring_passed"))),
                "collision_free_rate": float(np.mean(values("collision_free"))),
                "endpoint_feasibility_rate": float(np.mean(values("endpoint_feasible"))),
                "tracking_rmse_mean_m": float(np.mean(values("tracking_rmse_m"))),
                "tracking_rmse_p95_m": float(np.percentile(values("tracking_rmse_m"), 95)),
                "maximum_tracking_error_p95_m": float(np.percentile(values("maximum_tracking_error_m"), 95)),
                "crossing_time_error_mean_s": (
                    float(np.mean(finite_crossing_error)) if len(finite_crossing_error) else float("nan")
                ),
                "motor_saturation_mean": float(np.mean(values("motor_saturation_fraction"))),
                "maximum_tilt_p95_degrees": float(np.percentile(values("maximum_tilt_degrees"), 95)),
                "minimum_barrier": float(np.min(values("minimum_barrier"))),
            }
        )
    aggregate.sort(key=lambda item: (item["scenario"], item["method"]))
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (output / "simulation_parameters.json").write_text(
        json.dumps(
            {
                "physical_parameters": parameters.__dict__,
                "official_specification_fields": [
                    "mass", "arm_radius", "maximum_motor_thrust"
                ],
                "assumed_pending_identification": [
                    "inertia_xx", "inertia_yy", "inertia_zz", "motor_time_constant",
                    "yaw_moment_ratio", "linear_drag"
                ],
                "integration_dt": args.integration_dt,
                "control_dt": args.control_dt,
                "scenarios": SCENARIOS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    method_names = list(plans)
    x = np.arange(len(method_names))
    width = 0.25
    for scenario_index, scenario_name in enumerate(SCENARIOS):
        group = [
            next(item for item in aggregate if item["scenario"] == scenario_name and item["method"] == method)
            for method in method_names
        ]
        offset = (scenario_index - 1) * width
        axes[0].bar(x + offset, [item["tracking_rmse_mean_m"] for item in group], width, label=scenario_name)
        axes[1].bar(x + offset, [item["tracking_rmse_p95_m"] for item in group], width)
        axes[2].bar(x + offset, [item["ring_pass_rate"] for item in group], width)
    labels = [name.replace("_", "\n") for name in method_names]
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean tracking RMSE (m)")
    axes[1].set_ylabel("P95 tracking RMSE (m)")
    axes[2].set_ylabel("Closed-loop ring pass rate")
    axes[2].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=8)
    figure.suptitle("6-DoF closed-loop tracking under plant and sensing uncertainty")
    figure.tight_layout()
    figure.savefig(output / "closed_loop_robustness.png", dpi=220)
    plt.close(figure)

    example = np.load(output / "example_closed_loop.npz")
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for coordinate, label in enumerate("xyz"):
        axes[0].plot(example["times"], example["reference"][:, coordinate], linestyle="--", label=f"{label} ref")
        axes[0].plot(example["times"], example["actual"][:, coordinate], label=f"{label} actual")
    axes[0].set_ylabel("Position (m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(alpha=0.25)
    for motor in range(4):
        axes[1].plot(example["times"], 1000.0 * example["motor_thrusts"][:, motor], label=f"motor {motor + 1}")
    axes[1].axhline(1000.0 * parameters.maximum_motor_thrust, color="black", linestyle=":", label="motor limit")
    axes[1].set_xlabel("Physical time (s)")
    axes[1].set_ylabel("Motor thrust (mN)")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.suptitle("Representative proposed trajectory: reference and 6-DoF response")
    figure.tight_layout()
    figure.savefig(output / "closed_loop_example.png", dpi=220)
    plt.close(figure)
    print(f"Closed-loop results saved to: {output}")


if __name__ == "__main__":
    main()
