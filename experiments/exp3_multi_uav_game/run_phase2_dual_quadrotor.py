#!/usr/bin/env python3
"""Evaluate Phase-2 joint plans with two independent 6-DoF quadrotor plants."""

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

from safe_game_flow.simulation import (  # noqa: E402
    QuadrotorParameters,
    simulate_quadrotor_tracking,
)


SCENARIOS = {
    "nominal": {
        "wind": 0.0,
        "position_noise": 0.0,
        "velocity_noise": 0.0,
        "mass_range": (1.0, 1.0),
        "tau_range": (1.0, 1.0),
        "delay": 0.0,
    },
    "moderate": {
        "wind": 0.6,
        "position_noise": 0.005,
        "velocity_noise": 0.02,
        "mass_range": (0.95, 1.05),
        "tau_range": (0.8, 1.2),
        "delay": 0.01,
    },
    "severe": {
        "wind": 1.2,
        "position_noise": 0.015,
        "velocity_noise": 0.05,
        "mass_range": (0.90, 1.10),
        "tau_range": (0.7, 1.4),
        "delay": 0.02,
    },
}


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    vector[2] *= 0.35
    return vector / np.linalg.norm(vector)


def _crossing_time(
    times: np.ndarray, positions: np.ndarray, center: np.ndarray
) -> tuple[float, bool]:
    x = positions[:, 0]
    for index in np.flatnonzero((x[:-1] < 0.0) & (x[1:] >= 0.0)):
        denominator = x[index + 1] - x[index]
        fraction = -x[index] / denominator if abs(denominator) > 1e-12 else 0.0
        crossing = positions[index] + fraction * (positions[index + 1] - positions[index])
        if np.linalg.norm(crossing[1:] - center[1:]) <= 0.42:
            time_value = times[index] + fraction * (times[index + 1] - times[index])
            return float(time_value), True
    return float("nan"), False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--phase1-evaluation", type=Path, required=True)
    parser.add_argument("--phase2-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-trajectories", type=int, default=24)
    parser.add_argument("--seeds", default="8101,8102,8103")
    parser.add_argument("--integration-dt", type=float, default=0.002)
    parser.add_argument("--control-dt", type=float, default=0.002)
    parser.add_argument(
        "--scenarios",
        default="nominal,moderate,severe",
        help="Comma-separated subset of nominal,moderate,severe.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    phase1 = args.phase1_evaluation.resolve()
    phase2 = args.phase2_evaluation.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = np.load(data_dir / "conditions.npz")
    conditions = archive["values"]
    archive.close()
    count = min(args.num_trajectories, len(conditions))
    methods = {
        "nash_reference": np.load(data_dir / "joint_references.npy")[:count],
        "teacher_expert": np.load(data_dir / "joint_trajectories.npy")[:count],
        "nash_reference_hocbf": np.load(phase2 / "executed_nash_reference.npy")[:count],
        "phase1_cfm_hocbf": np.load(phase1 / "executed_game_cfm.npy")[:count],
        "phase2_guided_cfm_hocbf": np.load(phase2 / "executed_game_cfm.npy")[:count],
    }
    seeds = [int(item) for item in args.seeds.split(",")]
    selected_scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    unknown = sorted(set(selected_scenarios) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}")
    parameters = QuadrotorParameters()
    rows = []
    example_saved = False
    for scenario_name in selected_scenarios:
        scenario = SCENARIOS[scenario_name]
        for seed in seeds:
            for method, joint_trajectories in methods.items():
                print(
                    f"scenario={scenario_name} seed={seed} method={method}", flush=True
                )
                for index, flat_trajectory in enumerate(joint_trajectories):
                    joint = flat_trajectory.reshape(2, 3, -1).transpose(0, 2, 1)
                    center = conditions[index, 12:15]
                    horizon = float(conditions[index, -1])
                    results = []
                    for agent in range(2):
                        rng = np.random.default_rng(seed * 100_000 + index * 10 + agent)
                        wind = (
                            scenario["wind"] * _unit_vector(rng)
                            if scenario["wind"] > 0
                            else np.zeros(3)
                        )
                        result = simulate_quadrotor_tracking(
                            joint[agent],
                            horizon,
                            parameters=parameters,
                            integration_dt=args.integration_dt,
                            control_dt=args.control_dt,
                            command_delay=scenario["delay"],
                            disturbance_acceleration=wind,
                            position_noise_std=scenario["position_noise"],
                            velocity_noise_std=scenario["velocity_noise"],
                            mass_scale=rng.uniform(*scenario["mass_range"]),
                            motor_time_constant_scale=rng.uniform(*scenario["tau_range"]),
                            seed=seed * 100_000 + index * 10 + agent,
                        )
                        results.append(result)
                    separation = np.linalg.norm(
                        results[0].positions - results[1].positions, axis=1
                    )
                    crossings = [
                        _crossing_time(result.times, result.positions, center)
                        for result in results
                    ]
                    rows.append(
                        {
                            "scenario": scenario_name,
                            "seed": seed,
                            "method": method,
                            "trajectory_index": index,
                            "both_passed": int(all(item[1] for item in crossings)),
                            "collision": int(np.min(separation) < 0.25),
                            "margin_violation": int(np.min(separation) < 0.36),
                            "minimum_separation_m": float(np.min(separation)),
                            "crossing_gap_s": float(
                                abs(crossings[0][0] - crossings[1][0])
                                if all(item[1] for item in crossings)
                                else np.nan
                            ),
                            "tracking_rmse_mean_m": float(
                                np.mean([result.tracking_rmse for result in results])
                            ),
                            "maximum_tracking_error_m": float(
                                np.max([result.maximum_tracking_error for result in results])
                            ),
                            "maximum_tilt_degrees": float(
                                np.max([result.maximum_tilt_degrees for result in results])
                            ),
                            "motor_saturation_fraction": float(
                                np.mean([result.motor_saturation_fraction for result in results])
                            ),
                        }
                    )
                    if (
                        not example_saved
                        and scenario_name == "severe"
                        and method == "phase2_guided_cfm_hocbf"
                    ):
                        np.savez_compressed(
                            output / "example_dual_quadrotor.npz",
                            times=results[0].times,
                            positions_0=results[0].positions,
                            positions_1=results[1].positions,
                            reference_0=results[0].reference_positions,
                            reference_1=results[1].reference_positions,
                            center=center,
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
            return np.asarray([float(row[name]) for row in group])

        aggregate.append(
            {
                "scenario": scenario,
                "method": method,
                "runs": len(group),
                "both_pass_rate": float(np.mean(values("both_passed"))),
                "collision_rate": float(np.mean(values("collision"))),
                "margin_violation_rate": float(np.mean(values("margin_violation"))),
                "mean_minimum_separation_m": float(
                    np.mean(values("minimum_separation_m"))
                ),
                "worst_minimum_separation_m": float(
                    np.min(values("minimum_separation_m"))
                ),
                "tracking_rmse_mean_m": float(np.mean(values("tracking_rmse_mean_m"))),
                "tracking_rmse_p95_m": float(
                    np.quantile(values("tracking_rmse_mean_m"), 0.95)
                ),
                "maximum_tracking_error_p95_m": float(
                    np.quantile(values("maximum_tracking_error_m"), 0.95)
                ),
                "maximum_tilt_p95_degrees": float(
                    np.quantile(values("maximum_tilt_degrees"), 0.95)
                ),
                "motor_saturation_mean": float(
                    np.mean(values("motor_saturation_fraction"))
                ),
            }
        )
    aggregate.sort(key=lambda row: (row["scenario"], row["method"]))
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (output / "configuration.json").write_text(
        json.dumps(
            {
                "trajectories": count,
                "seeds": seeds,
                "integration_dt": args.integration_dt,
                "control_dt": args.control_dt,
                "scenarios": {name: SCENARIOS[name] for name in selected_scenarios},
                "quadrotor_parameters": parameters.__dict__,
                "coupling_note": (
                    "HOCBF is applied to the reference before 6-DoF tracking; "
                    "the two rigid-body plants are not coupled by a new online HOCBF."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plotted_scenario = "severe" if "severe" in selected_scenarios else selected_scenarios[-1]
    severe = [row for row in aggregate if row["scenario"] == plotted_scenario]
    x = np.arange(len(severe))
    labels = [row["method"].replace("_", "\n") for row in severe]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    panels = (
        ("margin_violation_rate", f"{plotted_scenario}: rate below 0.36 m"),
        ("mean_minimum_separation_m", f"{plotted_scenario}: mean minimum distance (m)"),
        ("tracking_rmse_mean_m", f"{plotted_scenario}: mean tracking RMSE (m)"),
    )
    for axis, (key, title) in zip(axes, panels, strict=True):
        axis.bar(x, [row[key] for row in severe])
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Phase-2 dual 6-DoF quadrotor simulation")
    figure.tight_layout()
    figure.savefig(output / "dual_quadrotor_severe_comparison.png", dpi=220)
    plt.close(figure)
    print(f"Dual-quadrotor results saved to: {output}")


if __name__ == "__main__":
    main()
