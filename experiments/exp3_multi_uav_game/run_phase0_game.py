#!/usr/bin/env python3
"""Phase-0 two-UAV competitive ring-access Monte Carlo experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safe_game_flow.game import project_inter_uav_hocbf, select_pure_nash_schedule  # noqa: E402
from safe_game_flow.trajectories import fit_quintic_segment  # noqa: E402


METHODS = (
    "independent",
    "fixed_priority",
    "hocbf_only",
    "nash_game",
    "nash_game_hocbf",
)


@dataclass(frozen=True)
class Reference:
    start: np.ndarray
    goal: np.ndarray
    center: np.ndarray
    duration: float
    delay: float
    first_duration: float
    first: object
    second: object

    @property
    def crossing_time(self) -> float:
        return self.delay + self.first_duration

    def evaluate(self, time_value: float, derivative: int = 0) -> np.ndarray:
        local = np.clip(time_value - self.delay, 0.0, self.duration)
        if time_value < self.delay:
            return np.zeros(3) if derivative else self.start.copy()
        if local <= self.first_duration:
            return self.first.evaluate(np.asarray([local]), derivative)[0]
        return self.second.evaluate(
            np.asarray([local - self.first_duration]), derivative
        )[0]


def _make_reference(
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    duration: float,
    delay: float,
) -> Reference:
    first_distance = np.linalg.norm(center - start)
    second_distance = np.linalg.norm(goal - center)
    first_duration = duration * first_distance / (first_distance + second_distance)
    second_duration = duration - first_duration
    zero = np.zeros(3)
    crossing_velocity = np.array([0.55, 0.0, 0.0])
    first = fit_quintic_segment(
        start, zero, zero, center, crossing_velocity, zero, first_duration
    )
    second = fit_quintic_segment(
        center, crossing_velocity, zero, goal, zero, zero, second_duration
    )
    return Reference(
        start, goal, center, duration, delay, first_duration, first, second
    )


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm <= limit else vector * limit / norm


def _crossing_time(times: np.ndarray, positions: np.ndarray) -> float:
    signed = positions[:, 0]
    for index in range(len(signed) - 1):
        if signed[index] < 0.0 <= signed[index + 1]:
            fraction = -signed[index] / (signed[index + 1] - signed[index])
            return float(times[index] + fraction * (times[index + 1] - times[index]))
    return float("nan")


def _actual_payoffs(crossing: tuple[float, float], collided: bool) -> tuple[float, float]:
    if collided or not np.all(np.isfinite(crossing)):
        return (-20.0, -20.0)
    first = 0 if crossing[0] < crossing[1] else 1
    rewards = [7.0, 7.0]
    rewards[first] = 10.0
    return (rewards[0] - 0.5 * crossing[0], rewards[1] - 0.5 * crossing[1])


def _simulate(
    starts: np.ndarray,
    goals: np.ndarray,
    center: np.ndarray,
    durations: tuple[float, float],
    delays: tuple[float, float],
    *,
    use_hocbf: bool,
    disturbance: np.ndarray,
    dt: float,
    physical_collision_distance: float = 0.25,
    hocbf_distance: float = 0.36,
) -> dict:
    references = tuple(
        _make_reference(starts[index], goals[index], center, durations[index], delays[index])
        for index in range(2)
    )
    total_time = max(delays) + max(durations) + 0.5
    times = np.arange(0.0, total_time + 0.5 * dt, dt)
    positions = starts.copy()
    velocities = np.zeros((2, 3))
    position_history = np.empty((len(times), 2, 3))
    velocity_history = np.empty_like(position_history)
    acceleration_history = np.empty_like(position_history)
    interventions = 0
    correction_sum = 0.0
    for time_index, time_value in enumerate(times):
        position_history[time_index] = positions
        velocity_history[time_index] = velocities
        nominal = np.empty((2, 3))
        for agent in range(2):
            reference_position = references[agent].evaluate(time_value, 0)
            reference_velocity = references[agent].evaluate(time_value, 1)
            reference_acceleration = references[agent].evaluate(time_value, 2)
            nominal[agent] = _clip_norm(
                reference_acceleration
                + 5.0 * (reference_position - positions[agent])
                + 3.5 * (reference_velocity - velocities[agent]),
                3.0,
            )
        if use_hocbf:
            projected = project_inter_uav_hocbf(
                positions,
                velocities,
                nominal,
                minimum_distance=hocbf_distance,
            )
            acceleration = projected.accelerations
            interventions += int(projected.intervened)
            correction_sum += projected.correction_norm
        else:
            acceleration = nominal
        acceleration_history[time_index] = acceleration
        if time_index == len(times) - 1:
            break
        velocities = velocities + (acceleration + disturbance) * dt
        positions = positions + velocities * dt

    separation = np.linalg.norm(
        position_history[:, 0, :] - position_history[:, 1, :], axis=1
    )
    crossings = (
        _crossing_time(times, position_history[:, 0]),
        _crossing_time(times, position_history[:, 1]),
    )
    radial_crossings = []
    for agent, crossing in enumerate(crossings):
        if np.isfinite(crossing):
            index = int(np.clip(round(crossing / dt), 0, len(times) - 1))
            radial_crossings.append(
                float(np.linalg.norm(position_history[index, agent, 1:3] - center[1:3]))
            )
        else:
            radial_crossings.append(float("inf"))
    both_passed = bool(np.all(np.isfinite(crossings)) and max(radial_crossings) <= 0.42)
    collided = bool(np.min(separation) < physical_collision_distance)
    payoffs = _actual_payoffs(crossings, collided)
    return {
        "both_passed": int(both_passed),
        "collision": int(collided),
        "deadlock": int(not np.all(np.isfinite(crossings))),
        "minimum_separation_m": float(np.min(separation)),
        "crossing_time_0_s": crossings[0],
        "crossing_time_1_s": crossings[1],
        "crossing_gap_s": abs(crossings[0] - crossings[1]),
        "completion_time_s": float(np.nanmax(crossings)),
        "payoff_0": payoffs[0],
        "payoff_1": payoffs[1],
        "social_payoff": sum(payoffs),
        "hocbf_intervention_rate": interventions / len(times),
        "mean_hocbf_correction": correction_sum / max(interventions, 1),
        "maximum_acceleration_mps2": float(
            np.max(np.linalg.norm(acceleration_history, axis=2))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=128)
    parser.add_argument("--seeds", default="1001,1002,1003")
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",")]
    rows: list[dict] = []
    for run_seed in seeds:
        for scenario_index in range(args.scenarios):
            rng = np.random.default_rng(run_seed * 100_000 + scenario_index)
            center = np.array([0.0, 0.0, rng.uniform(1.0, 1.5)])
            lateral = rng.uniform(0.18, 0.38, size=2)
            starts = np.array(
                [
                    [-rng.uniform(1.3, 1.8), -lateral[0], center[2] + rng.uniform(-0.08, 0.08)],
                    [-rng.uniform(1.3, 1.8), lateral[1], center[2] + rng.uniform(-0.08, 0.08)],
                ]
            )
            goals = np.array(
                [
                    [rng.uniform(1.3, 1.8), -lateral[0], center[2]],
                    [rng.uniform(1.3, 1.8), lateral[1], center[2]],
                ]
            )
            durations = (rng.uniform(3.6, 4.4), rng.uniform(3.6, 4.4))
            zero_delay_refs = (
                _make_reference(starts[0], goals[0], center, durations[0], 0.0),
                _make_reference(starts[1], goals[1], center, durations[1], 0.0),
            )
            nominal_crossings = tuple(item.crossing_time for item in zero_delay_refs)
            schedule = select_pure_nash_schedule(nominal_crossings)
            priority = 0 if nominal_crossings[0] <= nominal_crossings[1] else 1
            fixed_delays = (0.0, 0.8)
            disturbance = rng.normal(0.0, 0.08, size=(2, 3))
            configurations = {
                "independent": ((0.0, 0.0), False),
                "fixed_priority": (fixed_delays, False),
                "hocbf_only": ((0.0, 0.0), True),
                "nash_game": (schedule.delays, False),
                "nash_game_hocbf": (schedule.delays, True),
            }
            for method, (delays, use_hocbf) in configurations.items():
                result = _simulate(
                    starts,
                    goals,
                    center,
                    durations,
                    delays,
                    use_hocbf=use_hocbf,
                    disturbance=disturbance,
                    dt=args.dt,
                )
                rows.append(
                    {
                        "seed": run_seed,
                        "scenario_index": scenario_index,
                        "method": method,
                        "nominal_crossing_0_s": nominal_crossings[0],
                        "nominal_crossing_1_s": nominal_crossings[1],
                        "nash_action_0": schedule.actions[0],
                        "nash_action_1": schedule.actions[1],
                        "nominal_priority_agent": priority,
                        "delay_0_s": delays[0],
                        "delay_1_s": delays[1],
                        **result,
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
        aggregate.append(
            {
                "method": method,
                "runs": len(group),
                "both_pass_rate": float(np.mean(values("both_passed"))),
                "collision_rate": float(np.mean(values("collision"))),
                "deadlock_rate": float(np.mean(values("deadlock"))),
                "mean_minimum_separation_m": float(np.mean(values("minimum_separation_m"))),
                "minimum_separation_m": float(np.min(values("minimum_separation_m"))),
                "mean_crossing_gap_s": float(np.mean(values("crossing_gap_s"))),
                "mean_completion_time_s": float(np.mean(values("completion_time_s"))),
                "mean_social_payoff": float(np.mean(values("social_payoff"))),
                "hocbf_intervention_rate": float(np.mean(values("hocbf_intervention_rate"))),
                "mean_maximum_acceleration_mps2": float(np.mean(values("maximum_acceleration_mps2"))),
            }
        )
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (output / "configuration.json").write_text(
        json.dumps(
            {
                "scenarios": args.scenarios,
                "seeds": seeds,
                "dt": args.dt,
                "dynamics": "paired 3D double integrator",
                "physical_collision_distance": 0.25,
                "hocbf_distance": 0.36,
                "game_actions": ["go", "yield"],
                "game_safety_gap": 0.45,
                "game_safety_buffer": 0.05,
                "game_delay": "minimum scene-dependent safe delay",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(aggregate))
    labels = [item["method"].replace("_", "\n") for item in aggregate]
    plots = (
        ("collision_rate", "Collision rate"),
        ("mean_completion_time_s", "Mean completion time (s)"),
        ("mean_social_payoff", "Mean social payoff"),
    )
    for axis, (metric, title) in zip(axes, plots, strict=True):
        axis.bar(x, [item[metric] for item in aggregate])
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.0, 1.0)
    figure.suptitle("Experiment 3 Phase 0: two-UAV non-cooperative ring access")
    figure.tight_layout()
    figure.savefig(output / "phase0_game_comparison.png", dpi=220)
    plt.close(figure)
    print(f"Phase-0 game results saved to: {output}")


if __name__ == "__main__":
    main()
