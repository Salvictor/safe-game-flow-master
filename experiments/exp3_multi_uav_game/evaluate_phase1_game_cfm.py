#!/usr/bin/env python3
"""Evaluate joint game-conditioned Flow Matching against Phase-1 baselines."""

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

from safe_game_flow.flow_matching.model import (  # noqa: E402
    FlowMatching1D,
    model_config_from_checkpoint,
)
from safe_game_flow.flow_matching.normalization import denormalize_trajectory  # noqa: E402
from safe_game_flow.flow_matching.train import sample_ode  # noqa: E402
from safe_game_flow.game import simulate_joint_reference  # noqa: E402


METHODS = (
    "nash_reference",
    "teacher_expert",
    "game_cfm_raw",
    "game_cfm_anchored",
    "nash_reference_hocbf",
    "game_cfm_hocbf",
)


def _anchor_residual(residual: np.ndarray) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, residual.shape[-1])
    endpoint_line = (
        (1.0 - phase)[None, None, :] * residual[:, :, :1]
        + phase[None, None, :] * residual[:, :, -1:]
    )
    return residual - endpoint_line


def _dense_trajectory(trajectory: np.ndarray, factor: int = 10) -> np.ndarray:
    points = trajectory.shape[-1]
    source = np.linspace(0.0, 1.0, points)
    query = np.linspace(0.0, 1.0, (points - 1) * factor + 1)
    dense = np.empty((2, 3, len(query)))
    for agent in range(2):
        for axis in range(3):
            dense[agent, axis] = np.interp(query, source, trajectory[agent, axis])
    return dense


def _candidate_score(
    trajectory: np.ndarray,
    residual: np.ndarray,
    horizon: float,
    *,
    acceleration_weight: float = 0.0,
    jerk_weight: float = 0.0,
) -> float:
    """Score a joint candidate without access to its held-out expert label."""
    dense = _dense_trajectory(trajectory)
    minimum_separation = float(
        np.min(np.linalg.norm(dense[0] - dense[1], axis=0))
    )
    sample_times = np.linspace(0.0, horizon, trajectory.shape[-1])
    velocity = np.gradient(trajectory, sample_times, axis=2, edge_order=2)
    acceleration = np.gradient(velocity, sample_times, axis=2, edge_order=2)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    maximum_acceleration = float(np.max(acceleration_norm))
    jerk = np.gradient(acceleration, sample_times, axis=2, edge_order=2)
    jerk_norm = np.linalg.norm(jerk, axis=1)
    safety_penalty = 2_000.0 * max(0.0, 0.38 - minimum_separation) ** 2
    dynamic_penalty = 5.0 * max(0.0, maximum_acceleration - 3.0) ** 2
    residual_penalty = float(np.sqrt(np.mean(residual**2)))
    return (
        safety_penalty
        + dynamic_penalty
        + residual_penalty
        + acceleration_weight * float(np.mean(acceleration_norm**2))
        + jerk_weight * float(np.mean(jerk_norm**2))
    )


def _trajectory_metrics(
    trajectory: np.ndarray,
    expert: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    center: np.ndarray,
    *,
    horizon: float,
) -> dict[str, float]:
    dense = _dense_trajectory(trajectory)
    separation = np.linalg.norm(dense[0] - dense[1], axis=0)
    pass_flags = []
    for agent in range(2):
        x = dense[agent, 0]
        crossing_indices = np.flatnonzero((x[:-1] < 0.0) & (x[1:] >= 0.0))
        passed = False
        for index in crossing_indices:
            denominator = x[index + 1] - x[index]
            fraction = -x[index] / denominator if abs(denominator) > 1e-12 else 0.0
            crossing = dense[agent, :, index] + fraction * (
                dense[agent, :, index + 1] - dense[agent, :, index]
            )
            if np.linalg.norm(crossing[1:] - center[1:]) <= 0.42:
                passed = True
                break
        pass_flags.append(passed)
    sample_times = np.linspace(0.0, horizon, trajectory.shape[-1])
    velocity = np.gradient(trajectory, sample_times, axis=2, edge_order=2)
    acceleration = np.gradient(velocity, sample_times, axis=2, edge_order=2)
    return {
        "both_passed": float(all(pass_flags)),
        "collision": float(np.min(separation) < 0.25),
        "margin_violation": float(np.min(separation) < 0.36),
        "minimum_separation_m": float(np.min(separation)),
        "start_error_m": float(
            np.mean(np.linalg.norm(trajectory[:, :, 0] - starts, axis=1))
        ),
        "goal_error_m": float(
            np.mean(np.linalg.norm(trajectory[:, :, -1] - goals, axis=1))
        ),
        "expert_rmse_m": float(np.sqrt(np.mean((trajectory - expert) ** 2))),
        "maximum_speed_mps": float(np.max(np.linalg.norm(velocity, axis=1))),
        "maximum_acceleration_mps2": float(
            np.max(np.linalg.norm(acceleration, axis=1))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ode-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=53007)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--hocbf-distance",
        type=float,
        default=0.36,
        help="Execution-layer pairwise HOCBF reference distance.",
    )
    parser.add_argument("--score-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--score-jerk-weight", type=float, default=0.0)
    parser.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="Number of Flow Matching candidates sampled per game condition.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectories = np.load(data_dir / "joint_trajectories.npy")
    references = np.load(data_dir / "joint_references.npy")
    disturbances = np.load(data_dir / "disturbances.npy")
    condition_archive = np.load(data_dir / "conditions.npz")
    conditions = condition_archive["values"]
    condition_archive.close()
    if args.max_samples > 0:
        count = min(args.max_samples, len(trajectories))
        trajectories = trajectories[:count]
        references = references[:count]
        disturbances = disturbances[:count]
        conditions = conditions[:count]
    count, channels, points = trajectories.shape
    if args.candidates < 1:
        raise ValueError("candidates must be at least one")
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model = FlowMatching1D(**model_config_from_checkpoint(checkpoint))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    condition_mean = np.asarray(checkpoint["condition_mean"])
    condition_std = np.asarray(checkpoint["condition_std"])
    normalized_conditions = (conditions - condition_mean) / condition_std
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    start = time.perf_counter()
    repeated_conditions = np.repeat(normalized_conditions, args.candidates, axis=0)
    sampled_normalized = sample_ode(
        model,
        count * args.candidates,
        points,
        args.ode_steps,
        torch.device("cpu"),
        channels=channels,
        generator=generator,
        condition=torch.from_numpy(repeated_conditions).float(),
    )
    sampling_seconds = time.perf_counter() - start
    residual_candidates = denormalize_trajectory(
        sampled_normalized,
        checkpoint["mean"],
        checkpoint["std"],
    ).numpy().reshape(count, args.candidates, channels, points)
    anchored_candidates = _anchor_residual(
        residual_candidates.reshape(count * args.candidates, channels, points)
    ).reshape(count, args.candidates, channels, points)
    selected_indices = np.zeros(count, dtype=int)
    for index in range(count):
        horizon = float(conditions[index, -1])
        scores = [
            _candidate_score(
                (references[index] + anchored_candidates[index, candidate]).reshape(
                    2, 3, points
                ),
                anchored_candidates[index, candidate],
                horizon,
                acceleration_weight=args.score_acceleration_weight,
                jerk_weight=args.score_jerk_weight,
            )
            for candidate in range(args.candidates)
        ]
        selected_indices[index] = int(np.argmin(scores))
    residuals = residual_candidates[np.arange(count), selected_indices]
    selected_anchored_residuals = anchored_candidates[np.arange(count), selected_indices]
    raw_plans = references + residuals
    anchored_plans = references + selected_anchored_residuals
    np.save(output / "generated_residuals.npy", residuals)
    np.save(output / "generated_raw_plans.npy", raw_plans)
    np.save(output / "generated_anchored_plans.npy", anchored_plans)
    np.save(output / "selected_candidate_indices.npy", selected_indices)
    rows: list[dict[str, float | int | str]] = []
    executed_reference = np.empty_like(trajectories)
    executed_cfm = np.empty_like(trajectories)
    execution_times = np.linspace(0.0, 5.5, points)
    for index in range(count):
        starts = conditions[index, :6].reshape(2, 3)
        goals = conditions[index, 6:12].reshape(2, 3)
        center = conditions[index, 12:15]
        horizon = float(conditions[index, -1])
        candidates = {
            "nash_reference": references[index].reshape(2, 3, points),
            "teacher_expert": trajectories[index].reshape(2, 3, points),
            "game_cfm_raw": raw_plans[index].reshape(2, 3, points),
            "game_cfm_anchored": anchored_plans[index].reshape(2, 3, points),
        }
        rollout_reference = simulate_joint_reference(
            starts,
            candidates["nash_reference"],
            disturbances[index],
            horizon=horizon,
            dt=0.01,
            use_hocbf=True,
            hocbf_distance=args.hocbf_distance,
        )
        rollout_cfm = simulate_joint_reference(
            starts,
            candidates["game_cfm_anchored"],
            disturbances[index],
            horizon=horizon,
            dt=0.01,
            use_hocbf=True,
            hocbf_distance=args.hocbf_distance,
        )
        for name, rollout in (
            ("nash_reference_hocbf", rollout_reference),
            ("game_cfm_hocbf", rollout_cfm),
        ):
            sampled = np.empty((2, 3, points))
            for agent in range(2):
                for axis in range(3):
                    sampled[agent, axis] = np.interp(
                        execution_times, rollout.times, rollout.positions[:, agent, axis]
                    )
            candidates[name] = sampled
            if name == "nash_reference_hocbf":
                executed_reference[index] = sampled.reshape(6, points)
            else:
                executed_cfm[index] = sampled.reshape(6, points)
        for method in METHODS:
            metrics = _trajectory_metrics(
                candidates[method],
                trajectories[index].reshape(2, 3, points),
                starts,
                goals,
                center,
                horizon=horizon,
            )
            metrics["hocbf_intervention_rate"] = 0.0
            metrics["mean_hocbf_correction"] = 0.0
            if method == "nash_reference_hocbf":
                metrics["hocbf_intervention_rate"] = rollout_reference.intervention_rate
                metrics["mean_hocbf_correction"] = rollout_reference.mean_correction
            elif method == "game_cfm_hocbf":
                metrics["hocbf_intervention_rate"] = rollout_cfm.intervention_rate
                metrics["mean_hocbf_correction"] = rollout_cfm.mean_correction
            rows.append({"sample": index, "method": method, **metrics})
    np.save(output / "executed_nash_reference.npy", executed_reference)
    np.save(output / "executed_game_cfm.npy", executed_cfm)
    with (output / "per_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    aggregate = []
    for method in METHODS:
        group = grouped[method]
        aggregate.append(
            {
                "method": method,
                "samples": len(group),
                "both_pass_rate": float(np.mean([row["both_passed"] for row in group])),
                "collision_rate": float(np.mean([row["collision"] for row in group])),
                "margin_violation_rate": float(
                    np.mean([row["margin_violation"] for row in group])
                ),
                "mean_minimum_separation_m": float(
                    np.mean([row["minimum_separation_m"] for row in group])
                ),
                "worst_minimum_separation_m": float(
                    np.min([row["minimum_separation_m"] for row in group])
                ),
                "mean_start_error_m": float(
                    np.mean([row["start_error_m"] for row in group])
                ),
                "mean_goal_error_m": float(
                    np.mean([row["goal_error_m"] for row in group])
                ),
                "mean_expert_rmse_m": float(
                    np.mean([row["expert_rmse_m"] for row in group])
                ),
                "mean_maximum_speed_mps": float(
                    np.mean([row["maximum_speed_mps"] for row in group])
                ),
                "mean_maximum_acceleration_mps2": float(
                    np.mean([row["maximum_acceleration_mps2"] for row in group])
                ),
                "mean_hocbf_intervention_rate": float(
                    np.mean([row["hocbf_intervention_rate"] for row in group])
                ),
                "mean_hocbf_correction": float(
                    np.mean([row["mean_hocbf_correction"] for row in group])
                ),
                "batch_sampling_ms_per_trajectory": (
                    sampling_seconds * 1000.0 / count if "game_cfm" in method else 0.0
                ),
            }
        )
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
                "checkpoint": str(args.checkpoint.resolve()),
                "data_dir": str(data_dir),
                "samples": count,
                "ode_steps": args.ode_steps,
                "candidates_per_condition": args.candidates,
                "execution_hocbf_distance_m": args.hocbf_distance,
                "score_acceleration_weight": args.score_acceleration_weight,
                "score_jerk_weight": args.score_jerk_weight,
                "sampling_seconds": sampling_seconds,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(aggregate))
    labels = [item["method"].replace("_", "\n") for item in aggregate]
    panels = (
        ("margin_violation_rate", "Rate below 0.36 m margin"),
        ("mean_expert_rmse_m", "Mean expert RMSE (m)"),
        ("mean_hocbf_intervention_rate", "Execution HOCBF intervention"),
    )
    for axis, (key, title) in zip(axes, panels, strict=True):
        axis.bar(x, [item[key] for item in aggregate])
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Experiment 3 Phase 1: game-conditioned joint Flow Matching")
    figure.tight_layout()
    figure.savefig(output / "phase1_game_cfm_comparison.png", dpi=220)
    plt.close(figure)
    print(f"Phase-1 evaluation saved to: {output}")


if __name__ == "__main__":
    main()
