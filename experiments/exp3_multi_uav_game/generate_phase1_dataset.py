#!/usr/bin/env python3
"""Generate Nash+HOCBF expert trajectories for game-conditioned Flow Matching."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safe_game_flow.game import (  # noqa: E402
    CONDITION_SCHEMA,
    condition_vector,
    nominal_crossing_times,
    resample_joint_positions,
    resample_reference_positions,
    sample_scene,
    scheduled_references,
    select_pure_nash_schedule,
    simulate_expert,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--horizon", type=float, default=5.5)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()
    if args.samples < 1 or args.points < 2:
        raise ValueError("samples must be positive and points must be at least two")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectories = np.empty((args.samples, 6, args.points), dtype=np.float32)
    references = np.empty_like(trajectories)
    conditions = np.empty((args.samples, len(CONDITION_SCHEMA)), dtype=np.float32)
    disturbances = np.empty((args.samples, 2, 3), dtype=np.float32)
    minimum_separations = np.empty(args.samples)
    intervention_rates = np.empty(args.samples)
    mean_corrections = np.empty(args.samples)
    collision_flags = np.empty(args.samples, dtype=bool)
    actions = np.empty((args.samples, 2), dtype=np.int8)
    delays = np.empty((args.samples, 2), dtype=np.float32)
    start_time = time.perf_counter()
    for index in range(args.samples):
        rng = np.random.default_rng(args.seed * 1_000_003 + index)
        scene = sample_scene(rng)
        schedule = select_pure_nash_schedule(nominal_crossing_times(scene))
        rollout = simulate_expert(
            scene,
            schedule,
            horizon=args.horizon,
            dt=args.dt,
            use_hocbf=True,
        )
        scheduled = scheduled_references(scene, schedule)
        trajectories[index] = resample_joint_positions(rollout, args.points)
        references[index] = resample_reference_positions(
            scheduled, horizon=args.horizon, points=args.points
        )
        conditions[index] = condition_vector(scene, schedule, horizon=args.horizon)
        disturbances[index] = scene.disturbance
        minimum_separations[index] = rollout.minimum_separation
        intervention_rates[index] = rollout.intervention_rate
        mean_corrections[index] = rollout.mean_correction
        collision_flags[index] = rollout.collision
        actions[index] = [int(action == "yield") for action in schedule.actions]
        delays[index] = schedule.delays
        if (index + 1) % max(1, min(100, args.samples)) == 0:
            elapsed = time.perf_counter() - start_time
            print(f"Generated {index + 1}/{args.samples} experts in {elapsed:.1f} s")
    residuals = trajectories - references
    np.save(output / "joint_trajectories.npy", trajectories)
    np.save(output / "joint_references.npy", references)
    np.save(output / "joint_residuals.npy", residuals)
    np.save(output / "disturbances.npy", disturbances)
    np.savez(
        output / "conditions.npz",
        values=conditions,
        schema=np.asarray(CONDITION_SCHEMA),
    )
    np.savez(
        output / "expert_metrics.npz",
        minimum_separations=minimum_separations,
        intervention_rates=intervention_rates,
        mean_corrections=mean_corrections,
        collisions=collision_flags,
        actions=actions,
        delays=delays,
    )
    summary = {
        "samples": args.samples,
        "seed": args.seed,
        "points": args.points,
        "horizon_s": args.horizon,
        "integration_step_s": args.dt,
        "trajectory_shape": list(trajectories.shape),
        "condition_dimension": len(CONDITION_SCHEMA),
        "condition_schema": list(CONDITION_SCHEMA),
        "collision_rate": float(np.mean(collision_flags)),
        "hocbf_margin_violation_rate": float(np.mean(minimum_separations < 0.36)),
        "mean_minimum_separation_m": float(np.mean(minimum_separations)),
        "worst_minimum_separation_m": float(np.min(minimum_separations)),
        "mean_hocbf_intervention_rate": float(np.mean(intervention_rates)),
        "mean_hocbf_correction": float(np.mean(mean_corrections)),
        "residual_rms_m": float(np.sqrt(np.mean(residuals**2))),
        "yield_fraction_agent_0": float(np.mean(actions[:, 0])),
        "yield_fraction_agent_1": float(np.mean(actions[:, 1])),
        "mean_selected_delay_s": float(np.mean(np.max(delays, axis=1))),
        "generation_seconds": time.perf_counter() - start_time,
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
