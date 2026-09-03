#!/usr/bin/env python3
"""Generate reproducible, feasible 3D expert trajectories through circular rings."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import DATASET_CONFIG, DATA_DIR, DEFAULT_NUM_TRAJECTORIES, DEFAULT_SEED  # noqa: E402
from safe_game_flow.data.ring_trajectories import CONDITION_SCHEMA, generate_ring_dataset  # noqa: E402
from safe_game_flow.trajectories import encode_boundary_residual  # noqa: E402


def _ring_circle(center: np.ndarray, normal: np.ndarray, radius: float) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angles = np.linspace(0, 2 * np.pi, 100)
    return center + radius * (
        np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    )


def save_preview(samples, path: Path, maximum: int = 30) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for sample in samples[:maximum]:
        trajectory = sample.trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.45)
    representative = samples[0].ring
    circle = _ring_circle(
        representative.center, representative.normal, representative.major_radius
    )
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color="black", lw=2, label="Example ring")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Single-UAV ring traversal expert trajectories")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trajectories", type=int, default=DEFAULT_NUM_TRAJECTORIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--endpoint-offset-fraction",
        type=float,
        default=DATASET_CONFIG.endpoint_offset_fraction,
        help="maximum start/goal in-plane offset divided by clear aperture radius",
    )
    parser.add_argument(
        "--crossing-offset-fraction",
        type=float,
        default=DATASET_CONFIG.crossing_offset_fraction,
    )
    args = parser.parse_args()

    if args.endpoint_offset_fraction < 0 or args.crossing_offset_fraction < 0:
        raise ValueError("offset fractions must be non-negative")
    dataset_config = replace(
        DATASET_CONFIG,
        endpoint_offset_fraction=args.endpoint_offset_fraction,
        crossing_offset_fraction=args.crossing_offset_fraction,
    )
    samples = generate_ring_dataset(args.num_trajectories, args.seed, dataset_config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = np.stack([sample.trajectory.T for sample in samples])  # (N,3,H)
    velocities = np.stack([sample.velocity.T for sample in samples])
    accelerations = np.stack([sample.acceleration.T for sample in samples])
    conditions = np.stack([sample.condition for sample in samples])
    boundary_residuals = encode_boundary_residual(
        np.transpose(trajectories, (0, 2, 1)), conditions, CONDITION_SCHEMA
    )
    np.save(output_dir / "trajectories.npy", trajectories)
    np.save(output_dir / "boundary_residuals.npy", np.transpose(boundary_residuals, (0, 2, 1)))
    np.save(output_dir / "velocities.npy", velocities)
    np.save(output_dir / "accelerations.npy", accelerations)
    np.savez_compressed(
        output_dir / "conditions.npz",
        values=conditions,
        schema=np.asarray(CONDITION_SCHEMA),
    )

    summary = {
        "num_samples": len(samples),
        "seed": args.seed,
        "trajectory_shape": list(trajectories.shape),
        "condition_dimension": conditions.shape[1],
        "successful_ring_traversals": len(samples),
        "ring_pass_rate": 1.0,
        "minimum_barrier": float(min(sample.min_barrier for sample in samples)),
        "mean_minimum_barrier": float(np.mean([sample.min_barrier for sample in samples])),
        "maximum_speed": float(max(sample.max_speed for sample in samples)),
        "maximum_acceleration": float(max(sample.max_acceleration for sample in samples)),
        "mean_path_length": float(np.mean([sample.path_length for sample in samples])),
        "configured_speed_limit": dataset_config.maximum_speed,
        "configured_acceleration_limit": dataset_config.maximum_acceleration,
        "dataset_config": asdict(dataset_config),
    }
    metadata = {
        "coordinate_frame": "world",
        "units": {"position": "m", "velocity": "m/s", "acceleration": "m/s^2"},
        "trajectory_layout": "(N, 3, H)",
        "learned_representation": (
            "boundary_residuals.npy stores residuals around a quintic PVA-feasible skeleton; "
            "layout (N, 3, H)"
        ),
        "condition_schema": CONDITION_SCHEMA,
        "generator": "two C2-connected quintic polynomial segments",
        "dataset_config": asdict(dataset_config),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if not args.no_preview:
        save_preview(samples, output_dir / "dataset_preview.png")

    print(json.dumps(summary, indent=2))
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
