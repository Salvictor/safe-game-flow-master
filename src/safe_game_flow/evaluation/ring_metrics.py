"""Metrics for a single UAV traversing a three-dimensional circular ring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from safe_game_flow.data.ring_trajectories import CONDITION_SCHEMA
from safe_game_flow.geometry import RingGeometry


def _condition_mapping(
    condition: np.ndarray,
    schema: Sequence[str] = CONDITION_SCHEMA,
) -> dict[str, float]:
    values = np.asarray(condition, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"condition must have shape (D,), got {values.shape}")
    if len(schema) != len(values):
        raise ValueError(f"schema has {len(schema)} names for D={len(values)}")
    if len(set(schema)) != len(schema):
        raise ValueError("condition schema contains duplicate names")
    return dict(zip(schema, values, strict=True))


def ring_from_condition(
    condition: np.ndarray,
    schema: Sequence[str] = CONDITION_SCHEMA,
) -> RingGeometry:
    """Reconstruct ring geometry from a saved task-condition vector."""
    values = _condition_mapping(condition, schema)
    return RingGeometry(
        center=np.array(
            [values["ring_center_x"], values["ring_center_y"], values["ring_center_z"]]
        ),
        normal=np.array(
            [values["ring_normal_x"], values["ring_normal_y"], values["ring_normal_z"]]
        ),
        major_radius=values["ring_major_radius"],
        tube_radius=values["ring_tube_radius"],
        uav_radius=values["uav_radius"],
        margin=values["margin"],
    )


def _vector(values: dict[str, float], prefix: str) -> np.ndarray:
    return np.array([values[f"{prefix}_x"], values[f"{prefix}_y"], values[f"{prefix}_z"]])


def _densify_piecewise_linear(trajectory: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("dense_factor must be at least 1")
    if factor == 1:
        return trajectory
    source = np.arange(len(trajectory), dtype=float)
    target = np.linspace(0.0, len(trajectory) - 1, (len(trajectory) - 1) * factor + 1)
    return np.stack([np.interp(target, source, trajectory[:, axis]) for axis in range(3)], axis=1)


@dataclass(frozen=True)
class RingTrajectoryMetrics:
    """Per-trajectory task, safety, dynamics, and endpoint metrics."""

    successful: bool
    passed_ring: bool
    collision_free: bool
    dynamically_feasible: bool
    endpoint_feasible: bool
    minimum_barrier: float
    crossing_clearance: float
    maximum_speed: float
    maximum_acceleration: float
    path_length: float
    start_position_error: float
    goal_position_error: float
    start_velocity_error: float
    goal_velocity_error: float

    def to_dict(self) -> dict[str, bool | float]:
        return asdict(self)


def evaluate_ring_trajectory(
    trajectory: np.ndarray,
    condition: np.ndarray,
    schema: Sequence[str] = CONDITION_SCHEMA,
    *,
    maximum_speed: float = 2.0,
    maximum_acceleration: float = 3.0,
    endpoint_position_tolerance: float = 0.15,
    endpoint_velocity_tolerance: float = 0.25,
    dense_factor: int = 10,
) -> RingTrajectoryMetrics:
    """Evaluate one physical trajectory against its unnormalized task condition."""
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) < 3:
        raise ValueError(f"trajectory must have shape (H,3), H>=3; got {trajectory.shape}")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory contains NaN or infinite values")
    if maximum_speed <= 0 or maximum_acceleration <= 0:
        raise ValueError("dynamic limits must be positive")

    values = _condition_mapping(condition, schema)
    ring = ring_from_condition(condition, schema)
    flight_time = values["flight_time"]
    if flight_time <= 0:
        raise ValueError("flight_time must be positive")

    dense_trajectory = _densify_piecewise_linear(trajectory, dense_factor)
    crossing = ring.detect_ring_crossing(dense_trajectory, direction=1)
    minimum_barrier = float(np.min(ring.barrier_value(dense_trajectory)))
    collision_free = minimum_barrier >= -1e-9

    dt = flight_time / (len(trajectory) - 1)
    velocity = np.gradient(trajectory, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    maximum_observed_speed = float(np.max(np.linalg.norm(velocity, axis=1)))
    maximum_observed_acceleration = float(np.max(np.linalg.norm(acceleration, axis=1)))
    dynamically_feasible = (
        maximum_observed_speed <= maximum_speed
        and maximum_observed_acceleration <= maximum_acceleration
    )

    start_position_error = float(
        np.linalg.norm(trajectory[0] - _vector(values, "start_position"))
    )
    goal_position_error = float(
        np.linalg.norm(trajectory[-1] - _vector(values, "goal_position"))
    )
    start_velocity_error = float(
        np.linalg.norm(velocity[0] - _vector(values, "start_velocity"))
    )
    goal_velocity_error = float(
        np.linalg.norm(velocity[-1] - _vector(values, "goal_velocity"))
    )
    endpoint_feasible = (
        start_position_error <= endpoint_position_tolerance
        and goal_position_error <= endpoint_position_tolerance
        and start_velocity_error <= endpoint_velocity_tolerance
        and goal_velocity_error <= endpoint_velocity_tolerance
    )
    path_length = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))
    crossing_clearance = (
        float(crossing.aperture_clearance)
        if crossing.aperture_clearance is not None
        else float("nan")
    )
    passed_ring = crossing.passed
    successful = passed_ring and collision_free and dynamically_feasible and endpoint_feasible
    return RingTrajectoryMetrics(
        successful=successful,
        passed_ring=passed_ring,
        collision_free=collision_free,
        dynamically_feasible=dynamically_feasible,
        endpoint_feasible=endpoint_feasible,
        minimum_barrier=minimum_barrier,
        crossing_clearance=crossing_clearance,
        maximum_speed=maximum_observed_speed,
        maximum_acceleration=maximum_observed_acceleration,
        path_length=path_length,
        start_position_error=start_position_error,
        goal_position_error=goal_position_error,
        start_velocity_error=start_velocity_error,
        goal_velocity_error=goal_velocity_error,
    )


def summarize_ring_metrics(
    metrics: Sequence[RingTrajectoryMetrics],
) -> dict[str, int | float]:
    """Aggregate per-trajectory metrics into experiment-level statistics."""
    if not metrics:
        raise ValueError("metrics must contain at least one trajectory")

    def rate(name: str) -> float:
        return float(np.mean([bool(getattr(item, name)) for item in metrics]))

    def values(name: str) -> np.ndarray:
        return np.asarray([float(getattr(item, name)) for item in metrics])

    crossing_clearance = values("crossing_clearance")
    finite_clearance = crossing_clearance[np.isfinite(crossing_clearance)]
    return {
        "num_trajectories": len(metrics),
        "success_rate": rate("successful"),
        "ring_pass_rate": rate("passed_ring"),
        "collision_free_rate": rate("collision_free"),
        "dynamic_feasibility_rate": rate("dynamically_feasible"),
        "endpoint_feasibility_rate": rate("endpoint_feasible"),
        "minimum_barrier": float(np.min(values("minimum_barrier"))),
        "mean_minimum_barrier": float(np.mean(values("minimum_barrier"))),
        "mean_crossing_clearance": (
            float(np.mean(finite_clearance)) if len(finite_clearance) else float("nan")
        ),
        "mean_maximum_speed": float(np.mean(values("maximum_speed"))),
        "maximum_speed": float(np.max(values("maximum_speed"))),
        "mean_maximum_acceleration": float(np.mean(values("maximum_acceleration"))),
        "maximum_acceleration": float(np.max(values("maximum_acceleration"))),
        "mean_path_length": float(np.mean(values("path_length"))),
        "mean_start_position_error": float(np.mean(values("start_position_error"))),
        "mean_goal_position_error": float(np.mean(values("goal_position_error"))),
    }
