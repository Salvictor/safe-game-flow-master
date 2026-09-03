"""Velocity-profile projection and integration for ring traversal baselines."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _condition_values(condition: np.ndarray, schema: Sequence[str]) -> dict[str, float]:
    condition = np.asarray(condition, dtype=float)
    if condition.ndim != 1 or len(condition) != len(schema):
        raise ValueError(f"condition must have shape ({len(schema)},), got {condition.shape}")
    return dict(zip(schema, condition, strict=True))


def integrate_velocity_profile(
    velocity: np.ndarray,
    condition: np.ndarray,
    schema: Sequence[str],
) -> np.ndarray:
    """Integrate a physical velocity profile with the trapezoidal rule."""
    velocity = np.asarray(velocity, dtype=float)
    if velocity.ndim != 2 or velocity.shape[1] != 3 or len(velocity) < 2:
        raise ValueError(f"velocity must have shape (H,3), H>=2; got {velocity.shape}")
    values = _condition_values(condition, schema)
    duration = values["flight_time"]
    if duration <= 0:
        raise ValueError("flight_time must be positive")
    start = np.array(
        [values["start_position_x"], values["start_position_y"], values["start_position_z"]]
    )
    dt = duration / (len(velocity) - 1)
    increments = 0.5 * dt * (velocity[:-1] + velocity[1:])
    trajectory = np.empty_like(velocity)
    trajectory[0] = start
    trajectory[1:] = start + np.cumsum(increments, axis=0)
    return trajectory


def project_velocity_boundary_constraints(
    velocity: np.ndarray,
    condition: np.ndarray,
    schema: Sequence[str],
) -> np.ndarray:
    """Euclidean projection onto endpoint-velocity and displacement constraints.

    The projected profile exactly matches prescribed start/end velocities and
    has the correct trapezoidal integral, so its integrated position reaches
    the requested goal without applying a post-hoc position warp.
    """
    velocity = np.asarray(velocity, dtype=float)
    if velocity.ndim != 2 or velocity.shape[1] != 3 or len(velocity) < 4:
        raise ValueError(f"velocity must have shape (H,3), H>=4; got {velocity.shape}")
    values = _condition_values(condition, schema)
    duration = values["flight_time"]
    if duration <= 0:
        raise ValueError("flight_time must be positive")
    start_position = np.array(
        [values["start_position_x"], values["start_position_y"], values["start_position_z"]]
    )
    goal_position = np.array(
        [values["goal_position_x"], values["goal_position_y"], values["goal_position_z"]]
    )
    start_velocity = np.array(
        [values["start_velocity_x"], values["start_velocity_y"], values["start_velocity_z"]]
    )
    goal_velocity = np.array(
        [values["goal_velocity_x"], values["goal_velocity_y"], values["goal_velocity_z"]]
    )

    horizon = len(velocity)
    dt = duration / (horizon - 1)
    integral_weights = np.full(horizon, dt)
    integral_weights[[0, -1]] = 0.5 * dt
    constraint = np.zeros((3, horizon))
    constraint[0, 0] = 1.0
    constraint[1, -1] = 1.0
    constraint[2] = integral_weights
    gram_inverse = np.linalg.inv(constraint @ constraint.T)
    targets = np.stack([start_velocity, goal_velocity, goal_position - start_position])
    residual = targets - constraint @ velocity
    return velocity + constraint.T @ (gram_inverse @ residual)
