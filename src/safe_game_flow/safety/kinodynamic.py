"""Tracking-aware kinodynamic projection for boundary-conditioned residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from safe_game_flow.evaluation import RingTrajectoryMetrics, evaluate_ring_trajectory
from safe_game_flow.trajectories import decode_boundary_residual


@dataclass(frozen=True)
class KinodynamicProjectionResult:
    residual: np.ndarray
    trajectory: np.ndarray
    scale: float
    metrics: RingTrajectoryMetrics
    strategy: str = "unknown"


def _is_feasible(metrics: RingTrajectoryMetrics) -> bool:
    return (
        metrics.passed_ring
        and metrics.collision_free
        and metrics.dynamically_feasible
        and metrics.endpoint_feasible
    )


def _maximum_affine_norm_scale(
    base: np.ndarray,
    delta: np.ndarray,
    limit: float,
) -> float:
    """Largest scale in [0,1] satisfying ||base + scale*delta|| <= limit.

    The zero-residual boundary skeleton is assumed feasible. Each time sample
    yields a scalar quadratic interval containing zero; intersecting their
    positive roots gives the exact scale for the discretized derivative metric.
    """
    base = np.asarray(base, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if base.shape != delta.shape or base.ndim != 2:
        raise ValueError("base and delta must have matching (H,D) shapes")
    if np.any(np.linalg.norm(base, axis=1) > limit + 1e-8):
        raise ValueError("zero-residual skeleton violates the requested limit")
    quadratic = np.sum(delta * delta, axis=1)
    linear = 2.0 * np.sum(base * delta, axis=1)
    constant = np.sum(base * base, axis=1) - limit**2
    roots = np.ones(len(base))
    curved = quadratic > 1e-14
    discriminant = np.maximum(linear[curved] ** 2 - 4.0 * quadratic[curved] * constant[curved], 0.0)
    roots[curved] = (
        -linear[curved] + np.sqrt(discriminant)
    ) / (2.0 * quadratic[curved])
    linear_only = (~curved) & (linear > 1e-14)
    roots[linear_only] = -constant[linear_only] / linear[linear_only]
    return float(np.clip(np.min(roots), 0.0, 1.0))


def project_boundary_residual_to_feasibility(
    residual: np.ndarray,
    condition: np.ndarray,
    schema: Sequence[str],
    *,
    maximum_speed: float = 2.0,
    maximum_acceleration: float = 3.0,
    endpoint_position_tolerance: float = 0.15,
    endpoint_velocity_tolerance: float = 0.25,
    grid_steps: int = 32,
    refinement_steps: int = 10,
) -> KinodynamicProjectionResult:
    """Find the largest residual scale whose decoded trajectory is feasible.

    Scaling preserves the quintic boundary skeleton and therefore all endpoint
    constraints. A descending grid avoids assuming global monotonicity of the
    non-convex ring constraint; local bisection then refines the best interval.
    """
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2 or residual.shape[1] != 3:
        raise ValueError(f"residual must have shape (H,3), got {residual.shape}")
    if grid_steps < 2 or refinement_steps < 0:
        raise ValueError("grid_steps must be at least 2 and refinement_steps non-negative")

    def evaluate(scale: float) -> tuple[np.ndarray, RingTrajectoryMetrics]:
        trajectory = decode_boundary_residual(scale * residual, condition, schema)
        metrics = evaluate_ring_trajectory(
            trajectory,
            condition,
            schema,
            maximum_speed=maximum_speed,
            maximum_acceleration=maximum_acceleration,
            endpoint_position_tolerance=endpoint_position_tolerance,
            endpoint_velocity_tolerance=endpoint_velocity_tolerance,
        )
        return trajectory, metrics

    full_trajectory, full_metrics = evaluate(1.0)
    if _is_feasible(full_metrics):
        return KinodynamicProjectionResult(
            residual, full_trajectory, 1.0, full_metrics, "unchanged"
        )

    # Speed and acceleration of skeleton + scale*residual are affine in scale.
    # Solve their sampled norm constraints analytically instead of evaluating a
    # 32-point descending grid followed by ten bisection steps.
    zero_residual = np.zeros_like(residual)
    base_trajectory = decode_boundary_residual(zero_residual, condition, schema)
    try:
        time_index = tuple(schema).index("flight_time")
    except ValueError as error:
        raise ValueError("condition schema must contain flight_time") from error
    duration = float(np.asarray(condition)[time_index])
    if duration <= 0:
        raise ValueError("flight_time must be positive")
    dt = duration / (len(residual) - 1)
    full_delta = full_trajectory - base_trajectory
    base_velocity = np.gradient(base_trajectory, dt, axis=0, edge_order=2)
    delta_velocity = np.gradient(full_delta, dt, axis=0, edge_order=2)
    base_acceleration = np.gradient(base_velocity, dt, axis=0, edge_order=2)
    delta_acceleration = np.gradient(delta_velocity, dt, axis=0, edge_order=2)
    analytic_scale = min(
        _maximum_affine_norm_scale(base_velocity, delta_velocity, maximum_speed),
        _maximum_affine_norm_scale(
            base_acceleration, delta_acceleration, maximum_acceleration
        ),
    )
    # Pull back by a tiny numerical margin so the evaluator's <= comparison is
    # not determined by floating-point roundoff at the exact quadratic root.
    analytic_scale = max(0.0, analytic_scale * (1.0 - 1e-6))
    analytic_trajectory, analytic_metrics = evaluate(analytic_scale)
    if _is_feasible(analytic_metrics):
        return KinodynamicProjectionResult(
            residual=analytic_scale * residual,
            trajectory=analytic_trajectory,
            scale=analytic_scale,
            metrics=analytic_metrics,
            strategy="analytic",
        )

    # Rare fallback for non-monotone ring/endpoint feasibility along the scale
    # line. Search only below the analytic dynamic upper bound.
    grid = np.linspace(analytic_scale, 0.0, grid_steps + 1)
    feasible_scale = None
    feasible_trajectory = None
    feasible_metrics = None
    upper_scale = analytic_scale
    for index, scale in enumerate(grid[1:], start=1):
        trajectory, metrics = evaluate(float(scale))
        if _is_feasible(metrics):
            feasible_scale = float(scale)
            feasible_trajectory = trajectory
            feasible_metrics = metrics
            upper_scale = float(grid[index - 1])
            break
    if feasible_scale is None:
        raise RuntimeError(
            "The boundary skeleton is infeasible; expert generation or task limits are inconsistent"
        )

    lower_scale = feasible_scale
    for _ in range(refinement_steps):
        midpoint = 0.5 * (lower_scale + upper_scale)
        trajectory, metrics = evaluate(midpoint)
        if _is_feasible(metrics):
            lower_scale = midpoint
            feasible_trajectory = trajectory
            feasible_metrics = metrics
        else:
            upper_scale = midpoint

    return KinodynamicProjectionResult(
        residual=lower_scale * residual,
        trajectory=feasible_trajectory,
        scale=lower_scale,
        metrics=feasible_metrics,
        strategy="fallback_grid",
    )
