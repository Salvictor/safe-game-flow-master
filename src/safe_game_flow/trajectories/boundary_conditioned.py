"""Boundary-conditioned residual representation for fixed-horizon trajectories.

The decoder combines a quintic boundary-feasible skeleton with a learned
residual multiplied by ``64 s^3 (1-s)^3``. The multiplier and its first two
derivatives vanish at both endpoints, so decoded continuous trajectories
inherit the prescribed endpoint position, velocity, and zero acceleration.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .quintic import fit_quintic_segment


_REQUIRED_FIELDS = (
    "start_position_x", "start_position_y", "start_position_z",
    "start_velocity_x", "start_velocity_y", "start_velocity_z",
    "goal_position_x", "goal_position_y", "goal_position_z",
    "goal_velocity_x", "goal_velocity_y", "goal_velocity_z",
    "flight_time",
)


def _field_indices(schema: Sequence[str]) -> dict[str, int]:
    if len(set(schema)) != len(schema):
        raise ValueError("condition schema contains duplicate names")
    indices = {name: index for index, name in enumerate(schema)}
    missing = [name for name in _REQUIRED_FIELDS if name not in indices]
    if missing:
        raise ValueError(f"condition schema is missing required fields: {missing}")
    return indices


def boundary_envelope(horizon: int) -> np.ndarray:
    """Return a unit-scaled endpoint-vanishing envelope of shape ``(H,1)``."""
    if horizon < 3:
        raise ValueError("horizon must be at least 3")
    phase = np.linspace(0.0, 1.0, horizon)
    return (64.0 * phase**3 * (1.0 - phase) ** 3)[:, None]


def quintic_boundary_skeleton(
    conditions: np.ndarray,
    horizon: int,
    schema: Sequence[str],
) -> np.ndarray:
    """Construct quintic trajectories satisfying endpoint position/PVA constraints.

    ``conditions`` may have shape ``(D,)`` or ``(N,D)``. The returned layout is
    respectively ``(H,3)`` or ``(N,H,3)``.
    """
    conditions = np.asarray(conditions, dtype=float)
    single = conditions.ndim == 1
    if single:
        conditions = conditions[None, :]
    if conditions.ndim != 2 or conditions.shape[1] != len(schema):
        raise ValueError(
            f"conditions must have shape (D,) or (N,D) with D={len(schema)}, "
            f"got {conditions.shape}"
        )
    if not np.isfinite(conditions).all():
        raise ValueError("conditions contain NaN or infinite values")

    index = _field_indices(schema)
    start_position = conditions[:, [index[f"start_position_{axis}"] for axis in "xyz"]]
    start_velocity = conditions[:, [index[f"start_velocity_{axis}"] for axis in "xyz"]]
    goal_position = conditions[:, [index[f"goal_position_{axis}"] for axis in "xyz"]]
    goal_velocity = conditions[:, [index[f"goal_velocity_{axis}"] for axis in "xyz"]]
    flight_times = conditions[:, index["flight_time"]]
    if np.any(flight_times <= 0):
        raise ValueError("flight_time must be positive")

    zero_acceleration = np.zeros(3)
    skeletons = []
    for p0, v0, p1, v1, duration in zip(
        start_position,
        start_velocity,
        goal_position,
        goal_velocity,
        flight_times,
        strict=True,
    ):
        segment = fit_quintic_segment(
            p0,
            v0,
            zero_acceleration,
            p1,
            v1,
            zero_acceleration,
            float(duration),
        )
        skeletons.append(segment.evaluate(np.linspace(0.0, float(duration), horizon)))
    result = np.stack(skeletons)
    return result[0] if single else result


def encode_boundary_residual(
    trajectories: np.ndarray,
    conditions: np.ndarray,
    schema: Sequence[str],
) -> np.ndarray:
    """Encode physical ``(...,H,3)`` trajectories as boundary-free residuals."""
    trajectories = np.asarray(trajectories, dtype=float)
    single = trajectories.ndim == 2
    if single:
        trajectories = trajectories[None, ...]
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError(f"trajectories must have shape (H,3) or (N,H,3), got {trajectories.shape}")
    skeleton = quintic_boundary_skeleton(conditions, trajectories.shape[1], schema)
    if skeleton.ndim == 2:
        skeleton = skeleton[None, ...]
    if skeleton.shape != trajectories.shape:
        raise ValueError(
            f"trajectory/condition batch mismatch: {trajectories.shape} versus {skeleton.shape}"
        )
    envelope = boundary_envelope(trajectories.shape[1])
    residual = np.zeros_like(trajectories)
    mask = envelope[:, 0] > 1e-14
    residual[:, mask, :] = (
        trajectories[:, mask, :] - skeleton[:, mask, :]
    ) / envelope[mask][None, :, :]
    return residual[0] if single else residual


def decode_boundary_residual(
    residuals: np.ndarray,
    conditions: np.ndarray,
    schema: Sequence[str],
) -> np.ndarray:
    """Decode residuals to physical trajectories with structural endpoint constraints."""
    residuals = np.asarray(residuals, dtype=float)
    single = residuals.ndim == 2
    if single:
        residuals = residuals[None, ...]
    if residuals.ndim != 3 or residuals.shape[-1] != 3:
        raise ValueError(f"residuals must have shape (H,3) or (N,H,3), got {residuals.shape}")
    skeleton = quintic_boundary_skeleton(conditions, residuals.shape[1], schema)
    if skeleton.ndim == 2:
        skeleton = skeleton[None, ...]
    if skeleton.shape != residuals.shape:
        raise ValueError(
            f"residual/condition batch mismatch: {residuals.shape} versus {skeleton.shape}"
        )
    decoded = skeleton + boundary_envelope(residuals.shape[1])[None, :, :] * residuals
    return decoded[0] if single else decoded
