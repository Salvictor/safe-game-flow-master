"""Trajectory and velocity normalization helpers.

Trajectory tensors use ``(..., C, H)`` layout. Statistics use ``(C, 1)``
and are broadcast over leading batch dimensions. Positions are normalized as
``(x - mean) / std`` while velocities only change scale because translations
do not affect derivatives.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import torch


ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)


def _validate_shapes(data: ArrayLike, mean: ArrayLike, std: ArrayLike) -> None:
    if data.ndim < 2:
        raise ValueError(f"Expected data with shape (..., C, H), got {data.shape}")
    channels = data.shape[-2]
    if mean.shape != (channels, 1):
        raise ValueError(f"Expected mean shape ({channels}, 1), got {mean.shape}")
    if std.shape != (channels, 1):
        raise ValueError(f"Expected std shape ({channels}, 1), got {std.shape}")
    if bool((std <= 0).any()):
        raise ValueError("All standard deviations must be positive")


def _match_stats(data: ArrayLike, stat: ArrayLike) -> ArrayLike:
    """Move/cast statistics to the same backend, device and dtype as data."""
    if isinstance(data, torch.Tensor):
        return torch.as_tensor(stat, device=data.device, dtype=data.dtype)
    return np.asarray(stat, dtype=data.dtype)


def normalize_trajectory(data: ArrayLike, mean: ArrayLike, std: ArrayLike) -> ArrayLike:
    """Normalize positions in ``(..., C, H)`` layout."""
    mean = _match_stats(data, mean)
    std = _match_stats(data, std)
    _validate_shapes(data, mean, std)
    return (data - mean) / std


def denormalize_trajectory(data: ArrayLike, mean: ArrayLike, std: ArrayLike) -> ArrayLike:
    """Convert normalized positions back to physical coordinates."""
    mean = _match_stats(data, mean)
    std = _match_stats(data, std)
    _validate_shapes(data, mean, std)
    return data * std + mean


def normalize_velocity(velocity: ArrayLike, std: ArrayLike) -> ArrayLike:
    """Convert physical velocity to normalized-coordinate velocity.

    If ``x_norm = (x_phys - mean) / std``, then
    ``dx_norm/dt = (dx_phys/dt) / std``.
    """
    std = _match_stats(velocity, std)
    mean = torch.zeros_like(std) if isinstance(std, torch.Tensor) else np.zeros_like(std)
    _validate_shapes(velocity, mean, std)
    return velocity / std


def denormalize_velocity(velocity: ArrayLike, std: ArrayLike) -> ArrayLike:
    """Convert normalized-coordinate velocity to physical velocity."""
    std = _match_stats(velocity, std)
    mean = torch.zeros_like(std) if isinstance(std, torch.Tensor) else np.zeros_like(std)
    _validate_shapes(velocity, mean, std)
    return velocity * std
