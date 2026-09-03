"""Quintic polynomial segments with position/velocity/acceleration boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuinticSegment:
    """A vector-valued quintic polynomial over ``[0, duration]``.

    Coefficients have shape ``(D, 6)`` in ascending power order.
    """

    coefficients: np.ndarray
    duration: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=float)
        if coefficients.ndim != 2 or coefficients.shape[1] != 6:
            raise ValueError(
                f"coefficients must have shape (D,6), got {coefficients.shape}"
            )
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        object.__setattr__(self, "coefficients", coefficients.copy())

    @property
    def dimension(self) -> int:
        return self.coefficients.shape[0]

    def evaluate(self, time: np.ndarray | float, derivative: int = 0) -> np.ndarray:
        """Evaluate position or a derivative at one or more times.

        Returns ``(..., D)`` for an input with shape ``(...)``.
        """
        if derivative < 0 or derivative > 5:
            raise ValueError("derivative must be between 0 and 5")
        time_array = np.asarray(time, dtype=float)
        if np.any(time_array < -1e-12) or np.any(time_array > self.duration + 1e-12):
            raise ValueError(f"time must lie in [0, {self.duration}]")
        time_array = np.clip(time_array, 0.0, self.duration)

        powers = np.arange(6)
        factors = np.ones(6)
        for _ in range(derivative):
            factors *= np.maximum(powers, 0)
            powers = np.maximum(powers - 1, 0)
        factors[:derivative] = 0.0
        basis = factors * time_array[..., None] ** powers
        return basis @ self.coefficients.T


def fit_quintic_segment(
    start_position: np.ndarray,
    start_velocity: np.ndarray,
    start_acceleration: np.ndarray,
    end_position: np.ndarray,
    end_velocity: np.ndarray,
    end_acceleration: np.ndarray,
    duration: float,
) -> QuinticSegment:
    """Fit a quintic segment satisfying endpoint position, velocity and acceleration."""
    vectors = [
        np.asarray(value, dtype=float)
        for value in (
            start_position,
            start_velocity,
            start_acceleration,
            end_position,
            end_velocity,
            end_acceleration,
        )
    ]
    if any(value.ndim != 1 for value in vectors):
        raise ValueError("All boundary values must be one-dimensional vectors")
    dimension = vectors[0].shape[0]
    if dimension < 1 or any(value.shape != (dimension,) for value in vectors):
        raise ValueError("All boundary vectors must have the same non-zero shape")
    if duration <= 0:
        raise ValueError("duration must be positive")

    t = float(duration)
    matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, t, t**2, t**3, t**4, t**5],
            [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4],
            [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3],
        ],
        dtype=float,
    )
    boundary = np.stack(vectors, axis=0)
    coefficients = np.linalg.solve(matrix, boundary).T
    return QuinticSegment(coefficients=coefficients, duration=t)
