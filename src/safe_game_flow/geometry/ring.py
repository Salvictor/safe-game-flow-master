"""Three-dimensional ring geometry and robust traversal detection.

Coordinates are expressed in metres in a common world frame. A ring is
represented by its centre line radius and tube radius. The safety envelope
also includes the UAV radius and an additional user-selected margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RingCrossingResult:
    """Result returned by :meth:`RingGeometry.detect_ring_crossing`."""

    passed: bool
    reason: str
    direction: int
    segment_index: int | None = None
    crossing_point: np.ndarray | None = None
    radial_distance: float | None = None
    aperture_clearance: float | None = None

    def __bool__(self) -> bool:
        return self.passed


@dataclass(frozen=True)
class RingGeometry:
    """Smooth torus-style safety geometry for a circular flight ring.

    Args:
        center: Ring centre in world coordinates, shape ``(3,)``.
        normal: Ring-plane unit normal, shape ``(3,)``. It is normalized on
            construction and defines the positive traversal direction.
        major_radius: Radius from the ring centre to the frame centre line.
        tube_radius: Physical radius of the ring frame tube.
        uav_radius: Spherical UAV safety-envelope radius.
        margin: Additional clearance for tracking/localization uncertainty.
        epsilon: Numerical smoothing used for the radial norm near the axis.
    """

    center: np.ndarray
    normal: np.ndarray
    major_radius: float
    tube_radius: float
    uav_radius: float = 0.0
    margin: float = 0.0
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        normal = np.asarray(self.normal, dtype=float)
        if center.shape != (3,):
            raise ValueError(f"center must have shape (3,), got {center.shape}")
        if normal.shape != (3,):
            raise ValueError(f"normal must have shape (3,), got {normal.shape}")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-12:
            raise ValueError("normal must be non-zero")
        if self.major_radius <= 0:
            raise ValueError("major_radius must be positive")
        if self.tube_radius < 0 or self.uav_radius < 0 or self.margin < 0:
            raise ValueError("tube_radius, uav_radius and margin must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.clear_aperture_radius <= 0:
            raise ValueError(
                "Safety envelope closes the aperture: major_radius must exceed "
                "tube_radius + uav_radius + margin"
            )
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "normal", normal / normal_norm)

    @property
    def effective_tube_radius(self) -> float:
        """Frame radius after UAV size and safety margin are included."""
        return self.tube_radius + self.uav_radius + self.margin

    @property
    def clear_aperture_radius(self) -> float:
        """Maximum radial offset that remains inside the clear opening."""
        return self.major_radius - self.effective_tube_radius

    @staticmethod
    def _validate_points(points: np.ndarray | torch.Tensor) -> None:
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError(f"Expected points with shape (...,3), got {points.shape}")
        if isinstance(points, torch.Tensor) and not points.is_floating_point():
            raise TypeError("Torch geometry inputs must use a floating-point dtype")

    def _parameters_like(
        self, points: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if isinstance(points, torch.Tensor):
            center = torch.as_tensor(self.center, device=points.device, dtype=points.dtype)
            normal = torch.as_tensor(self.normal, device=points.device, dtype=points.dtype)
        else:
            dtype = points.dtype if np.issubdtype(points.dtype, np.floating) else np.float64
            center = np.asarray(self.center, dtype=dtype)
            normal = np.asarray(self.normal, dtype=dtype)
        return center, normal

    def decompose(
        self, points: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Return signed plane distance and in-plane radial distance."""
        self._validate_points(points)
        center, normal = self._parameters_like(points)
        q = points - center
        z = (q * normal).sum(axis=-1)
        q_perp = q - z[..., None] * normal
        radial_sq = (q_perp * q_perp).sum(axis=-1)
        if isinstance(points, torch.Tensor):
            rho = torch.sqrt(radial_sq + self.epsilon**2)
        else:
            rho = np.sqrt(radial_sq + self.epsilon**2)
        return z, rho

    def signed_plane_distance(self, points: np.ndarray | torch.Tensor):
        """Signed distance to the ring plane; positive follows ``normal``."""
        return self.decompose(points)[0]

    def aperture_clearance(self, points: np.ndarray | torch.Tensor):
        """Positive radial clearance inside the ring opening."""
        _, rho = self.decompose(points)
        return self.clear_aperture_radius - rho

    def barrier_value(self, points: np.ndarray | torch.Tensor):
        """Evaluate the frame safety function.

        ``h >= 0`` is outside the expanded torus frame, while ``h < 0``
        intersects the frame safety envelope.
        """
        z, rho = self.decompose(points)
        return (
            (rho - self.major_radius) ** 2
            + z**2
            - self.effective_tube_radius**2
        )

    def barrier_gradient(self, points: np.ndarray | torch.Tensor):
        """Analytic gradient of :meth:`barrier_value` with respect to points."""
        self._validate_points(points)
        center, normal = self._parameters_like(points)
        q = points - center
        z = (q * normal).sum(axis=-1)
        q_perp = q - z[..., None] * normal
        radial_sq = (q_perp * q_perp).sum(axis=-1)
        if isinstance(points, torch.Tensor):
            rho = torch.sqrt(radial_sq + self.epsilon**2)
        else:
            rho = np.sqrt(radial_sq + self.epsilon**2)
        radial_term = 2.0 * (rho - self.major_radius)[..., None] * q_perp / rho[..., None]
        plane_term = 2.0 * z[..., None] * normal
        return radial_term + plane_term

    def detect_ring_crossing(
        self,
        trajectory: np.ndarray,
        direction: int = 1,
        plane_tolerance: float = 1e-7,
        collision_tolerance: float = 1e-9,
    ) -> RingCrossingResult:
        """Detect a true plane transition through the clear aperture.

        Merely touching or stopping on the ring plane is not a traversal. The
        sampled trajectory must also remain outside the frame safety envelope.

        Args:
            trajectory: World-frame positions with shape ``(H,3)``.
            direction: ``1`` for negative-to-positive normal direction,
                ``-1`` for the reverse direction, or ``0`` to accept either.
            plane_tolerance: Values within this distance are treated as lying
                on the plane when identifying zero plateaus.
            collision_tolerance: Numerical tolerance for sampled barrier values.
        """
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) < 2:
            raise ValueError(
                f"trajectory must have shape (H,3) with H>=2, got {trajectory.shape}"
            )
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 or 1")
        if plane_tolerance < 0 or collision_tolerance < 0:
            raise ValueError("tolerances must be non-negative")

        min_h = float(np.min(self.barrier_value(trajectory)))
        if min_h < -collision_tolerance:
            return RingCrossingResult(
                passed=False,
                reason=f"trajectory intersects frame safety envelope (min_h={min_h:.6g})",
                direction=direction,
            )

        signed = np.asarray(self.signed_plane_distance(trajectory))
        signs = np.zeros(len(signed), dtype=int)
        signs[signed > plane_tolerance] = 1
        signs[signed < -plane_tolerance] = -1
        nonzero_indices = np.flatnonzero(signs)
        if len(nonzero_indices) < 2:
            return RingCrossingResult(False, "no two-sided plane transition", direction)

        saw_wrong_aperture = False
        for left, right in zip(nonzero_indices[:-1], nonzero_indices[1:], strict=False):
            sign_left, sign_right = signs[left], signs[right]
            if sign_left == sign_right:
                continue
            crossing_direction = 1 if sign_left < sign_right else -1
            if direction != 0 and crossing_direction != direction:
                continue

            zero_indices = np.arange(left + 1, right)[signs[left + 1 : right] == 0]
            if len(zero_indices):
                radial_values = np.asarray(self.decompose(trajectory[zero_indices])[1])
                best = int(zero_indices[int(np.argmin(radial_values))])
                crossing_point = trajectory[best].copy()
                segment_index = int(max(left, best - 1))
            else:
                denom = signed[right] - signed[left]
                if abs(denom) <= plane_tolerance:
                    continue
                weight = -signed[left] / denom
                crossing_point = (
                    (1.0 - weight) * trajectory[left] + weight * trajectory[right]
                )
                segment_index = int(left)

            radial = float(self.decompose(crossing_point)[1])
            clearance = self.clear_aperture_radius - radial
            if clearance < -plane_tolerance:
                saw_wrong_aperture = True
                continue
            return RingCrossingResult(
                passed=True,
                reason="valid ring-plane transition through clear aperture",
                direction=crossing_direction,
                segment_index=segment_index,
                crossing_point=crossing_point,
                radial_distance=radial,
                aperture_clearance=clearance,
            )

        reason = (
            "plane transition occurs outside clear aperture"
            if saw_wrong_aperture
            else "no plane transition in requested direction"
        )
        return RingCrossingResult(False, reason, direction)
