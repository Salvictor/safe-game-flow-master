"""Closed-form CBF projection for boundary-residual trajectory flows.

For each trajectory phase point, the ring-frame CBF defines one affine
constraint on the Flow Matching velocity. These constraints act on disjoint
three-dimensional residual coordinates, so the Euclidean QP projection has a
closed-form solution and can be evaluated in a batched manner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from safe_game_flow.trajectories import boundary_envelope, quintic_boundary_skeleton


def _indices(schema: Sequence[str]) -> dict[str, int]:
    if len(set(schema)) != len(schema):
        raise ValueError("condition schema contains duplicate names")
    required = (
        "ring_center_x", "ring_center_y", "ring_center_z",
        "ring_normal_x", "ring_normal_y", "ring_normal_z",
        "ring_major_radius", "ring_tube_radius", "uav_radius", "margin",
    )
    result = {name: index for index, name in enumerate(schema)}
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f"condition schema is missing ring fields: {missing}")
    return result


@dataclass(frozen=True)
class RingCBFDiagnostics:
    """Batched diagnostic tensors from one safety-projection call."""

    barrier: torch.Tensor
    active: torch.Tensor
    nominal_barrier_rate: torch.Tensor
    projected_barrier_rate: torch.Tensor
    correction_norm: torch.Tensor


class RingResidualCBFProjector:
    """Project normalized residual velocities onto ring-frame CBF constraints.

    The projector is bound to a batch of physical task conditions. Model state
    and velocity use layout ``(B,3,H)`` in normalized boundary-residual space.
    """

    def __init__(
        self,
        conditions: np.ndarray,
        schema: Sequence[str],
        horizon: int,
        residual_mean: np.ndarray | torch.Tensor,
        residual_std: np.ndarray | torch.Tensor,
        *,
        alpha: float = 5.0,
        epsilon: float = 1e-9,
        collocation_factor: int = 2,
        projection_iterations: int = 2,
    ) -> None:
        conditions = np.asarray(conditions, dtype=np.float32)
        if conditions.ndim != 2 or conditions.shape[1] != len(schema):
            raise ValueError(
                f"conditions must have shape (B,{len(schema)}), got {conditions.shape}"
            )
        if horizon < 3:
            raise ValueError("horizon must be at least 3")
        if alpha <= 0 or epsilon <= 0:
            raise ValueError("alpha and epsilon must be positive")
        if collocation_factor < 1 or projection_iterations < 1:
            raise ValueError("collocation_factor and projection_iterations must be positive")

        index = _indices(schema)
        center = conditions[:, [index[f"ring_center_{axis}"] for axis in "xyz"]]
        normal = conditions[:, [index[f"ring_normal_{axis}"] for axis in "xyz"]]
        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        if np.any(normal_norm < 1e-12):
            raise ValueError("ring normals must be non-zero")
        self.conditions = conditions
        self.schema = tuple(schema)
        self.horizon = horizon
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.collocation_factor = int(collocation_factor)
        self.projection_iterations = int(projection_iterations)
        self._skeleton = quintic_boundary_skeleton(
            conditions, horizon, schema
        ).astype(np.float32)
        self._envelope = boundary_envelope(horizon).astype(np.float32)
        self._center = center.astype(np.float32)
        self._normal = (normal / normal_norm).astype(np.float32)
        self._major_radius = conditions[:, index["ring_major_radius"]].astype(np.float32)
        self._effective_radius = (
            conditions[:, index["ring_tube_radius"]]
            + conditions[:, index["uav_radius"]]
            + conditions[:, index["margin"]]
        ).astype(np.float32)
        dense_phase = np.arange((horizon - 1) * collocation_factor + 1) / collocation_factor
        self._left_index = np.minimum(np.floor(dense_phase).astype(np.int64), horizon - 2)
        self._right_index = self._left_index + 1
        self._interpolation_weight = (dense_phase - self._left_index).astype(np.float32)

        mean = torch.as_tensor(residual_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(residual_std, dtype=torch.float32).reshape(-1)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError(
                f"residual statistics must contain three channels, got {mean.shape}, {std.shape}"
            )
        if torch.any(std <= 0):
            raise ValueError("residual standard deviations must be positive")
        self._residual_mean = mean.cpu()
        self._residual_std = std.cpu()
        self.last_diagnostics: RingCBFDiagnostics | None = None
        self.total_calls = 0
        self.total_constraints = 0
        self.total_active = 0
        self.total_correction_norm = 0.0
        self.maximum_correction_norm = 0.0
        self.state_projection_calls = 0
        self.state_projection_interventions = 0
        self.maximum_state_correction_norm = 0.0

    def _tensor(self, value, reference: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """Decode normalized state to physical positions with layout ``(B,H,3)``."""
        self._validate_state(state, "state")
        mean = self._tensor(self._residual_mean, state).view(1, 3, 1)
        std = self._tensor(self._residual_std, state).view(1, 3, 1)
        residual = (state * std + mean).transpose(1, 2)
        skeleton = self._tensor(self._skeleton, state)
        envelope = self._tensor(self._envelope, state).view(1, self.horizon, 1)
        return skeleton + envelope * residual

    def _validate_state(self, value: torch.Tensor, name: str) -> None:
        expected = (len(self.conditions), 3, self.horizon)
        if value.ndim != 3 or tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")

    def __call__(
        self,
        state: torch.Tensor,
        flow_time: torch.Tensor,
        nominal_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Return the minimum-norm velocity satisfying all pointwise CBF inequalities."""
        self._validate_state(state, "state")
        self._validate_state(nominal_velocity, "nominal_velocity")
        if flow_time.ndim != 1 or flow_time.shape[0] != state.shape[0]:
            raise ValueError(
                f"flow_time must have shape ({state.shape[0]},), got {tuple(flow_time.shape)}"
            )

        trajectory = self.decode(state)
        left_index = torch.as_tensor(self._left_index, device=state.device)
        right_index = torch.as_tensor(self._right_index, device=state.device)
        weight = self._tensor(self._interpolation_weight, state).view(1, -1, 1)
        dense_trajectory = (
            (1.0 - weight) * trajectory[:, left_index, :]
            + weight * trajectory[:, right_index, :]
        )
        center = self._tensor(self._center, state)[:, None, :]
        normal = self._tensor(self._normal, state)[:, None, :]
        q = dense_trajectory - center
        z = torch.sum(q * normal, dim=-1)
        q_perp = q - z[..., None] * normal
        rho = torch.sqrt(torch.sum(q_perp * q_perp, dim=-1) + self.epsilon**2)
        major_radius = self._tensor(self._major_radius, state)[:, None]
        effective_radius = self._tensor(self._effective_radius, state)[:, None]
        barrier = (rho - major_radius) ** 2 + z**2 - effective_radius**2
        gradient = (
            2.0 * (rho - major_radius)[..., None] * q_perp / rho[..., None]
            + 2.0 * z[..., None] * normal
        )

        envelope = self._tensor(self._envelope, state).view(1, self.horizon, 1)
        std = self._tensor(self._residual_std, state).view(1, 1, 3)
        state_scale = envelope * std
        left_normal = gradient * (1.0 - weight) * state_scale[:, left_index, :]
        right_normal = gradient * weight * state_scale[:, right_index, :]
        nominal_hc = nominal_velocity.transpose(1, 2)
        nominal_rate = (
            torch.sum(left_normal * nominal_hc[:, left_index, :], dim=-1)
            + torch.sum(right_normal * nominal_hc[:, right_index, :], dim=-1)
        )
        required_rate = -self.alpha * barrier
        squared_norm = (
            torch.sum(left_normal * left_normal, dim=-1)
            + torch.sum(right_normal * right_normal, dim=-1)
        )
        projected_hc = nominal_hc.clone()
        active = torch.zeros_like(barrier, dtype=torch.bool)
        initially_violated = (required_rate > nominal_rate) & (squared_norm > self.epsilon)
        # Safety-distilled/safe-prior flows are usually already feasible. Avoid
        # entering the Python active-set sweep when the nominal field satisfies
        # every affine CBF constraint; this preserves the exact projection while
        # making batch-size-one online inference substantially cheaper.
        if bool(torch.any(initially_violated).item()):
            for _ in range(self.projection_iterations):
                current_rate = (
                    torch.sum(left_normal * projected_hc[:, left_index, :], dim=-1)
                    + torch.sum(right_normal * projected_hc[:, right_index, :], dim=-1)
                )
                violated = (required_rate > current_rate) & (squared_norm > self.epsilon)
                active_indices = torch.nonzero(
                    torch.any(violated, dim=0), as_tuple=False
                ).flatten().tolist()
                if not active_indices:
                    break
                # Only visit currently violated half-spaces. The previous
                # implementation swept every dense collocation point whenever
                # one constraint was active, which dominated P95/P99 latency.
                for point_index in active_indices:
                    left = int(self._left_index[point_index])
                    right = int(self._right_index[point_index])
                    a_left = left_normal[:, point_index, :]
                    a_right = right_normal[:, point_index, :]
                    rate = (
                        torch.sum(a_left * projected_hc[:, left, :], dim=-1)
                        + torch.sum(a_right * projected_hc[:, right, :], dim=-1)
                    )
                    violation = required_rate[:, point_index] - rate
                    correctable = squared_norm[:, point_index] > self.epsilon
                    point_active = (violation > 0) & correctable
                    active[:, point_index] |= point_active
                    multiplier = torch.where(
                        point_active,
                        violation / squared_norm[:, point_index].clamp_min(self.epsilon),
                        torch.zeros_like(violation),
                    )
                    projected_hc[:, left, :] += multiplier[:, None] * a_left
                    projected_hc[:, right, :] += multiplier[:, None] * a_right

        projected_rate = (
            torch.sum(left_normal * projected_hc[:, left_index, :], dim=-1)
            + torch.sum(right_normal * projected_hc[:, right_index, :], dim=-1)
        )
        correction_hc = projected_hc - nominal_hc

        self.last_diagnostics = RingCBFDiagnostics(
            barrier=barrier.detach(),
            active=active.detach(),
            nominal_barrier_rate=nominal_rate.detach(),
            projected_barrier_rate=projected_rate.detach(),
            correction_norm=torch.linalg.vector_norm(correction_hc, dim=-1).detach(),
        )
        correction_norm = self.last_diagnostics.correction_norm
        self.total_calls += 1
        self.total_constraints += active.numel()
        self.total_active += int(active.sum().item())
        self.total_correction_norm += float(correction_norm.sum().item())
        self.maximum_correction_norm = max(
            self.maximum_correction_norm, float(correction_norm.max().item())
        )
        return projected_hc.transpose(1, 2)

    def project_state(
        self,
        state: torch.Tensor,
        flow_time: torch.Tensor | None = None,
        *,
        safety_margin: float = 1e-6,
        maximum_iterations: int = 12,
    ) -> torch.Tensor:
        """Correct a completed ODE step back into the sampled safe set.

        Continuous CBF inequalities constrain the vector field, but a finite
        Heun step can still cross the nonlinear torus boundary. This batched
        Jacobi projection linearizes each violated dense-point barrier and
        scatters minimum-norm corrections into the two adjacent residual
        coordinates. Endpoint conditions remain structural because the
        residual envelope is zero at both ends.
        """
        self._validate_state(state, "state")
        if safety_margin < 0 or maximum_iterations < 1:
            raise ValueError("invalid state-projection configuration")
        corrected = state.clone()
        left_index = torch.as_tensor(self._left_index, device=state.device)
        right_index = torch.as_tensor(self._right_index, device=state.device)
        weight = self._tensor(self._interpolation_weight, state).view(1, -1, 1)
        center = self._tensor(self._center, state)[:, None, :]
        normal = self._tensor(self._normal, state)[:, None, :]
        major_radius = self._tensor(self._major_radius, state)[:, None]
        effective_radius = self._tensor(self._effective_radius, state)[:, None]
        envelope = self._tensor(self._envelope, state).view(1, self.horizon, 1)
        std = self._tensor(self._residual_std, state).view(1, 1, 3)
        state_scale = envelope * std
        total_correction = torch.zeros_like(state).transpose(1, 2)
        intervened = False

        for _ in range(maximum_iterations):
            trajectory = self.decode(corrected)
            dense = (
                (1.0 - weight) * trajectory[:, left_index, :]
                + weight * trajectory[:, right_index, :]
            )
            q = dense - center
            z = torch.sum(q * normal, dim=-1)
            q_perp = q - z[..., None] * normal
            rho = torch.sqrt(torch.sum(q_perp * q_perp, dim=-1) + self.epsilon**2)
            barrier = (rho - major_radius) ** 2 + z**2 - effective_radius**2
            violation = torch.relu(safety_margin - barrier)
            if not bool(torch.any(violation > 0).item()):
                break
            intervened = True
            gradient = (
                2.0 * (rho - major_radius)[..., None] * q_perp / rho[..., None]
                + 2.0 * z[..., None] * normal
            )
            left_normal = gradient * (1.0 - weight) * state_scale[:, left_index, :]
            right_normal = gradient * weight * state_scale[:, right_index, :]
            squared_norm = (
                torch.sum(left_normal * left_normal, dim=-1)
                + torch.sum(right_normal * right_normal, dim=-1)
            )
            multiplier = torch.where(
                squared_norm > self.epsilon,
                violation / squared_norm.clamp_min(self.epsilon),
                torch.zeros_like(violation),
            )
            correction_hc = torch.zeros_like(corrected).transpose(1, 2)
            correction_hc.index_add_(
                1, left_index, multiplier[..., None] * left_normal
            )
            correction_hc.index_add_(
                1, right_index, multiplier[..., None] * right_normal
            )
            corrected = corrected + correction_hc.transpose(1, 2)
            total_correction = total_correction + correction_hc

        self.state_projection_calls += 1
        if intervened:
            self.state_projection_interventions += 1
        correction_norm = float(torch.linalg.vector_norm(total_correction).item())
        self.maximum_state_correction_norm = max(
            self.maximum_state_correction_norm, correction_norm
        )
        return corrected

    def summary(self) -> dict[str, int | float]:
        """Return cumulative intervention statistics since construction."""
        return {
            "projection_calls": self.total_calls,
            "constraints_evaluated": self.total_constraints,
            "active_constraints": self.total_active,
            "intervention_rate": (
                self.total_active / self.total_constraints if self.total_constraints else 0.0
            ),
            "mean_correction_norm": (
                self.total_correction_norm / self.total_constraints
                if self.total_constraints
                else 0.0
            ),
            "maximum_correction_norm": self.maximum_correction_norm,
            "state_projection_calls": self.state_projection_calls,
            "state_projection_interventions": self.state_projection_interventions,
            "state_projection_intervention_rate": (
                self.state_projection_interventions / self.state_projection_calls
                if self.state_projection_calls
                else 0.0
            ),
            "maximum_state_correction_norm": self.maximum_state_correction_norm,
        }
