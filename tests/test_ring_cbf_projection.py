import numpy as np
import torch

from safe_game_flow.data.ring_trajectories import CONDITION_SCHEMA, generate_ring_dataset
from safe_game_flow.evaluation import ring_from_condition
from safe_game_flow.safety import RingResidualCBFProjector
from safe_game_flow.trajectories import boundary_envelope


def test_ring_cbf_projection_satisfies_affine_barrier_constraints():
    sample = generate_ring_dataset(1, seed=71)[0]
    condition = sample.condition[None, :]
    horizon = len(sample.trajectory)
    projector = RingResidualCBFProjector(
        condition,
        CONDITION_SCHEMA,
        horizon,
        residual_mean=np.zeros((3, 1), dtype=np.float32),
        residual_std=np.ones((3, 1), dtype=np.float32),
        alpha=4.0,
    )
    state = torch.zeros(1, 3, horizon)
    physical = projector.decode(state)[0].numpy()
    gradient = ring_from_condition(sample.condition).barrier_gradient(physical)
    envelope = boundary_envelope(horizon)[:, 0]
    nominal_hc = np.zeros_like(gradient)
    mask = envelope > 1e-5
    nominal_hc[mask] = -gradient[mask] / envelope[mask, None]
    nominal = torch.from_numpy(nominal_hc.T).unsqueeze(0).float()

    projected = projector(state, torch.tensor([0.5]), nominal)
    diagnostics = projector.last_diagnostics

    assert projected.shape == nominal.shape
    assert diagnostics is not None
    cbf_value = diagnostics.projected_barrier_rate + 4.0 * diagnostics.barrier
    assert torch.min(cbf_value) >= -2e-5
    assert torch.any(diagnostics.active)
    assert torch.linalg.vector_norm(projected - nominal) > 0


def test_ring_cbf_projection_leaves_feasible_zero_velocity_unchanged():
    sample = generate_ring_dataset(1, seed=72)[0]
    horizon = len(sample.trajectory)
    projector = RingResidualCBFProjector(
        sample.condition[None, :],
        CONDITION_SCHEMA,
        horizon,
        residual_mean=np.zeros((3, 1), dtype=np.float32),
        residual_std=np.ones((3, 1), dtype=np.float32),
    )
    state = torch.zeros(1, 3, horizon)
    nominal = torch.zeros_like(state)

    projected = projector(state, torch.tensor([0.25]), nominal)

    assert torch.equal(projected, nominal)
    assert not torch.any(projector.last_diagnostics.active)


def test_post_step_state_projection_repairs_finite_step_collision():
    sample = generate_ring_dataset(1, seed=73)[0]
    horizon = len(sample.trajectory)
    projector = RingResidualCBFProjector(
        sample.condition[None, :],
        CONDITION_SCHEMA,
        horizon,
        residual_mean=np.zeros((3, 1), dtype=np.float32),
        residual_std=np.ones((3, 1), dtype=np.float32),
        collocation_factor=2,
    )
    ring = ring_from_condition(sample.condition)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(ring.normal @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    radial_axis = np.cross(ring.normal, reference)
    radial_axis /= np.linalg.norm(radial_axis)
    target = ring.center + (
        ring.major_radius + 0.5 * ring.effective_tube_radius
    ) * radial_axis
    state = torch.zeros(1, 3, horizon)
    midpoint = horizon // 2
    base = projector.decode(state)[0, midpoint].numpy()
    state[0, :, midpoint] = torch.from_numpy(
        (target - base) / boundary_envelope(horizon)[midpoint, 0]
    ).float()
    assert np.min(ring.barrier_value(projector.decode(state)[0].numpy())) < 0.0

    corrected = projector.project_state(
        state, torch.tensor([0.5]), maximum_iterations=30
    )
    trajectory = projector.decode(corrected)[0].numpy()
    source = np.arange(horizon, dtype=float)
    dense_phase = np.linspace(0.0, horizon - 1, 2 * (horizon - 1) + 1)
    dense = np.stack(
        [np.interp(dense_phase, source, trajectory[:, axis]) for axis in range(3)],
        axis=1,
    )

    assert np.min(ring.barrier_value(dense)) >= 0.0
    assert projector.summary()["state_projection_interventions"] == 1
