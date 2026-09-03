import numpy as np

from safe_game_flow.data.ring_trajectories import CONDITION_SCHEMA, generate_ring_dataset
from safe_game_flow.trajectories import (
    boundary_envelope,
    decode_boundary_residual,
    encode_boundary_residual,
    quintic_boundary_skeleton,
)


def test_boundary_envelope_and_skeleton_satisfy_endpoint_conditions():
    sample = generate_ring_dataset(1, seed=51)[0]
    horizon = len(sample.trajectory)
    envelope = boundary_envelope(horizon)
    skeleton = quintic_boundary_skeleton(sample.condition, horizon, CONDITION_SCHEMA)
    dt = sample.flight_time / (horizon - 1)
    velocity = np.gradient(skeleton, dt, axis=0, edge_order=2)

    assert envelope.shape == (horizon, 1)
    assert envelope[0, 0] == 0.0
    assert envelope[-1, 0] == 0.0
    np.testing.assert_allclose(skeleton[0], sample.condition[0:3], atol=1e-12)
    np.testing.assert_allclose(skeleton[-1], sample.condition[6:9], atol=1e-12)
    np.testing.assert_allclose(velocity[[0, -1]], 0.0, atol=5e-3)


def test_expert_trajectory_round_trips_through_boundary_residual():
    samples = generate_ring_dataset(3, seed=52)
    trajectories = np.stack([sample.trajectory for sample in samples])
    conditions = np.stack([sample.condition for sample in samples])

    residuals = encode_boundary_residual(trajectories, conditions, CONDITION_SCHEMA)
    decoded = decode_boundary_residual(residuals, conditions, CONDITION_SCHEMA)

    assert residuals.shape == trajectories.shape
    np.testing.assert_allclose(decoded, trajectories, atol=1e-10)


def test_arbitrary_residual_cannot_change_endpoint_positions():
    sample = generate_ring_dataset(1, seed=53)[0]
    rng = np.random.default_rng(2)
    residual = rng.normal(size=sample.trajectory.shape)

    decoded = decode_boundary_residual(residual, sample.condition, CONDITION_SCHEMA)

    np.testing.assert_allclose(decoded[0], sample.condition[0:3], atol=1e-12)
    np.testing.assert_allclose(decoded[-1], sample.condition[6:9], atol=1e-12)
