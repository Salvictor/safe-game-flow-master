import numpy as np

from safe_game_flow.data.ring_trajectories import CONDITION_SCHEMA, generate_ring_dataset
from safe_game_flow.trajectories import (
    integrate_velocity_profile,
    project_velocity_boundary_constraints,
)


def test_velocity_projection_enforces_endpoints_and_displacement():
    sample = generate_ring_dataset(1, seed=91)[0]
    rng = np.random.default_rng(3)
    nominal = rng.normal(size=sample.velocity.shape)

    projected = project_velocity_boundary_constraints(
        nominal, sample.condition, CONDITION_SCHEMA
    )
    trajectory = integrate_velocity_profile(projected, sample.condition, CONDITION_SCHEMA)

    np.testing.assert_allclose(projected[0], sample.condition[3:6], atol=1e-12)
    np.testing.assert_allclose(projected[-1], sample.condition[9:12], atol=1e-12)
    np.testing.assert_allclose(trajectory[0], sample.condition[0:3], atol=1e-12)
    np.testing.assert_allclose(trajectory[-1], sample.condition[6:9], atol=1e-12)


def test_expert_velocity_integrates_close_to_expert_positions():
    sample = generate_ring_dataset(1, seed=92)[0]

    trajectory = integrate_velocity_profile(
        sample.velocity, sample.condition, CONDITION_SCHEMA
    )

    np.testing.assert_allclose(trajectory, sample.trajectory, atol=2e-3)
