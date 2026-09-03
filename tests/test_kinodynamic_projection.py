import numpy as np

from safe_game_flow.data.ring_trajectories import CONDITION_SCHEMA, generate_ring_dataset
from safe_game_flow.safety import project_boundary_residual_to_feasibility
from safe_game_flow.trajectories import encode_boundary_residual


def test_feasible_expert_residual_is_not_modified():
    sample = generate_ring_dataset(1, seed=81)[0]
    residual = encode_boundary_residual(
        sample.trajectory, sample.condition, CONDITION_SCHEMA
    )

    result = project_boundary_residual_to_feasibility(
        residual, sample.condition, CONDITION_SCHEMA
    )

    assert result.scale == 1.0
    assert result.metrics.successful
    np.testing.assert_allclose(result.trajectory, sample.trajectory, atol=1e-10)


def test_large_rough_residual_is_scaled_to_feasibility():
    sample = generate_ring_dataset(1, seed=82)[0]
    rng = np.random.default_rng(4)
    rough_residual = 8.0 * rng.normal(size=sample.trajectory.shape)

    result = project_boundary_residual_to_feasibility(
        rough_residual,
        sample.condition,
        CONDITION_SCHEMA,
        grid_steps=24,
        refinement_steps=6,
    )

    assert 0.0 <= result.scale < 1.0
    assert result.metrics.successful
    assert result.metrics.maximum_speed <= 2.0
    assert result.metrics.maximum_acceleration <= 3.0
    assert result.strategy in {"analytic", "fallback_grid"}
