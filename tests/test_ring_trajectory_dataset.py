import numpy as np

from safe_game_flow.data import RingDatasetConfig, generate_ring_dataset
from safe_game_flow.trajectories import fit_quintic_segment


def test_quintic_segment_satisfies_boundary_conditions():
    p0, p1 = np.array([0.0, 1.0, 2.0]), np.array([2.0, -1.0, 3.0])
    v0, v1 = np.array([0.1, 0.0, 0.0]), np.array([0.4, 0.0, -0.1])
    a0, a1 = np.zeros(3), np.array([0.0, 0.2, 0.0])
    segment = fit_quintic_segment(p0, v0, a0, p1, v1, a1, duration=2.5)

    np.testing.assert_allclose(segment.evaluate(0.0), p0, atol=1e-10)
    np.testing.assert_allclose(segment.evaluate(0.0, 1), v0, atol=1e-10)
    np.testing.assert_allclose(segment.evaluate(0.0, 2), a0, atol=1e-10)
    np.testing.assert_allclose(segment.evaluate(2.5), p1, atol=1e-9)
    np.testing.assert_allclose(segment.evaluate(2.5, 1), v1, atol=1e-9)
    np.testing.assert_allclose(segment.evaluate(2.5, 2), a1, atol=1e-9)


def test_generated_ring_dataset_is_reproducible_and_feasible():
    config = RingDatasetConfig(horizon=31, dense_check_points=101)
    first = generate_ring_dataset(4, seed=11, config=config)
    second = generate_ring_dataset(4, seed=11, config=config)

    for sample_a, sample_b in zip(first, second, strict=True):
        np.testing.assert_allclose(sample_a.trajectory, sample_b.trajectory)
        assert sample_a.trajectory.shape == (31, 3)
        assert sample_a.condition.shape == (23,)
        assert sample_a.ring.detect_ring_crossing(sample_a.trajectory).passed
        assert sample_a.min_barrier >= 0
        assert sample_a.max_speed <= config.maximum_speed
        assert sample_a.max_acceleration <= config.maximum_acceleration
