import numpy as np

from safe_game_flow.data import RingDatasetConfig, generate_ring_dataset
from safe_game_flow.evaluation import evaluate_ring_trajectory, summarize_ring_metrics


def test_expert_ring_trajectory_satisfies_complete_evaluation():
    sample = generate_ring_dataset(1, seed=31, config=RingDatasetConfig())[0]

    metrics = evaluate_ring_trajectory(sample.trajectory, sample.condition)

    assert metrics.successful
    assert metrics.passed_ring
    assert metrics.collision_free
    assert metrics.dynamically_feasible
    assert metrics.endpoint_feasible
    assert metrics.minimum_barrier >= 0
    assert metrics.crossing_clearance > 0


def test_non_traversing_trajectory_fails_task_metric():
    sample = generate_ring_dataset(1, seed=32, config=RingDatasetConfig())[0]
    stationary = np.repeat(sample.trajectory[:1], len(sample.trajectory), axis=0)

    metrics = evaluate_ring_trajectory(stationary, sample.condition)

    assert not metrics.successful
    assert not metrics.passed_ring
    assert metrics.collision_free
    assert not metrics.endpoint_feasible


def test_metric_summary_reports_rates_and_extrema():
    samples = generate_ring_dataset(2, seed=33, config=RingDatasetConfig())
    good = evaluate_ring_trajectory(samples[0].trajectory, samples[0].condition)
    stationary = np.repeat(samples[1].trajectory[:1], len(samples[1].trajectory), axis=0)
    bad = evaluate_ring_trajectory(stationary, samples[1].condition)

    summary = summarize_ring_metrics([good, bad])

    assert summary["num_trajectories"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["ring_pass_rate"] == 0.5
    assert 0.0 <= summary["collision_free_rate"] <= 1.0
