import numpy as np

from safe_game_flow.game import (
    CONDITION_SCHEMA,
    condition_vector,
    nominal_crossing_times,
    resample_joint_positions,
    sample_scene,
    scheduled_references,
    select_pure_nash_schedule,
    simulate_expert,
    simulate_joint_reference,
)


def test_expert_rollout_and_condition_have_consistent_shapes():
    scene = sample_scene(np.random.default_rng(123))
    schedule = select_pure_nash_schedule(nominal_crossing_times(scene))
    rollout = simulate_expert(scene, schedule, horizon=5.5, dt=0.02)
    condition = condition_vector(scene, schedule, horizon=5.5)
    trajectory = resample_joint_positions(rollout, 32)

    assert condition.shape == (len(CONDITION_SCHEMA),)
    assert trajectory.shape == (6, 32)
    assert np.allclose(trajectory[:3, 0], scene.starts[0])
    assert np.allclose(trajectory[3:, 0], scene.starts[1])
    assert np.isfinite(rollout.positions).all()


def test_scheduled_reference_encodes_selected_yield_delay():
    scene = sample_scene(np.random.default_rng(7))
    schedule = select_pure_nash_schedule(nominal_crossing_times(scene))
    references = scheduled_references(scene, schedule)

    for index, action in enumerate(schedule.actions):
        assert np.isclose(references[index].delay, schedule.delays[index])
        assert (references[index].delay > 0.0) == (action == "yield")


def test_sampled_joint_reference_can_be_tracked_with_hocbf():
    scene = sample_scene(np.random.default_rng(99))
    schedule = select_pure_nash_schedule(nominal_crossing_times(scene))
    expert = simulate_expert(scene, schedule, horizon=5.5, dt=0.02)
    sampled = resample_joint_positions(expert, 48).reshape(2, 3, 48)
    tracked = simulate_joint_reference(
        scene.starts,
        sampled,
        scene.disturbance,
        horizon=5.5,
        dt=0.02,
        use_hocbf=True,
    )

    assert np.isfinite(tracked.positions).all()
    assert tracked.minimum_separation > 0.25
