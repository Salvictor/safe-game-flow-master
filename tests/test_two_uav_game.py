import numpy as np

from safe_game_flow.game import project_inter_uav_hocbf, select_pure_nash_schedule


def test_nash_schedule_breaks_simultaneous_ring_conflict():
    schedule = select_pure_nash_schedule((2.0, 2.0), safety_gap=0.45)
    assert schedule.actions in (("go", "yield"), ("yield", "go"))
    assert abs(schedule.expected_crossing_times[0] - schedule.expected_crossing_times[1]) >= 0.45
    assert schedule.actions == ("go", "yield")
    assert schedule.actions in schedule.pure_nash_equilibria


def test_nash_yield_delay_is_the_minimum_scene_dependent_safe_delay():
    schedule = select_pure_nash_schedule((2.0, 2.2))

    assert schedule.actions == ("go", "yield")
    assert np.isclose(schedule.delays[0], 0.0)
    assert np.isclose(schedule.delays[1], 0.3)
    assert np.isclose(
        schedule.expected_crossing_times[1] - schedule.expected_crossing_times[0],
        0.5,
    )


def test_hocbf_projection_satisfies_joint_affine_constraint():
    positions = np.array([[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0]])
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    nominal = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    result = project_inter_uav_hocbf(
        positions, velocities, nominal, minimum_distance=0.3
    )
    assert result.intervened
    assert result.nominal_constraint_value < 0.0
    assert result.projected_constraint_value >= -1e-10
    assert result.correction_norm > 0.0
