import numpy as np

from safe_game_flow.simulation import QuadrotorParameters, simulate_quadrotor_tracking


def test_default_brushless_thrust_to_weight_is_physical():
    parameters = QuadrotorParameters()
    ratio = 4.0 * parameters.maximum_motor_thrust / (parameters.mass * parameters.gravity)
    assert 3.4 < ratio < 3.6


def test_hover_reference_remains_near_hover():
    trajectory = np.repeat(np.asarray([[0.0, 0.0, 1.0]]), 21, axis=0)
    result = simulate_quadrotor_tracking(trajectory, 1.0, integration_dt=0.002, control_dt=0.01)
    assert result.tracking_rmse < 0.02
    assert result.maximum_tracking_error < 0.04
    assert result.maximum_tilt_degrees < 1.0


def test_smooth_translation_is_tracked_without_numerical_failure():
    phase = np.linspace(0.0, 1.0, 51)
    smooth = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
    trajectory = np.column_stack([smooth, np.zeros_like(smooth), np.ones_like(smooth)])
    result = simulate_quadrotor_tracking(trajectory, 2.0, seed=5)
    assert np.isfinite(result.positions).all()
    assert result.tracking_rmse < 0.20
    assert result.maximum_tracking_error < 0.35
