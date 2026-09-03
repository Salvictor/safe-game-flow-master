import numpy as np
import torch

from safe_game_flow.geometry import RingGeometry


def make_ring() -> RingGeometry:
    return RingGeometry(
        center=np.array([0.0, 0.0, 0.0]),
        normal=np.array([2.0, 0.0, 0.0]),
        major_radius=1.0,
        tube_radius=0.1,
        uav_radius=0.1,
    )


def test_normal_is_normalized_and_aperture_accounts_for_uav_size():
    ring = make_ring()

    np.testing.assert_allclose(ring.normal, [1.0, 0.0, 0.0])
    assert ring.effective_tube_radius == 0.2
    assert ring.clear_aperture_radius == 0.8


def test_barrier_gradient_matches_finite_difference():
    ring = make_ring()
    point = np.array([0.3, 0.45, -0.2])
    analytic = ring.barrier_gradient(point)
    numerical = np.zeros(3)
    eps = 1e-6
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = eps
        numerical[axis] = (
            ring.barrier_value(point + step) - ring.barrier_value(point - step)
        ) / (2 * eps)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-6)


def test_torch_barrier_supports_autograd():
    ring = make_ring()
    point = torch.tensor([[0.3, 0.45, -0.2]], dtype=torch.float64, requires_grad=True)
    value = ring.barrier_value(point).sum()
    value.backward()

    expected = ring.barrier_gradient(point.detach()).numpy()
    np.testing.assert_allclose(point.grad.numpy(), expected, rtol=1e-6, atol=1e-7)


def test_valid_forward_and_reverse_crossings():
    ring = make_ring()
    forward = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    reverse = forward[::-1].copy()

    forward_result = ring.detect_ring_crossing(forward, direction=1)
    reverse_result = ring.detect_ring_crossing(reverse, direction=-1)

    assert forward_result.passed and forward_result.direction == 1
    assert reverse_result.passed and reverse_result.direction == -1
    assert forward_result.aperture_clearance > 0


def test_touching_plane_without_side_change_is_not_a_crossing():
    ring = make_ring()
    trajectory = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    result = ring.detect_ring_crossing(trajectory)

    assert not result.passed


def test_frame_collision_is_rejected():
    ring = make_ring()
    trajectory = np.array([[-1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    result = ring.detect_ring_crossing(trajectory)

    assert not result.passed
    assert "intersects frame" in result.reason


def test_wrong_direction_is_rejected():
    ring = make_ring()
    reverse = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    result = ring.detect_ring_crossing(reverse, direction=1)

    assert not result.passed
    assert "requested direction" in result.reason
