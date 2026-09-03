"""Six-degree-of-freedom quadrotor tracking simulation.

The model contains translational and rotational rigid-body dynamics, four
independent first-order motor/thrust states, aerodynamic translation drag,
actuator saturation, sampled feedback, measurement noise, and command delay.
It is intentionally independent of a specific planner so all baselines are
evaluated by exactly the same closed-loop plant and controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuadrotorParameters:
    """Physical plant parameters in SI units.

    The default mass, arm radius, and maximum motor thrust match public
    Crazyflie 2.1 Brushless specifications. Inertia, motor lag, drag, and yaw
    moment ratio are simulation assumptions that must be identified before a
    hardware-claiming comparison.
    """

    mass: float = 0.034
    inertia_xx: float = 1.6e-5
    inertia_yy: float = 1.6e-5
    inertia_zz: float = 2.9e-5
    arm_radius: float = 0.05
    maximum_motor_thrust: float = 0.030 * 9.80665
    motor_time_constant: float = 0.025
    yaw_moment_ratio: float = 0.006
    linear_drag: float = 0.015
    gravity: float = 9.80665

    @property
    def inertia(self) -> np.ndarray:
        return np.diag([self.inertia_xx, self.inertia_yy, self.inertia_zz])

    @property
    def allocation_matrix(self) -> np.ndarray:
        radius = self.arm_radius
        xy = radius / np.sqrt(2.0)
        positions = np.asarray(
            [[xy, xy], [-xy, xy], [-xy, -xy], [xy, -xy]], dtype=float
        )
        spin = np.asarray([1.0, -1.0, 1.0, -1.0])
        return np.vstack(
            [
                np.ones(4),
                positions[:, 1],
                -positions[:, 0],
                self.yaw_moment_ratio * spin,
            ]
        )


@dataclass(frozen=True)
class ControllerGains:
    position: tuple[float, float, float] = (7.0, 7.0, 10.0)
    velocity: tuple[float, float, float] = (4.2, 4.2, 5.0)
    attitude: tuple[float, float, float] = (0.0035, 0.0035, 0.0018)
    angular_rate: tuple[float, float, float] = (0.00018, 0.00018, 0.00012)
    maximum_tilt_degrees: float = 40.0


@dataclass(frozen=True)
class QuadrotorSimulationResult:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    quaternions: np.ndarray
    angular_rates: np.ndarray
    motor_thrusts: np.ndarray
    commanded_motor_thrusts: np.ndarray
    reference_positions: np.ndarray
    reference_velocities: np.ndarray
    reference_accelerations: np.ndarray
    tracking_rmse: float
    maximum_tracking_error: float
    motor_saturation_fraction: float
    maximum_tilt_degrees: float


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _vee(matrix: np.ndarray) -> np.ndarray:
    return np.asarray([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def _reference_derivatives(trajectory: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    dt = duration / (len(trajectory) - 1)
    velocity = np.gradient(trajectory, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    return velocity, acceleration


def _interpolate(values: np.ndarray, phase: float) -> np.ndarray:
    coordinate = np.clip(phase, 0.0, 1.0) * (len(values) - 1)
    lower = min(int(np.floor(coordinate)), len(values) - 2)
    weight = coordinate - lower
    return (1.0 - weight) * values[lower] + weight * values[lower + 1]


def _desired_rotation(force: np.ndarray, yaw: float) -> np.ndarray:
    norm = np.linalg.norm(force)
    if norm < 1e-9:
        return np.eye(3)
    body_z = force / norm
    heading = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
    body_y = np.cross(body_z, heading)
    if np.linalg.norm(body_y) < 1e-8:
        heading = np.asarray([0.0, 1.0, 0.0])
        body_y = np.cross(body_z, heading)
    body_y /= np.linalg.norm(body_y)
    body_x = np.cross(body_y, body_z)
    return np.column_stack([body_x, body_y, body_z])


def _limit_force_tilt(force: np.ndarray, maximum_tilt_degrees: float) -> np.ndarray:
    vertical = max(float(force[2]), 1e-6)
    horizontal = force[:2]
    limit = vertical * np.tan(np.deg2rad(maximum_tilt_degrees))
    norm = np.linalg.norm(horizontal)
    if norm <= limit or norm < 1e-12:
        return force
    result = force.copy()
    result[:2] *= limit / norm
    return result


def _controller(
    state: np.ndarray,
    reference_position: np.ndarray,
    reference_velocity: np.ndarray,
    reference_acceleration: np.ndarray,
    parameters: QuadrotorParameters,
    gains: ControllerGains,
    position_noise: np.ndarray,
    velocity_noise: np.ndarray,
) -> np.ndarray:
    position = state[0:3] + position_noise
    velocity = state[3:6] + velocity_noise
    quaternion = state[6:10]
    angular_rate = state[10:13]
    rotation = _rotation_from_quaternion(quaternion)
    kp = np.asarray(gains.position)
    kv = np.asarray(gains.velocity)
    commanded_acceleration = (
        reference_acceleration
        + kp * (reference_position - position)
        + kv * (reference_velocity - velocity)
    )
    desired_force = parameters.mass * (
        commanded_acceleration + np.asarray([0.0, 0.0, parameters.gravity])
    )
    desired_force = _limit_force_tilt(desired_force, gains.maximum_tilt_degrees)
    desired_rotation = _desired_rotation(desired_force, yaw=0.0)
    attitude_error = 0.5 * _vee(
        desired_rotation.T @ rotation - rotation.T @ desired_rotation
    )
    moment = (
        -np.asarray(gains.attitude) * attitude_error
        -np.asarray(gains.angular_rate) * angular_rate
        + np.cross(angular_rate, parameters.inertia @ angular_rate)
    )
    total_thrust = max(0.0, float(desired_force @ rotation[:, 2]))
    wrench = np.concatenate([[total_thrust], moment])
    motor_command = np.linalg.solve(parameters.allocation_matrix, wrench)
    return np.clip(motor_command, 0.0, parameters.maximum_motor_thrust)


def _derivative(
    state: np.ndarray,
    motor_command: np.ndarray,
    parameters: QuadrotorParameters,
    disturbance_acceleration: np.ndarray,
) -> np.ndarray:
    velocity = state[3:6]
    quaternion = state[6:10] / np.linalg.norm(state[6:10])
    angular_rate = state[10:13]
    motor_thrust = state[13:17]
    rotation = _rotation_from_quaternion(quaternion)
    acceleration = (
        np.asarray([0.0, 0.0, -parameters.gravity])
        + rotation[:, 2] * np.sum(motor_thrust) / parameters.mass
        - parameters.linear_drag * velocity / parameters.mass
        + disturbance_acceleration
    )
    wrench = parameters.allocation_matrix @ motor_thrust
    moment = wrench[1:4]
    angular_acceleration = np.linalg.solve(
        parameters.inertia,
        moment - np.cross(angular_rate, parameters.inertia @ angular_rate),
    )
    quaternion_derivative = 0.5 * _quaternion_multiply(
        quaternion, np.concatenate([[0.0], angular_rate])
    )
    motor_derivative = (motor_command - motor_thrust) / parameters.motor_time_constant
    return np.concatenate(
        [velocity, acceleration, quaternion_derivative, angular_acceleration, motor_derivative]
    )


def _rk4_step(
    state: np.ndarray,
    motor_command: np.ndarray,
    parameters: QuadrotorParameters,
    disturbance_acceleration: np.ndarray,
    dt: float,
) -> np.ndarray:
    function = lambda value: _derivative(  # noqa: E731
        value, motor_command, parameters, disturbance_acceleration
    )
    k1 = function(state)
    k2 = function(state + 0.5 * dt * k1)
    k3 = function(state + 0.5 * dt * k2)
    k4 = function(state + dt * k3)
    result = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    result[6:10] /= np.linalg.norm(result[6:10])
    result[13:17] = np.clip(result[13:17], 0.0, parameters.maximum_motor_thrust)
    return result


def simulate_quadrotor_tracking(
    trajectory: np.ndarray,
    duration: float,
    *,
    parameters: QuadrotorParameters | None = None,
    gains: ControllerGains | None = None,
    integration_dt: float = 0.002,
    control_dt: float = 0.002,
    command_delay: float = 0.0,
    disturbance_acceleration: np.ndarray | None = None,
    position_noise_std: float = 0.0,
    velocity_noise_std: float = 0.0,
    mass_scale: float = 1.0,
    motor_time_constant_scale: float = 1.0,
    seed: int = 0,
) -> QuadrotorSimulationResult:
    """Track a uniformly timed 3D trajectory using a geometric controller."""
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) < 3:
        raise ValueError("trajectory must have shape (H,3), H >= 3")
    if duration <= 0 or integration_dt <= 0 or control_dt < integration_dt:
        raise ValueError("duration/dt values are inconsistent")
    if command_delay < 0:
        raise ValueError("command_delay must be non-negative")
    nominal_parameters = parameters or QuadrotorParameters()
    parameters = QuadrotorParameters(
        **{
            **nominal_parameters.__dict__,
            "mass": nominal_parameters.mass * mass_scale,
            "motor_time_constant": nominal_parameters.motor_time_constant
            * motor_time_constant_scale,
        }
    )
    gains = gains or ControllerGains()
    disturbance = (
        np.zeros(3)
        if disturbance_acceleration is None
        else np.asarray(disturbance_acceleration, dtype=float)
    )
    if disturbance.shape != (3,):
        raise ValueError("disturbance_acceleration must have shape (3,)")
    reference_velocity, reference_acceleration = _reference_derivatives(trajectory, duration)
    integration_steps = int(np.ceil(duration / integration_dt))
    times = np.linspace(0.0, duration, integration_steps + 1)
    actual_dt = duration / integration_steps
    control_stride = max(1, int(round(control_dt / actual_dt)))
    rng = np.random.default_rng(seed)

    state = np.zeros(17)
    state[0:3] = trajectory[0]
    state[3:6] = reference_velocity[0]
    state[6] = 1.0
    hover_per_motor = parameters.mass * parameters.gravity / 4.0
    state[13:17] = hover_per_motor
    motor_command = np.full(4, hover_per_motor)

    states = np.empty((len(times), 17))
    commands = np.empty((len(times), 4))
    reference_positions = np.empty((len(times), 3))
    reference_velocities = np.empty((len(times), 3))
    reference_accelerations = np.empty((len(times), 3))
    states[0] = state
    commands[0] = motor_command
    for index, time_value in enumerate(times):
        # Wireless/offboard delay affects the received reference timestamp. The
        # onboard attitude/motor loop itself continues to run at ``control_dt``.
        reference_time = max(0.0, time_value - command_delay)
        phase = reference_time / duration
        ref_p = _interpolate(trajectory, phase)
        ref_v = _interpolate(reference_velocity, phase)
        ref_a = _interpolate(reference_acceleration, phase)
        reference_positions[index] = ref_p
        reference_velocities[index] = ref_v
        reference_accelerations[index] = ref_a
        if index == len(times) - 1:
            break
        if index % control_stride == 0:
            requested_command = _controller(
                state,
                ref_p,
                ref_v,
                ref_a,
                nominal_parameters,
                gains,
                rng.normal(0.0, position_noise_std, 3),
                rng.normal(0.0, velocity_noise_std, 3),
            )
            motor_command = requested_command
        state = _rk4_step(state, motor_command, parameters, disturbance, actual_dt)
        states[index + 1] = state
        commands[index + 1] = motor_command

    errors = np.linalg.norm(states[:, 0:3] - reference_positions, axis=1)
    rotation_z = np.stack(
        [_rotation_from_quaternion(item)[2, 2] for item in states[:, 6:10]]
    )
    tilt = np.rad2deg(np.arccos(np.clip(rotation_z, -1.0, 1.0)))
    tolerance = 1e-8
    saturation = (commands <= tolerance) | (
        commands >= parameters.maximum_motor_thrust - tolerance
    )
    return QuadrotorSimulationResult(
        times=times,
        positions=states[:, 0:3],
        velocities=states[:, 3:6],
        quaternions=states[:, 6:10],
        angular_rates=states[:, 10:13],
        motor_thrusts=states[:, 13:17],
        commanded_motor_thrusts=commands,
        reference_positions=reference_positions,
        reference_velocities=reference_velocities,
        reference_accelerations=reference_accelerations,
        tracking_rmse=float(np.sqrt(np.mean(errors**2))),
        maximum_tracking_error=float(np.max(errors)),
        motor_saturation_fraction=float(np.mean(saturation)),
        maximum_tilt_degrees=float(np.max(tilt)),
    )
