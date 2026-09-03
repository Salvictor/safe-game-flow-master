"""Feasible single-UAV expert trajectories for three-dimensional ring traversal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safe_game_flow.geometry import RingGeometry
from safe_game_flow.trajectories import QuinticSegment, fit_quintic_segment


CONDITION_SCHEMA = (
    "start_position_x", "start_position_y", "start_position_z",
    "start_velocity_x", "start_velocity_y", "start_velocity_z",
    "goal_position_x", "goal_position_y", "goal_position_z",
    "goal_velocity_x", "goal_velocity_y", "goal_velocity_z",
    "ring_center_x", "ring_center_y", "ring_center_z",
    "ring_normal_x", "ring_normal_y", "ring_normal_z",
    "ring_major_radius", "ring_tube_radius", "uav_radius", "margin", "flight_time",
)


@dataclass(frozen=True)
class RingDatasetConfig:
    """Sampling ranges and feasibility limits, using SI units."""

    horizon: int = 100
    dense_check_points: int = 500
    max_attempts_per_sample: int = 100
    ring_center_low: tuple[float, float, float] = (-0.5, -0.5, 1.0)
    ring_center_high: tuple[float, float, float] = (0.5, 0.5, 2.0)
    maximum_normal_tilt_deg: float = 25.0
    major_radius_range: tuple[float, float] = (0.55, 0.85)
    tube_radius_range: tuple[float, float] = (0.035, 0.065)
    uav_radius: float = 0.09
    safety_margin: float = 0.04
    approach_distance_range: tuple[float, float] = (1.4, 2.2)
    exit_distance_range: tuple[float, float] = (1.4, 2.2)
    crossing_speed_range: tuple[float, float] = (0.35, 0.75)
    flight_time_range: tuple[float, float] = (4.0, 6.0)
    maximum_speed: float = 2.0
    maximum_acceleration: float = 3.0
    crossing_offset_fraction: float = 0.35
    endpoint_offset_fraction: float = 0.45

    def __post_init__(self) -> None:
        if self.horizon < 5:
            raise ValueError("horizon must be at least 5")
        if self.dense_check_points < self.horizon:
            raise ValueError("dense_check_points must be at least horizon")
        if self.max_attempts_per_sample < 1:
            raise ValueError("max_attempts_per_sample must be positive")
        if not 0 <= self.maximum_normal_tilt_deg < 90:
            raise ValueError("maximum_normal_tilt_deg must be in [0,90)")
        if self.maximum_speed <= 0 or self.maximum_acceleration <= 0:
            raise ValueError("dynamic limits must be positive")
        for name in (
            "major_radius_range", "tube_radius_range", "approach_distance_range",
            "exit_distance_range", "crossing_speed_range", "flight_time_range",
        ):
            low, high = getattr(self, name)
            if low <= 0 or high < low:
                raise ValueError(f"Invalid positive range for {name}: {(low, high)}")


@dataclass(frozen=True)
class RingTrajectorySample:
    trajectory: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    condition: np.ndarray
    ring: RingGeometry
    flight_time: float
    crossing_index: int
    min_barrier: float
    max_speed: float
    max_acceleration: float
    path_length: float


def _ring_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(normal, reference)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def _random_in_plane_offset(
    rng: np.random.Generator,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    maximum_radius: float,
) -> np.ndarray:
    radius = maximum_radius * np.sqrt(rng.uniform(0.0, 1.0))
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return radius * (np.cos(angle) * axis_u + np.sin(angle) * axis_v)


def _sample_normal(rng: np.random.Generator, maximum_tilt_deg: float) -> np.ndarray:
    tilt = np.deg2rad(rng.uniform(0.0, maximum_tilt_deg))
    azimuth = rng.uniform(0.0, 2.0 * np.pi)
    normal = np.array(
        [np.cos(tilt), np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth)]
    )
    return normal / np.linalg.norm(normal)


def _evaluate_piecewise(
    first: QuinticSegment,
    second: QuinticSegment,
    horizon: int,
    derivative: int,
) -> tuple[np.ndarray, int]:
    total_duration = first.duration + second.duration
    global_times = np.linspace(0.0, total_duration, horizon)
    first_mask = global_times <= first.duration
    values = np.empty((horizon, first.dimension), dtype=float)
    values[first_mask] = first.evaluate(global_times[first_mask], derivative)
    values[~first_mask] = second.evaluate(
        global_times[~first_mask] - first.duration, derivative
    )
    crossing_index = int(np.argmin(np.abs(global_times - first.duration)))
    return values, crossing_index


def _condition_vector(
    start: np.ndarray,
    start_velocity: np.ndarray,
    goal: np.ndarray,
    goal_velocity: np.ndarray,
    ring: RingGeometry,
    flight_time: float,
) -> np.ndarray:
    return np.concatenate(
        [
            start,
            start_velocity,
            goal,
            goal_velocity,
            ring.center,
            ring.normal,
            np.array(
                [
                    ring.major_radius,
                    ring.tube_radius,
                    ring.uav_radius,
                    ring.margin,
                    flight_time,
                ]
            ),
        ]
    )


def generate_ring_trajectory(
    rng: np.random.Generator,
    config: RingDatasetConfig,
) -> RingTrajectorySample:
    """Sample one C2-continuous, dynamically feasible expert trajectory."""
    center = rng.uniform(config.ring_center_low, config.ring_center_high)
    normal = _sample_normal(rng, config.maximum_normal_tilt_deg)
    ring = RingGeometry(
        center=center,
        normal=normal,
        major_radius=rng.uniform(*config.major_radius_range),
        tube_radius=rng.uniform(*config.tube_radius_range),
        uav_radius=config.uav_radius,
        margin=config.safety_margin,
    )
    axis_u, axis_v = _ring_plane_basis(ring.normal)
    start_offset = _random_in_plane_offset(
        rng, axis_u, axis_v, config.endpoint_offset_fraction * ring.clear_aperture_radius
    )
    goal_offset = _random_in_plane_offset(
        rng, axis_u, axis_v, config.endpoint_offset_fraction * ring.clear_aperture_radius
    )
    crossing_offset = _random_in_plane_offset(
        rng, axis_u, axis_v, config.crossing_offset_fraction * ring.clear_aperture_radius
    )
    start = center - rng.uniform(*config.approach_distance_range) * normal + start_offset
    goal = center + rng.uniform(*config.exit_distance_range) * normal + goal_offset
    crossing = center + crossing_offset
    start_velocity = np.zeros(3)
    goal_velocity = np.zeros(3)
    crossing_velocity = rng.uniform(*config.crossing_speed_range) * normal
    zero_acceleration = np.zeros(3)
    flight_time = float(rng.uniform(*config.flight_time_range))

    first_distance = np.linalg.norm(crossing - start)
    second_distance = np.linalg.norm(goal - crossing)
    first_duration = flight_time * first_distance / (first_distance + second_distance)
    second_duration = flight_time - first_duration
    first = fit_quintic_segment(
        start, start_velocity, zero_acceleration,
        crossing, crossing_velocity, zero_acceleration, first_duration,
    )
    second = fit_quintic_segment(
        crossing, crossing_velocity, zero_acceleration,
        goal, goal_velocity, zero_acceleration, second_duration,
    )

    trajectory, crossing_index = _evaluate_piecewise(first, second, config.horizon, 0)
    velocity, _ = _evaluate_piecewise(first, second, config.horizon, 1)
    acceleration, _ = _evaluate_piecewise(first, second, config.horizon, 2)

    dense_first_count = config.dense_check_points // 2 + 1
    dense_second_count = config.dense_check_points - dense_first_count
    dense_first_times = np.linspace(0.0, first.duration, dense_first_count)
    dense_second_times = np.linspace(0.0, second.duration, dense_second_count + 1)[1:]
    dense_position = np.concatenate(
        [first.evaluate(dense_first_times), second.evaluate(dense_second_times)], axis=0
    )
    dense_velocity = np.concatenate(
        [first.evaluate(dense_first_times, 1), second.evaluate(dense_second_times, 1)], axis=0
    )
    dense_acceleration = np.concatenate(
        [first.evaluate(dense_first_times, 2), second.evaluate(dense_second_times, 2)], axis=0
    )

    min_barrier = float(np.min(ring.barrier_value(dense_position)))
    max_speed = float(np.max(np.linalg.norm(dense_velocity, axis=1)))
    max_acceleration = float(np.max(np.linalg.norm(dense_acceleration, axis=1)))
    crossing_result = ring.detect_ring_crossing(trajectory, direction=1)
    if not crossing_result.passed:
        raise ValueError(f"Generated trajectory does not pass ring: {crossing_result.reason}")
    if min_barrier < 0:
        raise ValueError(f"Generated trajectory intersects ring frame: min_h={min_barrier}")
    if max_speed > config.maximum_speed:
        raise ValueError(f"Generated trajectory exceeds speed limit: {max_speed}")
    if max_acceleration > config.maximum_acceleration:
        raise ValueError(f"Generated trajectory exceeds acceleration limit: {max_acceleration}")

    condition = _condition_vector(
        start, start_velocity, goal, goal_velocity, ring, flight_time
    )
    path_length = float(np.sum(np.linalg.norm(np.diff(dense_position, axis=0), axis=1)))
    return RingTrajectorySample(
        trajectory=trajectory,
        velocity=velocity,
        acceleration=acceleration,
        condition=condition,
        ring=ring,
        flight_time=flight_time,
        crossing_index=crossing_index,
        min_barrier=min_barrier,
        max_speed=max_speed,
        max_acceleration=max_acceleration,
        path_length=path_length,
    )


def generate_ring_dataset(
    num_samples: int,
    seed: int,
    config: RingDatasetConfig | None = None,
) -> list[RingTrajectorySample]:
    """Generate a reproducible collection, resampling rejected conditions."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    config = config or RingDatasetConfig()
    rng = np.random.default_rng(seed)
    samples: list[RingTrajectorySample] = []
    attempts = 0
    maximum_attempts = num_samples * config.max_attempts_per_sample
    while len(samples) < num_samples and attempts < maximum_attempts:
        attempts += 1
        try:
            samples.append(generate_ring_trajectory(rng, config))
        except ValueError:
            continue
    if len(samples) != num_samples:
        raise RuntimeError(
            f"Generated only {len(samples)}/{num_samples} feasible trajectories "
            f"after {attempts} attempts"
        )
    return samples
