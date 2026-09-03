"""Reusable expert rollout utilities for two-UAV ring-access experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safe_game_flow.game.two_uav_ring import GameSchedule, project_inter_uav_hocbf
from safe_game_flow.trajectories import fit_quintic_segment


CONDITION_SCHEMA = (
    *(f"start_0_{axis}" for axis in "xyz"),
    *(f"start_1_{axis}" for axis in "xyz"),
    *(f"goal_0_{axis}" for axis in "xyz"),
    *(f"goal_1_{axis}" for axis in "xyz"),
    *(f"ring_center_{axis}" for axis in "xyz"),
    "duration_0",
    "duration_1",
    "nominal_crossing_0",
    "nominal_crossing_1",
    "yield_action_0",
    "yield_action_1",
    "delay_0",
    "delay_1",
    "horizon",
)


@dataclass(frozen=True)
class RingGameScene:
    starts: np.ndarray
    goals: np.ndarray
    center: np.ndarray
    durations: tuple[float, float]
    disturbance: np.ndarray


@dataclass(frozen=True)
class JointRollout:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    minimum_separation: float
    collision: bool
    intervention_rate: float
    mean_correction: float


@dataclass(frozen=True)
class QuinticReference:
    start: np.ndarray
    duration: float
    delay: float
    first_duration: float
    first: object
    second: object

    @property
    def crossing_time(self) -> float:
        return self.delay + self.first_duration

    def evaluate(self, time_value: float, derivative: int = 0) -> np.ndarray:
        if time_value < self.delay:
            return np.zeros(3) if derivative else self.start.copy()
        local = float(np.clip(time_value - self.delay, 0.0, self.duration))
        if local <= self.first_duration:
            return self.first.evaluate(np.asarray([local]), derivative)[0]
        return self.second.evaluate(
            np.asarray([local - self.first_duration]), derivative
        )[0]


def make_reference(
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    duration: float,
    delay: float,
) -> QuinticReference:
    first_distance = float(np.linalg.norm(center - start))
    second_distance = float(np.linalg.norm(goal - center))
    first_duration = duration * first_distance / (first_distance + second_distance)
    second_duration = duration - first_duration
    zero = np.zeros(3)
    crossing_velocity = np.array([0.55, 0.0, 0.0])
    first = fit_quintic_segment(
        start, zero, zero, center, crossing_velocity, zero, first_duration
    )
    second = fit_quintic_segment(
        center, crossing_velocity, zero, goal, zero, zero, second_duration
    )
    return QuinticReference(
        start=start.copy(),
        duration=float(duration),
        delay=float(delay),
        first_duration=float(first_duration),
        first=first,
        second=second,
    )


def sample_scene(rng: np.random.Generator) -> RingGameScene:
    center = np.array([0.0, 0.0, rng.uniform(1.0, 1.5)])
    lateral = rng.uniform(0.18, 0.38, size=2)
    starts = np.array(
        [
            [-rng.uniform(1.3, 1.8), -lateral[0], center[2] + rng.uniform(-0.08, 0.08)],
            [-rng.uniform(1.3, 1.8), lateral[1], center[2] + rng.uniform(-0.08, 0.08)],
        ]
    )
    goals = np.array(
        [
            [rng.uniform(1.3, 1.8), -lateral[0], center[2]],
            [rng.uniform(1.3, 1.8), lateral[1], center[2]],
        ]
    )
    durations = (float(rng.uniform(3.6, 4.4)), float(rng.uniform(3.6, 4.4)))
    disturbance = rng.normal(0.0, 0.08, size=(2, 3))
    return RingGameScene(starts, goals, center, durations, disturbance)


def nominal_crossing_times(scene: RingGameScene) -> tuple[float, float]:
    references = tuple(
        make_reference(
            scene.starts[index],
            scene.goals[index],
            scene.center,
            scene.durations[index],
            0.0,
        )
        for index in range(2)
    )
    return tuple(reference.crossing_time for reference in references)


def scheduled_references(
    scene: RingGameScene, schedule: GameSchedule
) -> tuple[QuinticReference, QuinticReference]:
    return tuple(
        make_reference(
            scene.starts[index],
            scene.goals[index],
            scene.center,
            scene.durations[index],
            schedule.delays[index],
        )
        for index in range(2)
    )


def reference_positions(
    references: tuple[QuinticReference, QuinticReference], times: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            [reference.evaluate(float(time_value), 0) for reference in references]
            for time_value in times
        ]
    )


def condition_vector(
    scene: RingGameScene,
    schedule: GameSchedule,
    *,
    horizon: float,
) -> np.ndarray:
    nominal = nominal_crossing_times(scene)
    return np.concatenate(
        [
            scene.starts.reshape(-1),
            scene.goals.reshape(-1),
            scene.center,
            np.asarray(scene.durations),
            np.asarray(nominal),
            np.asarray([float(action == "yield") for action in schedule.actions]),
            np.asarray(schedule.delays),
            np.asarray([horizon]),
        ]
    )


def simulate_expert(
    scene: RingGameScene,
    schedule: GameSchedule,
    *,
    horizon: float = 5.5,
    dt: float = 0.01,
    use_hocbf: bool = True,
    hocbf_distance: float = 0.36,
    physical_collision_distance: float = 0.25,
) -> JointRollout:
    if horizon <= 0 or dt <= 0:
        raise ValueError("horizon and dt must be positive")
    references = scheduled_references(scene, schedule)
    times = np.arange(0.0, horizon + 0.5 * dt, dt)
    positions = scene.starts.copy()
    velocities = np.zeros((2, 3))
    position_history = np.empty((len(times), 2, 3))
    velocity_history = np.empty_like(position_history)
    acceleration_history = np.empty_like(position_history)
    interventions = 0
    correction_sum = 0.0
    for time_index, time_value in enumerate(times):
        position_history[time_index] = positions
        velocity_history[time_index] = velocities
        nominal = np.empty((2, 3))
        for agent in range(2):
            reference_position = references[agent].evaluate(float(time_value), 0)
            reference_velocity = references[agent].evaluate(float(time_value), 1)
            reference_acceleration = references[agent].evaluate(float(time_value), 2)
            acceleration = (
                reference_acceleration
                + 5.0 * (reference_position - positions[agent])
                + 3.5 * (reference_velocity - velocities[agent])
            )
            norm = float(np.linalg.norm(acceleration))
            nominal[agent] = acceleration if norm <= 3.0 else acceleration * (3.0 / norm)
        if use_hocbf:
            projection = project_inter_uav_hocbf(
                positions,
                velocities,
                nominal,
                minimum_distance=hocbf_distance,
            )
            accelerations = projection.accelerations
            interventions += int(projection.intervened)
            correction_sum += projection.correction_norm
        else:
            accelerations = nominal
        acceleration_history[time_index] = accelerations
        if time_index == len(times) - 1:
            break
        velocities = velocities + (accelerations + scene.disturbance) * dt
        positions = positions + velocities * dt
    separation = np.linalg.norm(
        position_history[:, 0] - position_history[:, 1], axis=1
    )
    minimum_separation = float(np.min(separation))
    return JointRollout(
        times=times,
        positions=position_history,
        velocities=velocity_history,
        accelerations=acceleration_history,
        minimum_separation=minimum_separation,
        collision=minimum_separation < physical_collision_distance,
        intervention_rate=interventions / len(times),
        mean_correction=correction_sum / max(interventions, 1),
    )


def simulate_joint_reference(
    initial_positions: np.ndarray,
    trajectory: np.ndarray,
    disturbance: np.ndarray,
    *,
    horizon: float,
    dt: float = 0.01,
    use_hocbf: bool = True,
    hocbf_distance: float = 0.36,
    physical_collision_distance: float = 0.25,
) -> JointRollout:
    """Track a sampled joint position reference with the Phase-0 controller."""
    trajectory = np.asarray(trajectory, dtype=float)
    initial_positions = np.asarray(initial_positions, dtype=float)
    disturbance = np.asarray(disturbance, dtype=float)
    if trajectory.ndim != 3 or trajectory.shape[:2] != (2, 3):
        raise ValueError("trajectory must have shape (2,3,H)")
    if trajectory.shape[2] < 3:
        raise ValueError("trajectory requires at least three samples")
    if initial_positions.shape != (2, 3) or disturbance.shape != (2, 3):
        raise ValueError("initial_positions and disturbance must have shape (2,3)")
    sample_times = np.linspace(0.0, horizon, trajectory.shape[2])
    edge_order = 2 if trajectory.shape[2] >= 3 else 1
    reference_velocity = np.gradient(
        trajectory, sample_times, axis=2, edge_order=edge_order
    )
    reference_acceleration = np.gradient(
        reference_velocity, sample_times, axis=2, edge_order=edge_order
    )

    def interpolate(values: np.ndarray, time_value: float) -> np.ndarray:
        result = np.empty((2, 3))
        for agent in range(2):
            for axis in range(3):
                result[agent, axis] = np.interp(
                    time_value, sample_times, values[agent, axis]
                )
        return result

    times = np.arange(0.0, horizon + 0.5 * dt, dt)
    positions = initial_positions.copy()
    velocities = np.zeros((2, 3))
    position_history = np.empty((len(times), 2, 3))
    velocity_history = np.empty_like(position_history)
    acceleration_history = np.empty_like(position_history)
    interventions = 0
    correction_sum = 0.0
    for time_index, time_value in enumerate(times):
        position_history[time_index] = positions
        velocity_history[time_index] = velocities
        position_ref = interpolate(trajectory, float(time_value))
        velocity_ref = interpolate(reference_velocity, float(time_value))
        acceleration_ref = interpolate(reference_acceleration, float(time_value))
        nominal = acceleration_ref + 5.0 * (position_ref - positions) + 3.5 * (
            velocity_ref - velocities
        )
        for agent in range(2):
            norm = float(np.linalg.norm(nominal[agent]))
            if norm > 3.0:
                nominal[agent] *= 3.0 / norm
        if use_hocbf:
            projection = project_inter_uav_hocbf(
                positions,
                velocities,
                nominal,
                minimum_distance=hocbf_distance,
            )
            accelerations = projection.accelerations
            interventions += int(projection.intervened)
            correction_sum += projection.correction_norm
        else:
            accelerations = nominal
        acceleration_history[time_index] = accelerations
        if time_index == len(times) - 1:
            break
        velocities = velocities + (accelerations + disturbance) * dt
        positions = positions + velocities * dt
    separation = np.linalg.norm(
        position_history[:, 0] - position_history[:, 1], axis=1
    )
    minimum_separation = float(np.min(separation))
    return JointRollout(
        times=times,
        positions=position_history,
        velocities=velocity_history,
        accelerations=acceleration_history,
        minimum_separation=minimum_separation,
        collision=minimum_separation < physical_collision_distance,
        intervention_rate=interventions / len(times),
        mean_correction=correction_sum / max(interventions, 1),
    )


def resample_joint_positions(rollout: JointRollout, points: int) -> np.ndarray:
    if points < 2:
        raise ValueError("points must be at least 2")
    query = np.linspace(rollout.times[0], rollout.times[-1], points)
    sampled = np.empty((points, 2, 3))
    for agent in range(2):
        for axis in range(3):
            sampled[:, agent, axis] = np.interp(
                query, rollout.times, rollout.positions[:, agent, axis]
            )
    return sampled.transpose(1, 2, 0).reshape(6, points)


def resample_reference_positions(
    references: tuple[QuinticReference, QuinticReference],
    *,
    horizon: float,
    points: int,
) -> np.ndarray:
    times = np.linspace(0.0, horizon, points)
    positions = reference_positions(references, times)
    return positions.transpose(1, 2, 0).reshape(6, points)
