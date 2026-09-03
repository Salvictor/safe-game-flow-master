"""Two-UAV non-cooperative ring-access game and relative-degree-two HOCBF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GameSchedule:
    actions: tuple[str, str]
    delays: tuple[float, float]
    expected_crossing_times: tuple[float, float]
    payoffs: tuple[float, float]
    pure_nash_equilibria: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class HOCBFProjection:
    accelerations: np.ndarray
    intervened: bool
    nominal_constraint_value: float
    projected_constraint_value: float
    correction_norm: float


def _payoffs(
    crossing_times: tuple[float, float],
    *,
    safety_gap: float,
    collision_penalty: float,
    first_reward: float,
    second_reward: float,
    time_weight: float,
) -> tuple[float, float]:
    first, second = crossing_times
    unsafe = abs(first - second) < safety_gap
    if unsafe:
        return (
            -collision_penalty - time_weight * first,
            -collision_penalty - time_weight * second,
        )
    if first < second:
        rewards = (first_reward, second_reward)
    else:
        rewards = (second_reward, first_reward)
    return (
        rewards[0] - time_weight * first,
        rewards[1] - time_weight * second,
    )


def select_pure_nash_schedule(
    nominal_crossing_times: tuple[float, float],
    *,
    safety_gap: float = 0.45,
    safety_buffer: float = 0.05,
    minimum_yield_delay: float = 0.05,
    collision_penalty: float = 20.0,
    first_reward: float = 10.0,
    second_reward: float = 7.0,
    time_weight: float = 0.5,
) -> GameSchedule:
    """Solve a two-action simultaneous access game by pure best responses.

    Each UAV independently chooses ``go`` or ``yield``. A deterministic
    equilibrium selector resolves the usual two-equilibrium chicken-game tie
    in favor of the UAV with the earlier nominal arrival (UAV 0 on exact ties).
    """
    nominal = tuple(float(item) for item in nominal_crossing_times)
    if safety_gap <= 0 or safety_buffer < 0 or minimum_yield_delay <= 0:
        raise ValueError("game timing parameters must be positive")
    actions = ("go", "yield")
    # A yield action waits only as long as needed to enter after the other
    # agent's nominal passage plus the requested temporal safety margin.
    yield_delays = (
        max(
            minimum_yield_delay,
            nominal[1] + safety_gap + safety_buffer - nominal[0],
        ),
        max(
            minimum_yield_delay,
            nominal[0] + safety_gap + safety_buffer - nominal[1],
        ),
    )
    payoff_table: dict[tuple[str, str], tuple[float, float]] = {}
    crossing_table: dict[tuple[str, str], tuple[float, float]] = {}
    for action_0 in actions:
        for action_1 in actions:
            pair = (action_0, action_1)
            crossing = (
                nominal[0] + (yield_delays[0] if action_0 == "yield" else 0.0),
                nominal[1] + (yield_delays[1] if action_1 == "yield" else 0.0),
            )
            crossing_table[pair] = crossing
            payoff_table[pair] = _payoffs(
                crossing,
                safety_gap=safety_gap,
                collision_penalty=collision_penalty,
                first_reward=first_reward,
                second_reward=second_reward,
                time_weight=time_weight,
            )

    equilibria: list[tuple[str, str]] = []
    tolerance = 1e-12
    for pair, payoff in payoff_table.items():
        alternative_0 = ("yield" if pair[0] == "go" else "go", pair[1])
        alternative_1 = (pair[0], "yield" if pair[1] == "go" else "go")
        if (
            payoff[0] >= payoff_table[alternative_0][0] - tolerance
            and payoff[1] >= payoff_table[alternative_1][1] - tolerance
        ):
            equilibria.append(pair)
    if not equilibria:
        # This should be rare for the configured chicken game. A social-cost
        # fallback is explicit rather than silently pretending a Nash solution.
        selected = max(payoff_table, key=lambda pair: sum(payoff_table[pair]))
    elif len(equilibria) == 1:
        selected = equilibria[0]
    else:
        priority = 0 if nominal[0] <= nominal[1] else 1
        preferred = ("go", "yield") if priority == 0 else ("yield", "go")
        selected = preferred if preferred in equilibria else max(
            equilibria, key=lambda pair: sum(payoff_table[pair])
        )
    return GameSchedule(
        actions=selected,
        delays=(
            yield_delays[0] if selected[0] == "yield" else 0.0,
            yield_delays[1] if selected[1] == "yield" else 0.0,
        ),
        expected_crossing_times=crossing_table[selected],
        payoffs=payoff_table[selected],
        pure_nash_equilibria=tuple(equilibria),
    )


def project_inter_uav_hocbf(
    positions: np.ndarray,
    velocities: np.ndarray,
    nominal_accelerations: np.ndarray,
    *,
    minimum_distance: float,
    alpha_0: float = 16.0,
    alpha_1: float = 8.0,
) -> HOCBFProjection:
    """Minimum-norm joint acceleration projection for pairwise separation.

    For ``h = ||p0-p1||²-d²`` and double-integrator dynamics, enforce
    ``h_ddot + alpha_1 h_dot + alpha_0 h >= 0``. The single affine constraint
    has a closed-form Euclidean projection in the six-dimensional joint input.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    nominal = np.asarray(nominal_accelerations, dtype=float)
    if positions.shape != (2, 3) or velocities.shape != (2, 3) or nominal.shape != (2, 3):
        raise ValueError("positions, velocities, and accelerations must have shape (2,3)")
    if minimum_distance <= 0 or alpha_0 <= 0 or alpha_1 <= 0:
        raise ValueError("HOCBF parameters must be positive")
    relative_position = positions[0] - positions[1]
    relative_velocity = velocities[0] - velocities[1]
    barrier = float(relative_position @ relative_position - minimum_distance**2)
    barrier_rate = float(2.0 * relative_position @ relative_velocity)
    normal = np.concatenate([2.0 * relative_position, -2.0 * relative_position])
    offset = (
        2.0 * float(relative_velocity @ relative_velocity)
        + alpha_1 * barrier_rate
        + alpha_0 * barrier
    )
    nominal_flat = nominal.reshape(-1)
    nominal_value = float(normal @ nominal_flat + offset)
    squared_norm = float(normal @ normal)
    if nominal_value >= 0.0 or squared_norm < 1e-12:
        return HOCBFProjection(nominal.copy(), False, nominal_value, nominal_value, 0.0)
    correction = (-nominal_value / squared_norm) * normal
    projected = nominal_flat + correction
    projected_value = float(normal @ projected + offset)
    return HOCBFProjection(
        accelerations=projected.reshape(2, 3),
        intervened=True,
        nominal_constraint_value=nominal_value,
        projected_constraint_value=projected_value,
        correction_norm=float(np.linalg.norm(correction)),
    )
