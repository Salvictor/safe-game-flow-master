"""穿环路径规划：各机最大化穿环数量（贪心 + 优先序博弈）"""

from __future__ import annotations

import itertools

import numpy as np

from config import AGENTS, GOAL_X, RINGS


def _path_length(waypoints: list[tuple[float, float, float]]) -> float:
    pts = np.asarray(waypoints, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _best_ring_order(
    start: tuple[float, float, float], priority_offset: int = 0,
) -> tuple[list[dict], list[tuple[float, float, float]]]:
    """枚举四环排列，选最短路径（穿环数恒为 4）；优先序偏移体现博弈差异"""
    ring_list = list(RINGS)
    best_wps: list[tuple[float, float, float]] | None = None
    best_len = np.inf
    best_order: list[dict] = ring_list

    orders = list(itertools.permutations(ring_list))
    # 博弈：把以 priority_offset 起始的排列排在前面（同长度时优先）
    rotated = orders[priority_offset % 24 :] + orders[: priority_offset % 24]

    for perm in rotated:
        centers = [tuple(r["center"]) for r in perm]
        wps = [start] + centers + [(GOAL_X, start[1], start[2])]
        ln = _path_length(wps)
        if ln < best_len - 1e-6:
            best_len = ln
            best_wps = wps
            best_order = list(perm)

    assert best_wps is not None
    return best_order, best_wps


def plan_agent_waypoints(agent: dict) -> tuple[list[dict], list[tuple[float, float, float]]]:
    start = tuple(agent["start"])
    offset = int(agent.get("ring_priority_offset", 0))
    order, wps = _best_ring_order(start, offset)
    return order, wps


def plan_all_agents() -> tuple[list[list[dict]], list[list[tuple[float, float, float]]]]:
    orders, all_wps = [], []
    for ag in AGENTS:
        order, wps = plan_agent_waypoints(ag)
        orders.append(order)
        all_wps.append(wps)
    return orders, all_wps
