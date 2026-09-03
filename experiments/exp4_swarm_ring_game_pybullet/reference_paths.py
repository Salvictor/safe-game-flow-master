"""多环穿环参考轨迹：并列出发 → 依次穿环 → 出口"""

from __future__ import annotations

import numpy as np

from config import AGENTS, H
from ring_planner import plan_all_agents, plan_agent_waypoints


def _interp_waypoints(waypoints: list, n: int) -> np.ndarray:
    pts = np.asarray(waypoints, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.tile(pts[0], (n, 1))
    sq = np.linspace(0.0, s[-1], n)
    out = np.zeros((n, 3))
    j = 0
    for i, si in enumerate(sq):
        while j < len(s) - 2 and s[j + 1] < si:
            j += 1
        d = s[j + 1] - s[j]
        a = 0.0 if d < 1e-12 else (si - s[j]) / d
        out[i] = (1 - a) * pts[j] + a * pts[j + 1]
    return out


def build_agent_references(h: int = H) -> list[np.ndarray]:
    refs = []
    for ag in AGENTS:
        _, wps = plan_agent_waypoints(ag)
        refs.append(_interp_waypoints(wps, h))
    return refs


def get_planned_orders_and_waypoints():
    return plan_all_agents()


def nominal_velocity(
    traj: np.ndarray,
    target: np.ndarray,
    t: float,
    eps: float = 1e-3,
) -> np.ndarray:
    lam = 0.85 / (1.0 - t + eps)
    v = lam * (target - traj)
    if traj.shape[0] >= 3:
        smooth = np.zeros_like(traj)
        smooth[1:-1] = 0.10 * (traj[:-2] - 2.0 * traj[1:-1] + traj[2:])
        v += smooth
    return v
