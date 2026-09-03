"""穿环奖励：每穿一环得分 + 全环奖励 - 路径/控制/安全惩罚"""

from __future__ import annotations

import itertools

import numpy as np

from config import (
    AGENTS,
    D_SAFE,
    PENALTY_CONTROL,
    PENALTY_PATH_LENGTH,
    PENALTY_UNSAFE,
    REWARD_ALL_RINGS_BONUS,
    REWARD_PER_RING,
    RING_PASS_TOL,
    RING_RADIUS,
    RINGS,
)


def _ring_plane_x(ring: dict) -> float:
    return float(ring["center"][0])


def detect_rings_passed(traj: np.ndarray, rings: list[dict] | None = None) -> list[int]:
    """检测轨迹穿过的环 id 列表（按穿过顺序）"""
    rings = rings or RINGS
    passed: list[int] = []
    for ring in rings:
        cx, cy, cz = ring["center"]
        rx = float(cx)
        for k in range(len(traj) - 1):
            x0, x1 = traj[k, 0], traj[k + 1, 0]
            if (x0 - rx) * (x1 - rx) > 0:
                continue
            if abs(x1 - x0) < 1e-9:
                p = traj[k]
            else:
                a = (rx - x0) / (x1 - x0)
                p = (1 - a) * traj[k] + a * traj[k + 1]
            d_yz = float(np.hypot(p[1] - cy, p[2] - cz))
            if d_yz <= RING_RADIUS + RING_PASS_TOL:
                passed.append(int(ring["id"]))
                break
    return passed


def path_length(traj: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def min_sync_distance(trajs: list[np.ndarray]) -> float:
    """按流时间同步位置的最小机间距离"""
    n = len(trajs[0])
    m = np.inf
    for k in range(n):
        pts = [t[k] for t in trajs]
        for i, j in itertools.combinations(range(len(trajs)), 2):
            m = min(m, float(np.linalg.norm(pts[i] - pts[j])))
    return float(m)


def compute_agent_reward(
    traj: np.ndarray,
    rings_passed: list[int],
    u_norm_mean: float = 0.0,
    min_pairwise: float = np.inf,
) -> dict:
    n_rings = len(rings_passed)
    r_ring = REWARD_PER_RING * n_rings
    r_bonus = REWARD_ALL_RINGS_BONUS if n_rings >= len(RINGS) else 0.0
    r_path = -PENALTY_PATH_LENGTH * path_length(traj)
    r_ctrl = -PENALTY_CONTROL * u_norm_mean
    r_safe = -PENALTY_UNSAFE if min_pairwise < D_SAFE else 0.0
    total = r_ring + r_bonus + r_path + r_ctrl + r_safe
    return {
        "rings_passed": rings_passed,
        "n_rings": n_rings,
        "reward_ring": r_ring,
        "reward_all_bonus": r_bonus,
        "penalty_path": r_path,
        "penalty_control": r_ctrl,
        "penalty_unsafe": r_safe,
        "total_reward": total,
    }


def evaluate_swarm(
    trajs: list[np.ndarray],
    u_norm_log: np.ndarray | None = None,
    agent_names: list[str] | None = None,
) -> dict:
    agent_names = agent_names or [ag["name"] for ag in AGENTS]
    min_pw = min_sync_distance(trajs)
    u_mean = float(np.mean(u_norm_log)) if u_norm_log is not None and len(u_norm_log) else 0.0

    per_agent = []
    for i, traj in enumerate(trajs):
        passed = detect_rings_passed(traj)
        rw = compute_agent_reward(traj, passed, u_mean, min_pw)
        rw["agent"] = agent_names[i]
        rw["agent_id"] = i
        per_agent.append(rw)

    return {
        "per_agent": per_agent,
        "min_sync_distance_m": min_pw,
        "leaderboard": sorted(per_agent, key=lambda x: x["total_reward"], reverse=True),
    }
