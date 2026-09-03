"""N 机集群博弈 CBF：集中式 v-GNE 最小范数 QP（与 exp2 同框架，扩展至 3D × N 机）"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import sparse

try:
    import osqp
except ImportError as e:
    raise ImportError("需要 osqp: pip install osqp") from e

from config import D_SAFE, NUM_AGENTS, PHI0, PHI1_SCALE, STATIC_OBSTACLES


def phi_fun(t: float, h_val: float) -> float:
    if h_val >= 0:
        return PHI0
    t_clip = min(t, 1.0 - 1e-6)
    return PHI1_SCALE / ((1.0 - t_clip) ** 2)


def _prepare_static():
    out = []
    for o in STATIC_OBSTACLES:
        d = dict(o)
        d["c"] = np.asarray(d["c"], dtype=float)
        d["r_eff"] = float(d["r"] + d["margin"])
        out.append(d)
    return out


def solve_game_cbf_qp(
    trajs: list[np.ndarray],
    vels: list[np.ndarray],
    t: float,
    d_safe: float = D_SAFE,
) -> list[np.ndarray]:
    """
    min 0.5 * sum_i ||u_i||^2
    s.t. 机间 CBF + 静态障碍 CBF

    Args:
        trajs: N × (H, 3)
        vels:  N × (H, 3)

    Returns:
        corrections: N × (H, 3)
    """
    n_agents = len(trajs)
    h_len = trajs[0].shape[0]
    dim = 3
    n_vars = n_agents * h_len * dim

    a_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []

    def _agent_slice(i: int, k: int) -> slice:
        base = i * h_len * dim + k * dim
        return slice(base, base + dim)

    # 静态障碍（对每机每点）
    for obs in _prepare_static():
        c, r_eff = obs["c"], obs["r_eff"]
        for i in range(n_agents):
            traj, vel = trajs[i], vels[i]
            h_val = np.sum((traj - c) ** 2, axis=1) - r_eff**2
            grad = 2.0 * (traj - c)
            phi = np.array([phi_fun(t, float(h)) for h in h_val])
            bi = -np.sum(grad * vel, axis=1) - phi * h_val
            for k in range(h_len):
                row = np.zeros(n_vars)
                row[_agent_slice(i, k)] = grad[k]
                a_rows.append(row)
                b_rows.append(bi[k])

    # 机间博弈 CBF：所有无序对 (i, j), i < j
    for i, j in itertools.combinations(range(n_agents), 2):
        pi, pj = trajs[i], trajs[j]
        vi, vj = vels[i], vels[j]
        d = pi - pj
        h_ij = np.sum(d * d, axis=1) - d_safe**2
        a_i = 2.0 * d
        a_j = -2.0 * d
        phi = np.array([phi_fun(t, float(h)) for h in h_ij])
        bi = -np.sum(a_i * vi, axis=1) + np.sum(a_j * vj, axis=1) - phi * h_ij
        for k in range(h_len):
            row = np.zeros(n_vars)
            row[_agent_slice(i, k)] = a_i[k]
            row[_agent_slice(j, k)] = a_j[k]
            a_rows.append(row)
            b_rows.append(bi[k])

    if not a_rows:
        return [np.zeros((h_len, dim)) for _ in range(n_agents)]

    a_big = np.vstack(a_rows)
    b_big = np.asarray(b_rows)
    p_mat = sparse.eye(n_vars, format="csc")
    q_vec = np.zeros(n_vars)
    a_sp = sparse.csc_matrix(a_big)

    prob = osqp.OSQP()
    prob.setup(
        p_mat, q_vec, a_sp, b_big, np.inf * np.ones(len(b_big)),
        verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=8000,
    )
    res = prob.solve()

    if res.info.status != "solved":
        return _fallback_pairwise(trajs, vels, t, d_safe)

    u_flat = res.x
    out = []
    for i in range(n_agents):
        block = u_flat[i * h_len * dim : (i + 1) * h_len * dim]
        out.append(block.reshape(h_len, dim))
    return out


def _fallback_pairwise(
    trajs: list[np.ndarray],
    vels: list[np.ndarray],
    t: float,
    d_safe: float,
) -> list[np.ndarray]:
    """逐对闭式最小范数修正回退"""
    n = len(trajs)
    h_len = trajs[0].shape[0]
    us = [np.zeros((h_len, 3)) for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        d = trajs[i] - trajs[j]
        h_ij = np.sum(d * d, axis=1) - d_safe**2
        for k in range(h_len):
            a = 2.0 * d[k]
            a2 = float(np.dot(a, a) + 1e-12)
            b = (
                -float(np.dot(a, vels[i][k]))
                + float(np.dot(a, vels[j][k]))
                - phi_fun(t, float(h_ij[k])) * float(h_ij[k])
            )
            if b > 0:
                ui = (b / (2 * a2)) * a
                us[i][k] += ui
                us[j][k] -= ui
    return us


def min_pairwise_distance(trajs: list[np.ndarray]) -> float:
    m = np.inf
    for i, j in itertools.combinations(range(len(trajs)), 2):
        d = np.linalg.norm(trajs[i] - trajs[j], axis=1).min()
        m = min(m, d)
    return float(m)
