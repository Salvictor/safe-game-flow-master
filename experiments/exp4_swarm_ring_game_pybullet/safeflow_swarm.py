"""四机安全流匹配生成：名义流场 + 集群博弈 CBF-QP"""

from __future__ import annotations

import time

import numpy as np

from config import DT_FLOW, K, NUM_AGENTS
from game_cbf_qp import min_pairwise_distance, solve_game_cbf_qp
from reference_paths import build_agent_references, nominal_velocity


def generate_swarm_safe_flow(
    refs: list[np.ndarray] | None = None,
    k_steps: int = K,
    dt: float = DT_FLOW,
    naive: bool = False,
) -> dict:
    refs = refs or build_agent_references()
    trajs = [r.copy() for r in refs]
    n = NUM_AGENTS

    min_dist_log: list[float] = []
    u_norm_log: list[float] = []
    t0 = time.perf_counter()

    for step in range(k_steps):
        t = step / k_steps
        vels = [nominal_velocity(trajs[i], refs[i], t) for i in range(n)]
        if naive:
            us = [np.zeros_like(v) for v in vels]
        else:
            us = solve_game_cbf_qp(trajs, vels, t)
        for i in range(n):
            trajs[i] = trajs[i] + dt * (vels[i] + us[i])
        min_dist_log.append(min_pairwise_distance(trajs))
        u_norm_log.append(float(np.mean([np.linalg.norm(u) for u in us])))

    elapsed = time.perf_counter() - t0
    return {
        "trajs": trajs,
        "refs": refs,
        "min_dist": np.array(min_dist_log),
        "u_norm": np.array(u_norm_log),
        "plan_time_s": elapsed,
        "min_pairwise_dist": float(min(min_dist_log)),
        "method": "NaiveFlow" if naive else "SafeFlowGame",
    }
