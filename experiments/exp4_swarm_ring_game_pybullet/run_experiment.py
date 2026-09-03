#!/usr/bin/env python3
"""
实验4：四机穿环集群博弈 + PyBullet 仿真

场景：四环同高、XY 方阵；四机环外并列同时出发；目标尽可能多穿环。
方法：贪心穿环路径 + 安全流匹配 + 博弈 CBF-QP + 穿环奖励机制。

Usage:
  python experiments/exp4_swarm_ring_game_pybullet/run_experiment.py
  python experiments/exp4_swarm_ring_game_pybullet/run_experiment.py --gui
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

from config import (  # noqa: E402
    AGENTS,
    D_SAFE,
    FLIGHT_Z,
    H,
    K,
    OUTPUT_DIR,
    REWARD_ALL_RINGS_BONUS,
    REWARD_PER_RING,
    RING_RADIUS,
    RINGS,
    START_X,
)
from reference_paths import build_agent_references, get_planned_orders_and_waypoints  # noqa: E402
from ring_reward import evaluate_swarm  # noqa: E402
from safeflow_swarm import generate_swarm_safe_flow  # noqa: E402


def plot_3d_trajectories(
    safe_trajs: list[np.ndarray],
    naive_trajs: list[np.ndarray],
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 7))
    for ax_idx, (trajs, title) in enumerate([
        (naive_trajs, "Naive Flow (no game CBF)"),
        (safe_trajs, "Safe Flow Game (CBF)"),
    ]):
        ax = fig.add_subplot(1, 2, ax_idx + 1, projection="3d")
        for i, ag in enumerate(AGENTS):
            c = ag["color"]
            ax.plot(trajs[i][:, 0], trajs[i][:, 1], trajs[i][:, 2], color=c, lw=2, label=ag["name"])
        for ring in RINGS:
            cx, cy, cz = ring["center"]
            t = np.linspace(0, 2 * np.pi, 40)
            ax.plot(cx, cy + RING_RADIUS * np.cos(t), cz + RING_RADIUS * np.sin(t),
                    color=ring["color"], lw=1.5, alpha=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_trajectories_topdown(safe_trajs: list[np.ndarray], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_title("Top View (XY): coplanar rings & multi-ring paths", fontweight="bold")
    for ring in RINGS:
        cx, cy = ring["center"][0], ring["center"][1]
        circ = plt.Circle((cx, cy), RING_RADIUS, fill=False, color=ring["color"], lw=2.2)
        ax.add_patch(circ)
        ax.scatter(cx, cy, color=ring["color"], s=80, zorder=4)
        ax.text(cx, cy + RING_RADIUS + 0.3, ring["name"], ha="center", fontsize=8, color=ring["color"])
    ax.axvline(START_X, color="gold", ls="--", lw=2, label="Start line")
    for i, ag in enumerate(AGENTS):
        c = ag["color"]
        ax.plot(safe_trajs[i][:, 0], safe_trajs[i][:, 1], color=c, lw=2, label=ag["name"])
        ax.scatter(*ag["start"][:2], color=c, marker="s", s=60, edgecolors="k", zorder=5)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_rewards(reward_report: dict, save_path: Path) -> None:
    agents = reward_report["per_agent"]
    names = [a["agent"] for a in agents]
    totals = [a["total_reward"] for a in agents]
    n_rings = [a["n_rings"] for a in agents]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [AGENTS[i]["color"] for i in range(len(agents))]

    ax = axes[0]
    bars = ax.bar(names, totals, color=colors, alpha=0.85, edgecolor="k")
    ax.set_ylabel("Total reward")
    ax.set_title("Agent reward (maximize rings + bonus - costs)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, nr in zip(bars, n_rings):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{nr}/4 rings", ha="center", fontsize=9)

    ax2 = axes[1]
    comp_labels = ["Ring", "All-4 bonus", "Path", "Control", "Unsafe"]
    x = np.arange(len(names))
    w = 0.15
    comps = [
        [a["reward_ring"] for a in agents],
        [a["reward_all_bonus"] for a in agents],
        [a["penalty_path"] for a in agents],
        [a["penalty_control"] for a in agents],
        [a["penalty_unsafe"] for a in agents],
    ]
    for j, comp in enumerate(comps):
        ax2.bar(x + (j - 2) * w, comp, w, label=comp_labels[j], alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax2.set_ylabel("Reward component")
    ax2.set_title("Reward breakdown", fontweight="bold")
    ax2.legend(fontsize=7); ax2.grid(True, axis="y", alpha=0.3)
    ax2.axhline(0, color="k", lw=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_safety_metrics(safe_res: dict, naive_res: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t = np.linspace(0, 1, len(safe_res["min_dist"]))

    ax = axes[0]
    ax.plot(t, naive_res["min_dist"], color="#e67e22", lw=2, label="Naive")
    ax.plot(t, safe_res["min_dist"], color="#27ae60", lw=2, label="Safe Flow Game")
    ax.axhline(D_SAFE, color="r", ls="--", lw=1.5, label=f"d_safe={D_SAFE}m")
    ax.fill_between(t, 0, D_SAFE, color="r", alpha=0.08)
    ax.set_xlabel("Flow time t")
    ax.set_ylabel("Min pairwise distance (m)")
    ax.set_title("Inter-agent distance during generation", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    methods = ["Naive", "SafeFlow Game"]
    times = [naive_res["plan_time_s"], safe_res["plan_time_s"]]
    dists = [naive_res["min_pairwise_dist"], safe_res["min_pairwise_dist"]]
    x = np.arange(2)
    w = 0.35
    ax2.bar(x - w / 2, times, w, label="Plan time (s)", color="#3498db", alpha=0.85)
    ax2b = ax2.twinx()
    ax2b.bar(x + w / 2, dists, w, label="Min dist (m)", color="#2ecc71", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(methods)
    ax2.set_ylabel("Time (s)")
    ax2b.set_ylabel("Min distance (m)")
    ax2.set_title("Planning time vs safety", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_reward_table(report: dict) -> None:
    print("\n  Ring-passing reward (Safe Flow Game):")
    print(f"  {'Agent':<8} {'Rings':<12} {'RingR':>8} {'Bonus':>8} {'Total':>10}")
    print("  " + "-" * 50)
    for a in report["leaderboard"]:
        rings_str = ",".join(str(r) for r in a["rings_passed"]) or "-"
        print(
            f"  {a['agent']:<8} [{rings_str}]"
            f"  {a['reward_ring']:>8.0f} {a['reward_all_bonus']:>8.0f} {a['total_reward']:>10.1f}"
        )
    print(f"  Min sync distance: {report['min_sync_distance_m']:.3f} m (d_safe={D_SAFE}m)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pybullet", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Exp4: 4-UAV Multi-Ring Game — Safe Flow + Reward + PyBullet")
    print("=" * 72)

    orders, waypoints = get_planned_orders_and_waypoints()
    print(f"\nScene: {len(RINGS)} coplanar rings (Z={FLIGHT_Z}m), XY square half={RINGS[0]['center'][0]:+.1f}m")
    print(f"Reward: +{REWARD_PER_RING}/ring, +{REWARD_ALL_RINGS_BONUS} all-4 bonus")
    for i, ag in enumerate(AGENTS):
        ring_names = " -> ".join(r["name"] for r in orders[i])
        print(f"  {ag['name']}: start {ag['start']}  plan: {ring_names}")

    refs = build_agent_references(H)
    print(f"\nAgents: {len(AGENTS)}, H={H}, K={K}, d_safe={D_SAFE}m")

    print("\n[1] Safe Flow Game generation...")
    safe = generate_swarm_safe_flow(refs, naive=False)
    naive = generate_swarm_safe_flow(refs, naive=True)
    print(f"  Safe   min pairwise dist: {safe['min_pairwise_dist']:.3f} m,  time: {safe['plan_time_s']:.3f}s")
    print(f"  Naive  min pairwise dist: {naive['min_pairwise_dist']:.3f} m,  time: {naive['plan_time_s']:.3f}s")

    reward_safe = evaluate_swarm(safe["trajs"], safe["u_norm"])
    reward_naive = evaluate_swarm(naive["trajs"], naive["u_norm"])
    print_reward_table(reward_safe)

    for i, t in enumerate(safe["trajs"]):
        np.save(OUTPUT_DIR / f"traj_safe_agent{i}.npy", t)
        np.save(OUTPUT_DIR / f"traj_naive_agent{i}.npy", naive["trajs"][i])
        np.save(OUTPUT_DIR / f"ref_agent{i}.npy", refs[i])

    metrics = {
        "scene": {
            "flight_z": FLIGHT_Z,
            "ring_centers_xy": [r["center"][:2] for r in RINGS],
            "ring_radius": RING_RADIUS,
            "start_x": START_X,
        },
        "reward_config": {
            "per_ring": REWARD_PER_RING,
            "all_rings_bonus": REWARD_ALL_RINGS_BONUS,
        },
        "SafeFlowGame": {
            "min_pairwise_dist": safe["min_pairwise_dist"],
            "plan_time_s": safe["plan_time_s"],
            "rewards": reward_safe,
        },
        "NaiveFlow": {
            "min_pairwise_dist": naive["min_pairwise_dist"],
            "plan_time_s": naive["plan_time_s"],
            "rewards": reward_naive,
        },
        "d_safe": D_SAFE,
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_3d_trajectories(safe["trajs"], naive["trajs"], OUTPUT_DIR / "trajectories_3d.png")
    plot_trajectories_topdown(safe["trajs"], OUTPUT_DIR / "trajectories_topdown.png")
    plot_rewards(reward_safe, OUTPUT_DIR / "ring_rewards.png")
    plot_safety_metrics(safe, naive, OUTPUT_DIR / "safety_metrics.png")

    exec_trajs = None
    if not args.no_pybullet:
        print("\n[2] PyBullet multi-ring simulation...")
        try:
            from pybullet_sim import run_pybullet_swarm  # noqa: E402
            exec_trajs = run_pybullet_swarm(safe["trajs"], OUTPUT_DIR, gui=args.gui)
            if exec_trajs:
                exec_reward = evaluate_swarm(exec_trajs)
                print_reward_table(exec_reward)
                with open(OUTPUT_DIR / "exec_rewards.json", "w", encoding="utf-8") as f:
                    json.dump(exec_reward, f, indent=2, ensure_ascii=False)
        except ImportError as e:
            print(f"  PyBullet skipped: {e}")
    else:
        print("\n[2] PyBullet skipped.")

    print("\n" + "=" * 72)
    print(f"Done. Results: {OUTPUT_DIR}")
    print("  swarm_ring_flight.gif  trajectories_topdown.png  ring_rewards.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
