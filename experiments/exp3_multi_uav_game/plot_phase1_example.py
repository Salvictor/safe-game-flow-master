#!/usr/bin/env python3
"""Plot an automatically selected median Phase-1 trajectory and speed profile."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS = EXPERIMENT_DIR / "results"
EVALUATION = RESULTS / "phase1_v3_fast_best_of_4" / "evaluation"
DATA = RESULTS / "phase1_v1" / "data_test"
OUTPUT = RESULTS / "phase1_configuration_comparison"


def main() -> None:
    with (EVALUATION / "per_trajectory.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["method"] == "game_cfm_hocbf"
        ]
    margins = np.asarray([float(row["minimum_separation_m"]) for row in rows])
    median = float(np.median(margins))
    index = int(rows[int(np.argmin(np.abs(margins - median)))]["sample"])
    references = np.load(DATA / "joint_references.npy")[index].reshape(2, 3, -1)
    experts = np.load(DATA / "joint_trajectories.npy")[index].reshape(2, 3, -1)
    plans = np.load(EVALUATION / "generated_anchored_plans.npy")[index].reshape(2, 3, -1)
    executed = np.load(EVALUATION / "executed_game_cfm.npy")[index].reshape(2, 3, -1)
    points = references.shape[-1]
    times = np.linspace(0.0, 5.5, points)

    def speed(trajectory: np.ndarray) -> np.ndarray:
        velocity = np.gradient(trajectory, times, axis=2, edge_order=2)
        return np.linalg.norm(velocity, axis=1)

    def acceleration(trajectory: np.ndarray) -> np.ndarray:
        velocity = np.gradient(trajectory, times, axis=2, edge_order=2)
        accel = np.gradient(velocity, times, axis=2, edge_order=2)
        return np.linalg.norm(accel, axis=1)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ("tab:blue", "tab:orange")
    for agent in range(2):
        axes[0, 0].plot(
            references[agent, 0], references[agent, 1], "--", color=colors[agent], alpha=0.5
        )
        axes[0, 0].plot(
            plans[agent, 0], plans[agent, 1], ":", color=colors[agent], linewidth=2
        )
        axes[0, 0].plot(
            executed[agent, 0], executed[agent, 1], color=colors[agent], label=f"UAV {agent}"
        )
        axes[0, 1].plot(times, speed(plans)[agent], color=colors[agent], label=f"plan UAV {agent}")
        axes[0, 1].plot(
            times,
            speed(executed)[agent],
            "--",
            color=colors[agent],
            label=f"executed UAV {agent}",
        )
        axes[1, 1].plot(
            times, acceleration(plans)[agent], color=colors[agent], label=f"UAV {agent}"
        )
    axes[0, 0].axvline(0.0, color="black", linewidth=1, label="ring plane")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].set_title("Top view: dashed reference, dotted plan, solid execution")
    axes[0, 0].legend()
    axes[0, 0].axis("equal")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("speed (m/s)")
    axes[0, 1].set_title("Planned and executed speed")
    axes[0, 1].legend(fontsize=8)
    separation_plan = np.linalg.norm(plans[0] - plans[1], axis=0)
    separation_executed = np.linalg.norm(executed[0] - executed[1], axis=0)
    separation_expert = np.linalg.norm(experts[0] - experts[1], axis=0)
    axes[1, 0].plot(times, separation_plan, label="selected plan")
    axes[1, 0].plot(times, separation_executed, label="executed plan")
    axes[1, 0].plot(times, separation_expert, "--", label="expert")
    axes[1, 0].axhline(0.36, color="tab:green", linestyle=":", label="HOCBF margin")
    axes[1, 0].axhline(0.25, color="tab:red", linestyle=":", label="collision distance")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("inter-UAV separation (m)")
    axes[1, 0].set_title("Pairwise separation")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].axhline(3.0, color="tab:red", linestyle=":", label="3 m/s² limit")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("planned acceleration (m/s²)")
    axes[1, 1].set_title("Planned acceleration")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle(f"Phase-1 median-clearance test case (fixed selection rule, sample {index})")
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "phase1_median_case_trajectory_speed.png", dpi=220)
    plt.close(figure)
    (OUTPUT / "selected_example.txt").write_text(
        f"sample={index}\nselection=closest to median executed minimum separation\n",
        encoding="utf-8",
    )
    print(f"Saved median-case plot for sample {index}")


if __name__ == "__main__":
    main()
