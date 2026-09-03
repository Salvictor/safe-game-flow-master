#!/usr/bin/env python3
"""Summarize Phase-1 candidate-count and ODE-step trade-offs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS = EXPERIMENT_DIR / "results"
CONFIGURATIONS = (
    ("1 candidate / 16 steps", "phase1_v1", 1, 16),
    ("4 candidates / 8 steps", "phase1_v3_fast_best_of_4", 4, 8),
    ("8 candidates / 16 steps", "phase1_v2_best_of_8", 8, 16),
)


def main() -> None:
    output = RESULTS / "phase1_configuration_comparison"
    output.mkdir(parents=True, exist_ok=True)
    latency = {}
    with (RESULTS / "phase1_latency_v1" / "latency.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            latency[(int(row["candidates"]), int(row["ode_steps"]))] = row
    summaries = []
    for label, directory, candidates, steps in CONFIGURATIONS:
        with (RESULTS / directory / "evaluation" / "aggregate.json").open(
            encoding="utf-8"
        ) as handle:
            aggregate = {row["method"]: row for row in json.load(handle)}
        planned = aggregate["game_cfm_anchored"]
        executed = aggregate["game_cfm_hocbf"]
        timing = latency[(candidates, steps)]
        summaries.append(
            {
                "configuration": label,
                "candidates": candidates,
                "ode_steps": steps,
                "planned_collision_rate": planned["collision_rate"],
                "planned_margin_violation_rate": planned["margin_violation_rate"],
                "planned_mean_maximum_speed_mps": planned["mean_maximum_speed_mps"],
                "planned_mean_maximum_acceleration_mps2": planned[
                    "mean_maximum_acceleration_mps2"
                ],
                "execution_margin_violation_rate": executed["margin_violation_rate"],
                "execution_mean_minimum_separation_m": executed[
                    "mean_minimum_separation_m"
                ],
                "execution_hocbf_intervention_rate": executed[
                    "mean_hocbf_intervention_rate"
                ],
                "execution_hocbf_mean_correction": executed[
                    "mean_hocbf_correction"
                ],
                "execution_expert_rmse_m": executed["mean_expert_rmse_m"],
                "batch_ms_per_condition": executed["batch_sampling_ms_per_trajectory"],
                "single_query_median_ms": float(timing["median_ms"]),
                "single_query_p99_ms": float(timing["p99_ms"]),
            }
        )
    with (output / "configuration_tradeoff.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / "configuration_tradeoff.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    labels = [row["configuration"].replace(" / ", "\n") for row in summaries]
    x = np.arange(len(summaries))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = (
        ("planned_margin_violation_rate", "Planned rate below 0.36 m margin"),
        ("execution_hocbf_intervention_rate", "Execution HOCBF intervention"),
        ("execution_mean_minimum_separation_m", "Mean minimum separation (m)"),
        ("single_query_median_ms", "Single-query median CPU latency (ms)"),
    )
    for axis, (key, title) in zip(axes.flat, panels, strict=True):
        axis.bar(x, [row[key] for row in summaries])
        axis.set_xticks(x, labels, fontsize=9)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Phase-1 candidate and ODE-step safety/latency trade-off")
    figure.tight_layout()
    figure.savefig(output / "phase1_candidate_step_tradeoff.png", dpi=220)
    plt.close(figure)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
