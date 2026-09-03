#!/usr/bin/env python3
"""Paired statistical analysis for Experiment 3 Phase-0 results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "minimum_separation_m": "Minimum separation (m)",
    "hocbf_intervention_rate": "HOCBF intervention rate",
    "completion_time_s": "Completion time (s)",
    "social_payoff": "Social payoff",
    "maximum_acceleration_mps2": "Maximum acceleration (m/s^2)",
}


def _bootstrap_mean_ci(
    samples: np.ndarray, *, seed: int = 20260721, repetitions: int = 20_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(samples), size=(repetitions, len(samples)))
    means = np.mean(samples[indices], axis=1)
    return tuple(float(item) for item in np.quantile(means, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--hocbf-distance", type=float, default=0.36)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    with (result_dir / "all_runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed: dict[str, dict[tuple[int, int], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["seed"]), int(row["scenario_index"]))
        keyed[row["method"]][key] = row
    baseline = keyed["hocbf_only"]
    proposed = keyed["nash_game_hocbf"]
    keys = sorted(set(baseline) & set(proposed))
    comparisons = []
    for metric, label in METRICS.items():
        differences = np.asarray(
            [float(proposed[key][metric]) - float(baseline[key][metric]) for key in keys]
        )
        lower, upper = _bootstrap_mean_ci(differences)
        comparisons.append(
            {
                "metric": metric,
                "label": label,
                "paired_runs": len(keys),
                "proposed_minus_hocbf_mean": float(np.mean(differences)),
                "ci95_lower": lower,
                "ci95_upper": upper,
            }
        )
    base_violations = np.asarray(
        [float(baseline[key]["minimum_separation_m"]) < args.hocbf_distance for key in keys],
        dtype=float,
    )
    proposed_violations = np.asarray(
        [float(proposed[key]["minimum_separation_m"]) < args.hocbf_distance for key in keys],
        dtype=float,
    )
    violation_difference = proposed_violations - base_violations
    lower, upper = _bootstrap_mean_ci(violation_difference)
    comparisons.append(
        {
            "metric": "hocbf_margin_violation_rate",
            "label": f"Rate below {args.hocbf_distance:.2f} m HOCBF margin",
            "paired_runs": len(keys),
            "hocbf_only_rate": float(np.mean(base_violations)),
            "proposed_rate": float(np.mean(proposed_violations)),
            "proposed_minus_hocbf_mean": float(np.mean(violation_difference)),
            "ci95_lower": lower,
            "ci95_upper": upper,
        }
    )
    with (result_dir / "paired_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = sorted({field for row in comparisons for field in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    (result_dir / "paired_comparison.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )

    with (result_dir / "aggregate.json").open(encoding="utf-8") as handle:
        aggregate = json.load(handle)
    labels = [row["method"].replace("_", "\n") for row in aggregate]
    x = np.arange(len(aggregate))
    margin_violation_by_method = []
    for method in [row["method"] for row in aggregate]:
        values = [
            float(item["minimum_separation_m"]) < args.hocbf_distance
            for item in keyed[method].values()
        ]
        margin_violation_by_method.append(float(np.mean(values)))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = (
        ([row["collision_rate"] for row in aggregate], "Physical collision rate"),
        (margin_violation_by_method, f"Rate below {args.hocbf_distance:.2f} m margin"),
        ([row["hocbf_intervention_rate"] for row in aggregate], "HOCBF intervention rate"),
        ([row["mean_completion_time_s"] for row in aggregate], "Completion time (s)"),
    )
    for axis, (values, title) in zip(axes.flat, panels, strict=True):
        axis.bar(x, values)
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Experiment 3 Phase 0: safety-efficiency trade-off")
    figure.tight_layout()
    figure.savefig(result_dir / "phase0_safety_efficiency.png", dpi=220)
    plt.close(figure)
    print(f"Paired analysis saved to: {result_dir}")


if __name__ == "__main__":
    main()
