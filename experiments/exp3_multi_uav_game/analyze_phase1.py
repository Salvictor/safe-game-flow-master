#!/usr/bin/env python3
"""Paired bootstrap analysis for a Phase-1 evaluation directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "minimum_separation_m",
    "margin_violation",
    "hocbf_intervention_rate",
    "mean_hocbf_correction",
    "expert_rmse_m",
    "goal_error_m",
    "maximum_acceleration_mps2",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    args = parser.parse_args()
    evaluation = args.evaluation_dir.resolve()
    with (evaluation / "per_trajectory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["method"]][int(row["sample"])] = row
    baseline = grouped["nash_reference_hocbf"]
    proposed = grouped["game_cfm_hocbf"]
    keys = sorted(set(baseline) & set(proposed))
    rng = np.random.default_rng(20260721)
    results = []
    for metric in METRICS:
        difference = np.asarray(
            [float(proposed[key][metric]) - float(baseline[key][metric]) for key in keys]
        )
        indices = rng.integers(0, len(keys), size=(args.bootstrap, len(keys)))
        means = np.mean(difference[indices], axis=1)
        lower, upper = np.quantile(means, [0.025, 0.975])
        results.append(
            {
                "metric": metric,
                "paired_samples": len(keys),
                "proposed_minus_baseline_mean": float(np.mean(difference)),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
            }
        )
    with (evaluation / "paired_bootstrap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (evaluation / "paired_bootstrap.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Paired Phase-1 analysis saved to: {evaluation}")


if __name__ == "__main__":
    main()
