#!/usr/bin/env python3
"""Cluster-bootstrap paired closed-loop differences against proposed-8-step."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "tracking_rmse_m",
    "maximum_tracking_error_m",
    "crossing_time_error_s",
    "maximum_tilt_degrees",
)
BASELINES = (
    "quintic_skeleton",
    "boundary_cfm_steps8",
    "projected_velocity_steps8",
)
PROPOSED = "proposed_robust_steps8"


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")

    with args.input.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    keyed = {
        (row["scenario"], int(row["seed"]), int(row["trajectory_index"]), row["method"]): row
        for row in source_rows
    }
    scenarios = sorted({row["scenario"] for row in source_rows})
    trajectory_indices = sorted({int(row["trajectory_index"]) for row in source_rows})
    seeds = sorted({int(row["seed"]) for row in source_rows})
    rng = np.random.default_rng(args.seed)
    results = []
    for scenario in scenarios:
        for baseline in BASELINES:
            for metric in METRICS:
                differences_by_cluster = []
                for trajectory_index in trajectory_indices:
                    differences = []
                    for seed in seeds:
                        proposed_value = float(keyed[(scenario, seed, trajectory_index, PROPOSED)][metric])
                        baseline_value = float(keyed[(scenario, seed, trajectory_index, baseline)][metric])
                        if np.isfinite(proposed_value) and np.isfinite(baseline_value):
                            # Positive means the proposed method is better for
                            # these lower-is-better metrics.
                            differences.append(baseline_value - proposed_value)
                    differences_by_cluster.append(np.asarray(differences, dtype=float))
                observed = np.concatenate([item for item in differences_by_cluster if len(item)])
                bootstrap = np.empty(args.bootstrap_samples)
                for sample_index in range(args.bootstrap_samples):
                    selected = rng.integers(0, len(differences_by_cluster), len(differences_by_cluster))
                    values = [differences_by_cluster[index] for index in selected if len(differences_by_cluster[index])]
                    bootstrap[sample_index] = _finite_mean(np.concatenate(values))
                lower, upper = np.percentile(bootstrap, [2.5, 97.5])
                mean_difference = float(np.mean(observed))
                conclusion = (
                    "proposed_better" if lower > 0 else "proposed_worse" if upper < 0 else "inconclusive"
                )
                results.append(
                    {
                        "scenario": scenario,
                        "baseline": baseline,
                        "metric": metric,
                        "paired_runs": len(observed),
                        "baseline_minus_proposed_mean": mean_difference,
                        "ci95_lower": float(lower),
                        "ci95_upper": float(upper),
                        "conclusion": conclusion,
                    }
                )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired_bootstrap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output / "paired_bootstrap.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    counts: dict[str, int] = defaultdict(int)
    for result in results:
        counts[result["conclusion"]] += 1
    print(dict(counts))
    print(f"Paired bootstrap results saved to: {output}")


if __name__ == "__main__":
    main()
