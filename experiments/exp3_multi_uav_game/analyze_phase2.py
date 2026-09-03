#!/usr/bin/env python3
"""Paired Phase-2 guidance and severe 6-DoF bootstrap analysis."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS = EXPERIMENT_DIR / "results"


def _read_method(path: Path, method: str) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["sample"]): row
            for row in csv.DictReader(handle)
            if row["method"] == method
        }


def _paired_bootstrap(
    difference: np.ndarray, *, seed: int, repetitions: int = 20_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(repetitions, len(difference)))
    return tuple(float(value) for value in np.quantile(np.mean(difference[indices], axis=1), [0.025, 0.975]))


def main() -> None:
    output = RESULTS / "phase2_analysis"
    output.mkdir(parents=True, exist_ok=True)
    phase1_file = RESULTS / "phase1_v3_fast_best_of_4" / "evaluation" / "per_trajectory.csv"
    phase2_file = RESULTS / "phase2_v1_barrier_guided" / "evaluation_best_of_4" / "per_trajectory.csv"
    guidance_results = []
    for method in ("game_cfm_anchored", "game_cfm_hocbf"):
        phase1 = _read_method(phase1_file, method)
        phase2 = _read_method(phase2_file, method)
        keys = sorted(set(phase1) & set(phase2))
        for metric in (
            "collision",
            "margin_violation",
            "minimum_separation_m",
            "hocbf_intervention_rate",
            "mean_hocbf_correction",
            "expert_rmse_m",
            "maximum_acceleration_mps2",
        ):
            difference = np.asarray(
                [float(phase2[key][metric]) - float(phase1[key][metric]) for key in keys]
            )
            lower, upper = _paired_bootstrap(
                difference, seed=20260721 + len(guidance_results)
            )
            guidance_results.append(
                {
                    "method": method,
                    "metric": metric,
                    "samples": len(keys),
                    "phase2_minus_phase1_mean": float(np.mean(difference)),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    with (output / "guidance_paired_bootstrap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(guidance_results[0]))
        writer.writeheader()
        writer.writerows(guidance_results)

    six_dof_path = RESULTS / "phase2_dual_6dof_severe_v1" / "all_runs.csv"
    with six_dof_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[tuple[int, int], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["method"]][(int(row["trajectory_index"]), int(row["seed"]))] = row
    six_dof_results = []
    for baseline_name in ("nash_reference_hocbf", "phase1_cfm_hocbf"):
        baseline = grouped[baseline_name]
        proposed = grouped["phase2_guided_cfm_hocbf"]
        trajectory_indices = sorted({key[0] for key in baseline} & {key[0] for key in proposed})
        for metric in (
            "collision",
            "margin_violation",
            "minimum_separation_m",
            "tracking_rmse_mean_m",
            "maximum_tracking_error_m",
            "maximum_tilt_degrees",
        ):
            # Average the paired seed differences within each trajectory, then
            # bootstrap trajectory clusters rather than treating repeats as independent.
            cluster_difference = []
            for trajectory_index in trajectory_indices:
                keys = sorted(
                    key
                    for key in set(baseline) & set(proposed)
                    if key[0] == trajectory_index
                )
                cluster_difference.append(
                    np.mean(
                        [
                            float(proposed[key][metric]) - float(baseline[key][metric])
                            for key in keys
                        ]
                    )
                )
            difference = np.asarray(cluster_difference)
            lower, upper = _paired_bootstrap(
                difference, seed=20261721 + len(six_dof_results)
            )
            six_dof_results.append(
                {
                    "baseline": baseline_name,
                    "metric": metric,
                    "trajectory_clusters": len(difference),
                    "phase2_minus_baseline_mean": float(np.mean(difference)),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    with (output / "six_dof_cluster_bootstrap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(six_dof_results[0]))
        writer.writeheader()
        writer.writerows(six_dof_results)
    (output / "analysis.json").write_text(
        json.dumps(
            {"guidance": guidance_results, "six_dof": six_dof_results}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Phase-2 analysis saved to: {output}")


if __name__ == "__main__":
    main()
