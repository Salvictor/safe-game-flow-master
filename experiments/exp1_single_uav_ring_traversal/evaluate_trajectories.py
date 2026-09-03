#!/usr/bin/env python3
"""Evaluate generated single-UAV trajectories against ring task conditions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safe_game_flow.evaluation import (  # noqa: E402
    evaluate_ring_trajectory,
    summarize_ring_metrics,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_conditions(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "values" not in loaded or "schema" not in loaded:
                raise ValueError("Condition archive must contain 'values' and 'schema'")
            values = np.asarray(loaded["values"])
            schema = tuple(str(item) for item in loaded["schema"])
        finally:
            loaded.close()
    else:
        raise ValueError("Use conditions.npz so physical fields are identified by schema")
    return values, schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-offset", type=int, default=0)
    parser.add_argument("--maximum-speed", type=float, default=2.0)
    parser.add_argument("--maximum-acceleration", type=float, default=3.0)
    parser.add_argument("--endpoint-position-tolerance", type=float, default=0.15)
    parser.add_argument("--endpoint-velocity-tolerance", type=float, default=0.25)
    parser.add_argument("--dense-factor", type=int, default=10)
    args = parser.parse_args()

    trajectories = np.load(args.trajectories)
    if trajectories.ndim != 3:
        raise ValueError(f"Expected trajectory array with three dimensions, got {trajectories.shape}")
    if trajectories.shape[1] == 3:
        trajectories = np.transpose(trajectories, (0, 2, 1))
    elif trajectories.shape[2] != 3:
        raise ValueError(
            f"Expected layout (N,3,H) or (N,H,3), got {trajectories.shape}"
        )

    conditions, schema = _load_conditions(args.conditions)
    start = args.condition_offset
    stop = start + len(trajectories)
    if start < 0 or stop > len(conditions):
        raise ValueError(
            f"Requested condition rows [{start}:{stop}], but file has {len(conditions)} rows"
        )
    selected_conditions = conditions[start:stop]

    metrics = [
        evaluate_ring_trajectory(
            trajectory,
            condition,
            schema,
            maximum_speed=args.maximum_speed,
            maximum_acceleration=args.maximum_acceleration,
            endpoint_position_tolerance=args.endpoint_position_tolerance,
            endpoint_velocity_tolerance=args.endpoint_velocity_tolerance,
            dense_factor=args.dense_factor,
        )
        for trajectory, condition in zip(trajectories, selected_conditions, strict=True)
    ]
    summary = summarize_ring_metrics(metrics)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )

    rows = [item.to_dict() for item in metrics]
    with (output_dir / "per_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", *rows[0].keys()])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"index": index + start, **row})

    print(json.dumps(_json_safe(summary), indent=2))
    print(f"Evaluation saved to: {output_dir}")


if __name__ == "__main__":
    main()
