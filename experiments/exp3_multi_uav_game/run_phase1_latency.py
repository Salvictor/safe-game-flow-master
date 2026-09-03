#!/usr/bin/env python3
"""Batch-size-one CPU latency for Phase-1 candidate/ODE configurations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_phase1_game_cfm import _anchor_residual, _candidate_score  # noqa: E402
from safe_game_flow.flow_matching.model import (  # noqa: E402
    FlowMatching1D,
    model_config_from_checkpoint,
)
from safe_game_flow.flow_matching.normalization import denormalize_trajectory  # noqa: E402
from safe_game_flow.flow_matching.train import sample_ode  # noqa: E402


CONFIGURATIONS = ((1, 16), (4, 8), (8, 16))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = args.data_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    references = np.load(data / "joint_references.npy")[: args.queries]
    archive = np.load(data / "conditions.npz")
    conditions = archive["values"][: args.queries]
    archive.close()
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model = FlowMatching1D(**model_config_from_checkpoint(checkpoint))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalized = (conditions - np.asarray(checkpoint["condition_mean"])) / np.asarray(
        checkpoint["condition_std"]
    )
    channels = int(checkpoint["channels"])
    points = int(checkpoint["H"])
    rows = []
    for candidates, ode_steps in CONFIGURATIONS:
        for query in range(args.queries + args.warmup):
            index = query % args.queries
            condition = np.repeat(normalized[index : index + 1], candidates, axis=0)
            generator = torch.Generator(device="cpu").manual_seed(
                70001 + candidates * 100_000 + ode_steps * 1000 + query
            )
            start = time.perf_counter()
            sampled = sample_ode(
                model,
                candidates,
                points,
                ode_steps,
                torch.device("cpu"),
                channels=channels,
                generator=generator,
                condition=torch.from_numpy(condition).float(),
            )
            residuals = denormalize_trajectory(
                sampled, checkpoint["mean"], checkpoint["std"]
            ).numpy()
            anchored = _anchor_residual(residuals)
            scores = [
                _candidate_score(
                    (references[index] + anchored[candidate]).reshape(2, 3, points),
                    anchored[candidate],
                    float(conditions[index, -1]),
                )
                for candidate in range(candidates)
            ]
            _ = int(np.argmin(scores))
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if query >= args.warmup:
                rows.append(
                    {
                        "candidates": candidates,
                        "ode_steps": ode_steps,
                        "query": query - args.warmup,
                        "latency_ms": elapsed_ms,
                    }
                )
    with (output / "all_queries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for candidates, ode_steps in CONFIGURATIONS:
        values = np.asarray(
            [
                row["latency_ms"]
                for row in rows
                if row["candidates"] == candidates and row["ode_steps"] == ode_steps
            ]
        )
        aggregate.append(
            {
                "candidates": candidates,
                "ode_steps": ode_steps,
                "queries": len(values),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.quantile(values, 0.95)),
                "p99_ms": float(np.quantile(values, 0.99)),
                "maximum_ms": float(np.max(values)),
            }
        )
    with (output / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (output / "latency.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
