#!/usr/bin/env python3
"""Train joint game-conditioned Flow Matching on safe expert residuals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safe_game_flow.flow_matching.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--time-emb-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42003)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "data": str(data_dir / "joint_residuals.npy"),
        "conditions": str(data_dir / "conditions.npz"),
        "save_dir": str(save_dir),
        "resume": "",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": 0,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "hidden": args.hidden,
        "blocks": args.blocks,
        "time_emb_dim": args.time_emb_dim,
        "kernel_size": 3,
        "grad_clip": 1.0,
        "amp": False,
        "cpu": args.cpu,
        "seed": args.seed,
        "sample_only": False,
        "num_samples": 8,
        "ode_steps": 16,
        "trajectory_representation": "joint_game_safe_residual",
        "preview_interval": 0,
    }
    for path in (Path(configuration["data"]), Path(configuration["conditions"])):
        if not path.is_file():
            raise FileNotFoundError(path)
    (save_dir / "training_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    train(SimpleNamespace(**configuration))


if __name__ == "__main__":
    main()
