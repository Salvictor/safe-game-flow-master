#!/usr/bin/env python3
"""Train the 3D ring-conditioned Flow Matching baseline for Experiment 1."""

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

from config import DATA_DIR, RESULTS_DIR  # noqa: E402
from safe_game_flow.flow_matching.train import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--save-dir", type=Path, default=RESULTS_DIR / "conditional_fm")
    parser.add_argument(
        "--representation",
        choices=("boundary_residual", "position", "velocity"),
        default="boundary_residual",
        help="Model boundary-conditioned residuals (proposed) or raw positions (baseline).",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--time-emb-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--preview-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    representation_file = {
        "boundary_residual": "boundary_residuals.npy",
        "position": "trajectories.npy",
        "velocity": "velocities.npy",
    }
    trajectory_path = data_dir / representation_file[args.representation]
    condition_path = data_dir / "conditions.npz"
    for path in (trajectory_path, condition_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run generate_dataset.py before training."
            )

    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "data": str(trajectory_path),
        "conditions": str(condition_path),
        "save_dir": str(save_dir),
        "resume": str(args.resume.resolve()) if args.resume else "",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden": args.hidden,
        "blocks": args.blocks,
        "time_emb_dim": args.time_emb_dim,
        "kernel_size": args.kernel_size,
        "grad_clip": args.grad_clip,
        "amp": args.amp,
        "cpu": args.cpu,
        "seed": args.seed,
        "sample_only": False,
        "num_samples": 8,
        "ode_steps": args.ode_steps,
        "trajectory_representation": args.representation,
        "preview_interval": args.preview_interval,
    }
    (save_dir / "training_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    train(SimpleNamespace(**configuration))


if __name__ == "__main__":
    main()
