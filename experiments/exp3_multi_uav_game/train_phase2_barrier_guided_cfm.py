#!/usr/bin/env python3
"""Fine-tune game-conditioned Flow Matching with barrier and dynamics guidance."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safe_game_flow.flow_matching.model import (  # noqa: E402
    FlowMatching1D,
    model_config_from_checkpoint,
)


def _anchor_residual(residual: torch.Tensor) -> torch.Tensor:
    phase = torch.linspace(0.0, 1.0, residual.shape[-1], device=residual.device)
    endpoint_line = (
        (1.0 - phase)[None, None, :] * residual[:, :, :1]
        + phase[None, None, :] * residual[:, :, -1:]
    )
    return residual - endpoint_line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--safe-weight", type=float, default=40.0)
    parser.add_argument("--dynamic-weight", type=float, default=0.02)
    parser.add_argument("--safety-distance", type=float, default=0.38)
    parser.add_argument("--seed", type=int, default=62003)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    data_dir = args.data_dir.resolve()
    checkpoint = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
    residual_physical = torch.from_numpy(np.load(data_dir / "joint_residuals.npy")).float()
    references = torch.from_numpy(np.load(data_dir / "joint_references.npy")).float()
    archive = np.load(data_dir / "conditions.npz")
    conditions_physical = torch.from_numpy(archive["values"]).float()
    archive.close()
    mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["std"], dtype=torch.float32)
    condition_mean = torch.as_tensor(checkpoint["condition_mean"], dtype=torch.float32)
    condition_std = torch.as_tensor(checkpoint["condition_std"], dtype=torch.float32)
    residuals = (residual_physical - mean) / std
    conditions = (conditions_physical - condition_mean) / condition_std
    dataset = TensorDataset(residuals, conditions, references)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = FlowMatching1D(**model_config_from_checkpoint(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = torch.Generator(device=device).manual_seed(args.seed + 1234)
    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    history = []
    horizon = float(conditions_physical[0, -1])
    sample_dt = horizon / (residuals.shape[-1] - 1)
    for epoch in range(args.epochs):
        model.train()
        totals = {"total": 0.0, "fm": 0.0, "safe": 0.0, "dynamic": 0.0}
        batches = 0
        epoch_start = time.perf_counter()
        for x1, condition, reference in loader:
            batch = x1.shape[0]
            x0 = torch.randn(x1.shape, generator=rng)
            t = torch.rand(batch, generator=rng).clamp(1e-3, 1.0 - 1e-3)
            t_bc = t[:, None, None]
            x_t = (1.0 - t_bc) * x0 + t_bc * x1
            target_velocity = x1 - x0
            predicted_velocity = model(x_t, t, condition)
            fm_loss = F.mse_loss(predicted_velocity, target_velocity)
            predicted_terminal = x_t + (1.0 - t_bc) * predicted_velocity
            residual_prediction = predicted_terminal * std + mean
            residual_prediction = _anchor_residual(residual_prediction)
            plan = (reference + residual_prediction).reshape(batch, 2, 3, -1)
            separation = torch.linalg.vector_norm(plan[:, 0] - plan[:, 1], dim=1)
            safety_violation = F.relu(args.safety_distance - separation)
            safe_loss = (
                safety_violation.max(dim=1).values.square() * t.square()
            ).mean()
            velocity = torch.diff(plan, dim=-1) / sample_dt
            acceleration = torch.diff(velocity, dim=-1) / sample_dt
            acceleration_norm = torch.linalg.vector_norm(acceleration, dim=2)
            dynamic_violation = F.relu(acceleration_norm - 3.0)
            dynamic_loss = (
                dynamic_violation.flatten(1).max(dim=1).values.square() * t.square()
            ).mean()
            loss = (
                fm_loss
                + args.safe_weight * safe_loss
                + args.dynamic_weight * dynamic_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["total"] += float(loss.detach())
            totals["fm"] += float(fm_loss.detach())
            totals["safe"] += float(safe_loss.detach())
            totals["dynamic"] += float(dynamic_loss.detach())
            batches += 1
        record = {
            "epoch": epoch + 1,
            "total_loss": totals["total"] / batches,
            "fm_loss": totals["fm"] / batches,
            "safe_loss": totals["safe"] / batches,
            "dynamic_loss": totals["dynamic"] / batches,
            "duration_seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | total {record['total_loss']:.5f} | "
            f"fm {record['fm_loss']:.5f} | safe {record['safe_loss']:.6f} | "
            f"dynamic {record['dynamic_loss']:.5f}",
            flush=True,
        )
        torch.save(
            {
                **checkpoint,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "epoch": epoch + 1,
                "training_history": history,
                "trajectory_representation": "joint_game_barrier_guided_residual",
                "guidance_config": {
                    "safe_weight": args.safe_weight,
                    "dynamic_weight": args.dynamic_weight,
                    "safety_distance": args.safety_distance,
                    "terminal_estimator": "x_t + (1-t) v_theta",
                    "endpoint_anchoring": True,
                },
            },
            save_dir / "checkpoint.pt",
        )
        (save_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    (save_dir / "training_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "data_dir": str(data_dir),
                "resume": str(args.resume.resolve()),
                "save_dir": str(save_dir),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
