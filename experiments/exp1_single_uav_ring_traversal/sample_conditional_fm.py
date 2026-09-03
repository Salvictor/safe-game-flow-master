#!/usr/bin/env python3
"""Sample and decode boundary-conditioned ring traversal trajectories."""

from __future__ import annotations

import argparse
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

from safe_game_flow.flow_matching.model import (  # noqa: E402
    FlowMatching1D,
    model_config_from_checkpoint,
)
from safe_game_flow.flow_matching.normalization import denormalize_trajectory  # noqa: E402
from safe_game_flow.flow_matching.train import sample_ode  # noqa: E402
from safe_game_flow.evaluation import ring_from_condition  # noqa: E402
from safe_game_flow.safety import (  # noqa: E402
    RingResidualCBFProjector,
    project_boundary_residual_to_feasibility,
)
from safe_game_flow.trajectories import (  # noqa: E402
    decode_boundary_residual,
    integrate_velocity_profile,
    project_velocity_boundary_constraints,
)


def _densify(trajectory: np.ndarray, factor: int = 10) -> np.ndarray:
    source = np.arange(len(trajectory), dtype=float)
    target = np.linspace(0.0, len(trajectory) - 1, (len(trajectory) - 1) * factor + 1)
    return np.stack(
        [np.interp(target, source, trajectory[:, axis]) for axis in range(3)], axis=1
    )


def _sample_safe_passage_prior(
    projector: RingResidualCBFProjector,
    conditions: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
    maximum_attempts: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Reject unsafe Gaussian priors and fall back to the boundary skeleton."""
    if maximum_attempts < 1:
        raise ValueError("prior maximum attempts must be positive")
    batch = len(conditions)
    horizon = projector.horizon
    accepted = torch.zeros(batch, dtype=torch.bool)
    result = torch.empty(batch, 3, horizon, device=device)
    rejected = 0
    for _ in range(maximum_attempts):
        unresolved = torch.nonzero(~accepted, as_tuple=False).flatten()
        if len(unresolved) == 0:
            break
        candidate = torch.randn(
            batch, 3, horizon, device=device, generator=generator, dtype=torch.float32
        )
        physical = projector.decode(candidate).detach().cpu().numpy()
        for index in unresolved.tolist():
            ring = ring_from_condition(conditions[index])
            dense = _densify(physical[index])
            crossing = ring.detect_ring_crossing(dense, direction=1)
            if crossing.passed:
                result[index] = candidate[index]
                accepted[index] = True
            else:
                rejected += 1

    unresolved = torch.nonzero(~accepted, as_tuple=False).flatten()
    fallback_count = len(unresolved)
    if fallback_count:
        base = (-mean / std).view(1, 3, 1).expand(batch, 3, horizon)
        physical_base = projector.decode(base).detach().cpu().numpy()
        for index in unresolved.tolist():
            ring = ring_from_condition(conditions[index])
            if not ring.detect_ring_crossing(_densify(physical_base[index]), direction=1).passed:
                raise RuntimeError(f"Boundary skeleton is not a safe passage for condition {index}")
            result[index] = base[index]
    return result, {
        "rejected_candidates": rejected,
        "fallback_skeletons": fallback_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-offset", type=int, default=0)
    parser.add_argument("--num-conditions", type=int, default=8)
    parser.add_argument("--samples-per-condition", type=int, default=1)
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable-cbf", action="store_true")
    parser.add_argument("--cbf-alpha", type=float, default=5.0)
    parser.add_argument("--unsafe-gaussian-prior", action="store_true")
    parser.add_argument("--prior-max-attempts", type=int, default=20)
    parser.add_argument("--cbf-collocation-factor", type=int, default=2)
    parser.add_argument("--cbf-projection-iterations", type=int, default=2)
    parser.add_argument(
        "--post-step-safety-correction",
        action="store_true",
        help="project every completed Heun state back into the dense ring-safe set",
    )
    parser.add_argument("--kinodynamic-project", action="store_true")
    parser.add_argument("--maximum-speed", type=float, default=2.0)
    parser.add_argument("--maximum-acceleration", type=float, default=3.0)
    parser.add_argument("--project-velocity-boundary", action="store_true")
    args = parser.parse_args()

    if args.num_conditions < 1 or args.samples_per_condition < 1:
        raise ValueError("num-conditions and samples-per-condition must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    representation = checkpoint.get("trajectory_representation", "position")
    if representation not in ("boundary_residual", "position", "velocity"):
        raise ValueError(
            f"Unsupported trajectory_representation={representation!r}"
        )
    if representation in ("position", "velocity") and (
        not args.disable_cbf or not args.unsafe_gaussian_prior or args.kinodynamic_project
    ):
        raise ValueError(
            "Raw position/velocity baselines require --disable-cbf "
            "--unsafe-gaussian-prior and do not support --kinodynamic-project"
        )
    if representation != "velocity" and args.project_velocity_boundary:
        raise ValueError("--project-velocity-boundary requires a velocity checkpoint")
    model_config = model_config_from_checkpoint(checkpoint)
    if int(model_config.get("condition_dim", 0)) <= 0:
        raise ValueError("Checkpoint is not a conditional Flow Matching model")
    if model_config["in_channels"] != 3:
        raise ValueError("Ring traversal sampler expects a three-channel model")
    model = FlowMatching1D(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    loaded = np.load(args.conditions)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("conditions must be an .npz archive with values and schema")
    try:
        raw_conditions = np.asarray(loaded["values"], dtype=np.float32)
        schema = tuple(str(item) for item in loaded["schema"])
    finally:
        loaded.close()
    stop = args.condition_offset + args.num_conditions
    if args.condition_offset < 0 or stop > len(raw_conditions):
        raise ValueError(
            f"Requested conditions [{args.condition_offset}:{stop}], file has {len(raw_conditions)}"
        )
    selected = raw_conditions[args.condition_offset:stop]
    repeated_conditions = np.repeat(selected, args.samples_per_condition, axis=0)

    checkpoint_schema = checkpoint.get("condition_schema")
    if checkpoint_schema is not None and tuple(checkpoint_schema) != schema:
        raise ValueError("Condition schema differs from the schema stored in the checkpoint")
    condition_mean = torch.as_tensor(checkpoint["condition_mean"], dtype=torch.float32)
    condition_std = torch.as_tensor(checkpoint["condition_std"], dtype=torch.float32)
    normalized_conditions = (
        torch.from_numpy(repeated_conditions) - condition_mean.cpu()
    ) / condition_std.cpu()

    mean = torch.as_tensor(checkpoint["mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(checkpoint["std"], device=device, dtype=torch.float32)
    geometry_projector = None
    projector = None
    if representation == "boundary_residual":
        geometry_projector = RingResidualCBFProjector(
            repeated_conditions,
            schema,
            int(checkpoint["H"]),
            mean,
            std,
            alpha=args.cbf_alpha,
            collocation_factor=args.cbf_collocation_factor,
            projection_iterations=args.cbf_projection_iterations,
        )
        projector = None if args.disable_cbf else geometry_projector
    generator = torch.Generator(device=device).manual_seed(args.seed)
    planning_start = time.perf_counter()
    initial_state = None
    prior_summary = {"rejected_candidates": 0, "fallback_skeletons": 0}
    if representation == "boundary_residual" and not args.unsafe_gaussian_prior:
        initial_state, prior_summary = _sample_safe_passage_prior(
            geometry_projector,
            repeated_conditions,
            mean,
            std,
            generator,
            device,
            args.prior_max_attempts,
        )
    sampling_start = time.perf_counter()
    modeled_norm = sample_ode(
        model,
        num_samples=len(repeated_conditions),
        H=int(checkpoint["H"]),
        ode_steps=args.ode_steps,
        device=device,
        generator=generator,
        condition=normalized_conditions.to(device),
        velocity_projector=projector,
        state_projector=(
            geometry_projector.project_state
            if args.post_step_safety_correction and geometry_projector is not None
            else None
        ),
        initial_state=initial_state,
    )
    ode_sampling_seconds = time.perf_counter() - sampling_start
    modeled = denormalize_trajectory(modeled_norm, mean, std).cpu().numpy()
    modeled_hc = np.transpose(modeled, (0, 2, 1))
    kinodynamic_scales = np.ones(len(modeled_hc))
    kinodynamic_strategies: list[str] = []
    kinodynamic_seconds = 0.0
    if representation == "position":
        trajectories_hc = modeled_hc
    elif representation == "velocity":
        if args.project_velocity_boundary:
            modeled_hc = np.stack(
                [
                    project_velocity_boundary_constraints(item, condition, schema)
                    for item, condition in zip(
                        modeled_hc, repeated_conditions, strict=True
                    )
                ]
            )
            modeled = np.transpose(modeled_hc, (0, 2, 1))
        trajectories_hc = np.stack(
            [
                integrate_velocity_profile(item, condition, schema)
                for item, condition in zip(modeled_hc, repeated_conditions, strict=True)
            ]
        )
    elif args.kinodynamic_project:
        projection_start = time.perf_counter()
        projection_results = [
            project_boundary_residual_to_feasibility(
                item,
                condition,
                schema,
                maximum_speed=args.maximum_speed,
                maximum_acceleration=args.maximum_acceleration,
            )
            for item, condition in zip(modeled_hc, repeated_conditions, strict=True)
        ]
        kinodynamic_seconds = time.perf_counter() - projection_start
        modeled_hc = np.stack([item.residual for item in projection_results])
        trajectories_hc = np.stack([item.trajectory for item in projection_results])
        kinodynamic_scales = np.asarray([item.scale for item in projection_results])
        kinodynamic_strategies = [item.strategy for item in projection_results]
        modeled = np.transpose(modeled_hc, (0, 2, 1))
    else:
        trajectories_hc = decode_boundary_residual(
            modeled_hc, repeated_conditions, schema
        )
    end_to_end_seconds = time.perf_counter() - planning_start

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    representation_filename = {
        "boundary_residual": "boundary_residuals.npy",
        "position": "positions.npy",
        "velocity": "velocities.npy",
    }[representation]
    np.save(output_dir / representation_filename, modeled)
    np.save(output_dir / "trajectories.npy", np.transpose(trajectories_hc, (0, 2, 1)))
    np.savez_compressed(
        output_dir / "conditions_used.npz",
        values=repeated_conditions,
        schema=np.asarray(schema),
    )
    sampling_summary = {
        "num_conditions": len(selected),
        "samples_per_condition": args.samples_per_condition,
        "num_trajectories": len(trajectories_hc),
        "ode_steps": args.ode_steps,
        "seed": args.seed,
        "trajectory_representation": representation,
        "ode_sampling_seconds": ode_sampling_seconds,
        "ode_milliseconds_per_trajectory": (
            1000.0 * ode_sampling_seconds / len(repeated_conditions)
        ),
        "end_to_end_planning_seconds": end_to_end_seconds,
        "end_to_end_milliseconds_per_trajectory": (
            1000.0 * end_to_end_seconds / len(repeated_conditions)
        ),
        "cbf_enabled": projector is not None,
        "cbf_alpha": args.cbf_alpha if projector is not None else None,
        "cbf_projection": projector.summary() if projector is not None else None,
        "cbf_collocation_factor": args.cbf_collocation_factor if projector is not None else None,
        "cbf_projection_iterations": (
            args.cbf_projection_iterations if projector is not None else None
        ),
        "safe_passage_prior": not args.unsafe_gaussian_prior,
        "prior_sampling": prior_summary,
        "kinodynamic_projection": args.kinodynamic_project,
        "kinodynamic_projection_seconds": kinodynamic_seconds,
        "kinodynamic_projection_strategies": {
            strategy: kinodynamic_strategies.count(strategy)
            for strategy in sorted(set(kinodynamic_strategies))
        },
        "mean_residual_scale": float(np.mean(kinodynamic_scales)),
        "minimum_residual_scale": float(np.min(kinodynamic_scales)),
        "fraction_scaled": float(np.mean(kinodynamic_scales < 1.0 - 1e-12)),
        "velocity_boundary_projection": args.project_velocity_boundary,
        "post_step_safety_correction": args.post_step_safety_correction,
    }
    (output_dir / "sampling_summary.json").write_text(
        json.dumps(sampling_summary, indent=2), encoding="utf-8"
    )
    print(
        f"Saved {len(trajectories_hc)} decoded trajectories for "
        f"{len(selected)} conditions to: {output_dir}"
    )


if __name__ == "__main__":
    main()
