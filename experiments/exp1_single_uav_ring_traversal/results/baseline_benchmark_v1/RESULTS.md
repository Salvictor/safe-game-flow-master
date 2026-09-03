# Experiment 1 — Controlled Baseline Benchmark (v1)

> **Superseded development result.** A subsequent velocity-planning validation
> found that the two expert polynomial segments were sampled with equal point
> counts rather than a uniform physical-time grid. The v1 dynamic metrics were
> therefore computed using an inconsistent `dt`. The issue is fixed in the data
> generator; v1 is retained for traceability and must not be used as a paper
> result. All formal tables must be regenerated as v2 or later.

## Setup

- Training trajectories: 512
- Independent test conditions: 128
- Training epochs: 100
- Model: 3 residual blocks, 64 hidden channels, 64-dimensional time embedding
- ODE integration steps: 20
- Device: CPU (`torch 2.11.0+cpu`)
- Identical model seed and capacity for raw-position CFM and boundary-residual CFM

## Main results

| Method | Success | Ring pass | Collision free | Dynamic feasible | Endpoint feasible | RMSE to expert | Sampling ms/trajectory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Quintic skeleton | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.1662 | 0.12 |
| Raw CFM | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.7342 | 6.70 |
| Boundary CFM | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0.2470 | 6.05 |
| Boundary CFM + safe prior | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0.2442 | 6.60 |
| Boundary CFM + safe prior + CBF | 0.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0.2436 | 45.41 |
| Proposed full method | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.1652 | 37.95 |

The full method additionally required 3.26 s for kinodynamic projection over
128 trajectories. Its mean retained residual scale was 0.1284, indicating that
the feasibility layer removed most of the learned residual.

## What this experiment demonstrates

1. Boundary-conditioned residual parameterization raises endpoint feasibility
   from 0% to 100% independently of training quality.
2. The safe prior and ring CBF preserve ring traversal and collision avoidance.
3. Kinodynamic projection raises dynamic feasibility from 0% to 100%.
4. The complete pipeline is substantially more reliable than unconstrained CFM.

## What this experiment does not demonstrate

The quintic skeleton also achieves 100% task success and is much faster. The
full method improves trajectory RMSE by only about 0.6% (`0.1662 -> 0.1652`).
Therefore, the obstacle-free single-ring task is insufficient evidence that a
generative planner is necessary or globally superior.

The next discriminative experiment must include obstacle configurations or
other non-convex route choices for which the direct quintic skeleton is unsafe,
while retaining the same endpoint, safety, dynamics, and timing metrics.
