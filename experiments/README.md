# Experimental Programme

The experiments are split by scientific question rather than by figure count.
Only experiments with generated result files are treated as completed.

## Experiment 1 — Single-UAV method validation

Question: can the proposed boundary-conditioned, barrier-projected Flow
Matching planner produce executable ring-traversal trajectories, and what are
its accuracy, robustness, and response-time trade-offs?

The current implementation is in `exp1_single_uav_ring_traversal`. It includes
matched planning baselines, ODE-step sweeps, batch-size-one tail latency,
bounded tracking error, and 6-DoF closed-loop validation. Hardware is reserved
for the final subsection and does not introduce obstacles beyond the ring
frame.

## Experiment 2 — Ablation and controlled comparison

Question: which contribution is responsible for boundary feasibility, safety,
dynamic feasibility, robustness, and latency?

The implemented controlled comparison in `exp2_ablation_comparison` reuses the
same checkpoint, test scenarios, and seeds while adding one component at a
time: boundary parameterization, safe prior, CBF projection, kinodynamic
projection, and execution margin. It includes trajectory metrics,
batch-size-one tail latency, paired 6-DoF closed-loop evaluation, and clustered
bootstrap intervals. Strong quintic and velocity-CFM baselines remain in the
table even when they outperform the proposed method on a metric.

## Experiment 3 — Multi-UAV non-cooperative game

Question: after the single-agent planner is validated, can game-conditioned
Flow Matching model strategic interaction without losing safety or online
responsiveness?

This stage will introduce joint multi-UAV state conditions, game roles/payoffs,
inter-UAV CBF constraints, and equilibrium/expert-policy distillation. It must
not be used to retroactively tune Experiment 1 test results.
