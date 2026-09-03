# Experiment 1 Baseline Benchmark — v2

This is the first valid benchmark after correcting expert trajectories to a
uniform physical-time grid. It uses 512 training trajectories, 128 independent
test conditions, 100 epochs, and 20 ODE steps on CPU.

| Method | Success | Dynamic feasible | Endpoint feasible | RMSE to expert | ODE ms/traj |
|---|---:|---:|---:|---:|---:|
| Quintic skeleton | 100.0% | 100.0% | 100.0% | 0.1582 | 0.13 |
| Raw trajectory CFM | 0.0% | 0.0% | 0.0% | 0.7307 | 6.34 |
| Boundary trajectory CFM | 0.0% | 0.0% | 100.0% | 0.2556 | 7.34 |
| Raw velocity CFM | 53.9% | 86.7% | 57.8% | 0.1702 | 6.64 |
| Boundary-projected velocity CFM | 63.3% | 63.3% | 100.0% | **0.1564** | 5.89 |
| Boundary CFM + safe prior | 0.0% | 0.0% | 100.0% | 0.2540 | 6.29 |
| Boundary CFM + prior + CBF | 0.0% | 0.0% | 100.0% | 0.2524 | 40.96 |
| Proposed full trajectory method | 100.0% | 100.0% | 100.0% | 0.1583 | 37.77 |

All methods were evaluated without static obstacles, matching the planned
hardware experiment. The analytical quintic baseline remains the fastest and
fully feasible method for this simple geometry. The proposed method's value in
this experiment is constraint reliability across learned samples, not a claim
of global superiority over the analytical special-case solution.
