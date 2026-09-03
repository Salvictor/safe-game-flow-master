# ODE-Step Trajectory/Velocity Planning Sweep — v2

Five planners were evaluated at 1, 2, 4, 8, 16, 32, and 64 ODE steps using
128 conditions and three independent random seeds (105 complete runs).

| Planner | ODE steps | End-to-end ms/traj | Success | Dynamic feasible | Mean max acceleration |
|---|---:|---:|---:|---:|---:|
| Projected velocity | 4 | 1.72 | 59.1% | 59.1% | 2.99 m/s² |
| Projected velocity | 8 | 2.94 | 70.8% | 70.8% | 2.70 m/s² |
| Projected velocity | 16 | 6.31 | 71.9% | 71.9% | 2.65 m/s² |
| Proposed trajectory | 1 | 43.02 | 100.0% | 100.0% | 2.99 m/s² |
| Proposed trajectory | 4 | 51.16 | 100.0% | 100.0% | 2.99 m/s² |
| Proposed trajectory | 8 | 63.36 | 100.0% | 100.0% | 2.97 m/s² |
| Proposed trajectory | 64 | 161.17 | 100.0% | 100.0% | 2.95 m/s² |

The reported time is end-to-end batched throughput per trajectory after model
loading. It includes safe-prior construction, CBF integration, and kinodynamic
projection for the proposed method. It is not yet the batch-size-one hardware
control-loop latency.

The results expose a practical operating choice: projected velocity planning at
8–16 steps is fast but only about 71–72% feasible, whereas the proposed
trajectory method is invariant to the tested ODE step count but costs 43–161 ms.
