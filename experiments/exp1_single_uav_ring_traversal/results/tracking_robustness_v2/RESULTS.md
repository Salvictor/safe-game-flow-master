# Bounded Tracking-Error Robustness — v2

This trajectory-level stress test adds smooth, endpoint-vanishing position
errors with a prescribed maximum amplitude. Each level uses 128 conditions and
three independently planned trajectory sets. It is not a replacement for the
future full quadrotor closed-loop simulator.

All evaluated planners retained 100% ring-passage and collision-free rates for
maximum position errors from 0 to 8 cm. The standard proposed planner was less
robust dynamically because its feasibility projection used nearly all of the
3 m/s² acceleration budget.

| Planner | Max tracking error | Dynamic feasible |
|---|---:|---:|
| Standard proposed, 1 ODE step | 0.5 cm | 46.6% |
| Standard proposed, 1 ODE step | 1.0 cm | 37.8% |
| Robust-margin proposed, 1 step | 0.5 cm | **100.0%** |
| Robust-margin proposed, 1 step | 1.0 cm | **100.0%** |
| Robust-margin proposed, 1 step | 2.0 cm | 95.3% |
| Robust-margin proposed, 8 steps | 0.5 cm | **100.0%** |
| Robust-margin proposed, 8 steps | 1.0 cm | **100.0%** |
| Robust-margin proposed, 8 steps | 2.0 cm | 95.6% |

The robust planner uses planning limits of 1.95 m/s and 2.0 m/s², while
evaluation retains the physical limits of 2.0 m/s and 3.0 m/s². This is a
measured improvement introduced after the initial stress test exposed poor
dynamic robustness; the original degraded results remain in `all_runs.csv`.
