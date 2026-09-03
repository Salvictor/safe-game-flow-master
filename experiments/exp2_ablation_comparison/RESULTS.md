# Experiment 2 — Ablation and controlled comparison

## Protocol

All learned boundary ablations use the same frozen checkpoint, 128 held-out
conditions, eight Heun ODE steps, and fixed inference configuration. Components
are added cumulatively. Batch-size-one timing excludes checkpoint loading and
includes all online planning stages. Closed-loop results use 24 task conditions
and three uncertainty seeds, giving 72 paired 6-DoF flights per method under
the severe uncertainty setting from Experiment 1.

## Trajectory-level ablation

| Method | Success | Collision-free | Dynamic feasible | Mean peak acceleration (m/s²) | Batched time (ms/traj) |
|---|---:|---:|---:|---:|---:|
| Quintic skeleton | 100.0% | 100.0% | 100.0% | 0.90 | <0.01 |
| Projected velocity CFM | 65.6% | 100.0% | 65.6% | 2.98 | 3.39 |
| Boundary only | 0.0% | 100.0% | 0.0% | 33.54 | 3.04 |
| + safe prior | 0.8% | 100.0% | 0.8% | 32.22 | 3.82 |
| + CBF | 0.8% | 100.0% | 0.8% | 33.11 | 10.86 |
| + kinodynamic projection | 100.0% | 100.0% | 100.0% | 2.98 | 12.44 |
| + execution margin | 100.0% | 100.0% | 100.0% | 1.99 | 13.29 |

Boundary parameterization guarantees endpoint feasibility but does not control
interior derivatives. Safe-prior and ring-CBF components address geometric
safety, not speed or acceleration. Kinodynamic projection is the component that
changes physical feasibility. Its mean retained residual scale is 0.107; the
execution-margin version retains 0.069, showing that the current learned
residual is frequently too aggressive.

The original run had one post-CBF collision because the CBF constrained a
two-times dense grid while evaluation checked ten-times dense segments. The
improved method uses sparse active-constraint projection, post-Heun state
correction, and the same ten-times collocation density as evaluation. The
fixed-seed collision count is reduced from 1/128 to 0/128. This is a matched
discrete certificate, not a claim about the mathematical continuum between
arbitrary real-valued times.

## Batch-size-one online latency

| Method | Median (ms) | P95 (ms) | P99 (ms) | Success |
|---|---:|---:|---:|---:|
| Quintic skeleton | 0.21 | 0.35 | 0.38 | 100.0% |
| Projected velocity CFM | 21.68 | 27.05 | 27.99 | 68.0% |
| Boundary only | 20.47 | 25.20 | 28.71 | 0.0% |
| + safe prior | 24.95 | 32.45 | 37.02 | 1.6% |
| + CBF | 48.91 | 62.90 | 70.20 | 1.6% |
| + kinodynamic projection | 54.79 | 72.50 | 84.57 | 100.0% |
| + execution margin | 51.77 | 68.94 | 84.25 | 100.0% |

Sparse active-set CBF projection removes the previous long-tail sweep, and the
analytic derivative-constraint intersection removes repeated kinodynamic
search. For the full method, median latency falls from 71.06 to 51.77 ms and
P99 falls from 741.04 to 84.25 ms while retaining 100% feasibility. The
execution margin itself adds no measurable latency penalty in this run.

## Paired 6-DoF closed-loop ablation

| Method | Mean RMSE (m) | P95 RMSE (m) | Crossing-time error (s) | P95 tilt (deg) | Ring pass |
|---|---:|---:|---:|---:|---:|
| Quintic skeleton | 0.162 | 0.210 | 0.084 | 19.9 | 100.0% |
| Projected velocity CFM | 0.164 | 0.210 | 0.130 | 38.7 | 100.0% |
| Boundary only | 0.359 | 0.572 | 0.227 | 70.7 | 88.9% |
| + safe prior | 0.360 | 0.577 | 0.248 | 70.0 | 80.6% |
| + CBF | 0.360 | 0.577 | 0.248 | 70.0 | 80.6% |
| + kinodynamic projection | 0.167 | 0.217 | 0.087 | 32.8 | 100.0% |
| + execution margin | 0.164 | 0.210 | 0.085 | 24.2 | 100.0% |

Relative to `plus_kinodynamic`, the execution-margin version improves paired
mean RMSE by 0.00326 m (95% cluster-bootstrap CI 0.00184–0.00545 m), maximum
tracking error by 0.00998 m (CI 0.00654–0.01510 m), and maximum tilt by 4.93
degrees (CI 4.09–5.79 degrees). Crossing-time improvement is inconclusive.

Relative to projected velocity CFM, the full method improves crossing-time
error by 0.0454 s (CI 0.0250–0.0688 s) and maximum tilt by 4.61 degrees (CI
1.58–7.99 degrees); RMSE is inconclusive and maximum pointwise error is
slightly worse. The task-specific quintic baseline remains better in RMSE and
tilt on this simple obstacle-free task distribution, so the experiment does
not support a global-superiority claim over quintic.

## Supported conclusion

The ablation supports a narrow causal statement: boundary conditioning is
insufficient for executable trajectories; adding geometry-only safety modules
does not repair derivative violations; kinodynamic projection is necessary for
feasibility; reserving an execution margin significantly improves closed-loop
tracking and attitude demand with negligible additional median planning cost.

It also exposes two priorities before Experiment 3: reduce CBF tail latency and
replace the overly easy direct-quintic task distribution with a versioned
ring-passage prior and wider lateral start/goal offsets.
