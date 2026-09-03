# Experiment 3 Phase 0 — Non-cooperative two-UAV ring access

## Scope and protocol

Phase 0 isolates the two components needed before game-conditioned Flow
Matching: a pure-strategy go/yield Nash scheduler and a relative-degree-two
inter-UAV HOCBF. It uses paired 3-D double-integrator dynamics, a 0.01 s
integration step, constant per-flight acceleration disturbances, 128 scenarios,
and three seeds (384 paired runs per method). This is a game/safety validation,
not the final learned-policy or 6-DoF experiment.

The Nash yielding delay is not fixed. For every scene it is the minimum delay
that places the yielding vehicle behind its opponent by the 0.45 s safety gap
plus a 0.05 s buffer. `fixed_priority` retains a fixed 0.8 s delay and always
prioritizes UAV 0.

## Aggregate results

| Method | Both pass | Collision | Mean min. separation (m) | Worst separation (m) | HOCBF intervention | Completion (s) | Social payoff |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent | 100.0% | 90.9% | 0.116 | 0.009 | 0.0% | 2.098 | -34.989 |
| Fixed priority | 100.0% | 6.0% | 0.464 | 0.083 | 0.0% | 2.801 | 11.330 |
| HOCBF only | 100.0% | 0.0% | 0.367 | 0.340 | 36.7% | 2.205 | 14.993 |
| Nash game | 100.0% | 0.8% | 0.304 | 0.242 | 0.0% | 2.411 | 14.413 |
| Nash game + HOCBF | 100.0% | 0.0% | 0.410 | 0.364 | 13.3% | 2.490 | 14.830 |

`Both pass` alone is intentionally not treated as the main result because all
controllers eventually cross the ring plane. The differentiating quantities
are collision, robust safety margin, safety-filter burden, and response time.

## Paired comparison with reactive HOCBF

For every metric below, the difference is `Nash game + HOCBF` minus `HOCBF
only`, using the same 384 initial states and disturbances. Confidence intervals
are paired nonparametric bootstrap 95% intervals with 20,000 resamples.

| Metric | Paired mean difference | 95% CI | Interpretation |
|---|---:|---:|---|
| Minimum separation | +0.0428 m | [0.0406, 0.0450] | Larger robust clearance |
| HOCBF intervention rate | -0.2343 | [-0.2440, -0.2241] | 63.8% relative reduction |
| Rate below 0.36 m margin | -0.2708 | [-0.3151, -0.2266] | 27.1% to 0.0% |
| Completion time | +0.2850 s | [0.2693, 0.3001] | Explicit coordination cost |
| Social payoff | -0.1628 | [-0.1699, -0.1555] | Small cost from extra delay |
| Maximum acceleration | +0.0051 m/s² | [0.0006, 0.0094] | Negligible practical change |

## Supported conclusion

Reactive HOCBF is the fastest safe baseline in this phase, so the result does
not justify claiming that the game layer is universally faster. The supported
claim is narrower: game-level temporal coordination moves conflict resolution
upstream, significantly reducing online HOCBF activity and preventing the
disturbed trajectories from dropping below the 0.36 m design margin, at an
average completion-time cost of 0.285 s. Neither the Nash scheduler alone nor
the HOCBF alone achieves all three properties simultaneously.

## Next phase

Phase 1 will use the selected game action, scene-dependent delay, opponent
state, and ring condition as conditioning variables for a joint two-agent Flow
Matching model. The Nash+HOCBF trajectories form safe expert labels, while the
reactive HOCBF remains at execution time as a residual safety layer. The next
comparison must therefore test whether distillation preserves the Phase-0
safety-margin benefit while reducing online planning and game-solver latency.

# Experiment 3 Phase 1 — Game-conditioned joint Flow Matching

## Data and model

The Phase-1 training set contains 1,024 Nash+HOCBF expert rollouts and the test
set contains 256 independently seeded rollouts. Every rollout uses a 0.01 s
closed-loop step and is resampled to 64 joint position points. The learned
tensor has six channels (three coordinates for each UAV). Its 24-dimensional
condition contains both initial states, both goals, ring center, nominal
durations and crossing times, the two game actions, selected delays, and the
planning horizon.

The model learns the joint closed-loop correction residual around the Nash
reference. Training loss decreases from 1.965 to 0.323 over 100 CPU epochs.
The fast proposed configuration samples four residual candidates with eight
Heun steps, anchors their two endpoints, and selects a candidate using only
deployable clearance, acceleration and residual-magnitude terms. The held-out
expert trajectory is never used by the selector.

## Fast best-of-four result

| Method | Collision | Below 0.36 m | Mean minimum distance | HOCBF intervention | Mean correction | Expert RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Nash reference plan | 0.0% | 99.6% | 0.304 m | — | — | 0.0185 m |
| Unfiltered selected CFM plan | 0.0% | 51.6% | 0.359 m | — | — | 0.0186 m |
| Nash reference + HOCBF | 0.0% | 0.78% | 0.408 m | 12.04% | 0.637 | 0.0016 m |
| Selected CFM + HOCBF | 0.0% | 0.39% | 0.424 m | 9.68% | 0.490 | 0.0106 m |

Relative to Nash-reference HOCBF execution, selected CFM+HOCBF increases the
paired minimum separation by 0.0168 m (95% bootstrap CI 0.0152–0.0183 m),
reduces HOCBF intervention by 0.0237 (CI 0.0210–0.0264), and reduces mean
correction magnitude by 0.147 (CI 0.128–0.167). Its expert RMSE increases by
0.0090 m and maximum acceleration increases by 0.038 m/s², so the learned
method does not dominate the teacher in tracking fidelity.

## Candidate/ODE-step trade-off

| Configuration | Planned collision | Planned margin violation | Execution intervention | Mean separation | Median CPU latency | P99 latency |
|---|---:|---:|---:|---:|---:|---:|
| 1 candidate, 16 steps | 0.78% | 82.0% | 11.33% | 0.416 m | 63.3 ms | 78.8 ms |
| 4 candidates, 8 steps | 0.0% | 51.6% | 9.68% | 0.424 m | 84.0 ms | 96.0 ms |
| 8 candidates, 16 steps | 0.0% | 28.1% | 8.81% | 0.426 m | 252.4 ms | 308.3 ms |

The four-candidate/eight-step configuration is the practical CPU operating
point: it removes planned physical collisions and materially reduces execution
safety intervention while keeping P99 below 100 ms. The eight-candidate model
is safer before execution but is not suitable for high-rate replanning on the
tested CPU.

## Supported conclusion and limitation

Phase 1 supports the claim that game-conditioned multi-candidate Flow Matching
can move part of the reactive collision-avoidance correction into the generated
joint reference, reducing HOCBF intervention and increasing average clearance.
It does **not** support removing HOCBF: 51.6% of fast selected plans still fall
below the conservative 0.36 m design margin even though none crosses the 0.25 m
physical collision threshold. Phase 2 should therefore introduce a train-time
safety-distillation loss or differentiable pairwise barrier guidance, then move
the comparison to 6-DoF quadrotor dynamics.

# Experiment 3 Phase 2 — Barrier guidance and dual 6-DoF validation

## Training-time barrier guidance

Phase 2 resumes the Phase-1 checkpoint and fine-tunes it for 60 epochs. The
terminal estimate `x_t + (1-t)v_theta` is decoded into a joint trajectory; its
maximum pairwise-distance violation supplies the barrier loss, and its maximum
acceleration violation supplies the dynamics loss. Model, data, normalization,
and test seeds remain unchanged.

| Metric | Phase-1 | Phase-2 | Paired Phase-2 minus Phase-1 |
|---|---:|---:|---:|
| Planned margin violation | 51.6% | 39.5% | -12.1 points, CI [-16.8, -7.8] |
| Planned mean minimum distance | 0.359 m | 0.367 m | +0.0080 m, CI [0.0063, 0.0098] |
| Executed margin violation | 0.39% | 0.0% | -0.39 points, CI [-1.17, 0.0] |
| Executed mean minimum distance | 0.424 m | 0.427 m | +0.0031 m, CI [0.0023, 0.0039] |
| HOCBF intervention | 9.68% | 9.31% | -0.36 points, CI [-0.51, -0.22] |
| Mean HOCBF correction | 0.490 | 0.468 | -0.022, CI [-0.033, -0.011] |

Four-candidate/eight-step CPU latency is 85.3 ms median and 101.8 ms P99,
essentially unchanged because guidance changes training, not inference.

## Dual 6-DoF severe-uncertainty experiment

The high-fidelity experiment uses two rigid-body quadrotor plants with
quaternion attitude dynamics, four first-order motor states, thrust saturation,
drag, sampled feedback, measurement noise, mass and motor-time-constant errors,
1.2 m/s² wind acceleration, and 20 ms command delay. Integration and control
both run at 0.002 s. Eight trajectory clusters and three uncertainty seeds give
24 paired joint flights per method.

The initial 0.36 m and 0.50 m reference margins were insufficient under severe
tracking errors. A common 0.65 m robust reference HOCBF margin is therefore
used for Nash, Phase-1 CFM, and Phase-2 CFM. This margin is a separate
robustness mechanism and is not attributed to Flow Matching.

| Method | Collision | Below 0.36 m | Mean min. distance | Worst distance | Mean RMSE | P95 tilt |
|---|---:|---:|---:|---:|---:|---:|
| Nash reference | 33.3% | 70.8% | 0.303 m | 0.118 m | 0.166 m | 27.6° |
| Teacher expert (0.36 m) | 33.3% | 50.0% | 0.352 m | 0.171 m | 0.166 m | 25.0° |
| Nash + robust HOCBF reference | 0.0% | 16.7% | 0.497 m | 0.326 m | 0.167 m | 27.1° |
| Phase-1 CFM + robust HOCBF | 0.0% | 16.7% | 0.496 m | 0.326 m | 0.167 m | 34.4° |
| Phase-2 CFM + robust HOCBF | 0.0% | 16.7% | 0.495 m | 0.326 m | 0.167 m | 33.8° |

The robust reference layer eliminates physical collisions in all 24 severe
runs, while non-robust Nash and expert references collide in 8/24 runs. Four
robust runs still cross the conservative 0.36 m margin.

Barrier guidance does not show an additional 6-DoF clearance advantage over
Phase 1: the cluster-bootstrap minimum-distance difference is -0.0006 m with CI
[-0.0026, 0.0011], and mean-tilt difference is 0.04° with CI
[-0.61°, 0.62°]. Against the Nash robust reference, Phase-2 requires 2.19° more
mean maximum tilt (CI [1.01°, 3.53°]).

## Supported Phase-2 conclusion

Training-time barrier guidance improves generated joint safety and reduces
reactive HOCBF effort in the translational model without adding inference cost.
Under severe 6-DoF uncertainty, collision elimination is driven by the
tracking-aware robust HOCBF reference margin; the learned residual does not yet
outperform the simpler robust Nash reference and increases attitude demand.
The next method revision should condition on tracking uncertainty or optimize
attitude/thrust demand directly instead of adding more geometric barrier weight.
