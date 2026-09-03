# 6-DoF closed-loop quadrotor robustness

Each method was evaluated on 24 held-out task conditions and three random
seeds: 72 closed-loop flights per method and uncertainty level. Comparisons are
paired: for a fixed `(scenario, seed, trajectory)` every planner receives the
same wind direction, mass deviation, motor-lag scale, sensing noise, and
reference delay.

The plant includes rigid-body position/attitude dynamics, angular rates, four
first-order motor thrust states, drag, thrust saturation, sampled geometric
feedback, sensor noise, and offboard reference delay. RK4 integration and the
feedback controller both use 4 ms steps.

## Main closed-loop results

| Scenario | Method | Mean RMSE (m) | P95 RMSE (m) | P95 max error (m) | Mean crossing-time error (s) | P95 tilt (deg) | Ring pass |
|---|---|---:|---:|---:|---:|---:|---:|
| Nominal | Quintic skeleton | 0.049 | 0.064 | 0.096 | 0.050 | 9.8 | 100% |
| Nominal | Boundary CFM | 0.279 | 0.485 | 1.158 | 0.206 | 68.8 | 91.7% |
| Nominal | Projected velocity CFM | 0.043 | 0.056 | 0.087 | 0.057 | 30.8 | 100% |
| Nominal | Proposed, 1 step | 0.053 | 0.063 | 0.120 | 0.053 | 26.7 | 100% |
| Nominal | Proposed, 8 steps | 0.050 | 0.064 | 0.097 | 0.051 | 18.3 | 100% |
| Moderate | Quintic skeleton | 0.087 | 0.130 | 0.165 | 0.057 | 12.3 | 100% |
| Moderate | Boundary CFM | 0.291 | 0.541 | 1.205 | 0.207 | 68.8 | 84.7% |
| Moderate | Projected velocity CFM | 0.085 | 0.126 | 0.152 | 0.070 | 30.0 | 100% |
| Moderate | Proposed, 1 step | 0.090 | 0.136 | 0.181 | 0.059 | 26.6 | 100% |
| Moderate | Proposed, 8 steps | 0.088 | 0.132 | 0.174 | 0.058 | 21.1 | 100% |
| Severe | Quintic skeleton | 0.163 | 0.214 | 0.271 | 0.089 | 21.9 | 100% |
| Severe | Boundary CFM | 0.348 | 0.577 | 1.377 | 0.235 | 71.8 | 80.6% |
| Severe | Projected velocity CFM | 0.165 | 0.217 | 0.282 | 0.124 | 40.6 | 100% |
| Severe | Proposed, 1 step | 0.167 | 0.220 | 0.294 | 0.090 | 38.3 | 100% |
| Severe | Proposed, 8 steps | 0.165 | 0.215 | 0.275 | 0.090 | 31.6 | 100% |

## Interpretation

- Boundary CFM without execution-aware projection produces high-curvature
  references: it has the worst RMSE, large tilt, nonzero motor saturation, and
  loses ring passages. This verifies that trajectory-level endpoints alone are
  insufficient for physical execution.
- Against projected velocity CFM under severe uncertainty, proposed-8-step
  reduces mean crossing-time error from 0.124 s to 0.090 s and P95 tilt from
  40.6 degrees to 31.6 degrees, with similar tracking RMSE.
- The direct quintic skeleton is still the strongest overall baseline on this
  obstacle-free single-ring distribution. The proposed method matches its
  ring-passing reliability but does not improve its RMSE, tilt, or response
  time. This is a genuine negative result, not a plotting error.
- Severe endpoint feasibility is low for all methods (19.4%–23.6% for the
  dynamically reasonable planners) because the evaluation also requires a
  tight terminal-velocity match at the exact flight-time boundary. Ring
  passage remains 100%; hardware experiments should include a post-passage
  settling window rather than reinterpret this metric as collision failure.

A 10,000-sample cluster bootstrap (resampling the 24 task conditions and
keeping the three seeds within each cluster) confirms the limited claim against
projected velocity CFM in the severe scenario. The paired mean crossing-time
improvement is 0.0340 s (95% CI 0.0234–0.0452 s) and the paired maximum-tilt
improvement is 4.78 degrees (95% CI 1.89–7.80 degrees). RMSE difference is
inconclusive: 0.00054 m (95% CI -0.00154–0.00266 m). Comparisons against the
quintic baseline are mostly in favor of quintic; no global superiority claim
is supported by this dataset.

## Scope of the claim

Mass, motor-centre radius, and maximum per-motor thrust use public Crazyflie
2.1 Brushless specifications. Inertia, motor lag, drag, and yaw moment ratio
are declared simulation assumptions pending hardware identification. These
results validate robustness to a physically structured closed loop; they are
not yet evidence of high-fidelity hardware-model agreement.
