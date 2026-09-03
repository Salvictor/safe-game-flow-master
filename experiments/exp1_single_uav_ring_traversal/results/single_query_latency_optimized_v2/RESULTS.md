# Batch-size-one online latency (optimized CBF fast path)

This benchmark contains 128 held-out task conditions per configuration on CPU
with one Torch thread. Checkpoint/model loading and warm-up are excluded. Each
timed query includes condition normalization, prior construction, ODE
integration, decoding, and method-specific boundary/CBF/kinodynamic
projection. Evaluation is outside the timed interval.

| Method | ODE steps | Median (ms) | P95 (ms) | P99 (ms) | Planning success |
|---|---:|---:|---:|---:|---:|
| Quintic skeleton | — | 0.19 | 0.31 | 0.35 | 100.0% |
| Boundary trajectory CFM | 1 | 3.89 | 5.08 | 5.89 | 0.0% |
| Projected velocity CFM | 4 | 11.45 | 14.04 | 15.67 | 60.2% |
| Projected velocity CFM | 8 | 20.61 | 28.09 | 31.22 | 70.3% |
| Proposed safe trajectory | 1 | 46.93 | 117.27 | 132.68 | 100.0% |
| Proposed safe trajectory | 4 | 62.47 | 313.10 | 535.95 | 100.0% |
| Proposed safe trajectory | 8 | 77.06 | 562.32 | 1070.04 | 100.0% |
| Proposed safe trajectory | 16 | 110.29 | 1130.66 | 2141.39 | 100.0% |

## What the data show

- The proposed one-step configuration is the best current safety/latency
  compromise: every held-out query is feasible, with a 46.93 ms median.
- The strong task-specific quintic baseline remains much faster and fully
  successful in this simple obstacle-free single-ring distribution. The
  proposed method cannot honestly claim global response-time superiority here.
- Projected velocity CFM is faster but its success saturates near 71%; adding
  ODE steps beyond eight provides little benefit.
- The proposed method has heavy tail latency. CBF constraints are rarely
  active, but active-set sweeps and kinodynamic line search dominate the hard
  cases. Median-only reporting would hide this limitation.

The previous `single_query_latency_v2` directory is a pre-optimization run. It
is retained for reproducibility and must not be mixed with this table.
