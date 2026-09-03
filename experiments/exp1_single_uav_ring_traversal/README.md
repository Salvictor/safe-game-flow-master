# Experiment 1: Single-UAV Safe Ring Traversal

This experiment replaces the first vehicle prototype with a three-dimensional
single-UAV ring traversal task. It currently includes expert-data generation,
23-dimensional task-conditioned Flow Matching, a structural boundary-residual
trajectory representation, conditional sampling, ring-aware CBF projection,
kinodynamic projection, online latency measurement, bounded-error robustness,
and a common six-degree-of-freedom closed-loop quadrotor evaluation.

Generate a small smoke-test dataset:

```bash
python experiments/exp1_single_uav_ring_traversal/generate_dataset.py \
  --num-trajectories 20
```

The saved trajectory layout is `(N, 3, H)` in metres. Each trajectory is made
of two quintic segments joined at the ring plane with continuous position,
velocity and acceleration. `boundary_residuals.npy` is the proposed learned
representation: a quintic boundary-feasible skeleton plus a residual whose
multiplier and first two derivatives vanish at both endpoints.

Train the boundary-conditioned conditional Flow Matching model:

```bash
python experiments/exp1_single_uav_ring_traversal/train_conditional_fm.py \
  --data-dir experiments/exp1_single_uav_ring_traversal/data \
  --epochs 200
```

Use `--representation position` for the raw-position baseline.

Sample and decode physical trajectories:

```bash
python experiments/exp1_single_uav_ring_traversal/sample_conditional_fm.py \
  --checkpoint experiments/exp1_single_uav_ring_traversal/results/conditional_fm/checkpoint.pt \
  --conditions experiments/exp1_single_uav_ring_traversal/data/conditions.npz \
  --output-dir experiments/exp1_single_uav_ring_traversal/results/generated
```

Sampling uses a ring-passage-safe prior and a ring-frame CBF velocity
projection by default. Use `--disable-cbf` for the matched ablation, or
`--unsafe-gaussian-prior` to reproduce the unconstrained Gaussian-prior
baseline. Projection intervention statistics are written to
`sampling_summary.json`.

Evaluate task success, safety, dynamics, and endpoint feasibility:

```bash
python experiments/exp1_single_uav_ring_traversal/evaluate_trajectories.py \
  --trajectories experiments/exp1_single_uav_ring_traversal/results/generated/trajectories.npy \
  --conditions experiments/exp1_single_uav_ring_traversal/results/generated/conditions_used.npz \
  --output-dir experiments/exp1_single_uav_ring_traversal/results/evaluation
```

Run the complete controlled baseline benchmark:

```bash
python experiments/exp1_single_uav_ring_traversal/run_baseline_benchmark.py \
  --train-size 512 --test-size 128 --epochs 100 --cpu
```

This compares a quintic skeleton, raw conditional FM, boundary-conditioned FM,
direct velocity FM, boundary-projected velocity FM, safe-prior and CBF
ablations, and the complete method. It writes
`comparison.csv`, `comparison.json`, `comparison_rates.png`, per-trajectory
CSV files, checkpoints, and environment metadata under the selected work
directory.

Run the ODE-step response-time and quality sweep:

```bash
python experiments/exp1_single_uav_ring_traversal/run_ode_step_sweep.py \
  --benchmark-dir experiments/exp1_single_uav_ring_traversal/results/baseline_benchmark_v2 \
  --output-dir experiments/exp1_single_uav_ring_traversal/results/ode_step_sweep_v2 \
  --cpu
```

Run the bounded tracking-error stress test:

```bash
python experiments/exp1_single_uav_ring_traversal/run_tracking_robustness.py \
  --benchmark-dir experiments/exp1_single_uav_ring_traversal/results/baseline_benchmark_v2 \
  --sweep-dir experiments/exp1_single_uav_ring_traversal/results/ode_step_sweep_v2 \
  --output-dir experiments/exp1_single_uav_ring_traversal/results/tracking_robustness_v2
```

Measure true online batch-size-one latency (model loading excluded, all online
planning/projection stages included):

```bash
python experiments/exp1_single_uav_ring_traversal/run_single_query_latency.py \
  --benchmark-dir experiments/exp1_single_uav_ring_traversal/results/baseline_benchmark_v2 \
  --output-dir experiments/exp1_single_uav_ring_traversal/results/single_query_latency_optimized_v2 \
  --steps 1,4,8,16 --queries 128 --warmup 8 --cpu
```

Run the 6-DoF rigid-body, motor-lag, sampled-controller robustness benchmark:

```bash
python experiments/exp1_single_uav_ring_traversal/run_closed_loop_quadrotor.py \
  --benchmark-dir experiments/exp1_single_uav_ring_traversal/results/baseline_benchmark_v2 \
  --ode-sweep-dir experiments/exp1_single_uav_ring_traversal/results/ode_step_sweep_v2 \
  --robustness-dir experiments/exp1_single_uav_ring_traversal/results/tracking_robustness_v2 \
  --output-dir experiments/exp1_single_uav_ring_traversal/results/closed_loop_quadrotor_v1 \
  --num-trajectories 24 --seeds 701,702,703 \
  --integration-dt 0.004 --control-dt 0.004
```

The closed-loop simulation uses public Crazyflie 2.1 Brushless values for
mass, motor-centre radius, and maximum single-motor thrust. Inertia, motor time
constant, drag, and yaw moment ratio are explicitly marked as assumptions in
`simulation_parameters.json`; they must be identified before hardware results
are used to validate absolute model fidelity.
