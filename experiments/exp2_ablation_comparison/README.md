# Experiment 2: Component Ablation and Controlled Comparison

This experiment holds the test conditions, trained checkpoint, random seed,
ODE solver, and evaluation thresholds fixed while changing one inference-time
component at a time.

The ordered ablation is:

1. `boundary_only`: boundary-conditioned representation with a Gaussian prior;
2. `plus_safe_prior`: passage-safe prior, no CBF;
3. `plus_cbf`: passage-safe prior and ring CBF;
4. `plus_kinodynamic`: adds speed/acceleration feasibility projection;
5. `plus_execution_margin`: tightens planning limits to leave closed-loop margin.

`quintic_skeleton` and `projected_velocity_cfm` are retained as strong
non-ablation baselines.

Run planning ablations:

```bash
python experiments/exp2_ablation_comparison/run_planning_ablation.py \
  --benchmark-dir experiments/exp1_single_uav_ring_traversal/results/baseline_benchmark_v2 \
  --output-dir experiments/exp2_ablation_comparison/results/planning_v3_discrete_safe \
  --cpu
```

The experiment reuses a frozen checkpoint; it does not retrain a different
network for each safety component.

Run batch-size-one latency ablations:

```bash
python experiments/exp2_ablation_comparison/run_ablation_latency.py \
  --benchmark-dir experiments/exp1_single_uav_ring_traversal/results/baseline_benchmark_v2 \
  --output-dir experiments/exp2_ablation_comparison/results/latency_v2_improved \
  --queries 128 --warmup 8 --ode-steps 8 --cpu
```

Run paired 6-DoF closed-loop ablations under severe uncertainty:

```bash
python experiments/exp2_ablation_comparison/run_closed_loop_ablation.py \
  --planning-dir experiments/exp2_ablation_comparison/results/planning_v3_discrete_safe \
  --output-dir experiments/exp2_ablation_comparison/results/closed_loop_v2_improved \
  --num-trajectories 24 --seeds 901,902,903 \
  --integration-dt 0.004 --control-dt 0.004 \
  --bootstrap-samples 5000
```

Formal results and their limitations are summarized in `RESULTS.md`.
