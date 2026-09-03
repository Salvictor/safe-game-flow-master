# Experiment 3: Two-UAV Non-Cooperative Ring-Access Game

Phase 0 validates the game labels and inter-UAV safety layer before training a
game-conditioned Flow Matching policy.

The initial comparison contains:

- independent aggressive traversal;
- fixed-priority scheduling;
- HOCBF-only reactive safety;
- pure-Nash go/yield scheduling;
- pure-Nash scheduling plus robust inter-UAV HOCBF.

Run the Phase-0 Monte Carlo experiment:

```bash
python experiments/exp3_multi_uav_game/run_phase0_game.py \
  --output-dir experiments/exp3_multi_uav_game/results/phase0_v1_formal \
  --scenarios 128 --seeds 1001,1002,1003

python experiments/exp3_multi_uav_game/analyze_phase0.py \
  experiments/exp3_multi_uav_game/results/phase0_v1_formal
```

This phase uses double-integrator translational dynamics to isolate game and
HOCBF correctness. It is not yet the final 6-DoF or learned-policy experiment.
The Nash yield duration is scene dependent: it is the minimum delay that puts
the yielding UAV behind the opponent by the requested temporal safety gap.
This separates adaptive game coordination from the fixed-delay priority
baseline.

## Phase 1: game-conditioned joint Flow Matching

Generate safe expert residuals, train the six-channel joint model, and evaluate
the fast best-of-four configuration:

```bash
python experiments/exp3_multi_uav_game/generate_phase1_dataset.py \
  --output-dir experiments/exp3_multi_uav_game/results/phase1_v1/data_train \
  --samples 1024 --seed 31001 --points 64 --dt 0.01

python experiments/exp3_multi_uav_game/train_phase1_game_cfm.py \
  --data-dir experiments/exp3_multi_uav_game/results/phase1_v1/data_train \
  --save-dir experiments/exp3_multi_uav_game/results/phase1_v1/model \
  --epochs 100 --batch-size 64 --hidden 128 --blocks 4 --time-emb-dim 64 --cpu

python experiments/exp3_multi_uav_game/evaluate_phase1_game_cfm.py \
  --data-dir experiments/exp3_multi_uav_game/results/phase1_v1/data_test \
  --checkpoint experiments/exp3_multi_uav_game/results/phase1_v1/model/checkpoint.pt \
  --output-dir experiments/exp3_multi_uav_game/results/phase1_v3_fast_best_of_4/evaluation \
  --ode-steps 8 --candidates 4
```

Phase 1 learns the joint HOCBF correction residual around the scene-dependent
Nash reference. Candidate scoring uses only deployable quantities (pairwise
clearance, acceleration and residual magnitude), never the held-out expert.

## Phase 2: barrier guidance and 6-DoF validation

```bash
python experiments/exp3_multi_uav_game/train_phase2_barrier_guided_cfm.py \
  --data-dir experiments/exp3_multi_uav_game/results/phase1_v1/data_train \
  --resume experiments/exp3_multi_uav_game/results/phase1_v1/model/checkpoint.pt \
  --save-dir experiments/exp3_multi_uav_game/results/phase2_v1_barrier_guided/model \
  --epochs 60 --safe-weight 2000 --dynamic-weight 0.1 --safety-distance 0.38

python experiments/exp3_multi_uav_game/run_phase2_dual_quadrotor.py \
  --data-dir experiments/exp3_multi_uav_game/results/phase1_v1/data_test \
  --phase1-evaluation experiments/exp3_multi_uav_game/results/phase2_robust_margin/phase1_best_of_4_hocbf065 \
  --phase2-evaluation experiments/exp3_multi_uav_game/results/phase2_robust_margin/phase2_best_of_4_hocbf065 \
  --output-dir experiments/exp3_multi_uav_game/results/phase2_dual_6dof_severe_v1 \
  --num-trajectories 8 --seeds 8101,8102,8103 --scenarios severe \
  --integration-dt 0.002 --control-dt 0.002
```

The 6-DoF script records that HOCBF corrects the reference before rigid-body
tracking. It does not claim a new online safety proof for the actual coupled
6-DoF states.
