# Experiment 3: Statistical Comparison of Safe Multi-Agent Methods

## 📋 Overview

This experiment performs a statistical comparison of three trajectory generation methods for multi-agent intersection scenarios through **100 randomized trials**.

## 🎯 Methods Compared

### 1. Pure Flow Matching (Baseline)
- **Description**: Direct application of learned velocity fields without safety constraints
- **Purpose**: Demonstrate the necessity of safety constraints
- **Expected**: High collision rate, best distribution preservation

### 2. Non-Game-Theoretic CBF (Independent Optimization)
- **Description**: Each agent independently solves its own QP, treating the other agent as a moving obstacle
- **Key Feature**: Independent decision-making without coordination
- **Expected**: Moderate collision rate, moderate distribution preservation

### 3. Game-Theoretic CBF (Joint Optimization) - **Ours** ⭐
- **Description**: Joint QP optimization with game-theoretic constraints
- **Key Feature**: Coordinated decision-making considering both agents' corrections
- **Expected**: Low collision rate, good distribution preservation

## 📊 Evaluation Metrics

### Safety Metrics 🛡️
- **Collision Rate (%)**: Percentage of experiments with min distance < d_safe
- **Minimum Distance Statistics**: Mean, std, min values across all experiments
- **Violation Severity**: How much the safety distance is violated when collisions occur

### Distribution Preservation Metrics 📈
- **Wasserstein Distance**: Distance between generated and training endpoint distributions
- **Total Correction Magnitude**: Sum of all CBF corrections (lower = closer to learned behavior)
- **Trajectory Features**: Path length, curvature, etc.

### Efficiency Metrics ⚡
- **Computation Time**: Mean time per step (milliseconds)
- **Solver Success Rate**: Percentage of successful QP solves

## 🚀 How to Run

### Step 1: Run Experiments
```bash
cd /home/karl/safe-game-flow/experiments/exp3_statistical_comparison
python run_experiments.py
```

This will:
- Run 100 randomized experiments for each method
- Save raw results to `results/all_results.pkl`
- Save summary statistics to `results/summary_stats.txt`
- Print progress and final statistics

**Expected runtime**: ~10-30 minutes (depends on OSQP solver performance)

### Step 2: Visualize Results
```bash
python visualize_results.py
```

This will generate `results/comparison_results.png` with 9 subplots:
1. Collision rate comparison (bar chart)
2. Minimum distance distribution (box plot)
3. Computation time (bar chart with error bars)
4. Total correction magnitude (box plot)
5. Wasserstein distance (bar chart)
6. Safety vs Distribution trade-off (scatter plot)
7-9. Example trajectories for each method

## 📁 Output Files

```
results/
├── all_results.pkl           # Raw results (Python pickle format)
├── summary_stats.txt          # Human-readable summary statistics
└── comparison_results.png     # Main visualization (9 subplots)
```

## 🔍 Key Implementation Details

### Random Sampling
Start and goal positions are randomly sampled along road centerlines:
- **Agent 1 (West→East)**: 
  - Start X ∈ [-40, -20], Y ∈ [-3, 3]
  - Goal X ∈ [20, 40], Y ∈ [-3, 3]
- **Agent 2 (South→North)**:
  - Start X ∈ [-3, 3], Y ∈ [-40, -20]
  - Goal X ∈ [-3, 3], Y ∈ [20, 40]

### Safety Method Interface
```python
class SafetyMethod:
    def compute_correction(self, traj_1, traj_2, v1, v2, t, obstacles, d_safe):
        """
        Returns:
            u1, u2: (H, 2) correction velocities
            solver_status: bool indicating success
        """
```

### QP Formulations

**Non-Game CBF** (two independent QPs):
```
Agent 1:  min ||u1||^2  s.t. constraints_1
Agent 2:  min ||u2||^2  s.t. constraints_2
```

**Game-Theoretic CBF** (one joint QP):
```
min ||u1||^2 + ||u2||^2  
s.t. 
  - a1^T u1 >= b1  (Agent 1 obstacles)
  - a2^T u2 >= b2  (Agent 2 obstacles)
  - a^T u1 - a^T u2 >= b12  (inter-agent safety)
```

## 📈 Expected Results

| Method | Collision Rate | Distribution | Comp Time |
|--------|----------------|-------------|-----------|
| Pure FM | ~40-60% | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐⭐⭐ Fastest |
| Non-Game CBF | ~10-20% | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Fast |
| Game CBF (Ours) | ~0-5% | ⭐⭐⭐⭐ Good | ⭐⭐⭐ Moderate |

## 🎓 Conclusions for Paper

1. **Pure FM** proves that learned models alone don't guarantee safety
2. **Non-Game CBF** shows independent optimization improves safety but lacks coordination
3. **Game-Theoretic CBF** achieves the best balance between safety and distribution preservation through coordinated optimization

## ⚙️ Configuration

Edit `run_experiments.py` to adjust:
- `N_EXPERIMENTS`: Number of random trials (default: 100)
- `H`: Discrete points per trajectory (default: 50)
- `K`: Integration steps (default: 100)
- `D_SAFE`: Safety distance threshold (default: 8.0)
- Random sampling ranges

## 🐛 Troubleshooting

### OSQP solver warnings
If you see many "OSQP solver status: solved inaccurate", consider:
- Increasing `eps_abs` and `eps_rel` tolerance
- Increasing `max_iter` in OSQP setup
- Adjusting CBF parameters (`PHI0`, `PHI1_SCALE`)

### High collision rates for Game CBF
This may indicate:
- Safety distance `D_SAFE` too large for the scenario
- Integration time step `dt` too large
- Need to tune CBF parameters

### Memory issues
If running 100+ experiments causes memory issues:
- Reduce `N_EXPERIMENTS`
- Don't store full trajectories in results
- Process and save metrics incrementally

## 📚 References

This experiment design follows the statistical evaluation methodology commonly used in:
- Multi-agent reinforcement learning
- Motion planning benchmarks
- Control barrier function validation studies
