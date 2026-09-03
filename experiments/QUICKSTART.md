# Quick Start: Running Experiments

## ⚡ Fastest Way

```bash
# Make script executable
chmod +x experiments/run_all_experiments.sh

# Run all experiments
./experiments/run_all_experiments.sh
```

This will automatically run both experiments and save all results.

---

## 📋 Individual Experiments

### Experiment 1: Single Agent Obstacle Avoidance

```bash
cd experiments/exp1_single_agent_obstacle_avoidance
python single_agent_safe_generation.py
```

**Output:**
- `results/comparison_original_vs_safe.png` - Main comparison figure
- `results/safety_metrics.png` - Safety statistics
- `results/*.npy` - Raw data files

### Experiment 2: Intersection Multi-Agent

```bash
cd experiments/exp2_intersection_multi_agent
python intersection_multi_agent.py
```

**Output:**
- `results/intersection_multi_agent_results.png` - Main 4-panel figure
- `results/*.npy` - Raw data files

---

## 📊 Expected Results

### Experiment 1
- **Left panel**: Elliptical trajectories (original learned distribution)
- **Right panel**: Trajectories avoiding 4 randomly placed obstacles
- **Demonstrates**: Ability to adapt learned distribution to unseen obstacles

### Experiment 2
- **Panel 1**: Two agents crossing intersection (blue=W→E, red=S→N)
- **Panel 2**: Velocity magnitudes and CBF corrections
- **Panel 3**: Inter-agent distance over time (should stay above safety threshold)
- **Panel 4**: Final distance at each discrete point
- **Demonstrates**: Game-theoretic multi-agent collision avoidance

---

## ⚠️ Prerequisites

Before running experiments, ensure models are trained:

```bash
# Check if models exist
ls runs/model_west_to_east/checkpoint.pt
ls runs/model_south_to_north/checkpoint.pt

# If not, train them:
chmod +x scripts/train_two_models.sh
./scripts/train_two_models.sh
```

---

## 🎯 For Your Thesis

**Key figures to include:**

1. **Experiment 1**: `comparison_original_vs_safe.png`
   - Shows distribution adaptation capability
   - Demonstrates static obstacle avoidance
   - Quantifies safety with barrier functions

2. **Experiment 2**: `intersection_multi_agent_results.png`
   - Shows multi-agent interaction
   - Demonstrates game-theoretic safety
   - Proves collision avoidance capability

**Key metrics to report:**
- Safety violation rate (should be 0%)
- Mean correction magnitude (shows how much CBF intervenes)
- Minimum distance achieved (should be ≥ d_safe)
- Computational efficiency (time per iteration)

---

## 🔧 Quick Customization

### Change safety parameters (Exp 2):

Edit `intersection_multi_agent.py`:
```python
D_SAFE = 10.0  # Increase safety distance
H = 100        # More discrete points
K = 200        # More integration steps
```

### Add more obstacles (Exp 1):

Edit `single_agent_safe_generation.py`:
```python
OBSTACLES = [
    {'c': np.array([x, y]), 'r': radius, 'margin': safety_margin},
    # Add your obstacles here
]
```

---

## 📁 Results Location

All results are saved in:
```
experiments/
├── exp1_single_agent_obstacle_avoidance/results/
└── exp2_intersection_multi_agent/results/
```

Each contains:
- `.png` figures (high resolution, 300 DPI)
- `.npy` data files (for further analysis)

---

## 🐛 Troubleshooting

**"Model not found"**
→ Train models first: `./scripts/train_two_models.sh`

**"QP solver warning"**
→ This is normal, fallback solver is used automatically

**"Safety violations detected"**
→ Increase `D_SAFE` or reduce `dt` for stricter safety

---

## 📖 More Information

- Full documentation: `experiments/README.md`
- Training guide: `docs/training_guide.md` (if exists)
- Source code: `experiments/exp*/`
