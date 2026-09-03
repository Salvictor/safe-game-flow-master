#!/usr/bin/env python3
"""
Experiment 1: Single Agent Static Obstacle Avoidance

This experiment demonstrates the safe trajectory generation capability for a single agent
in the presence of static obstacles. It compares:
- Original learned distribution (elliptical trajectories from training data)
- Safe generation with CBF-based constraints (avoiding obstacles)

Key visualization:
- Left plot: Original learned trajectories (elliptical distribution)
- Right plot: Safe generated trajectories (avoiding randomly placed obstacles)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import osqp
from scipy import sparse

from safe_game_flow.flow_matching.model import FlowMatching1D

# ==================== Configuration ====================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/karl/safe-game-flow/runs/model_west_to_east/checkpoint.pt"
DATA_PATH = "/home/karl/safe-game-flow/datasets/trajectories_west_to_east.npy"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Trajectory parameters
H = 200  # Number of discrete points in trajectory
T = 1.0  # Total time
K = 100  # Number of integration steps
dt = T / K

# Obstacles (randomly placed in the middle area)
np.random.seed(42)
OBSTACLES = [
    {'c': np.array([0.0, 10.0]), 'r':3.0, 'margin': 1.0},
    {'c': np.array([00.0, -5.0]), 'r': 1.5, 'margin': 1.0},
    {'c': np.array([15.0, 5.8]), 'r': 1, 'margin': 1.0},
    {'c': np.array([10.0, -8.0]), 'r': 3.0, 'margin': 1.0},
]

for obs in OBSTACLES:
    obs['r_eff'] = obs['r'] + obs['margin']

# CBF parameters
PHI0 = 1.0
PHI1_SCALE = 1

NUM_SAMPLES = 20  # Number of trajectories to generate


# ==================== Load Model and Dataset ====================
print("=" * 80)
print("Experiment 1: Single Agent Static Obstacle Avoidance")
print("=" * 80)

print(f"\nLoading dataset from: {DATA_PATH}")
training_data = np.load(DATA_PATH)  # (N, 2, H)
print(f"✓ Dataset loaded successfully")
print(f"  Shape: {training_data.shape}")
print(f"  Number of training trajectories: {training_data.shape[0]}")

print(f"\nLoading model from: {MODEL_PATH}")
model = FlowMatching1D(
    in_channels=2,
    hidden_channels=256,
    num_blocks=6,
    time_emb_dim=128,
    kernel_size=3
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

mean = checkpoint["mean"].to(DEVICE)  # (2, 1)
std = checkpoint["std"].to(DEVICE)    # (2, 1)

print(f"✓ Model loaded successfully")
print(f"  Device: {DEVICE}")
print(f"  Mean: {mean.squeeze().cpu().numpy()}")
print(f"  Std: {std.squeeze().cpu().numpy()}")


# ==================== Velocity Field Function ====================
@torch.no_grad()
def velocity_field(traj, t):
    """Compute velocity field from the learned model"""
    t_input = torch.tensor(t, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    traj_input = torch.tensor(traj.T, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    v = model(traj_input, t_input).detach().cpu().numpy().squeeze(0).T
    return v


# ==================== CBF Functions ====================
def phi_fun(t, h_val):
    """CBF barrier function"""
    if h_val >= 0:
        return PHI0
    t_clip = min(t, 1.0 - 1e-6)
    return PHI1_SCALE / ((1.0 - t_clip)*(1.0 - t_clip))


def h_of(x, c, r_eff):
    """Barrier function value"""
    dx = x - c
    return np.sum(dx**2, axis=1) - r_eff**2


def grad_h_of(x, c):
    """Gradient of barrier function"""
    return 2.0 * (x - c)


def solve_min_norm_qp(traj, v, t, obstacles):
    """
    Solve minimum-norm QP for safe trajectory generation using OSQP:
    min 0.5 * ||u||^2
    s.t. a_i^T u_i >= b_i  for all i, obstacles
    
    Returns:
        u: (H, 2) correction velocity
        h_min: (H,) minimum barrier values
    """
    H_traj = traj.shape[0]
    A_rows = []
    b_rows = []
    h_all = []
    
    for obs in obstacles:
        c = obs['c']
        r_eff = obs['r_eff']
        h_val = h_of(traj, c, r_eff)
        a = grad_h_of(traj, c)
        phi = np.array([phi_fun(t, h_i) for h_i in h_val])
        bi = -np.sum(a * v, axis=1) - phi * h_val
        
        for i in range(H_traj):
            row = np.zeros(2 * H_traj, dtype=float)
            row[2*i:2*i+2] = a[i]
            A_rows.append(row)
        b_rows.append(bi)
        h_all.append(h_val)
    
    if len(A_rows) == 0:
        return np.zeros((H_traj, 2)), np.array([0.0] * H_traj)
    
    A_big = np.vstack(A_rows)
    b_big = np.concatenate(b_rows, axis=0)
    
    # Setup OSQP problem
    # minimize    (1/2) u^T P u + q^T u
    # subject to  l <= A u <= u_bound
    # For our problem: P = I, q = 0, l = b_big, u_bound = inf
    
    n_vars = 2 * H_traj
    P = sparse.eye(n_vars, format='csc')  # Identity matrix (sparse)
    q = np.zeros(n_vars)
    
    # Inequality constraints: A u >= b  =>  -inf <= A u <= inf, with lower bound = b
    A_sparse = sparse.csc_matrix(A_big)
    l = b_big  # Lower bound
    u_bound = np.inf * np.ones(len(b_big))  # Upper bound (infinity)
    
    # Create OSQP solver
    prob = osqp.OSQP()
    prob.setup(P, q, A_sparse, l, u_bound, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    
    # Solve
    res = prob.solve()
    
    if res.info.status != 'solved':
        print(f"  Warning: OSQP solver status: {res.info.status} at t={t:.3f}")
        # Fallback to simple solution
        u = np.zeros((H_traj, 2))
        for i in range(H_traj):
            best_u = np.zeros(2)
            best_gain = 0.0
            for obs in obstacles:
                c = obs['c']
                r_eff = obs['r_eff']
                a_i = 2.0 * (traj[i] - c)
                a_n2 = np.dot(a_i, a_i) + 1e-12
                h_i = np.dot(traj[i] - c, traj[i] - c) - r_eff**2
                b_i = -np.dot(a_i, v[i]) - phi_fun(t, h_i) * h_i
                if b_i > 0:
                    u_i = (b_i / a_n2) * a_i
                    gain = np.dot(u_i, u_i)
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u_i
            u[i] = best_u
    else:
        u = res.x.reshape(H_traj, 2)
    
    h_min = np.min(np.vstack(h_all), axis=0)
    return u, h_min


# ==================== Sample Initial Trajectories from Dataset ====================
print("\n" + "-" * 80)
print("Sampling initial trajectories from training dataset...")
print("-" * 80)

# Randomly sample indices from the dataset
np.random.seed(42)
sampled_indices = np.random.choice(training_data.shape[0], NUM_SAMPLES, replace=False)
initial_trajectories = training_data[sampled_indices]  # (NUM_SAMPLES, 2, H)
initial_trajectories = initial_trajectories.transpose(0, 2, 1)  # (NUM_SAMPLES, H, 2)

print(f"✓ Sampled {NUM_SAMPLES} initial trajectories from dataset")
print(f"  Initial shape: {initial_trajectories.shape}")


# ==================== Generate Original Trajectories (No Obstacles) ====================
print("\n" + "-" * 80)
print("Generating original trajectories (learned distribution, no obstacles)...")
print("Starting from t=0 sampled from dataset...")
print("-" * 80)

original_trajectories = []
for i in range(NUM_SAMPLES):
    # Start from sampled trajectory at t=0
    traj = initial_trajectories[i].copy()
    
    for k in range(K):
        t = k / K
        v = velocity_field(traj, t)
        traj = traj + dt * v
    
    original_trajectories.append(traj)
    if (i + 1) % 10 == 0:
        print(f"  Generated {i+1}/{NUM_SAMPLES} original trajectories")

original_trajectories = np.array(original_trajectories)  # (NUM_SAMPLES, H, 2)
print(f"✓ Original trajectories shape: {original_trajectories.shape}")


# ==================== Generate Safe Trajectories (With Obstacles) ====================
print("\n" + "-" * 80)
print("Generating safe trajectories (with obstacle avoidance)...")
print("Starting from same t=0 sampled from dataset...")
print("-" * 80)

safe_trajectories = []
min_h_values = []
mean_u_norms = []

for i in range(NUM_SAMPLES):
    # Start from the SAME sampled trajectory at t=0 as original
    traj = initial_trajectories[i].copy()
    h_min_list = []
    u_norm_list = []
    
    for k in range(K):
        t = k / K
        v = velocity_field(traj, t)
        u, h_min = solve_min_norm_qp(traj, v, t, OBSTACLES)
        
        traj = traj + dt * (v + u)
        h_min_list.append(h_min.min())
        u_norm_list.append(np.linalg.norm(u, axis=1).mean())
    
    safe_trajectories.append(traj)
    min_h_values.append(h_min_list)
    mean_u_norms.append(u_norm_list)
    
    if (i + 1) % 10 == 0:
        print(f"  Generated {i+1}/{NUM_SAMPLES} safe trajectories")

safe_trajectories = np.array(safe_trajectories)  # (NUM_SAMPLES, H, 2)
min_h_values = np.array(min_h_values)  # (NUM_SAMPLES, K)
mean_u_norms = np.array(mean_u_norms)  # (NUM_SAMPLES, K)

print(f"✓ Safe trajectories shape: {safe_trajectories.shape}")


# ==================== Save Data ====================
print("\n" + "-" * 80)
print("Saving data...")
print("-" * 80)

np.save(OUTPUT_DIR / "initial_trajectories.npy", initial_trajectories)
np.save(OUTPUT_DIR / "original_trajectories.npy", original_trajectories)
np.save(OUTPUT_DIR / "safe_trajectories.npy", safe_trajectories)
np.save(OUTPUT_DIR / "min_h_values.npy", min_h_values)
np.save(OUTPUT_DIR / "mean_u_norms.npy", mean_u_norms)

print(f"✓ Data saved to: {OUTPUT_DIR}")
print(f"  - initial_trajectories.npy (t=0 from dataset)")
print(f"  - original_trajectories.npy")
print(f"  - safe_trajectories.npy")
print(f"  - min_h_values.npy")
print(f"  - mean_u_norms.npy")


# ==================== Statistics ====================
print("\n" + "-" * 80)
print("Safety Statistics:")
print("-" * 80)

# Check violations
violations = (min_h_values < 0).sum()
total_checks = min_h_values.size
violation_rate = violations / total_checks * 100

print(f"Total safety checks: {total_checks}")
print(f"Safety violations: {violations} ({violation_rate:.2f}%)")
print(f"Minimum h value: {min_h_values.min():.4f}")
print(f"Mean correction norm: {mean_u_norms.mean():.4f}")


# ==================== Visualization ====================
print("\n" + "-" * 80)
print("Creating visualizations...")
print("-" * 80)

fig, axes = plt.subplots(2, 1, figsize=(16, 14))

# Colormap for trajectories
colors = plt.cm.viridis(np.linspace(0, 1, NUM_SAMPLES))

# Top plot: Original trajectories (no obstacles)
ax1 = axes[0]
for i in range(NUM_SAMPLES):
    traj = original_trajectories[i]
    ax1.plot(traj[:, 0], traj[:, 1], '-', alpha=0.5, linewidth=2, color=colors[i], zorder=5)
    ax1.scatter(traj[0, 0], traj[0, 1], color='green', s=40, alpha=0.7, zorder=10)
    ax1.scatter(traj[-1, 0], traj[-1, 1], color='red', s=40, alpha=0.7, zorder=10)

ax1.plot([], [], 'o', color='green', markersize=10, label='Start')
ax1.plot([], [], 's', color='red', markersize=10, label='Goal')
ax1.set_xlabel('X (m)', fontsize=14)
ax1.set_ylabel('Y (m)', fontsize=14)
ax1.set_title('Original Learned Distribution\n(Elliptical Trajectories, No Obstacles)', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12, loc='upper right')
ax1.set_aspect('equal', adjustable='box')
ax1.text(0.02, 0.98, f'N = {NUM_SAMPLES} trajectories\nInitialized from dataset (t=0)', 
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Bottom plot: Safe trajectories (with obstacles)
ax2 = axes[1]

# Draw obstacles first (lower zorder so trajectories are on top)
theta = np.linspace(0, 2 * np.pi, 100)
for j, obs in enumerate(OBSTACLES):
    c = obs['c']
    r = obs['r']
    r_eff = obs['r_eff']
    
    # Inner circle (actual obstacle)
    x_inner = c[0] + r * np.cos(theta)
    y_inner = c[1] + r * np.sin(theta)
    ax2.fill(x_inner, y_inner, color='darkred', alpha=0.6, zorder=1, 
             label='Obstacle' if j == 0 else None)
    
    # Outer circle (safety margin) - thinner line, lower zorder
    x_outer = c[0] + r_eff * np.cos(theta)
    y_outer = c[1] + r_eff * np.sin(theta)
    ax2.plot(x_outer, y_outer, 'r--', linewidth=1, alpha=0.6, zorder=2,
             label='Safety Margin' if j == 0 else None)

# Draw trajectories on top
for i in range(NUM_SAMPLES):
    traj = safe_trajectories[i]
    ax2.plot(traj[:, 0], traj[:, 1], '-', alpha=0.5, linewidth=2, color=colors[i], zorder=5)
    ax2.scatter(traj[0, 0], traj[0, 1], color='green', s=40, alpha=0.7, zorder=10)
    ax2.scatter(traj[-1, 0], traj[-1, 1], color='red', s=40, alpha=0.7, zorder=10)

ax2.plot([], [], 'o', color='green', markersize=10, label='Start')
ax2.plot([], [], 's', color='red', markersize=10, label='Goal')
ax2.set_xlabel('X (m)', fontsize=14)
ax2.set_ylabel('Y (m)', fontsize=14)
ax2.set_title('Safe Generation with CBF\n(Avoiding {0} Obstacles)'.format(len(OBSTACLES)), fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12, loc='upper right')
ax2.set_aspect('equal', adjustable='box')
ax2.text(0.02, 0.98, f'N = {NUM_SAMPLES} trajectories\nSame t=0 as top panel\n{len(OBSTACLES)} obstacles',
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
output_path = OUTPUT_DIR / "comparison_original_vs_safe.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved: {output_path}")

# Additional plot: Safety metrics over time
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Minimum h values over time
ax_h = axes2[0]
time_steps = np.arange(K) * dt
for i in range(min(10, NUM_SAMPLES)):  # Show first 10 trajectories
    ax_h.plot(time_steps, min_h_values[i], alpha=0.5, linewidth=1.5)
ax_h.axhline(0, color='r', linestyle='--', linewidth=2, label='Safety boundary')
ax_h.fill_between(time_steps, -30, 0, color='red', alpha=0.1)
ax_h.set_xlabel('Time t', fontsize=12)
ax_h.set_ylabel('Minimum h(x)', fontsize=12)
ax_h.set_title('Barrier Function Values Over Time', fontsize=14, fontweight='bold')
ax_h.grid(True, alpha=0.3)
ax_h.legend(fontsize=10)

# Plot 2: Mean correction norms over time
ax_u = axes2[1]
for i in range(min(10, NUM_SAMPLES)):
    ax_u.plot(time_steps, mean_u_norms[i], alpha=0.5, linewidth=1.5)
ax_u.set_xlabel('Time t', fontsize=12)
ax_u.set_ylabel('Mean ||u|| (correction magnitude)', fontsize=12)
ax_u.set_title('CBF Correction Magnitude Over Time', fontsize=14, fontweight='bold')
ax_u.grid(True, alpha=0.3)

plt.tight_layout()
metrics_path = OUTPUT_DIR / "safety_metrics.png"
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
print(f"✓ Safety metrics plot saved: {metrics_path}")

plt.show()

print("\n" + "=" * 80)
print("✓ Experiment 1 completed successfully!")
print("=" * 80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  - comparison_original_vs_safe.png : Main comparison visualization")
print("  - safety_metrics.png : Safety statistics over time")
print("  - initial_trajectories.npy : Initial states (t=0) from dataset")
print("  - original_trajectories.npy : Original trajectory data (no obstacles)")
print("  - safe_trajectories.npy : Safe trajectory data (with CBF)")
print("  - min_h_values.npy : Barrier function values")
print("  - mean_u_norms.npy : Correction magnitudes")
print("\nKey insight:")
print("  Both original and safe trajectories start from the SAME t=0 states")
print("  (sampled from training dataset), showing how CBF modifies the flow.")