#!/usr/bin/env python3
"""
Experiment 2: Intersection Multi-Agent Game-Theoretic Interaction

This experiment demonstrates safe trajectory generation for two agents crossing
an intersection using game-theoretic CBF constraints. Each agent uses a different
learned velocity field model:
- Agent 1: West-to-East model (horizontal motion)
- Agent 2: South-to-North model (vertical motion)

Key features:
- Game-theoretic QP solver for multi-agent interaction
- Distributed safety constraints (each agent's perspective)
- Real-time collision avoidance with minimal deviation
- Static obstacle avoidance (optional)
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
MODEL_1_PATH = "/home/karl/safe-game-flow/runs/model_west_to_east/checkpoint.pt"
MODEL_2_PATH = "/home/karl/safe-game-flow/runs/model_south_to_north/checkpoint.pt"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Agent 1: West to East trajectory
START_1 = np.array([-30.0, 0.0])
GOAL_1 = np.array([10.0, 0.0])

# Agent 2: South to North trajectory
START_2 = np.array([0.0, -20.0])
GOAL_2 = np.array([0.0, 20.0])

# Trajectory parameters
H = 50  # Number of discrete points
T = 1.0  # Total time
K = 100  # Integration steps
dt = T / K

# Safety distance between agents
D_SAFE = 8.0

# Static obstacles (empty for now, can be added)
OBSTACLES = [
    # {'c': np.array([0.0, 0.0]), 'r': 5.0, 'margin': 0.2},
]

for obs in OBSTACLES:
    obs['r_eff'] = obs['r'] + obs['margin']

# CBF parameters
PHI0 = 1.0
PHI1_SCALE = 1.0


# ==================== Load Models ====================
print("=" * 80)
print("Experiment 2: Intersection Multi-Agent Game-Theoretic Interaction")
print("=" * 80)

print(f"\nLoading Agent 1 model (West→East): {MODEL_1_PATH}")
model_1 = FlowMatching1D(
    in_channels=2,
    hidden_channels=256,
    num_blocks=6,
    time_emb_dim=128,
    kernel_size=3
).to(DEVICE)
ckpt_1 = torch.load(MODEL_1_PATH, map_location=DEVICE)
model_1.load_state_dict(ckpt_1["model"])
model_1.eval()
print("✓ Agent 1 model loaded")

print(f"\nLoading Agent 2 model (South→North): {MODEL_2_PATH}")
model_2 = FlowMatching1D(
    in_channels=2,
    hidden_channels=256,
    num_blocks=6,
    time_emb_dim=128,
    kernel_size=3
).to(DEVICE)
ckpt_2 = torch.load(MODEL_2_PATH, map_location=DEVICE)
model_2.load_state_dict(ckpt_2["model"])
model_2.eval()
print("✓ Agent 2 model loaded")


# ==================== Velocity Field Functions ====================
@torch.no_grad()
def velocity_field_1(traj, t):
    """Agent 1 velocity field (West→East model)"""
    t_input = torch.tensor(t, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    traj_input = torch.tensor(traj.T, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    v = model_1(traj_input, t_input).detach().cpu().numpy().squeeze(0).T
    return v


@torch.no_grad()
def velocity_field_2(traj, t):
    """Agent 2 velocity field (South→North model)"""
    t_input = torch.tensor(t, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    traj_input = torch.tensor(traj.T, device=DEVICE, dtype=torch.float32).unsqueeze(0)
    v = model_2(traj_input, t_input).detach().cpu().numpy().squeeze(0).T
    return v


# ==================== CBF Functions ====================
def phi_fun(t, h_val):
    """CBF barrier function"""
    if h_val >= 0:
        return PHI0
    t_clip = min(t, 1.0 - 1e-6)
    return PHI1_SCALE / ((1.0 - t_clip)*(1.0 - t_clip))


def h_of(x, c, r_eff):
    """Barrier function value for obstacles"""
    dx = x - c
    return np.sum(dx**2, axis=1) - r_eff**2


def grad_h_of(x, c):
    """Gradient of barrier function for obstacles"""
    return 2.0 * (x - c)


def solve_min_norm_qp_two_agents(traj1, v1, traj2, v2, t, obstacles1, obstacles2, d_safe):
    """
    Solve game-theoretic minimum-norm QP for two agents:
    min 0.5 * (||u1||^2 + ||u2||^2)
    s.t. 
        - Agent 1 static obstacle constraints
        - Agent 2 static obstacle constraints
        - Inter-agent collision avoidance constraint
    
    Args:
        traj1, v1: Agent 1 trajectory and nominal velocity (H, 2)
        traj2, v2: Agent 2 trajectory and nominal velocity (H, 2)
        t: Current time ∈ [0, 1)
        obstacles1, obstacles2: Static obstacles for each agent
        d_safe: Minimum safe distance between agents
        
    Returns:
        u1, u2: (H, 2) Correction velocities for both agents
    """
    H_traj = traj1.shape[0]
    A_rows = []
    b_rows = []
    
    # 1) Agent 1 static obstacle constraints
    for obs in obstacles1:
        c = obs['c']
        r_eff = obs['r_eff']
        h1 = h_of(traj1, c, r_eff)
        a1 = grad_h_of(traj1, c)
        phi1 = np.array([phi_fun(t, h) for h in h1])
        b1 = -np.sum(a1 * v1, axis=1) - phi1 * h1
        
        for i in range(H_traj):
            row = np.zeros(4 * H_traj, dtype=float)
            row[2*i:2*i+2] = a1[i]
            A_rows.append(row)
        b_rows.append(b1)
    
    # 2) Agent 2 static obstacle constraints
    for obs in obstacles2:
        c = obs['c']
        r_eff = obs['r_eff']
        h2 = h_of(traj2, c, r_eff)
        a2 = grad_h_of(traj2, c)
        phi2 = np.array([phi_fun(t, h) for h in h2])
        b2 = -np.sum(a2 * v2, axis=1) - phi2 * h2
        
        for i in range(H_traj):
            row = np.zeros(4 * H_traj, dtype=float)
            row[2*H_traj + 2*i:2*H_traj + 2*i + 2] = a2[i]
            A_rows.append(row)
        b_rows.append(b2)
    
    # 3) Inter-agent collision avoidance constraint
    d = traj1 - traj2
    h12 = np.sum(d * d, axis=1) - d_safe**2
    a12 = 2.0 * d  # Gradient w.r.t. agent 1
    phi12 = np.array([phi_fun(t, h) for h in h12])
    b12 = -np.sum(a12 * v1, axis=1) + np.sum(a12 * v2, axis=1) - phi12 * h12
    
    for i in range(H_traj):
        row = np.zeros(4 * H_traj, dtype=float)
        row[2*i:2*i+2] = a12[i]  # Agent 1 coefficient
        row[2*H_traj + 2*i:2*H_traj + 2*i + 2] = -a12[i]  # Agent 2 coefficient
        A_rows.append(row)
    b_rows.append(b12)
    
    if len(A_rows) == 0:
        return np.zeros((H_traj, 2)), np.zeros((H_traj, 2))
    
    A_big = np.vstack(A_rows)
    b_big = np.concatenate(b_rows, axis=0)
    
    # Setup OSQP problem
    # minimize    (1/2) u^T P u + q^T u
    # subject to  l <= A u <= u_bound
    # For our problem: P = I, q = 0, l = b_big, u_bound = inf
    
    n_vars = 4 * H_traj  # Both agents: u1 and u2
    P = sparse.eye(n_vars, format='csc')  # Identity matrix (sparse)
    q = np.zeros(n_vars)
    
    # Inequality constraints: A u >= b  =>  l = b, u_bound = inf
    A_sparse = sparse.csc_matrix(A_big)
    l = b_big  # Lower bound
    u_bound = np.inf * np.ones(len(b_big))  # Upper bound (infinity)
    
    # Create OSQP solver
    prob = osqp.OSQP()
    prob.setup(P, q, A_sparse, l, u_bound, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    
    # Solve
    res = prob.solve()
    
    if res.info.status != 'solved':
        print(f"  Warning: OSQP two-agent solver status: {res.info.status} at t={t:.3f}")
        # Fallback: simple closed-form solution for inter-agent constraint only
        u1 = np.zeros((H_traj, 2))
        u2 = np.zeros((H_traj, 2))
        for i in range(H_traj):
            a = a12[i]
            a_n2 = np.dot(a, a) + 1e-12
            b = b12[i]
            if b > 0:
                u1[i] = (b / (2 * a_n2)) * a
                u2[i] = (-b / (2 * a_n2)) * a
        return u1, u2
    
    u_flat = res.x
    u1 = u_flat[:2*H_traj].reshape(H_traj, 2)
    u2 = u_flat[2*H_traj:].reshape(H_traj, 2)
    
    return u1, u2


# ==================== Initialize Trajectories ====================
print("\n" + "-" * 80)
print("Initializing trajectories...")
print("-" * 80)

target_traj_1 = np.linspace(START_1, GOAL_1, H)
target_traj_2 = np.linspace(START_2, GOAL_2, H)

traj_1 = target_traj_1.copy()
traj_2 = target_traj_2.copy()

print(f"Agent 1: {START_1} → {GOAL_1}")
print(f"Agent 2: {START_2} → {GOAL_2}")
print(f"Safety distance: {D_SAFE:.2f}")
print(f"Discrete points per trajectory: {H}")
print(f"Integration steps: {K}")


# ==================== Simulation Loop ====================
print("\n" + "-" * 80)
print("Running simulation...")
print("-" * 80)

# Storage for metrics
mean_v1_list, mean_v2_list = [], []
mean_u1_list, mean_u2_list = [], []
min_dist_list = []
mean_dist_list = []

for k in range(K):
    t = k / K
    
    # Compute nominal velocities from learned models
    v1 = velocity_field_1(traj_1, t)
    v2 = velocity_field_2(traj_2, t)
    
    # Solve game-theoretic QP for safe corrections
    u1, u2 = solve_min_norm_qp_two_agents(traj_1, v1, traj_2, v2, t, 
                                          OBSTACLES, OBSTACLES, D_SAFE)
    
    # Update trajectories
    traj_1 = traj_1 + dt * (v1 + u1)
    traj_2 = traj_2 + dt * (v2 + u2)
    
    # Compute inter-agent distances
    dist_between = np.linalg.norm(traj_1 - traj_2, axis=1)
    min_dist_list.append(dist_between.min())
    mean_dist_list.append(dist_between.mean())
    
    # Record metrics
    mean_v1_list.append(np.linalg.norm(v1, axis=1).mean())
    mean_v2_list.append(np.linalg.norm(v2, axis=1).mean())
    mean_u1_list.append(np.linalg.norm(u1, axis=1).mean())
    mean_u2_list.append(np.linalg.norm(u2, axis=1).mean())
    
    if (k + 1) % 20 == 0:
        print(f"  Step {k+1}/{K}, t={t:.3f}, min_dist={dist_between.min():.3f}")

trajectory_1 = traj_1
trajectory_2 = traj_2

print("✓ Simulation completed")


# ==================== Statistics ====================
min_dist_array = np.array(min_dist_list)
mean_dist_array = np.array(mean_dist_list)
mean_v1_array = np.array(mean_v1_list)
mean_v2_array = np.array(mean_v2_list)
mean_u1_array = np.array(mean_u1_list)
mean_u2_array = np.array(mean_u2_list)

print("\n" + "=" * 80)
print("Safety Statistics:")
print("=" * 80)
print(f"Safety distance threshold (d_safe): {D_SAFE:.2f}")
print(f"Minimum distance (global): {min_dist_array.min():.4f}")
print(f"Minimum distance (mean): {min_dist_array.mean():.4f}")
print(f"Mean distance (global average): {mean_dist_array.mean():.4f}")
violations = np.sum(min_dist_array < D_SAFE)
print(f"Safety violations: {violations} / {K} time steps")
if violations > 0:
    max_violation = (D_SAFE - min_dist_array[min_dist_array < D_SAFE]).max()
    print(f"Maximum violation: {max_violation:.4f}")
print("=" * 80)


# ==================== Save Data ====================
print("\n" + "-" * 80)
print("Saving data...")
print("-" * 80)

np.save(OUTPUT_DIR / "trajectory_agent1.npy", trajectory_1)
np.save(OUTPUT_DIR / "trajectory_agent2.npy", trajectory_2)
np.save(OUTPUT_DIR / "min_distance_over_time.npy", min_dist_array)
np.save(OUTPUT_DIR / "mean_distance_over_time.npy", mean_dist_array)
np.save(OUTPUT_DIR / "mean_velocities.npy", np.stack([mean_v1_array, mean_v2_array]))
np.save(OUTPUT_DIR / "mean_corrections.npy", np.stack([mean_u1_array, mean_u2_array]))

print(f"✓ Data saved to: {OUTPUT_DIR}")


# ==================== Visualization ====================
print("\n" + "-" * 80)
print("Creating visualizations...")
print("-" * 80)

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Trajectories + Obstacles
ax = axs[0, 0]

# Agent 1 trajectory (West→East) - Blue
sc1 = ax.scatter(trajectory_1[:, 0], trajectory_1[:, 1], 
                 c=np.linspace(0, 1, H), cmap='Blues', s=30, alpha=0.8, 
                 vmin=0.2, vmax=1.0, zorder=5)
ax.plot(trajectory_1[:, 0], trajectory_1[:, 1], 
        color='blue', linestyle='-', alpha=0.6, linewidth=3)

# Agent 2 trajectory (South→North) - Red
sc2 = ax.scatter(trajectory_2[:, 0], trajectory_2[:, 1], 
                 c=np.linspace(0, 1, H), cmap='Reds', s=30, alpha=0.8, 
                 vmin=0.2, vmax=1.0, zorder=5)
ax.plot(trajectory_2[:, 0], trajectory_2[:, 1], 
        color='red', linestyle='-', alpha=0.6, linewidth=3)

# Obstacles
theta = np.linspace(0, 2 * np.pi, 100)
for j, obs in enumerate(OBSTACLES):
    c = obs['c']
    r_eff = obs['r_eff']
    x_circle = c[0] + r_eff * np.cos(theta)
    y_circle = c[1] + r_eff * np.sin(theta)
    ax.plot(x_circle, y_circle, 'gray', linestyle='--', linewidth=2, alpha=0.6)

# # Start and goal markers with larger size and labels
# ax.plot([START_1[0]], [START_1[1]], 'o', color='blue', markersize=15, 
#         markeredgewidth=2, markeredgecolor='darkblue', alpha=0.9, zorder=10, label='Agent 1: Start')
# ax.plot([GOAL_1[0]], [GOAL_1[1]], 's', color='blue', markersize=15, 
#         markeredgewidth=2, markeredgecolor='darkblue', alpha=0.9, zorder=10, label='Agent 1: Goal')
# ax.plot([START_2[0]], [START_2[1]], 'o', color='red', markersize=15, 
#         markeredgewidth=2, markeredgecolor='darkred', alpha=0.9, zorder=10, label='Agent 2: Start')
# ax.plot([GOAL_2[0]], [GOAL_2[1]], 's', color='red', markersize=15, 
#         markeredgewidth=2, markeredgecolor='darkred', alpha=0.9, zorder=10, label='Agent 2: Goal')

# Add text annotations for agents
ax.text(trajectory_1[H//4, 0], trajectory_1[H//4, 1] + 3, 'Agent 1\n(West→East)', 
        fontsize=11, fontweight='bold', color='blue', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='blue', alpha=0.8))
ax.text(trajectory_2[H//4, 0] + 7, trajectory_2[H//4, 1], 'Agent 2\n(South→North)', 
        fontsize=11, fontweight='bold', color='red', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))

ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='upper left', framealpha=0.95, edgecolor='black', fancybox=True)
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_title('Safe Trajectory Generation with Game-Theoretic CBF\n(Multi-Agent Intersection Crossing)', 
             fontsize=13, fontweight='bold')

# Plot 2: Velocities and Corrections
ax2 = axs[0, 1]
tt = np.arange(K) * dt
ax2.plot(tt, mean_v1_array, label='Agent 1: Mean |v₁|', linewidth=2)
ax2.plot(tt, mean_v2_array, label='Agent 2: Mean |v₂|', linewidth=2)
ax2.plot(tt, mean_u1_array, '--', label='Agent 1: Mean |u₁| (CBF correction)', linewidth=2)
ax2.plot(tt, mean_u2_array, '--', label='Agent 2: Mean |u₂| (CBF correction)', linewidth=2)
ax2.axhline(0.0, color='k', lw=0.8)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.set_xlabel('Time t', fontsize=12)
ax2.set_ylabel('Velocity magnitude', fontsize=12)
ax2.set_title('Nominal Velocities and CBF Corrections', fontsize=13, fontweight='bold')

# Plot 3: Inter-agent Distance Over Time
ax3 = axs[1, 0]
ax3.plot(tt, min_dist_array, label='Minimum distance', linewidth=2.5, color='darkblue')
ax3.plot(tt, mean_dist_array, label='Mean distance', linewidth=2, alpha=0.7, color='steelblue')
ax3.axhline(D_SAFE, color='red', linestyle='--', linewidth=2.5, 
            label=f'Safety threshold (d_safe={D_SAFE:.1f})')
ax3.fill_between(tt, 0, D_SAFE, color='red', alpha=0.15, label='Unsafe region')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_xlabel('Time t', fontsize=12)
ax3.set_ylabel('Distance (m)', fontsize=12)
ax3.set_title('Inter-Agent Distance Over Time', fontsize=13, fontweight='bold')
ax3.set_ylim(bottom=0)

# Plot 4: Final Trajectory Inter-Agent Distance
ax4 = axs[1, 1]
final_dist = np.linalg.norm(trajectory_1 - trajectory_2, axis=1)
point_indices = np.arange(H)
ax4.plot(point_indices, final_dist, 'o-', linewidth=2, markersize=5, color='darkgreen')
ax4.axhline(D_SAFE, color='red', linestyle='--', linewidth=2.5, 
            label=f'Safety threshold (d_safe={D_SAFE:.1f})')
ax4.fill_between(point_indices, 0, D_SAFE, color='red', alpha=0.15, label='Unsafe region')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.set_xlabel('Discrete point index', fontsize=12)
ax4.set_ylabel('Distance (m)', fontsize=12)
ax4.set_title(f'Final Inter-Agent Distance (min={final_dist.min():.3f} m)', 
              fontsize=13, fontweight='bold')
ax4.set_ylim(bottom=0)

plt.tight_layout()
output_path = OUTPUT_DIR / "intersection_multi_agent_results.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Main results plot saved: {output_path}")

plt.show()

print("\n" + "=" * 80)
print("✓ Experiment 2 completed successfully!")
print("=" * 80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  - intersection_multi_agent_results.png : Main visualization")
print("  - trajectory_agent1.npy : Agent 1 trajectory data")
print("  - trajectory_agent2.npy : Agent 2 trajectory data")
print("  - min_distance_over_time.npy : Minimum distance time series")
print("  - mean_distance_over_time.npy : Mean distance time series")
print("  - mean_velocities.npy : Velocity statistics")
print("  - mean_corrections.npy : CBF correction statistics")
