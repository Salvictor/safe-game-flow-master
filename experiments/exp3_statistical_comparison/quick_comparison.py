#!/usr/bin/env python3
"""
Quick comparison of Non-Game CBF vs Game-Theoretic CBF

This script visualizes the trajectory differences between:
- Non-Game CBF: Independent optimization (each agent treats the other as obstacle)
- Game-Theoretic CBF: Joint optimization (game-theoretic coordination)

Run a few cases to see the behavioral difference.
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
OUTPUT_DIR = Path(__file__).parent / "quick_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Trajectory parameters
H = 50  # Discrete points per trajectory
T = 1.0  # Total time
K = 100  # Integration steps
dt = T / K
D_SAFE = 10.0  # Safety distance

# CBF parameters
PHI0 = 1.0
PHI1_SCALE = 1.0

OBSTACLES = []  # No static obstacles

print("=" * 80)
print("Quick Comparison: Non-Game CBF vs Game-Theoretic CBF")
print("=" * 80)

# ==================== Load Models ====================
print("\nLoading models...")
model_1 = FlowMatching1D(
    in_channels=2, hidden_channels=256, num_blocks=6,
    time_emb_dim=128, kernel_size=3
).to(DEVICE)
ckpt_1 = torch.load(MODEL_1_PATH, map_location=DEVICE)
model_1.load_state_dict(ckpt_1["model"])
model_1.eval()
print("✓ Agent 1 model loaded (West→East)")

model_2 = FlowMatching1D(
    in_channels=2, hidden_channels=256, num_blocks=6,
    time_emb_dim=128, kernel_size=3
).to(DEVICE)
ckpt_2 = torch.load(MODEL_2_PATH, map_location=DEVICE)
model_2.load_state_dict(ckpt_2["model"])
model_2.eval()
print("✓ Agent 2 model loaded (South→North)")


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


# ==================== CBF Helper Functions ====================
def phi_fun(t, h_val):
    """CBF barrier function"""
    if h_val >= 0:
        return PHI0
    t_clip = min(t, 1.0 - 1e-6)
    return PHI1_SCALE / ((1.0 - t_clip) * (1.0 - t_clip))


def h_of(x, c, r_eff):
    """Barrier function value for obstacles"""
    dx = x - c
    return np.sum(dx**2, axis=1) - r_eff**2


def grad_h_of(x, c):
    """Gradient of barrier function for obstacles"""
    return 2.0 * (x - c)


# ==================== Non-Game CBF (Independent Optimization) ====================
def solve_non_game_cbf(traj_1, v1, traj_2, v2, t, d_safe):
    """
    Non-Game-Theoretic CBF: Each agent independently solves its own QP
    Agent 1: min ||u1||^2  s.t. constraints (treating Agent 2 as obstacle)
    Agent 2: min ||u2||^2  s.t. constraints (treating Agent 1 as obstacle)
    """
    # Agent 1's QP
    u1 = solve_single_agent_qp(traj_1, v1, traj_2, d_safe, t)
    
    # Agent 2's QP
    u2 = solve_single_agent_qp(traj_2, v2, traj_1, d_safe, t)
    
    return u1, u2


def solve_single_agent_qp(my_traj, my_v, other_traj, d_safe, t):
    """Single agent QP: treat other agent as static obstacle"""
    H_traj = my_traj.shape[0]
    A_rows = []
    b_rows = []
    
    # Treat other agent as static obstacle
    d = my_traj - other_traj
    h_other = np.sum(d * d, axis=1) - d_safe**2
    a_other = 2.0 * d
    phi_other = np.array([phi_fun(t, h) for h in h_other])
    b_other = -np.sum(a_other * my_v, axis=1) - phi_other * h_other
    
    for i in range(H_traj):
        row = np.zeros(2 * H_traj, dtype=float)
        row[2*i:2*i+2] = a_other[i]
        A_rows.append(row)
    b_rows.append(b_other)
    
    if len(A_rows) == 0:
        return np.zeros((H_traj, 2))
    
    A_big = np.vstack(A_rows)
    b_big = np.concatenate(b_rows, axis=0)
    
    # Setup OSQP
    n_vars = 2 * H_traj
    P = sparse.eye(n_vars, format='csc')
    q = np.zeros(n_vars)
    A_sparse = sparse.csc_matrix(A_big)
    l = b_big
    u_bound = np.inf * np.ones(len(b_big))
    
    prob = osqp.OSQP()
    prob.setup(P, q, A_sparse, l, u_bound, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    res = prob.solve()
    
    if res.info.status != 'solved':
        # Fallback
        u = np.zeros((H_traj, 2))
        for i in range(H_traj):
            if b_other[i] > 0:
                a_n2 = np.dot(a_other[i], a_other[i]) + 1e-12
                u[i] = (b_other[i] / a_n2) * a_other[i]
        return u
    
    u = res.x.reshape(H_traj, 2)
    return u


# ==================== Game-Theoretic CBF (Joint Optimization) ====================
def solve_game_cbf(traj_1, v1, traj_2, v2, t, d_safe):
    """
    Game-Theoretic CBF: Joint optimization
    min ||u1||^2 + ||u2||^2
    s.t. a^T u1 - a^T u2 >= b  (coordinated constraint)
    """
    H_traj = traj_1.shape[0]
    A_rows = []
    b_rows = []
    
    # Inter-agent collision avoidance (game-theoretic)
    d = traj_1 - traj_2
    h12 = np.sum(d * d, axis=1) - d_safe**2
    a12 = 2.0 * d
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
    
    # Setup OSQP
    n_vars = 4 * H_traj
    P = sparse.eye(n_vars, format='csc')
    q = np.zeros(n_vars)
    A_sparse = sparse.csc_matrix(A_big)
    l = b_big
    u_bound = np.inf * np.ones(len(b_big))
    
    prob = osqp.OSQP()
    prob.setup(P, q, A_sparse, l, u_bound, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    res = prob.solve()
    
    if res.info.status != 'solved':
        # Fallback
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


# ==================== Test Cases ====================
# Define a few test cases
test_cases = [
    {
        'name': 'Case 1: Head-on intersection',
        'start_1': np.array([-30.0, 0.0]),
        'goal_1': np.array([10.0, 0.0]),
        'start_2': np.array([0.0, -20.0]),
        'goal_2': np.array([0.0, 20.0]),
    },
    {
        'name': 'Case 2: Offset intersection',
        'start_1': np.array([-35.0, 2.0]),
        'goal_1': np.array([25.0, 0.0]),
        'start_2': np.array([-0.0, -25.0]),
        'goal_2': np.array([0.0, 35.0]),
    },
    {
        'name': 'Case 3: Wide intersection',
        'start_1': np.array([-25.0, 0.0]),
        'goal_1': np.array([35.0, 0.0]),
        'start_2': np.array([0.0, -35.0]),
        'goal_2': np.array([0.0, 25.0]),
    },
]


# ==================== Run Comparisons ====================
print("\n" + "=" * 80)
print("Running comparison experiments...")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for case_idx, test_case in enumerate(test_cases):
    print(f"\n{test_case['name']}")
    
    start_1 = test_case['start_1']
    goal_1 = test_case['goal_1']
    start_2 = test_case['start_2']
    goal_2 = test_case['goal_2']
    
    # ========== Non-Game CBF ==========
    print("  Running Non-Game CBF...")
    traj_1_ng = np.linspace(start_1, goal_1, H)
    traj_2_ng = np.linspace(start_2, goal_2, H)
    
    total_u1_ng = 0
    total_u2_ng = 0
    peak_u_mean_ng = 0.0
    min_dist_over_time_ng = []
    
    for k in range(K):
        t = k / K
        v1 = velocity_field_1(traj_1_ng, t)
        v2 = velocity_field_2(traj_2_ng, t)
        u1, u2 = solve_non_game_cbf(traj_1_ng, v1, traj_2_ng, v2, t, D_SAFE)
        
        traj_1_ng = traj_1_ng + dt * (v1 + u1)
        traj_2_ng = traj_2_ng + dt * (v2 + u2)

        # Diagnostics over time (NOT the final safety metric)
        dist_t = np.linalg.norm(traj_1_ng - traj_2_ng, axis=1)
        min_dist_over_time_ng.append(dist_t.min())
        u_mean_t = 0.5 * (np.linalg.norm(u1, axis=1).mean() + np.linalg.norm(u2, axis=1).mean())
        peak_u_mean_ng = max(peak_u_mean_ng, float(u_mean_t))

        total_u1_ng += np.linalg.norm(u1, axis=1).sum()
        total_u2_ng += np.linalg.norm(u2, axis=1).sum()
    
    # Evaluate safety ONLY on the final generated trajectories (after full integration)
    final_dist_ng = np.linalg.norm(traj_1_ng - traj_2_ng, axis=1)
    min_dist_ng = final_dist_ng.min()
    collision_ng = min_dist_ng < D_SAFE
    min_dist_time_min_ng = float(np.min(min_dist_over_time_ng)) if len(min_dist_over_time_ng) else float("nan")
    
    # ========== Game-Theoretic CBF ==========
    print("  Running Game-Theoretic CBF...")
    traj_1_game = np.linspace(start_1, goal_1, H)
    traj_2_game = np.linspace(start_2, goal_2, H)
    
    total_u1_game = 0
    total_u2_game = 0
    peak_u_mean_game = 0.0
    min_dist_over_time_game = []
    
    for k in range(K):
        t = k / K
        v1 = velocity_field_1(traj_1_game, t)
        v2 = velocity_field_2(traj_2_game, t)
        u1, u2 = solve_game_cbf(traj_1_game, v1, traj_2_game, v2, t, D_SAFE)
        
        traj_1_game = traj_1_game + dt * (v1 + u1)
        traj_2_game = traj_2_game + dt * (v2 + u2)

        # Diagnostics over time (NOT the final safety metric)
        dist_t = np.linalg.norm(traj_1_game - traj_2_game, axis=1)
        min_dist_over_time_game.append(dist_t.min())
        u_mean_t = 0.5 * (np.linalg.norm(u1, axis=1).mean() + np.linalg.norm(u2, axis=1).mean())
        peak_u_mean_game = max(peak_u_mean_game, float(u_mean_t))

        total_u1_game += np.linalg.norm(u1, axis=1).sum()
        total_u2_game += np.linalg.norm(u2, axis=1).sum()
    
    # Evaluate safety ONLY on the final generated trajectories (after full integration)
    final_dist_game = np.linalg.norm(traj_1_game - traj_2_game, axis=1)
    min_dist_game = final_dist_game.min()
    collision_game = min_dist_game < D_SAFE
    min_dist_time_min_game = float(np.min(min_dist_over_time_game)) if len(min_dist_over_time_game) else float("nan")
    
    # Print results
    print(f"  Non-Game CBF:  min_dist={min_dist_ng:.3f} {'⚠️ COLLISION' if collision_ng else '✓ Safe'}, "
          f"total_correction={total_u1_ng + total_u2_ng:.2f}, "
          f"min_dist_over_time={min_dist_time_min_ng:.3f}, peak_mean|u|={peak_u_mean_ng:.4f}")
    print(f"  Game CBF:      min_dist={min_dist_game:.3f} {'⚠️ COLLISION' if collision_game else '✓ Safe'}, "
          f"total_correction={total_u1_game + total_u2_game:.2f}, "
          f"min_dist_over_time={min_dist_time_min_game:.3f}, peak_mean|u|={peak_u_mean_game:.4f}")
    
    # ========== Plot Non-Game CBF ==========
    ax_ng = axes[0, case_idx]
    ax_ng.plot(traj_1_ng[:, 0], traj_1_ng[:, 1], '-', color='blue', linewidth=3, alpha=0.7, label='Agent 1 (W→E)')
    ax_ng.plot(traj_2_ng[:, 0], traj_2_ng[:, 1], '-', color='red', linewidth=3, alpha=0.7, label='Agent 2 (S→N)')
    
    # Start/goal markers
    ax_ng.scatter(start_1[0], start_1[1], s=150, color='blue', marker='o', edgecolor='black', linewidth=2, zorder=10)
    ax_ng.scatter(goal_1[0], goal_1[1], s=150, color='blue', marker='s', edgecolor='black', linewidth=2, zorder=10)
    ax_ng.scatter(start_2[0], start_2[1], s=150, color='red', marker='o', edgecolor='black', linewidth=2, zorder=10)
    ax_ng.scatter(goal_2[0], goal_2[1], s=150, color='red', marker='s', edgecolor='black', linewidth=2, zorder=10)
    
    # Add info text
    status_color = 'red' if collision_ng else 'green'
    status_text = '⚠️ COLLISION' if collision_ng else '✓ Safe'
    info_text = (
        f"{status_text}\n"
        f"Final min: {min_dist_ng:.2f}m\n"
        f"Min(t): {min_dist_time_min_ng:.2f}m\n"
        f"Total corr: {total_u1_ng + total_u2_ng:.1f}\n"
        f"Peak mean|u|: {peak_u_mean_ng:.3f}"
    )
    ax_ng.text(0.02, 0.98, info_text, transform=ax_ng.transAxes, fontsize=10,
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
    
    ax_ng.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax_ng.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax_ng.set_title(f'Non-Game CBF\n(Independent Optimization)', fontsize=12, fontweight='bold')
    ax_ng.grid(True, alpha=0.3)
    ax_ng.axis('equal')
    if case_idx == 0:
        ax_ng.legend(fontsize=10, loc='lower right')
    
    # ========== Plot Game CBF ==========
    ax_game = axes[1, case_idx]
    ax_game.plot(traj_1_game[:, 0], traj_1_game[:, 1], '-', color='blue', linewidth=3, alpha=0.7, label='Agent 1 (W→E)')
    ax_game.plot(traj_2_game[:, 0], traj_2_game[:, 1], '-', color='red', linewidth=3, alpha=0.7, label='Agent 2 (S→N)')
    
    # Start/goal markers
    ax_game.scatter(start_1[0], start_1[1], s=150, color='blue', marker='o', edgecolor='black', linewidth=2, zorder=10)
    ax_game.scatter(goal_1[0], goal_1[1], s=150, color='blue', marker='s', edgecolor='black', linewidth=2, zorder=10)
    ax_game.scatter(start_2[0], start_2[1], s=150, color='red', marker='o', edgecolor='black', linewidth=2, zorder=10)
    ax_game.scatter(goal_2[0], goal_2[1], s=150, color='red', marker='s', edgecolor='black', linewidth=2, zorder=10)
    
    # Add info text
    status_color = 'red' if collision_game else 'green'
    status_text = '⚠️ COLLISION' if collision_game else '✓ Safe'
    info_text = (
        f"{status_text}\n"
        f"Final min: {min_dist_game:.2f}m\n"
        f"Min(t): {min_dist_time_min_game:.2f}m\n"
        f"Total corr: {total_u1_game + total_u2_game:.1f}\n"
        f"Peak mean|u|: {peak_u_mean_game:.3f}"
    )
    ax_game.text(0.02, 0.98, info_text, transform=ax_game.transAxes, fontsize=10,
                 verticalalignment='top', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
    
    ax_game.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax_game.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax_game.set_title(f'Game-Theoretic CBF (Ours)\n(Joint Optimization)', fontsize=12, fontweight='bold')
    ax_game.grid(True, alpha=0.3)
    ax_game.axis('equal')
    if case_idx == 0:
        ax_game.legend(fontsize=10, loc='lower right')

# Add main title
fig.suptitle('Comparison: Non-Game CBF vs Game-Theoretic CBF\n' +
             'Top Row: Independent Optimization | Bottom Row: Joint Optimization',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()

# Save figure
output_path = OUTPUT_DIR / "trajectory_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Comparison plot saved: {output_path}")

plt.show()

print("\n" + "=" * 80)
print("✓ Quick comparison completed!")
print("=" * 80)
print("\nKey observations:")
print("  - Non-Game CBF: Each agent optimizes independently")
print("  - Game CBF: Coordinated optimization considers both agents' corrections")
print("  - Game CBF typically achieves better safety with less total correction")
