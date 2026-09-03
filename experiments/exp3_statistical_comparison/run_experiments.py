#!/usr/bin/env python3
"""
Experiment 3: Statistical Comparison of Safe Multi-Agent Methods

This experiment compares three methods through randomized trials:
1. Pure Flow Matching (no safety)
2. Non-Game-Theoretic CBF (independent optimization)
3. Game-Theoretic CBF (joint optimization) - Ours

Metrics:
- Safety: collision rate, minimum distance statistics
- Distribution preservation: Wasserstein distance, correction magnitude
- Efficiency: computation time
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import osqp
from scipy import sparse
from scipy.stats import wasserstein_distance
import time
import pickle

from safe_game_flow.flow_matching.model import FlowMatching1D

# ==================== Configuration ====================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_1_PATH = "/home/karl/safe-game-flow/runs/model_west_to_east/checkpoint.pt"
MODEL_2_PATH = "/home/karl/safe-game-flow/runs/model_south_to_north/checkpoint.pt"
DATA_1_PATH = "/home/karl/safe-game-flow/datasets/trajectories_west_to_east.npy"
DATA_2_PATH = "/home/karl/safe-game-flow/datasets/trajectories_south_to_north.npy"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment parameters
N_EXPERIMENTS = 100  # Number of random experiments
H = 50  # Discrete points per trajectory
T = 1.0  # Total time
K = 100  # Integration steps
dt = T / K
D_SAFE = 10.0  # Safety distance

# Random sampling ranges (along road centerlines)
AGENT_1_START_X_RANGE = [-40, -20]  # West side
AGENT_1_START_Y_RANGE = [-3, 3]      # Road width
AGENT_1_GOAL_X_RANGE = [20, 40]      # East side
AGENT_1_GOAL_Y_RANGE = [-3, 3]

AGENT_2_START_X_RANGE = [-3, 3]
AGENT_2_START_Y_RANGE = [-40, -20]   # South side
AGENT_2_GOAL_X_RANGE = [-3, 3]
AGENT_2_GOAL_Y_RANGE = [20, 40]      # North side

# CBF parameters
PHI0 = 1.0
PHI1_SCALE = 1.0

OBSTACLES = []  # No static obstacles for this experiment

print("=" * 80)
print("Experiment 3: Statistical Comparison of Safe Multi-Agent Methods")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Number of experiments: {N_EXPERIMENTS}")
print(f"  Discrete points (H): {H}")
print(f"  Integration steps (K): {K}")
print(f"  Safety distance: {D_SAFE}")
print(f"  Device: {DEVICE}")


# ==================== Load Models ====================
print(f"\nLoading models...")

model_1 = FlowMatching1D(
    in_channels=2, hidden_channels=256, num_blocks=6,
    time_emb_dim=128, kernel_size=3
).to(DEVICE)
ckpt_1 = torch.load(MODEL_1_PATH, map_location=DEVICE)
model_1.load_state_dict(ckpt_1["model"])
model_1.eval()
print(f"✓ Agent 1 model loaded (West→East)")

model_2 = FlowMatching1D(
    in_channels=2, hidden_channels=256, num_blocks=6,
    time_emb_dim=128, kernel_size=3
).to(DEVICE)
ckpt_2 = torch.load(MODEL_2_PATH, map_location=DEVICE)
model_2.load_state_dict(ckpt_2["model"])
model_2.eval()
print(f"✓ Agent 2 model loaded (South→North)")

# Load training data for distribution comparison
training_data_1 = np.load(DATA_1_PATH)  # (N, 2, H)
training_data_2 = np.load(DATA_2_PATH)
print(f"✓ Training data loaded")
print(f"  Agent 1 training samples: {training_data_1.shape[0]}")
print(f"  Agent 2 training samples: {training_data_2.shape[0]}")


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


# ==================== Safety Method Classes ====================
class SafetyMethod:
    """Base class for all safety methods"""
    def __init__(self, name):
        self.name = name
    
    def compute_correction(self, traj_1, traj_2, v1, v2, t, obstacles, d_safe):
        """
        Compute safety correction
        
        Args:
            traj_1, traj_2: (H, 2) current trajectories
            v1, v2: (H, 2) nominal velocity fields
            t: float, current time
            obstacles: list, static obstacles
            d_safe: float, safety distance
            
        Returns:
            u1, u2: (H, 2) correction velocities
            solver_status: bool, True if solver succeeded
        """
        raise NotImplementedError


class PureFlowMatching(SafetyMethod):
    """Method 1: Pure Flow Matching, no safety constraints"""
    def __init__(self):
        super().__init__("Pure Flow Matching")
    
    def compute_correction(self, traj_1, traj_2, v1, v2, t, obstacles, d_safe):
        H_traj = traj_1.shape[0]
        return np.zeros((H_traj, 2)), np.zeros((H_traj, 2)), True


class NonGameCBF(SafetyMethod):
    """Method 2: Non-Game-Theoretic CBF, independent optimization"""
    def __init__(self):
        super().__init__("Non-Game CBF")
    
    def compute_correction(self, traj_1, traj_2, v1, v2, t, obstacles, d_safe):
        # Agent 1: solve independently, treating Agent 2 as obstacle
        u1, status1 = self._solve_single_agent_qp(traj_1, v1, traj_2, obstacles, d_safe, t)
        
        # Agent 2: solve independently, treating Agent 1 as obstacle
        u2, status2 = self._solve_single_agent_qp(traj_2, v2, traj_1, obstacles, d_safe, t)
        
        return u1, u2, (status1 and status2)
    
    def _solve_single_agent_qp(self, my_traj, my_v, other_traj, obstacles, d_safe, t):
        """
        Single agent QP:
        min ||u||^2
        s.t. 
          - Static obstacle constraints
          - Treat other_traj as a static obstacle
        """
        H_traj = my_traj.shape[0]
        A_rows = []
        b_rows = []
        
        # Static obstacles
        for obs in obstacles:
            c = obs['c']
            r_eff = obs['r_eff']
            h_val = h_of(my_traj, c, r_eff)
            a = grad_h_of(my_traj, c)
            phi = np.array([phi_fun(t, h) for h in h_val])
            bi = -np.sum(a * my_v, axis=1) - phi * h_val
            
            for i in range(H_traj):
                row = np.zeros(2 * H_traj, dtype=float)
                row[2*i:2*i+2] = a[i]
                A_rows.append(row)
            b_rows.append(bi)
        
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
            return np.zeros((H_traj, 2)), True
        
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
        # Align with exp2: do not override max_iter (use OSQP default), keep tolerances
        prob.setup(P, q, A_sparse, l, u_bound, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
        res = prob.solve()
        
        if res.info.status != 'solved':
            # Fallback
            u = np.zeros((H_traj, 2))
            for i in range(H_traj):
                if b_other[i] > 0:
                    a_n2 = np.dot(a_other[i], a_other[i]) + 1e-12
                    u[i] = (b_other[i] / a_n2) * a_other[i]
            return u, False
        
        u = res.x.reshape(H_traj, 2)
        return u, True


class GameTheoreticCBF(SafetyMethod):
    """Method 3: Game-Theoretic CBF, joint optimization (Ours)"""
    def __init__(self):
        super().__init__("Game-Theoretic CBF")
    
    def compute_correction(self, traj_1, traj_2, v1, v2, t, obstacles, d_safe):
        H_traj = traj_1.shape[0]
        A_rows = []
        b_rows = []
        
        # Agent 1 static obstacles
        for obs in obstacles:
            c = obs['c']
            r_eff = obs['r_eff']
            h1 = h_of(traj_1, c, r_eff)
            a1 = grad_h_of(traj_1, c)
            phi1 = np.array([phi_fun(t, h) for h in h1])
            b1 = -np.sum(a1 * v1, axis=1) - phi1 * h1
            
            for i in range(H_traj):
                row = np.zeros(4 * H_traj, dtype=float)
                row[2*i:2*i+2] = a1[i]
                A_rows.append(row)
            b_rows.append(b1)
        
        # Agent 2 static obstacles
        for obs in obstacles:
            c = obs['c']
            r_eff = obs['r_eff']
            h2 = h_of(traj_2, c, r_eff)
            a2 = grad_h_of(traj_2, c)
            phi2 = np.array([phi_fun(t, h) for h in h2])
            b2 = -np.sum(a2 * v2, axis=1) - phi2 * h2
            
            for i in range(H_traj):
                row = np.zeros(4 * H_traj, dtype=float)
                row[2*H_traj + 2*i:2*H_traj + 2*i + 2] = a2[i]
                A_rows.append(row)
            b_rows.append(b2)
        
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
            return np.zeros((H_traj, 2)), np.zeros((H_traj, 2)), True
        
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
        # Align with exp2: do not override max_iter (use OSQP default), keep tolerances
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
            return u1, u2, False
        
        u_flat = res.x
        u1 = u_flat[:2*H_traj].reshape(H_traj, 2)
        u2 = u_flat[2*H_traj:].reshape(H_traj, 2)
        
        return u1, u2, True


# ==================== Helper Functions ====================
def sample_random_start_goal():
    """Sample random start and goal positions for both agents"""
    start_1 = np.array([
        np.random.uniform(*AGENT_1_START_X_RANGE),
        np.random.uniform(*AGENT_1_START_Y_RANGE)
    ])
    goal_1 = np.array([
        np.random.uniform(*AGENT_1_GOAL_X_RANGE),
        np.random.uniform(*AGENT_1_GOAL_Y_RANGE)
    ])
    
    start_2 = np.array([
        np.random.uniform(*AGENT_2_START_X_RANGE),
        np.random.uniform(*AGENT_2_START_Y_RANGE)
    ])
    goal_2 = np.array([
        np.random.uniform(*AGENT_2_GOAL_X_RANGE),
        np.random.uniform(*AGENT_2_GOAL_Y_RANGE)
    ])
    
    return start_1, goal_1, start_2, goal_2


def run_single_experiment(method, start_1, goal_1, start_2, goal_2):
    """Run a single experiment with given method and start/goal positions"""
    # Initialize trajectories
    traj_1 = np.linspace(start_1, goal_1, H)
    traj_2 = np.linspace(start_2, goal_2, H)
    
    # Storage for metrics
    min_dists_over_time = []
    mean_dists_over_time = []
    # Match quick_comparison.py: accumulate sum_k sum_i ||u[k,i]||
    total_u1 = 0.0
    total_u2 = 0.0
    # Peak correction intensity (mean over points) over time
    peak_mean_u = 0.0
    comp_times = []
    solver_successes = []
    
    # Main simulation loop
    for k in range(K):
        t = k / K
        
        # Compute nominal velocities
        v1 = velocity_field_1(traj_1, t)
        v2 = velocity_field_2(traj_2, t)
        
        # Compute correction (with timing)
        start_time = time.time()
        u1, u2, solver_success = method.compute_correction(
            traj_1, traj_2, v1, v2, t, OBSTACLES, D_SAFE
        )
        comp_time = time.time() - start_time
        
        # Update trajectories
        traj_1 = traj_1 + dt * (v1 + u1)
        traj_2 = traj_2 + dt * (v2 + u2)
        
        # Compute inter-agent distances
        dist = np.linalg.norm(traj_1 - traj_2, axis=1)
        min_dists_over_time.append(dist.min())
        mean_dists_over_time.append(dist.mean())
        
        # Record metrics
        u1_norm_per_point = np.linalg.norm(u1, axis=1)
        u2_norm_per_point = np.linalg.norm(u2, axis=1)
        total_u1 += float(u1_norm_per_point.sum())
        total_u2 += float(u2_norm_per_point.sum())
        mean_u_t = 0.5 * (float(u1_norm_per_point.mean()) + float(u2_norm_per_point.mean()))
        peak_mean_u = max(peak_mean_u, mean_u_t)
        comp_times.append(comp_time)
        solver_successes.append(solver_success)
    
    # Compute final metrics
    # Safety over the rollout (min across time steps)
    min_dist_over_time = float(min(min_dists_over_time)) if len(min_dists_over_time) else float("nan")
    # Final generated trajectories safety (min along final discrete trajectories)
    final_dist = np.linalg.norm(traj_1 - traj_2, axis=1)
    min_dist_final = float(final_dist.min())
    # Primary safety definition for Experiment 3:
    # collision is judged on the FINAL generated trajectories (not intermediate rollout states)
    collision = min_dist_final < D_SAFE
    # Keep rollout-based collision as a diagnostic
    collision_over_time = min_dist_over_time < D_SAFE
    
    metrics = {
        'collision': collision,
        'collision_over_time': bool(collision_over_time),
        # Safety metrics
        'min_distance_over_time': min_dist_over_time,
        'min_distance_final': min_dist_final,
        'mean_min_distance_over_time': float(np.mean(min_dists_over_time)) if len(min_dists_over_time) else float("nan"),
        'mean_dist_over_time': float(np.mean(mean_dists_over_time)) if len(mean_dists_over_time) else float("nan"),
        # Deviation / distribution preservation
        'total_correction_1': float(total_u1),
        'total_correction_2': float(total_u2),
        'total_correction': float(total_u1 + total_u2),
        'peak_mean_u': float(peak_mean_u),
        'mean_comp_time_ms': np.mean(comp_times) * 1000,
        'solver_success_rate': np.mean(solver_successes) * 100,
        'final_trajectory_1': traj_1,
        'final_trajectory_2': traj_2,
    }
    
    return metrics


# ==================== Run Batch Experiments ====================
print("\n" + "=" * 80)
print("Running batch experiments...")
print("=" * 80)

# Initialize methods
methods = {
    'Pure FM': PureFlowMatching(),
    'Non-Game CBF': NonGameCBF(),
    'Game CBF (Ours)': GameTheoreticCBF(),
}

# Storage for all results
all_results = {name: [] for name in methods.keys()}

# Set random seed for reproducibility
np.random.seed(42)

# Run experiments
for exp_idx in range(N_EXPERIMENTS):
    # Sample random start and goal
    start_1, goal_1, start_2, goal_2 = sample_random_start_goal()
    
    # Run each method
    for method_name, method in methods.items():
        try:
            metrics = run_single_experiment(method, start_1, goal_1, start_2, goal_2)
            all_results[method_name].append(metrics)
        except Exception as e:
            print(f"  Error in {method_name} at experiment {exp_idx}: {e}")
            # Add dummy metrics
            all_results[method_name].append({
                'collision': True,
                'min_distance': 0.0,
                'mean_min_distance': 0.0,
                'mean_dist': 0.0,
                'total_correction_1': 0.0,
                'total_correction_2': 0.0,
                'mean_correction': 0.0,
                'mean_comp_time_ms': 0.0,
                'solver_success_rate': 0.0,
                'final_trajectory_1': np.zeros((H, 2)),
                'final_trajectory_2': np.zeros((H, 2)),
            })
    
    if (exp_idx + 1) % 10 == 0:
        print(f"  Completed {exp_idx + 1}/{N_EXPERIMENTS} experiments")

print("✓ All experiments completed!")


# ==================== Save Results ====================
print("\n" + "-" * 80)
print("Saving results...")
print("-" * 80)

with open(OUTPUT_DIR / "all_results.pkl", 'wb') as f:
    pickle.dump(all_results, f)
print(f"✓ Raw results saved to: {OUTPUT_DIR / 'all_results.pkl'}")

# Save summary statistics
summary_stats = {}
for method_name, results in all_results.items():
    summary_stats[method_name] = {
        'collision_rate': np.mean([r['collision'] for r in results]) * 100,
        # Safety over rollout
        'min_distance_over_time_mean': np.mean([r['min_distance_over_time'] for r in results]),
        'min_distance_over_time_std': np.std([r['min_distance_over_time'] for r in results]),
        'min_distance_over_time_min': np.min([r['min_distance_over_time'] for r in results]),
        # Final trajectory safety (reference)
        'min_distance_final_mean': np.mean([r['min_distance_final'] for r in results]),
        'min_distance_final_std': np.std([r['min_distance_final'] for r in results]),
        'min_distance_final_min': np.min([r['min_distance_final'] for r in results]),
        # Deviation / distribution preservation
        'total_correction_mean': np.mean([r['total_correction'] for r in results]),
        'total_correction_std': np.std([r['total_correction'] for r in results]),
        'peak_mean_u_mean': np.mean([r['peak_mean_u'] for r in results]),
        'peak_mean_u_std': np.std([r['peak_mean_u'] for r in results]),
        'mean_comp_time_ms': np.mean([r['mean_comp_time_ms'] for r in results]),
        'solver_success_rate': np.mean([r['solver_success_rate'] for r in results]),
    }

with open(OUTPUT_DIR / "summary_stats.txt", 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("Experiment 3: Summary Statistics\n")
    f.write("=" * 80 + "\n\n")
    
    for method_name, stats in summary_stats.items():
        f.write(f"\n{method_name}:\n")
        f.write(f"  Collision Rate: {stats['collision_rate']:.2f}%\n")
        f.write(f"  Min Distance over time: {stats['min_distance_over_time_mean']:.3f} ± {stats['min_distance_over_time_std']:.3f}\n")
        f.write(f"  Min Distance over time (global min): {stats['min_distance_over_time_min']:.3f}\n")
        f.write(f"  Min Distance final: {stats['min_distance_final_mean']:.3f} ± {stats['min_distance_final_std']:.3f}\n")
        f.write(f"  Min Distance final (global min): {stats['min_distance_final_min']:.3f}\n")
        f.write(f"  Total Correction: {stats['total_correction_mean']:.2f} ± {stats['total_correction_std']:.2f}\n")
        f.write(f"  Peak mean|u|: {stats['peak_mean_u_mean']:.4f} ± {stats['peak_mean_u_std']:.4f}\n")
        f.write(f"  Comp Time: {stats['mean_comp_time_ms']:.2f} ms/step\n")
        f.write(f"  Solver Success Rate: {stats['solver_success_rate']:.1f}%\n")

print(f"✓ Summary statistics saved to: {OUTPUT_DIR / 'summary_stats.txt'}")

# Print to console
print("\n" + "=" * 80)
print("Summary Statistics:")
print("=" * 80)
for method_name, stats in summary_stats.items():
    print(f"\n{method_name}:")
    print(f"  Collision Rate: {stats['collision_rate']:.2f}%")
    print(f"  Min Distance over time: {stats['min_distance_over_time_mean']:.3f} ± {stats['min_distance_over_time_std']:.3f}")
    print(f"  Min Distance final: {stats['min_distance_final_mean']:.3f} ± {stats['min_distance_final_std']:.3f}")
    print(f"  Total Correction: {stats['total_correction_mean']:.2f} ± {stats['total_correction_std']:.2f}")
    print(f"  Peak mean|u|: {stats['peak_mean_u_mean']:.4f} ± {stats['peak_mean_u_std']:.4f}")
    print(f"  Comp Time: {stats['mean_comp_time_ms']:.2f} ms/step")

print("\n" + "=" * 80)
print("✓ Experiment 3 completed successfully!")
print("=" * 80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print("\nNext step: Run visualization script to generate plots")
