#!/usr/bin/env python3
"""
Visualization script for Experiment 3 results

Generates plots comparing the three methods:
1. Collision rate comparison
2. Minimum distance box plots
3. Computation time comparison
4. Safety vs Distribution preservation scatter
5. Example trajectory comparisons
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import wasserstein_distance
import os

OUTPUT_DIR = Path(__file__).parent / "results"
DATA_1_PATH = "/home/karl/safe-game-flow/datasets/trajectories_west_to_east.npy"
DATA_2_PATH = "/home/karl/safe-game-flow/datasets/trajectories_south_to_north.npy"

print("=" * 80)
print("Visualizing Experiment 3 Results")
print("=" * 80)

# Load results
print("\nLoading results...")
with open(OUTPUT_DIR / "all_results.pkl", 'rb') as f:
    all_results = pickle.load(f)

# Load training data for distribution comparison
training_data_1 = np.load(DATA_1_PATH)  # (N, 2, H)
training_data_2 = np.load(DATA_2_PATH)
print(f"✓ Results and training data loaded")

# Safety threshold (override if you ran experiments with different D_SAFE)
# Example:
#   D_SAFE=10 python visualize_results.py
D_SAFE = float(os.environ.get("D_SAFE", "8.0"))

# ==================== Extract Metrics ====================
method_names = list(all_results.keys())
n_methods = len(method_names)

# Safety metrics
collision_rates = {}
min_dist_over_time = {}
min_dist_final = {}

# Distribution metrics
total_corrections = {}
peak_mean_u = {}
wasserstein_dists_1 = {}
wasserstein_dists_2 = {}

# Efficiency metrics
comp_times = {}

for method_name in method_names:
    results = all_results[method_name]
    
    # Safety
    collision_rates[method_name] = np.mean([r['collision'] for r in results]) * 100
    # New schema (preferred): min_distance_over_time / min_distance_final
    # Backward compatibility: fall back to old keys if needed
    min_dist_over_time[method_name] = [
        float(r.get("min_distance_over_time", r.get("min_distance", np.nan))) for r in results
    ]
    min_dist_final[method_name] = [
        float(r.get("min_distance_final", np.nan)) for r in results
    ]
    
    # Distribution preservation
    total_corrections[method_name] = [
        float(r.get("total_correction", r.get("total_correction_1", 0.0) + r.get("total_correction_2", 0.0)))
        for r in results
    ]
    peak_mean_u[method_name] = [
        float(r.get("peak_mean_u", np.nan)) for r in results
    ]
    
    # Compute Wasserstein distance for each experiment
    w_dists_1 = []
    w_dists_2 = []
    for r in results:
        # Compare endpoint positions
        final_1 = r['final_trajectory_1'][-1]  # (2,)
        final_2 = r['final_trajectory_2'][-1]  # (2,)
        
        # Get training endpoints
        train_endpoints_1 = training_data_1[:, :, -1]  # (N, 2)
        train_endpoints_2 = training_data_2[:, :, -1]  # (N, 2)
        
        # Wasserstein distance in X and Y separately
        w1_x = wasserstein_distance([final_1[0]], train_endpoints_1[:, 0])
        w1_y = wasserstein_distance([final_1[1]], train_endpoints_1[:, 1])
        w2_x = wasserstein_distance([final_2[0]], train_endpoints_2[:, 0])
        w2_y = wasserstein_distance([final_2[1]], train_endpoints_2[:, 1])
        
        w_dists_1.append(w1_x + w1_y)
        w_dists_2.append(w2_x + w2_y)
    
    wasserstein_dists_1[method_name] = w_dists_1
    wasserstein_dists_2[method_name] = w_dists_2
    
    # Efficiency
    comp_times[method_name] = [r['mean_comp_time_ms'] for r in results]

print(f"✓ Metrics extracted")


# ==================== Create Visualizations ====================
print("\nGenerating visualizations (each figure saved separately)...")

# Color scheme
colors = {
    'Pure FM': '#E74C3C',  # Red
    'Non-Game CBF': '#F39C12',  # Orange
    'Game CBF (Ours)': '#27AE60',  # Green
}


def _color_for(name: str) -> str:
    return colors.get(name, "#4C72B0")


def _save(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {path}")
    plt.close(fig)


def _boxplot(ax, data_by_method, title, ylabel, add_threshold=True):
    bp = ax.boxplot(data_by_method, labels=method_names, patch_artist=True, showfliers=True, notch=True)
    for patch, name in zip(bp["boxes"], method_names):
        patch.set_facecolor(_color_for(name))
        patch.set_alpha(0.7)
    if add_threshold:
        ax.axhline(D_SAFE, color="red", linestyle="--", linewidth=2.0, label=f"d_safe={D_SAFE}", zorder=0)
        ax.fill_between([-0.5, n_methods + 0.5], 0, D_SAFE, color="red", alpha=0.08)
        ax.legend(fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xticklabels(method_names, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)


def _annotate_means(ax, values_by_method, title="Mean values"):
    """Add a small text box listing mean values for each method."""
    lines = [title]
    for name, vals in zip(method_names, values_by_method):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean_val = float("nan")
        else:
            mean_val = float(np.mean(vals))
        lines.append(f"{name}: {mean_val:.3f}")
    ax.text(
        0.98,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


# 1) Collision rate
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(n_methods)
collision_vals = [collision_rates[name] for name in method_names]
bars = ax.bar(x_pos, collision_vals, color=[_color_for(n) for n in method_names], alpha=0.85,
              edgecolor="black", linewidth=1.0)
for bar, val in zip(bars, collision_vals):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Safety: Collision Rate (final trajectories)", fontsize=13, fontweight="bold")
ax.set_ylabel("Collision Rate (%)", fontsize=12, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(method_names, rotation=15, ha="right")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_ylim(0, max(collision_vals) * 1.2 if max(collision_vals) > 0 else 5)
_save(fig, "01_collision_rate.png")

# 2) Min distance over time
fig, ax = plt.subplots(figsize=(8, 5))
_boxplot(
    ax,
    [min_dist_over_time[name] for name in method_names],
    title="Safety: Min Distance over Time (rollout)",
    ylabel="Min distance (m)",
    add_threshold=True,
)
_save(fig, "02_min_distance_over_time.png")

# 3) Min distance final
fig, ax = plt.subplots(figsize=(8, 5))
_boxplot(
    ax,
    [min_dist_final[name] for name in method_names],
    title="Safety: Min Distance on Final Trajectories",
    ylabel="Min distance (m)",
    add_threshold=True,
)
# Plot 3: make safety threshold dashed line thinner (per user request)
for line in ax.lines:
    # the dashed threshold line is the only constant horizontal line we added
    if getattr(line, "get_linestyle", lambda: "")() == "--":
        line.set_linewidth(1.0)
_save(fig, "03_min_distance_final.png")

# 4) Computation time
fig, ax = plt.subplots(figsize=(8, 5))
mean_times = [np.mean(comp_times[name]) for name in method_names]
std_times = [np.std(comp_times[name]) for name in method_names]
bars = ax.bar(x_pos, mean_times, yerr=std_times, color=[_color_for(n) for n in method_names],
              alpha=0.85, edgecolor="black", linewidth=1.0, capsize=5)
for i, (bar, val) in enumerate(zip(bars, mean_times)):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + std_times[i] + 0.005, f"{val:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Efficiency: Computation Time", fontsize=13, fontweight="bold")
ax.set_ylabel("Time per step (ms)", fontsize=12, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(method_names, rotation=15, ha="right")
ax.grid(axis="y", alpha=0.3, linestyle="--")
_save(fig, "04_computation_time.png")

# 5) Total correction
fig, ax = plt.subplots(figsize=(8, 5))
_boxplot(
    ax,
    [total_corrections[name] for name in method_names],
    title="Distribution Preservation: Total Correction",
    ylabel="Total correction (Σ||u|| over time & points)",
    add_threshold=False,
)
ax.text(0.02, 0.98, "Lower is better\n(less deviation from flow)",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
_annotate_means(ax, [total_corrections[name] for name in method_names], title="Mean total correction")
_save(fig, "05_total_correction.png")

# 6) Peak mean |u|
fig, ax = plt.subplots(figsize=(8, 5))
_boxplot(
    ax,
    [peak_mean_u[name] for name in method_names],
    title="Distribution Preservation: Peak Correction Intensity",
    ylabel="Peak mean ||u|| (over time)",
    add_threshold=False,
)
ax.text(0.02, 0.98, "Lower is better\n(less conservative / smoother)",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
_annotate_means(ax, [peak_mean_u[name] for name in method_names], title="Mean peak mean||u||")
_save(fig, "06_peak_mean_u.png")

# 7) Wasserstein distance
fig, ax = plt.subplots(figsize=(8, 5))
# Compute distribution distance to Pure FM generated trajectories (not to training data)
if "Pure FM" not in all_results:
    raise KeyError("Expected 'Pure FM' in results for distribution-distance baseline.")

def _endpoints(results, key):
    arr = np.array([r[key][-1] for r in results], dtype=float)  # (N, 2)
    return arr

pure_end_1 = _endpoints(all_results["Pure FM"], "final_trajectory_1")
pure_end_2 = _endpoints(all_results["Pure FM"], "final_trajectory_2")

wd_to_pure = []
for name in method_names:
    end_1 = _endpoints(all_results[name], "final_trajectory_1")
    end_2 = _endpoints(all_results[name], "final_trajectory_2")
    w = 0.0
    # Agent 1 endpoints (x,y)
    w += wasserstein_distance(end_1[:, 0], pure_end_1[:, 0])
    w += wasserstein_distance(end_1[:, 1], pure_end_1[:, 1])
    # Agent 2 endpoints (x,y)
    w += wasserstein_distance(end_2[:, 0], pure_end_2[:, 0])
    w += wasserstein_distance(end_2[:, 1], pure_end_2[:, 1])
    wd_to_pure.append(float(w))

bars = ax.bar(x_pos, wd_to_pure, color=[_color_for(n) for n in method_names],
              alpha=0.85, edgecolor="black", linewidth=1.0)
for bar, val in zip(bars, wd_to_pure):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() , f"{val:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_title("Distribution Distance to Pure FM (Endpoints, Wasserstein)", fontsize=13, fontweight="bold")
ax.set_ylabel("Wasserstein distance to Pure FM (lower is better)", fontsize=12, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(method_names, rotation=15, ha="right")
ax.grid(axis="y", alpha=0.3, linestyle="--")
_save(fig, "07_wasserstein_distance.png")

# 8) Trade-off scatter
fig, ax = plt.subplots(figsize=(8, 6))
for name in method_names:
    safety_score = 100 - collision_rates[name]
    dist_score = 100 / (1 + np.mean(total_corrections[name]))
    peak_u_mean_val = np.nanmean(peak_mean_u[name])
    ax.scatter(safety_score, dist_score, s=260, color=_color_for(name), alpha=0.85,
               edgecolor="black", linewidth=1.5, zorder=10)
    ax.text(safety_score + 1.5, dist_score, f"{name}\npeak|u|={peak_u_mean_val:.3f}", fontsize=9, fontweight="bold")
ax.set_title("Trade-off: Safety vs Distribution Preservation", fontsize=13, fontweight="bold")
ax.set_xlabel("Safety score (100 - collision rate) →", fontsize=12, fontweight="bold")
ax.set_ylabel("Preservation score (1 / (1 + total correction)) →", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(-5, 105)
ax.set_ylim(bottom=0)
ax.text(0.98, 0.02, "Ideal: Top-Right Corner", transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7))
_save(fig, "08_tradeoff.png")

# 9) Example trajectories (one row, 3 methods)
exp_idx = 0
fig, axs = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
if n_methods == 1:
    axs = [axs]
for i, method_name in enumerate(method_names):
    ax = axs[i]
    result = all_results[method_name][exp_idx]
    traj_1 = result["final_trajectory_1"]
    traj_2 = result["final_trajectory_2"]
    ax.plot(traj_1[:, 0], traj_1[:, 1], "-", color="blue", linewidth=2.5, alpha=0.7, label="Agent 1 (W→E)")
    ax.plot(traj_2[:, 0], traj_2[:, 1], "-", color="red", linewidth=2.5, alpha=0.7, label="Agent 2 (S→N)")
    ax.scatter(traj_1[0, 0], traj_1[0, 1], s=90, color="blue", marker="o", edgecolor="black", linewidth=1.5, zorder=10)
    ax.scatter(traj_1[-1, 0], traj_1[-1, 1], s=90, color="blue", marker="s", edgecolor="black", linewidth=1.5, zorder=10)
    ax.scatter(traj_2[0, 0], traj_2[0, 1], s=90, color="red", marker="o", edgecolor="black", linewidth=1.5, zorder=10)
    ax.scatter(traj_2[-1, 0], traj_2[-1, 1], s=90, color="red", marker="s", edgecolor="black", linewidth=1.5, zorder=10)

    min_fin = float(result.get("min_distance_final", np.nan))
    peak_u = float(result.get("peak_mean_u", np.nan))
    txt = (
        f"{'⚠️ COLLISION' if result['collision'] else '✓ Safe'}\n"
        f"Final min: {min_fin:.2f}m\n"
        f"peak mean|u|: {peak_u:.3f}"
    )
    color = "red" if result["collision"] else "green"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10, va="top", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.25))
    ax.set_title(method_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    if i == 0:
        ax.legend(fontsize=9, loc="lower right")
fig.suptitle(f"Example final trajectories (experiment #{exp_idx})", fontsize=14, fontweight="bold")
_save(fig, "09_example_trajectories.png")

print("\n" + "=" * 80)
print("✓ Visualization completed successfully!")
print("=" * 80)
