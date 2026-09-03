#!/bin/bash
# Run all experiments for Master's thesis
# This script executes both experiments and saves all results

set -e  # Exit on error

echo "=========================================================================="
echo "           Master's Thesis Experiments: Safe Trajectory Generation       "
echo "=========================================================================="
echo ""
echo "This script will run:"
echo "  1. Experiment 1: Single Agent Obstacle Avoidance"
echo "  2. Experiment 2: Intersection Multi-Agent Game-Theoretic Interaction"
echo "  4. Experiment 4: 4-UAV Ring Game (PyBullet)"
echo ""
echo "=========================================================================="
echo ""

# Check if models exist
echo "Checking prerequisites..."
echo "--------------------------------------------------------------------------"

MODEL_1="/home/karl/safe-game-flow/runs/model_west_to_east/checkpoint.pt"
MODEL_2="/home/karl/safe-game-flow/runs/model_south_to_north/checkpoint.pt"

if [ ! -f "$MODEL_1" ]; then
    echo "ERROR: Model 1 (West→East) not found at: $MODEL_1"
    echo "Please train the models first using: ./scripts/train_two_models.sh"
    exit 1
fi

if [ ! -f "$MODEL_2" ]; then
    echo "ERROR: Model 2 (South→North) not found at: $MODEL_2"
    echo "Please train the models first using: ./scripts/train_two_models.sh"
    exit 1
fi

echo "✓ Model 1 (West→East) found"
echo "✓ Model 2 (South→North) found"
echo ""

# Experiment 1
echo "=========================================================================="
echo "EXPERIMENT 1: Single Agent Obstacle Avoidance"
echo "=========================================================================="
echo ""
cd experiments/exp1_single_agent_obstacle_avoidance
python single_agent_safe_generation.py
cd ../..
echo ""
echo "✓ Experiment 1 completed"
echo ""

# Experiment 2
echo "=========================================================================="
echo "EXPERIMENT 2: Intersection Multi-Agent Game-Theoretic Interaction"
echo "=========================================================================="
echo ""
cd experiments/exp2_intersection_multi_agent
python intersection_multi_agent.py
cd ../..
echo ""
echo "✓ Experiment 2 completed"
echo ""

# Experiment 4 (optional: no GUI for headless servers)
echo "=========================================================================="
echo "EXPERIMENT 4: 4-UAV Ring Game (Safe Flow + PyBullet)"
echo "=========================================================================="
echo ""
cd experiments/exp4_swarm_ring_game_pybullet
python run_experiment.py || echo "Warning: Experiment 4 failed"
cd ../..
echo ""
echo "✓ Experiment 4 completed (or skipped on failure)"
echo ""

# Summary
echo "=========================================================================="
echo "                         ALL EXPERIMENTS COMPLETED                        "
echo "=========================================================================="
echo ""
echo "Results saved to:"
echo ""
echo "Experiment 1 (Single Agent):"
echo "  📁 experiments/exp1_single_agent_obstacle_avoidance/results/"
echo "     - comparison_original_vs_safe.png"
echo "     - safety_metrics.png"
echo "     - *.npy data files"
echo ""
echo "Experiment 2 (Multi-Agent):"
echo "  📁 experiments/exp2_intersection_multi_agent/results/"
echo "     - intersection_multi_agent_results.png"
echo "     - *.npy data files"
echo ""
echo "Experiment 4 (Ring Game Swarm):"
echo "  📁 experiments/exp4_swarm_ring_game_pybullet/results/"
echo "     - swarm_ring_flight.gif"
echo "     - trajectories_3d.png"
echo "     - safety_metrics.png"
echo ""
echo "=========================================================================="
echo ""
echo "Next steps:"
echo "  • Review generated figures in results/ directories"
echo "  • Analyze .npy data files for quantitative results"
echo "  • Customize experiments by editing the Python scripts"
echo "  • See experiments/README.md for detailed documentation"
echo ""
echo "=========================================================================="
