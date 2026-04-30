#!/bin/bash
# Run complete pipeline: train all models, evaluate, generate figures
# Usage: ./scripts_shell/run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "MJO Probabilistic Correction Pipeline"
echo "=========================================="

# Step 1: Train DBC model
echo ""
echo "Step 1: Training DBC model..."
"$SCRIPT_DIR/run_train_dbc.sh"

# Step 2: Train baselines
echo ""
echo "Step 2: Training baselines..."
"$SCRIPT_DIR/run_train_baselines.sh"

# Step 3: Evaluate
echo ""
echo "Step 3: Evaluating all methods..."
"$SCRIPT_DIR/run_evaluation.sh"

# Step 4: Generate figures
echo ""
echo "Step 4: Generating figures..."
mkdir -p outputs/figures
python plotting/plot_metrics_per_lead.py
python plotting/plot_heatmap_comparison.py

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - Models: */seed*/models/"
echo "  - Results: */seed*/results/"
echo "  - Figures: outputs/figures/"
