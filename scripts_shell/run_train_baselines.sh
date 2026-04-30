#!/bin/bash
# Train all baseline methods on all datasets
# Usage: ./scripts_shell/run_train_baselines.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

DATASETS=("BoM" "JMA" "CNRM")

echo "Training baseline methods..."
echo "============================"

# Deterministic baselines
echo ""
echo "=== Deterministic Baselines ==="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "Dataset: $DATASET"
    echo "-------------------"

    echo "Training Kim..."
    python baselines/deterministic/kim.py --dataset "$DATASET" --seed 42

    echo "Training Silini..."
    python baselines/deterministic/silini.py --dataset "$DATASET" --seed 42

    echo "Training UAR..."
    python baselines/deterministic/uar.py --dataset "$DATASET" --seed 42
done

# Probabilistic baselines
echo ""
echo "=== Probabilistic Baselines ==="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "Dataset: $DATASET"
    echo "-------------------"

    echo "Training EMOS..."
    python baselines/probabilistic/emos.py --dataset "$DATASET"

    echo "Training BMA..."
    python baselines/probabilistic/bma.py --dataset "$DATASET"
done

echo ""
echo "Baseline training complete!"
