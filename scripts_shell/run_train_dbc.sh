#!/bin/bash
# Train DBC model on all datasets
# Usage: ./scripts_shell/run_train_dbc.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Training DBC model..."
echo "====================="

# Train on each dataset with multiple seeds
for DATASET in bom jma cnrm; do
    echo ""
    echo "Dataset: $DATASET"
    echo "-------------------"

    python scripts/train_multiseed.py \
        --dataset "$DATASET" \
        --normalize \
        --seeds 42 123 456 789 1024
done

echo ""
echo "Training complete!"
echo "Results saved to: {BoM,JMA,CNRM}_norm/seed*/results/"
