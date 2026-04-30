#!/bin/bash
# Evaluate all methods and generate metrics
# Usage: ./scripts_shell/run_evaluation.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Evaluating all methods..."
echo "========================="

# Compute metrics
python scripts/evaluate.py

# Generate tables
echo ""
echo "Generating comparison tables..."
python scripts/generate_tables.py

echo ""
echo "Evaluation complete!"
