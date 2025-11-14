#!/bin/bash
# Script to train all QM9 targets sequentially

set -e  # Exit on error

echo "=========================================="
echo "Training Allegro on all QM9 targets"
echo "=========================================="
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "allegro-reproduction" ]]; then
    echo "Error: Please activate the allegro-reproduction environment first:"
    echo "  conda activate allegro-reproduction"
    exit 1
fi

# Get the base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$SCRIPT_DIR/../.."

cd "$BASE_DIR"

# Array of targets
TARGETS=("U0" "U" "H" "G")

echo "Targets to train: ${TARGETS[@]}"
echo ""

# Train each target
for TARGET in "${TARGETS[@]}"; do
    echo "=========================================="
    echo "Training target: $TARGET"
    echo "=========================================="
    echo ""

    CONFIG_FILE="models/qm9/config_${TARGET}.yaml"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found: $CONFIG_FILE"
        continue
    fi

    echo "Using config: $CONFIG_FILE"
    echo "Start time: $(date)"
    echo ""

    # Run training
    nequip-train "$CONFIG_FILE"

    EXIT_CODE=$?

    echo ""
    echo "End time: $(date)"
    echo "Exit code: $EXIT_CODE"
    echo ""

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Warning: Training for $TARGET failed with exit code $EXIT_CODE"
        read -p "Continue with next target? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit $EXIT_CODE
        fi
    else
        echo "Training for $TARGET completed successfully!"
    fi

    echo ""
done

echo "=========================================="
echo "All QM9 training completed!"
echo "=========================================="
echo ""
echo "Results saved to: results/qm9/"
echo ""
echo "To analyze results, run:"
echo "  jupyter lab notebooks/results-analysis.ipynb"
echo ""
