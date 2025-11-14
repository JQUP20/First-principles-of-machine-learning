#!/bin/bash
#
# 自动运行Table 1的所有实验
# Automatically run all Table 1 experiments
#
# Usage:
#   bash run_table1_experiments.sh
#   bash run_table1_experiments.sh --gpus 0,1,2,3
#

set -e  # 遇到错误立即退出

# 默认GPU设置
GPUS="0,1,2,3"
N_TRAIN=1000
N_VAL=100

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --n-train)
            N_TRAIN="$2"
            shift 2
            ;;
        --n-val)
            N_VAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpus 0,1,2,3] [--n-train 1000] [--n-val 100]"
            exit 1
            ;;
    esac
done

echo "======================================================================================================"
echo "NequIP Table 1 Experiments"
echo "======================================================================================================"
echo "Training set size: $N_TRAIN"
echo "Validation set size: $N_VAL"
echo "GPUs: $GPUS"
echo ""
echo "This will train NequIP models on all 8 MD17 molecules with the exact hyperparameters from the paper."
echo "Estimated time: 1-2 days with 4x V100 GPUs"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# 记录开始时间
START_TIME=$(date +%s)

echo ""
echo "======================================================================================================"
echo "Step 1: Verify data availability"
echo "======================================================================================================"

MOLECULES=("aspirin" "benzene" "ethanol" "malonaldehyde" "naphthalene" "salicylic_acid" "toluene" "uracil")

for mol in "${MOLECULES[@]}"; do
    DATA_FILE="../data/md17_${mol}/${mol}.xyz"
    if [ ! -f "$DATA_FILE" ]; then
        echo "✗ Data missing for $mol: $DATA_FILE"
        echo "Please run: python scripts/download_md17.py --molecules $mol"
        exit 1
    else
        echo "✓ Found $mol data"
    fi
done

echo ""
echo "======================================================================================================"
echo "Step 2: Train all molecules"
echo "======================================================================================================"

python train_all_molecules.py \
    --molecules all \
    --gpus "$GPUS" \
    --n-train "$N_TRAIN" \
    --n-val "$N_VAL"

echo ""
echo "======================================================================================================"
echo "Step 3: Evaluate all models and generate Table 1"
echo "======================================================================================================"

python evaluate_and_generate_table1.py \
    --results_dir ../results \
    --n_test 10000 \
    --output ../results/table1_reproduction

echo ""
echo "======================================================================================================"
echo "Step 4: Generate plots"
echo "======================================================================================================"

# 检查是否有matplotlib
if python -c "import matplotlib" 2>/dev/null; then
    echo "Generating visualization plots..."

    # 学习曲线
    cat > plot_all_learning_curves.py << 'EOF'
import matplotlib.pyplot as plt
import pandas as pd
import os

molecules = ["aspirin", "benzene", "ethanol", "malonaldehyde",
             "naphthalene", "salicylic_acid", "toluene", "uracil"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, mol in enumerate(molecules):
    csv_file = f'../results/{mol}/metrics_epoch.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        axes[i].plot(df['epoch'], df['val_e_mae'] * 1000, label='Energy')
        axes[i].plot(df['epoch'], df['val_f_mae'] * 1000, label='Force')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('MAE (meV)')
        axes[i].set_yscale('log')
        axes[i].set_title(mol.replace('_', ' ').title())
        axes[i].legend()
        axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../results/all_learning_curves.pdf', dpi=300)
print("✓ Saved: ../results/all_learning_curves.pdf")
EOF

    python plot_all_learning_curves.py
    rm plot_all_learning_curves.py
else
    echo "Matplotlib not found, skipping plots"
fi

# 计算总时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "======================================================================================================"
echo "Experiments Completed!"
echo "======================================================================================================"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results summary:"
echo "  - Training logs: ../results/*/training.log"
echo "  - Table 1: ../results/table1_reproduction.csv"
echo "  - Comparison plots: ../results/table1_reproduction.pdf"
echo "  - Learning curves: ../results/all_learning_curves.pdf"
echo ""
echo "Next steps:"
echo "  1. Review ../results/table1_reproduction.md for detailed results"
echo "  2. Compare with paper results (should be within 15% error)"
echo "  3. Run data efficiency experiments: python scripts/data_efficiency_experiment.py"
echo ""
echo "======================================================================================================"
