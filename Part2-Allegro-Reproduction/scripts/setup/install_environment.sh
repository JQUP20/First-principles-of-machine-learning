#!/bin/bash
# Installation script for Allegro reproduction environment

set -e  # Exit on error

echo "=========================================="
echo "Allegro Reproduction Environment Setup"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "Step 1: Creating conda environment 'allegro-reproduction'..."
conda env create -f ../../environment-allegro.yml

echo ""
echo "Step 2: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate allegro-reproduction

echo ""
echo "Step 3: Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Step 4: Verifying NequIP installation..."
python -c "import nequip; print(f'NequIP version: {nequip.__version__}')"

echo ""
echo "Step 5: Verifying Allegro installation..."
python -c "import allegro; print(f'Allegro version: {allegro.__version__}')"

echo ""
echo "Step 6: Installing development version of Allegro from cloned repo (optional)..."
read -p "Do you want to install the development version from the cloned repo? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd ../../allegro
    pip install -e .
    cd ../scripts/setup
    echo "Development version installed!"
else
    echo "Skipping development installation."
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate allegro-reproduction"
echo ""
echo "To verify the installation, run:"
echo "  python -c 'import torch, nequip, allegro; print(\"All packages loaded successfully!\")"
echo ""
echo "Next steps:"
echo "  1. Download datasets (run scripts/setup/download_datasets.sh)"
echo "  2. Configure training runs (see configs/ directory)"
echo "  3. Start training (see README.md for instructions)"
echo ""
