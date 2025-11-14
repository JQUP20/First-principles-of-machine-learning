#!/bin/bash
# Script to download QM9 and revMD17 datasets

set -e  # Exit on error

echo "=========================================="
echo "Downloading Datasets for Allegro Reproduction"
echo "=========================================="
echo ""

# Create data directories
mkdir -p ../../data/qm9
mkdir -p ../../data/revmd17
mkdir -p ../../data/raw

echo "Step 1: Downloading QM9 dataset..."
echo "QM9 dataset will be downloaded automatically by NequIP during first use."
echo "Manual download option available at: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904"
echo ""

echo "Step 2: Downloading revMD17 dataset..."
echo "revMD17 (sGDML) dataset information:"
echo "  - This dataset is available through the sGDML package"
echo "  - NequIP has built-in support for sGDML datasets"
echo "  - Datasets will be downloaded automatically during training"
echo ""
echo "Available molecules in revMD17:"
echo "  - Aspirin"
echo "  - Benzene"
echo "  - Ethanol"
echo "  - Malonaldehyde"
echo "  - Naphthalene"
echo "  - Salicylic acid"
echo "  - Toluene"
echo "  - Uracil"
echo ""

echo "Step 3: Alternative manual download..."
read -p "Do you want to manually download datasets now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Manual download instructions:"
    echo ""
    echo "For QM9:"
    echo "  1. Visit: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904"
    echo "  2. Download dsgdb9nsd.xyz.tar.bz2"
    echo "  3. Extract to data/qm9/"
    echo ""
    echo "For revMD17/sGDML:"
    echo "  1. Visit: http://quantum-machine.org/gdml/"
    echo "  2. Download desired molecules"
    echo "  3. Extract to data/revmd17/"
    echo ""
    echo "Or use the NequIP built-in data loaders (recommended)."
else
    echo "Datasets will be downloaded automatically during training."
fi

echo ""
echo "=========================================="
echo "Dataset setup information displayed!"
echo "=========================================="
echo ""
echo "Data directories created at:"
echo "  - ../../data/qm9/"
echo "  - ../../data/revmd17/"
echo "  - ../../data/raw/"
echo ""
echo "The NequIP/Allegro framework will automatically download datasets"
echo "when you run training with the appropriate config files."
echo ""
