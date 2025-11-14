#!/usr/bin/env python3
"""
Download and prepare revMD17 (sGDML) dataset for Allegro training.

The revMD17 dataset contains molecular dynamics trajectories from ab initio simulations.
"""

import os
import sys
from pathlib import Path


def download_revmd17(data_dir: Path):
    """
    Download revMD17 dataset.

    NequIP has built-in support for sGDML/revMD17 datasets.
    This function provides information about the dataset and prepares directories.

    Args:
        data_dir: Path to store the dataset
    """
    print("=" * 60)
    print("revMD17 (sGDML) Dataset Download")
    print("=" * 60)
    print()

    # Create directories
    revmd17_dir = data_dir / "revmd17"
    revmd17_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {revmd17_dir}")
    print()

    print("revMD17 Dataset Information:")
    print("-" * 60)
    print("Source: Ab initio MD simulations (DFT with PBE functional)")
    print("Properties: Energies and atomic forces")
    print("Temperature: 300K - 500K depending on molecule")
    print()

    molecules = {
        "aspirin": {
            "formula": "C9H8O4",
            "atoms": 21,
            "description": "Acetylsalicylic acid"
        },
        "benzene": {
            "formula": "C6H6",
            "atoms": 12,
            "description": "Benzene"
        },
        "ethanol": {
            "formula": "C2H6O",
            "atoms": 9,
            "description": "Ethyl alcohol"
        },
        "malonaldehyde": {
            "formula": "C3H4O2",
            "atoms": 9,
            "description": "Propanedial"
        },
        "naphthalene": {
            "formula": "C10H8",
            "atoms": 18,
            "description": "Bicyclic aromatic hydrocarbon"
        },
        "salicylic_acid": {
            "formula": "C7H6O3",
            "atoms": 16,
            "description": "2-Hydroxybenzoic acid"
        },
        "toluene": {
            "formula": "C7H8",
            "atoms": 15,
            "description": "Methylbenzene"
        },
        "uracil": {
            "formula": "C4H4N2O2",
            "atoms": 12,
            "description": "Pyrimidine nucleobase"
        }
    }

    print("Available molecules:")
    print("-" * 60)
    for name, info in molecules.items():
        print(f"{name:20s} {info['formula']:10s} {info['atoms']:3d} atoms - {info['description']}")
    print()

    print("Paper Results to Reproduce:")
    print("-" * 60)
    print("Average performance on revMD17:")
    print("  - Energy MAE: 3.84 ± 0.08 meV")
    print("  - Force MAE: 12.98 ± 0.17 meV/Å")
    print("  - Evaluation temperature: 300K")
    print()

    print("Training set sizes (typical):")
    print("  - Small: 50-200 structures")
    print("  - Medium: 500-1000 structures")
    print("  - Large: 5000+ structures")
    print()

    print("Download methods:")
    print("-" * 60)
    print("1. Automatic (via NequIP):")
    print("   NequIP's sGDML_CCSD_DataModule will download datasets automatically")
    print("   when you run training for the first time.")
    print("   Example config:")
    print("     data:")
    print("       _target_: nequip.data.datamodule.sGDML_CCSD_DataModule")
    print("       dataset: aspirin")
    print()
    print("2. Manual download:")
    print("   Visit: http://quantum-machine.org/gdml/")
    print("   Download individual molecule datasets")
    print(f"   Extract to: {revmd17_dir}/")
    print()

    print("Creating molecule directories...")
    for mol_name in molecules.keys():
        mol_dir = revmd17_dir / mol_name
        mol_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {mol_dir}")
    print()

    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print("The revMD17 datasets will be automatically downloaded when you run:")
    print("  nequip-train configs/revmd17_*.yaml")
    print()
    print("Or you can manually download and place the data in:")
    print(f"  {revmd17_dir}/<molecule_name>/")
    print()


if __name__ == "__main__":
    # Get the data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "../../data"
    data_dir = data_dir.resolve()

    download_revmd17(data_dir)
