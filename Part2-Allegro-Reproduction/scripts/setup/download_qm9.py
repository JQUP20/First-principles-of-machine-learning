#!/usr/bin/env python3
"""
Download and prepare QM9 dataset for Allegro training.

The QM9 dataset contains quantum mechanical properties for ~134k organic molecules
with up to 9 heavy atoms (C, O, N, F).
"""

import os
import sys
from pathlib import Path


def download_qm9(data_dir: Path):
    """
    Download QM9 dataset.

    NequIP has built-in support for QM9 dataset through its data loaders.
    This function provides information about the dataset and prepares directories.

    Args:
        data_dir: Path to store the dataset
    """
    print("=" * 60)
    print("QM9 Dataset Download")
    print("=" * 60)
    print()

    # Create directories
    qm9_dir = data_dir / "qm9"
    qm9_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {qm9_dir}")
    print()

    print("QM9 Dataset Information:")
    print("-" * 60)
    print("Size: ~134,000 molecules")
    print("Atoms: Up to 9 heavy atoms (C, O, N, F, plus H)")
    print("Properties: 13 quantum mechanical properties")
    print("Source: DFT calculations at B3LYP/6-31G(2df,p) level")
    print()

    print("Properties included:")
    properties = [
        "A: Rotational constant (GHz)",
        "B: Rotational constant (GHz)",
        "C: Rotational constant (GHz)",
        "μ: Dipole moment (Debye)",
        "α: Isotropic polarizability (Bohr³)",
        "ε_HOMO: HOMO energy (Hartree)",
        "ε_LUMO: LUMO energy (Hartree)",
        "Δε: Gap (Hartree)",
        "⟨R²⟩: Electronic spatial extent (Bohr²)",
        "ZPVE: Zero point vibrational energy (Hartree)",
        "U₀: Internal energy at 0K (Hartree)",
        "U: Internal energy at 298.15K (Hartree)",
        "H: Enthalpy at 298.15K (Hartree)",
        "G: Free energy at 298.15K (Hartree)",
        "c_v: Heat capacity at 298.15K (cal/mol·K)"
    ]
    for prop in properties:
        print(f"  - {prop}")
    print()

    print("Targets for reproduction (energy properties):")
    print("  - U₀: Internal energy at 0K")
    print("  - U: Internal energy at 298.15K")
    print("  - H: Enthalpy at 298.15K")
    print("  - G: Free energy at 298.15K")
    print()

    print("Standard split:")
    print("  - Training: 110,000 molecules")
    print("  - Validation: 10,000 molecules")
    print("  - Test: 13,885 molecules")
    print()

    print("Download methods:")
    print("-" * 60)
    print("1. Automatic (via NequIP):")
    print("   NequIP's QM9DataModule will download the dataset automatically")
    print("   when you run training for the first time.")
    print()
    print("2. Manual download:")
    print("   Visit: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904")
    print("   Download: dsgdb9nsd.xyz.tar.bz2")
    print(f"   Extract to: {qm9_dir}/")
    print()

    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print("The QM9 dataset will be automatically downloaded when you run:")
    print("  nequip-train configs/qm9_*.yaml")
    print()
    print("Or you can manually download and place the data in:")
    print(f"  {qm9_dir}/")
    print()


if __name__ == "__main__":
    # Get the data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "../../data"
    data_dir = data_dir.resolve()

    download_qm9(data_dir)
