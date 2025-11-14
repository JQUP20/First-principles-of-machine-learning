# Allegro Nature Paper Reproduction

## Paper Information

**Title:** Learning local equivariant representations for large-scale atomistic dynamics

**Authors:** Albert Musaelian, Simon Batzner, Anders Johansson, Lixin Sun, Cameron J. Owen, Mordechai Kornbluth, and Boris Kozinsky

**Published:** Nature Communications 14, Article 579 (2023)

**DOI:** https://doi.org/10.1038/s41467-023-36329-y

**arXiv:** https://arxiv.org/abs/2204.05249

**Code Repository:** https://github.com/mir-group/allegro

## Paper Abstract

The long-standing goal of accurately and efficiently describing the potential energy surface of molecules and materials remains a central challenge in computational chemistry and materials science. While atom-centered message passing neural networks (MPNNs) have shown remarkable accuracy, their information propagation has limited the accessible length-scales. Local methods, conversely, scale to large simulations but have suffered from inferior accuracy.

This work introduces **Allegro**, a strictly local equivariant deep neural network interatomic potential architecture that simultaneously exhibits excellent accuracy and scalability. Allegro represents a many-body potential using iterated tensor products of learned equivariant representations without atom-centered message passing.

## Key Contributions

1. **Novel Architecture**: Allegro uses local equivariant representations based on tensor products instead of message passing
2. **State-of-the-art Performance**: Outperforms existing models on QM9 and revMD17 benchmarks
3. **Scalability**: Demonstrated with simulations of 100 million atoms
4. **Single Layer Performance**: A single tensor product layer outperforms deep MPNNs and transformers on QM9

## Benchmark Results to Reproduce

### 1. QM9 Dataset

The QM9 dataset contains quantum mechanical properties for ~134k organic molecules with up to 9 heavy atoms (C, O, N, F).

**Targets (Energy Properties):**
- U₀: Internal energy at T = 0 K
- U: Internal energy at T = 298.15 K
- H: Enthalpy at T = 298.15 K
- G: Free energy at T = 298.15 K

**Allegro Results (MAE in meV):**
- U₀: 4 meV
- U: 4 meV
- H: 4 meV
- G: 5 meV

### 2. revMD17 Dataset

The revised MD17 dataset contains molecular dynamics trajectories from ab initio simulations.

**Metrics:**
- Energy MAE: 3.84 (±0.08) meV
- Force MAE: 12.98 (±0.17) meV/Å

**Evaluation Temperature:** 300K

### 3. Scalability Demonstrations

- Large-scale MD simulations (100M atoms)
- Performance comparison with other methods (NequIP, SchNet, DimeNet++)

## Technical Details

### Model Architecture

- **Core Principle**: Local equivariant representations via tensor products
- **Equivariance**: E(3) equivariance (rotation and translation invariance)
- **Key Innovation**: Avoids message passing while maintaining accuracy
- **Layers**: Single tensor product layer achieves state-of-the-art results

### Datasets

1. **QM9**
   - Size: ~134,000 molecules
   - Properties: Multiple quantum mechanical properties
   - Split: 110,000 train / 10,000 val / 13,885 test (standard split)

2. **revMD17**
   - Molecules: Aspirin, Benzene, Ethanol, Malonaldehyde, Naphthalene, Salicylic acid, Toluene, Uracil
   - Properties: Energies and forces from DFT calculations
   - Split: Various sizes per molecule

## Reproduction Plan

### Phase 1: Environment Setup ✓
- [x] Create project structure
- [ ] Clone Allegro repository
- [ ] Install dependencies (PyTorch, NequIP, Allegro)
- [ ] Set up conda environment
- [ ] Verify GPU availability

### Phase 2: Data Preparation
- [ ] Download QM9 dataset
- [ ] Preprocess QM9 data
- [ ] Download revMD17 dataset
- [ ] Preprocess revMD17 data
- [ ] Verify data integrity

### Phase 3: QM9 Reproduction
- [ ] Configure training for U₀ target
- [ ] Configure training for U target
- [ ] Configure training for H target
- [ ] Configure training for G target
- [ ] Run training for all targets
- [ ] Evaluate and record results
- [ ] Compare with paper results

### Phase 4: revMD17 Reproduction
- [ ] Configure training for all molecules
- [ ] Run training and evaluation
- [ ] Record energy and force MAE
- [ ] Compare with paper results

### Phase 5: Analysis and Documentation
- [ ] Generate comparison tables
- [ ] Create visualization plots
- [ ] Document findings
- [ ] Write reproduction report

## Directory Structure

```
Part2-Allegro-Reproduction/
├── README.md                    # This file
├── docs/                        # Documentation and notes
│   ├── paper-summary.md         # Detailed paper summary
│   ├── methodology.md           # Technical methodology
│   └── reproduction-log.md      # Reproduction log
├── scripts/                     # Training and evaluation scripts
│   ├── setup/                   # Setup scripts
│   ├── training/                # Training scripts
│   ├── evaluation/              # Evaluation scripts
│   └── analysis/                # Analysis scripts
├── data/                        # Datasets (gitignored)
│   ├── qm9/                     # QM9 dataset
│   └── revmd17/                 # revMD17 dataset
├── models/                      # Trained models and configs
│   ├── qm9/                     # QM9 models
│   └── revmd17/                 # revMD17 models
├── results/                     # Results and analysis
│   ├── qm9/                     # QM9 results
│   ├── revmd17/                 # revMD17 results
│   └── comparisons/             # Comparison with paper
└── notebooks/                   # Jupyter notebooks for analysis
    ├── data-exploration.ipynb
    ├── results-analysis.ipynb
    └── visualization.ipynb
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- conda or mamba

### Setup Steps

```bash
# Create conda environment
conda create -n allegro python=3.9
conda activate allegro

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install NequIP and Allegro
pip install nequip-allegro

# Install additional dependencies
pip install numpy scipy matplotlib pandas jupyter ase
```

## Usage

### 1. Data Preparation
```bash
cd scripts/setup
python download_qm9.py
python download_revmd17.py
```

### 2. Training
```bash
# QM9 training
cd scripts/training
python train_qm9.py --target U0

# revMD17 training
python train_revmd17.py --molecule aspirin
```

### 3. Evaluation
```bash
cd scripts/evaluation
python evaluate_qm9.py --model path/to/model
python evaluate_revmd17.py --model path/to/model
```

## References

1. **Main Paper:**
   Musaelian, A., Batzner, S., Johansson, A. et al. Learning local equivariant representations for large-scale atomistic dynamics. Nat Commun 14, 579 (2023). https://doi.org/10.1038/s41467-023-36329-y

2. **NequIP (Predecessor):**
   Batzner, S., Musaelian, A., Sun, L. et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nat Commun 13, 2453 (2022). https://doi.org/10.1038/s41467-022-29939-5

3. **QM9 Dataset:**
   Ramakrishnan, R., Dral, P., Rupp, M. et al. Quantum chemistry structures and properties of 134 kilo molecules. Sci Data 1, 140022 (2014). https://doi.org/10.1038/sdata.2014.22

4. **MD17 Dataset:**
   Chmiela, S., Tkatchenko, A., Sauceda, H. et al. Machine learning of accurate energy-conserving molecular force fields. Sci Adv 3, e1603015 (2017).

## License

This reproduction study is for educational purposes. The Allegro code is licensed under MIT License.

## Contact

For questions or issues with this reproduction, please open an issue in the repository.

---

**Last Updated:** 2025-11-14
