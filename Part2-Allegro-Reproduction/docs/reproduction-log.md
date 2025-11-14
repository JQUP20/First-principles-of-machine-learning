# Allegro Reproduction Log

## Overview

This document tracks the progress of reproducing the results from the Allegro Nature Communications paper (2023).

**Paper:** Learning local equivariant representations for large-scale atomistic dynamics

**Start Date:** 2025-11-14

**Goal:** Reproduce QM9 and revMD17 benchmark results reported in the paper

---

## Timeline

### 2025-11-14: Project Initialization

#### Setup Tasks Completed ✓
- [x] Created project directory structure
- [x] Written comprehensive README
- [x] Documented paper summary
- [x] Created reproduction log (this file)

#### Next Steps
- [ ] Clone Allegro repository
- [ ] Set up conda environment
- [ ] Install dependencies
- [ ] Verify GPU availability
- [ ] Download datasets

---

## Environment Information

### System Specifications
- **OS:** Linux
- **Python Version:** TBD
- **CUDA Version:** TBD
- **GPU:** TBD
- **PyTorch Version:** TBD

### Installed Packages
(To be updated after environment setup)

```
# Package list will be added here after installation
```

---

## Datasets

### QM9 Dataset
- **Status:** Not yet downloaded
- **Size:** TBD
- **Location:** `data/qm9/`
- **Download Date:** TBD
- **MD5 Checksum:** TBD

### revMD17 Dataset
- **Status:** Not yet downloaded
- **Size:** TBD
- **Location:** `data/revmd17/`
- **Download Date:** TBD
- **MD5 Checksum:** TBD

---

## Reproduction Results

### QM9 Benchmark

#### Target: U₀ (Internal Energy at 0K)
- **Paper Result:** 4 meV
- **Our Result:** TBD
- **Status:** Not started
- **Training Time:** TBD
- **Model Config:** TBD
- **Notes:**

#### Target: U (Internal Energy at 298.15K)
- **Paper Result:** 4 meV
- **Our Result:** TBD
- **Status:** Not started
- **Training Time:** TBD
- **Model Config:** TBD
- **Notes:**

#### Target: H (Enthalpy at 298.15K)
- **Paper Result:** 4 meV
- **Our Result:** TBD
- **Status:** Not started
- **Training Time:** TBD
- **Model Config:** TBD
- **Notes:**

#### Target: G (Free Energy at 298.15K)
- **Paper Result:** 5 meV
- **Our Result:** TBD
- **Status:** Not started
- **Training Time:** TBD
- **Model Config:** TBD
- **Notes:**

### revMD17 Benchmark

#### Overall Results
- **Paper Energy MAE:** 3.84 ± 0.08 meV
- **Our Energy MAE:** TBD
- **Paper Force MAE:** 12.98 ± 0.17 meV/Å
- **Our Force MAE:** TBD
- **Status:** Not started

#### Per-Molecule Results

##### Aspirin
- **Energy MAE (Paper):** TBD
- **Energy MAE (Ours):** TBD
- **Force MAE (Paper):** TBD
- **Force MAE (Ours):** TBD
- **Status:** Not started

##### Benzene
- **Energy MAE (Paper):** TBD
- **Energy MAE (Ours):** TBD
- **Force MAE (Paper):** TBD
- **Force MAE (Ours):** TBD
- **Status:** Not started

##### Ethanol
- **Energy MAE (Paper):** TBD
- **Energy MAE (Ours):** TBD
- **Force MAE (Paper):** TBD
- **Force MAE (Ours):** TBD
- **Status:** Not started

##### Malonaldehyde
- **Energy MAE (Paper):** TBD
- **Energy MAE (Ours):** TBD
- **Force MAE (Paper):** TBD
- **Force MAE (Ours):** TBD
- **Status:** Not started

##### Naphthalene
- **Energy MAE (Paper):** TBD
- **Energy MAE (Ours):** TBD
- **Force MAE (Paper):** TBD
- **Force MAE (Ours):** TBD
- **Status:** Not started

##### Salicylic Acid
- **Energy MAE (Paper):** TBD
- **Energy MAE (Ours):** TBD
- **Force MAE (Paper):** TBD
- **Force MAE (Ours):** TBD
- **Status:** Not started

##### Toluene
- **Energy MAE (Paper):** TBD
- **Energy MAE (Ours):** TBD
- **Force MAE (Paper):** TBD
- **Force MAE (Ours):** TBD
- **Status:** Not started

##### Uracil
- **Energy MAE (Paper):** TBD
- **Energy MAE (Ours):** TBD
- **Force MAE (Paper):** TBD
- **Force MAE (Ours):** TBD
- **Status:** Not started

---

## Issues and Challenges

### Encountered Issues

(To be documented as they arise)

### Solutions

(To be documented)

---

## Observations and Insights

### Training Observations

(To be filled during training)

### Model Behavior

(To be filled during evaluation)

### Differences from Paper

(To be documented if results differ from paper)

---

## Hyperparameters Used

### QM9 Models

```yaml
# To be filled with actual hyperparameters used
```

### revMD17 Models

```yaml
# To be filled with actual hyperparameters used
```

---

## Computational Resources

### Training Resources
- **Hardware:** TBD
- **Total GPU Hours:** TBD
- **Cost Estimate:** TBD

### Storage Requirements
- **Models:** TBD
- **Datasets:** TBD
- **Results:** TBD
- **Total:** TBD

---

## Comparison with Paper

### Summary Table

| Metric | Paper | Ours | Difference | % Error |
|--------|-------|------|------------|---------|
| QM9 U₀ | 4 meV | TBD | TBD | TBD |
| QM9 U | 4 meV | TBD | TBD | TBD |
| QM9 H | 4 meV | TBD | TBD | TBD |
| QM9 G | 5 meV | TBD | TBD | TBD |
| revMD17 Energy | 3.84 meV | TBD | TBD | TBD |
| revMD17 Force | 12.98 meV/Å | TBD | TBD | TBD |

---

## Conclusions

(To be filled after completing reproduction)

---

## Appendix

### Useful Commands

```bash
# Training
nequip-train config.yaml

# Evaluation
nequip-evaluate --model model.pth --dataset test.xyz

# Deployment
nequip-deploy model.pth deployed_model.pth
```

### Configuration Files

Links to configuration files used:
- QM9: `models/qm9/config.yaml`
- revMD17: `models/revmd17/config.yaml`

---

**Last Updated:** 2025-11-14
