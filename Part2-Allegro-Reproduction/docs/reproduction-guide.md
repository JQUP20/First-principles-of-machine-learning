# Allegro Paper Reproduction Guide

This guide provides step-by-step instructions for reproducing the results from the Allegro Nature Communications paper (2023).

## Prerequisites

- Linux/macOS system (or Windows with WSL)
- CUDA-capable GPU (recommended, but CPU training is possible)
- Anaconda or Miniconda installed
- At least 50GB free disk space (for datasets and models)
- At least 16GB RAM (32GB+ recommended)

## Step 1: Environment Setup

### 1.1 Create Conda Environment

```bash
cd Part2-Allegro-Reproduction
conda env create -f environment-allegro.yml
conda activate allegro-reproduction
```

Or use the installation script:

```bash
cd scripts/setup
chmod +x install_environment.sh
./install_environment.sh
```

### 1.2 Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check NequIP
python -c "import nequip; print(f'NequIP: {nequip.__version__}')"

# Check Allegro
python -c "import allegro; print(f'Allegro: {allegro.__version__}')"
```

## Step 2: Dataset Preparation

### 2.1 QM9 Dataset

The QM9 dataset will be downloaded automatically by NequIP during the first training run. Alternatively, you can prepare it manually:

```bash
cd scripts/setup
python download_qm9.py
```

**Note:** The first time you run training on QM9, NequIP will download the dataset (~3GB) automatically. This may take some time depending on your internet connection.

### 2.2 revMD17 Dataset

Similarly, the revMD17 datasets will be downloaded automatically:

```bash
cd scripts/setup
python download_revmd17.py
```

## Step 3: QM9 Reproduction

### 3.1 Training on QM9 Targets

Train models for each of the four energy targets (U₀, U, H, G):

```bash
# Activate environment
conda activate allegro-reproduction

# Navigate to the reproduction directory
cd Part2-Allegro-Reproduction

# Train U₀ (Internal Energy at 0K)
nequip-train models/qm9/config_U0.yaml

# Train U (Internal Energy at 298.15K)
nequip-train models/qm9/config_U.yaml

# Train H (Enthalpy at 298.15K)
nequip-train models/qm9/config_H.yaml

# Train G (Free Energy at 298.15K)
nequip-train models/qm9/config_G.yaml
```

### 3.2 Expected Training Time

- **Per target:** 2-8 hours on a modern GPU (e.g., RTX 3090, A100)
- **Total for all 4 targets:** ~10-30 hours depending on hardware

### 3.3 Monitor Training

Training logs and checkpoints will be saved to:
- `results/qm9/<target>/<date>/<time>/`

You can monitor training with TensorBoard:

```bash
tensorboard --logdir results/qm9/
```

### 3.4 Evaluate Models

After training, evaluate the best model on the test set:

```bash
# The test evaluation is automatically run at the end of training
# Check the final output for test metrics

# To manually evaluate a checkpoint:
nequip-evaluate --checkpoint results/qm9/U0/<date>/<time>/checkpoints/best_model.ckpt
```

### 3.5 Expected Results

According to the paper, Allegro should achieve:

| Target | Expected MAE | Units |
|--------|--------------|-------|
| U₀     | 4 meV        | meV   |
| U      | 4 meV        | meV   |
| H      | 4 meV        | meV   |
| G      | 5 meV        | meV   |

## Step 4: revMD17 Reproduction

### 4.1 Training on revMD17

Train on the aspirin molecule (example):

```bash
nequip-train models/revmd17/config_aspirin.yaml
```

To train on other molecules, create similar config files by copying `config_aspirin.yaml` and changing:
- `dataset:` parameter (benzene, ethanol, malonaldehyde, naphthalene, salicylic_acid, toluene, uracil)
- `chemical_symbols:` list (based on molecule composition)
- Output directory in `hydra.run.dir`

### 4.2 Expected Training Time

- **Per molecule:** 4-12 hours depending on molecule size
- **All 8 molecules:** ~40-80 hours total

### 4.3 Expected Results

Average performance across all molecules:
- **Energy MAE:** 3.84 ± 0.08 meV
- **Force MAE:** 12.98 ± 0.17 meV/Å

## Step 5: Results Analysis

### 5.1 Collect Results

After training completes, results are stored in:
- `results/qm9/<target>/<date>/<time>/`
- `results/revmd17/<molecule>/<date>/<time>/`

### 5.2 Compare with Paper

Create comparison tables and plots using the provided notebook:

```bash
jupyter lab notebooks/results-analysis.ipynb
```

### 5.3 Document Results

Update the reproduction log with your results:

```bash
nano docs/reproduction-log.md
```

## Step 6: Advanced Options

### 6.1 Hyperparameter Tuning

To tune hyperparameters, modify the config files:

**Key hyperparameters to tune:**
- `num_scalar_features`: 64, 128, 256
- `num_tensor_features`: 32, 64, 128
- `l_max`: 1, 2, 3 (higher = more accurate but slower)
- `num_layers`: 1, 2, 3
- `learning_rate`: 0.0001 - 0.01

### 6.2 Multi-GPU Training

To use multiple GPUs, modify the trainer config:

```yaml
trainer:
  devices: 2  # or [0, 1] for specific GPUs
  strategy: ddp
```

### 6.3 Mixed Precision Training

For faster training with minimal accuracy loss:

```yaml
trainer:
  precision: 16-mixed
```

### 6.4 Using Pre-trained Models

If pre-trained models are available:

```bash
# Download pre-trained model
wget https://example.com/allegro_qm9_u0.ckpt

# Evaluate
nequip-evaluate --checkpoint allegro_qm9_u0.ckpt --data path/to/test/data
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
- Reduce `batch_size` in config
- Reduce `num_scalar_features` or `num_tensor_features`
- Use gradient accumulation

### Issue: Dataset Download Fails

**Solution:**
- Check internet connection
- Manually download datasets and place in `data/` directory
- Check firewall settings

### Issue: Training is Too Slow

**Solution:**
- Verify GPU is being used: `torch.cuda.is_available()`
- Reduce model size (fewer features, layers)
- Use mixed precision training
- Enable PyTorch compilation (PyTorch >= 2.6.0)

### Issue: Results Don't Match Paper

**Possible causes:**
- Different random seeds
- Different hyperparameters
- Different dataset versions
- Insufficient training epochs

**Solutions:**
- Train for more epochs
- Tune hyperparameters
- Verify dataset versions
- Run multiple seeds and average

## Tips for Successful Reproduction

1. **Start Small:** Test with a small subset of data first to verify setup
2. **Monitor Closely:** Watch training curves for convergence
3. **Save Regularly:** Ensure checkpoints are being saved
4. **Document Everything:** Keep detailed notes of hyperparameters and results
5. **Be Patient:** Full reproduction may take several days of compute time

## Resources

- **Allegro Paper:** https://www.nature.com/articles/s41467-023-36329-y
- **Allegro Code:** https://github.com/mir-group/allegro
- **NequIP Docs:** https://nequip.readthedocs.io/
- **QM9 Dataset:** https://figshare.com/collections/978904
- **MD17 Dataset:** http://quantum-machine.org/gdml/

## Citation

If you use this reproduction in your work, please cite the original paper:

```bibtex
@article{musaelian2023allegro,
  title={Learning local equivariant representations for large-scale atomistic dynamics},
  author={Musaelian, Albert and Batzner, Simon and Johansson, Anders and Sun, Lixin and Owen, Cameron J and Kornbluth, Mordechai and Kozinsky, Boris},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={579},
  year={2023},
  publisher={Nature Publishing Group}
}
```

---

**Good luck with your reproduction!**
