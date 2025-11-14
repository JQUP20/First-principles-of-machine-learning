# 快速开始指南 | Quick Start Guide

本指南帮助你在**30分钟**内开始复现NequIP论文。

## 第一步：环境准备（10分钟）

```bash
# 1. 创建conda环境
conda create -n nequip-repro python=3.9 -y
conda activate nequip-repro

# 2. 安装PyTorch
conda install pytorch==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 安装NequIP和依赖
pip install e3nn==0.5.1
pip install nequip==0.5.6
pip install ase wandb matplotlib seaborn pandas scipy tqdm pyyaml

# 4. 验证安装
python -c "import torch, e3nn, nequip; print('✓ All packages installed')"
```

## 第二步：下载数据（10分钟）

```bash
# 进入项目目录
cd Part4-Equivariant-NNP/03-Paper-Reproduction/NequIP-Nature-2022

# 下载单个分子（快速测试）
python scripts/download_md17.py --molecules ethanol

# 或下载所有分子（推荐，但需要更长时间）
python scripts/download_md17.py --all
```

## 第三步：快速训练测试（10分钟）

### 选项A：使用小数据集快速测试

```bash
# 在Ethanol上用50个样本快速测试（~5分钟）
nequip-train <<EOF
root: ./data/md17_ethanol
dataset: ase
dataset_file_name: ethanol.xyz
chemical_symbols: [H, C, O]
n_train: 50
n_val: 10
num_layers: 3
l_max: 1
num_features: 32
num_basis: 8
r_max: 4.0
batch_size: 5
max_epochs: 100
learning_rate: 0.005
loss_coeffs:
  forces: [100, PerSpeciesL1Loss]
  total_energy: [1, PerAtomMSELoss]
model_save_dir: ./quick_test
wandb: false
EOF
```

### 选项B：使用论文完整配置

```bash
# 在Aspirin上用论文配置训练（~2-4小时）
python scripts/train_all_molecules.py \
  --molecules aspirin \
  --gpus 0 \
  --n-train 1000 \
  --n-val 100
```

## 第四步：检查结果

训练完成后，检查结果：

```bash
# 查看训练曲线
ls results/aspirin/  # 或 quick_test/

# 如果使用wandb，打开链接查看
# 否则查看日志文件
tail -f results/aspirin/training.log
```

---

## 完整复现流程（约7天）

如果你想完整复现论文中的所有结果：

### 1. 下载所有数据（~4小时）

```bash
python scripts/download_md17.py --all --verify
```

### 2. 训练所有8个分子（~2-3天）

```bash
# 使用4个GPU并行训练
python scripts/train_all_molecules.py \
  --molecules all \
  --gpus 0,1,2,3 \
  --n-train 1000 \
  --n-val 100
```

### 3. 评估并生成Table 1（~2小时）

```bash
python scripts/evaluate_and_generate_table1.py \
  --results_dir results \
  --n_test 10000 \
  --output paper_table1
```

### 4. 数据效率实验（~2-3天）

```bash
# 测试不同训练集大小
for n_train in 50 100 200 500 1000 2000; do
  python scripts/train_all_molecules.py \
    --molecules aspirin ethanol \
    --gpus 0,1 \
    --n-train $n_train
done
```

---

## 常见问题

### Q: 训练很慢怎么办？

**A**: 尝试以下方法：
1. 减小模型：`num_features: 32`, `l_max: 1`
2. 减小batch size：`batch_size: 1`
3. 使用更好的GPU（V100/A100）

### Q: 内存不足？

**A**:
```yaml
# 在配置中添加：
batch_size: 1
num_features: 32
l_max: 1
```

### Q: 如何查看训练进度？

**A**:
```bash
# 实时查看日志
tail -f results/aspirin/training.log

# 查看metrics
cat results/aspirin/metrics_epoch.csv

# 使用TensorBoard（如果配置了）
tensorboard --logdir results/
```

### Q: 结果与论文不一致？

**A**: 检查：
1. 数据集版本是否正确
2. 超参数是否完全一致
3. 训练是否收敛（至少1000 epochs）
4. Random seed的影响（运行多次取平均）

---

## 最小可复现示例

如果你只想验证代码能运行：

```python
# minimal_test.py
from nequip.data import AtomicDataDict
from nequip.nn import NequIPModel
import torch

# 创建最小模型
model = NequIPModel.from_config({
    'num_layers': 2,
    'l_max': 1,
    'num_features': 16,
    'num_basis': 8,
    'r_max': 4.0,
    'chemical_symbols': ['H', 'C', 'O']
})

# 测试前向传播
batch = {
    AtomicDataDict.POSITIONS_KEY: torch.randn(10, 3),
    AtomicDataDict.ATOM_TYPE_KEY: torch.randint(0, 3, (10,)),
    AtomicDataDict.EDGE_INDEX_KEY: torch.randint(0, 10, (2, 20))
}

output = model(batch)
print(f"✓ Model works! Output energy shape: {output['total_energy'].shape}")
```

运行：
```bash
python minimal_test.py
```

---

## 下一步

- 📖 阅读完整的 [README.md](README.md)
- 💻 查看 [Jupyter Notebook教程](notebooks/paper_reproduction.ipynb)
- 📊 查看论文 [补充材料](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-29939-5/MediaObjects/41467_2022_29939_MOESM1_ESM.pdf)

---

**祝复现顺利！Happy reproducing! 🎉**
