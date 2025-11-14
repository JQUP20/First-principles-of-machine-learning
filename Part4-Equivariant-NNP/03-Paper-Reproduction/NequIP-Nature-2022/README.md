# 复现Nature Communications论文：NequIP
# Reproducing Nature Communications Paper: NequIP

**论文标题**: E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials

**作者**: Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger, Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E. Smidt, Boris Kozinsky

**期刊**: Nature Communications 13, 2453 (2022)

**DOI**: [10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5)

**代码仓库**: https://github.com/mir-group/nequip

---

## 📋 论文摘要

NequIP是一个基于E(3)等变图神经网络的原子间势函数，通过保持三维旋转和平移对称性，实现了前所未有的数据效率和预测精度。

**主要贡献**：
1. 提出E(3)等变消息传递神经网络架构
2. 使用不可约表示和张量积操作保持等变性
3. 在小数据集（~1000样本）上达到DFT级别精度
4. 相比之前的模型提高10倍数据效率

---

## 🎯 论文主要结果

### Table 1: MD17数据集上的性能（训练集大小：1000）

| 分子 | 能量MAE (meV) |  | 力MAE (meV/Å) |  |
|------|--------------|--------------|--------------|--------------|
|      | **NequIP** | SchNet | **NequIP** | SchNet |
| Aspirin | **2.9** | 8.5 | **8.8** | 33.0 |
| Benzene | **0.8** | 1.8 | **2.6** | 7.2 |
| Ethanol | **2.4** | 4.3 | **7.2** | 19.0 |
| Malonaldehyde | **2.7** | 4.8 | **6.9** | 18.5 |
| Naphthalene | **2.1** | 5.0 | **5.3** | 17.4 |
| Salicylic acid | **3.1** | 7.2 | **9.2** | 27.3 |
| Toluene | **2.5** | 4.5 | **6.4** | 16.5 |
| Uracil | **2.6** | 4.7 | **7.0** | 18.9 |
| **平均** | **2.4** | 5.1 | **6.7** | 19.7 |

**结果说明**：NequIP在能量和力的预测上都显著优于SchNet，平均提升2-3倍精度。

### Figure 2: 数据效率对比

论文展示了NequIP在不同训练集大小下的性能：
- **50个样本**：NequIP已经达到SchNet用1000个样本的精度
- **1000个样本**：NequIP接近化学精度（1 kcal/mol ≈ 43 meV）
- 数据效率提升约**10-20倍**

### Figure 3: 复杂体系的应用

- **水的相变**：正确预测冰的熔化温度
- **分子动力学**：长时程MD模拟保持稳定性
- **迁移学习**：在新体系上快速适应

---

## 🔧 环境配置

### 系统要求

```bash
- Python >= 3.7
- PyTorch >= 1.11.0
- CUDA >= 11.3 (GPU推荐)
- 内存 >= 16GB
- 存储 >= 50GB (用于数据集)
```

### 安装步骤

#### 1. 创建Conda环境

```bash
conda create -n nequip-paper python=3.9 -y
conda activate nequip-paper

# 安装PyTorch (根据你的CUDA版本调整)
conda install pytorch==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装e3nn
pip install e3nn==0.5.1

# 安装NequIP (使用论文发表时的版本)
pip install nequip==0.5.6

# 其他依赖
pip install ase wandb matplotlib seaborn pandas scipy tqdm pyyaml
```

#### 2. 验证安装

```python
python -c "import torch; import e3nn; import nequip; print('All packages installed successfully!')"
```

---

## 📊 数据集准备

### MD17数据集

MD17包含8个小分子的DFT计算轨迹，每个分子约100,000个构型。

#### 自动下载脚本

```bash
cd data
python ../scripts/download_md17.py --all
```

或手动下载：

```bash
# 单个分子
wget http://www.quantum-machine.org/gdml/data/xyz/md17_aspirin.xyz.gz
gunzip md17_aspirin.xyz.gz
```

#### 数据集信息

| 分子 | 原子数 | 构型数 | 大小 |
|------|--------|--------|------|
| Aspirin | 21 | 211,762 | ~800 MB |
| Benzene | 12 | 627,983 | ~1.2 GB |
| Ethanol | 9 | 555,092 | ~800 MB |
| Malonaldehyde | 9 | 993,237 | ~1.4 GB |
| Naphthalene | 18 | 326,250 | ~950 MB |
| Salicylic acid | 16 | 320,231 | ~820 MB |
| Toluene | 15 | 442,790 | ~1.1 GB |
| Uracil | 12 | 133,770 | ~260 MB |

---

## ⚙️ 模型配置

### 论文中的超参数设置

根据论文的补充材料(Supplementary Information)，NequIP的配置如下：

```yaml
# configs/nequip_paper_aspirin.yaml
# 完全按照Nature Communications论文的配置

# 数据集
root: ./data/md17_aspirin
dataset: ase
dataset_file_name: aspirin.xyz
chemical_symbols:
  - H
  - C
  - O

# 数据划分
n_train: 1000
n_val: 100
dataset_statistics_stride: 1

# 网络架构（论文设置）
num_layers: 5                # L = 5 交互层
l_max: 2                     # lmax = 2
parity: true                 # 包含奇偶宇称
num_features: 64             # F = 64 特征通道数
invariant_layers: 2          # 不变层数
invariant_neurons: 64        # 不变层神经元

# 径向网络
num_basis: 8                 # K = 8 径向基函数
BesselBasis_trainable: true
PolynomialCutoff_p: 6
r_max: 4.0                   # rcut = 4.0 Å

# 归一化
avg_num_neighbors: auto
use_sc: true                 # Self-connection

# 优化器（Adam）
optimizer_name: Adam
optimizer_params:
  amsgrad: false
  betas: [0.9, 0.999]
  eps: 1.0e-8

# 学习率调度
learning_rate: 0.005         # 初始学习率
lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 50
lr_scheduler_factor: 0.5
lr_scheduler_min_lr: 1.0e-6

# 训练参数
batch_size: 5                # 论文使用batch_size=5
max_epochs: 10000
early_stopping_patiences:
  validation_loss: 1000

# 损失函数
loss_coeffs:
  forces:
    - 100                    # λF = 100 (力的权重)
    - PerSpeciesL1Loss
  total_energy:
    - 1                      # λE = 1 (能量权重)
    - PerAtomMSELoss

# 正则化
weight_decay: 0.0            # 无L2正则化

# 梯度裁剪
gradient_clip_val: 10.0

# 输出
model_save_dir: ./results/aspirin
wandb: false
verbose: info
```

### 关键超参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_layers` | 5 | 消息传递层数，论文测试了3-6层 |
| `l_max` | 2 | 最大角动量，包含标量(l=0)、矢量(l=1)、张量(l=2) |
| `num_features` | 64 | 特征通道数，论文测试了32-128 |
| `num_basis` | 8 | 径向基函数数量 |
| `r_max` | 4.0Å | 截断半径，适合小分子 |
| `force_weight` | 100 | 力的损失权重，相对于能量 |
| `batch_size` | 5 | 小批量，适合GPU内存 |

---

## 🚀 运行复现

### 方法1：单个分子训练

```bash
# 训练Aspirin（约需要2-4小时，取决于GPU）
nequip-train configs/nequip_paper_aspirin.yaml

# 监控训练
tensorboard --logdir results/aspirin/
```

### 方法2：自动化训练所有分子

```bash
# 使用提供的自动化脚本
python scripts/train_all_molecules.py \
  --molecules aspirin benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil \
  --n_train 1000 \
  --n_val 100 \
  --gpus 0,1,2,3
```

### 方法3：使用Jupyter Notebook交互式复现

```bash
jupyter notebook notebooks/paper_reproduction.ipynb
```

---

## 📈 评估和结果分析

### 评估单个模型

```bash
# 评估训练好的模型
python scripts/evaluate_model.py \
  --model results/aspirin/deployed_model.pth \
  --dataset data/md17_aspirin/aspirin.xyz \
  --n_test 10000 \
  --output results/aspirin/test_results.json
```

### 生成论文中的图表

```bash
# 生成Table 1的结果对比
python scripts/generate_table1.py \
  --results_dir results/ \
  --output paper_table1.csv

# 生成Figure 2的学习曲线
python scripts/plot_learning_curves.py \
  --results_dir results/ \
  --output paper_figure2.pdf

# 生成数据效率对比图
python scripts/plot_data_efficiency.py \
  --molecules aspirin ethanol \
  --train_sizes 50 100 200 500 1000 2000 5000 \
  --output paper_data_efficiency.pdf
```

---

## 🔬 论文关键实验复现

### 实验1：Table 1复现（MD17基准测试）

**目标**：在8个分子上训练NequIP，使用1000个训练样本。

```bash
# 自动运行所有8个分子
bash scripts/run_table1_experiments.sh
```

**预期结果**（能量MAE，单位meV）：
- Aspirin: 2.9 ± 0.2
- Benzene: 0.8 ± 0.1
- Ethanol: 2.4 ± 0.2
- 其他分子见上表

**验证标准**：
- ✓ 能量MAE误差 < 10%
- ✓ 力MAE误差 < 15%
- ✓ 相对SchNet提升 > 2倍

### 实验2：数据效率测试（Figure 2）

**目标**：测试不同训练集大小的性能。

```bash
# 在Aspirin上测试不同数据量
python scripts/data_efficiency_experiment.py \
  --molecule aspirin \
  --train_sizes 50 100 200 500 1000 2000 5000 \
  --n_trials 3 \
  --output results/data_efficiency/
```

**预期趋势**：
- 50样本：能量MAE ~10-15 meV
- 100样本：能量MAE ~5-8 meV
- 1000样本：能量MAE ~2-3 meV

### 实验3：消融实验（Supplementary）

**测试不同超参数的影响**：

```bash
# 测试不同的l_max
python scripts/ablation_study.py \
  --molecule aspirin \
  --parameter l_max \
  --values 0 1 2 3 \
  --n_trials 3

# 测试不同的层数
python scripts/ablation_study.py \
  --molecule aspirin \
  --parameter num_layers \
  --values 3 4 5 6 \
  --n_trials 3

# 测试不同的特征维度
python scripts/ablation_study.py \
  --molecule aspirin \
  --parameter num_features \
  --values 32 64 128 \
  --n_trials 3
```

### 实验4：迁移学习

**测试预训练模型在新分子上的迁移能力**：

```bash
# 在Aspirin上预训练
nequip-train configs/pretrain_aspirin.yaml

# 在Salicylic acid上微调（结构相似）
python scripts/transfer_learning.py \
  --pretrained_model results/aspirin/best_model.pth \
  --target_molecule salicylic_acid \
  --n_finetune 100
```

---

## 📊 结果可视化

### 1. 预测精度散点图

```python
import matplotlib.pyplot as plt
import numpy as np

# 加载结果
results = np.load('results/aspirin/predictions.npz')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 能量
axes[0].scatter(results['energy_true'], results['energy_pred'], alpha=0.5, s=10)
axes[0].plot([results['energy_true'].min(), results['energy_true'].max()],
             [results['energy_true'].min(), results['energy_true'].max()], 'r--')
axes[0].set_xlabel('DFT Energy (eV)')
axes[0].set_ylabel('NequIP Energy (eV)')
axes[0].set_title('Energy Prediction')

# 力
axes[1].scatter(results['forces_true'], results['forces_pred'], alpha=0.3, s=1)
axes[1].plot([results['forces_true'].min(), results['forces_true'].max()],
             [results['forces_true'].min(), results['forces_true'].max()], 'r--')
axes[1].set_xlabel('DFT Forces (eV/Å)')
axes[1].set_ylabel('NequIP Forces (eV/Å)')
axes[1].set_title('Force Prediction')

plt.tight_layout()
plt.savefig('prediction_parity.pdf')
```

### 2. 学习曲线对比

```python
import pandas as pd
import seaborn as sns

# 读取训练历史
df = pd.read_csv('results/aspirin/metrics_epoch.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 能量MAE
axes[0].plot(df['epoch'], df['train_e_mae'] * 1000, label='Train')
axes[0].plot(df['epoch'], df['val_e_mae'] * 1000, label='Validation')
axes[0].axhline(2.9, color='r', linestyle='--', label='Paper Result')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Energy MAE (meV)')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 力MAE
axes[1].plot(df['epoch'], df['train_f_mae'] * 1000, label='Train')
axes[1].plot(df['epoch'], df['val_f_mae'] * 1000, label='Validation')
axes[1].axhline(8.8, color='r', linestyle='--', label='Paper Result')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Force MAE (meV/Å)')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.pdf')
```

### 3. 数据效率曲线（Figure 2复现）

```python
import matplotlib.pyplot as plt
import numpy as np

train_sizes = [50, 100, 200, 500, 1000, 2000, 5000]

# NequIP结果
nequip_energy = [12.5, 7.2, 4.8, 3.5, 2.9, 2.5, 2.2]
nequip_forces = [35.0, 20.0, 14.0, 10.5, 8.8, 7.5, 6.8]

# SchNet结果（用于对比）
schnet_energy = [45.0, 28.0, 18.0, 12.0, 8.5, 6.8, 5.5]
schnet_forces = [120.0, 80.0, 55.0, 40.0, 33.0, 28.0, 24.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 能量
axes[0].loglog(train_sizes, nequip_energy, 'o-', linewidth=2, markersize=8, label='NequIP')
axes[0].loglog(train_sizes, schnet_energy, 's-', linewidth=2, markersize=8, label='SchNet')
axes[0].axhline(1, color='gray', linestyle='--', alpha=0.5, label='Chemical accuracy')
axes[0].set_xlabel('Training Set Size', fontsize=12)
axes[0].set_ylabel('Energy MAE (meV)', fontsize=12)
axes[0].set_title('Data Efficiency: Energy', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# 力
axes[1].loglog(train_sizes, nequip_forces, 'o-', linewidth=2, markersize=8, label='NequIP')
axes[1].loglog(train_sizes, schnet_forces, 's-', linewidth=2, markersize=8, label='SchNet')
axes[1].set_xlabel('Training Set Size', fontsize=12)
axes[1].set_ylabel('Force MAE (meV/Å)', fontsize=12)
axes[1].set_title('Data Efficiency: Forces', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_data_efficiency.pdf', dpi=300)
```

---

## 🧪 复现清单

### 必须完成（核心结果）

- [ ] **Table 1**: 在MD17的8个分子上达到论文精度
  - [ ] Aspirin: E_MAE < 3.5 meV, F_MAE < 10 meV
  - [ ] Benzene: E_MAE < 1.5 meV, F_MAE < 4 meV
  - [ ] Ethanol: E_MAE < 3.0 meV, F_MAE < 9 meV
  - [ ] Malonaldehyde: E_MAE < 3.5 meV, F_MAE < 9 meV
  - [ ] Naphthalene: E_MAE < 3.0 meV, F_MAE < 7 meV
  - [ ] Salicylic acid: E_MAE < 4.0 meV, F_MAE < 11 meV
  - [ ] Toluene: E_MAE < 3.5 meV, F_MAE < 8 meV
  - [ ] Uracil: E_MAE < 3.5 meV, F_MAE < 9 meV

- [ ] **Figure 2**: 数据效率曲线
  - [ ] 训练集大小: 50, 100, 200, 500, 1000样本
  - [ ] 展示NequIP优于SchNet的数据效率

### 可选（补充材料）

- [ ] **消融实验**: 测试l_max、num_layers、num_features的影响
- [ ] **迁移学习**: 预训练+微调实验
- [ ] **MD模拟**: 使用NequIP势函数进行长时程MD
- [ ] **计算效率**: 测量训练时间和推理速度

---

## 💡 调试技巧

### 常见问题

#### 1. 训练loss不下降

**可能原因**：
- 学习率太高或太低
- 数据归一化问题
- 梯度爆炸

**解决方法**：
```yaml
# 降低学习率
learning_rate: 0.001  # 从0.005降低

# 增加梯度裁剪
gradient_clip_val: 5.0  # 从10.0降低

# 检查数据统计
dataset_statistics_stride: 1
```

#### 2. 验证loss震荡

**解决方法**：
```yaml
# 增加batch size
batch_size: 10  # 从5增加

# 使用更稳定的学习率调度
lr_scheduler_patience: 100  # 增加patience
```

#### 3. 内存不足

**解决方法**：
```yaml
# 减小batch size
batch_size: 1

# 减小模型
num_features: 32  # 从64降低
l_max: 1  # 从2降低
```

#### 4. 训练速度慢

**优化方法**：
```bash
# 使用多GPU
CUDA_VISIBLE_DEVICES=0,1 nequip-train config.yaml --distributed

# 使用混合精度
# 在config中添加：
default_dtype: float32
allow_tf32: true  # A100 GPU
```

---

## 📚 补充资源

### 论文相关

- **主论文**: [Nature Communications](https://doi.org/10.1038/s41467-022-29939-5)
- **补充材料**: [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-29939-5/MediaObjects/41467_2022_29939_MOESM1_ESM.pdf)
- **代码**: [GitHub - mir-group/nequip](https://github.com/mir-group/nequip)

### 相关论文

1. **e3nn**: Geiger & Smidt, "e3nn: Euclidean neural networks", arXiv:2207.09453
2. **SchNet**: Schütt et al., "SchNet: A continuous-filter convolutional neural network", NeurIPS 2017
3. **DimeNet++**: Klicpera et al., "Fast and uncertainty-aware directional message passing", NeurIPS 2020

### 教程和文档

- [NequIP官方文档](https://nequip.readthedocs.io/)
- [e3nn教程](https://docs.e3nn.org/en/stable/guide/convolution.html)
- [MD17数据集](http://www.quantum-machine.org/gdml/)

---

## 🎯 预期时间表

| 任务 | 预计时间 | GPU需求 |
|------|----------|---------|
| 环境配置 | 1-2小时 | N/A |
| 数据下载 | 2-4小时 | N/A |
| 单分子训练 | 2-4小时 | 1x V100/A100 |
| 8分子全部训练 | 1-2天 | 4x V100/A100 |
| 数据效率实验 | 1-2天 | 2x V100/A100 |
| 消融实验 | 2-3天 | 4x V100/A100 |
| 结果分析和可视化 | 4-8小时 | N/A |
| **总计** | **5-7天** | **多GPU推荐** |

---

## ✅ 成功标准

### 定量指标

复现被认为成功如果满足：

1. **精度达标**（相对论文误差<15%）：
   - MD17平均能量MAE: 2.0-3.0 meV
   - MD17平均力MAE: 6.0-8.0 meV/Å

2. **数据效率**（相对SchNet）：
   - 50样本时提升 > 3倍
   - 1000样本时提升 > 2倍

3. **训练稳定性**：
   - 验证loss平稳下降
   - 无梯度爆炸或消失

### 定性指标

- 学习曲线趋势与论文一致
- MD模拟保持稳定性（能量守恒）
- 可视化结果合理

---

## 📞 获取帮助

如果复现遇到问题：

1. **检查Issues**: [NequIP GitHub Issues](https://github.com/mir-group/nequip/issues)
2. **讨论区**: [GitHub Discussions](https://github.com/mir-group/nequip/discussions)
3. **邮件**: 联系论文作者（见论文通讯作者信息）

---

## 📝 引用

如果这个复现对你的研究有帮助，请引用原论文：

```bibtex
@article{batzner2022nequip,
  title={E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials},
  author={Batzner, Simon and Musaelian, Albert and Sun, Lixin and Geiger, Mario and Mailoa, Jonathan P and Kornbluth, Mordechai and Molinari, Nicola and Smidt, Tess E and Kozinsky, Boris},
  journal={Nature communications},
  volume={13},
  number={1},
  pages={2453},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

---

**祝复现顺利！Good luck with the reproduction! 🚀**

如有任何问题，请提交Issue或Pull Request。
