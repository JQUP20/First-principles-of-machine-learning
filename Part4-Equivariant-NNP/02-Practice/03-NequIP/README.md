# NequIP等变神经网络势函数实践
# NequIP Equivariant Neural Network Potential Practice

本教程提供NequIP的完整实践指南，从安装配置到复现Nature Communications论文结果。

## 学习目标

1. 掌握NequIP的安装和环境配置
2. 学会编写NequIP配置文件
3. 训练NequIP模型并监控训练过程
4. 复现NequIP在Nature Communications论文中的结果
5. 理解球谐函数和张量积的实际应用

## NequIP简介

**NequIP** (Neural Equivariant Interatomic Potentials)是由Batzner等人在2021年发表于Nature Communications的等变神经网络势函数。

**核心特点**：
- 基于E(3)等变消息传递
- 使用球谐函数表示不可约表示
- 张量积操作保持等变性
- 最先进的预测精度和数据效率

**论文**：
Batzner et al., "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials", Nature Communications 13, 2453 (2022)

**代码**：https://github.com/mir-group/nequip

## 项目结构

```
03-NequIP/
├── README.md
├── installation.sh           # 安装脚本
├── configs/                  # 配置文件
│   ├── minimal.yaml         # 最小配置
│   ├── aspirin.yaml         # Aspirin训练配置
│   └── ethanol.yaml         # 乙醇训练配置
├── data/                     # 数据目录
│   ├── md17_aspirin/
│   └── custom_data/
├── scripts/                  # 脚本
│   ├── prepare_data.py      # 数据准备
│   ├── monitor_training.py  # 训练监控
│   └── evaluate_model.py    # 模型评估
└── notebooks/                # Jupyter notebooks
    └── nature_paper_reproduction.ipynb
```

## 第一部分：NequIP安装

### 1.1 环境要求

```bash
# 系统要求
- Python >= 3.7
- PyTorch >= 1.11.0
- CUDA >= 11.3 (如果使用GPU)
```

### 1.2 通过Conda安装（推荐）

```bash
#!/bin/bash
# installation.sh - NequIP安装脚本

set -e

echo "Creating NequIP environment..."

# 创建环境
conda create -n nequip python=3.9 -y
conda activate nequip

# 安装PyTorch (CUDA 11.8)
conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装e3nn (核心依赖)
pip install e3nn==0.5.1

# 安装NequIP
pip install nequip

# 验证安装
python -c "import nequip; print(f'NequIP version: {nequip.__version__}')"
python -c "import e3nn; print(f'e3nn version: {e3nn.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "NequIP installation completed!"
```

### 1.3 从源码安装（开发版本）

```bash
# 克隆仓库
git clone https://github.com/mir-group/nequip.git
cd nequip

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .

# 验证
nequip-train --help
nequip-evaluate --help
```

### 1.4 安装验证

```python
"""
test_installation.py - 验证NequIP安装
"""
import torch
import e3nn
import nequip

print("=" * 60)
print("NequIP Installation Check")
print("=" * 60)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"e3nn: {e3nn.__version__}")
print(f"NequIP: {nequip.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# 测试e3nn
from e3nn import o3

# 创建球谐函数
irreps = o3.Irreps("1x0e + 1x1o + 1x2e")
print(f"\nIrreps: {irreps}")
print(f"Dimension: {irreps.dim}")

# 测试张量积
tp = o3.FullyConnectedTensorProduct(
    "1x0e + 1x1o",  # 输入1
    "1x1o",         # 输入2
    "1x0e + 1x1o + 1x2e",  # 输出
    shared_weights=False
)
print(f"\nTensor product parameters: {sum(p.numel() for p in tp.parameters())}")

print("\n✓ All checks passed!")
```

## 第二部分：配置文件编写

### 2.1 最小配置示例

```yaml
# configs/minimal.yaml
# 最小可运行的NequIP配置

# 数据集
root: ./data/md17_aspirin
dataset: ase                    # ASE数据集格式
dataset_file_name: aspirin.xyz  # 数据文件
chemical_symbols:
  - H
  - C
  - O

# 网络架构
num_layers: 4                   # 交互层数
l_max: 2                        # 最大角动量量子数
parity: true                    # 是否包含奇宇称
num_features: 32                # 特征维度
num_basis: 8                    # 径向基函数数量
r_max: 4.0                      # 截断半径(Å)

# 训练
n_train: 1000                   # 训练集大小
n_val: 100                      # 验证集大小
batch_size: 5
max_epochs: 100
learning_rate: 0.005

# 损失函数
loss_coeffs:
  forces: 100                   # 力的权重
  total_energy:
    - 1
    - PerAtomMSELoss            # 每原子能量MSE

# 输出
wandb: false                    # 是否使用Weights & Biases
verbose: info
model_save_dir: ./results/minimal
```

### 2.2 完整配置详解

```yaml
# configs/aspirin.yaml
# Aspirin分子的完整训练配置

# ========== 数据配置 ==========
root: ./data/md17_aspirin
dataset: ase
dataset_file_name: aspirin.xyz

# 化学元素
chemical_symbols:
  - H
  - C
  - O

# 数据划分
n_train: 1000
n_val: 100
# test集自动为剩余数据

# 数据增强（旋转、平移不变性）
dataset_statistics_stride: 1

# ========== 网络架构 ==========
# 不可约表示
# l_max控制角动量，越高表示能力越强但计算量越大
# l_max=0: 仅标量
# l_max=1: 标量+矢量
# l_max=2: 标量+矢量+2阶张量
l_max: 2
parity: true  # 包含奇偶宇称

# 特征数
num_features: 32      # 隐藏特征维度
num_layers: 4         # 消息传递层数

# 径向网络
num_basis: 8          # 径向基函数数量
BesselBasis_trainable: true  # 贝塞尔基是否可训练
PolynomialCutoff_p: 6        # 多项式截断次数
r_max: 4.0                    # 截断半径

# 卷积层类型
invariant_layers: 2          # 不变层数
invariant_neurons: 64        # 不变层神经元数
avg_num_neighbors: auto      # 平均邻居数（用于归一化）

# ========== 训练配置 ==========
# 优化器
optimizer_name: Adam
optimizer_params:
  amsgrad: false
  betas:
    - 0.9
    - 0.999

# 学习率
learning_rate: 0.005
lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 50
lr_scheduler_factor: 0.5

# 训练参数
batch_size: 5
max_epochs: 1000
train_val_split: random      # 随机划分
shuffle: true                # 打乱训练集
metrics_key: validation_loss # 用于保存最佳模型的指标

# Early stopping
early_stopping_patiences:
  validation_loss: 100

# 梯度
gradient_clip_val: 10.0

# ========== 损失函数 ==========
# NequIP支持多任务学习
loss_coeffs:
  forces:
    - 100                    # 力的权重（相对能量）
    - PerSpeciesL1Loss       # 每种元素的L1损失
  total_energy:
    - 1
    - PerAtomMSELoss         # 每原子能量MSE

# ========== 日志和保存 ==========
# Weights & Biases
wandb: true
wandb_project: nequip-aspirin
wandb_entity: your_username

# 日志
verbose: info
log_batch_freq: 10
log_epoch_freq: 1

# 模型保存
model_save_dir: ./results/aspirin
save_checkpoint_freq: -1     # -1表示只保存最佳模型
save_ema: true               # 保存指数移动平均模型
save_ema_decay: 0.99

# ========== 硬件配置 ==========
device: cuda                 # cuda或cpu
default_dtype: float32       # 数据类型
allow_tf32: false            # 是否允许TF32（A100+）

# 性能
compile_model: false         # PyTorch 2.0编译
```

### 2.3 配置参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `l_max` | 最大角动量 | 1-2（小分子），2-3（晶体） |
| `num_features` | 特征维度 | 32-128 |
| `num_layers` | 交互层数 | 3-5 |
| `num_basis` | 径向基数量 | 8-12 |
| `r_max` | 截断半径 | 分子：4-5Å，晶体：6-8Å |
| `batch_size` | 批大小 | 根据GPU内存，通常1-10 |
| `forces` loss weight | 力权重 | 10-1000 |

## 第三部分：数据准备

### 3.1 数据格式

NequIP支持多种格式：
- **ASE** (推荐): Extended XYZ格式
- **NPZ**: NumPy压缩格式
- **LAMMPS**: LAMMPS数据文件

### 3.2 从MD17下载数据

```python
"""
scripts/prepare_data.py - 准备MD17数据集
"""
import urllib.request
import os
import gzip
import shutil
from ase.io import read, write

def download_md17(molecule='aspirin', data_dir='./data'):
    """
    下载MD17数据集

    可选分子:
    - aspirin, benzene, ethanol, malonaldehyde,
      naphthalene, salicylic_acid, toluene, uracil
    """
    os.makedirs(data_dir, exist_ok=True)

    # MD17 URL
    base_url = "http://www.quantum-machine.org/gdml/data/xyz/"
    url = f"{base_url}md17_{molecule}.xyz.gz"

    output_dir = os.path.join(data_dir, f'md17_{molecule}')
    os.makedirs(output_dir, exist_ok=True)

    gz_file = os.path.join(output_dir, f'{molecule}.xyz.gz')
    xyz_file = os.path.join(output_dir, f'{molecule}.xyz')

    # 下载
    print(f"Downloading {molecule} from MD17...")
    urllib.request.urlretrieve(url, gz_file)

    # 解压
    print("Extracting...")
    with gzip.open(gz_file, 'rb') as f_in:
        with open(xyz_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_file)

    # 验证
    atoms_list = read(xyz_file, index=':')
    print(f"Successfully downloaded {len(atoms_list)} configurations")
    print(f"Saved to: {xyz_file}")

    return xyz_file

def convert_to_nequip_format(input_file, output_file=None):
    """
    转换到NequIP格式（确保能量和力的标签正确）
    """
    atoms_list = read(input_file, index=':')

    if output_file is None:
        output_file = input_file.replace('.xyz', '_nequip.xyz')

    # NequIP要求Extended XYZ格式，包含：
    # - energy (标量)
    # - forces (Nx3数组)

    processed = []
    for atoms in atoms_list:
        # 检查是否有能量和力
        if not hasattr(atoms, 'calc') or atoms.calc is None:
            # 从info字典读取
            if 'energy' in atoms.info and 'forces' in atoms.arrays:
                # 创建单点计算器
                from ase.calculators.singlepoint import SinglePointCalculator
                calc = SinglePointCalculator(
                    atoms,
                    energy=atoms.info['energy'],
                    forces=atoms.arrays['forces']
                )
                atoms.calc = calc
            else:
                print(f"Warning: Missing energy or forces, skipping frame")
                continue

        processed.append(atoms)

    # 保存
    write(output_file, processed, format='extxyz')
    print(f"Converted {len(processed)} configurations to {output_file}")

    return output_file

if __name__ == '__main__':
    # 下载Aspirin
    xyz_file = download_md17('aspirin')

    # 转换格式
    convert_to_nequip_format(xyz_file)
```

### 3.3 自定义数据集

```python
"""
创建自定义数据集
"""
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

def create_custom_dataset():
    """
    从DFT计算创建NequIP数据集
    """
    atoms_list = []

    # 示例：从VASP/GPAW输出读取
    for i in range(100):
        # 假设你有坐标、能量、力
        positions = np.random.randn(10, 3)  # 10个原子
        symbols = ['C'] * 10

        atoms = Atoms(symbols=symbols, positions=positions)

        # 能量和力（从你的DFT计算）
        energy = -1234.56  # eV
        forces = np.random.randn(10, 3)  # eV/Å

        # 附加计算器
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.calc = calc

        atoms_list.append(atoms)

    # 保存为Extended XYZ
    write('custom_dataset.xyz', atoms_list, format='extxyz')

    print(f"Created custom dataset with {len(atoms_list)} configurations")
```

## 第四部分：训练NequIP模型

### 4.1 启动训练

```bash
# 基础训练
nequip-train configs/aspirin.yaml

# 指定GPU
CUDA_VISIBLE_DEVICES=0 nequip-train configs/aspirin.yaml

# 从检查点恢复
nequip-train configs/aspirin.yaml --restart-from results/aspirin/best_model.pth

# 多GPU训练（数据并行）
CUDA_VISIBLE_DEVICES=0,1 nequip-train configs/aspirin.yaml --distributed
```

### 4.2 训练监控脚本

```python
"""
scripts/monitor_training.py - 实时监控训练进度
"""
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TrainingMonitor(FileSystemEventHandler):
    """监控NequIP训练日志"""

    def __init__(self, log_dir, metrics_file='metrics_epoch.csv'):
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, metrics_file)
        self.last_modified = 0

    def on_modified(self, event):
        if event.src_path.endswith('metrics_epoch.csv'):
            # 避免重复读取
            current_time = os.path.getmtime(self.metrics_file)
            if current_time > self.last_modified:
                self.last_modified = current_time
                self.plot_metrics()

    def plot_metrics(self):
        """绘制训练曲线"""
        try:
            df = pd.read_csv(self.metrics_file)
        except:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 能量损失
        if 'train_loss' in df.columns and 'validation_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train')
            axes[0, 0].plot(df['epoch'], df['validation_loss'], label='Validation')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].set_yscale('log')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            axes[0, 0].set_title('Total Loss')

        # 能量MAE
        if 'train_e_mae' in df.columns:
            axes[0, 1].plot(df['epoch'], df['train_e_mae'], label='Train')
            if 'validation_e_mae' in df.columns:
                axes[0, 1].plot(df['epoch'], df['validation_e_mae'], label='Validation')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Energy MAE (eV)')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
            axes[0, 1].set_title('Energy MAE')

        # 力MAE
        if 'train_f_mae' in df.columns:
            axes[1, 0].plot(df['epoch'], df['train_f_mae'], label='Train')
            if 'validation_f_mae' in df.columns:
                axes[1, 0].plot(df['epoch'], df['validation_f_mae'], label='Validation')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Force MAE (eV/Å)')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
            axes[1, 0].set_title('Force MAE')

        # 学习率
        if 'lr' in df.columns:
            axes[1, 1].plot(df['epoch'], df['lr'])
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(alpha=0.3)
            axes[1, 1].set_title('Learning Rate')

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'), dpi=150)
        plt.close()

        print(f"Updated training curves at epoch {df['epoch'].iloc[-1]}")

def monitor_training(log_dir):
    """启动监控"""
    print(f"Monitoring training in: {log_dir}")
    print("Press Ctrl+C to stop")

    monitor = TrainingMonitor(log_dir)
    observer = Observer()
    observer.schedule(monitor, log_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == '__main__':
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else './results/aspirin'
    monitor_training(log_dir)
```

### 4.3 使用TensorBoard监控

```bash
# 如果配置了wandb，可以在线查看
# 本地可以使用TensorBoard

# 安装
pip install tensorboard

# 启动（如果NequIP输出了TensorBoard日志）
tensorboard --logdir ./results/aspirin
```

## 第五部分：模型评估

### 5.1 评估脚本

```bash
# 在测试集上评估
nequip-evaluate \
  --model results/aspirin/best_model.pth \
  --dataset-config configs/aspirin.yaml \
  --metrics-config configs/eval_metrics.yaml \
  --output results/aspirin/test_metrics.txt
```

### 5.2 自定义评估

```python
"""
scripts/evaluate_model.py - 详细模型评估
"""
import torch
import numpy as np
from nequip.data import AtomicDataDict, AtomicData
from nequip.ase import NequIPCalculator
from ase.io import read
import matplotlib.pyplot as plt

def load_nequip_model(model_path):
    """加载NequIP模型"""
    return NequIPCalculator.from_deployed_model(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

def evaluate_on_dataset(calc, dataset_file, n_samples=None):
    """
    在数据集上评估模型

    Args:
        calc: NequIP calculator
        dataset_file: 数据文件
        n_samples: 评估样本数（None表示全部）
    """
    # 读取数据
    atoms_list = read(dataset_file, index=':')

    if n_samples is not None:
        atoms_list = atoms_list[:n_samples]

    energy_predictions = []
    energy_targets = []
    force_predictions = []
    force_targets = []

    print(f"Evaluating on {len(atoms_list)} configurations...")

    for i, atoms in enumerate(atoms_list):
        if i % 100 == 0:
            print(f"  {i}/{len(atoms_list)}")

        # 真实值
        energy_true = atoms.get_potential_energy()
        forces_true = atoms.get_forces()

        # 预测
        atoms.calc = calc
        energy_pred = atoms.get_potential_energy()
        forces_pred = atoms.get_forces()

        energy_predictions.append(energy_pred)
        energy_targets.append(energy_true)
        force_predictions.append(forces_pred.flatten())
        force_targets.append(forces_true.flatten())

    # 转换为数组
    energy_predictions = np.array(energy_predictions)
    energy_targets = np.array(energy_targets)
    force_predictions = np.concatenate(force_predictions)
    force_targets = np.concatenate(force_targets)

    # 计算指标
    energy_mae = np.mean(np.abs(energy_predictions - energy_targets))
    energy_rmse = np.sqrt(np.mean((energy_predictions - energy_targets)**2))

    force_mae = np.mean(np.abs(force_predictions - force_targets))
    force_rmse = np.sqrt(np.mean((force_predictions - force_targets)**2))

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Energy MAE:  {energy_mae:.6f} eV")
    print(f"Energy RMSE: {energy_rmse:.6f} eV")
    print(f"Force MAE:   {force_mae:.6f} eV/Å")
    print(f"Force RMSE:  {force_rmse:.6f} eV/Å")
    print("=" * 60)

    return {
        'energy_pred': energy_predictions,
        'energy_true': energy_targets,
        'force_pred': force_predictions,
        'force_true': force_targets,
        'energy_mae': energy_mae,
        'energy_rmse': energy_rmse,
        'force_mae': force_mae,
        'force_rmse': force_rmse
    }

def plot_evaluation_results(results, save_dir='./'):
    """绘制评估结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 能量对比图
    axes[0].scatter(
        results['energy_true'],
        results['energy_pred'],
        alpha=0.5,
        s=10
    )
    min_e = min(results['energy_true'].min(), results['energy_pred'].min())
    max_e = max(results['energy_true'].max(), results['energy_pred'].max())
    axes[0].plot([min_e, max_e], [min_e, max_e], 'r--', linewidth=2, label='Ideal')
    axes[0].set_xlabel('DFT Energy (eV)', fontsize=12)
    axes[0].set_ylabel('NequIP Energy (eV)', fontsize=12)
    axes[0].set_title(f'Energy Prediction (MAE={results["energy_mae"]:.4f} eV)', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 力对比图
    axes[1].scatter(
        results['force_true'],
        results['force_pred'],
        alpha=0.3,
        s=1
    )
    min_f = min(results['force_true'].min(), results['force_pred'].min())
    max_f = max(results['force_true'].max(), results['force_pred'].max())
    axes[1].plot([min_f, max_f], [min_f, max_f], 'r--', linewidth=2, label='Ideal')
    axes[1].set_xlabel('DFT Forces (eV/Å)', fontsize=12)
    axes[1].set_ylabel('NequIP Forces (eV/Å)', fontsize=12)
    axes[1].set_title(f'Force Prediction (MAE={results["force_mae"]:.4f} eV/Å)', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/evaluation_results.png', dpi=300)
    plt.close()

    print(f"Plots saved to {save_dir}/evaluation_results.png")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to deployed model')
    parser.add_argument('--dataset', required=True, help='Path to test dataset')
    parser.add_argument('--n-samples', type=int, default=None, help='Number of samples to evaluate')
    parser.add_argument('--output', default='./', help='Output directory')
    args = parser.parse_args()

    # 加载模型
    print(f"Loading model from {args.model}...")
    calc = load_nequip_model(args.model)

    # 评估
    results = evaluate_on_dataset(calc, args.dataset, args.n_samples)

    # 绘图
    plot_evaluation_results(results, args.output)

if __name__ == '__main__':
    main()
```

### 5.3 运行评估

```bash
python scripts/evaluate_model.py \
  --model results/aspirin/deployed_model.pth \
  --dataset data/md17_aspirin/aspirin.xyz \
  --n-samples 1000 \
  --output results/aspirin/
```

## 第六部分：复现Nature Communications论文

### 6.1 论文中的主要结果

论文报告了NequIP在MD17数据集上的性能：

| 分子 | 训练样本 | 能量MAE (meV) | 力MAE (meV/Å) |
|------|----------|---------------|---------------|
| Aspirin | 1000 | 2.9 | 8.8 |
| Ethanol | 1000 | 2.4 | 7.2 |
| Malonaldehyde | 1000 | 2.7 | 6.9 |
| Naphthalene | 1000 | 2.1 | 5.3 |
| Salicylic acid | 1000 | 3.1 | 9.2 |
| Toluene | 1000 | 2.5 | 6.4 |
| Uracil | 1000 | 2.6 | 7.0 |

### 6.2 复现步骤

```python
"""
reproduce_nature_paper.py - 复现Nature Communications论文结果
"""
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 论文报告的结果（参考值）
PAPER_RESULTS = {
    'aspirin': {'e_mae': 2.9, 'f_mae': 8.8},
    'ethanol': {'e_mae': 2.4, 'f_mae': 7.2},
    'malonaldehyde': {'e_mae': 2.7, 'f_mae': 6.9},
    'naphthalene': {'e_mae': 2.1, 'f_mae': 5.3},
    'salicylic_acid': {'e_mae': 3.1, 'f_mae': 9.2},
    'toluene': {'e_mae': 2.5, 'f_mae': 6.4},
    'uracil': {'e_mae': 2.6, 'f_mae': 7.0}
}

def create_paper_config(molecule, output_dir='./configs'):
    """
    创建论文中使用的配置
    """
    os.makedirs(output_dir, exist_ok=True)

    config = f"""
# Nature Communications论文配置: {molecule}

root: ./data/md17_{molecule}
dataset: ase
dataset_file_name: {molecule}.xyz

# 网络架构（论文设置）
num_layers: 4
l_max: 2
parity: true
num_features: 64
num_basis: 8
r_max: 4.0

# 训练（论文设置）
n_train: 1000
n_val: 100
batch_size: 5
max_epochs: 10000

# 优化器
learning_rate: 0.01
lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 50
lr_scheduler_factor: 0.5

# 损失函数
loss_coeffs:
  forces:
    - 100
    - PerSpeciesL1Loss
  total_energy:
    - 1
    - PerAtomMSELoss

# Early stopping
early_stopping_patiences:
  validation_loss: 200

# 输出
model_save_dir: ./results/{molecule}_paper
wandb: false
verbose: info
"""

    config_file = os.path.join(output_dir, f'{molecule}_paper.yaml')
    with open(config_file, 'w') as f:
        f.write(config)

    return config_file

def train_all_molecules():
    """训练所有MD17分子"""
    results = {}

    for molecule in PAPER_RESULTS.keys():
        print(f"\n{'='*60}")
        print(f"Training {molecule}")
        print(f"{'='*60}\n")

        # 创建配置
        config_file = create_paper_config(molecule)

        # 训练
        cmd = f"nequip-train {config_file}"
        subprocess.run(cmd, shell=True, check=True)

        # 评估
        model_path = f"./results/{molecule}_paper/best_model.pth"
        dataset_path = f"./data/md17_{molecule}/{molecule}.xyz"

        # 读取结果（从日志或重新评估）
        # 这里简化，实际应该调用evaluate函数
        results[molecule] = {
            'e_mae': 0.0,  # 替换为实际结果
            'f_mae': 0.0
        }

    return results

def compare_with_paper(our_results):
    """对比我们的结果与论文"""
    molecules = list(PAPER_RESULTS.keys())

    paper_e = [PAPER_RESULTS[m]['e_mae'] for m in molecules]
    paper_f = [PAPER_RESULTS[m]['f_mae'] for m in molecules]

    our_e = [our_results[m]['e_mae'] for m in molecules]
    our_f = [our_results[m]['f_mae'] for m in molecules]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(molecules))
    width = 0.35

    # 能量MAE对比
    axes[0].bar(x - width/2, paper_e, width, label='Paper', alpha=0.8)
    axes[0].bar(x + width/2, our_e, width, label='Our Implementation', alpha=0.8)
    axes[0].set_ylabel('Energy MAE (meV)')
    axes[0].set_title('Energy Prediction Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(molecules, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # 力MAE对比
    axes[1].bar(x - width/2, paper_f, width, label='Paper', alpha=0.8)
    axes[1].bar(x + width/2, our_f, width, label='Our Implementation', alpha=0.8)
    axes[1].set_ylabel('Force MAE (meV/Å)')
    axes[1].set_title('Force Prediction Accuracy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(molecules, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('paper_reproduction_comparison.png', dpi=300)
    plt.close()

    print("\nComparison with paper saved to paper_reproduction_comparison.png")

if __name__ == '__main__':
    # 训练所有分子
    # results = train_all_molecules()

    # 对比结果
    # compare_with_paper(results)

    # 示例：只训练一个分子
    molecule = 'ethanol'
    config = create_paper_config(molecule)
    print(f"Created config for {molecule}: {config}")
    print("\nTo train, run:")
    print(f"  nequip-train {config}")
```

### 6.3 结果分析notebook

```python
# notebooks/nature_paper_reproduction.ipynb

"""
Jupyter notebook用于交互式分析和可视化
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ase.io import read

# %%
# 加载训练结果
metrics = pd.read_csv('./results/aspirin_paper/metrics_epoch.csv')

# 绘制学习曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(metrics['epoch'], metrics['validation_e_mae'] * 1000, label='Energy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Energy MAE (meV)')
axes[0].set_title('Energy Learning Curve')
axes[0].axhline(2.9, color='r', linestyle='--', label='Paper Result')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(metrics['epoch'], metrics['validation_f_mae'] * 1000, label='Force')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Force MAE (meV/Å)')
axes[1].set_title('Force Learning Curve')
axes[1].axhline(8.8, color='r', linestyle='--', label='Paper Result')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# 加载模型进行预测
from nequip.ase import NequIPCalculator

calc = NequIPCalculator.from_deployed_model(
    './results/aspirin_paper/deployed_model.pth'
)

# 测试集评估
test_data = read('./data/md17_aspirin/aspirin.xyz', index='100000:')

predictions = []
for atoms in test_data[:100]:  # 评估100个
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    predictions.append({'energy': e, 'forces': f})

# %%
# 可视化预测分布
# ... (添加更多分析)
```

## 练习题

### 练习1：基础训练
在MD17的ethanol分子上训练NequIP，尝试复现论文结果。

### 练习2：超参数影响
研究`l_max`、`num_features`、`num_layers`对精度和训练时间的影响。

### 练习3：小数据集挑战
用不同训练集大小(100, 500, 1000, 5000)训练，绘制学习曲线。

### 练习4：多元素体系
在包含多种元素的晶体数据集上训练NequIP。

### 练习5：与其他模型对比
对比NequIP、PaiNN、SchNet在同一数据集上的表现。

## 常见问题

### Q1: 训练很慢？

**优化方法**：
- 减小`batch_size`（但可能影响收敛）
- 减少`l_max`（从2降到1）
- 减少`num_features`
- 使用更少的`num_layers`
- 启用`compile_model: true`（PyTorch 2.0+）

### Q2: 内存不足？

**解决方法**：
- 设置`batch_size: 1`
- 减小模型（`num_features`, `l_max`）
- 使用`default_dtype: float16`（可能损失精度）
- 减小`r_max`截断半径

### Q3: 结果无法复现论文？

**检查**：
- 数据集版本是否一致？
- 数据划分是否相同？（random seed）
- 超参数是否完全一致？
- 训练时长是否足够？（论文可能训练数千epoch）
- PyTorch/e3nn版本差异

### Q4: 如何在LAMMPS中使用？

NequIP支持导出为LAMMPS可用格式：

```bash
# 部署模型
nequip-deploy \
  --from-config results/aspirin/best_model.pth \
  --out-file aspirin_deployed.pth

# 转换为LAMMPS格式（需要额外工具）
# 使用pair_allegro或pair_nequip接口
```

## 总结

NequIP代表了神经网络势函数的最新进展：

**优势**：
- 等变性保证物理对称性
- 数据效率极高（1000样本即可达到优秀精度）
- 力预测非常准确
- 理论基础扎实（群论、不可约表示）

**挑战**：
- 计算成本较高（相比SchNet）
- 实现复杂度高
- 需要深入理解e3nn和群论

**适用场景**：
- 小数据集（<10,000样本）
- 需要极高精度的力预测
- 研究性质需要等变性（偶极矩、极化等）
- 有充足计算资源

## 参考资料

- **NequIP论文**: [Nature Communications (2022)](https://doi.org/10.1038/s41467-022-29939-5)
- **NequIP GitHub**: https://github.com/mir-group/nequip
- **e3nn文档**: https://docs.e3nn.org/
- **MD17数据集**: http://www.quantum-machine.org/gdml/
- **教程视频**: https://www.youtube.com/watch?v=... (如果有)

---

**返回**: [Part 4主页](../../README.md)
