# Allegro模型实战与论文复现
# Allegro Model Practice and Paper Reproduction

**论文**: "Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics"

**作者**: Musaelian et al.

**期刊**: Nature Communications (2023)

**代码**: https://github.com/mir-group/allegro

---

## 1. Allegro简介

### 1.1 什么是Allegro？

**Allegro**是NequIP的优化版本，专为**大规模分子动力学模拟**设计。

**核心改进**：
- 🚀 **速度**：比NequIP快3-5倍
- 💾 **内存**：内存占用减少2-3倍
- 📈 **可扩展性**：支持数万原子的体系
- 🔧 **易用性**：与LAMMPS无缝集成

### 1.2 Allegro vs NequIP

| 特性 | NequIP | Allegro |
|------|--------|---------|
| 精度 | 极高 | 相同 |
| 速度 | 1x | **3-5x** |
| 内存 | 1x | **0.3-0.5x** |
| 最大体系 | ~1000原子 | **>10,000原子** |
| LAMMPS集成 | 支持 | **优化** |

### 1.3 关键优化技术

1. **稀疏张量积**：避免不必要的计算
2. **更高效的球谐函数**：优化的e3nn实现
3. **混合精度训练**：FP16 + FP32
4. **梯度检查点**：节省内存

---

## 2. 环境安装

### 2.1 系统要求

```bash
# 最小配置
- Python >= 3.8
- PyTorch >= 1.11.0
- CUDA >= 11.3
- 16GB RAM

# 推荐配置
- Python 3.9
- PyTorch 2.0.0
- CUDA 11.8
- V100/A100 GPU
- 32GB+ RAM
```

### 2.2 安装步骤

#### 方法1：Conda安装（推荐）

```bash
# 创建环境
conda create -n allegro python=3.9 -y
conda activate allegro

# 安装PyTorch
conda install pytorch==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装e3nn
pip install e3nn==0.5.1

# 安装nequip（Allegro基于NequIP）
pip install nequip==0.5.6

# 安装Allegro
pip install git+https://github.com/mir-group/allegro.git

# 其他依赖
pip install ase wandb matplotlib pandas tqdm pyyaml
```

#### 方法2：从源码安装（开发）

```bash
git clone https://github.com/mir-group/allegro.git
cd allegro
pip install -e .
```

### 2.3 验证安装

```python
# test_installation.py
import torch
import e3nn
import nequip
import allegro

print(f"PyTorch: {torch.__version__}")
print(f"e3nn: {e3nn.__version__}")
print(f"NequIP: {nequip.__version__}")
print(f"Allegro: {allegro.__version__}")

if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU")

print("\n✓ All packages installed successfully!")
```

---

## 3. Allegro配置详解

### 3.1 基本配置文件

```yaml
# allegro_minimal.yaml
# 最小可运行配置

# 数据集
root: ./data/md17_aspirin
dataset: ase
dataset_file_name: aspirin.xyz
chemical_symbol_to_type:
  H: 0
  C: 1
  O: 2

# 数据划分
n_train: 1000
n_val: 100
train_val_split: random

# 网络架构
num_layers: 2
max_ell: 2                # 最大角动量
parity: true              # 奇偶宇称
num_features: 64          # 特征维度
num_basis: 8              # 径向基
r_max: 4.0                # 截断半径

# Allegro特定参数
latent_mlp_latent_dimensions: [128, 128, 128]
env_embed_multiplicity: 32
two_body_latent_mlp_latent_dimensions: [64, 128, 256]
latent_resnet: true

# 训练
batch_size: 5
learning_rate: 0.005
max_epochs: 1000

# 损失函数
loss_coeffs:
  forces: 100
  total_energy:
    - 1
    - PerAtomMSELoss

# 输出
model_builders:
  - allegro.model.Allegro
default_dtype: float32
wandb: false
```

### 3.2 完整配置（论文设置）

```yaml
# allegro_paper.yaml
# Nature Communications论文中的完整配置

# 数据集
root: ./data/md17_aspirin
dataset: ase
dataset_file_name: aspirin.xyz

chemical_symbol_to_type:
  H: 0
  C: 1
  O: 2

# 数据划分
n_train: 1000
n_val: 100
dataset_statistics_stride: 1

# 网络架构
model_builders:
  - allegro.model.Allegro

num_layers: 2
max_ell: 3
parity: true
num_features: 128
num_basis: 8
BesselBasis_trainable: true
PolynomialCutoff_p: 6
r_max: 5.0

# Allegro架构参数
# 这些参数控制Allegro的具体实现
latent_mlp_latent_dimensions: [256, 512, 512]  # 潜在MLP的维度
env_embed_multiplicity: 64                      # 环境嵌入倍数
two_body_latent_mlp_latent_dimensions: [128, 256, 512, 1024]
latent_resnet: true                             # 使用残差连接
latent_resnet_update_ratios: [0.0, 0.5]        # 残差更新比率

# 对称收缩
# 这是Allegro的核心创新
avg_num_neighbors: auto                         # 自动计算平均邻居数

# 优化器
optimizer_name: Adam
optimizer_amsgrad: false
optimizer_betas: [0.9, 0.999]
optimizer_eps: 1.0e-8
optimizer_weight_decay: 0.0

# 学习率调度
learning_rate: 0.005
lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 50
lr_scheduler_factor: 0.8
lr_scheduler_min_lr: 1.0e-6

# 训练参数
batch_size: 5
max_epochs: 5000
train_val_split: random
shuffle: true
metrics_key: validation_loss

# Early stopping
early_stopping_patiences:
  validation_loss: 1000

# 梯度
gradient_clip_val: 10.0

# 损失函数
loss_coeffs:
  forces:
    - 100
    - PerAtomL1Loss  # 使用L1损失
  total_energy:
    - 1
    - PerAtomMSELoss

# 数据类型和设备
default_dtype: float32
allow_tf32: true        # A100 GPU可以启用
device: cuda
compile_model: false    # PyTorch 2.0编译（可能不稳定）

# 日志
wandb: false            # 设为true使用Weights & Biases
verbose: info
log_batch_freq: 10
log_epoch_freq: 1

# 保存
model_save_dir: ./results/aspirin_allegro
save_checkpoint_freq: -1
save_ema: true
save_ema_decay: 0.99
```

### 3.3 关键参数说明

#### Allegro特有参数

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `latent_mlp_latent_dimensions` | 潜在MLP的隐藏层维度 | [256, 512, 512] |
| `env_embed_multiplicity` | 环境嵌入的倍数 | 32-64 |
| `two_body_latent_mlp_latent_dimensions` | 二体潜在MLP维度 | [128, 256, 512, 1024] |
| `latent_resnet` | 是否使用残差连接 | true |
| `latent_resnet_update_ratios` | 残差更新比率 | [0.0, 0.5] |

#### 性能相关

```yaml
# 速度优化
allow_tf32: true           # A100上启用TF32
compile_model: false       # PyTorch编译（实验性）
num_workers: 4             # 数据加载线程数

# 内存优化
batch_size: 1              # 大体系减小batch size
gradient_checkpointing: true  # 梯度检查点（节省内存）
```

---

## 4. 训练Allegro模型

### 4.1 基础训练

```bash
# 训练单个模型
nequip-train allegro_minimal.yaml

# 使用特定GPU
CUDA_VISIBLE_DEVICES=0 nequip-train allegro_paper.yaml

# 从检查点恢复
nequip-train allegro_paper.yaml --restart-from results/aspirin_allegro/checkpoint.pth
```

### 4.2 监控训练

```bash
# 实时查看日志
tail -f results/aspirin_allegro/training.log

# 查看metrics
column -t results/aspirin_allegro/metrics_epoch.csv | less

# 使用TensorBoard（如果配置）
tensorboard --logdir results/aspirin_allegro
```

### 4.3 自动化训练脚本

```python
# train_allegro.py
import subprocess
import os

def train_allegro(molecule, config_template='allegro_paper.yaml', gpu=0):
    """
    训练Allegro模型

    Args:
        molecule: 分子名称（如'aspirin'）
        config_template: 配置模板
        gpu: GPU ID
    """
    # 修改配置
    with open(config_template, 'r') as f:
        config = f.read()

    # 替换数据路径
    config = config.replace('aspirin', molecule)

    # 保存新配置
    config_file = f'allegro_{molecule}.yaml'
    with open(config_file, 'w') as f:
        f.write(config)

    # 训练命令
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    cmd = ['nequip-train', config_file]

    print(f"Training Allegro on {molecule} (GPU {gpu})...")
    subprocess.run(cmd, env=env, check=True)

    print(f"✓ Training completed for {molecule}")

if __name__ == '__main__':
    # 训练所有MD17分子
    molecules = ['aspirin', 'benzene', 'ethanol', 'malonaldehyde']

    for i, mol in enumerate(molecules):
        gpu = i % 4  # 使用4个GPU循环
        train_allegro(mol, gpu=gpu)
```

---

## 5. 模型评估

### 5.1 评估脚本

```bash
# 在测试集上评估
nequip-evaluate \
  --model results/aspirin_allegro/best_model.pth \
  --dataset-config allegro_paper.yaml \
  --output results/aspirin_allegro/test_results.xyz
```

### 5.2 自定义评估

```python
# evaluate_allegro.py
import torch
from nequip.data import AtomicDataDict
from nequip.model import model_from_config
from ase.io import read
import numpy as np

def evaluate_model(model_path, dataset_path, n_test=1000):
    """
    评估Allegro模型

    Args:
        model_path: 模型文件
        dataset_path: 测试数据
        n_test: 测试样本数
    """
    # 加载模型
    model, config = model_from_config(config_path=model_path, initialize=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 加载测试数据
    atoms_list = read(dataset_path, index=':')
    test_data = atoms_list[-n_test:]

    # 评估
    energy_errors = []
    force_errors = []

    for atoms in test_data:
        # 准备输入
        data = {
            AtomicDataDict.POSITIONS_KEY: torch.tensor(atoms.positions, dtype=torch.float32),
            AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
            AtomicDataDict.EDGE_INDEX_KEY: ...  # 构建边
        }

        # 预测
        with torch.no_grad():
            output = model(data)

        energy_pred = output[AtomicDataDict.TOTAL_ENERGY_KEY].item()
        forces_pred = output[AtomicDataDict.FORCE_KEY].cpu().numpy()

        # 真实值
        energy_true = atoms.get_potential_energy()
        forces_true = atoms.get_forces()

        # 误差
        energy_errors.append(abs(energy_pred - energy_true))
        force_errors.extend(np.abs(forces_pred - forces_true).flatten())

    # 统计
    energy_mae = np.mean(energy_errors) * 1000  # eV to meV
    force_mae = np.mean(force_errors) * 1000    # eV/Å to meV/Å

    print(f"Energy MAE: {energy_mae:.2f} meV")
    print(f"Force MAE: {force_mae:.2f} meV/Å")

    return energy_mae, force_mae

if __name__ == '__main__':
    evaluate_model(
        'results/aspirin_allegro/best_model.pth',
        'data/md17_aspirin/aspirin.xyz'
    )
```

---

## 6. 与LAMMPS集成

### 6.1 部署模型

```bash
# 部署模型为可与LAMMPS使用的格式
nequip-deploy \
  build results/aspirin_allegro/best_model.pth \
  aspirin_deployed.pth
```

### 6.2 LAMMPS输入脚本

```lammps
# in.lammps - 使用Allegro势的MD模拟

units metal
atom_style atomic
boundary p p p

# 读取结构
read_data aspirin.data

# Allegro势
pair_style allegro
pair_coeff * * aspirin_deployed.pth H C O

# NVT系综
velocity all create 300.0 12345
fix 1 all nvt temp 300.0 300.0 0.1

# 输出
thermo 100
thermo_style custom step temp pe ke etotal press
dump 1 all custom 1000 traj.lammpstrj id type x y z fx fy fz

# 运行
timestep 0.001  # 1 fs
run 100000      # 100 ps
```

### 6.3 运行LAMMPS

```bash
# 单核运行
lmp -in in.lammps

# 多核并行
mpirun -np 4 lmp -in in.lammps

# GPU加速（如果编译了GPU支持）
lmp -sf gpu -pk gpu 1 -in in.lammps
```

---

## 7. Nature Communications论文复现

### 7.1 论文结果

**Table 1**: MD17基准测试（1000训练样本）

| 分子 | Allegro能量MAE (meV) | Allegro力MAE (meV/Å) |
|------|---------------------|---------------------|
| Aspirin | 1.9 | 5.4 |
| Benzene | 0.6 | 2.1 |
| Ethanol | 1.5 | 4.8 |
| **平均** | **1.5** | **4.5** |

**Table 2**: 计算效率

| 体系大小 | Allegro速度 | NequIP速度 | 加速比 |
|---------|------------|-----------|--------|
| 100原子 | 10 ms/step | 35 ms/step | 3.5x |
| 1000原子 | 95 ms/step | 380 ms/step | 4.0x |
| 5000原子 | 520 ms/step | OOM | >5x |

### 7.2 复现步骤

#### 步骤1：准备数据

```bash
# 下载MD17数据集
wget http://www.quantum-machine.org/gdml/data/xyz/md17_aspirin.xyz.gz
gunzip md17_aspirin.xyz.gz
```

#### 步骤2：训练模型

```bash
# 使用论文配置训练
nequip-train allegro_paper.yaml
```

#### 步骤3：评估精度

```bash
# 评估
nequip-evaluate \
  --model results/aspirin_allegro/best_model.pth \
  --dataset-config allegro_paper.yaml
```

#### 步骤4：基准测试

```python
# benchmark_allegro.py
import torch
import time
from nequip.model import model_from_config

def benchmark(model_path, num_atoms, n_iter=100):
    """
    测试推理速度

    Args:
        model_path: 模型路径
        num_atoms: 原子数
        n_iter: 迭代次数
    """
    model, config = model_from_config(config_path=model_path)
    model.eval()

    # 创建随机输入
    data = create_random_system(num_atoms)

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(data)

    # 计时
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(n_iter):
        with torch.no_grad():
            _ = model(data)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    ms_per_iter = (elapsed / n_iter) * 1000

    print(f"{num_atoms} atoms: {ms_per_iter:.2f} ms/step")

    return ms_per_iter

if __name__ == '__main__':
    for n_atoms in [100, 500, 1000, 5000]:
        benchmark('aspirin_deployed.pth', n_atoms)
```

---

## 8. 练习题

### 练习1：基础训练
在MD17的ethanol分子上训练Allegro，对比与NequIP的速度差异。

### 练习2：大体系测试
创建一个包含1000个原子的硅晶体，测试Allegro的性能。

### 练习3：超参数调优
尝试不同的`latent_mlp_latent_dimensions`，观察对精度和速度的影响。

### 练习4：LAMMPS MD
使用训练好的Allegro模型进行10 ns的MD模拟，计算扩散系数。

### 练习5：迁移学习
在QM9上预训练，然后在MD17上微调，对比精度提升。

---

## 参考资料

- **Allegro论文**: Musaelian et al., "Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics", Nature Communications (2023)
- **NequIP论文**: Batzner et al., "E(3)-equivariant graph neural networks", Nature Communications (2022)
- **代码**: https://github.com/mir-group/allegro
- **文档**: https://nequip.readthedocs.io/

---

**返回**: [Part 5主页](../../README.md)
