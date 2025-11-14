# SchNet实践：从训练到分子动力学模拟
# SchNet Practice: From Training to Molecular Dynamics Simulation

本教程提供SchNet模型的完整实践流程，从数据准备、模型训练、精度评估到分子动力学模拟和性质计算。

## 学习目标

1. 掌握QM9数据集的准备和预处理
2. 实现并训练SchNet模型预测能量和力
3. 评估模型在测试集上的准确度
4. 将训练好的模型用于分子动力学模拟
5. 计算径向分布函数(RDF)等统计性质

## 项目结构

```
01-SchNet-Practice/
├── README.md
├── requirements.txt
├── data/                     # 数据目录
│   └── QM9/                 # QM9数据集
├── models/                   # 模型定义
│   ├── __init__.py
│   ├── schnet.py            # SchNet实现
│   ├── layers.py            # 自定义层
│   └── utils.py             # 辅助函数
├── scripts/                  # 训练和评估脚本
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评估脚本
│   ├── md_simulation.py     # MD模拟脚本
│   └── analyze_rdf.py       # RDF分析脚本
└── configs/                  # 配置文件
    ├── schnet_default.yaml  # 默认配置
    └── schnet_forces.yaml   # 力训练配置
```

## 环境安装

### 依赖包

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
torch>=2.0.0
torch-geometric>=2.3.0
torch-scatter>=2.1.0
ase>=3.22.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.62.0
pyyaml>=6.0
scipy>=1.7.0
```

### 验证安装

```python
import torch
import torch_geometric
from ase import Atoms
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 第一部分：QM9数据集准备

### 1.1 数据集下载

```python
from torch_geometric.datasets import QM9

# 自动下载QM9数据集
dataset = QM9(root='./data/QM9')

print(f"数据集大小: {len(dataset)}")
print(f"节点特征维度: {dataset.num_node_features}")
print(f"边特征维度: {dataset.num_edge_features}")

# 查看单个样本
data = dataset[0]
print(data)
# Data(x=[5, 11], edge_index=[2, 8], edge_attr=[8, 4], y=[1, 19], pos=[5, 3], z=[5])
```

### 1.2 数据集分析

```python
"""
data_analysis.py - 分析QM9数据集
"""
import torch
from torch_geometric.datasets import QM9
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
dataset = QM9(root='./data/QM9')

# 统计原子数分布
num_atoms_list = [data.num_nodes for data in dataset]

plt.figure(figsize=(10, 6))
plt.hist(num_atoms_list, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Atoms')
plt.ylabel('Frequency')
plt.title('Distribution of Molecule Sizes in QM9')
plt.savefig('qm9_size_distribution.png', dpi=300)
plt.close()

# 统计元素分布
element_counts = {}
for data in dataset[:1000]:  # 采样分析
    for z in data.z:
        element = int(z.item())
        element_counts[element] = element_counts.get(element, 0) + 1

print("Element distribution:")
for element, count in sorted(element_counts.items()):
    print(f"  Z={element}: {count}")

# 分析能量分布
energies = [data.y[0, 0].item() for data in dataset[:10000]]

plt.figure(figsize=(10, 6))
plt.hist(energies, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Energy (Ha)')
plt.ylabel('Frequency')
plt.title('Distribution of Internal Energy at 0K in QM9')
plt.savefig('qm9_energy_distribution.png', dpi=300)
plt.close()

print(f"Energy range: [{min(energies):.2f}, {max(energies):.2f}] Ha")
```

### 1.3 数据预处理

```python
"""
data_preprocessing.py - 数据预处理和划分
"""
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import numpy as np

def preprocess_qm9(root='./data/QM9', target_property=0):
    """
    预处理QM9数据集

    Args:
        root: 数据集根目录
        target_property: 目标性质索引
            0: μ (dipole moment)
            1: α (isotropic polarizability)
            2: ε_HOMO (HOMO energy)
            3: ε_LUMO (LUMO energy)
            4: Δε (HOMO-LUMO gap)
            5: <R²> (electronic spatial extent)
            6: ZPVE (zero point vibrational energy)
            7: U₀ (internal energy at 0K)
            8: U (internal energy at 298.15K)
            9: H (enthalpy at 298.15K)
            10: G (free energy at 298.15K)
            11: c_v (heat capacity at 298.15K)
    """
    dataset = QM9(root=root)

    # 过滤无效数据
    valid_indices = []
    for i, data in enumerate(dataset):
        if not torch.isnan(data.y[0, target_property]):
            valid_indices.append(i)

    print(f"Total samples: {len(dataset)}")
    print(f"Valid samples: {len(valid_indices)}")

    # 数据划分：train 80%, val 10%, test 10%
    np.random.seed(42)
    np.random.shuffle(valid_indices)

    n_samples = len(valid_indices)
    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)

    train_indices = valid_indices[:n_train]
    val_indices = valid_indices[n_train:n_train+n_val]
    test_indices = valid_indices[n_train+n_val:]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    return train_indices, val_indices, test_indices

def get_dataloaders(root='./data/QM9', target_property=7, batch_size=32):
    """
    获取数据加载器
    """
    train_indices, val_indices, test_indices = preprocess_qm9(root, target_property)

    dataset = QM9(root=root)

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders(
        target_property=7,  # U₀: internal energy at 0K
        batch_size=32
    )

    # 测试数据加载
    batch = next(iter(train_loader))
    print(f"Batch: {batch}")
    print(f"Batch size: {batch.num_graphs}")
```

## 第二部分：SchNet模型实现

### 2.1 核心组件实现

```python
"""
models/layers.py - SchNet核心层实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class ShiftedSoftplus(nn.Module):
    """Shifted softplus activation function"""
    def forward(self, x):
        return F.softplus(x) - torch.log(torch.tensor(2.0))

class GaussianRBF(nn.Module):
    """径向基函数层"""
    def __init__(self, num_rbf=50, cutoff=5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # 高斯中心均匀分布
        self.register_buffer('centers', torch.linspace(0, cutoff, num_rbf))

        # 高斯宽度
        self.gamma = 10.0 / cutoff

    def forward(self, distances):
        """
        Args:
            distances: (num_edges,) 原子间距离
        Returns:
            rbf: (num_edges, num_rbf) RBF展开
        """
        distances = distances.unsqueeze(-1)  # (num_edges, 1)
        rbf = torch.exp(-self.gamma * (distances - self.centers)**2)
        return rbf

def cosine_cutoff(distances, cutoff):
    """
    余弦截断函数

    Args:
        distances: (num_edges,) 原子间距离
        cutoff: 截断半径
    Returns:
        cutoff_values: (num_edges,) 截断权重
    """
    # f(r) = 0.5 * (cos(π*r/r_c) + 1) for r < r_c, else 0
    cutoff_values = 0.5 * (torch.cos(distances * torch.pi / cutoff) + 1.0)
    cutoff_values = cutoff_values * (distances < cutoff).float()
    return cutoff_values

class CFConv(nn.Module):
    """连续滤波器卷积层"""
    def __init__(self, n_atom_basis, n_filters, n_rbf):
        super().__init__()

        # 滤波器生成网络
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_filters),
            ShiftedSoftplus(),
            nn.Linear(n_filters, n_filters)
        )

        # 原子特征更新网络
        self.dense = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            ShiftedSoftplus(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )

    def forward(self, x, edge_index, edge_weight, edge_attr):
        """
        Args:
            x: (num_nodes, n_atom_basis) 节点特征
            edge_index: (2, num_edges) 边索引
            edge_weight: (num_edges,) 截断权重
            edge_attr: (num_edges, n_rbf) RBF展开
        Returns:
            out: (num_nodes, n_atom_basis) 更新后的节点特征
        """
        # 生成距离依赖的滤波器
        W = self.filter_net(edge_attr)  # (num_edges, n_filters)
        W = W * edge_weight.unsqueeze(-1)

        # 消息传递：邻居特征 * 滤波器
        row, col = edge_index
        messages = x[col] * W  # (num_edges, n_filters)

        # 聚合消息
        aggregated = scatter(messages, row, dim=0, reduce='sum')

        # 更新特征
        out = self.dense(aggregated)

        return out

class InteractionBlock(nn.Module):
    """SchNet交互层"""
    def __init__(self, n_atom_basis=64, n_filters=64, n_rbf=50):
        super().__init__()
        self.cfconv = CFConv(n_atom_basis, n_filters, n_rbf)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        """
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_weight: 截断权重
            edge_attr: RBF特征
        Returns:
            x: 更新后的节点特征
        """
        # 卷积
        v = self.cfconv(x, edge_index, edge_weight, edge_attr)

        # 残差连接
        x = x + v

        return x
```

### 2.2 完整SchNet模型

```python
"""
models/schnet.py - 完整SchNet模型
"""
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from .layers import GaussianRBF, cosine_cutoff, InteractionBlock, ShiftedSoftplus

class SchNet(nn.Module):
    """
    SchNet模型

    参考文献:
    Schütt et al., "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions", NeurIPS 2017
    """
    def __init__(
        self,
        num_elements=100,
        embedding_dim=64,
        n_filters=64,
        n_rbf=50,
        n_interactions=3,
        cutoff=5.0,
        max_neighbors=32
    ):
        super().__init__()

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

        # 原子嵌入
        self.embedding = nn.Embedding(num_elements, embedding_dim)

        # RBF展开
        self.rbf = GaussianRBF(n_rbf, cutoff)

        # 交互层
        self.interactions = nn.ModuleList([
            InteractionBlock(embedding_dim, n_filters, n_rbf)
            for _ in range(n_interactions)
        ])

        # 输出网络（原子能量）
        self.output_net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 1)
        )

    def forward(self, z, pos, batch=None):
        """
        Args:
            z: (num_nodes,) 原子序数
            pos: (num_nodes, 3) 原子坐标
            batch: (num_nodes,) 批次索引
        Returns:
            energy: 总能量（如果batch提供则返回每个分子的能量）
            atom_energies: (num_nodes,) 每个原子的能量贡献
        """
        # 构建边（半径图）
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_neighbors
        )

        # 计算距离
        row, col = edge_index
        diff = pos[row] - pos[col]
        distances = torch.norm(diff, dim=1)

        # 截断函数
        edge_weight = cosine_cutoff(distances, self.cutoff)

        # RBF展开
        edge_attr = self.rbf(distances)

        # 原子嵌入
        x = self.embedding(z)

        # 交互层
        for interaction in self.interactions:
            x = interaction(x, edge_index, edge_weight, edge_attr)

        # 原子能量
        atom_energies = self.output_net(x).squeeze(-1)

        # 总能量（按分子求和）
        if batch is not None:
            from torch_scatter import scatter
            energy = scatter(atom_energies, batch, dim=0, reduce='sum')
        else:
            energy = atom_energies.sum()

        return energy, atom_energies
```

### 2.3 力的计算

```python
"""
models/utils.py - 辅助函数
"""
import torch

def compute_forces(model, z, pos, batch=None):
    """
    通过自动微分计算力

    Args:
        model: SchNet模型
        z: 原子序数
        pos: 原子坐标（需要梯度）
        batch: 批次索引
    Returns:
        forces: (num_nodes, 3) 原子受力
    """
    # 确保坐标需要梯度
    pos.requires_grad_(True)

    # 前向传播
    energy, _ = model(z, pos, batch)

    # 如果是批次数据，对所有分子的能量求和
    if batch is not None:
        total_energy = energy.sum()
    else:
        total_energy = energy

    # 自动微分: F = -dE/dr
    forces = -torch.autograd.grad(
        total_energy,
        pos,
        grad_outputs=torch.ones_like(total_energy),
        create_graph=True,
        retain_graph=True
    )[0]

    return forces

def energy_and_forces(model, z, pos, batch=None):
    """
    同时计算能量和力
    """
    pos.requires_grad_(True)
    energy, atom_energies = model(z, pos, batch)

    if batch is not None:
        total_energy = energy.sum()
    else:
        total_energy = energy

    forces = -torch.autograd.grad(
        total_energy,
        pos,
        grad_outputs=torch.ones_like(total_energy),
        create_graph=True
    )[0]

    return energy, forces
```

## 第三部分：模型训练

### 3.1 训练脚本

```python
"""
scripts/train.py - SchNet训练脚本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.schnet import SchNet
from models.utils import compute_forces

def train_energy_only(model, loader, optimizer, device):
    """仅训练能量"""
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # 前向传播
        energy_pred, _ = model(batch.z, batch.pos, batch.batch)

        # 目标能量 (U₀)
        energy_true = batch.y[:, 7]  # Index 7 is U₀

        # 损失
        loss = F.mse_loss(energy_pred, energy_true)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def train_energy_and_forces(model, loader, optimizer, device, force_weight=100.0):
    """同时训练能量和力"""
    model.train()
    total_loss = 0
    energy_loss_total = 0
    force_loss_total = 0

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # 需要计算梯度的坐标
        pos = batch.pos.clone().requires_grad_(True)

        # 前向传播
        energy_pred, _ = model(batch.z, pos, batch.batch)

        # 计算力
        forces_pred = compute_forces(model, batch.z, pos, batch.batch)

        # 目标
        energy_true = batch.y[:, 7]  # U₀
        # 注意：QM9数据集不包含力，这里仅作示例
        # 实际应用中需要使用MD17等包含力的数据集

        # 能量损失
        energy_loss = F.mse_loss(energy_pred, energy_true)

        # 总损失
        loss = energy_loss
        # 如果有力的标签：loss = energy_loss + force_weight * force_loss

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        energy_loss_total += energy_loss.item()

    return total_loss / len(loader), energy_loss_total / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    """验证"""
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        # 前向传播
        energy_pred, _ = model(batch.z, batch.pos, batch.batch)

        # 目标
        energy_true = batch.y[:, 7]

        # 损失
        loss = F.mse_loss(energy_pred, energy_true)
        total_loss += loss.item()

    return total_loss / len(loader)

def main(config_file):
    # 加载配置
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据加载
    print("Loading QM9 dataset...")
    dataset = QM9(root=config['data']['root'])

    # 数据划分
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 模型
    model = SchNet(
        num_elements=config['model']['num_elements'],
        embedding_dim=config['model']['embedding_dim'],
        n_filters=config['model']['n_filters'],
        n_rbf=config['model']['n_rbf'],
        n_interactions=config['model']['n_interactions'],
        cutoff=config['model']['cutoff']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")

        # 训练
        train_loss = train_energy_only(model, train_loader, optimizer, device)

        # 验证
        val_loss = validate(model, val_loader, device)

        # 学习率调整
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved best model!")

    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('SchNet Training Curve')
    plt.savefig('training_curve.png', dpi=300)
    plt.close()

    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/schnet_default.yaml')
    args = parser.parse_args()

    main(args.config)
```

### 3.2 配置文件

```yaml
# configs/schnet_default.yaml
data:
  root: './data/QM9'
  target_property: 7  # U₀: internal energy at 0K

model:
  num_elements: 100
  embedding_dim: 128
  n_filters: 128
  n_rbf: 50
  n_interactions: 6
  cutoff: 5.0

training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 300
  force_weight: 100.0
```

### 3.3 运行训练

```bash
# 使用默认配置
python scripts/train.py --config configs/schnet_default.yaml

# 查看GPU使用
nvidia-smi -l 1
```

## 第四部分：模型评估

### 4.1 评估脚本

```python
"""
scripts/evaluate.py - 模型评估
"""
import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.schnet import SchNet
from models.utils import compute_forces

def evaluate_model(model, loader, device):
    """
    评估模型在测试集上的表现
    """
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # 预测
            energy_pred, _ = model(batch.z, batch.pos, batch.batch)

            # 目标
            energy_true = batch.y[:, 7]

            predictions.extend(energy_pred.cpu().numpy())
            targets.extend(energy_true.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # 计算指标
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    r2 = 1 - np.sum((predictions - targets)**2) / np.sum((targets - np.mean(targets))**2)

    print(f"Test MAE: {mae:.6f} Ha ({mae * 627.5:.2f} kcal/mol)")
    print(f"Test RMSE: {rmse:.6f} Ha ({rmse * 627.5:.2f} kcal/mol)")
    print(f"Test R²: {r2:.6f}")

    return predictions, targets, mae, rmse, r2

def plot_parity(predictions, targets, save_path='parity_plot.png'):
    """绘制预测vs真实值对比图"""
    plt.figure(figsize=(8, 8))

    # 散点图
    plt.scatter(targets, predictions, alpha=0.3, s=10)

    # 理想线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')

    plt.xlabel('DFT Energy (Ha)', fontsize=14)
    plt.ylabel('SchNet Predicted Energy (Ha)', fontsize=14)
    plt.title('Energy Prediction: SchNet vs DFT', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Parity plot saved to {save_path}")

def plot_error_distribution(predictions, targets, save_path='error_distribution.png'):
    """绘制误差分布"""
    errors = predictions - targets

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 误差直方图
    axes[0].hist(errors * 627.5, bins=100, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Error (kcal/mol)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14)
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2)

    # 累积分布
    sorted_errors = np.sort(np.abs(errors * 627.5))
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1].plot(sorted_errors, cumulative, linewidth=2)
    axes[1].set_xlabel('Absolute Error (kcal/mol)', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_title('Cumulative Error Distribution', fontsize=14)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Error distribution saved to {save_path}")

def main():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据集
    print("Loading QM9 dataset...")
    dataset = QM9(root='./data/QM9')

    # 测试集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_dataset = dataset[train_size+val_size:]

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Test dataset size: {len(test_dataset)}")

    # 加载模型
    print("Loading model...")
    model = SchNet(
        num_elements=100,
        embedding_dim=128,
        n_filters=128,
        n_rbf=50,
        n_interactions=6,
        cutoff=5.0
    ).to(device)

    model.load_state_dict(torch.load('best_model.pt'))
    print("Model loaded successfully!")

    # 评估
    print("\nEvaluating model...")
    predictions, targets, mae, rmse, r2 = evaluate_model(model, test_loader, device)

    # 绘图
    plot_parity(predictions, targets)
    plot_error_distribution(predictions, targets)

    print("\nEvaluation completed!")

if __name__ == '__main__':
    main()
```

### 4.2 运行评估

```bash
python scripts/evaluate.py
```

## 第五部分：分子动力学模拟

### 5.1 MD模拟脚本

```python
"""
scripts/md_simulation.py - 使用SchNet进行分子动力学模拟
"""
import torch
import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.io import write
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.schnet import SchNet

class SchNetCalculator:
    """
    ASE Calculator接口，使用SchNet模型计算能量和力
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def get_potential_energy(self, atoms):
        """计算势能"""
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).to(self.device)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            energy, _ = self.model(z, pos)

        # 转换为eV
        return energy.item() * 27.211  # Ha to eV

    def get_forces(self, atoms):
        """计算力"""
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).to(self.device)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, requires_grad=True).to(self.device)

        energy, _ = self.model(z, pos)

        # 自动微分
        forces = -torch.autograd.grad(
            energy,
            pos,
            grad_outputs=torch.ones_like(energy)
        )[0]

        # 转换为eV/Å
        return forces.cpu().detach().numpy() * 27.211  # Ha/Bohr to eV/Å

def run_md_simulation(
    atoms,
    calculator,
    temperature=300,  # K
    timestep=1.0,     # fs
    num_steps=1000,
    output_interval=10,
    trajectory_file='md_trajectory.traj'
):
    """
    运行分子动力学模拟

    Args:
        atoms: ASE Atoms对象
        calculator: 势能计算器
        temperature: 温度(K)
        timestep: 时间步长(fs)
        num_steps: 总步数
        output_interval: 输出间隔
        trajectory_file: 轨迹文件
    """
    # 初始化速度
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    # MD积分器（Velocity Verlet）
    dyn = VelocityVerlet(atoms, timestep * units.fs)

    # 存储轨迹
    trajectory = []
    energies = []
    temperatures = []

    print(f"Starting MD simulation:")
    print(f"  Temperature: {temperature} K")
    print(f"  Timestep: {timestep} fs")
    print(f"  Total steps: {num_steps}")
    print(f"  Total time: {num_steps * timestep / 1000:.2f} ps\n")

    for step in range(num_steps):
        # 计算能量和力
        energy = calculator.get_potential_energy(atoms)
        forces = calculator.get_forces(atoms)

        # 更新位置和速度
        atoms.set_calculator(None)  # ASE需要
        atoms.positions += atoms.get_velocities() * timestep * units.fs
        atoms.set_momenta(atoms.get_momenta() + forces * timestep * units.fs / 2)

        # 记录
        if step % output_interval == 0:
            trajectory.append(atoms.copy())
            energies.append(energy)

            # 计算瞬时温度
            kinetic_energy = atoms.get_kinetic_energy()
            temp = 2 * kinetic_energy / (3 * len(atoms) * units.kB)
            temperatures.append(temp)

            print(f"Step {step:5d}: E = {energy:10.4f} eV, T = {temp:6.1f} K")

    # 保存轨迹
    write(trajectory_file, trajectory)
    print(f"\nTrajectory saved to {trajectory_file}")

    return trajectory, np.array(energies), np.array(temperatures)

def main():
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SchNet(
        num_elements=100,
        embedding_dim=128,
        n_filters=128,
        n_rbf=50,
        n_interactions=6,
        cutoff=5.0
    ).to(device)

    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    print("Model loaded!\n")

    # 创建计算器
    calculator = SchNetCalculator(model, device)

    # 创建测试分子（水分子）
    atoms = Atoms(
        'H2O',
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.24, 0.93, 0.0]
        ]
    )

    # 运行MD
    trajectory, energies, temperatures = run_md_simulation(
        atoms,
        calculator,
        temperature=300,
        timestep=0.5,
        num_steps=1000,
        output_interval=10,
        trajectory_file='water_md.traj'
    )

    # 绘制能量和温度曲线
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    time = np.arange(len(energies)) * 10 * 0.5  # fs

    axes[0].plot(time, energies)
    axes[0].set_xlabel('Time (fs)')
    axes[0].set_ylabel('Energy (eV)')
    axes[0].set_title('Total Energy vs Time')
    axes[0].grid(alpha=0.3)

    axes[1].plot(time, temperatures)
    axes[1].axhline(300, color='r', linestyle='--', label='Target')
    axes[1].set_xlabel('Time (fs)')
    axes[1].set_ylabel('Temperature (K)')
    axes[1].set_title('Temperature vs Time')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('md_analysis.png', dpi=300)
    plt.close()

    print("\nMD simulation completed!")

if __name__ == '__main__':
    main()
```

### 5.2 运行MD模拟

```bash
python scripts/md_simulation.py
```

## 第六部分：径向分布函数计算

### 6.1 RDF计算脚本

```python
"""
scripts/analyze_rdf.py - 计算径向分布函数
"""
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def compute_rdf(trajectory, rmax=10.0, nbins=200, elements=None):
    """
    计算径向分布函数g(r)

    Args:
        trajectory: ASE轨迹列表
        rmax: 最大距离(Å)
        nbins: 分箱数
        elements: 元素对，例如[('H', 'H'), ('O', 'H')]
    Returns:
        r: 距离数组
        rdf: 径向分布函数
    """
    dr = rmax / nbins
    r = np.linspace(0, rmax, nbins)

    # 初始化RDF
    rdf = np.zeros(nbins)

    # 遍历所有帧
    for atoms in trajectory:
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        cell = atoms.get_cell()

        # 计算所有原子对的距离
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                # 如果指定了元素对，检查是否匹配
                if elements is not None:
                    pair = (symbols[i], symbols[j])
                    if pair not in elements and (pair[1], pair[0]) not in elements:
                        continue

                # 计算距离（考虑周期性边界条件）
                diff = positions[i] - positions[j]

                # 最小镜像约定（如果有晶胞）
                if cell is not None and np.any(cell.lengths() > 0):
                    diff = diff - cell.lengths() * np.round(diff / cell.lengths())

                distance = np.linalg.norm(diff)

                # 更新直方图
                if distance < rmax:
                    bin_idx = int(distance / dr)
                    if bin_idx < nbins:
                        rdf[bin_idx] += 1

    # 归一化
    natoms = len(trajectory[0])
    nframes = len(trajectory)

    # 理想气体密度
    if trajectory[0].get_cell() is not None:
        volume = trajectory[0].get_volume()
        density = natoms / volume
    else:
        # 如果没有晶胞，估算密度
        density = 0.001  # Å⁻³

    # 归一化因子
    for i in range(nbins):
        r_inner = i * dr
        r_outer = (i + 1) * dr
        shell_volume = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        rdf[i] /= (shell_volume * density * natoms * nframes / 2)

    return r, rdf

def plot_rdf(r, rdf, save_path='rdf.png', smooth=True):
    """绘制RDF"""
    plt.figure(figsize=(10, 6))

    if smooth:
        # 平滑处理
        rdf_smooth = savgol_filter(rdf, window_length=11, polyorder=3)
        plt.plot(r, rdf_smooth, linewidth=2, label='Smoothed')
        plt.plot(r, rdf, alpha=0.3, linewidth=1, label='Raw')
    else:
        plt.plot(r, rdf, linewidth=2)

    plt.xlabel('Distance r (Å)', fontsize=14)
    plt.ylabel('g(r)', fontsize=14)
    plt.title('Radial Distribution Function', fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0, r[-1])
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"RDF plot saved to {save_path}")

def analyze_peaks(r, rdf, prominence=0.5):
    """分析RDF峰位置"""
    from scipy.signal import find_peaks

    peaks, properties = find_peaks(rdf, prominence=prominence)

    print("\nRDF Peak Analysis:")
    print("  r (Å)    g(r)    Peak Height")
    print("-" * 40)
    for peak in peaks:
        print(f"  {r[peak]:5.2f}    {rdf[peak]:5.2f}    {properties['prominences'][list(peaks).index(peak)]:.2f}")

    return r[peaks], rdf[peaks]

def main():
    # 读取轨迹
    print("Loading trajectory...")
    trajectory = read('water_md.traj', index=':')
    print(f"Loaded {len(trajectory)} frames")

    # 计算RDF
    print("\nComputing RDF...")
    r, rdf = compute_rdf(trajectory, rmax=5.0, nbins=200)

    # 绘图
    plot_rdf(r, rdf)

    # 分析峰
    analyze_peaks(r, rdf)

    # 如果是多元素体系，计算部分RDF
    if len(set(trajectory[0].get_chemical_symbols())) > 1:
        print("\nComputing partial RDFs...")

        # O-H RDF
        r_oh, rdf_oh = compute_rdf(trajectory, rmax=5.0, nbins=200, elements=[('O', 'H')])

        # H-H RDF
        r_hh, rdf_hh = compute_rdf(trajectory, rmax=5.0, nbins=200, elements=[('H', 'H')])

        # 绘制部分RDF
        plt.figure(figsize=(10, 6))
        plt.plot(r, rdf, linewidth=2, label='Total')
        plt.plot(r_oh, rdf_oh, linewidth=2, label='O-H', linestyle='--')
        plt.plot(r_hh, rdf_hh, linewidth=2, label='H-H', linestyle=':')
        plt.xlabel('Distance r (Å)', fontsize=14)
        plt.ylabel('g(r)', fontsize=14)
        plt.title('Partial Radial Distribution Functions', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.xlim(0, 5.0)
        plt.tight_layout()
        plt.savefig('partial_rdf.png', dpi=300)
        plt.close()

        print("Partial RDF plot saved to partial_rdf.png")

    print("\nRDF analysis completed!")

if __name__ == '__main__':
    main()
```

### 6.2 运行RDF分析

```bash
python scripts/analyze_rdf.py
```

## 练习题

### 练习1：基础训练
在QM9数据集的小子集(1000个分子)上训练SchNet，观察学习曲线。

**提示**：修改数据加载部分，使用`dataset[:1000]`

### 练习2：超参数调优
尝试不同的超参数组合：
- 不同的交互层数(3, 6, 9)
- 不同的嵌入维度(64, 128, 256)
- 不同的RBF数量(25, 50, 100)

记录训练时间和最终精度，找出最佳配置。

### 练习3：多任务学习
修改模型同时预测多个性质（能量、HOMO、LUMO、gap等）。

**提示**：修改输出网络，输出多个值而不是单个能量。

### 练习4：迁移学习
在QM9上预训练模型，然后在MD17数据集上微调。

**提示**：
```python
# 加载预训练模型
model.load_state_dict(torch.load('qm9_pretrained.pt'))

# 冻结部分层
for param in model.embedding.parameters():
    param.requires_grad = False
```

### 练习5：不确定性量化
实现Monte Carlo Dropout来估计预测不确定性。

**提示**：
```python
def predict_with_uncertainty(model, data, n_samples=100):
    model.train()  # 启用dropout
    predictions = []
    for _ in range(n_samples):
        pred, _ = model(data.z, data.pos, data.batch)
        predictions.append(pred)
    mean = torch.stack(predictions).mean(0)
    std = torch.stack(predictions).std(0)
    return mean, std
```

## 常见问题

### Q1: 训练很慢怎么办？

**解决方法**：
1. 减小batch size
2. 使用GPU训练
3. 减少交互层数
4. 使用混合精度训练：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Q2: 内存不足(OOM)？

**解决方法**：
1. 减小batch size
2. 使用梯度累积
3. 减小嵌入维度
4. 使用CPU训练（虽然慢）

### Q3: MD模拟不稳定？

**检查**：
1. 时间步长是否太大？（建议0.5-1.0 fs）
2. 初始结构是否合理？
3. 模型是否训练充分？
4. 是否需要使用恒温器？

### Q4: RDF计算结果不合理？

**检查**：
1. 轨迹是否足够长？（至少需要几百帧）
2. 是否正确处理了周期性边界条件？
3. 归一化是否正确？
4. 是否已经平衡？（忽略前几十帧）

## 扩展方向

### 1. 实现DimeNet++
添加角度信息，提高精度：
```python
class DimeNetPP(nn.Module):
    def __init__(self):
        # 添加角度嵌入
        self.angle_emb = AngleEmbedding()
        # 添加方向性消息传递
        self.directional_mp = DirectionalMessagePassing()
```

### 2. 主动学习
自动识别不确定性高的构型，进行DFT计算并加入训练集。

### 3. 集成学习
训练多个模型并ensemble，提高预测精度和可靠性。

### 4. 长时程MD
结合传统力场和神经网络势，进行微秒级模拟。

## 参考资料

- **SchNet论文**: Schütt et al., "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions", NeurIPS 2017
- **SchNetPack**: https://github.com/atomistic-machine-learning/schnetpack
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **ASE文档**: https://wiki.fysik.dtu.dk/ase/

---

**返回**: [Part 4主页](../../README.md)
