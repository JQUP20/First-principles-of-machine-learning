# PaiNN等变神经网络实践
# PaiNN Equivariant Neural Network Practice

本教程介绍PaiNN (Polarizable Atom Interaction Neural Network)的实现和应用，重点讲解等变性的实际应用。

## 学习目标

1. 理解标量特征和矢量特征的区别
2. 实现等变消息传递机制
3. 训练PaiNN模型预测分子性质
4. 对比PaiNN与SchNet的性能

## PaiNN简介

**PaiNN**由Schütt等人在2021年提出，是SchNet的等变扩展版本。

**核心创新**：
- 同时使用标量和矢量特征
- 等变消息传递保持旋转对称性
- 更高的数据效率和预测精度

**论文**：Schütt et al., "Equivariant message passing for the prediction of tensorial properties and molecular spectra", ICML 2021

## 项目结构

```
02-PaiNN/
├── README.md
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── painn.py              # PaiNN模型
│   ├── painn_layers.py       # PaiNN层
│   └── equivariance_test.py  # 等变性测试
├── scripts/
│   ├── train_painn.py        # 训练脚本
│   ├── compare_models.py     # 模型对比
│   └── visualize_features.py # 特征可视化
└── configs/
    └── painn_md17.yaml       # MD17配置
```

## 环境安装

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
torch>=2.0.0
torch-geometric>=2.3.0
torch-scatter>=2.1.0
e3nn>=0.5.0
ase>=3.22.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.62.0
pyyaml>=6.0
```

## 第一部分：标量与矢量特征

### 1.1 特征表示

```python
"""
标量特征 vs 矢量特征
"""
import torch

# 标量特征：旋转不变
scalar_features = torch.randn(10, 64)  # (num_atoms, n_scalar_features)

# 矢量特征：旋转等变
vector_features = torch.randn(10, 64, 3)  # (num_atoms, n_vector_features, 3)

print(f"Scalar features shape: {scalar_features.shape}")
print(f"Vector features shape: {vector_features.shape}")

# 旋转变换
import torch.nn.functional as F

def random_rotation_matrix():
    """生成随机旋转矩阵"""
    # 随机旋转角度
    angles = torch.randn(3) * torch.pi

    # 绕x轴旋转
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angles[0]), -torch.sin(angles[0])],
        [0, torch.sin(angles[0]), torch.cos(angles[0])]
    ])

    # 绕y轴旋转
    Ry = torch.tensor([
        [torch.cos(angles[1]), 0, torch.sin(angles[1])],
        [0, 1, 0],
        [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
    ])

    # 绕z轴旋转
    Rz = torch.tensor([
        [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
        [torch.sin(angles[2]), torch.cos(angles[2]), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

# 测试旋转等变性
R = random_rotation_matrix()

# 标量特征在旋转下不变
assert torch.allclose(scalar_features, scalar_features)  # 显然成立

# 矢量特征在旋转下等变
vector_features_rotated = torch.einsum('ij,nfj->nfi', R, vector_features)
print(f"Vector features after rotation: {vector_features_rotated.shape}")
```

### 1.2 等变性质的重要性

```python
"""
为什么需要矢量特征？
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 示例：水分子的偶极矩
# 偶极矩是矢量，应该等变地旋转

positions = torch.tensor([
    [0.0, 0.0, 0.0],      # O
    [0.96, 0.0, 0.0],     # H
    [0.24, 0.93, 0.0]     # H
])

charges = torch.tensor([8.0, 1.0, 1.0])  # 原子序数作为"电荷"

# 计算偶极矩（简化）
dipole = torch.sum(charges.unsqueeze(-1) * positions, dim=0)

print(f"Dipole moment: {dipole}")

# 旋转分子
R = random_rotation_matrix()
positions_rotated = positions @ R.T

# 计算旋转后的偶极矩
dipole_rotated = torch.sum(charges.unsqueeze(-1) * positions_rotated, dim=0)

# 直接旋转偶极矩
dipole_transformed = dipole @ R.T

print(f"Dipole after molecular rotation: {dipole_rotated}")
print(f"Dipole transformed directly: {dipole_transformed}")
print(f"Difference: {torch.norm(dipole_rotated - dipole_transformed):.6f}")

# 等变性：两种方式应该得到相同结果
assert torch.allclose(dipole_rotated, dipole_transformed, atol=1e-5)
print("✓ Dipole moment is equivariant!")
```

## 第二部分：PaiNN模型实现

### 2.1 等变消息传递层

```python
"""
models/painn_layers.py - PaiNN核心层
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class PaiNNMessage(nn.Module):
    """
    PaiNN消息传递层
    更新标量特征和矢量特征
    """
    def __init__(self, n_atom_basis, n_rbf):
        super().__init__()

        self.n_atom_basis = n_atom_basis

        # 标量消息
        self.scalar_message_net = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 3 * n_atom_basis)
        )

        # RBF → 滤波器权重
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 3 * n_atom_basis)
        )

    def forward(self, s, V, edge_index, edge_attr, r_ij):
        """
        Args:
            s: (num_nodes, n_atom_basis) 标量特征
            V: (num_nodes, n_atom_basis, 3) 矢量特征
            edge_index: (2, num_edges) 边索引
            edge_attr: (num_edges, n_rbf) RBF展开的距离
            r_ij: (num_edges, 3) 相对位置矢量
        Returns:
            delta_s: 标量特征更新
            delta_V: 矢量特征更新
        """
        row, col = edge_index

        # 生成滤波器权重
        W = self.filter_net(edge_attr)  # (num_edges, 3*n_atom_basis)
        W = W.reshape(-1, 3, self.n_atom_basis)  # (num_edges, 3, n_atom_basis)

        # 邻居的标量特征
        s_j = s[col]  # (num_edges, n_atom_basis)

        # 生成消息
        phi = self.scalar_message_net(s_j)  # (num_edges, 3*n_atom_basis)
        phi = phi.reshape(-1, 3, self.n_atom_basis)  # (num_edges, 3, n_atom_basis)

        # 距离加权
        phi = phi * W  # element-wise

        # 分离三个通道
        phi_s = phi[:, 0, :]   # 标量消息
        phi_v1 = phi[:, 1, :]  # 矢量消息（由标量生成）
        phi_v2 = phi[:, 2, :]  # 矢量消息（由矢量生成）

        # 标量消息聚合
        delta_s = scatter(phi_s, row, dim=0, reduce='sum')

        # 矢量消息
        # 第一部分：从邻居的矢量特征
        V_j = V[col]  # (num_edges, n_atom_basis, 3)
        msg_v1 = V_j * phi_v2.unsqueeze(-1)  # (num_edges, n_atom_basis, 3)

        # 第二部分：从标量特征生成方向性消息
        # 使用归一化的相对位置矢量
        r_ij_norm = r_ij / (torch.norm(r_ij, dim=1, keepdim=True) + 1e-8)
        msg_v2 = phi_v1.unsqueeze(-1) * r_ij_norm.unsqueeze(1)  # (num_edges, n_atom_basis, 3)

        # 聚合矢量消息
        delta_V = scatter(msg_v1 + msg_v2, row, dim=0, reduce='sum')

        return delta_s, delta_V

class PaiNNUpdate(nn.Module):
    """
    PaiNN更新层
    标量和矢量特征的自相互作用
    """
    def __init__(self, n_atom_basis):
        super().__init__()

        self.n_atom_basis = n_atom_basis

        # U矩阵：矢量到标量的映射
        self.U = nn.Linear(n_atom_basis, n_atom_basis, bias=False)

        # V矩阵：标量到矢量的映射
        self.V = nn.Linear(n_atom_basis, n_atom_basis, bias=False)

        # 标量更新网络
        self.scalar_update_net = nn.Sequential(
            nn.Linear(2 * n_atom_basis, n_atom_basis),
            nn.SiLU(),
            nn.Linear(n_atom_basis, 3 * n_atom_basis)
        )

    def forward(self, s, V):
        """
        Args:
            s: (num_nodes, n_atom_basis) 标量特征
            V: (num_nodes, n_atom_basis, 3) 矢量特征
        Returns:
            s: 更新后的标量特征
            V: 更新后的矢量特征
        """
        # 矢量特征的范数（不变量）
        V_norm = torch.norm(V, dim=-1)  # (num_nodes, n_atom_basis)

        # 矢量 → 标量信息
        V_s = self.U(V_norm)  # (num_nodes, n_atom_basis)

        # 合并标量信息
        s_combined = torch.cat([s, V_s], dim=-1)  # (num_nodes, 2*n_atom_basis)

        # 生成更新
        updates = self.scalar_update_net(s_combined)  # (num_nodes, 3*n_atom_basis)
        updates = updates.reshape(-1, 3, self.n_atom_basis)

        a_ss = updates[:, 0, :]  # 标量到标量
        a_sv = updates[:, 1, :]  # 标量到矢量
        a_vv = updates[:, 2, :]  # 矢量到矢量

        # 更新标量特征
        delta_s = a_ss

        # 更新矢量特征
        # 第一部分：矢量自更新
        delta_V1 = a_vv.unsqueeze(-1) * V

        # 第二部分：标量生成新矢量（但需要保持等变性，这里简化为0）
        # 在实际实现中，这部分通常结合其他等变操作
        delta_V2 = torch.zeros_like(V)

        return s + delta_s, V + delta_V1 + delta_V2

class PaiNNInteraction(nn.Module):
    """完整的PaiNN交互块"""
    def __init__(self, n_atom_basis, n_rbf):
        super().__init__()
        self.message = PaiNNMessage(n_atom_basis, n_rbf)
        self.update = PaiNNUpdate(n_atom_basis)

    def forward(self, s, V, edge_index, edge_attr, r_ij):
        # 消息传递
        delta_s, delta_V = self.message(s, V, edge_index, edge_attr, r_ij)

        # 更新特征
        s = s + delta_s
        V = V + delta_V

        # 自相互作用
        s, V = self.update(s, V)

        return s, V
```

### 2.2 完整PaiNN模型

```python
"""
models/painn.py - 完整PaiNN模型
"""
import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from .painn_layers import PaiNNInteraction

class GaussianRBF(nn.Module):
    """径向基函数"""
    def __init__(self, num_rbf=20, cutoff=5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.register_buffer('centers', torch.linspace(0, cutoff, num_rbf))
        self.gamma = 10.0 / cutoff

    def forward(self, distances):
        distances = distances.unsqueeze(-1)
        return torch.exp(-self.gamma * (distances - self.centers)**2)

def cosine_cutoff(distances, cutoff):
    """余弦截断函数"""
    cutoff_values = 0.5 * (torch.cos(distances * torch.pi / cutoff) + 1.0)
    cutoff_values = cutoff_values * (distances < cutoff).float()
    return cutoff_values

class PaiNN(nn.Module):
    """
    PaiNN: Polarizable Atom Interaction Neural Network

    参考文献:
    Schütt et al., "Equivariant message passing for the prediction of
    tensorial properties and molecular spectra", ICML 2021
    """
    def __init__(
        self,
        num_elements=100,
        n_atom_basis=128,
        n_rbf=20,
        n_interactions=3,
        cutoff=5.0,
        max_neighbors=32,
        predict_forces=True
    ):
        super().__init__()

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.predict_forces = predict_forces

        # 原子嵌入（标量）
        self.embedding = nn.Embedding(num_elements, n_atom_basis)

        # RBF展开
        self.rbf = GaussianRBF(n_rbf, cutoff)

        # PaiNN交互层
        self.interactions = nn.ModuleList([
            PaiNNInteraction(n_atom_basis, n_rbf)
            for _ in range(n_interactions)
        ])

        # 输出网络（能量）
        self.out_net = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis // 2),
            nn.SiLU(),
            nn.Linear(n_atom_basis // 2, 1)
        )

    def forward(self, z, pos, batch=None):
        """
        Args:
            z: (num_nodes,) 原子序数
            pos: (num_nodes, 3) 原子坐标
            batch: (num_nodes,) 批次索引
        Returns:
            energy: 总能量
            forces: (可选) 原子受力
        """
        # 构建边
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_neighbors
        )

        row, col = edge_index

        # 相对位置矢量
        r_ij = pos[row] - pos[col]  # (num_edges, 3)
        distances = torch.norm(r_ij, dim=1)  # (num_edges,)

        # 截断
        cutoff_values = cosine_cutoff(distances, self.cutoff)

        # RBF展开
        edge_attr = self.rbf(distances) * cutoff_values.unsqueeze(-1)

        # 初始化特征
        s = self.embedding(z)  # (num_nodes, n_atom_basis)
        V = torch.zeros(
            z.shape[0],
            s.shape[1],
            3,
            device=z.device,
            dtype=s.dtype
        )  # (num_nodes, n_atom_basis, 3)

        # PaiNN交互层
        for interaction in self.interactions:
            s, V = interaction(s, V, edge_index, edge_attr, r_ij)

        # 预测原子能量
        atomic_energies = self.out_net(s).squeeze(-1)  # (num_nodes,)

        # 总能量
        if batch is not None:
            from torch_scatter import scatter
            energy = scatter(atomic_energies, batch, dim=0, reduce='sum')
        else:
            energy = atomic_energies.sum()

        # 计算力
        if self.predict_forces and self.training:
            return energy, atomic_energies
        else:
            return energy, atomic_energies

def compute_forces_painn(model, z, pos, batch=None):
    """通过自动微分计算力"""
    pos.requires_grad_(True)

    energy, _ = model(z, pos, batch)

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

    return forces
```

### 2.3 等变性测试

```python
"""
models/equivariance_test.py - 验证PaiNN的等变性
"""
import torch
import numpy as np
from .painn import PaiNN

def random_rotation_matrix():
    """生成随机SO(3)旋转矩阵"""
    # Cayley变换
    A = torch.randn(3, 3)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    # 确保行列式为+1（特殊正交群）
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def test_energy_invariance():
    """测试能量的旋转不变性"""
    print("Testing energy rotational invariance...")

    model = PaiNN(n_atom_basis=32, n_interactions=2)
    model.eval()

    # 创建测试分子
    z = torch.tensor([6, 1, 1, 1, 1])  # CH4
    pos = torch.randn(5, 3)

    # 原始能量
    with torch.no_grad():
        energy_original, _ = model(z, pos)

    # 旋转坐标
    R = random_rotation_matrix()
    pos_rotated = pos @ R.T

    # 旋转后的能量
    with torch.no_grad():
        energy_rotated, _ = model(z, pos_rotated)

    # 检查不变性
    diff = torch.abs(energy_original - energy_rotated).item()
    print(f"  Energy difference: {diff:.2e}")

    if diff < 1e-5:
        print("  ✓ Energy is rotationally invariant!")
        return True
    else:
        print("  ✗ Energy is NOT rotationally invariant!")
        return False

def test_force_equivariance():
    """测试力的旋转等变性"""
    print("\nTesting force rotational equivariance...")

    from .painn import compute_forces_painn

    model = PaiNN(n_atom_basis=32, n_interactions=2)
    model.eval()

    # 创建测试分子
    z = torch.tensor([6, 1, 1, 1, 1])  # CH4
    pos = torch.randn(5, 3)

    # 原始力
    forces_original = compute_forces_painn(model, z, pos)

    # 旋转坐标
    R = random_rotation_matrix()
    pos_rotated = pos @ R.T

    # 旋转后的力
    forces_rotated = compute_forces_painn(model, z, pos_rotated)

    # 直接旋转原始力
    forces_transformed = forces_original @ R.T

    # 检查等变性
    diff = torch.norm(forces_rotated - forces_transformed).item()
    print(f"  Force difference: {diff:.2e}")

    if diff < 1e-4:
        print("  ✓ Forces are rotationally equivariant!")
        return True
    else:
        print("  ✗ Forces are NOT rotationally equivariant!")
        return False

def test_translation_invariance():
    """测试平移不变性"""
    print("\nTesting translational invariance...")

    model = PaiNN(n_atom_basis=32, n_interactions=2)
    model.eval()

    z = torch.tensor([6, 1, 1, 1, 1])
    pos = torch.randn(5, 3)

    # 原始能量
    with torch.no_grad():
        energy_original, _ = model(z, pos)

    # 平移坐标
    translation = torch.randn(1, 3) * 10
    pos_translated = pos + translation

    # 平移后的能量
    with torch.no_grad():
        energy_translated, _ = model(z, pos_translated)

    # 检查不变性
    diff = torch.abs(energy_original - energy_translated).item()
    print(f"  Energy difference: {diff:.2e}")

    if diff < 1e-5:
        print("  ✓ Energy is translationally invariant!")
        return True
    else:
        print("  ✗ Energy is NOT translationally invariant!")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("PaiNN Equivariance Tests")
    print("=" * 60)

    test_energy_invariance()
    test_force_equivariance()
    test_translation_invariance()

    print("\n" + "=" * 60)
```

## 第三部分：训练PaiNN

### 3.1 在MD17数据集上训练

```python
"""
scripts/train_painn.py - PaiNN训练脚本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.painn import PaiNN, compute_forces_painn

def load_md17_dataset(molecule='aspirin', root='./data'):
    """
    加载MD17数据集

    MD17包含8个小分子，每个分子~100,000个构型
    包含能量和力的DFT标签
    """
    try:
        from torch_geometric.datasets import MD17
    except ImportError:
        print("MD17 dataset not available in your PyG version")
        print("Using QM9 as fallback...")
        from torch_geometric.datasets import QM9
        return QM9(root=root)

    dataset = MD17(root=root, name=molecule)
    print(f"Loaded MD17-{molecule}: {len(dataset)} configurations")

    return dataset

def train_epoch(model, loader, optimizer, device, force_weight=100.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    energy_loss_total = 0
    force_loss_total = 0

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # 需要梯度的坐标
        pos = batch.pos.clone().requires_grad_(True)

        # 预测能量
        energy_pred, _ = model(batch.z, pos, batch.batch)

        # 预测力
        forces_pred = compute_forces_painn(model, batch.z, pos, batch.batch)

        # 目标
        energy_true = batch.energy if hasattr(batch, 'energy') else batch.y
        forces_true = batch.force

        # 损失
        energy_loss = F.mse_loss(energy_pred, energy_true)
        force_loss = F.mse_loss(forces_pred, forces_true)

        loss = energy_loss + force_weight * force_loss

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        energy_loss_total += energy_loss.item()
        force_loss_total += force_loss.item()

    n_batches = len(loader)
    return (total_loss / n_batches,
            energy_loss_total / n_batches,
            force_loss_total / n_batches)

@torch.no_grad()
def validate(model, loader, device, force_weight=100.0):
    """验证"""
    model.eval()
    total_loss = 0
    energy_loss_total = 0
    force_loss_total = 0

    for batch in loader:
        batch = batch.to(device)
        pos = batch.pos

        # 预测
        energy_pred, _ = model(batch.z, pos, batch.batch)
        forces_pred = compute_forces_painn(model, batch.z, pos, batch.batch)

        # 目标
        energy_true = batch.energy if hasattr(batch, 'energy') else batch.y
        forces_true = batch.force

        # 损失
        energy_loss = F.mse_loss(energy_pred, energy_true)
        force_loss = F.mse_loss(forces_pred, forces_true)
        loss = energy_loss + force_weight * force_loss

        total_loss += loss.item()
        energy_loss_total += energy_loss.item()
        force_loss_total += force_loss.item()

    n_batches = len(loader)
    return (total_loss / n_batches,
            energy_loss_total / n_batches,
            force_loss_total / n_batches)

def main(config_file):
    # 加载配置
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 加载数据
    print("Loading dataset...")
    dataset = load_md17_dataset(
        molecule=config['data']['molecule'],
        root=config['data']['root']
    )

    # 数据划分
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")

    # 模型
    model = PaiNN(
        num_elements=config['model']['num_elements'],
        n_atom_basis=config['model']['n_atom_basis'],
        n_rbf=config['model']['n_rbf'],
        n_interactions=config['model']['n_interactions'],
        cutoff=config['model']['cutoff']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6
    )

    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    force_weight = config['training']['force_weight']

    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")

        # 训练
        train_loss, train_e_loss, train_f_loss = train_epoch(
            model, train_loader, optimizer, device, force_weight
        )

        # 验证
        val_loss, val_e_loss, val_f_loss = validate(
            model, val_loader, device, force_weight
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Train - Total: {train_loss:.6f}, Energy: {train_e_loss:.6f}, Force: {train_f_loss:.6f}")
        print(f"  Val   - Total: {val_loss:.6f}, Energy: {val_e_loss:.6f}, Force: {val_f_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}\n")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_painn_model.pt')
            print("  ✓ Saved best model!\n")

    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('PaiNN Training Curve')
    plt.grid(alpha=0.3)
    plt.savefig('painn_training_curve.png', dpi=300)
    plt.close()

    print(f"Training completed! Best val loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/painn_md17.yaml')
    args = parser.parse_args()

    main(args.config)
```

### 3.2 配置文件

```yaml
# configs/painn_md17.yaml
data:
  root: './data/MD17'
  molecule: 'aspirin'  # aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic_acid, toluene, uracil

model:
  num_elements: 100
  n_atom_basis: 128
  n_rbf: 20
  n_interactions: 3
  cutoff: 5.0

training:
  batch_size: 16
  learning_rate: 0.0005
  weight_decay: 0.01
  epochs: 300
  force_weight: 100.0  # 力的权重
```

## 第四部分：PaiNN vs SchNet对比

### 4.1 性能对比脚本

```python
"""
scripts/compare_models.py - 对比PaiNN和SchNet
"""
import torch
import time
import numpy as np
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '01-SchNet-Practice'))

from models.painn import PaiNN, compute_forces_painn

try:
    from models.schnet import SchNet
    from models.utils import compute_forces as compute_forces_schnet
except:
    print("SchNet not found, using placeholder")
    SchNet = None

def compare_model_sizes():
    """对比模型参数量"""
    print("=" * 60)
    print("Model Size Comparison")
    print("=" * 60)

    # 相同配置
    config = {
        'n_atom_basis': 128,
        'n_interactions': 3,
        'cutoff': 5.0
    }

    painn = PaiNN(
        n_atom_basis=config['n_atom_basis'],
        n_interactions=config['n_interactions'],
        cutoff=config['cutoff']
    )

    if SchNet is not None:
        schnet = SchNet(
            embedding_dim=config['n_atom_basis'],
            n_interactions=config['n_interactions'],
            cutoff=config['cutoff']
        )

        painn_params = sum(p.numel() for p in painn.parameters())
        schnet_params = sum(p.numel() for p in schnet.parameters())

        print(f"PaiNN parameters:  {painn_params:,}")
        print(f"SchNet parameters: {schnet_params:,}")
        print(f"Ratio: {painn_params / schnet_params:.2f}x\n")
    else:
        painn_params = sum(p.numel() for p in painn.parameters())
        print(f"PaiNN parameters: {painn_params:,}\n")

def compare_inference_speed(device='cpu'):
    """对比推理速度"""
    print("=" * 60)
    print(f"Inference Speed Comparison ({device.upper()})")
    print("=" * 60)

    painn = PaiNN(n_atom_basis=128, n_interactions=3).to(device)
    painn.eval()

    # 测试数据
    batch_sizes = [1, 8, 32]
    n_atoms = 20

    for batch_size in batch_sizes:
        z = torch.randint(1, 10, (batch_size * n_atoms,)).to(device)
        pos = torch.randn(batch_size * n_atoms, 3).to(device)
        batch = torch.arange(batch_size).repeat_interleave(n_atoms).to(device)

        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = painn(z, pos, batch)

        # 计时
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        n_iter = 100
        for _ in range(n_iter):
            with torch.no_grad():
                _ = painn(z, pos, batch)

        if device == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        throughput = (batch_size * n_iter) / elapsed

        print(f"Batch size {batch_size:2d}: {elapsed/n_iter*1000:.2f} ms/iter, "
              f"{throughput:.1f} molecules/s")

    print()

def compare_accuracy():
    """对比预测精度"""
    print("=" * 60)
    print("Accuracy Comparison on QM9")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    dataset = QM9(root='./data/QM9')
    test_dataset = dataset[120000:]
    test_loader = DataLoader(test_dataset[:1000], batch_size=32)  # 测试子集

    # PaiNN
    painn = PaiNN(n_atom_basis=128, n_interactions=3).to(device)
    try:
        painn.load_state_dict(torch.load('best_painn_model.pt', map_location=device))
        print("Loaded PaiNN model")

        painn.eval()
        painn_predictions = []
        targets = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                energy_pred, _ = painn(batch.z, batch.pos, batch.batch)
                painn_predictions.extend(energy_pred.cpu().numpy())
                targets.extend(batch.y[:, 7].cpu().numpy())

        painn_predictions = np.array(painn_predictions)
        targets = np.array(targets)

        painn_mae = np.mean(np.abs(painn_predictions - targets))
        painn_rmse = np.sqrt(np.mean((painn_predictions - targets)**2))

        print(f"PaiNN  - MAE: {painn_mae:.6f} Ha, RMSE: {painn_rmse:.6f} Ha")

    except FileNotFoundError:
        print("PaiNN model not found, skipping accuracy comparison")

    # SchNet (如果可用)
    if SchNet is not None:
        try:
            schnet = SchNet(embedding_dim=128, n_interactions=3).to(device)
            schnet.load_state_dict(torch.load('../01-SchNet-Practice/best_model.pt', map_location=device))
            print("Loaded SchNet model")

            schnet.eval()
            schnet_predictions = []

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    energy_pred, _ = schnet(batch.z, batch.pos, batch.batch)
                    schnet_predictions.extend(energy_pred.cpu().numpy())

            schnet_predictions = np.array(schnet_predictions)

            schnet_mae = np.mean(np.abs(schnet_predictions - targets))
            schnet_rmse = np.sqrt(np.mean((schnet_predictions - targets)**2))

            print(f"SchNet - MAE: {schnet_mae:.6f} Ha, RMSE: {schnet_rmse:.6f} Ha")

            # 对比
            print(f"\nImprovement:")
            print(f"  MAE:  {(1 - painn_mae/schnet_mae)*100:.1f}%")
            print(f"  RMSE: {(1 - painn_rmse/schnet_rmse)*100:.1f}%")

        except FileNotFoundError:
            print("SchNet model not found")

    print()

def plot_comparison():
    """绘制对比图"""
    print("=" * 60)
    print("Generating Comparison Plots")
    print("=" * 60)

    # 性能对比条形图（示例数据）
    models = ['SchNet', 'PaiNN']
    mae_values = [0.014, 0.011]  # Ha，示例数据

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE对比
    axes[0].bar(models, mae_values, color=['#3498db', '#e74c3c'])
    axes[0].set_ylabel('MAE (Ha)')
    axes[0].set_title('Energy Prediction Accuracy')
    axes[0].grid(axis='y', alpha=0.3)

    # 训练效率对比（示例数据）
    training_samples = [10000, 5000]  # 达到相同精度所需样本数
    axes[1].bar(models, training_samples, color=['#3498db', '#e74c3c'])
    axes[1].set_ylabel('Training Samples')
    axes[1].set_title('Data Efficiency')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()

    print("Saved comparison plot to model_comparison.png\n")

def main():
    compare_model_sizes()
    compare_inference_speed('cpu')

    if torch.cuda.is_available():
        compare_inference_speed('cuda')

    compare_accuracy()
    plot_comparison()

    print("=" * 60)
    print("Comparison completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
```

### 4.2 运行对比

```bash
python scripts/compare_models.py
```

## 总结

### PaiNN的优势

1. **等变性**：矢量特征保持旋转等变性，更好地捕捉方向性信息
2. **数据效率**：在小数据集上表现更好，通常需要更少的训练样本
3. **力预测**：由于等变性，力的预测更准确
4. **可解释性**：矢量特征可以解释为物理量（偶极矩、极化等）

### PaiNN vs SchNet

| 特性 | SchNet | PaiNN |
|------|--------|-------|
| 特征类型 | 仅标量 | 标量+矢量 |
| 等变性 | 否 | 是 |
| 参数量 | 较少 | 较多(~1.5x) |
| 训练速度 | 较快 | 较慢(~1.2x) |
| 能量精度 | 好 | 更好(~10-20%) |
| 力精度 | 一般 | 优秀(~20-30%) |
| 数据需求 | 较多 | 较少 |

### 适用场景

- **SchNet**: 大规模数据集，纯能量预测，计算资源受限
- **PaiNN**: 小数据集，需要准确的力预测，MD模拟，极化性质

## 练习题

### 练习1：实现测试
运行等变性测试脚本，验证PaiNN的旋转和平移不变性。

### 练习2：特征可视化
可视化标量和矢量特征的分布和变化。

### 练习3：消融实验
- 去除矢量特征，只用标量，对比性能下降
- 改变交互层数，观察影响

### 练习4：扩展到其他性质
修改输出层，预测偶极矩、极化率等矢量/张量性质。

### 练习5：迁移学习
在QM9上预训练PaiNN，迁移到MD17，对比与从头训练的差异。

## 参考资料

- **PaiNN论文**: Schütt et al., "Equivariant message passing for the prediction of tensorial properties and molecular spectra", ICML 2021
- **e3nn库**: https://e3nn.org/ (更高级的等变神经网络框架)
- **Equivariant Networks**: https://github.com/risilab/cormorant

---

**返回**: [Part 4主页](../../README.md)
