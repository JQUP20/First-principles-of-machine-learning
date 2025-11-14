# SchNet图神经网络实践
# SchNet Graph Neural Network Practice

本目录包含使用SchNet进行分子和材料性质预测的完整实践项目。

## 项目简介

实现SchNet模型，用于预测分子能量和力，展示图神经网络在原子系统中的应用。

## 学习目标

1. 理解图神经网络的数据表示
2. 实现SchNet的核心组件
3. 使用自动微分计算力
4. 在QM9数据集上训练和评估模型
5. 可视化模型预测结果

## 项目结构

```
03-SchNet-GNN/
├── README.md
├── requirements.txt
├── train.py              # 训练脚本
├── test.py               # 测试脚本
├── data/                 # 数据目录
├── models/               # 模型定义
│   ├── __init__.py
│   ├── schnet.py         # SchNet实现
│   └── layers.py         # 自定义层
└── utils/                # 工具函数
    ├── __init__.py
    ├── data_loader.py    # 数据加载
    └── evaluation.py     # 评估指标
```

## 安装依赖

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
torch>=2.0.0
torch-geometric>=2.3.0
ase>=3.22.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.62.0
```

## 数据集

### QM9数据集

- **内容**: 134,000个有机小分子
- **性质**: 能量、HOMO-LUMO gap、偶极矩等
- **来源**: 量子化学计算(DFT B3LYP/6-31G(2df,p))

### 数据加载

```python
from torch_geometric.datasets import QM9

# 下载并加载数据
dataset = QM9(root='./data/QM9')

print(f"数据集大小: {len(dataset)}")
print(f"节点特征维度: {dataset.num_node_features}")
print(f"边特征维度: {dataset.num_edge_features}")

# 查看单个样本
data = dataset[0]
print(data)
# Data(x=[5, 11], edge_index=[2, 8], edge_attr=[8, 4], y=[1, 19], pos=[5, 3])
```

## SchNet模型实现

### 核心组件

#### 1. 原子嵌入层

```python
class AtomEmbedding(nn.Module):
    def __init__(self, num_elements=100, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_elements, embedding_dim)

    def forward(self, z):
        # z: 原子序数
        return self.embedding(z)
```

#### 2. 径向基函数(RBF)展开

```python
class GaussianRBF(nn.Module):
    def __init__(self, num_rbf=50, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        # 中心点均匀分布在[0, cutoff]
        self.register_buffer('centers',
                           torch.linspace(0, cutoff, num_rbf))
        # 宽度
        self.gamma = 10.0 / cutoff

    def forward(self, distances):
        # distances: (num_edges,)
        # 计算高斯基函数
        distances = distances.unsqueeze(-1)  # (num_edges, 1)
        rbf = torch.exp(-self.gamma * (distances - self.centers)**2)
        return rbf  # (num_edges, num_rbf)
```

#### 3. 连续滤波器卷积层

```python
class CFConv(nn.Module):
    def __init__(self, n_atom_basis, n_filters, n_rbf):
        super().__init__()
        # 滤波器生成网络
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_filters),
            ShiftedSoftplus(),
            nn.Linear(n_filters, n_filters)
        )
        # 原子更新网络
        self.dense = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis),
            ShiftedSoftplus(),
            nn.Linear(n_atom_basis, n_atom_basis)
        )

    def forward(self, x, edge_index, edge_weight, edge_attr):
        # x: 节点特征 (num_nodes, n_atom_basis)
        # edge_index: 边索引 (2, num_edges)
        # edge_weight: 截断权重 (num_edges,)
        # edge_attr: RBF展开 (num_edges, n_rbf)

        # 生成滤波器
        W = self.filter_net(edge_attr)  # (num_edges, n_filters)
        W = W * edge_weight.unsqueeze(-1)

        # 聚合邻居信息
        row, col = edge_index
        messages = x[col] * W  # (num_edges, n_filters)

        # 求和聚合
        from torch_scatter import scatter
        out = scatter(messages, row, dim=0, reduce='sum')

        # 更新
        out = self.dense(out)

        return out
```

#### 4. SchNet交互层

```python
class InteractionBlock(nn.Module):
    def __init__(self, n_atom_basis=64, n_filters=64, n_rbf=50):
        super().__init__()
        self.cfconv = CFConv(n_atom_basis, n_filters, n_rbf)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        # 卷积
        v = self.cfconv(x, edge_index, edge_weight, edge_attr)

        # 残差连接
        x = x + v

        return x
```

#### 5. 完整SchNet模型

```python
class SchNet(nn.Module):
    def __init__(self,
                 num_elements=100,
                 embedding_dim=64,
                 n_filters=64,
                 n_rbf=50,
                 n_interactions=3,
                 cutoff=5.0):
        super().__init__()

        # 原子嵌入
        self.embedding = AtomEmbedding(num_elements, embedding_dim)

        # RBF展开
        self.rbf = GaussianRBF(n_rbf, cutoff)

        # 交互层
        self.interactions = nn.ModuleList([
            InteractionBlock(embedding_dim, n_filters, n_rbf)
            for _ in range(n_interactions)
        ])

        # 输出网络
        self.output_net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            ShiftedSoftplus(),
            nn.Linear(64, 1)
        )

        self.cutoff = cutoff

    def forward(self, z, pos, edge_index):
        # z: 原子序数 (num_nodes,)
        # pos: 原子坐标 (num_nodes, 3)
        # edge_index: 边索引 (2, num_edges)

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

        # 总能量
        energy = atom_energies.sum()

        return energy, atom_energies
```

## 训练流程

```python
import torch
from torch_geometric.data import DataLoader

# 加载数据
dataset = QM9(root='./data/QM9')
train_dataset = dataset[:110000]
val_dataset = dataset[110000:120000]
test_dataset = dataset[120000:]

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 模型
model = SchNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # 前向传播
        energy_pred, _ = model(batch.z, batch.pos, batch.edge_index)

        # 损失(能量)
        energy_true = batch.y[:, 0]  # 第一个性质是能量
        loss = F.mse_loss(energy_pred, energy_true)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.6f}")
```

## 计算力

使用自动微分：

```python
def compute_forces(model, z, pos, edge_index):
    # 需要坐标的梯度
    pos.requires_grad_(True)

    # 计算能量
    energy, _ = model(z, pos, edge_index)

    # 自动微分得到力
    forces = -torch.autograd.grad(
        energy,
        pos,
        grad_outputs=torch.ones_like(energy),
        create_graph=True
    )[0]

    return forces
```

## 练习题

### 练习1：基础实现
实现SchNet的基本组件，在小数据集上测试。

### 练习2：力的预测
修改训练流程，同时优化能量和力。

### 练习3：可视化
可视化学习曲线和预测vs真实值的对比图。

### 练习4：超参数调优
尝试不同的`n_interactions`、`n_filters`等参数。

### 练习5：迁移学习
在QM9上预训练，然后在MD17数据集上微调。

## 扩展

- 实现DimeNet++（添加角度信息）
- 使用PyTorch Geometric的内置SchNet
- 添加不确定性量化(贝叶斯神经网络)
- 实现主动学习流程

## 参考资料

- [SchNet论文](https://arxiv.org/abs/1706.08566)
- [PyTorch Geometric文档](https://pytorch-geometric.readthedocs.io/)
- [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)

---

**返回**: [Part 3主页](../../README.md)
