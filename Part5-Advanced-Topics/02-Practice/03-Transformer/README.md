# Transformer模型实现与实践
# Transformer Model Implementation and Practice

## 目录 | Contents

1. [实验简介](#1-实验简介)
2. [从零实现Transformer](#2-从零实现transformer)
3. [用于原子系统的Transformer](#3-用于原子系统的transformer)
4. [Equiformer实现](#4-equiformer实现)
5. [训练与评估](#5-训练与评估)
6. [实际应用](#6-实际应用)

---

## 1. 实验简介

### 1.1 学习目标

- 从零实现标准Transformer（Attention is All You Need）
- 理解Self-Attention和Multi-Head Attention机制
- 将Transformer应用于原子系统
- 实现Equiformer（等变Transformer）
- 对比Transformer与GNN的性能

### 1.2 环境准备

```bash
# 创建环境
conda create -n transformer-practice python=3.9 -y
conda activate transformer-practice

# 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install numpy matplotlib pandas seaborn
pip install torch-geometric  # 图神经网络
pip install e3nn  # 等变神经网络
pip install ase  # 原子模拟
pip install wandb  # 实验跟踪
pip install einops  # 张量操作
```

### 1.3 预期时间

| 任务 | 时间 |
|------|------|
| 实现基础Transformer | 1-2小时 |
| 实现Atomic Transformer | 2-3小时 |
| 实现Equiformer | 3-4小时 |
| 训练和评估 | 视数据集而定 |

---

## 2. 从零实现Transformer

### 2.1 Scaled Dot-Product Attention

```python
# attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        参数:
            Q: Query [batch_size, num_heads, seq_len, d_k]
            K: Key [batch_size, num_heads, seq_len, d_k]
            V: Value [batch_size, num_heads, seq_len, d_v]
            mask: [batch_size, 1, seq_len, seq_len] or None

        返回:
            output: [batch_size, num_heads, seq_len, d_v]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        d_k = Q.size(-1)

        # 计算attention scores: QK^T / √d_k
        # [B, H, L, d_k] @ [B, H, d_k, L] -> [B, H, L, L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 应用mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和：[B, H, L, L] @ [B, H, L, d_v] -> [B, H, L, d_v]
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

# 测试
def test_attention():
    batch_size = 2
    num_heads = 8
    seq_len = 10
    d_k = 64

    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_k)

    attn = ScaledDotProductAttention()
    output, weights = attn(Q, K, V)

    print(f"Output shape: {output.shape}")  # [2, 8, 10, 64]
    print(f"Attention weights shape: {weights.shape}")  # [2, 8, 10, 10]
    print(f"Attention weights sum (should be 1): {weights[0, 0, 0].sum().item():.4f}")

if __name__ == '__main__':
    test_attention()
```

### 2.2 Multi-Head Attention

```python
# multi_head_attention.py
import torch
import torch.nn as nn
from attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    将d_model维度分成h个头，每个头计算独立的attention
    """
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # 输出投影

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        拆分成多个头
        [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.size()

        # Reshape: [B, L, d_model] -> [B, L, num_heads, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose: [B, L, num_heads, d_k] -> [B, num_heads, L, d_k]
        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x):
        """
        合并多个头
        [batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # Transpose: [B, num_heads, L, d_k] -> [B, L, num_heads, d_k]
        x = x.transpose(1, 2).contiguous()

        # Reshape: [B, L, num_heads, d_k] -> [B, L, d_model]
        x = x.view(batch_size, seq_len, self.d_model)

        return x

    def forward(self, Q, K, V, mask=None):
        """
        参数:
            Q, K, V: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len] or None

        返回:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # 线性投影
        Q = self.W_Q(Q)  # [B, L, d_model]
        K = self.W_K(K)
        V = self.W_V(V)

        # 拆分成多头
        Q = self.split_heads(Q)  # [B, num_heads, L, d_k]
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 计算attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # attn_output: [B, num_heads, L, d_k]

        # 合并多头
        output = self.combine_heads(attn_output)  # [B, L, d_model]

        # 输出投影
        output = self.W_O(output)
        output = self.dropout(output)

        return output, attn_weights

# 测试
def test_multi_head_attention():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)

    # Self-attention: Q=K=V
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn = mha(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")

    # 检查参数数量
    num_params = sum(p.numel() for p in mha.parameters())
    print(f"Number of parameters: {num_params:,}")

if __name__ == '__main__':
    test_multi_head_attention()
```

### 2.3 Position-wise Feed-Forward Network

```python
# feed_forward.py
import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

    通常 d_ff = 4 * d_model
    """
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 或 ReLU

    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, d_model]
        返回:
            output: [batch_size, seq_len, d_model]
        """
        # x: [B, L, d_model]
        x = self.linear1(x)  # [B, L, d_ff]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)  # [B, L, d_model]
        x = self.dropout(x)

        return x
```

### 2.4 Transformer Encoder Layer

```python
# encoder_layer.py
import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import PositionWiseFeedForward

class TransformerEncoderLayer(nn.Module):
    """
    单个Transformer Encoder层

    包含：
    1. Multi-Head Self-Attention
    2. Add & Norm
    3. Feed-Forward Network
    4. Add & Norm
    """
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        参数:
            x: [batch_size, seq_len, d_model]
            mask: attention mask
        返回:
            output: [batch_size, seq_len, d_model]
        """
        # Self-Attention + Residual + Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer Normalization

        # Feed-Forward + Residual + Norm
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x
```

### 2.5 位置编码

```python
# positional_encoding.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    正弦位置编码

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # 注册为buffer（不参与训练）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, d_model]
        返回:
            output: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# 可视化位置编码
def visualize_positional_encoding():
    import matplotlib.pyplot as plt

    pe = PositionalEncoding(d_model=128, max_len=100, dropout=0)

    # 生成dummy input
    dummy_input = torch.zeros(1, 100, 128)
    output = pe(dummy_input)

    # 提取位置编码
    pos_enc = output[0].detach().numpy()  # [100, 128]

    plt.figure(figsize=(12, 6))
    plt.imshow(pos_enc.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Visualization')
    plt.savefig('positional_encoding.pdf')
    plt.show()

if __name__ == '__main__':
    visualize_positional_encoding()
```

### 2.6 完整Transformer Encoder

```python
# transformer.py
import torch
import torch.nn as nn
from encoder_layer import TransformerEncoderLayer
from positional_encoding import PositionalEncoding

class TransformerEncoder(nn.Module):
    """
    完整的Transformer Encoder
    """
    def __init__(
        self,
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.d_model = d_model

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 多层Encoder
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        参数:
            x: [batch_size, seq_len, d_model]
            mask: attention mask
        返回:
            output: [batch_size, seq_len, d_model]
        """
        # 添加位置编码
        x = self.pos_encoding(x)

        # 逐层传播
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        return x

# 测试
def test_transformer():
    batch_size = 4
    seq_len = 20
    d_model = 512

    transformer = TransformerEncoder(
        num_layers=6,
        d_model=d_model,
        num_heads=8,
        d_ff=2048,
        dropout=0.1
    )

    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output = transformer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 参数统计
    num_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

if __name__ == '__main__':
    test_transformer()
```

---

## 3. 用于原子系统的Transformer

### 3.1 原子系统的表示

原子系统不是序列数据，需要特殊处理：

1. **节点特征**：原子类型、电荷、磁矩等
2. **边特征**：原子间距离、角度等
3. **全局特征**：总能量、体积等

### 3.2 Graph-to-Sequence Transformer

```python
# atomic_transformer.py
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
import math

class AtomicTransformer(nn.Module):
    """
    用于原子系统的Transformer

    架构:
    1. 原子Embedding层
    2. 边Embedding层
    3. Transformer Encoder
    4. 输出头（预测能量、力等）
    """
    def __init__(
        self,
        num_species=100,  # 元素种类
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        r_max=5.0,
        num_rbf=20,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.r_max = r_max

        # 原子类型Embedding
        self.species_embedding = nn.Embedding(num_species, d_model)

        # 径向基函数（RBF）
        self.rbf = GaussianRBF(r_max=r_max, num_rbf=num_rbf)

        # 边Embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(num_rbf, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Transformer Encoder
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=500  # 最多支持500个原子
        )

        # 输出头
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, data):
        """
        参数:
            data: PyG Data对象，包含
                - pos: [N, 3] 原子坐标
                - species: [N] 原子类型
                - edge_index: [2, E] 边索引
                - batch: [N] batch索引
        返回:
            energy: [batch_size] 总能量
        """
        pos = data.pos
        species = data.species
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(len(pos), dtype=torch.long)

        # 原子Embedding
        node_feats = self.species_embedding(species)  # [N, d_model]

        # 计算边特征
        row, col = edge_index
        edge_vec = pos[row] - pos[col]  # [E, 3]
        edge_dist = torch.norm(edge_vec, dim=1)  # [E]

        # RBF编码距离
        edge_rbf = self.rbf(edge_dist)  # [E, num_rbf]
        edge_feats = self.edge_embedding(edge_rbf)  # [E, d_model]

        # 添加边信息到节点
        # 这里简化：对每个节点，平均其所有边的特征
        node_edge_feats = torch.zeros_like(node_feats)
        node_edge_feats.index_add_(0, row, edge_feats)

        # 计算每个节点的度数
        degree = torch.bincount(row, minlength=len(node_feats)).unsqueeze(1).float()
        node_edge_feats = node_edge_feats / (degree + 1e-6)

        # 合并节点和边特征
        x = node_feats + node_edge_feats  # [N, d_model]

        # 转换为序列格式（每个batch独立）
        # 这里假设batch内原子数相同，实际需要padding
        batch_size = batch.max().item() + 1
        max_atoms = (batch == 0).sum().item()  # 假设batch 0的原子数最多

        # Padding到相同长度
        x_padded = torch.zeros(batch_size, max_atoms, self.d_model, device=x.device)
        for b in range(batch_size):
            mask = batch == b
            atoms_in_batch = mask.sum().item()
            x_padded[b, :atoms_in_batch] = x[mask]

        # Transformer
        transformer_output = self.transformer(x_padded)  # [batch_size, max_atoms, d_model]

        # 预测每个原子的能量贡献
        atom_energies = self.energy_head(transformer_output).squeeze(-1)  # [batch_size, max_atoms]

        # 求和得到总能量
        total_energies = atom_energies.sum(dim=1)  # [batch_size]

        return total_energies

class GaussianRBF(nn.Module):
    """高斯径向基函数"""
    def __init__(self, r_max=5.0, num_rbf=20):
        super().__init__()
        self.r_max = r_max

        # 均匀分布的中心
        centers = torch.linspace(0, r_max, num_rbf)
        self.register_buffer('centers', centers)

        # 高斯宽度
        gamma = 1.0 / (r_max / num_rbf)
        self.register_buffer('gamma', torch.tensor(gamma))

    def forward(self, distances):
        """
        distances: [E]
        返回: [E, num_rbf]
        """
        distances = distances.unsqueeze(-1)  # [E, 1]
        return torch.exp(-self.gamma * (distances - self.centers)**2)
```

### 3.3 训练示例

```python
# train_atomic_transformer.py
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from ase.io import read
from torch_geometric.data import Data

def atoms_to_graph(atoms, r_max=5.0):
    """将ASE Atoms转换为PyG Data"""
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    species = torch.tensor(atoms.numbers, dtype=torch.long)

    # 构建边（距离小于r_max的原子对）
    from scipy.spatial import cKDTree
    tree = cKDTree(atoms.positions)
    pairs = tree.query_pairs(r_max)

    edge_index = torch.tensor(list(pairs), dtype=torch.long).t()
    # 添加反向边
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # 能量
    energy = torch.tensor([atoms.get_potential_energy()], dtype=torch.float32)

    # 力
    forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)

    data = Data(
        pos=pos,
        species=species,
        edge_index=edge_index,
        energy=energy,
        forces=forces
    )

    return data

# 准备数据
def prepare_dataset(traj_file):
    atoms_list = read(traj_file, ':')
    dataset = [atoms_to_graph(atoms) for atoms in atoms_list]
    return dataset

# 训练
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)

        # 前向
        pred_energy = model(batch)
        target_energy = batch.energy

        # 损失
        loss = F.mse_loss(pred_energy, target_energy)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    dataset = prepare_dataset('train.traj')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型
    model = AtomicTransformer(
        num_species=100,
        d_model=256,
        num_layers=4,
        num_heads=8
    ).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练
    for epoch in range(100):
        loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

if __name__ == '__main__':
    main()
```

---

## 4. Equiformer实现

Equiformer结合了Transformer和等变性，是当前最先进的原子系统模型之一。

### 4.1 等变Self-Attention

```python
# equiformer.py
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import FullyConnectedTensorProduct

class EquivariantMultiHeadAttention(nn.Module):
    """
    等变Multi-Head Attention

    使用e3nn的不可约表示处理几何信息
    """
    def __init__(
        self,
        irreps_in='64x0e + 32x1o + 16x2e',
        irreps_out='64x0e + 32x1o + 16x2e',
        num_heads=8,
        irreps_head='8x0e + 4x1o + 2x2e'
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_heads = num_heads
        self.irreps_head = o3.Irreps(irreps_head)

        # Q, K, V投影
        self.linear_Q = o3.Linear(self.irreps_in, num_heads * self.irreps_head)
        self.linear_K = o3.Linear(self.irreps_in, num_heads * self.irreps_head)
        self.linear_V = o3.Linear(self.irreps_in, num_heads * self.irreps_head)

        # 输出投影
        self.linear_out = o3.Linear(num_heads * self.irreps_head, self.irreps_out)

        # Attention权重（只在标量上计算）
        # 提取标量部分
        self.scalar_idx = [i for i, (mul, ir) in enumerate(self.irreps_head) if ir.l == 0]

    def forward(self, x, edge_index, edge_sh):
        """
        x: [N, irreps_in] 节点特征
        edge_index: [2, E] 边索引
        edge_sh: [E, irreps_sh] 边的球谐函数
        """
        # 投影到Q, K, V
        Q = self.linear_Q(x)  # [N, num_heads * irreps_head]
        K = self.linear_K(x)
        V = self.linear_V(x)

        # Reshape成多头格式
        # 这里简化处理，实际需要正确地分离不同l的分量

        # 计算attention（只在标量通道上）
        row, col = edge_index

        # 提取标量特征（l=0）
        Q_scalar = Q[row, :self.num_heads * len(self.scalar_idx)]
        K_scalar = K[col, :self.num_heads * len(self.scalar_idx)]

        # Dot product attention
        attn_scores = (Q_scalar * K_scalar).sum(dim=-1, keepdim=True) / math.sqrt(Q_scalar.size(-1))
        attn_weights = torch.softmax(attn_scores, dim=0)  # [E, 1]

        # 应用attention到V（所有不可约分量）
        V_weighted = V[col] * attn_weights  # [E, num_heads * irreps_head]

        # 聚合（按行求和）
        output = torch.zeros_like(x)
        output.index_add_(0, row, V_weighted)

        # 输出投影
        output = self.linear_out(output)

        return output

class EquiformerLayer(nn.Module):
    """
    Equiformer Layer

    包含：
    1. 等变Self-Attention
    2. 等变Feed-Forward
    3. Layer Norm
    """
    def __init__(self, irreps='64x0e + 32x1o + 16x2e', num_heads=8):
        super().__init__()

        self.irreps = o3.Irreps(irreps)

        # Self-Attention
        self.self_attn = EquivariantMultiHeadAttention(
            irreps_in=self.irreps,
            irreps_out=self.irreps,
            num_heads=num_heads
        )

        # Feed-Forward (只对标量做非线性)
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps if ir.l == 0])
        irreps_others = o3.Irreps([(mul, ir) for mul, ir in self.irreps if ir.l > 0])

        self.ffn_scalars = nn.Sequential(
            nn.Linear(irreps_scalars.dim, irreps_scalars.dim * 4),
            nn.SiLU(),
            nn.Linear(irreps_scalars.dim * 4, irreps_scalars.dim)
        )

        # Layer Norm (只对标量)
        self.norm1 = nn.LayerNorm(irreps_scalars.dim)
        self.norm2 = nn.LayerNorm(irreps_scalars.dim)

    def forward(self, x, edge_index, edge_sh):
        # Self-Attention + Residual
        attn_out = self.self_attn(x, edge_index, edge_sh)
        x = x + attn_out

        # 提取标量进行归一化和FFN
        # 这里简化，实际需要正确分离和重组

        # Feed-Forward + Residual
        # ...

        return x

class Equiformer(nn.Module):
    """完整的Equiformer模型"""
    def __init__(
        self,
        num_species=100,
        irreps_hidden='64x0e + 32x1o + 16x2e',
        num_layers=6,
        num_heads=8,
        r_max=5.0
    ):
        super().__init__()

        self.irreps_hidden = o3.Irreps(irreps_hidden)

        # 原子Embedding
        self.species_embedding = nn.Embedding(num_species, 64)
        self.embedding_to_irreps = o3.Linear('64x0e', self.irreps_hidden)

        # 球谐函数（边特征）
        self.sh = o3.SphericalHarmonics(
            irreps_out=o3.Irreps.spherical_harmonics(lmax=2),
            normalize=True,
            normalization='component'
        )

        # Equiformer层
        self.layers = nn.ModuleList([
            EquiformerLayer(irreps=self.irreps_hidden, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # 输出头（能量：标量）
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0])
        self.energy_head = nn.Sequential(
            nn.Linear(irreps_scalars.dim, irreps_scalars.dim // 2),
            nn.SiLU(),
            nn.Linear(irreps_scalars.dim // 2, 1)
        )

    def forward(self, data):
        pos = data.pos
        species = data.species
        edge_index = data.edge_index

        # Embedding
        x = self.species_embedding(species)  # [N, 64]
        x = self.embedding_to_irreps(x)  # [N, irreps_hidden]

        # 边特征（球谐函数）
        row, col = edge_index
        edge_vec = pos[row] - pos[col]  # [E, 3]
        edge_sh = self.sh(edge_vec)  # [E, irreps_sh]

        # Equiformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_sh)

        # 输出（只取标量部分）
        # 提取标量
        x_scalar = x[:, :64]  # 假设前64维是标量

        # 预测能量
        atom_energies = self.energy_head(x_scalar).squeeze(-1)  # [N]

        # 按batch求和
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(len(pos), dtype=torch.long)
        total_energies = torch.zeros(batch.max().item() + 1, device=pos.device)
        total_energies.index_add_(0, batch, atom_energies)

        return total_energies
```

---

## 5. 训练与评估

### 5.1 完整训练脚本

```python
# train_equiformer.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from equiformer import Equiformer
import wandb

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        # 前向
        pred_energy = model(batch)
        target_energy = batch.energy

        # 损失
        loss = F.mse_loss(pred_energy, target_energy)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        pred_energy = model(batch)
        target_energy = batch.energy

        loss = F.mse_loss(pred_energy, target_energy)
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)

def main():
    # 配置
    config = {
        'num_species': 100,
        'irreps_hidden': '128x0e + 64x1o + 32x2e',
        'num_layers': 6,
        'num_heads': 8,
        'r_max': 5.0,
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 500,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # W&B
    wandb.init(project='equiformer', config=config)

    # 数据
    train_dataset = ...  # 加载训练数据
    val_dataset = ...  # 加载验证数据

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # 模型
    model = Equiformer(
        num_species=config['num_species'],
        irreps_hidden=config['irreps_hidden'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        r_max=config['r_max']
    ).to(config['device'])

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=20
    )

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, config['device'])
        val_loss = evaluate(model, val_loader, config['device'])

        scheduler.step(val_loss)

        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_equiformer.pth')

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    wandb.finish()

if __name__ == '__main__':
    main()
```

---

## 6. 实际应用

### 6.1 对比Transformer与GNN

```python
# compare_models.py
from atomic_transformer import AtomicTransformer
from schnet import SchNet  # 假设已实现
import time

models = {
    'Transformer': AtomicTransformer(...),
    'SchNet': SchNet(...)
}

# 在相同数据上测试
test_data = ...

for name, model in models.items():
    # 推理速度
    start = time.time()
    with torch.no_grad():
        output = model(test_data)
    end = time.time()

    print(f"{name}:")
    print(f"  Time: {(end-start)*1000:.2f} ms")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
```

### 6.2 注意力可视化

```python
# visualize_attention.py
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_map(model, data):
    """可视化attention权重"""
    model.eval()

    # 提取attention权重（需要修改模型返回attention）
    with torch.no_grad():
        output, attn_weights = model(data, return_attention=True)

    # attn_weights: [num_heads, N, N]
    attn_avg = attn_weights.mean(dim=0).cpu().numpy()  # [N, N]

    plt.figure(figsize=(10, 8))
    plt.imshow(attn_avg, cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Atom Index')
    plt.ylabel('Atom Index')
    plt.title('Average Attention Map')
    plt.savefig('attention_map.pdf')
    plt.show()
```

---

## 总结 | Summary

本实验实现了：

1. **标准Transformer**：从零实现Attention机制
2. **Atomic Transformer**：将Transformer应用于原子系统
3. **Equiformer**：等变Transformer，保持SO(3)对称性

**关键要点**：
- Transformer通过Self-Attention捕捉长程相互作用
- 等变性保证了物理对称性
- Transformer在大体系中可能比GNN更有效

**进一步探索**：
- TorchMD-NET: https://github.com/torchmd/torchmd-net
- Equiformer: https://github.com/atomicarchitects/equiformer
- Graphormer: https://github.com/microsoft/Graphormer

---

*Happy Transforming! 🤖*
