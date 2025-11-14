# Attention is All You Need——Transformer模型
# Transformer Model: Attention is All You Need

## 1. 为什么需要Transformer？

### 1.1 序列建模的挑战

在许多任务中，我们需要处理**序列数据**：
- 📝 自然语言：文本是词的序列
- 🧬 生物序列：DNA、蛋白质是碱基/氨基酸序列
- ⚛️ 原子体系：可以看作原子的序列
- 📊 时间序列：温度、压力等随时间变化

**传统方法**：
- **RNN (循环神经网络)**：顺序处理，慢且难以并行
- **CNN (卷积神经网络)**：局部感受野，难以捕捉长程依赖

**Transformer的优势**：
- ✅ **并行化**：可以同时处理整个序列
- ✅ **长程依赖**：直接建模任意距离的关系
- ✅ **可解释性**：注意力权重可视化

### 1.2 Transformer的核心思想

**"Attention is All You Need"**（注意力机制就是全部）

核心机制：**Self-Attention（自注意力）**
- 每个元素可以"关注"序列中的所有其他元素
- 权重动态计算，反映元素间的相关性

## 2. 注意力机制（Attention Mechanism）

### 2.1 什么是注意力？

**人类的注意力**：
当你阅读这段文字时，你的注意力会聚焦在某些关键词上，而忽略不重要的词。

**神经网络的注意力**：
给定查询（Query），在一组键值对（Key-Value）中找到最相关的信息。

### 2.2 Scaled Dot-Product Attention

**数学定义**：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

其中：
- **Q (Query)**：查询向量，维度 (n, d_k)
- **K (Key)**：键向量，维度 (m, d_k)
- **V (Value)**：值向量，维度 (m, d_v)
- **d_k**：键的维度，用于缩放
- **n**：查询数量（目标序列长度）
- **m**：键值对数量（源序列长度）

**计算步骤**：

1. **计算相似度**：Q和K的点积
   ```
   scores = QK^T / √d_k
   # shape: (n, m)
   ```

2. **归一化**：softmax转换为概率分布
   ```
   attention_weights = softmax(scores)
   # shape: (n, m)，每行和为1
   ```

3. **加权求和**：用权重聚合值
   ```
   output = attention_weights @ V
   # shape: (n, d_v)
   ```

**PyTorch实现**：

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention

    Args:
        Q: Query tensor, shape (batch, n, d_k)
        K: Key tensor, shape (batch, m, d_k)
        V: Value tensor, shape (batch, m, d_v)
        mask: Optional mask, shape (batch, n, m)

    Returns:
        output: Attention output, shape (batch, n, d_v)
        attention_weights: Attention weights, shape (batch, n, m)
    """
    d_k = Q.size(-1)

    # 1. 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    # shape: (batch, n, m)

    # 2. 应用mask（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)
    # shape: (batch, n, m)

    # 4. 加权求和
    output = torch.matmul(attention_weights, V)
    # shape: (batch, n, d_v)

    return output, attention_weights
```

**为什么要除以√d_k？**

- 防止点积结果过大
- 使softmax的梯度更稳定
- 当d_k很大时，点积方差为d_k，除以√d_k使方差为1

### 2.3 Multi-Head Attention（多头注意力）

**动机**：单个注意力头可能只关注某一方面的信息。多个头可以并行学习不同的表示子空间。

**数学定义**：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

参数：
- **h**：头的数量（通常8或16）
- **W_i^Q, W_i^K, W_i^V**：第i个头的投影矩阵
- **W^O**：输出投影矩阵

**维度分析**：

假设模型维度d_model = 512，h = 8个头：
- 每个头的维度：d_k = d_v = d_model / h = 64
- Q, K, V的投影：(d_model, d_k) = (512, 64)
- 输出投影：(h * d_v, d_model) = (512, 512)

**实现**：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V投影（所有头一起投影）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: shape (batch, seq_len, d_model)
            mask: shape (batch, seq_len, seq_len)

        Returns:
            output: shape (batch, seq_len, d_model)
        """
        batch_size = Q.size(0)

        # 1. 线性投影并分割为多个头
        # shape: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 应用注意力
        attn_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        # shape: (batch, num_heads, seq_len, d_k)

        # 3. 拼接所有头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # shape: (batch, seq_len, d_model)

        # 4. 最终线性投影
        output = self.W_O(attn_output)

        return output, attention_weights
```

## 3. Transformer架构

### 3.1 完整架构图

```
输入序列
   ↓
输入嵌入 + 位置编码
   ↓
┌─────────────────────┐
│  Encoder (N层)      │
│  ┌───────────────┐  │
│  │ Multi-Head    │  │
│  │ Self-Attention│  │
│  └───────┬───────┘  │
│          ↓          │
│  ┌───────────────┐  │
│  │ Add & Norm    │  │
│  └───────┬───────┘  │
│          ↓          │
│  ┌───────────────┐  │
│  │ Feed Forward  │  │
│  └───────┬───────┘  │
│          ↓          │
│  ┌───────────────┐  │
│  │ Add & Norm    │  │
│  └───────────────┘  │
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  Decoder (N层)      │
│  （如果需要）       │
└─────────┬───────────┘
          ↓
     输出层
```

### 3.2 Position Encoding（位置编码）

**问题**：Attention机制是permutation-invariant（排列不变），无法区分序列顺序。

**解决**：添加位置信息

**正弦/余弦位置编码**：

```python
def positional_encoding(seq_len, d_model):
    """
    生成正弦余弦位置编码

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: 序列长度
        d_model: 模型维度

    Returns:
        pe: shape (seq_len, d_model)
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

# 使用
pe = positional_encoding(100, 512)
plt.imshow(pe.numpy(), cmap='RdBu', aspect='auto')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding Visualization')
plt.colorbar()
```

**可学习的位置编码**：

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pe(positions)
```

### 3.3 Feed-Forward Network（前馈网络）

每个Transformer层包含一个FFN：

```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

**特点**：
- 在每个位置独立应用（position-wise）
- 两层线性变换 + ReLU激活
- 通常d_ff = 4 * d_model

### 3.4 Layer Normalization & Residual Connection

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # 子层
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Multi-Head Self-Attention + Residual + Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Feed-Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
```

### 3.5 完整的Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

## 4. Transformer用于原子体系建模

### 4.1 为什么Transformer适合原子体系？

**原子体系的特点**：
- 原子之间存在长程相互作用
- 需要全局信息（如总能量）
- 对称性要求（旋转、平移、排列不变）

**Transformer的优势**：
- **全局感受野**：每个原子可以"看到"所有其他原子
- **并行计算**：比RNN快
- **灵活性**：容易加入先验知识

### 4.2 Equivariant Transformer

**挑战**：标准Transformer不满足E(3)等变性

**解决方案**：Equiformer (ICLR 2023)

```python
class EquivariantTransformer(nn.Module):
    """
    E(3)-equivariant Transformer for atomic systems

    使用球谐函数表示方向信息
    """
    def __init__(self, irreps_in, irreps_out, num_layers=6, num_heads=8):
        super().__init__()

        from e3nn import o3

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        # Equivariant attention layers
        self.layers = nn.ModuleList([
            EquivariantTransformerLayer(
                irreps_in=self.irreps_in,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])

    def forward(self, node_features, positions, edge_index):
        """
        Args:
            node_features: (N, C) 节点特征（不可约表示）
            positions: (N, 3) 原子坐标
            edge_index: (2, E) 边索引
        """
        x = node_features

        for layer in self.layers:
            x = layer(x, positions, edge_index)

        return x
```

### 4.3 实际应用：TorchMD-NET

**TorchMD-NET** (NeurIPS 2022)：使用Transformer预测分子性质

```python
from torchmdnet.models import TorchMD_ET

model = TorchMD_ET(
    hidden_channels=128,
    num_layers=6,
    num_rbf=50,
    rbf_type='expnorm',
    trainable_rbf=True,
    activation='silu',
    attn_activation='silu',
    neighbor_embedding=True,
    num_heads=8,
    distance_influence='both',
    cutoff_lower=0.0,
    cutoff_upper=5.0
)

# 预测能量和力
energy, forces = model(z, pos, batch)
```

**特点**：
- 基于Transformer的注意力机制
- 等变性（通过careful设计）
- 在MD17、QM9等基准上SOTA

## 5. Transformer的变体和优化

### 5.1 Efficient Attention

**问题**：标准attention的复杂度是O(n²)，对于长序列很慢。

**解决方案**：

#### Linformer（线性Transformer）
```python
# 将K和V投影到低维
K_proj = K @ projection  # (n, d) -> (k, d), k << n
V_proj = V @ projection
attn = Q @ K_proj.T @ V_proj  # O(nk) instead of O(n²)
```

#### Performer（基于kernel方法）
```python
# 使用随机特征近似softmax
Q_prime = phi(Q)  # 特征映射
K_prime = phi(K)
attn = Q_prime @ (K_prime.T @ V)  # O(n) 复杂度
```

### 5.2 Sparse Attention

**局部注意力**：只关注邻近的token

```python
# 仅计算r范围内的注意力
mask = (distances < r).float()
scores = scores.masked_fill(mask == 0, float('-inf'))
```

### 5.3 Flash Attention

**IO-aware attention**：优化GPU内存访问

```python
# 使用Triton或CUDA实现
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=False)
# 速度提升2-4倍，内存节省10-20倍
```

## 6. 实战示例：序列预测

### 6.1 分子性质预测

```python
class MoleculeTransformer(nn.Module):
    """
    使用Transformer预测分子性质
    """
    def __init__(self, num_atom_types=100, d_model=256, num_layers=6, num_heads=8):
        super().__init__()

        # 原子嵌入
        self.atom_embedding = nn.Embedding(num_atom_types, d_model)

        # 位置编码
        self.pos_encoding = LearnedPositionalEncoding(max_seq_len=200, d_model=d_model)

        # Transformer编码器
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads
        )

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, atom_types, positions=None):
        """
        Args:
            atom_types: (batch, seq_len) 原子类型
            positions: (batch, seq_len, 3) 原子坐标（可选）

        Returns:
            property: (batch,) 预测的性质
        """
        # 1. 原子嵌入
        x = self.atom_embedding(atom_types)  # (batch, seq_len, d_model)

        # 2. 添加位置编码
        x = self.pos_encoding(x)

        # 3. Transformer编码
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # 4. 全局池化
        x = x.mean(dim=1)  # (batch, d_model)

        # 5. 预测
        output = self.output_head(x).squeeze(-1)  # (batch,)

        return output
```

### 6.2 训练循环

```python
# 模型
model = MoleculeTransformer(num_atom_types=100, d_model=256, num_layers=6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练
for epoch in range(100):
    for batch in train_loader:
        atom_types = batch.z
        target = batch.y

        # 前向传播
        pred = model(atom_types)

        # 损失
        loss = F.mse_loss(pred, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

## 7. Transformer vs GNN

### 7.1 对比

| 特性 | GNN (如SchNet) | Transformer |
|------|----------------|-------------|
| 感受野 | 局部（r_cut） | 全局 |
| 复杂度 | O(N × 邻居数) | O(N²) |
| 等变性 | 天然支持 | 需要特殊设计 |
| 并行性 | 中等 | 高 |
| 可解释性 | 中等 | 高（attention weights） |

### 7.2 何时使用Transformer？

**适合Transformer**：
- ✅ 需要捕捉长程依赖
- ✅ 序列数据（时间序列、轨迹）
- ✅ 全局性质预测
- ✅ 有充足的数据

**适合GNN**：
- ✅ 局部相互作用主导
- ✅ 大体系（避免O(N²)复杂度）
- ✅ 需要严格的等变性
- ✅ 数据有限

### 7.3 混合方法

**最佳实践**：结合两者优势

```python
class HybridModel(nn.Module):
    """
    GNN捕捉局部 + Transformer捕捉全局
    """
    def __init__(self):
        super().__init__()
        self.gnn = SchNet(...)  # 局部特征
        self.transformer = TransformerEncoder(...)  # 全局特征

    def forward(self, z, pos, batch):
        # 1. GNN提取局部特征
        local_features = self.gnn.get_features(z, pos, batch)

        # 2. Transformer聚合全局信息
        global_features = self.transformer(local_features)

        # 3. 组合预测
        energy = self.output_net(global_features)

        return energy
```

## 8. 最新进展

### 8.1 Equiformer (ICLR 2023)

- E(3)-equivariant Transformer
- 在OC20、QM9上SOTA
- 使用球谐函数和Clebsch-Gordan张量积

### 8.2 TorchMD-NET (NeurIPS 2022)

- Transformer用于分子动力学
- 等变attention机制
- 开源且易用

### 8.3 GraphGPS (NeurIPS 2022)

- 图Transformer的通用框架
- 结合message passing和attention
- 在多个基准上提升性能

## 参考文献

1. Vaswani et al., "Attention is All You Need", NeurIPS 2017

2. Liao & Smidt, "Equiformer: Equivariant Graph Attention Transformer", ICLR 2023

3. Thölke & De Fabritiis, "TorchMD-NET: Equivariant Transformers for Neural Network based Molecular Potentials", NeurIPS 2022

4. Rampášek et al., "Recipe for a General, Powerful, Scalable Graph Transformer", NeurIPS 2022

---

**下一章**: [05-MACE-Framework.md](./05-MACE-Framework.md)

**返回**: [Part 5主页](../README.md)
