# 常见的等变模型——PaiNN、NequIP和Allegro模型
# Common Equivariant Models: PaiNN, NequIP and Allegro

## 目录
- [PaiNN——通过距离矩阵实现等变性](#painn通过距离矩阵实现等变性)
- [NequIP和Allegro——通过不可约表示实现等变性](#nequip和allegro通过不可约表示实现等变性)
- [易于大规模并行的等变模型——Allegro](#易于大规模并行的等变模型allegro)

---

## PaiNN——通过距离矩阵实现等变性

### PaiNN简介

**PaiNN** (Polarizable Atom Interaction Neural Network) 由Schütt等人于2021年提出。

**核心思想**：使用标量特征和向量特征的组合，通过巧妙的操作保持等变性。

**论文**：*Equivariant message passing for the prediction of tensorial properties and molecular spectra*, ICML 2021

### PaiNN架构

**节点特征**：
- **标量特征** $\mathbf{s}_i \in \mathbb{R}^F$：旋转不变
- **向量特征** $\mathbf{V}_i \in \mathbb{R}^{F \times 3}$：旋转等变

**两阶段更新**：
1. **消息传递**：从邻居收集信息
2. **特征更新**：标量和向量的自相互作用

### 消息传递阶段

#### 标量消息

$$
\Delta \mathbf{s}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{s}_j \odot \phi_s(r_{ij})
$$

其中 $\phi_s$ 是径向基函数网络。

#### 向量消息

$$
\Delta \mathbf{V}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{V}_j \odot \phi_v(r_{ij}) + \mathbf{s}_j \otimes \frac{\mathbf{r}_{ij}}{r_{ij}} \odot \phi_{vv}(r_{ij})
$$

**关键**：
- 第一项：等变向量的加权和
- 第二项：标量通过方向向量 $\hat{\mathbf{r}}_{ij}$ 提升为向量

### 特征更新阶段

**标量更新**（包含向量信息）：

$$
\mathbf{s}_i \leftarrow \mathbf{s}_i + U(\mathbf{s}_i) + U_V(\|\mathbf{V}_i\|^2)
$$

其中 $\|\mathbf{V}_i\|^2$ 是向量特征的模平方（不变量）。

**向量更新**（由标量调制）：

$$
\mathbf{V}_i \leftarrow \mathbf{V}_i + U_s(\mathbf{s}_i) \mathbf{V}_i
$$

标量 $U_s(\mathbf{s}_i)$ 作为门控因子。

### 保持等变性的操作

**允许的操作**：

1. **向量的线性组合**：
   $$
   \mathbf{V}^{new} = \alpha \mathbf{V}_1 + \beta \mathbf{V}_2
   $$

2. **标量乘向量**：
   $$
   \mathbf{V}^{new} = s \mathbf{V}
   $$

3. **向量到标量**（范数）：
   $$
   s = \|\mathbf{V}\|
   $$

4. **标量通过方向提升**：
   $$
   \mathbf{V} = s \hat{\mathbf{r}}
   $$

**禁止的操作**：
- ❌ 向量的逐元素乘法（破坏等变性）
- ❌ 向量通过非线性激活函数

### PaiNN实现伪代码

```python
class PaiNNLayer(nn.Module):
    def __init__(self, n_atom_basis):
        self.n_basis = n_atom_basis

        # 消息网络
        self.phi_s = nn.Sequential(...)
        self.phi_v = nn.Sequential(...)

        # 更新网络
        self.U = nn.Sequential(...)
        self.U_V = nn.Sequential(...)

    def forward(self, s, V, edge_index, edge_attr, r_ij):
        # edge_attr: RBF(r_ij)
        # r_ij: 方向向量 (num_edges, 3)

        # === 消息传递阶段 ===

        # 标量消息
        scalar_msg = s[edge_index[1]] * self.phi_s(edge_attr)
        delta_s = scatter_sum(scalar_msg, edge_index[0], dim=0)

        # 向量消息
        # 1. 向量传递
        vector_msg_1 = V[edge_index[1]] * self.phi_v(edge_attr).unsqueeze(-1)

        # 2. 标量提升
        r_hat = r_ij / torch.norm(r_ij, dim=1, keepdim=True)
        vector_msg_2 = s[edge_index[1]].unsqueeze(-1) * r_hat.unsqueeze(1)
        vector_msg_2 = vector_msg_2 * self.phi_vv(edge_attr).unsqueeze(-1)

        delta_V = scatter_sum(vector_msg_1 + vector_msg_2,
                             edge_index[0], dim=0)

        # 应用消息
        s = s + delta_s
        V = V + delta_V

        # === 更新阶段 ===

        # 向量范数（不变量）
        V_norm = torch.sum(V ** 2, dim=-1)  # (N, F)

        # 标量更新
        s = s + self.U(s) + self.U_V(V_norm)

        # 向量更新（门控）
        gate = self.U_s(s)  # (N, F)
        V = V + gate.unsqueeze(-1) * V

        return s, V
```

### PaiNN的优点

- ✅ 简单直观的等变设计
- ✅ 计算效率高（只有向量，无高阶张量）
- ✅ 比SchNet准确30-50%
- ✅ 易于实现和理解

### PaiNN的局限

- ❌ 只考虑距离和方向，不考虑角度
- ❌ 对强方向性体系（如氢键）可能不够准确

---

## NequIP和Allegro——通过不可约表示实现等变性

### NequIP简介

**NequIP** (Neural Equivariant Interatomic Potentials) 由Batzner等人于2022年提出。

**核心创新**：使用E(3)等变图神经网络和张量积。

**论文**：*E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials*, Nature Communications 2022

### 球谐张量积表示

**特征表示**：

节点特征是不可约表示的直和：
$$
\mathbf{h}_i = \mathbf{h}_i^{(0)} \oplus \mathbf{h}_i^{(1)} \oplus \mathbf{h}_i^{(2)} \oplus ...
$$

其中：
- $\mathbf{h}_i^{(0)}$：标量（$l=0$，维度1）
- $\mathbf{h}_i^{(1)}$：向量（$l=1$，维度3）
- $\mathbf{h}_i^{(2)}$：秩2张量（$l=2$，维度5）
- ...

### 张量积卷积

**核心操作**：张量积(Tensor Product)

$$
\mathbf{m}_{ij} = \mathbf{h}_j \otimes Y(\hat{\mathbf{r}}_{ij}) \otimes W(r_{ij})
$$

其中：
- $Y(\hat{\mathbf{r}}_{ij})$：球谐函数（编码方向）
- $W(r_{ij})$：径向网络（可学习）

**张量积分解**：

$$
l_1 \otimes l_2 = \bigoplus_{l=|l_1-l_2|}^{l_1+l_2} l
$$

例如：
$$
\text{scalar}(0) \otimes \text{vector}(1) = \text{vector}(1)
$$
$$
\text{vector}(1) \otimes \text{vector}(1) = \text{scalar}(0) \oplus \text{vector}(1) \oplus \text{tensor}(2)
$$

### NequIP层

```python
class NequIPLayer(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        # irreps_in: 输入不可约表示，如 "32x0e + 16x1o + 8x2e"
        # irreps_out: 输出不可约表示

        # 张量积
        self.tp = TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2="1o",  # 边球谐函数
            irreps_out=irreps_out,
            instructions=[...]  # Clebsch-Gordan分解规则
        )

        # 自相互作用
        self.linear = Linear(irreps_in, irreps_out)

    def forward(self, node_features, edge_index, edge_sh, edge_weight):
        # node_features: 节点特征（各种l）
        # edge_sh: 边的球谐函数 Y(r_hat)
        # edge_weight: 径向网络 W(r)

        # 张量积卷积
        messages = self.tp(node_features[edge_index[1]],
                          edge_sh,
                          edge_weight)

        # 聚合
        out = scatter_sum(messages, edge_index[0], dim=0)

        # 自相互作用
        out = out + self.linear(node_features)

        return out
```

### 球谐函数编码方向

**计算球谐函数**：

对于方向向量 $\hat{\mathbf{r}} = (x, y, z)$：

$$
\begin{aligned}
Y_0^0 &= 1 \\
Y_1^{-1} &= y \\
Y_1^0 &= z \\
Y_1^1 &= x \\
Y_2^{-2} &= \sqrt{3}xy \\
Y_2^{-1} &= \sqrt{3}yz \\
&\vdots
\end{aligned}
$$

### NequIP vs PaiNN

| 特性 | PaiNN | NequIP |
|-----|-------|--------|
| **表示** | 标量+向量 | 多阶不可约表示 |
| **角度信息** | 间接（通过方向） | 显式（球谐函数） |
| **准确性** | 高 | 更高 |
| **复杂度** | O(N×E×F) | O(N×E×F×L²) |
| **实现难度** | 中 | 高 |

### Allegro简介

**Allegro**是NequIP的加速版本，专为大规模并行设计。

**改进**：
- 优化的张量积实现
- GPU友好的内存布局
- 高效的消息传递

**性能**：
- 比NequIP快2-5倍
- 可扩展到百万原子

---

## 易于大规模并行的等变模型——Allegro

### Allegro的设计目标

1. **高吞吐量**：最大化GPU利用率
2. **内存效率**：减少内存占用
3. **可扩展性**：支持大规模系统

### 关键优化

#### 1. 严格局部性

**限制感受野**：
- 只考虑最近邻
- 固定的最大邻居数

**好处**：
- 规则的内存访问模式
- 易于批处理

#### 2. 优化的张量积

**预计算**：
- 预计算Clebsch-Gordan系数
- 缓存球谐函数

**融合操作**：
- 将多个小操作合并为大kernel
- 减少内存传输

#### 3. 混合精度训练

**策略**：
- 前向传播：FP16
- 反向传播：FP32（关键部分）
- 梯度：FP16

**加速比**：2-3×

### Allegro架构

```
输入: 原子坐标和类型
  ↓
嵌入层 (多阶表示)
  ↓
Allegro层 ×N
  ├─ 局部环境编码
  ├─ 优化的张量积
  ├─ 两体和多体相互作用
  └─ 残差连接
  ↓
读出层
  ↓
能量 (标量)
```

### 性能基准

**MD22数据集**（大分子）：

| 模型 | 训练时间 | 推理速度 | 力MAE |
|------|---------|---------|-------|
| NequIP | 10 h | 100 steps/s | 0.08 |
| Allegro | 4 h | 400 steps/s | 0.08 |

相同准确度，速度提升4×！

### 使用Allegro

#### 安装

```bash
# 安装依赖
pip install torch-geometric e3nn

# 安装Allegro
git clone https://github.com/mir-group/allegro.git
cd allegro
pip install -e .
```

#### 配置文件

```yaml
# allegro_config.yaml
model:
  num_layers: 4
  max_ell: 2  # 最大角动量
  parity: true
  num_features: 32
  env_embed_dim: 64
  two_body_latent_dim: 64
  avg_num_neighbors: 30

training:
  batch_size: 10
  learning_rate: 0.005
  max_epochs: 10000
  loss:
    energy_weight: 1.0
    force_weight: 100.0
```

#### 训练

```bash
allegro-train \
    --config allegro_config.yaml \
    --dataset path/to/data \
    --output-dir ./checkpoints
```

### 与LAMMPS集成

```lammps
# LAMMPS脚本
pair_style allegro
pair_coeff * * deployed_model.pth Cu O

# 运行MD
fix 1 all nvt temp 300 300 0.1
run 100000
```

### Allegro的应用

**1. 大规模MD模拟**：
- 百万原子体系
- 纳秒时间尺度

**2. 高通量筛选**：
- 快速评估候选材料
- 并行计算数千个体系

**3. 在线学习**：
- 边模拟边训练
- 主动学习流程

---

## 模型总结对比

| 模型 | 准确性 | 速度 | 内存 | 实现难度 | 适用场景 |
|------|-------|------|------|----------|---------|
| **SchNet** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | 大系统、快速筛选 |
| **PaiNN** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 均衡的选择 |
| **NequIP** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 高精度需求 |
| **Allegro** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 大规模生产 |

### 选择建议

**SchNet**：
- 快速原型
- 大规模系统（>10K原子）
- 只需要能量

**PaiNN**：
- 中等规模系统
- 需要力和能量
- 平衡准确性和速度

**NequIP**：
- 小分子高精度
- 研究用途
- 数据有限

**Allegro**：
- 生产环境
- 大规模MD
- GPU集群

---

## 小结

1. **PaiNN**通过标量和向量特征实现等变性，简单高效
2. **NequIP**使用完整的SO(3)不可约表示，准确度最高
3. **Allegro**是NequIP的优化版，适合大规模并行计算

---

## 扩展阅读

1. Schütt, K. T., et al. (2021). Equivariant message passing for the prediction of tensorial properties and molecular spectra. *ICML*.
2. Batzner, S., et al. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*.
3. Musaelian, A., et al. (2023). Learning local equivariant representations for large-scale atomistic dynamics. *Nature Communications*.
4. [e3nn Documentation](https://docs.e3nn.org/)

---

**上一节**: [从不变性到等变性](./01-Equivariance.md)
**返回**: [Part 4主页](../README.md)
