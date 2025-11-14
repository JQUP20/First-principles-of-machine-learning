# 通用原子体系大模型——MACE框架
# Universal Atomic System Foundation Model: MACE Framework

## 1. MACE简介

**MACE** (Multi-Atomic Cluster Expansion) 是当前最先进的神经网络势函数框架之一，由Ilyes Batatia等人在2022-2023年开发。

**核心论文**：
- MACE: Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", NeurIPS 2022
- MACE-OFF: Batatia et al., "A foundation model for atomistic materials chemistry", arXiv:2401.00096 (2024)

### 1.1 MACE的创新点

相比于之前的模型（SchNet、PaiNN、NequIP），MACE的主要创新：

1. **原子簇展开（ACE）**：系统地考虑多体相互作用
2. **更高阶的等变性**：使用多个角动量通道
3. **消息传递 + ACE**：结合MPNN和ACE的优势
4. **基础模型**：在大规模数据上预训练
5. **少样本学习**：只需很少数据就能微调

**性能对比**（MD17数据集，1000训练样本）：

| 模型 | 能量MAE (meV) | 力MAE (meV/Å) |
|------|---------------|---------------|
| SchNet | 8.5 | 33.0 |
| PaiNN | 5.2 | 18.3 |
| NequIP | 2.9 | 8.8 |
| **MACE** | **1.8** | **5.2** |

## 2. 原子簇展开（ACE）方法

### 2.1 从二体到多体

**二体势**（如Lennard-Jones）：
```
E = Σ_{i<j} V_2(r_ij)
```

只考虑原子对，无法描述键角、二面角等。

**三体势**：
```
E = Σ_{i<j<k} V_3(r_ij, r_ik, r_jk)
```

考虑键角，但组合爆炸（N³项）。

**ACE的思想**：
系统地展开为多体项，同时保持：
- ✅ **完备性**：理论上可以表示任意势函数
- ✅ **等变性**：保持旋转对称性
- ✅ **线性复杂度**：O(N)而不是O(N^ν)

### 2.2 ACE的数学表示

**基函数**：

ACE使用**球谐函数**和**径向基函数**的组合：

```
B_{nlm}(r) = R_n(r) Y_l^m(r̂)
```

其中：
- R_n(r)：径向基函数（如Bessel函数）
- Y_l^m(r̂)：球谐函数
- n：径向量子数
- l：角量子数
- m：磁量子数

**原子密度展开**：

```
ρ_i = Σ_j B(r_ij) = Σ_{nlm} c_{nlm} R_n(r_ij) Y_l^m(r̂_ij)
```

**多体相关**：

通过Clebsch-Gordan耦合构造高阶项：

```
A_{i}^{(ν)} = Σ_{j_1...j_ν} B(r_{ij_1}) ⊗ ... ⊗ B(r_{ij_ν})
```

其中⊗表示等变张量积。

**能量表达式**：

```
E = Σ_i Σ_ν w_ν · A_i^{(ν)}
```

其中w_ν是可学习的权重。

### 2.3 ACE vs 神经网络

**传统ACE**：
- 线性模型（快速但表达能力有限）
- 需要手工选择基函数
- 难以处理复杂体系

**神经网络势**（如SchNet）：
- 非线性（强大）
- 自动学习特征
- 但是黑箱，难以理解

**MACE = ACE + NN**：
- 使用ACE的数学框架保证等变性和完备性
- 使用神经网络学习非线性组合
- 最佳的权衡

## 3. MACE架构详解

### 3.1 整体架构

```
输入：原子序数{Z_i}，坐标{r_i}
                ↓
        ┌────────────────┐
        │  原子嵌入层    │
        └────────┬───────┘
                 ↓
        ┌────────────────┐
        │  MACE Layer 1  │ ← 等变消息传递
        └────────┬───────┘
                 ↓
        ┌────────────────┐
        │  MACE Layer 2  │
        └────────┬───────┘
                 ↓
              ...
                 ↓
        ┌────────────────┐
        │  MACE Layer L  │
        └────────┬───────┘
                 ↓
        ┌────────────────┐
        │  ReadOut层     │
        └────────┬───────┘
                 ↓
         原子能量{E_i}
                 ↓
          E_total = ΣE_i
```

### 3.2 MACE消息传递层

**核心思想**：在每一层，每个原子聚合邻居的等变特征。

```python
class MACELayer(nn.Module):
    """
    MACE消息传递层

    输入：原子的等变特征（多个irreps）
    输出：更新后的等变特征
    """
    def __init__(self, irreps_in, irreps_out, correlation, num_elements):
        super().__init__()

        from e3nn import o3

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.correlation = correlation  # 2 or 3（二体或三体相关）

        # 1. 线性层（等变）
        self.linear_up = o3.Linear(self.irreps_in, self.irreps_out)

        # 2. 张量积卷积（核心）
        self.conv = TensorProductConv(
            irreps_node=self.irreps_out,
            irreps_edge=o3.Irreps.spherical_harmonics(lmax=3),
            irreps_out=self.irreps_out
        )

        # 3. 非线性（只对标量）
        self.nonlinearity = Nonlinearity(self.irreps_out)

        # 4. 对称张量积（多体相关）
        if correlation >= 2:
            self.tensor_product = SymmetricTensorProduct(
                irreps_in=self.irreps_out,
                correlation=correlation
            )

    def forward(self, node_features, edge_index, edge_attr):
        """
        Args:
            node_features: (N, C) 节点的等变特征
            edge_index: (2, E) 边索引
            edge_attr: (E, C_edge) 边的等变特征（球谐函数）
        """
        # 1. 线性投影
        h = self.linear_up(node_features)

        # 2. 等变卷积（消息传递）
        h = self.conv(h, edge_index, edge_attr)

        # 3. 非线性激活
        h = self.nonlinearity(h)

        # 4. 对称张量积（多体相关）
        if hasattr(self, 'tensor_product'):
            h = self.tensor_product(h)

        return h
```

### 3.3 关键组件

#### (1) 对称张量积（Symmetric Tensor Product）

**二体相关**（correlation=2）：

```
h^{(2)} = h ⊗ h
```

这里⊗是等变的张量积，保持SO(3)对称性。

**数学细节**：

使用Clebsch-Gordan系数：

```
[Y_{l_1} ⊗ Y_{l_2}]_{LM} = Σ_{m_1,m_2} C_{l_1 m_1, l_2 m_2}^{LM} Y_{l_1}^{m_1} Y_{l_2}^{m_2}
```

其中C是Clebsch-Gordan系数。

**e3nn实现**：

```python
from e3nn import o3

# 两个1o (l=1, 奇宇称)张量积
tp = o3.FullTensorProduct("1o", "1o")
print(tp.irreps_out)
# 输出: 0e + 1o + 2e
# 解释：l1=1, l2=1 → L=0,1,2（三角不等式）
```

#### (2) 等变非线性

标准的ReLU、SiLU等激活函数破坏等变性！

**解决方案**：只对标量（l=0）应用非线性

```python
class Nonlinearity(nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)

        # 找出标量和非标量
        self.scalar_indices = []
        self.nonscalar_indices = []

        idx = 0
        for mul, (l, p) in self.irreps:
            if l == 0:
                self.scalar_indices.extend(range(idx, idx + mul))
            else:
                self.nonscalar_indices.extend(range(idx, idx + mul * (2*l+1)))
            idx += mul * (2*l+1)

    def forward(self, x):
        out = x.clone()
        # 只对标量应用激活
        out[..., self.scalar_indices] = F.silu(x[..., self.scalar_indices])
        # 非标量保持不变（或归一化）
        return out
```

#### (3) ReadOut层

最后，将等变特征转换为标量能量：

```python
class ReadOut(nn.Module):
    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)

        # 只提取标量部分
        self.irreps_scalar = o3.Irreps([(mul, (0, p)) for mul, (l, p) in self.irreps_in if l == 0])

        # 线性层：标量 → 原子能量
        self.linear = o3.Linear(self.irreps_scalar, o3.Irreps("1x0e"))

    def forward(self, node_features):
        # 提取标量
        scalar_features = extract_scalars(node_features, self.irreps_in)

        # 预测原子能量
        atomic_energies = self.linear(scalar_features).squeeze(-1)

        return atomic_energies
```

## 4. MACE的训练

### 4.1 损失函数

**多任务学习**：

```python
loss = λ_E · loss_energy + λ_F · loss_forces + λ_S · loss_stress + λ_V · loss_virials

其中：
  loss_energy = MSE(E_pred, E_true) / N_atoms
  loss_forces = MSE(F_pred, F_true)
  loss_stress = MSE(σ_pred, σ_true)  # 应力张量
  loss_virials = MSE(Ξ_pred, Ξ_true)  # Virial张量
```

**权重设置**（MACE论文）：

```python
config = {
    'energy_weight': 1.0,
    'forces_weight': 100.0,  # 力更重要
    'stress_weight': 10.0,
    'virials_weight': 1.0
}
```

### 4.2 数据增强

**旋转增强**：

```python
def rotate_system(atoms, energy, forces):
    """随机旋转整个体系"""
    from scipy.spatial.transform import Rotation

    R = Rotation.random().as_matrix()

    positions_rotated = atoms.positions @ R.T
    forces_rotated = forces @ R.T

    # 能量是标量，不变
    return positions_rotated, energy, forces_rotated
```

**平移增强**（自动满足，无需额外操作）

**排列增强**：

```python
# 随机打乱原子顺序
indices = torch.randperm(N_atoms)
atoms_permuted = atoms[indices]
```

### 4.3 训练技巧

**1. 学习率调度**：

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.8,
    patience=50,
    min_lr=1e-6
)
```

**2. 能量/力的缩放**：

```python
# 数据标准化
energy_mean = train_dataset.energies.mean()
energy_std = train_dataset.energies.std()

energy_normalized = (energy - energy_mean) / energy_std
```

**3. 梯度裁剪**：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

**4. EMA (指数移动平均)**：

```python
from torch_ema import ExponentialMovingAverage

ema = ExponentialMovingAverage(model.parameters(), decay=0.99)

# 训练循环
for batch in train_loader:
    loss.backward()
    optimizer.step()
    ema.update()  # 更新EMA参数

# 评估时使用EMA模型
with ema.average_parameters():
    evaluate(model, val_loader)
```

## 5. MACE基础模型（Foundation Model）

### 5.1 MACE-MP-0

**MACE-MP-0**：在**Materials Project**数据集上预训练的大模型

**训练数据**：
- 150万个晶体结构
- 5000万个能量/力计算
- 覆盖118种元素

**模型规模**：
- 1亿参数
- 6层MACE层
- correlation=3（三体相关）

**性能**：
- MD17（微调100样本）：能量MAE < 1 meV
- SPICE（有机分子）：力MAE < 5 meV/Å
- 泛化能力极强

### 5.2 使用预训练模型

```python
from mace import pretrained

# 加载MACE-MP-0
model = pretrained.mace_mp_0(device='cuda')

# 推理
from ase import Atoms

atoms = Atoms(
    'H2O',
    positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
)

energy = model.get_potential_energy(atoms)
forces = model.get_forces(atoms)

print(f"Energy: {energy:.4f} eV")
print(f"Forces:\n{forces}")
```

### 5.3 微调（Fine-tuning）

**场景**：你有一个新体系，只有少量DFT数据（如50-100个构型）

```python
# 1. 加载预训练模型
model = pretrained.mace_mp_0()

# 2. 冻结底层
for name, param in model.named_parameters():
    if 'layer_0' in name or 'layer_1' in name:
        param.requires_grad = False  # 冻结前两层

# 3. 微调
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4  # 小学习率
)

for epoch in range(100):
    for batch in small_dataset:
        # 标准训练循环
        ...
```

**微调 vs 从头训练**：

| 方法 | 需要数据量 | 训练时间 | 精度 |
|------|-----------|---------|------|
| 从头训练 | 1000+ | 数小时 | 好 |
| 微调MACE-MP-0 | 50-100 | <1小时 | **更好** |

## 6. MACE vs 其他模型

### 6.1 架构对比

| 特性 | SchNet | NequIP | MACE |
|------|--------|--------|------|
| 等变性 | ❌ 不变 | ✅ 等变 | ✅ 等变 |
| 多体相关 | ❌ 隐式 | ⚠️ 二体 | ✅ 三体+ |
| 理论基础 | 经验 | 群论 | ACE |
| 数据效率 | 低 | 中 | **高** |
| 泛化能力 | 中 | 好 | **极好** |
| 预训练 | ❌ | ❌ | ✅ |

### 6.2 性能基准

**QM9数据集**（1000训练样本）：

```
能量MAE (meV):
  SchNet:  14.0
  PaiNN:    8.5
  NequIP:   5.2
  MACE:     2.8  ← 最佳
```

**MD17 Aspirin**（1000训练样本）：

```
力MAE (meV/Å):
  SchNet:  33.0
  PaiNN:   18.3
  NequIP:   8.8
  MACE:     5.2  ← 最佳
```

## 7. MACE的局限性和未来方向

### 7.1 当前局限

**1. 计算成本**：
- 张量积操作复杂度高
- 对于超大体系（>1000原子）较慢

**解决方向**：
- 稀疏化
- 低秩近似
- 专用硬件加速

**2. 长程相互作用**：
- 仍然基于截断半径
- 需要额外模块处理电荷

**解决方向**：
- 集成Ewald求和
- 学习长程修正

**3. 磁性材料**：
- 标准MACE不处理自旋

**解决方向**：
- SpinMACE（自旋等变）

### 7.2 研究前沿

**1. Multi-scale MACE**：
- 同时建模电子和原子尺度

**2. MACE for Reactions**：
- 化学反应势能面
- 过渡态搜索

**3. Uncertainty Quantification**：
- 贝叶斯MACE
- 集成学习

## 8. 实践建议

### 8.1 何时使用MACE？

**推荐场景**：
- ✅ 小数据集（<1000样本）→ 使用MACE-MP-0微调
- ✅ 需要极高精度
- ✅ 多种元素混合体系
- ✅ 需要泛化到新结构

**不推荐场景**：
- ❌ 超大体系（>5000原子）→ 考虑SchNet
- ❌ 实时推理（毫秒级）→ 考虑更快的模型
- ❌ 只有CPU → MACE对GPU优化

### 8.2 超参数选择

```python
# 小分子（<30原子）
config_small = {
    'num_layers': 3,
    'max_ell': 2,  # lmax
    'correlation': 2,
    'hidden_irreps': '64x0e + 32x1o + 16x2e',
    'r_max': 5.0
}

# 晶体材料
config_crystal = {
    'num_layers': 4,
    'max_ell': 3,
    'correlation': 3,
    'hidden_irreps': '128x0e + 64x1o + 32x2e + 16x3o',
    'r_max': 6.0
}
```

## 参考文献

1. **MACE**: Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", NeurIPS 2022

2. **MACE-OFF**: Batatia et al., "A foundation model for atomistic materials chemistry", arXiv:2401.00096 (2024)

3. **ACE**: Drautz, "Atomic cluster expansion for accurate and transferable interatomic potentials", Phys. Rev. B 99, 014104 (2019)

4. **e3nn**: Geiger & Smidt, "e3nn: Euclidean neural networks", arXiv:2207.09453

---

**下一步**: 查看 [MACE实战教程](../02-Practice/04-MACE/)

**返回**: [Part 5主页](../README.md)
