# 神经网络势函数中的长程相互作用
# Long-Range Interactions in Neural Network Potentials

## 1. 为什么需要长程相互作用？

### 1.1 局域近似的局限性

在之前的课程中，我们学习的神经网络势函数（如SchNet、PaiNN、NequIP）都基于**局域近似**：

```
E_total = Σ E_i(r_i, {r_j | |r_j - r_i| < r_cut})
```

这个假设在以下情况下是合理的：
- ✅ 共价键体系（Si、C等）
- ✅ 金属体系（Cu、Al等）
- ✅ 小分子（不带电荷）

但在以下情况下会失效：
- ❌ 离子晶体（NaCl、MgO等）
- ❌ 带电分子和溶液体系
- ❌ 电解质和电池材料
- ❌ 生物大分子（蛋白质、DNA）

### 1.2 长程相互作用的物理本质

**库仑相互作用**（Coulomb interaction）：

```
E_coulomb = (1/2) Σ_i Σ_j (q_i q_j) / |r_i - r_j|
```

特点：
- **长程性**：~1/r衰减，理论上无限远
- **各向同性**：只依赖于距离
- **主导作用**：在离子体系中能量占比可达50%以上

**为什么神经网络难以学习？**

1. **截断半径限制**：r_cut = 5Å时，只能看到近邻原子
2. **周期性边界条件**：需要Ewald求和处理无限多镜像
3. **电荷守恒**：总电荷必须守恒
4. **屏蔽效应**：在金属中库仑相互作用被屏蔽

## 2. 长程相互作用的处理方法

### 2.1 传统方法：Ewald求和

**基本思想**：将长程相互作用分解为实空间和倒空间两部分。

**Ewald求和公式**：

```
E_coulomb = E_real + E_reciprocal + E_self

E_real = (1/2) Σ_{i,j,n} q_i q_j erfc(α|r_ij + n|) / |r_ij + n|

E_reciprocal = (1/2πV) Σ_{k≠0} (|Σ_i q_i exp(ik·r_i)|²/k²) exp(-k²/4α²)

E_self = -(α/√π) Σ_i q_i²
```

其中：
- α：分离参数，控制实空间和倒空间的收敛速度
- n：晶格矢量
- k：倒空间矢量

**优点**：
- 精确计算周期性体系的库仑能
- 收敛速度快（O(N^1.5)）

**缺点**：
- 需要已知所有原子电荷
- 计算复杂，难以与神经网络结合

### 2.2 机器学习方法框架

**核心思路**：将长程相互作用分为两步

```
第一步：神经网络预测原子电荷
  {r_i} → NN → {q_i}

第二步：用电荷计算长程能量
  {q_i, r_i} → Ewald/PME → E_long
```

**总能量**：

```
E_total = E_short(NN) + E_long(q, Ewald)

其中：
  E_short：短程能量（神经网络直接预测）
  E_long：长程库仑能（通过预测的电荷计算）
```

## 3. 神经网络预测原子电荷

### 3.1 电荷的物理意义

**Mulliken电荷**：基于原子轨道的电荷分配

```
q_i^Mulliken = Z_i - Σ_μ∈i (PS)_μμ
```

**Hirshfeld电荷**：基于原子密度的划分

```
q_i^Hirshfeld = Z_i - ∫ ρ(r) w_i(r) dr
```

**Bader电荷**：基于电荷密度拓扑分析

```
q_i^Bader = Z_i - ∫_{Ω_i} ρ(r) dr
```

在机器学习中，我们通常使用**部分电荷（partial charges）**，它们满足：

1. **电荷守恒**：Σ_i q_i = Q_total
2. **旋转不变性**：电荷是标量
3. **平移不变性**：电荷不依赖于坐标原点

### 3.2 电荷预测的神经网络架构

#### 方法1：直接预测（SchNet + Charge）

```python
class SchNetWithCharges(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.schnet = SchNet(...)  # 标准SchNet

        # 电荷预测头
        self.charge_net = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.SiLU(),
            nn.Linear(n_features // 2, 1)
        )

    def forward(self, z, pos, batch):
        # 原子特征
        h = self.schnet.get_features(z, pos, batch)

        # 预测原子电荷
        q_raw = self.charge_net(h).squeeze(-1)

        # 电荷守恒约束
        from torch_scatter import scatter
        q_sum = scatter(q_raw, batch, dim=0, reduce='sum')
        q_correction = scatter(q_sum[batch] / scatter(torch.ones_like(q_raw), batch, dim=0, reduce='sum')[batch],
                               batch, dim=0, reduce='mean')
        q = q_raw - q_correction[batch]

        # 短程能量
        E_short = self.schnet.energy_head(h)

        # 长程能量（Ewald求和）
        E_long = self.compute_ewald(q, pos, batch)

        return E_short + E_long, q
```

#### 方法2：等变电荷预测（PaiNN + Charge）

对于PaiNN等等变模型，电荷仍然是**标量**，因此只使用标量特征：

```python
class PaiNNWithCharges(nn.Module):
    def forward(self, z, pos, batch):
        # PaiNN前向传播
        s, V = self.painn(z, pos, batch)  # s: 标量, V: 矢量

        # 只用标量特征预测电荷
        q = self.charge_net(s).squeeze(-1)

        # 电荷守恒
        q = self.apply_charge_conservation(q, batch)

        # 短程 + 长程
        E_short = self.energy_net(s)
        E_long = self.compute_ewald(q, pos, batch)

        return E_short + E_long, q
```

### 3.3 电荷守恒约束

**方法1：后处理校正**

```python
def apply_charge_conservation(q_raw, batch, total_charge=0):
    """
    确保每个分子的总电荷正确

    Args:
        q_raw: 原始预测电荷 (num_atoms,)
        batch: 批次索引 (num_atoms,)
        total_charge: 期望的总电荷（标量或与batch匹配的张量）
    """
    from torch_scatter import scatter

    # 计算每个分子的预测总电荷
    q_sum = scatter(q_raw, batch, dim=0, reduce='sum')

    # 计算每个分子的原子数
    n_atoms = scatter(torch.ones_like(q_raw), batch, dim=0, reduce='sum')

    # 计算校正量（均匀分配）
    correction = (q_sum - total_charge) / n_atoms

    # 应用校正
    q_corrected = q_raw - correction[batch]

    return q_corrected
```

**方法2：约束优化**

```python
def charge_preserving_layer(q_raw, batch):
    """
    使用拉格朗日乘子确保电荷守恒
    """
    from torch_scatter import scatter

    # 电荷守恒约束：Σ q_i = 0
    # 使用投影到约束子空间

    q_mean = scatter(q_raw, batch, dim=0, reduce='mean')
    q_centered = q_raw - q_mean[batch]

    return q_centered
```

**方法3：电荷平衡损失**

```python
def charge_balance_loss(q, batch, total_charge=0):
    """
    软约束：惩罚电荷不守恒
    """
    from torch_scatter import scatter

    q_sum = scatter(q, batch, dim=0, reduce='sum')
    loss = F.mse_loss(q_sum, torch.zeros_like(q_sum) + total_charge)

    return loss
```

## 4. 长程能量的高效计算

### 4.1 Ewald求和的实现

```python
def ewald_summation(charges, positions, cell, alpha=0.25, k_max=5):
    """
    Ewald求和计算库仑能

    Args:
        charges: (N,) 原子电荷
        positions: (N, 3) 原子坐标
        cell: (3, 3) 晶胞矩阵
        alpha: 分离参数
        k_max: 倒空间截断

    Returns:
        energy: 库仑能 (eV)
    """
    import numpy as np
    from scipy.special import erfc

    N = len(charges)
    volume = np.linalg.det(cell)

    # 1. 实空间部分
    E_real = 0.0
    r_cut = 3.0 / alpha  # 实空间截断半径

    for i in range(N):
        for j in range(i+1, N):
            r_ij = positions[j] - positions[i]

            # 考虑周期性镜像
            for n1 in range(-2, 3):
                for n2 in range(-2, 3):
                    for n3 in range(-2, 3):
                        n = np.array([n1, n2, n3])
                        r = r_ij + cell.T @ n
                        dist = np.linalg.norm(r)

                        if dist < r_cut and dist > 1e-8:
                            E_real += charges[i] * charges[j] * erfc(alpha * dist) / dist

    # 2. 倒空间部分
    E_reciprocal = 0.0
    reciprocal_cell = 2 * np.pi * np.linalg.inv(cell).T

    for k1 in range(-k_max, k_max+1):
        for k2 in range(-k_max, k_max+1):
            for k3 in range(-k_max, k_max+1):
                if k1 == 0 and k2 == 0 and k3 == 0:
                    continue

                k = np.array([k1, k2, k3])
                k_vec = reciprocal_cell @ k
                k2 = np.dot(k_vec, k_vec)

                # 结构因子
                S_k = np.sum(charges * np.exp(1j * k_vec @ positions.T))

                E_reciprocal += (np.abs(S_k)**2 / k2) * np.exp(-k2 / (4 * alpha**2))

    E_reciprocal /= (2 * np.pi * volume)

    # 3. 自作用修正
    E_self = -alpha / np.sqrt(np.pi) * np.sum(charges**2)

    # 4. 背景电荷修正（如果体系带电）
    Q_total = np.sum(charges)
    E_background = -np.pi * Q_total**2 / (2 * volume * alpha**2)

    # 单位转换：Hartree to eV
    E_total = (E_real + E_reciprocal + E_self + E_background) * 27.211

    return E_total
```

### 4.2 PyTorch可微分实现

为了能够反向传播，我们需要PyTorch版本：

```python
class EwaldEnergy(nn.Module):
    """
    可微分的Ewald求和
    """
    def __init__(self, alpha=0.25, k_max=5):
        super().__init__()
        self.alpha = alpha
        self.k_max = k_max

        # 预计算k向量
        self.register_buffer('k_vectors', self._generate_k_vectors())

    def _generate_k_vectors(self):
        """生成倒空间k向量"""
        k_list = []
        for k1 in range(-self.k_max, self.k_max+1):
            for k2 in range(-self.k_max, self.k_max+1):
                for k3 in range(-self.k_max, self.k_max+1):
                    if k1 == 0 and k2 == 0 and k3 == 0:
                        continue
                    k_list.append([k1, k2, k3])
        return torch.tensor(k_list, dtype=torch.float32)

    def forward(self, charges, positions, cell):
        """
        Args:
            charges: (N,) 或 (batch_size, N)
            positions: (N, 3) 或 (batch_size, N, 3)
            cell: (3, 3) 或 (batch_size, 3, 3)
        """
        # 实空间（简化版，只考虑原胞）
        E_real = self._real_space(charges, positions)

        # 倒空间
        E_reciprocal = self._reciprocal_space(charges, positions, cell)

        # 自作用
        E_self = -self.alpha / torch.sqrt(torch.tensor(torch.pi)) * torch.sum(charges**2)

        return E_real + E_reciprocal + E_self

    def _real_space(self, charges, positions):
        """实空间部分（简化）"""
        from torch_cluster import radius_graph

        # 构建近邻图
        edge_index = radius_graph(positions, r=3.0/self.alpha)
        row, col = edge_index

        # 距离
        r_ij = positions[row] - positions[col]
        dist = torch.norm(r_ij, dim=1)

        # erfc函数
        from scipy.special import erfc
        erfc_values = torch.tensor([erfc(self.alpha * d.item()) for d in dist])

        # 能量
        E = charges[row] * charges[col] * erfc_values / dist

        return 0.5 * torch.sum(E)  # 0.5避免重复计数

    def _reciprocal_space(self, charges, positions, cell):
        """倒空间部分"""
        volume = torch.det(cell)
        reciprocal_cell = 2 * torch.pi * torch.inverse(cell).T

        E = 0.0
        for k in self.k_vectors:
            k_vec = reciprocal_cell @ k
            k2 = torch.sum(k_vec**2)

            # 结构因子
            S_k = torch.sum(charges * torch.exp(1j * k_vec @ positions.T))

            E += (torch.abs(S_k)**2 / k2) * torch.exp(-k2 / (4 * self.alpha**2))

        return E / (2 * torch.pi * volume)
```

### 4.3 更高效的方法：PME (Particle Mesh Ewald)

对于大体系，PME更高效（O(N log N)）：

```python
# 使用现成的库
from ase.calculators.lammps import LAMMPS

calc = LAMMPS(
    pair_style='coul/long 12.0',
    kspace_style='pppm 1.0e-4'
)
```

## 5. 训练策略

### 5.1 多任务学习

同时训练能量、力和电荷：

```python
def train_with_charges(model, batch, optimizer):
    optimizer.zero_grad()

    # 前向传播
    E_pred, q_pred = model(batch.z, batch.pos, batch.batch)

    # 目标
    E_true = batch.energy
    q_true = batch.charges  # 从DFT计算获得
    F_true = batch.forces

    # 计算力
    F_pred = -torch.autograd.grad(
        E_pred.sum(), batch.pos,
        create_graph=True
    )[0]

    # 多任务损失
    loss_energy = F.mse_loss(E_pred, E_true)
    loss_forces = F.mse_loss(F_pred, F_true)
    loss_charges = F.mse_loss(q_pred, q_true)
    loss_charge_balance = charge_balance_loss(q_pred, batch.batch)

    # 总损失
    loss = loss_energy + 100 * loss_forces + 10 * loss_charges + loss_charge_balance

    loss.backward()
    optimizer.step()

    return loss.item()
```

### 5.2 课程学习（Curriculum Learning）

```python
# 阶段1：先训练短程能量和电荷
for epoch in range(100):
    for batch in train_loader:
        # 只优化短程和电荷
        loss = loss_short + loss_charges

# 阶段2：联合训练短程和长程
for epoch in range(100, 200):
    for batch in train_loader:
        # 优化全部
        loss = loss_short + loss_long + loss_charges
```

## 6. 实际应用案例

### 6.1 离子液体

**挑战**：
- 强库仑相互作用
- 复杂的离子配对
- 动态的电荷转移

**解决方案**：
```python
# PhysNet模型（包含电荷预测和长程）
from physnet import PhysNet

model = PhysNet(
    n_atom_basis=128,
    n_interactions=5,
    cutoff=10.0,
    predict_charges=True,
    use_electrostatics=True
)
```

### 6.2 电解质溶液

**示例**：NaCl水溶液

```python
# 训练数据需要包含DFT电荷
# 使用Bader电荷分析或Hirshfeld电荷
from ase.io import read

atoms = read('nacl_solution.xyz')
charges = atoms.get_array('charges')  # 从DFT获得

# 训练
model.train(atoms, energies, forces, charges)
```

### 6.3 固态电解质

**Li-ion导体**：

```python
# 关注Li+的迁移
# 需要准确的电荷和长程能量

model = SchNetWithCharges(...)
# 训练后用于MD模拟
# 分析Li+扩散系数和电导率
```

## 7. 最新进展

### 7.1 SpookyNet (2021)

- 同时预测能量、力、偶极矩、极化率
- 内置电荷守恒
- 包含色散修正

### 7.2 MACE-OFF (2023)

- 使用原子簇展开（ACE）
- 更准确的电荷预测
- 大规模预训练模型

### 7.3 未来方向

1. **自适应电荷**：电荷随化学环境动态变化
2. **多极展开**：不止偶极，还有四极、八极
3. **极化效应**：考虑感应偶极矩
4. **量子效应**：核量子效应和零点能

## 参考文献

1. Unke et al., "SpookyNet: Learning force fields with electronic degrees of freedom and nonlocal effects", Nature Communications 12, 7273 (2021)

2. Ko et al., "A fourth-generation high-dimensional neural network potential with accurate electrostatics including non-local charge transfer", Nature Communications 12, 398 (2021)

3. Gao & Gärtner, "Incorporating long-range physics in neural network potentials", J. Chem. Phys. 153, 194101 (2020)

---

**下一章**: [02-Magnetic-Materials.md](./02-Magnetic-Materials.md)

**返回**: [Part 5主页](../README.md)
