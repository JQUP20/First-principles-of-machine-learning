# 磁性材料的第一性原理计算和建模
# First-Principles Calculations and Modeling of Magnetic Materials

## 目录 | Contents

1. [磁性材料基础](#1-磁性材料基础)
2. [磁性材料的第一性原理计算](#2-磁性材料的第一性原理计算)
3. [传统哈密顿模型](#3-传统哈密顿模型)
4. [机器学习建模磁性势能面](#4-机器学习建模磁性势能面)
5. [挑战与前沿](#5-挑战与前沿)

---

## 1. 磁性材料基础

### 1.1 磁性的起源

磁性材料中的磁性主要来源于：

1. **电子自旋**：内禀角动量
2. **轨道角动量**：电子轨道运动
3. **自旋-轨道耦合**：两者的相互作用

对于大多数3d过渡金属，自旋磁矩占主导。

### 1.2 磁性的类型

| 磁性类型 | 特征 | 典型材料 |
|---------|------|---------|
| 铁磁性(Ferromagnetic) | 自发磁化，磁矩平行排列 | Fe, Co, Ni |
| 反铁磁性(Antiferromagnetic) | 磁矩反平行排列，零净磁矩 | MnO, FeO, NiO |
| 亚铁磁性(Ferrimagnetic) | 磁矩反平行但不等，有净磁矩 | Fe₃O₄, YIG |
| 顺磁性(Paramagnetic) | 无序排列，外场下磁化 | Al, Pt |

### 1.3 关键物理量

**磁矩（Magnetic Moment）**：
```
μ = gₛ μ_B S
```
- gₛ ≈ 2：电子g因子
- μ_B = 9.274 × 10⁻²⁴ J/T：玻尔磁子
- S：自旋量子数

**交换相互作用（Exchange Interaction）**：
```
H_ex = -∑ᵢⱼ Jᵢⱼ Sᵢ·Sⱼ
```
- Jᵢⱼ > 0：铁磁耦合
- Jᵢⱼ < 0：反铁磁耦合

---

## 2. 磁性材料的第一性原理计算

### 2.1 自旋极化DFT

**基本方程**：

对于自旋极化体系，电荷密度分为两个自旋通道：

```
ρ(r) = ρ↑(r) + ρ↓(r)
磁化密度：m(r) = ρ↑(r) - ρ↓(r)
```

**Kohn-Sham方程（自旋极化）**：

```
[-∇²/2 + V_eff,σ(r)]ψᵢ,σ(r) = εᵢ,σ ψᵢ,σ(r)
```

其中 σ = ↑,↓ 表示自旋方向。

**有效势能**：
```
V_eff,σ(r) = V_ext(r) + V_H(r) + V_xc,σ[ρ↑,ρ↓](r)
```

### 2.2 交换关联泛函

**局域自旋密度近似（LSDA）**：

```
E_xc[ρ↑,ρ↓] = ∫ ρ(r) ε_xc(ρ↑(r), ρ↓(r)) dr
```

**广义梯度近似（GGA）**：
```
E_xc[ρ↑,ρ↓] = ∫ f(ρ↑, ρ↓, ∇ρ↑, ∇ρ↓) dr
```

### 2.3 DFT+U方法

对于强关联体系（如过渡金属氧化物），标准DFT失效，需要加入在位库仑排斥U：

```
E_DFT+U = E_DFT + (U-J)/2 ∑ᵢ,σ nᵢ,σ(1 - nᵢ,σ)
```

- nᵢ,σ：d或f轨道占据数
- U：在位库仑相互作用
- J：交换相互作用

### 2.4 VASP计算示例

```bash
# INCAR for spin-polarized calculation
SYSTEM = Fe bulk
ISTART = 0
ICHARG = 2
ISPIN = 2          # 开启自旋极化
MAGMOM = 4*2.2     # 初始磁矩（4个Fe原子，每个2.2 μ_B）

# Electronic relaxation
ENCUT = 500
PREC = Accurate
EDIFF = 1E-6
ALGO = Fast
ISMEAR = 1
SIGMA = 0.2

# Ionic relaxation
NSW = 100
IBRION = 2
ISIF = 3
EDIFFG = -0.01
```

对于反铁磁体系：
```bash
# MAGMOM for antiferromagnetic ordering
# 例如：MnO (岩盐结构，AFM-II型)
MAGMOM = 5 -5 5 -5 8*0  # Mn原子交替自旋，O原子无磁矩
```

### 2.5 非共线磁性

对于复杂磁性结构（如螺旋磁性、斯格明子），需要非共线磁性计算：

```bash
LNONCOLLINEAR = .TRUE.
LSORBIT = .TRUE.        # 包含自旋-轨道耦合
MAGMOM = 0 0 2.2  0 0 -2.2  # (m_x, m_y, m_z)
```

---

## 3. 传统哈密顿模型

### 3.1 Heisenberg模型

最简单的磁性模型，只考虑最近邻交换相互作用：

```
H = -∑⟨i,j⟩ Jᵢⱼ Sᵢ·Sⱼ
```

**扩展形式**：
```
H = -∑ᵢⱼ Jᵢⱼ Sᵢ·Sⱼ - D∑ᵢ(Sᵢᶻ)² - ∑ᵢ gμ_B B·Sᵢ
```
- 第一项：交换相互作用
- 第二项：单离子各向异性（D > 0偏向xy平面）
- 第三项：Zeeman项（外磁场）

### 3.2 从第一性原理提取J

**方法1：总能量映射**

计算不同磁构型的总能量，拟合到Heisenberg模型：

```python
import numpy as np
from scipy.optimize import curve_fit

# 磁构型：FM, AFM1, AFM2, ...
configs = np.array([
    [1, 1, 1, 1],    # 铁磁
    [1, -1, 1, -1],  # 反铁磁
    [1, 1, -1, -1],  # 另一种AFM
])

# DFT能量（相对于铁磁态）
energies = np.array([0.0, 0.15, 0.22])  # eV

# Heisenberg能量
def heisen_energy(config, J1, J2):
    E = 0
    N = len(config)
    for i in range(N):
        j = (i + 1) % N  # 最近邻
        E -= J1 * config[i] * config[j]
        j = (i + 2) % N  # 次近邻
        E -= J2 * config[i] * config[j]
    return E

# 拟合
def fit_func(idx, J1, J2):
    return np.array([heisen_energy(configs[i], J1, J2) for i in idx])

popt, _ = curve_fit(fit_func, np.arange(len(configs)), energies)
J1, J2 = popt
print(f"J1 = {J1:.3f} eV, J2 = {J2:.3f} eV")
```

**方法2：磁性力常数**

使用线性响应理论（frozen magnon approach）：

```python
# 从VASP计算磁性力常数
# 需要进行小幅磁扰动计算

def extract_J_from_force_constants():
    """
    通过对磁矩施加小扰动，计算能量二阶导数
    """
    # 读取OSZICAR获取磁矩和能量
    import pymatgen as mg

    # 原始态
    E0 = -10.523  # eV
    m0 = np.array([2.2, 2.2, 2.2, 2.2])

    # 扰动态：翻转第1个原子自旋
    E1 = -10.358
    m1 = np.array([-2.2, 2.2, 2.2, 2.2])

    # 能量差
    dE = E1 - E0

    # 对于Heisenberg模型：dE = 2J * S²
    S = 2.2 / 2  # S = m/(gμ_B) ≈ m/2
    J = dE / (8 * S**2)

    return J

J_nn = extract_J_from_force_constants()
print(f"Nearest-neighbor J = {J_nn*1000:.1f} meV")
```

### 3.3 蒙特卡洛模拟

使用提取的J参数进行Monte Carlo模拟：

```python
import numpy as np
import matplotlib.pyplot as plt

class HeisenbergMC:
    def __init__(self, L, J, T):
        """
        L: 晶格尺寸 (L×L×L)
        J: 交换常数 (eV)
        T: 温度 (K)
        """
        self.L = L
        self.J = J
        self.kB = 8.617e-5  # eV/K
        self.beta = 1 / (self.kB * T)

        # 初始化自旋（随机）
        self.spins = np.random.choice([-1, 1], size=(L, L, L))

    def energy(self):
        """计算总能量"""
        E = 0
        L = self.L
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    s = self.spins[i,j,k]
                    # 最近邻（周期边界条件）
                    neighbors = [
                        self.spins[(i+1)%L, j, k],
                        self.spins[(i-1)%L, j, k],
                        self.spins[i, (j+1)%L, k],
                        self.spins[i, (j-1)%L, k],
                        self.spins[i, j, (k+1)%L],
                        self.spins[i, j, (k-1)%L],
                    ]
                    E += -self.J * s * sum(neighbors)
        return E / 2  # 每个键计算了两次

    def magnetization(self):
        """计算磁化强度"""
        return np.abs(np.mean(self.spins))

    def mc_step(self):
        """单次Monte Carlo步"""
        L = self.L
        # 随机选择一个格点
        i, j, k = np.random.randint(0, L, 3)

        # 计算翻转前的局域能量
        s = self.spins[i,j,k]
        neighbors = [
            self.spins[(i+1)%L, j, k],
            self.spins[(i-1)%L, j, k],
            self.spins[i, (j+1)%L, k],
            self.spins[i, (j-1)%L, k],
            self.spins[i, j, (k+1)%L],
            self.spins[i, j, (k-1)%L],
        ]
        E_before = -self.J * s * sum(neighbors)

        # 翻转
        self.spins[i,j,k] *= -1
        s = -s
        E_after = -self.J * s * sum(neighbors)

        # Metropolis准则
        dE = E_after - E_before
        if dE < 0 or np.random.rand() < np.exp(-self.beta * dE):
            # 接受
            pass
        else:
            # 拒绝，翻转回去
            self.spins[i,j,k] *= -1

    def run(self, n_steps, n_measure=100):
        """运行模拟"""
        magnetizations = []
        energies = []

        for step in range(n_steps):
            self.mc_step()

            if step % n_measure == 0:
                magnetizations.append(self.magnetization())
                energies.append(self.energy())

        return np.array(magnetizations), np.array(energies)

# 计算居里温度
def compute_curie_temperature(L=10, J=0.01, T_range=(100, 1000, 20)):
    """通过扫描温度计算居里温度"""
    temperatures = np.linspace(*T_range)
    magnetizations = []

    for T in temperatures:
        mc = HeisenbergMC(L, J, T)
        # 平衡
        mc.run(n_steps=10000, n_measure=1000)
        # 测量
        mags, _ = mc.run(n_steps=20000, n_measure=100)
        magnetizations.append(np.mean(mags[100:]))  # 去除前100步

    magnetizations = np.array(magnetizations)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(temperatures, magnetizations, 'o-')
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Magnetization', fontsize=12)
    plt.title(f'Heisenberg Model MC Simulation (L={L}, J={J} eV)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig('curie_temperature.pdf', dpi=300)

    # 估计Tc（磁化强度降到0.5的温度）
    idx = np.argmin(np.abs(magnetizations - 0.5))
    Tc = temperatures[idx]
    print(f"Estimated Curie temperature: {Tc:.1f} K")

    return temperatures, magnetizations, Tc

# 示例：Fe的J ≈ 10 meV
# compute_curie_temperature(L=10, J=0.01, T_range=(100, 1500, 30))
```

---

## 4. 机器学习建模磁性势能面

### 4.1 为什么需要机器学习？

传统Heisenberg模型的局限性：

1. **只考虑磁自由度**：忽略了原子位置和磁矩的耦合
2. **固定的交换常数**：实际上J依赖于原子间距和角度
3. **简化的形式**：无法描述复杂的多体磁相互作用

**机器学习的优势**：
- 同时建模原子位置和磁矩
- 自动学习J(r)的距离依赖性
- 捕捉多体效应

### 4.2 挑战

1. **数据集稀缺**：磁性材料的高质量DFT数据库很少
2. **模型架构**：需要同时处理位置和自旋自由度
3. **对称性**：自旋旋转对称性、时间反演对称性

### 4.3 输入表示

**方法1：将自旋作为额外特征**

```python
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate

class SpinNequIP(nn.Module):
    """
    NequIP扩展：加入自旋特征
    """
    def __init__(
        self,
        num_species=3,
        num_layers=4,
        irreps_hidden='32x0e + 32x1o + 16x2e',
        r_max=4.0,
        num_basis=8,
    ):
        super().__init__()

        # 原子类型embedding
        self.species_embedding = nn.Embedding(num_species, 32)

        # 自旋embedding（向量）
        # 输入：3D自旋向量 -> 不可约表示
        self.spin_embedding = o3.Linear(
            irreps_in='1x1o',  # 自旋是1o（奇宇称向量）
            irreps_out=irreps_hidden
        )

        # 径向基函数
        self.radial_basis = BesselBasis(r_max=r_max, num_basis=num_basis)

        # NequIP层
        self.layers = nn.ModuleList([
            NequIPLayer(
                irreps_in=irreps_hidden,
                irreps_out=irreps_hidden,
                num_basis=num_basis
            )
            for _ in range(num_layers)
        ])

        # 输出：能量（标量）
        self.energy_head = o3.Linear(irreps_hidden, '1x0e')

        # 输出：力和自旋力矩
        self.force_head = o3.Linear(irreps_hidden, '1x1o')
        self.torque_head = o3.Linear(irreps_hidden, '1x1o')

    def forward(self, data):
        """
        data: 包含
            - pos: [N, 3] 原子位置
            - spins: [N, 3] 自旋向量
            - species: [N] 原子类型
            - edge_index: [2, E] 边索引
            - edge_vec: [E, 3] 边向量
        """
        # 原子特征
        node_feats_species = self.species_embedding(data['species'])  # [N, 32]

        # 自旋特征（需要转换为不可约表示）
        spins = data['spins']  # [N, 3]
        node_feats_spin = self.spin_embedding(spins)  # [N, irreps_hidden]

        # 合并特征
        node_feats = node_feats_species + node_feats_spin  # 简化版本

        # 边特征
        edge_vec = data['edge_vec']
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_hidden.lmax,
            x=edge_vec,
            normalize=True
        )
        edge_radial = self.radial_basis(edge_length)

        # 传播
        for layer in self.layers:
            node_feats = layer(
                node_feats,
                data['edge_index'],
                edge_sh,
                edge_radial
            )

        # 输出
        energy = self.energy_head(node_feats).sum()  # 总能量
        forces = self.force_head(node_feats)  # [N, 3]
        torques = self.torque_head(node_feats)  # [N, 3] 自旋力矩

        return {
            'energy': energy,
            'forces': forces,
            'torques': torques
        }
```

**方法2：SpinGCN（离散自旋）**

对于Ising或XY模型，自旋是离散的：

```python
class SpinGraphConv(nn.Module):
    """考虑自旋的图卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.lin_spin = nn.Linear(1, out_channels)  # 自旋相关项

    def forward(self, x, edge_index, spins):
        """
        x: [N, in_channels] 节点特征
        edge_index: [2, E]
        spins: [N] 自旋值（+1或-1）
        """
        row, col = edge_index

        # 自相关
        out = self.lin_self(x)

        # 邻居聚合
        neigh_feats = x[col]  # [E, in_channels]

        # 自旋项：Sᵢ * Sⱼ
        spin_products = (spins[row] * spins[col]).unsqueeze(1)  # [E, 1]
        spin_term = self.lin_spin(spin_products)  # [E, out_channels]

        # 加权聚合
        neigh_msg = self.lin_neigh(neigh_feats) * spin_term

        # Scatter add
        out.index_add_(0, row, neigh_msg)

        return out

class SpinGNN(nn.Module):
    """完整的自旋GNN"""
    def __init__(self, num_species=3, hidden_dim=64, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(num_species, hidden_dim)

        self.convs = nn.ModuleList([
            SpinGraphConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = self.embedding(data['species'])

        for conv in self.convs:
            x = conv(x, data['edge_index'], data['spins'])
            x = F.silu(x)

        # 总能量
        energies = self.energy_head(x)
        total_energy = energies.sum()

        return {'energy': total_energy}
```

### 4.4 训练

```python
import torch
from torch_geometric.data import Data, DataLoader

# 准备数据
def prepare_spin_data(traj_file):
    """
    从VASP轨迹读取数据
    需要包含MAGMOM信息
    """
    from ase.io import read

    atoms_list = read(traj_file, ':')
    data_list = []

    for atoms in atoms_list:
        # 获取磁矩（从OUTCAR或MAGMOM）
        magmoms = atoms.get_magnetic_moments()  # [N, 3] 或 [N]

        # 如果是标量磁矩，扩展为向量（沿z方向）
        if magmoms.ndim == 1:
            spins = torch.zeros(len(atoms), 3)
            spins[:, 2] = torch.tensor(magmoms)
        else:
            spins = torch.tensor(magmoms, dtype=torch.float32)

        # 构建图
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        species = torch.tensor(atoms.numbers, dtype=torch.long)

        # 边（截断半径内）
        edge_index, edge_vec = get_edges(pos, r_max=4.0, pbc=atoms.pbc, cell=atoms.cell)

        # 能量
        energy = atoms.get_potential_energy()

        # 力
        forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)

        data = Data(
            pos=pos,
            spins=spins,
            species=species,
            edge_index=edge_index,
            edge_vec=edge_vec,
            energy=torch.tensor([energy], dtype=torch.float32),
            forces=forces
        )

        data_list.append(data)

    return data_list

# 训练循环
def train_spin_model():
    # 数据
    train_data = prepare_spin_data('train.traj')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    # 模型
    model = SpinNequIP(num_species=3, num_layers=4).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 损失权重
    energy_weight = 1.0
    force_weight = 100.0
    torque_weight = 10.0  # 自旋力矩权重

    for epoch in range(500):
        total_loss = 0

        for batch in train_loader:
            batch = batch.cuda()

            # 前向
            pred = model(batch)

            # 损失
            loss_energy = F.mse_loss(pred['energy'], batch.energy)
            loss_force = F.mse_loss(pred['forces'], batch.forces)

            # 自旋力矩损失（如果有标签）
            if hasattr(batch, 'torques'):
                loss_torque = F.mse_loss(pred['torques'], batch.torques)
            else:
                loss_torque = 0

            loss = (energy_weight * loss_energy +
                   force_weight * loss_force +
                   torque_weight * loss_torque)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

    return model
```

### 4.5 应用：磁性分子动力学

使用训练好的模型进行自旋动力学模拟：

```python
class SpinDynamics:
    """
    耦合原子-自旋动力学
    """
    def __init__(self, model, atoms, spins, dt=0.5):
        """
        model: 训练好的SpinNequIP模型
        atoms: ASE Atoms对象
        spins: 初始自旋构型 [N, 3]
        dt: 时间步长（fs）
        """
        self.model = model
        self.atoms = atoms
        self.spins = torch.tensor(spins, dtype=torch.float32)
        self.dt = dt

        # 质量和磁矩大小
        self.masses = torch.tensor(atoms.get_masses())
        self.spin_magnitudes = torch.norm(self.spins, dim=1)

    def step(self):
        """单步动力学"""
        # 构建数据
        data = self.build_data()

        # 模型预测
        with torch.enable_grad():
            data['pos'].requires_grad = True
            data['spins'].requires_grad = True

            pred = self.model(data)

            # 力：dE/dR
            forces = -torch.autograd.grad(
                pred['energy'],
                data['pos'],
                create_graph=False
            )[0]

            # 自旋力矩：dE/dS
            torques = -torch.autograd.grad(
                pred['energy'],
                data['spins'],
                create_graph=False
            )[0]

        # 更新位置（Velocity Verlet）
        pos = torch.tensor(self.atoms.positions)
        vel = torch.tensor(self.atoms.get_velocities())

        vel += 0.5 * self.dt * forces / self.masses.unsqueeze(1)
        pos += self.dt * vel
        vel += 0.5 * self.dt * forces / self.masses.unsqueeze(1)

        self.atoms.positions = pos.numpy()
        self.atoms.set_velocities(vel.numpy())

        # 更新自旋（Landau-Lifshitz方程）
        # dS/dt = -γ S × H_eff, H_eff = -dE/dS
        gamma = 1.76e11  # 旋磁比（rad/s/T）

        # S × H_eff
        cross_product = torch.cross(self.spins, torques)
        self.spins += self.dt * gamma * cross_product

        # 重归一化（保持磁矩大小不变）
        self.spins = F.normalize(self.spins, dim=1) * self.spin_magnitudes.unsqueeze(1)

    def run(self, steps=1000):
        """运行模拟"""
        trajectory = []
        spin_trajectory = []

        for i in range(steps):
            self.step()
            trajectory.append(self.atoms.copy())
            spin_trajectory.append(self.spins.clone())

            if i % 100 == 0:
                E = self.model(self.build_data())['energy'].item()
                print(f"Step {i}: E = {E:.4f} eV")

        return trajectory, spin_trajectory

    def build_data(self):
        """构建模型输入"""
        pos = torch.tensor(self.atoms.positions, dtype=torch.float32)
        species = torch.tensor(self.atoms.numbers, dtype=torch.long)
        edge_index, edge_vec = get_edges(pos, r_max=4.0)

        return {
            'pos': pos,
            'spins': self.spins,
            'species': species,
            'edge_index': edge_index,
            'edge_vec': edge_vec
        }

# 使用示例
"""
from ase.build import bulk
atoms = bulk('Fe', 'bcc', a=2.87) * (3, 3, 3)
spins = np.random.randn(len(atoms), 3)
spins = spins / np.linalg.norm(spins, axis=1, keepdims=True) * 2.2  # 归一化到2.2 μ_B

md = SpinDynamics(model, atoms, spins, dt=0.5)
traj, spin_traj = md.run(steps=5000)
"""
```

---

## 5. 挑战与前沿

### 5.1 当前挑战

| 挑战 | 描述 | 可能的解决方案 |
|------|------|---------------|
| **数据稀缺** | 磁性材料DFT数据库少 | 主动学习、迁移学习 |
| **长时间尺度** | 磁性相变慢（ns-μs） | 粗粒化、增强采样 |
| **温度效应** | 磁矩大小随温度变化 | 多温度训练、热涨落模型 |
| **强关联** | DFT+U依赖U值选择 | 从实验数据学习、DMFT |
| **量子效应** | 自旋波、磁激发 | 量子自旋动力学 |

### 5.2 前沿方向

**1. 磁性材料的主动学习**

```python
def active_learning_for_magnets():
    """
    针对磁性材料的主动学习策略
    """
    # 初始数据：不同磁构型
    initial_configs = generate_magnetic_configs(
        structures=['FM', 'AFM-I', 'AFM-II'],
        temperatures=[0, 300, 600]
    )

    # 运行DFT
    initial_data = run_dft_batch(initial_configs)

    # 训练初始模型
    model = train_spin_model(initial_data)

    # 主动学习循环
    for iteration in range(10):
        # 1. 生成候选（MD模拟）
        candidates = run_spin_md(model, steps=10000, T=500)

        # 2. 不确定性估计（ensemble）
        uncertainties = estimate_uncertainty(model, candidates)

        # 3. 选择最不确定的
        selected = candidates[np.argsort(uncertainties)[-10:]]

        # 4. DFT标注
        new_data = run_dft_batch(selected)

        # 5. 重新训练
        all_data = initial_data + new_data
        model = train_spin_model(all_data)

        print(f"Iteration {iteration}: Added {len(new_data)} structures")

    return model
```

**2. 基础模型方法**

类似于MACE-MP-0，训练通用的磁性材料势函数：

- 数据：从Materials Project, OQMD等收集所有磁性材料
- 架构：SpinNequIP或等变Transformer
- 任务：能量、力、磁矩、磁各向异性能

**3. 与实验结合**

```python
def fit_to_experiment():
    """
    结合实验数据（如中子散射）优化模型
    """
    # DFT数据
    dft_data = load_dft_data()

    # 实验：磁化曲线
    exp_T, exp_M = load_magnetization_curve()

    # 损失函数
    def loss_fn(model):
        # DFT损失
        loss_dft = compute_dft_loss(model, dft_data)

        # 实验损失：模拟磁化曲线
        sim_M = []
        for T in exp_T:
            mc = HeisenbergMC_with_ML(model, T=T)
            M = mc.run_and_measure()
            sim_M.append(M)

        loss_exp = np.mean((np.array(sim_M) - exp_M)**2)

        return loss_dft + 0.1 * loss_exp

    # 优化
    model = optimize_model(loss_fn)
    return model
```

### 5.3 软件工具

| 工具 | 功能 | 网站 |
|------|------|------|
| **VAMPIRE** | 原子级自旋动力学 | https://vampire.york.ac.uk/ |
| **Spirit** | 自旋系统能量最小化和动力学 | https://github.com/spirit-code/spirit |
| **UppASD** | Uppsala原子自旋动力学 | https://github.com/UppASD/UppASD |
| **SpinW** | 自旋波计算（MATLAB） | https://spinw.org/ |
| **TB2J** | 从DFT提取磁交换常数 | https://tb2j.readthedocs.io/ |

### 5.4 学习资源

**教科书**：
1. *Introduction to the Theory of Ferromagnetism* - Aharoni
2. *Magnetism: From Fundamentals to Nanoscale Dynamics* - Stöhr & Siegmann
3. *Computational Materials Science* - Kauffman & Peppas

**综述论文**：
- "Machine learning for quantum materials" - Nature Physics 2021
- "Density functional theory for magnetism" - Rev. Mod. Phys. 2015

**在线课程**：
- MIT OpenCourseWare: Magnetic Materials

---

## 总结 | Summary

本章介绍了磁性材料的第一性原理计算和机器学习建模：

1. **基础知识**：磁性的起源、类型、关键物理量
2. **DFT计算**：自旋极化DFT、DFT+U、非共线磁性
3. **传统模型**：Heisenberg模型、交换常数提取、Monte Carlo模拟
4. **ML建模**：扩展神经网络势函数到磁性体系、自旋动力学
5. **前沿**：主动学习、基础模型、实验数据整合

**关键要点**：
- 磁性材料的建模需要同时处理原子位置和自旋自由度
- 机器学习可以超越传统Heisenberg模型的局限性
- 数据稀缺是当前最大挑战，需要主动学习和迁移学习

**下一步**：
- 实践：使用VASP计算Fe的磁构型
- 实践：实现SpinGNN并在简单体系上测试
- 探索：查阅磁性材料数据库（如MAGNDATA）

---

*Happy learning magnetism! 🧲*
