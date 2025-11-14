# 神经网络建模DFT哈密顿量
# Neural Network Modeling of DFT Hamiltonians

## 目录 | Contents

1. [引言：为什么建模哈密顿量](#1-引言为什么建模哈密顿量)
2. [DFT哈密顿量基础](#2-dft哈密顿量基础)
3. [DeepH方法](#3-deeph方法)
4. [实现与训练](#4-实现与训练)
5. [应用场景](#5-应用场景)
6. [前沿方向](#6-前沿方向)

---

## 1. 引言：为什么建模哈密顿量

### 1.1 传统神经网络势函数的局限性

传统的NNP（如SchNet, NequIP）预测的是**总能量**：

```
E_total = f_NNP(R)
```

**局限性**：
- ❌ 无法获得电子结构信息（能带、态密度）
- ❌ 无法计算响应性质（介电常数、极化率）
- ❌ 无法描述激发态
- ❌ 无法处理电场、磁场等外场

### 1.2 直接建模哈密顿量的优势

如果我们能预测**哈密顿矩阵** H(R)：

```
H(R) = f_NN(R)
```

然后求解本征值问题：
```
H(R) ψᵢ = εᵢ ψᵢ
```

**优势**：
- ✅ 获得完整电子结构（能带、波函数、态密度）
- ✅ 计算所有从H导出的性质
- ✅ 加速大规模DFT计算（从O(N³)到O(N))
- ✅ 实现第一性原理精度的量子输运计算

### 1.3 应用场景

| 应用 | 传统DFT计算量 | DeepH加速比 |
|------|--------------|------------|
| 大尺寸超胞能带 | 数天 | ~100x |
| 表面态计算 | 数小时 | ~50x |
| 缺陷态扫描 | 数周 | ~500x |
| 量子输运 | 难以计算 | 可行 |
| 声子计算（DFPT） | 数天 | ~100x |

---

## 2. DFT哈密顿量基础

### 2.1 Kohn-Sham哈密顿量

在DFT中，Kohn-Sham方程为：

```
Ĥ_KS ψᵢ(r) = εᵢ ψᵢ(r)
```

其中：
```
Ĥ_KS = -∇²/2 + V_ext(r) + V_H(r) + V_xc(r)
```

- V_ext：外势（原子核）
- V_H：Hartree势（电子-电子库仑排斥）
- V_xc：交换关联势

### 2.2 基组展开

在实际计算中，使用原子轨道基组展开：

```
ψᵢ(r) = ∑_μ c_μi φ_μ(r)
```

φ_μ(r)：原子轨道基函数（如局域原子轨道、平面波等）

### 2.3 哈密顿矩阵元

在基组表示下：

```
H_μν = ⟨φ_μ| Ĥ_KS |φ_ν⟩
```

这是一个 N_basis × N_basis 的矩阵，其中 N_basis 是基函数总数。

**紧束缚近似（Tight-Binding）**：

只保留近邻hopping：

```
H_μν = ε_μ δ_μν + ∑_⟨μν⟩ t_μν
```

- ε_μ：原子轨道能级（on-site energy）
- t_μν：hopping积分（off-diagonal）

### 2.4 DFT软件中的哈密顿量

**VASP**：
- 平面波基组
- 输出：WAVECAR（波函数），EIGENVAL（本征值）
- 不直接输出H矩阵

**OpenMX**：
- 局域原子轨道基组（LCAO）
- 可输出哈密顿矩阵：`scfout`文件
- 格式：稀疏矩阵（只存储非零元）

**Quantum ESPRESSO**：
- 平面波基组
- 可通过wannier90转换到局域轨道

### 2.5 从DFT提取哈密顿量

```python
import numpy as np
from openmx_reader import read_scfout  # 自定义读取器

def extract_hamiltonian(scfout_file):
    """
    从OpenMX的scfout文件提取哈密顿矩阵
    """
    data = read_scfout(scfout_file)

    # 哈密顿矩阵（实空间）
    H_R = data['Hamiltonian']  # Dict: {R: H(R)}
    # R: 晶胞向量, H(R): [N_orb, N_orb] 矩阵

    # 重叠矩阵
    S_R = data['Overlap']

    # 原子位置
    positions = data['positions']

    # 基函数信息
    orbitals = data['orbitals']  # 每个原子的轨道数量

    return H_R, S_R, positions, orbitals

# 计算k空间哈密顿量
def real_to_k_space(H_R, k_point, lattice_vectors):
    """
    H(k) = ∑_R H(R) e^{ik·R}
    """
    H_k = np.zeros_like(H_R[(0,0,0)], dtype=complex)

    for R, H in H_R.items():
        R_vec = np.dot(R, lattice_vectors)
        phase = np.exp(1j * np.dot(k_point, R_vec))
        H_k += H * phase

    return H_k

# 求解能带
def compute_band_structure(H_R, S_R, k_path, lattice):
    """
    沿k路径计算能带
    """
    bands = []

    for k in k_path:
        # 构建k空间哈密顿量和重叠矩阵
        H_k = real_to_k_space(H_R, k, lattice)
        S_k = real_to_k_space(S_R, k, lattice)

        # 求解广义本征值问题：H|ψ⟩ = ε S|ψ⟩
        eigenvalues, eigenvectors = scipy.linalg.eigh(H_k, S_k)

        bands.append(eigenvalues)

    return np.array(bands)
```

---

## 3. DeepH方法

### 3.1 DeepH框架

**核心思想**：使用神经网络学习哈密顿矩阵元与局部原子环境的映射关系。

```
H_μν = f_NN(环境_μ, 环境_ν, R_μν)
```

**论文**：
- *Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation*
- Nature Computational Science, 2022
- https://github.com/mzjb/DeepH-pack

### 3.2 关键创新点

1. **局域性假设**：
   - H_μν 只依赖于 μ, ν 附近的局部环境
   - 类似于神经网络势函数的局域能量假设

2. **环境描述符**：
   - 使用图神经网络编码原子环境
   - 输出：每个轨道的嵌入向量

3. **对称性**：
   - 平移对称性：H(R)
   - 旋转对称性：使用等变神经网络
   - 时间反演对称性：H = H*（实数哈密顿量）

### 3.3 架构

```
输入：原子坐标 R, 原子类型 Z
   ↓
[环境编码器] (GNN / NequIP)
   ↓
节点嵌入：h_i for each atom i
   ↓
[轨道编码器]
   ↓
轨道嵌入：h_μ for each orbital μ
   ↓
[哈密顿预测器]
   ↓
H_μν = MLP(h_μ, h_ν, r_μν)
```

### 3.4 数学形式

**环境编码**（使用NequIP风格）：

```python
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import Gate

class OrbitalEncoder(nn.Module):
    """
    将原子环境编码为轨道特征
    """
    def __init__(self, irreps_hidden='64x0e + 32x1o + 16x2e'):
        super().__init__()

        # 原子级别的GNN（例如NequIP）
        self.atom_gnn = NequIPModel(...)

        # 轨道展开
        # 例如：s, p, d轨道
        self.orbital_types = ['s', 'p', 'd']
        self.irreps_orbitals = {
            's': '1x0e',   # l=0
            'p': '1x1o',   # l=1
            'd': '1x2e',   # l=2
        }

        # 从原子特征到轨道特征
        self.orbital_expansion = nn.ModuleDict({
            orb: o3.Linear(
                irreps_in=irreps_hidden,
                irreps_out=self.irreps_orbitals[orb]
            )
            for orb in self.orbital_types
        })

    def forward(self, data):
        """
        data: 包含 pos, species, edge_index, edge_attr
        返回：orbital_features [N_orbitals, irreps_hidden]
        """
        # 原子特征
        atom_features = self.atom_gnn(data)  # [N_atoms, irreps_hidden]

        # 展开到轨道
        orbital_features = []
        orbital_indices = []  # 记录每个轨道属于哪个原子

        for i, atom_type in enumerate(data['species']):
            atom_feat = atom_features[i]

            # 根据原子类型决定有哪些轨道
            # 例如：C有2s, 2p
            orbitals = get_orbitals_for_atom(atom_type)

            for orb in orbitals:
                orb_feat = self.orbital_expansion[orb](atom_feat)
                orbital_features.append(orb_feat)
                orbital_indices.append(i)

        orbital_features = torch.stack(orbital_features)  # [N_orb, irreps]
        orbital_indices = torch.tensor(orbital_indices)

        return orbital_features, orbital_indices
```

**哈密顿预测器**：

```python
class HamiltonianPredictor(nn.Module):
    """
    从轨道特征预测哈密顿矩阵元
    """
    def __init__(self, irreps_in='64x0e', hidden_dim=128, num_layers=3):
        super().__init__()

        # 将不可约表示投影到标量
        self.to_scalar = o3.Linear(irreps_in, '1x0e')

        # MLP预测H_μν
        layers = []
        in_dim = 2 + 1  # h_μ + h_ν + r_μν (都是标量)

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))  # 输出：H_μν

        self.mlp = nn.Sequential(*layers)

    def forward(self, orbital_feats, edge_index, edge_dist):
        """
        orbital_feats: [N_orb, irreps_in]
        edge_index: [2, E] 轨道对索引
        edge_dist: [E] 轨道间距离

        返回：H [E] 哈密顿矩阵元
        """
        # 转换为标量
        scalar_feats = self.to_scalar(orbital_feats).squeeze(-1)  # [N_orb]

        # 取出边对应的特征
        src, dst = edge_index
        feat_src = scalar_feats[src]  # [E]
        feat_dst = scalar_feats[dst]  # [E]

        # 拼接
        edge_feats = torch.stack([feat_src, feat_dst, edge_dist], dim=1)  # [E, 3]

        # 预测
        H_elements = self.mlp(edge_feats).squeeze(-1)  # [E]

        return H_elements

class DeepHModel(nn.Module):
    """完整的DeepH模型"""
    def __init__(self):
        super().__init__()
        self.orbital_encoder = OrbitalEncoder()
        self.hamiltonian_predictor = HamiltonianPredictor()

    def forward(self, data):
        """
        预测哈密顿矩阵

        data包含：
            - atomic_positions
            - atomic_species
            - orbital_pairs: 需要计算的轨道对
        """
        # 编码轨道
        orbital_feats, orbital_atom_idx = self.orbital_encoder(data)

        # 构建轨道对图
        orbital_edge_index, orbital_edge_dist = self.build_orbital_graph(
            data['atomic_positions'],
            orbital_atom_idx,
            r_max=10.0  # 截断半径
        )

        # 预测H矩阵元
        H_elements = self.hamiltonian_predictor(
            orbital_feats,
            orbital_edge_index,
            orbital_edge_dist
        )

        return H_elements, orbital_edge_index

    @staticmethod
    def build_orbital_graph(atom_pos, orbital_atom_idx, r_max):
        """构建轨道对图"""
        N_orb = len(orbital_atom_idx)

        # 计算所有轨道对的距离
        edges = []
        dists = []

        for i in range(N_orb):
            for j in range(N_orb):
                atom_i = orbital_atom_idx[i]
                atom_j = orbital_atom_idx[j]

                r_ij = torch.norm(atom_pos[atom_i] - atom_pos[atom_j])

                if r_ij < r_max:
                    edges.append([i, j])
                    dists.append(r_ij)

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_dist = torch.tensor(dists, dtype=torch.float32)

        return edge_index, edge_dist
```

### 3.5 损失函数

**直接损失**（L1）：

```python
def hamiltonian_loss(pred_H, target_H, edge_index):
    """
    预测的H矩阵元 vs DFT的H矩阵元
    """
    loss = F.l1_loss(pred_H, target_H)
    return loss
```

**能带损失**（更物理）：

```python
def band_structure_loss(pred_H, target_bands, k_points, S_matrix):
    """
    通过求解本征值，比较能带

    pred_H: 预测的哈密顿矩阵元
    target_bands: DFT计算的能带 [N_k, N_bands]
    k_points: k点 [N_k, 3]
    S_matrix: 重叠矩阵
    """
    total_loss = 0

    for i, k in enumerate(k_points):
        # 构建k空间哈密顿量
        H_k = construct_H_k(pred_H, edge_index, k, lattice)
        S_k = construct_S_k(S_matrix, edge_index, k, lattice)

        # 求解本征值
        eigenvalues = torch.linalg.eigh(H_k, S_k)[0]

        # 比较
        loss_k = F.mse_loss(eigenvalues, target_bands[i])
        total_loss += loss_k

    return total_loss / len(k_points)
```

**混合损失**：

```python
def total_loss(pred, target, weight_H=1.0, weight_band=0.1):
    loss_H = hamiltonian_loss(pred['H'], target['H'], pred['edge_index'])
    loss_band = band_structure_loss(
        pred['H'],
        target['bands'],
        target['k_points'],
        target['S_matrix']
    )

    return weight_H * loss_H + weight_band * loss_band
```

---

## 4. 实现与训练

### 4.1 数据准备

```python
import os
import numpy as np
from ase.io import read

class DeepHDataset:
    """
    DeepH数据集

    需要：
        - 结构文件（POSCAR）
        - 哈密顿矩阵（scfout或HDF5）
        - 能带（可选，用于验证）
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.structures = []
        self.hamiltonians = []

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载所有训练样本"""
        for subdir in os.listdir(self.data_dir):
            struct_file = os.path.join(self.data_dir, subdir, 'POSCAR')
            hamil_file = os.path.join(self.data_dir, subdir, 'hamiltonian.h5')

            if os.path.exists(struct_file) and os.path.exists(hamil_file):
                # 读取结构
                atoms = read(struct_file)

                # 读取哈密顿量
                H_data = self.load_hamiltonian(hamil_file)

                self.structures.append(atoms)
                self.hamiltonians.append(H_data)

    def load_hamiltonian(self, file_path):
        """
        读取哈密顿矩阵（格式依赖于DFT软件）
        """
        import h5py

        with h5py.File(file_path, 'r') as f:
            # 假设存储为稀疏格式
            H_ilist = f['hamiltonian']['ilist'][:]  # [E] 行索引
            H_jlist = f['hamiltonian']['jlist'][:]  # [E] 列索引
            H_values = f['hamiltonian']['values'][:]  # [E] 矩阵元
            H_Rlist = f['hamiltonian']['Rlist'][:]  # [E, 3] R向量

            # 重叠矩阵
            S_ilist = f['overlap']['ilist'][:]
            S_jlist = f['overlap']['jlist'][:]
            S_values = f['overlap']['values'][:]
            S_Rlist = f['overlap']['Rlist'][:]

        return {
            'H': (H_ilist, H_jlist, H_values, H_Rlist),
            'S': (S_ilist, S_jlist, S_values, S_Rlist)
        }

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        """返回一个样本"""
        atoms = self.structures[idx]
        H_data = self.hamiltonians[idx]

        # 转换为PyTorch Geometric格式
        data = self.atoms_to_graph(atoms, H_data)

        return data

    def atoms_to_graph(self, atoms, H_data):
        """将ASE Atoms转换为图"""
        from torch_geometric.data import Data

        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        species = torch.tensor(atoms.numbers, dtype=torch.long)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)

        # 哈密顿矩阵元
        H_ilist, H_jlist, H_values, H_Rlist = H_data['H']
        edge_index_H = torch.tensor([H_ilist, H_jlist], dtype=torch.long)
        edge_attr_H = torch.tensor(H_values, dtype=torch.float32)
        edge_R_H = torch.tensor(H_Rlist, dtype=torch.long)

        # 类似处理S矩阵
        S_ilist, S_jlist, S_values, S_Rlist = H_data['S']
        edge_index_S = torch.tensor([S_ilist, S_jlist], dtype=torch.long)
        edge_attr_S = torch.tensor(S_values, dtype=torch.float32)

        data = Data(
            pos=pos,
            species=species,
            cell=cell,
            edge_index_H=edge_index_H,
            edge_attr_H=edge_attr_H,
            edge_R_H=edge_R_H,
            edge_index_S=edge_index_S,
            edge_attr_S=edge_attr_S
        )

        return data
```

### 4.2 训练脚本

```python
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

def train_deeph_model(
    train_dataset,
    val_dataset,
    model,
    num_epochs=500,
    batch_size=4,
    lr=1e-3,
    device='cuda'
):
    """训练DeepH模型"""

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=20
    )

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            # 前向
            pred_H, edge_index = model(batch)

            # 损失
            target_H = batch.edge_attr_H
            loss = F.l1_loss(pred_H, target_H)

            # 反向
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                pred_H, edge_index = model(batch)
                target_H = batch.edge_attr_H

                loss = F.l1_loss(pred_H, target_H)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_deeph_model.pth')

        # 日志
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")

    return model

# 使用示例
"""
# 准备数据
train_data = DeepHDataset('data/train')
val_data = DeepHDataset('data/val')

# 创建模型
model = DeepHModel().cuda()

# 训练
model = train_deeph_model(
    train_data,
    val_data,
    model,
    num_epochs=500,
    batch_size=4,
    lr=1e-3
)
"""
```

### 4.3 推理与能带计算

```python
def predict_band_structure(model, atoms, k_path, device='cuda'):
    """
    使用DeepH预测能带结构

    atoms: ASE Atoms对象
    k_path: k点路径 [N_k, 3]
    """
    model.eval()

    # 准备输入
    data = atoms_to_graph_simple(atoms)
    data = data.to(device)

    # 预测哈密顿矩阵
    with torch.no_grad():
        pred_H, edge_index = model(data)

    # 重构完整的H矩阵（实空间）
    N_orb = get_num_orbitals(atoms)
    H_R = reconstruct_H_matrix(pred_H, edge_index, N_orb)

    # 计算能带
    bands = []
    for k in k_path:
        # k空间哈密顿量
        H_k = fourier_transform(H_R, k, atoms.cell)

        # 求解本征值
        eigenvalues = torch.linalg.eigvalsh(H_k)
        bands.append(eigenvalues.cpu().numpy())

    bands = np.array(bands)  # [N_k, N_bands]

    return bands

def plot_comparison(dft_bands, deeph_bands, k_path):
    """对比DFT和DeepH的能带"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # DFT
    for i in range(dft_bands.shape[1]):
        axes[0].plot(k_path, dft_bands[:, i], 'b-', alpha=0.6)
    axes[0].set_title('DFT', fontsize=14)
    axes[0].set_ylabel('Energy (eV)', fontsize=12)

    # DeepH
    for i in range(deeph_bands.shape[1]):
        axes[1].plot(k_path, deeph_bands[:, i], 'r-', alpha=0.6)
    axes[1].set_title('DeepH', fontsize=14)

    # MAE
    mae = np.mean(np.abs(dft_bands - deeph_bands))
    plt.suptitle(f'Band Structure Comparison (MAE = {mae*1000:.2f} meV)', fontsize=16)

    plt.tight_layout()
    plt.savefig('band_comparison.pdf', dpi=300)
    plt.show()

    return mae
```

---

## 5. 应用场景

### 5.1 大尺寸超胞能带计算

```python
def large_supercell_bands(model, unit_cell, supercell_size=(5, 5, 1)):
    """
    计算大超胞的能带（展开到原胞BZ）

    传统DFT：计算量 ~ N³（N是原子数）
    DeepH：计算量 ~ N（线性标度）
    """
    from ase.build import make_supercell

    # 构建超胞
    supercell = make_supercell(unit_cell, np.diag(supercell_size))

    print(f"Supercell size: {len(supercell)} atoms")
    print(f"DFT would take ~{(len(supercell)/len(unit_cell))**3:.0f}x longer")

    # DeepH预测
    import time
    t0 = time.time()

    k_path = get_k_path(unit_cell, npoints=100)
    bands = predict_band_structure(model, supercell, k_path)

    t1 = time.time()
    print(f"DeepH time: {t1-t0:.2f} s")

    return bands
```

### 5.2 缺陷态扫描

```python
def defect_scanning(model, pristine_structure, defect_types, positions):
    """
    高通量扫描缺陷态

    在不同位置引入不同类型的缺陷，计算缺陷能级
    """
    results = []

    for defect_type in defect_types:
        for pos in positions:
            # 创建缺陷结构
            defect_struct = create_defect(
                pristine_structure,
                defect_type,
                pos
            )

            # DeepH预测能带
            bands = predict_band_structure(model, defect_struct, k_path)

            # 提取缺陷能级（带隙内的态）
            defect_levels = extract_defect_levels(bands, E_vbm, E_cbm)

            results.append({
                'defect_type': defect_type,
                'position': pos,
                'defect_levels': defect_levels,
                'structure': defect_struct
            })

    return results
```

### 5.3 量子输运计算

DeepH可以实现第一性原理精度的量子输运：

```python
def quantum_transport(model, device_structure, bias_voltages):
    """
    计算I-V曲线

    device_structure: 左电极 | 散射区 | 右电极
    bias_voltages: 偏压列表
    """
    from negf import NEGF  # Non-Equilibrium Green's Function

    currents = []

    for V in bias_voltages:
        # DeepH预测散射区哈密顿量
        H_C = predict_hamiltonian(model, device_structure['center'], V)

        # 电极自能
        Sigma_L = compute_self_energy(model, device_structure['left'], V)
        Sigma_R = compute_self_energy(model, device_structure['right'], V)

        # NEGF计算电流
        I = NEGF_current(H_C, Sigma_L, Sigma_R, V)

        currents.append(I)

    return np.array(currents)
```

### 5.4 DFPT加速（声子计算）

```python
def accelerated_phonons(model, structure, q_points):
    """
    使用DeepH加速声子计算

    传统DFPT：需要多次自洽计算
    DeepH：直接预测微扰后的哈密顿量
    """
    phonon_frequencies = []

    for q in q_points:
        # 预测动力学矩阵
        # D(q)_αβ = ∂²E / ∂u_α(q) ∂u_β(-q)

        D_q = compute_dynamical_matrix(model, structure, q)

        # 求解声子频率
        omega_q = np.sqrt(np.linalg.eigvalsh(D_q))

        phonon_frequencies.append(omega_q)

    return np.array(phonon_frequencies)

def compute_dynamical_matrix(model, atoms, q_vector):
    """
    通过有限差分计算动力学矩阵

    对每个原子施加小位移，用DeepH计算H的变化
    """
    N = len(atoms)
    D = np.zeros((3*N, 3*N))

    delta = 0.01  # Å

    for i in range(N):
        for alpha in range(3):
            # 正向位移
            atoms_plus = atoms.copy()
            atoms_plus.positions[i, alpha] += delta

            H_plus = predict_hamiltonian(model, atoms_plus)

            # 负向位移
            atoms_minus = atoms.copy()
            atoms_minus.positions[i, alpha] -= delta

            H_minus = predict_hamiltonian(model, atoms_minus)

            # 有限差分
            dH_d_u = (H_plus - H_minus) / (2 * delta)

            # 计算力常数矩阵
            for j in range(N):
                for beta in range(3):
                    # 这里需要从dH计算力常数，简化版本
                    D[3*i+alpha, 3*j+beta] = compute_force_constant(dH_d_u, i, j, alpha, beta)

    # Fourier变换到q空间
    D_q = fourier_transform_force_constants(D, q_vector, atoms.cell)

    return D_q
```

---

## 6. 前沿方向

### 6.1 多尺度建模

结合NNP和DeepH：

```
粗粒度（NNP）-> 精细化（DeepH）
```

例如：
1. 用NNP进行长时间MD
2. 对关键构型用DeepH计算电子结构
3. 反馈到NNP训练

### 6.2 激发态

扩展到含时DFT（TDDFT）：

```
H(R, t) = f_NN(R, t)
```

应用：
- 光吸收谱
- 激子动力学
- 非绝热MD

### 6.3 强关联体系

结合DMFT（动力学平均场论）：

```
H_eff = H_DFT + Σ_DMFT
```

使用DeepH学习自能Σ：

```python
class DeepDMFT(nn.Module):
    """DeepH + DMFT"""
    def __init__(self):
        super().__init__()
        self.deeph = DeepHModel()
        self.dmft_self_energy = SelfEnergyNN()

    def forward(self, data, frequency_grid):
        # DFT部分
        H_DFT = self.deeph(data)

        # DMFT自能
        Sigma = self.dmft_self_energy(data, frequency_grid)

        # 有效哈密顿量
        H_eff = H_DFT + Sigma

        return H_eff
```

### 6.4 基础模型

类似MACE-MP-0，训练通用的哈密顿量预测模型：

**数据来源**：
- Materials Project: ~150k材料
- OQMD: ~1M结构
- JARVIS-DFT: ~40k材料

**架构**：
- Transformer + Equivariant layers
- 多任务学习：能带 + 态密度 + 介电函数

### 6.5 实验数据整合

将实验测量（如ARPES、STM）整合到训练中：

```python
def loss_with_experiment(pred_H, exp_data):
    """
    包含实验约束的损失函数
    """
    # DFT损失
    loss_dft = hamiltonian_loss(pred_H, dft_H)

    # 实验损失（例如：ARPES测量的费米面）
    pred_fermi_surface = compute_fermi_surface(pred_H)
    exp_fermi_surface = exp_data['fermi_surface']
    loss_exp = F.mse_loss(pred_fermi_surface, exp_fermi_surface)

    return loss_dft + 0.1 * loss_exp
```

---

## 总结 | Summary

DeepH方法通过神经网络直接建模DFT哈密顿量，实现了：

1. **完整电子结构**：能带、态密度、波函数
2. **线性标度**：大尺寸体系计算（1000+原子）
3. **新应用**：量子输运、DFPT加速
4. **高精度**：能带MAE < 10 meV

**关键技术**：
- 等变图神经网络编码局部环境
- 轨道分辨的特征学习
- 物理约束的损失函数

**展望**：
- 基础模型：通用哈密顿量预测
- 实验整合：结合ARPES、STM等实验数据
- 多尺度：与NNP耦合
- 激发态：扩展到TDDFT

---

## 参考文献

1. *Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation*, Nature Computational Science 2022
2. *DeepH-E3: End-to-end equivariant network for large-scale electronic structure calculation*, arXiv 2023
3. *Learning the exciton properties of azo-dyes*, Nature Communications 2021

**软件工具**：
- DeepH-pack: https://github.com/mzjb/DeepH-pack
- OpenMX: http://www.openmx-square.org/
- Wannier90: http://www.wannier.org/

---

*Happy learning Hamiltonians! 🌐*
