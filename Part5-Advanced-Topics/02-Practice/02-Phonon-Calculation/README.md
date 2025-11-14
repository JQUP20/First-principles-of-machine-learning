# 声子谱计算实验
# Phonon Spectrum Calculation with Neural Network Potentials

## 目录 | Contents

1. [实验简介](#1-实验简介)
2. [理论基础](#2-理论基础)
3. [使用Phonopy](#3-使用phonopy)
4. [NNP加速声子计算](#4-nnp加速声子计算)
5. [完整工作流程](#5-完整工作流程)
6. [实际案例](#6-实际案例)

---

## 1. 实验简介

### 1.1 学习目标

- 理解声子的物理意义和计算方法
- 掌握Phonopy软件的使用
- 学会用神经网络势函数加速声子计算
- 计算声子色散、态密度、热力学性质

### 1.2 所需软件

```bash
# 基础环境
conda create -n phonon python=3.9 -y
conda activate phonon

# DFT软件（选一个）
# VASP (需要license)
# Quantum ESPRESSO (开源)
conda install -c conda-forge quantum-espresso

# Phonopy
pip install phonopy h5py matplotlib

# ASE
pip install ase

# 机器学习势函数（选择）
pip install nequip  # NequIP
# 或
pip install torch-dftd  # 其他NNP
```

### 1.3 预期时间

| 任务 | DFT时间 | NNP时间 | 加速比 |
|------|---------|---------|-------|
| Si (2原子) | ~10分钟 | ~1分钟 | 10x |
| Si (64原子) | ~10小时 | ~10分钟 | 60x |
| 蛋白质 (1000原子) | 不可行 | ~1小时 | ∞ |

---

## 2. 理论基础

### 2.1 声子是什么？

声子是晶格振动的量子化准粒子。

**经典图像**：
- 原子在平衡位置附近做小幅振动
- 原子间通过化学键耦合，形成集体振动模式

**量子图像**：
- 每个振动模式的能量量子化：E = ℏω(n + 1/2)
- 声子：一个振动量子（n=1）

### 2.2 简正模式

对于N个原子的体系，有3N个振动自由度（3N个简正模式）。

**简正模式坐标**：
```
u_i(t) = A_i e^{i(q·R_i - ωt)}
```

- q：波矢（动量）
- ω：频率（能量 = ℏω）
- R_i：原子i的平衡位置

### 2.3 动力学矩阵

声子频率由动力学矩阵的本征值决定：

```
D_αβ(q) = (1/√(M_i M_j)) ∑_R Φ_αβ(0,R) e^{iq·R}
```

- Φ_αβ(0,R)：力常数矩阵
  ```
  Φ_αβ(i,j) = ∂²E / ∂u_iα ∂u_jβ
  ```

**本征值问题**：
```
D(q) e_λ(q) = ω²_λ(q) e_λ(q)
```

- ω_λ(q)：第λ支声子在波矢q的频率
- e_λ(q)：对应的本征矢量（极化方向）

### 2.4 计算方法

**方法1：有限差分（Frozen Phonon）**

1. 对每个原子施加小位移 Δu
2. 计算力 F = -∂E/∂u
3. 用有限差分计算二阶导数（力常数）

```
Φ_αβ(i,j) ≈ [F_β(u_iα=+Δ) - F_β(u_iα=-Δ)] / (2Δ)
```

**方法2：密度泛函微扰理论（DFPT）**

直接计算线性响应函数，无需做超胞。

优点：精确，效率高（小体系）
缺点：实现复杂，大体系仍慢

### 2.5 超胞方法

为了计算长波长声子（小q），需要大超胞。

关系：
```
q_min = 2π / L
```

L：超胞尺寸

例如：
- 2×2×2超胞 → 只能计算q = (0.5, 0, 0)等几个点
- 4×4×4超胞 → 可以计算更密集的q网格

---

## 3. 使用Phonopy

### 3.1 快速开始：Si的声子谱

#### 步骤1：准备原胞结构

```python
# generate_structure.py
from ase.build import bulk
from ase.io import write

# Si 金刚石结构
atoms = bulk('Si', 'diamond', a=5.43)
write('POSCAR_unitcell', atoms, format='vasp')

print(f"Unit cell: {len(atoms)} atoms")
print(atoms)
```

#### 步骤2：生成超胞（带位移）

```bash
# 使用Phonopy生成2×2×2超胞，并创建位移
phonopy -d --dim="2 2 2" --pa="auto" -c POSCAR_unitcell

# 这会生成：
#   SPOSCAR: 超胞
#   POSCAR-001, POSCAR-002, ...: 带位移的超胞
```

查看生成了多少个位移：
```bash
ls POSCAR-* | wc -l
# 对于Si，应该是几个（取决于对称性）
```

#### 步骤3：DFT计算力

对每个位移构型，计算力：

```bash
# INCAR for force calculation
cat > INCAR << 'EOF'
SYSTEM = Si phonon
ISTART = 0
ICHARG = 2
ENCUT = 500
PREC = Accurate
EDIFF = 1E-8
ISMEAR = 0
SIGMA = 0.01
LREAL = .FALSE.
LWAVE = .FALSE.
LCHARG = .FALSE.
NSW = 0          # 不优化，只计算力
IBRION = -1
EOF

# 批量运行
for i in POSCAR-*; do
    dir=${i#POSCAR-}
    mkdir -p disp-$dir
    cp $i disp-$dir/POSCAR
    cp INCAR POTCAR KPOINTS disp-$dir/
    cd disp-$dir
    mpirun -np 4 vasp_std > log
    cd ..
done
```

#### 步骤4：收集力，计算声子

```bash
# 创建force_sets文件
phonopy -f disp-*/vasprun.xml

# 或者从OUTCAR读取
phonopy -f disp-*/OUTCAR

# 计算声子色散和DOS
phonopy --dim="2 2 2" -c POSCAR_unitcell --pa="auto" -p band.conf

# 绘图
phonopy-load --graph save
```

#### band.conf 配置文件

```yaml
# band.conf
DIM = 2 2 2
ATOM_NAME = Si
PRIMITIVE_AXES = AUTO
BAND = 0.5 0.0 0.0  0.0 0.0 0.0  0.5 0.5 0.5
BAND_POINTS = 101
MESH = 20 20 20
DOS = .TRUE.
DOS_RANGE = 0 20 0.1
WRITE_MESH = .FALSE.
```

### 3.2 Python接口

```python
# phonon_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase.io import read

# 读取结构
atoms_ase = read('POSCAR_unitcell')

# 转换为Phonopy格式
unitcell = PhonopyAtoms(
    symbols=atoms_ase.get_chemical_symbols(),
    cell=atoms_ase.cell,
    scaled_positions=atoms_ase.get_scaled_positions()
)

# 创建Phonopy对象
phonon = Phonopy(
    unitcell,
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix='auto'
)

# 生成位移
phonon.generate_displacements(distance=0.01)  # 0.01 Å

# 获取超胞（带位移）
supercells = phonon.supercells_with_displacements

print(f"Number of displacements: {len(supercells)}")

# 保存位移超胞
from phonopy.interface.vasp import write_vasp
for i, scell in enumerate(supercells):
    write_vasp(f'POSCAR-{i+1:03d}', scell)

# === 假设已经计算了力 ===

# 读取力（从VASP输出）
from phonopy.interface.vasp import parse_set_of_forces

set_of_forces = parse_set_of_forces(
    len(supercells),
    filename='FORCE_SETS'  # Phonopy生成的文件
)

# 或者手动设置力
# forces = [...]  # List of [N_atoms, 3] arrays
# phonon.forces = forces

phonon.produce_force_constants()

# 计算能带结构
bands = [
    [0.5, 0.0, 0.0],  # X
    [0.0, 0.0, 0.0],  # Γ
    [0.5, 0.5, 0.5],  # L
]

phonon.run_band_structure(
    bands,
    with_eigenvectors=True,
    is_band_connection=True
)

# 绘图
phonon.plot_band_structure().show()

# 计算态密度
phonon.run_mesh(mesh=[20, 20, 20])
phonon.run_total_dos()

# 绘制DOS
plt.figure(figsize=(8, 6))
dos_data = phonon.get_total_dos_dict()
plt.plot(dos_data['frequency_points'], dos_data['total_dos'])
plt.xlabel('Frequency (THz)')
plt.ylabel('DOS')
plt.title('Phonon Density of States')
plt.grid(alpha=0.3)
plt.savefig('phonon_dos.pdf')
plt.show()

# 计算热力学性质
phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)

# 获取自由能、熵、热容
tp_dict = phonon.get_thermal_properties_dict()
temperatures = tp_dict['temperatures']
free_energy = tp_dict['free_energy']  # kJ/mol
entropy = tp_dict['entropy']  # J/K/mol
heat_capacity = tp_dict['heat_capacity']  # J/K/mol

# 绘制
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(temperatures, free_energy)
axes[0].set_xlabel('Temperature (K)')
axes[0].set_ylabel('Free Energy (kJ/mol)')
axes[0].grid(alpha=0.3)

axes[1].plot(temperatures, entropy)
axes[1].set_xlabel('Temperature (K)')
axes[1].set_ylabel('Entropy (J/K/mol)')
axes[1].grid(alpha=0.3)

axes[2].plot(temperatures, heat_capacity)
axes[2].set_xlabel('Temperature (K)')
axes[2].set_ylabel('Heat Capacity (J/K/mol)')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('thermal_properties.pdf')
plt.show()
```

---

## 4. NNP加速声子计算

### 4.1 为什么用NNP？

**传统DFT声子计算的瓶颈**：

1. 需要计算多个位移构型的力（通常10-100个）
2. 每个构型都需要自洽迭代
3. 大超胞计算量 ~ N³

**NNP的优势**：

1. 单次力计算 < 1秒（vs DFT的几分钟到几小时）
2. 总加速比：50-1000x
3. 可以用于大体系（1000+原子）

### 4.2 使用NequIP计算声子

#### 准备训练好的NequIP模型

假设你已经有了一个训练好的模型 `deployed_model.pth`。

```python
# phonon_with_nequip.py
import numpy as np
from ase.io import read
from ase.calculators.nequip import NequIP
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

# 1. 读取结构
atoms = read('POSCAR_unitcell')

# 2. 设置NequIP计算器
calc = NequIP(
    model_path='deployed_model.pth',
    device='cuda'
)

# 3. 创建Phonopy对象
unitcell_phonopy = PhonopyAtoms(
    symbols=atoms.get_chemical_symbols(),
    cell=atoms.cell.array,
    scaled_positions=atoms.get_scaled_positions()
)

phonon = Phonopy(
    unitcell_phonopy,
    supercell_matrix=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],  # 更大的超胞！
    primitive_matrix='auto'
)

# 4. 生成位移
phonon.generate_displacements(distance=0.01)
supercells = phonon.supercells_with_displacements

print(f"Number of displacements: {len(supercells)}")
print(f"Supercell size: {len(phonon.supercell)} atoms")

# 5. 用NequIP计算所有位移的力
forces = []

for i, scell in enumerate(supercells):
    # 转换为ASE Atoms
    atoms_disp = Atoms(
        symbols=scell.symbols,
        cell=scell.cell,
        scaled_positions=scell.scaled_positions,
        pbc=True
    )
    atoms_disp.set_calculator(calc)

    # 计算力
    f = atoms_disp.get_forces()
    forces.append(f)

    print(f"Displacement {i+1}/{len(supercells)}: computed forces")

# 6. 设置力到Phonopy
phonon.forces = forces

# 7. 计算力常数
phonon.produce_force_constants()

# 8. 计算声子色散
bands = [
    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],  # Γ-X
    [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],  # X-M
    [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]],  # M-Γ
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],  # Γ-R
]

phonon.run_band_structure(
    bands,
    with_eigenvectors=True,
    is_band_connection=False
)

# 9. 绘图
import matplotlib.pyplot as plt
phonon.plot_band_structure().savefig('phonon_band_nequip.pdf')
plt.show()

# 10. 计算DOS
phonon.run_mesh(mesh=[30, 30, 30])
phonon.run_total_dos()
phonon.plot_total_dos().savefig('phonon_dos_nequip.pdf')
plt.show()

# 11. 保存结果
phonon.save('phonopy_nequip.yaml')
print("Phonon calculation completed!")
```

### 4.3 对比DFT和NNP结果

```python
# compare_phonons.py
import numpy as np
import matplotlib.pyplot as plt
from phonopy import load

# 加载两个计算结果
phonon_dft = load('phonopy_dft.yaml')
phonon_nnp = load('phonopy_nequip.yaml')

# 提取能带数据
band_dft = phonon_dft.get_band_structure_dict()
band_nnp = phonon_nnp.get_band_structure_dict()

# 绘制对比图
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# DFT
for path in band_dft['paths']:
    qpoints = path['qpoints']
    frequencies = path['frequencies']
    for band in frequencies.T:
        axes[0].plot(qpoints, band, 'b-', alpha=0.6)
axes[0].set_title('DFT', fontsize=14)
axes[0].set_ylabel('Frequency (THz)', fontsize=12)
axes[0].set_xlabel('q-point', fontsize=12)
axes[0].grid(alpha=0.3)

# NNP
for path in band_nnp['paths']:
    qpoints = path['qpoints']
    frequencies = path['frequencies']
    for band in frequencies.T:
        axes[1].plot(qpoints, band, 'r-', alpha=0.6)
axes[1].set_title('NequIP', fontsize=14)
axes[1].set_xlabel('q-point', fontsize=12)
axes[1].grid(alpha=0.3)

# 计算误差
freq_dft = np.concatenate([p['frequencies'] for p in band_dft['paths']])
freq_nnp = np.concatenate([p['frequencies'] for p in band_nnp['paths']])
mae = np.mean(np.abs(freq_dft - freq_nnp))
rmse = np.sqrt(np.mean((freq_dft - freq_nnp)**2))

plt.suptitle(f'Phonon Band Comparison (MAE = {mae:.3f} THz, RMSE = {rmse:.3f} THz)', fontsize=16)
plt.tight_layout()
plt.savefig('phonon_comparison.pdf', dpi=300)
plt.show()

print(f"Mean Absolute Error: {mae:.4f} THz ({mae*33.36:.2f} cm⁻¹)")
print(f"Root Mean Square Error: {rmse:.4f} THz ({rmse*33.36:.2f} cm⁻¹)")
```

---

## 5. 完整工作流程

### 5.1 端到端示例：金刚石

```bash
# workflow.sh - 完整声子计算流程

set -e

echo "=== Step 1: Generate structure ==="
python << 'EOF'
from ase.build import bulk
from ase.io import write

atoms = bulk('C', 'diamond', a=3.567)
write('POSCAR', atoms, format='vasp')
print(f"Created diamond structure with {len(atoms)} atoms")
EOF

echo "=== Step 2: DFT calculation of unit cell (for NNP training) ==="
# 这里假设你已经有训练好的NequIP模型
# 如果没有，需要先生成训练数据并训练模型

echo "=== Step 3: Generate supercell with displacements ==="
python << 'EOF'
from ase.io import read
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase.calculators.nequip import NequIP
from ase import Atoms
import numpy as np

# 读取原胞
atoms = read('POSCAR')

# Phonopy设置
unitcell = PhonopyAtoms(
    symbols=atoms.get_chemical_symbols(),
    cell=atoms.cell.array,
    scaled_positions=atoms.get_scaled_positions()
)

phonon = Phonopy(
    unitcell,
    supercell_matrix=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
    primitive_matrix='auto'
)

phonon.generate_displacements(distance=0.01)
supercells = phonon.supercells_with_displacements

print(f"Generated {len(supercells)} displaced supercells")
print(f"Each supercell has {len(phonon.supercell)} atoms")

# 计算力（使用NequIP）
calc = NequIP(model_path='deployed_nequip.pth', device='cuda')

forces = []
for i, scell in enumerate(supercells):
    atoms_disp = Atoms(
        symbols=scell.symbols,
        cell=scell.cell,
        scaled_positions=scell.scaled_positions,
        pbc=True
    )
    atoms_disp.set_calculator(calc)
    f = atoms_disp.get_forces()
    forces.append(f)
    print(f"  Computed forces for displacement {i+1}/{len(supercells)}")

phonon.forces = forces
phonon.produce_force_constants()

# 计算能带
bands_path = [
    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]],  # Γ-X
    [[0.5, 0.0, 0.5], [0.5, 0.25, 0.75]],  # X-W
    [[0.5, 0.25, 0.75], [0.375, 0.375, 0.75]],  # W-K
    [[0.375, 0.375, 0.75], [0.0, 0.0, 0.0]],  # K-Γ
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],  # Γ-L
]

phonon.run_band_structure(bands_path, with_eigenvectors=False, is_band_connection=False)
phonon.write_yaml_band_structure()

# DOS
phonon.run_mesh(mesh=[30, 30, 30])
phonon.run_total_dos()
phonon.write_total_dos()

# 热力学性质
phonon.run_thermal_properties(t_min=0, t_max=2000, t_step=10)
phonon.write_yaml_thermal_properties()

print("Phonon calculation complete!")
EOF

echo "=== Step 4: Plot results ==="
python << 'EOF'
import matplotlib.pyplot as plt
import yaml

# 读取能带
with open('band.yaml', 'r') as f:
    data = yaml.safe_load(f)

# 绘制能带
plt.figure(figsize=(8, 6))
for path in data['phonon']:
    for band in path['band']:
        freqs = [b['frequency'] for b in band]
        qpts = range(len(freqs))
        plt.plot(qpts, freqs, 'b-', alpha=0.7)

plt.xlabel('q-point')
plt.ylabel('Frequency (THz)')
plt.title('Diamond Phonon Dispersion (NequIP)')
plt.grid(alpha=0.3)
plt.savefig('diamond_phonon_band.pdf')
print("Saved: diamond_phonon_band.pdf")

# 读取DOS
dos_data = np.loadtxt('total_dos.dat')
freq = dos_data[:, 0]
dos = dos_data[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(freq, dos, 'r-', linewidth=2)
plt.fill_between(freq, dos, alpha=0.3)
plt.xlabel('Frequency (THz)')
plt.ylabel('Density of States')
plt.title('Diamond Phonon DOS (NequIP)')
plt.grid(alpha=0.3)
plt.savefig('diamond_phonon_dos.pdf')
print("Saved: diamond_phonon_dos.pdf")

# 读取热力学性质
with open('thermal_properties.yaml', 'r') as f:
    tp = yaml.safe_load(f)

T = [t['temperature'] for t in tp['thermal_properties']]
F = [t['free_energy'] for t in tp['thermal_properties']]
S = [t['entropy'] for t in tp['thermal_properties']]
Cv = [t['heat_capacity'] for t in tp['thermal_properties']]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(T, F, 'b-', linewidth=2)
axes[0].set_xlabel('Temperature (K)')
axes[0].set_ylabel('Free Energy (kJ/mol)')
axes[0].grid(alpha=0.3)

axes[1].plot(T, S, 'g-', linewidth=2)
axes[1].set_xlabel('Temperature (K)')
axes[1].set_ylabel('Entropy (J/K/mol)')
axes[1].grid(alpha=0.3)

axes[2].plot(T, Cv, 'r-', linewidth=2)
axes[2].set_xlabel('Temperature (K)')
axes[2].set_ylabel('Heat Capacity (J/K/mol)')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diamond_thermal.pdf')
print("Saved: diamond_thermal.pdf")
EOF

echo "=== All done! ==="
```

### 5.2 自动化脚本

```python
# auto_phonon.py - 自动化声子计算

import argparse
from ase.io import read
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np

def compute_phonon_with_nnp(
    structure_file,
    model_path,
    supercell_matrix=[[4,0,0],[0,4,0],[0,0,4]],
    mesh=[20,20,20],
    displacement=0.01,
    device='cuda'
):
    """
    使用NNP自动计算声子

    参数:
        structure_file: 结构文件（POSCAR, CIF, etc.）
        model_path: NNP模型路径
        supercell_matrix: 超胞矩阵
        mesh: DOS计算的k网格
        displacement: 位移大小（Å）
        device: cuda或cpu
    """
    print(f"Reading structure from {structure_file}...")
    atoms = read(structure_file)

    print(f"Loading NNP model from {model_path}...")
    from ase.calculators.nequip import NequIP
    calc = NequIP(model_path=model_path, device=device)

    print("Setting up Phonopy...")
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions()
    )

    phonon = Phonopy(
        unitcell,
        supercell_matrix=supercell_matrix,
        primitive_matrix='auto'
    )

    print(f"Generating displacements (distance={displacement} Å)...")
    phonon.generate_displacements(distance=displacement)
    supercells = phonon.supercells_with_displacements

    print(f"Computing forces for {len(supercells)} configurations...")
    print(f"Supercell size: {len(phonon.supercell)} atoms")

    forces = []
    for i, scell in enumerate(supercells):
        atoms_disp = Atoms(
            symbols=scell.symbols,
            cell=scell.cell,
            scaled_positions=scell.scaled_positions,
            pbc=True
        )
        atoms_disp.set_calculator(calc)
        f = atoms_disp.get_forces()
        forces.append(f)

        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(supercells)}")

    print("Computing force constants...")
    phonon.forces = forces
    phonon.produce_force_constants()

    print(f"Computing phonon DOS (mesh={mesh})...")
    phonon.run_mesh(mesh=mesh)
    phonon.run_total_dos()

    print("Computing thermal properties...")
    phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)

    print("Saving results...")
    phonon.save('phonopy_params.yaml')
    phonon.write_yaml_band_structure()
    phonon.write_total_dos()
    phonon.write_yaml_thermal_properties()

    print("Done! Results saved to:")
    print("  - phonopy_params.yaml")
    print("  - band.yaml")
    print("  - total_dos.dat")
    print("  - thermal_properties.yaml")

    return phonon

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute phonons with NNP')
    parser.add_argument('structure', help='Structure file')
    parser.add_argument('model', help='NNP model path')
    parser.add_argument('--supercell', type=int, nargs=3, default=[4,4,4])
    parser.add_argument('--mesh', type=int, nargs=3, default=[20,20,20])
    parser.add_argument('--displacement', type=float, default=0.01)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    supercell_matrix = np.diag(args.supercell)

    phonon = compute_phonon_with_nnp(
        args.structure,
        args.model,
        supercell_matrix=supercell_matrix,
        mesh=args.mesh,
        displacement=args.displacement,
        device=args.device
    )

    # 绘图
    import matplotlib.pyplot as plt
    phonon.plot_total_dos().savefig('phonon_dos.pdf')
    print("Saved: phonon_dos.pdf")
```

使用：
```bash
python auto_phonon.py POSCAR deployed_model.pth --supercell 5 5 5 --mesh 30 30 30
```

---

## 6. 实际案例

### 6.1 案例1：MOF材料的声子

金属有机框架（MOF）通常有几百个原子，DFT计算极其昂贵。

```python
# mof_phonon.py
from ase.io import read

# 读取MOF结构（例如：MOF-5）
mof = read('MOF5.cif')
print(f"MOF-5: {len(mof)} atoms")

# 用NequIP计算声子
from auto_phonon import compute_phonon_with_nnp

phonon = compute_phonon_with_nnp(
    'MOF5.cif',
    'mof_nequip_model.pth',
    supercell_matrix=[[2,0,0],[0,2,0],[0,0,2]],  # 小超胞，因为原胞已经很大
    mesh=[10,10,10],  # 粗网格
    device='cuda'
)

# 分析低频模式（柔性）
dos_dict = phonon.get_total_dos_dict()
freqs = dos_dict['frequency_points']
dos = dos_dict['total_dos']

# 找到低频峰（< 2 THz）
low_freq_mask = freqs < 2.0
low_freq_dos = dos[low_freq_mask]

print(f"Low-frequency DOS integral: {np.trapz(low_freq_dos, freqs[low_freq_mask]):.3f}")
print("This indicates the flexibility of the MOF framework.")
```

### 6.2 案例2：温度依赖的声子

某些材料的声子谱随温度变化（非谐效应）。

```python
# temperature_dependent_phonon.py
# 需要在不同温度下采样MD轨迹，训练temperature-dependent NNP

import numpy as np
from ase.io import read

temperatures = [300, 500, 700, 900]  # K
phonons = {}

for T in temperatures:
    # 加载对应温度的模型（或使用通用模型）
    model_path = f'nequip_T{T}K.pth'

    phonon = compute_phonon_with_nnp(
        'POSCAR',
        model_path,
        supercell_matrix=[[4,0,0],[0,4,0],[0,0,4]],
        mesh=[20,20,20]
    )

    phonons[T] = phonon

# 对比不同温度的声子DOS
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for T in temperatures:
    dos_dict = phonons[T].get_total_dos_dict()
    freqs = dos_dict['frequency_points']
    dos = dos_dict['total_dos']

    plt.plot(freqs, dos, label=f'{T} K', linewidth=2)

plt.xlabel('Frequency (THz)', fontsize=12)
plt.ylabel('DOS', fontsize=12)
plt.title('Temperature-Dependent Phonon DOS', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('phonon_dos_temperature.pdf')
plt.show()
```

### 6.3 案例3：缺陷对声子的影响

```python
# defect_phonon.py
from ase.build import bulk
from ase import Atoms
import numpy as np

# 完美晶体
perfect = bulk('Si', 'diamond', a=5.43) * (3, 3, 3)

# 引入空位缺陷
defect = perfect.copy()
del defect[0]  # 删除一个原子

print(f"Perfect: {len(perfect)} atoms")
print(f"Defect: {len(defect)} atoms")

# 计算两者的声子
phonon_perfect = compute_phonon_with_nnp(
    perfect,  # 可以直接传ASE Atoms
    'si_nequip.pth',
    supercell_matrix=[[2,0,0],[0,2,0],[0,0,2]]
)

phonon_defect = compute_phonon_with_nnp(
    defect,
    'si_nequip.pth',
    supercell_matrix=[[2,0,0],[0,2,0],[0,0,2]]
)

# 对比DOS
import matplotlib.pyplot as plt

dos_perfect = phonon_perfect.get_total_dos_dict()
dos_defect = phonon_defect.get_total_dos_dict()

plt.figure(figsize=(10, 6))
plt.plot(dos_perfect['frequency_points'], dos_perfect['total_dos'],
         'b-', label='Perfect', linewidth=2)
plt.plot(dos_defect['frequency_points'], dos_defect['total_dos'],
         'r--', label='With Vacancy', linewidth=2)
plt.xlabel('Frequency (THz)')
plt.ylabel('DOS')
plt.title('Effect of Vacancy on Phonon DOS')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('defect_vs_perfect_phonon.pdf')
plt.show()

# 分析局域振动模式（defect modes）
# 通常出现在带隙中
```

---

## 总结 | Summary

本实验介绍了使用Phonopy和神经网络势函数计算声子谱：

### 关键要点

1. **声子理论**：
   - 动力学矩阵和简正模式
   - 有限差分方法（frozen phonon）
   - 超胞方法

2. **Phonopy使用**：
   - 生成位移超胞
   - 收集力并计算力常数
   - 计算能带、DOS、热力学性质

3. **NNP加速**：
   - 50-1000x加速比
   - 可处理大体系（1000+原子）
   - 精度接近DFT（MAE < 0.1 THz）

4. **实际应用**：
   - MOF等大分子体系
   - 温度依赖声子
   - 缺陷态分析

### 最佳实践

- 超胞大小：至少4×4×4（取决于体系）
- 位移：0.01 Å（标准值）
- DOS网格：20×20×20或更密
- 检查虚频：<0.5 THz可忽略，>1 THz需检查结构

### 下一步

- 实践：计算你感兴趣材料的声子谱
- 探索：非谐效应（温度依赖）
- 应用：热导率计算（Boltzmann输运方程）

---

## 参考资源

**软件**：
- Phonopy: https://phonopy.github.io/phonopy/
- ASE: https://wiki.fysik.dtu.dk/ase/
- ALAMODE: https://alamode.readthedocs.io/ (非谐声子)

**教程**：
- Phonopy官方教程
- "Phonons and Lattice Dynamics" - M. Dove (教科书)

**论文**：
- "First principles phonon calculations in materials science" - Rev. Mod. Phys. 2001
- "Machine-learning interatomic potentials enable first-principles multiscale modeling of lattice thermal conductivity in graphene/borophene heterostructures" - Mater. Horiz. 2020

---

*Happy phonon hunting! 🎵*
