# GPAW第一性原理计算 | GPAW First-Principles Calculations

## 目录 | Table of Contents

1. [GPAW简介](#1-gpaw简介)
2. [基本计算流程](#2-基本计算流程)
3. [参数设置](#3-参数设置)
4. [收敛性测试](#4-收敛性测试)
5. [性质计算](#5-性质计算)
6. [高级功能](#6-高级功能)

---

## 1. GPAW简介

### 1.1 什么是GPAW?

**GPAW** (Grid-based Projector Augmented Wave) 是基于密度泛函理论的第一性原理计算软件。

**特点**：
- 🐍 **Python编写**: 易于使用和扩展
- 🆓 **开源免费**: GPL许可
- 🔧 **多种模式**: 平面波、实空间网格、原子轨道
- 🔬 **功能丰富**: 能量、力、应力、电荷密度、能带等
- 🚀 **高效并行**: 支持MPI和OpenMP

### 1.2 安装GPAW

```bash
# 方法1: 使用conda（推荐）
$ conda install -c conda-forge gpaw

# 方法2: 使用pip
$ pip install gpaw

# 安装PAW数据集
$ gpaw install-data /path/to/gpaw-setups
```

### 1.3 验证安装

```python
import gpaw
print(f"GPAW版本: {gpaw.__version__}")

# 测试安装
from gpaw import GPAW
print("GPAW安装成功！")
```

---

## 2. 基本计算流程

### 2.1 最简单的计算

```python
from ase.build import bulk
from gpaw import GPAW, PW

# 1. 创建结构
si = bulk('Si', 'diamond', a=5.43)

# 2. 设置计算器
calc = GPAW(mode=PW(400),  # 平面波，截断能400 eV
            xc='PBE',      # PBE泛函
            kpts=(4, 4, 4),  # k点网格
            txt='si.txt')  # 输出文件

# 3. 附加计算器
si.calc = calc

# 4. 计算
energy = si.get_potential_energy()
forces = si.get_forces()

print(f"能量: {energy:.4f} eV")
print(f"力:\n{forces}")
```

### 2.2 自洽计算步骤

GPAW执行以下自洽循环：

```
初始化
  ↓
猜测初始电子密度
  ↓
┌─────────────────┐
│ 计算有效势      │
│      ↓          │
│ 求解Kohn-Sham  │
│      ↓          │
│ 计算新密度      │
│      ↓          │
│ 检查收敛？      │
└─────────────────┘
  ↓ (收敛)
计算总能量、力等
  ↓
输出结果
```

### 2.3 读取计算结果

```python
from gpaw import GPAW
from ase.io import read

# 方法1: 从gpw文件读取
calc = GPAW('si.gpw')
atoms = calc.get_atoms()
energy = atoms.get_potential_energy()

# 方法2: 使用ASE读取
atoms = read('si.gpw')
energy = atoms.get_potential_energy()

# 获取更多信息
forces = calc.get_forces(atoms)
stress = calc.get_stress(atoms)
fermi_level = calc.get_fermi_level()

print(f"费米能级: {fermi_level:.3f} eV")
```

---

## 3. 参数设置

### 3.1 计算模式

#### 平面波模式 (PW)

```python
from gpaw import GPAW, PW

# 适合周期性体系
calc = GPAW(mode=PW(400),  # 截断能 (eV)
            kpts=(8, 8, 8))

# 不同截断能
calc_300 = GPAW(mode=PW(300), kpts=(8, 8, 8))  # 粗糙
calc_400 = GPAW(mode=PW(400), kpts=(8, 8, 8))  # 标准
calc_600 = GPAW(mode=PW(600), kpts=(8, 8, 8))  # 精细
```

#### 实空间网格模式 (FD)

```python
from gpaw import GPAW

# 适合分子、表面等
calc = GPAW(mode='fd',     # 有限差分
            h=0.2,         # 网格间距 (Å)
            kpts=(1, 1, 1))

# 不同网格精度
calc_coarse = GPAW(mode='fd', h=0.3)  # 粗糙
calc_fine = GPAW(mode='fd', h=0.18)   # 精细
```

#### 原子轨道模式 (LCAO)

```python
from gpaw import GPAW

# 最快，适合大体系
calc = GPAW(mode='lcao',        # 线性组合原子轨道
            basis='dzp',        # 基组: sz, dz, szp, dzp, tzp
            kpts=(4, 4, 4))
```

### 3.2 交换关联泛函

```python
from gpaw import GPAW, PW

# LDA
calc_lda = GPAW(mode=PW(400), xc='LDA')

# GGA - PBE
calc_pbe = GPAW(mode=PW(400), xc='PBE')

# GGA - RPBE (改善吸附能)
calc_rpbe = GPAW(mode=PW(400), xc='RPBE')

# GGA - PBEsol (固体优化)
calc_pbesol = GPAW(mode=PW(400), xc='PBEsol')

# 范德华修正
calc_vdw = GPAW(mode=PW(400), xc='vdW-DF')
calc_beef = GPAW(mode=PW(400), xc='BEEF-vdW')
```

### 3.3 k点设置

```python
from gpaw import GPAW, PW

# 均匀网格
calc = GPAW(mode=PW(400), kpts=(8, 8, 8))

# 不同方向不同密度（板模型）
calc_slab = GPAW(mode=PW(400), kpts=(8, 8, 1))

# Monkhorst-Pack
calc_mp = GPAW(mode=PW(400), kpts={'size': (8, 8, 8), 'gamma': True})

# 指定k点密度（k点/Å）
calc_density = GPAW(mode=PW(400), kpts={'density': 5.0})

# Gamma点计算（分子）
calc_gamma = GPAW(mode=PW(400), kpts=(1, 1, 1))
```

### 3.4 收敛标准

```python
from gpaw import GPAW, PW

calc = GPAW(
    mode=PW(400),
    xc='PBE',
    kpts=(8, 8, 8),

    # 收敛标准
    convergence={
        'energy': 0.0001,  # 能量收敛 (eV)
        'density': 1.0e-4,  # 密度收敛
        'eigenstates': 1.0e-4,  # 本征态收敛
        'bands': -10  # 所有能带
    },

    # 最大迭代次数
    maxiter=300,

    # 混合参数
    mixer={'backend': 'pulay',  # Pulay混合
           'beta': 0.05,        # 混合系数
           'nmaxold': 5,        # 历史步数
           'weight': 100.0},    # 权重

    txt='calculation.txt'
)
```

### 3.5 其他重要参数

```python
from gpaw import GPAW, PW, FermiDirac

calc = GPAW(
    mode=PW(400),
    xc='PBE',
    kpts=(8, 8, 8),

    # 占据数平滑
    occupations=FermiDirac(0.1),  # 展宽 0.1 eV

    # 对称性
    symmetry={'point_group': True,  # 使用点群
              'time_reversal': True},  # 时间反演对称

    # 自旋极化
    spinpol=False,  # 不考虑自旋（默认）
    # spinpol=True,   # 考虑自旋

    # 电荷
    charge=0,  # 中性体系

    # 并行
    parallel={'domain': 1,  # 实空间并行
              'band': 1},    # 能带并行

    txt='output.txt'
)
```

---

## 4. 收敛性测试

### 4.1 截断能收敛测试

```python
from ase.build import bulk
from gpaw import GPAW, PW
import numpy as np
import matplotlib.pyplot as plt

def test_encut_convergence():
    """测试截断能收敛性"""

    si = bulk('Si', 'diamond', a=5.43)

    # 测试范围
    encuts = np.arange(200, 650, 50)
    energies = []

    for encut in encuts:
        calc = GPAW(mode=PW(encut),
                    xc='PBE',
                    kpts=(8, 8, 8),
                    txt=f'encut_{encut}.txt')

        si.calc = calc
        energy = si.get_potential_energy()
        energies.append(energy / len(si))  # 每原子能量

        print(f"ENCUT = {encut} eV: E = {energy/len(si):.4f} eV/atom")

    # 计算相对参考值的差异
    energies = np.array(energies)
    ref_energy = energies[-1]  # 最高截断能为参考
    diff = (energies - ref_energy) * 1000  # meV

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(encuts, diff, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', label='1 meV threshold')
    plt.xlabel('Cutoff Energy (eV)', fontsize=12)
    plt.ylabel('Energy Difference (meV/atom)', fontsize=12)
    plt.title('Cutoff Energy Convergence', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('encut_convergence.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    test_encut_convergence()
```

### 4.2 k点收敛测试

```python
from ase.build import bulk
from gpaw import GPAW, PW
import numpy as np
import matplotlib.pyplot as plt

def test_kpts_convergence():
    """测试k点收敛性"""

    si = bulk('Si', 'diamond', a=5.43)

    # 测试不同k点密度
    kpts_list = [(2,2,2), (4,4,4), (6,6,6), (8,8,8), (10,10,10), (12,12,12)]
    energies = []

    for kpts in kpts_list:
        calc = GPAW(mode=PW(400),
                    xc='PBE',
                    kpts=kpts,
                    txt=f'kpts_{kpts[0]}x{kpts[1]}x{kpts[2]}.txt')

        si.calc = calc
        energy = si.get_potential_energy()
        energies.append(energy / len(si))

        print(f"k-points = {kpts}: E = {energy/len(si):.4f} eV/atom")

    # 绘图
    k_values = [k[0] for k in kpts_list]
    energies = np.array(energies)
    ref_energy = energies[-1]
    diff = (energies - ref_energy) * 1000  # meV

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, diff, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', label='1 meV threshold')
    plt.xlabel('k-point Grid (n×n×n)', fontsize=12)
    plt.ylabel('Energy Difference (meV/atom)', fontsize=12)
    plt.title('k-point Convergence', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kpts_convergence.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    test_kpts_convergence()
```

### 4.3 检查自洽收敛

```python
def check_scf_convergence(txt_file='output.txt'):
    """检查自洽迭代收敛情况"""

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # 提取迭代信息
    iterations = []
    energies = []

    for line in lines:
        if 'iter:' in line:
            parts = line.split()
            iter_num = int(parts[1])
            energy = float(parts[4])
            iterations.append(iter_num)
            energies.append(energy)

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, 'o-', linewidth=2)
    plt.xlabel('SCF Iteration', fontsize=12)
    plt.ylabel('Total Energy (eV)', fontsize=12)
    plt.title('SCF Convergence', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scf_convergence.png', dpi=300)
    plt.show()

    # 检查是否收敛
    if len(energies) > 1:
        final_diff = abs(energies[-1] - energies[-2])
        print(f"最后两步能量差: {final_diff:.6f} eV")
        if final_diff < 1e-4:
            print("✓ 已收敛")
        else:
            print("✗ 未完全收敛")
```

---

## 5. 性质计算

### 5.1 能量计算

```python
from ase.build import bulk
from gpaw import GPAW, PW

si = bulk('Si', 'diamond', a=5.43)

calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(8, 8, 8),
            txt='si.txt')

si.calc = calc

# 总能量
energy = si.get_potential_energy()
print(f"总能量: {energy:.4f} eV")

# 每原子能量
energy_per_atom = energy / len(si)
print(f"每原子能量: {energy_per_atom:.4f} eV")

# 费米能级
fermi = calc.get_fermi_level()
print(f"费米能级: {fermi:.3f} eV")
```

### 5.2 力和应力计算

```python
from ase.build import bulk
from gpaw import GPAW, PW
import numpy as np

si = bulk('Si', 'diamond', a=5.43)

# 略微变形
si.set_cell(si.cell * 1.01, scale_atoms=True)

calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(8, 8, 8),
            txt='si_stress.txt')

si.calc = calc

# 计算力
forces = si.get_forces()
print("原子受力 (eV/Å):")
print(forces)
print(f"最大力: {np.max(np.abs(forces)):.4f} eV/Å")

# 计算应力
stress = si.get_stress()  # Voigt形式：xx, yy, zz, yz, xz, xy
print("\n应力张量 (eV/Å³):")
print(stress)

# 转换为GPa
stress_gpa = stress * 160.21766208  # 转换因子
print(stress_gpa)
```

### 5.3 电荷密度

```python
from ase.build import bulk
from gpaw import GPAW, PW

si = bulk('Si', 'diamond', a=5.43)

calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(8, 8, 8),
            txt='si.txt')

si.calc = calc
si.get_potential_energy()  # 先计算

# 获取电荷密度
density = calc.get_pseudo_density()
print(f"电荷密度形状: {density.shape}")

# 保存电荷密度
from ase.io import write
write('density.cube', si, data=density)

# 绘制平面图
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(density[:, :, 0], cmap='viridis')
plt.colorbar(label='Charge Density (e/Å³)')
plt.title('Electron Density (z=0 plane)')
plt.tight_layout()
plt.savefig('charge_density.png', dpi=300)
```

### 5.4 能带结构

```python
from ase.build import bulk
from gpaw import GPAW, PW

# 第一步：自洽计算
si = bulk('Si', 'diamond', a=5.43)

calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(8, 8, 8),
            txt='si_scf.txt')

si.calc = calc
si.get_potential_energy()
calc.write('si_gs.gpw')

# 第二步：能带计算
calc = GPAW('si_gs.gpw', txt='si_bands.txt')

# 定义高对称点路径
kpts = {'path': 'GXWKL',  # Γ-X-W-K-L
        'npoints': 60}

calc.get_fermi_level()
calc_bands = calc.get_band_structure(kpts=kpts)

# 绘制能带
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
calc_bands.plot(ax=ax, emin=-10, emax=10)
plt.tight_layout()
plt.savefig('si_bands.png', dpi=300)
```

### 5.5 态密度 (DOS)

```python
from ase.build import bulk
from gpaw import GPAW, PW

si = bulk('Si', 'diamond', a=5.43)

calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(12, 12, 12),  # DOS需要密集k点
            txt='si_dos.txt')

si.calc = calc
si.get_potential_energy()

# 计算DOS
energies, dos = calc.get_dos(npts=1000, width=0.1)

# 绘图
import matplotlib.pyplot as plt
import numpy as np

fermi = calc.get_fermi_level()
energies -= fermi  # 相对费米能级

plt.figure(figsize=(8, 6))
plt.plot(energies, dos, linewidth=2)
plt.axvline(x=0, color='r', linestyle='--', label='Fermi Level')
plt.xlabel('Energy (eV)', fontsize=12)
plt.ylabel('DOS (states/eV)', fontsize=12)
plt.title('Silicon Density of States', fontsize=14)
plt.xlim(-10, 10)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('si_dos.png', dpi=300)
```

---

## 6. 高级功能

### 6.1 结构优化

```python
from ase.build import bulk
from ase.optimize import BFGS
from gpaw import GPAW, PW

# 创建初始结构（稍微偏离平衡）
si = bulk('Si', 'diamond', a=5.5)

calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(8, 8, 8),
            txt='si_opt.txt')

si.calc = calc

# 结构优化
opt = BFGS(si, trajectory='si_optimization.traj', logfile='opt.log')
opt.run(fmax=0.01)  # 最大力 < 0.01 eV/Å

# 结果
final_a = si.cell[0, 0]
final_energy = si.get_potential_energy()

print(f"优化后晶格常数: {final_a:.3f} Å")
print(f"最终能量: {final_energy:.4f} eV")

# 分析轨迹
from ase.io import read
traj = read('si_optimization.traj', ':')
print(f"优化步数: {len(traj)}")
```

### 6.2 分子动力学

```python
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from gpaw import GPAW

# 创建小体系
si = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)

# 使用快速模式
calc = GPAW(mode='lcao',
            basis='sz',
            kpts=(2, 2, 2),
            txt='md.txt')

si.calc = calc

# 设置初始速度（300 K）
MaxwellBoltzmannDistribution(si, temperature_K=300)

# MD积分器
dyn = VelocityVerlet(si, timestep=2.0 * units.fs, trajectory='md.traj')

# 运行50步
dyn.run(50)

# 分析温度
from ase.md.analysis import DiffusionCoefficient
traj = read('md.traj', ':')
temps = [atoms.get_temperature() for atoms in traj]

print(f"平均温度: {np.mean(temps):.1f} K")
```

### 6.3 表面吸附能计算

```python
from ase.build import surface, molecule, add_adsorbate
from gpaw import GPAW, PW

def calculate_adsorption_energy():
    """计算CO在Pt(111)表面的吸附能"""

    # 1. 干净表面
    slab = surface('Pt', (1, 1, 1), layers=4, vacuum=10.0)

    calc = GPAW(mode=PW(400),
                xc='PBE',
                kpts=(4, 4, 1),
                txt='slab.txt')

    slab.calc = calc
    e_slab = slab.get_potential_energy()

    # 2. 吸附体系
    slab_ads = slab.copy()
    co = molecule('CO')
    add_adsorbate(slab_ads, co, height=2.0, position='ontop')

    slab_ads.calc = calc
    e_slab_ads = slab_ads.get_potential_energy()

    # 3. 气相CO
    co_gas = molecule('CO')
    co_gas.center(vacuum=10.0)

    calc_mol = GPAW(mode=PW(400),
                    xc='PBE',
                    txt='co.txt')

    co_gas.calc = calc_mol
    e_co = co_gas.get_potential_energy()

    # 4. 计算吸附能
    e_ads = e_slab_ads - e_slab - e_co

    print(f"表面能量: {e_slab:.3f} eV")
    print(f"吸附体系能量: {e_slab_ads:.3f} eV")
    print(f"CO能量: {e_co:.3f} eV")
    print(f"吸附能: {e_ads:.3f} eV")

    return e_ads

if __name__ == '__main__':
    e_ads = calculate_adsorption_energy()
```

### 6.4 自旋极化计算

```python
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac

# O2分子（三重态）
o2 = Atoms('O2', positions=[[0, 0, 0], [1.2, 0, 0]])
o2.center(vacuum=6.0)

# 自旋极化计算
calc = GPAW(mode=PW(400),
            xc='PBE',
            spinpol=True,  # 开启自旋极化
            occupations=FermiDirac(0.1),
            txt='o2_spin.txt')

o2.calc = calc
energy = o2.get_potential_energy()

# 获取磁矩
magmom = o2.get_magnetic_moment()
print(f"总磁矩: {magmom:.2f} μB")

# 每个原子的磁矩
magmoms = o2.get_magnetic_moments()
print(f"原子磁矩: {magmoms}")
```

---

## 7. 实用脚本模板

### 7.1 批量计算模板

```python
"""
批量计算不同材料的晶格常数和体模量
"""
from ase.build import bulk
from gpaw import GPAW, PW
from ase.eos import calculate_eos
import numpy as np

def calculate_lattice_constant_and_bulk_modulus(element, structure, a_guess):
    """
    计算平衡晶格常数和体模量

    Parameters:
    -----------
    element : str
        元素符号
    structure : str
        晶体结构 ('fcc', 'bcc', 'diamond'等)
    a_guess : float
        初始晶格常数猜测值
    """

    # 扫描晶格常数
    a_values = np.linspace(a_guess * 0.95, a_guess * 1.05, 7)
    energies = []
    volumes = []

    for a in a_values:
        atoms = bulk(element, structure, a=a)

        calc = GPAW(mode=PW(400),
                    xc='PBE',
                    kpts=(12, 12, 12),
                    txt=f'{element}_{a:.3f}.txt')

        atoms.calc = calc
        e = atoms.get_potential_energy()
        v = atoms.get_volume()

        energies.append(e / len(atoms))
        volumes.append(v / len(atoms))

        print(f"{element} a={a:.3f}: E={e/len(atoms):.4f} eV/atom")

    # 拟合EOS
    from ase.eos import EquationOfState
    eos = EquationOfState(volumes, energies)
    v0, e0, B = eos.fit()

    # 计算平衡晶格常数
    atoms_ref = bulk(element, structure, a=a_guess)
    v_cell = atoms_ref.get_volume()
    n_atoms = len(atoms_ref)
    a0 = a_guess * (v0 * n_atoms / v_cell) ** (1/3)

    print(f"\n结果:")
    print(f"平衡晶格常数: {a0:.3f} Å")
    print(f"平衡能量: {e0:.4f} eV/atom")
    print(f"体模量: {B/1e9:.1f} GPa")

    return a0, e0, B

# 使用示例
if __name__ == '__main__':
    materials = [
        ('Si', 'diamond', 5.43),
        ('Cu', 'fcc', 3.61),
        ('Fe', 'bcc', 2.87)
    ]

    results = {}
    for element, structure, a_guess in materials:
        print(f"\n{'='*50}")
        print(f"计算 {element} ({structure})")
        print('='*50)
        a0, e0, B = calculate_lattice_constant_and_bulk_modulus(
            element, structure, a_guess)
        results[element] = {'a0': a0, 'e0': e0, 'B': B}

    # 输出汇总
    print(f"\n{'='*50}")
    print("汇总结果")
    print('='*50)
    for element, data in results.items():
        print(f"{element:3s}: a0 = {data['a0']:.3f} Å, "
              f"B = {data['B']/1e9:.1f} GPa")
```

---

## 8. 学习检查清单

- [ ] 理解GPAW的基本工作流程
- [ ] 掌握不同计算模式的使用
- [ ] 会设置合适的计算参数
- [ ] 能进行收敛性测试
- [ ] 会计算常见物理量（能量、力、应力等）
- [ ] 能计算能带和态密度
- [ ] 掌握结构优化方法
- [ ] 了解自旋极化计算

---

## 参考资源

- [GPAW官方文档](https://wiki.fysik.dtu.dk/gpaw/)
- [GPAW教程](https://wiki.fysik.dtu.dk/gpaw/tutorialsexercises/tutorialsexercises.html)
- [ASE+GPAW组合](https://wiki.fysik.dtu.dk/ase/ase/calculators/gpaw.html)

**完整示例**：查看 `examples/` 目录 📁

**恭喜完成第一部分的学习！🎉**
