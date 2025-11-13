# ASE原子建模 | ASE Atomic Modeling

## 目录 | Table of Contents

1. [ASE简介](#1-ase简介)
2. [Atoms对象](#2-atoms对象)
3. [构建晶体结构](#3-构建晶体结构)
4. [构建分子和表面](#4-构建分子和表面)
5. [结构操作](#5-结构操作)
6. [文件输入输出](#6-文件输入输出)
7. [与GPAW结合](#7-与gpaw结合)

---

## 1. ASE简介

### 1.1 什么是ASE？

**ASE** (Atomic Simulation Environment) 是用Python编写的原子模拟工具集。

**核心功能**：
- 🔨 **结构构建**: 晶体、分子、表面、纳米结构
- 📐 **几何操作**: 平移、旋转、超胞、切割
- 🔍 **结构分析**: 键长、键角、配位数
- 📁 **文件I/O**: 支持100+种文件格式
- 🖥️ **计算器接口**: 统一接口连接各种DFT软件
- 📊 **可视化**: 内置查看器

### 1.2 安装ASE

```bash
# 使用conda安装（推荐）
$ conda install -c conda-forge ase

# 使用pip安装
$ pip install ase

# 验证安装
$ python -c "import ase; print(ase.__version__)"
3.22.1
```

### 1.3 第一个ASE程序

```python
from ase import Atoms
from ase.visualize import view

# 创建水分子
water = Atoms('H2O',
              positions=[[0.0, 0.0, 0.0],
                        [0.96, 0.0, 0.0],
                        [-0.24, 0.93, 0.0]])

# 设置晶胞
water.set_cell([10, 10, 10])
water.center()

# 查看结构
view(water)

# 保存文件
from ase.io import write
write('water.xyz', water)
```

---

## 2. Atoms对象

### 2.1 创建Atoms对象

```python
from ase import Atoms
import numpy as np

# 方法1: 指定元素符号和位置
atoms = Atoms('H2O',
              positions=[[0.0, 0.0, 0.0],
                        [0.96, 0.0, 0.0],
                        [-0.24, 0.93, 0.0]])

# 方法2: 分别指定
atoms = Atoms(symbols=['H', 'H', 'O'],
              positions=[[0.0, 0.0, 0.0],
                        [0.96, 0.0, 0.0],
                        [-0.24, 0.93, 0.0]])

# 方法3: 使用列表
atoms = Atoms(['H', 'H', 'O'],
              positions=np.array([[0.0, 0.0, 0.0],
                                 [0.96, 0.0, 0.0],
                                 [-0.24, 0.93, 0.0]]))
```

### 2.2 Atoms对象的属性

```python
from ase.build import bulk

# 创建硅晶体
si = bulk('Si', 'diamond', a=5.43)

# 基本属性
print(f"原子数: {len(si)}")  # 2
print(f"化学式: {si.get_chemical_formula()}")  # Si2
print(f"元素符号: {si.get_chemical_symbols()}")  # ['Si', 'Si']
print(f"原子序数: {si.get_atomic_numbers()}")  # [14, 14]

# 位置信息
print(f"原子位置:\n{si.get_positions()}")
print(f"缩放坐标:\n{si.get_scaled_positions()}")

# 晶胞信息
print(f"晶胞参数: {si.get_cell()}")
print(f"晶胞体积: {si.get_volume():.2f} Å³")
print(f"周期性: {si.get_pbc()}")  # [True, True, True]

# 质量
print(f"原子质量: {si.get_masses()}")
print(f"总质量: {si.get_masses().sum():.2f} amu")
```

### 2.3 修改Atoms对象

```python
from ase.build import bulk

si = bulk('Si', 'diamond', a=5.43)

# 设置晶胞
si.set_cell([[5.43, 0, 0],
             [0, 5.43, 0],
             [0, 0, 5.43]])

# 设置周期性边界条件
si.set_pbc([True, True, True])  # 三个方向都周期
si.set_pbc([True, True, False])  # 只xy方向周期（板模型）

# 修改位置
positions = si.get_positions()
positions[0] += [0.1, 0.0, 0.0]  # 移动第一个原子
si.set_positions(positions)

# 设置速度（用于分子动力学）
import numpy as np
velocities = np.random.randn(len(si), 3) * 0.01
si.set_velocities(velocities)
```

---

## 3. 构建晶体结构

### 3.1 使用bulk()函数

```python
from ase.build import bulk
from ase.io import write

# 常见晶体结构
# FCC (面心立方)
cu = bulk('Cu', 'fcc', a=3.61)
write('Cu_fcc.cif', cu)

# BCC (体心立方)
fe = bulk('Fe', 'bcc', a=2.87)

# Diamond (金刚石)
si = bulk('Si', 'diamond', a=5.43)
c = bulk('C', 'diamond', a=3.57)

# HCP (六方密排)
mg = bulk('Mg', 'hcp', a=3.21, c=5.21)

# Rocksalt (岩盐结构)
nacl = bulk('NaCl', 'rocksalt', a=5.64)

# Perovskite (钙钛矿)
srtio3 = bulk('SrTiO3', 'perovskite', a=3.905)

# 简单立方
po = bulk('Po', 'sc', a=3.35)

# Wurtzite (纤锌矿)
zno = bulk('ZnO', 'wurtzite', a=3.25, c=5.21)

# 查看结构信息
print(f"Cu: {cu.get_chemical_formula()}, 原子数={len(cu)}")
print(f"Si: {si.get_chemical_formula()}, 原子数={len(si)}")
print(f"NaCl: {nacl.get_chemical_formula()}, 原子数={len(nacl)}")
```

### 3.2 创建超胞

```python
from ase.build import bulk

# 创建原胞
si = bulk('Si', 'diamond', a=5.43)
print(f"原胞原子数: {len(si)}")  # 2

# 创建超胞 (2×2×2)
si_222 = si * (2, 2, 2)
print(f"2×2×2超胞原子数: {len(si_222)}")  # 16

# 创建非立方超胞
si_234 = si * (2, 3, 4)
print(f"2×3×4超胞原子数: {len(si_234)}")  # 48

# 使用repeat方法
si_222_alt = si.repeat((2, 2, 2))
```

### 3.3 手动构建晶体

```python
from ase import Atoms
import numpy as np

# 构建简单立方
a = 3.0  # 晶格常数
cell = [[a, 0, 0],
        [0, a, 0],
        [0, 0, a]]

positions = [[0, 0, 0]]  # 原子在(0,0,0)位置
atoms = Atoms('Na', positions=positions, cell=cell, pbc=True)

# 构建FCC
a = 4.0
cell = [[a, 0, 0],
        [0, a, 0],
        [0, 0, a]]

# FCC的原子位置
positions = [[0.0, 0.0, 0.0],  # 角
             [0.5, 0.5, 0.0],  # 面心
             [0.5, 0.0, 0.5],
             [0.0, 0.5, 0.5]]

atoms = Atoms('Cu' * 4,
              scaled_positions=positions,  # 使用缩放坐标
              cell=cell,
              pbc=True)

# 构建层状结构 (石墨烯)
a = 2.46
c = 20.0  # 大的c方向以分离层
cell = [[a, 0, 0],
        [-a/2, a*np.sqrt(3)/2, 0],
        [0, 0, c]]

positions = [[0, 0, 0],
             [1/3, 2/3, 0]]

graphene = Atoms('C2',
                 scaled_positions=positions,
                 cell=cell,
                 pbc=[True, True, False])  # 只在xy方向周期
```

---

## 4. 构建分子和表面

### 4.1 构建分子

```python
from ase.build import molecule
from ase.io import write

# 预定义分子
h2o = molecule('H2O')
ch4 = molecule('CH4')
co2 = molecule('CO2')
benzene = molecule('C6H6')

# 查看可用分子
from ase.data.molecules import molecule_names
print(f"可用分子数: {len(molecule_names)}")
print(f"示例: {molecule_names[:10]}")

# 将分子放入盒子中
h2o.set_cell([10, 10, 10])
h2o.center()  # 居中
write('h2o_in_box.xyz', h2o)

# 手动构建分子
from ase import Atoms

# CO2分子 O=C=O
co2 = Atoms('OCO',
            positions=[[-1.16, 0, 0],
                      [0, 0, 0],
                      [1.16, 0, 0]])
co2.set_cell([10, 10, 10])
co2.center()
```

### 4.2 构建表面

```python
from ase.build import surface, add_adsorbate
from ase.io import write

# 创建FCC(111)表面
slab = surface('Au', (1, 1, 1), layers=4)
print(f"表面原子数: {len(slab)}")

# 添加真空层
slab.center(vacuum=10.0, axis=2)  # z方向加10Å真空
print(f"晶胞高度: {slab.cell[2, 2]:.2f} Å")

# 保存
write('Au_111_surface.cif', slab)

# 不同的表面
fcc111 = surface('Cu', (1, 1, 1), layers=4)  # (111)面
fcc100 = surface('Cu', (1, 0, 0), layers=4)  # (100)面
fcc110 = surface('Cu', (1, 1, 0), layers=4)  # (110)面

# 在表面上添加吸附物
from ase.build import molecule

# 创建表面
slab = surface('Pt', (1, 1, 1), layers=4)
slab.center(vacuum=10.0, axis=2)

# 创建CO分子
co = molecule('CO')

# 添加到表面顶位 (top site)
add_adsorbate(slab, co, height=2.0, position='ontop')
write('Pt111_CO.xyz', slab)

# 添加到桥位 (bridge site)
slab2 = surface('Pt', (1, 1, 1), layers=4)
slab2.center(vacuum=10.0, axis=2)
add_adsorbate(slab2, 'O', height=1.5, position='bridge')

# 添加到三重位 (hollow site)
slab3 = surface('Pt', (1, 1, 1), layers=4)
slab3.center(vacuum=10.0, axis=2)
add_adsorbate(slab3, 'N', height=1.5, position='hcp')
```

### 4.3 构建纳米结构

```python
from ase.cluster import FaceCenteredCubic

# 纳米团簇
# FCC纳米粒子
surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers = [6, 9, 5]
nanoparticle = FaceCenteredCubic('Au',
                                  surfaces,
                                  layers,
                                  latticeconstant=4.08)

print(f"纳米粒子原子数: {len(nanoparticle)}")
write('Au_nanoparticle.xyz', nanoparticle)

# 纳米管 (简化示例)
from ase.build import nanotube
cnt = nanotube(6, 6, length=4)  # (6,6)碳纳米管
print(f"纳米管原子数: {len(cnt)}")
```

---

## 5. 结构操作

### 5.1 几何变换

```python
from ase.build import bulk
import numpy as np

si = bulk('Si', 'diamond', a=5.43)

# 平移
si.translate([1.0, 0.0, 0.0])  # x方向平移1Å

# 居中
si.center()  # 在晶胞中居中
si.center(vacuum=10.0, axis=2)  # z方向加真空并居中

# 旋转
si.rotate(45, 'z')  # 绕z轴旋转45度
si.rotate('x', 'y')  # 将x方向转到y方向

# 缩放
cell = si.get_cell()
si.set_cell(cell * 1.1, scale_atoms=True)  # 放大10%
```

### 5.2 结构修改

```python
from ase.build import bulk

si = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)

# 删除原子
del si[0]  # 删除第一个原子
del si[[0, 1, 2]]  # 删除多个原子

# 添加原子
from ase import Atom
si.append(Atom('C', position=[0, 0, 0]))  # 添加碳原子

# 替换原子
si[0].symbol = 'Ge'  # 将第一个原子改为Ge

# 掺杂
# 随机替换一个Si为P
import random
index = random.randint(0, len(si) - 1)
si[index].symbol = 'P'

# 创建空位
positions = si.get_positions()
center = positions.mean(axis=0)
distances = np.linalg.norm(positions - center, axis=1)
vacancy_index = np.argmin(distances)
del si[vacancy_index]
```

### 5.3 结构分析

```python
from ase.build import bulk
from ase.neighborlist import NeighborList, natural_cutoffs

si = bulk('Si', 'diamond', a=5.43)

# 获取最近邻
cutoffs = natural_cutoffs(si)
nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
nl.update(si)

# 查找原子0的近邻
indices, offsets = nl.get_neighbors(0)
print(f"原子0的近邻: {indices}")
print(f"近邻数: {len(indices)}")

# 计算键长
from ase.geometry import get_distances
positions = si.get_positions()
atom0_pos = positions[0]
atom1_pos = positions[1]
distance = np.linalg.norm(atom1_pos - atom0_pos)
print(f"键长: {distance:.3f} Å")

# 使用更高级的分析
from ase.geometry.analysis import Analysis
ana = Analysis(si)

# 获取所有键
bonds = ana.all_bonds[0]  # 第一个原子的所有键
print(f"键: {bonds}")
```

---

## 6. 文件输入输出

### 6.1 读取结构文件

```python
from ase.io import read

# 读取单个结构
atoms = read('structure.cif')
atoms = read('POSCAR')  # VASP格式
atoms = read('geometry.in')  # FHI-aims格式
atoms = read('structure.xyz')

# 读取多个结构（轨迹文件）
images = read('trajectory.traj', index=':')  # 读取所有
images = read('trajectory.traj', index='0:10')  # 前10个
images = read('trajectory.traj', index='-1')  # 最后一个

# 读取特定格式
atoms = read('file.xyz', format='xyz')
```

### 6.2 写入结构文件

```python
from ase.io import write
from ase.build import bulk

si = bulk('Si', 'diamond', a=5.43)

# 写入不同格式
write('si.cif', si)  # CIF格式
write('si.xyz', si)  # XYZ格式
write('POSCAR', si)  # VASP格式
write('si.pdb', si)  # PDB格式
write('si.json', si)  # JSON格式

# 写入图片
write('si_structure.png', si, rotation='10x,10y,10z')

# 写入多个结构
images = [si, si * (2, 1, 1), si * (3, 1, 1)]
write('structures.traj', images)
```

### 6.3 支持的文件格式

```python
from ase.io.formats import all_formats

# 查看所有支持的格式
print("ASE支持的文件格式:")
for format_name, (description, code) in all_formats.items():
    print(f"  {format_name}: {description}")

# 常用格式:
# - cif: Crystallographic Information File
# - xyz: XYZ coordinates
# - pdb: Protein Data Bank
# - traj: ASE trajectory
# - json: JavaScript Object Notation
# - vasp: VASP POSCAR/CONTCAR
# - espresso-in: Quantum ESPRESSO input
# - aims: FHI-aims geometry
```

---

## 7. 与GPAW结合

### 7.1 基本计算设置

```python
from ase.build import bulk
from gpaw import GPAW, PW

# 创建结构
si = bulk('Si', 'diamond', a=5.43)

# 设置GPAW计算器
calc = GPAW(mode=PW(400),  # 平面波，截断能400 eV
            xc='PBE',  # 交换关联泛函
            kpts=(4, 4, 4),  # k点网格
            txt='si_calculation.txt')  # 输出文件

# 将计算器附加到原子对象
si.calc = calc

# 计算能量和力
energy = si.get_potential_energy()
forces = si.get_forces()

print(f"能量: {energy:.3f} eV")
print(f"力:\n{forces}")
```

### 7.2 结构优化

```python
from ase.build import bulk
from ase.optimize import BFGS
from gpaw import GPAW, PW

# 创建稍微变形的结构
si = bulk('Si', 'diamond', a=5.5)  # 略大于平衡值

# 设置计算器
calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(4, 4, 4),
            txt='optimization.txt')

si.calc = calc

# 优化结构
opt = BFGS(si, trajectory='si_opt.traj')
opt.run(fmax=0.01)  # 力小于0.01 eV/Å时停止

# 获取优化后的结果
optimized_a = si.cell[0, 0]
final_energy = si.get_potential_energy()

print(f"优化后晶格常数: {optimized_a:.3f} Å")
print(f"最终能量: {final_energy:.3f} eV")
```

### 7.3 完整示例：晶格常数扫描

```python
from ase.build import bulk
from gpaw import GPAW, PW
import numpy as np
import matplotlib.pyplot as plt

def calculate_energy_vs_volume():
    """计算不同体积下的能量"""

    # 扫描范围
    lattice_constants = np.linspace(5.2, 5.6, 9)
    energies = []
    volumes = []

    for a in lattice_constants:
        # 创建结构
        si = bulk('Si', 'diamond', a=a)

        # 设置计算器
        calc = GPAW(mode=PW(400),
                    xc='PBE',
                    kpts=(8, 8, 8),
                    txt=f'si_a{a:.2f}.txt')

        si.calc = calc

        # 计算
        energy = si.get_potential_energy()
        volume = si.get_volume()

        energies.append(energy / len(si))  # 每原子能量
        volumes.append(volume / len(si))  # 每原子体积

        print(f"a = {a:.3f} Å, E/atom = {energy/len(si):.4f} eV")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(volumes, energies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Volume per atom (Å³)', fontsize=12)
    plt.ylabel('Energy per atom (eV)', fontsize=12)
    plt.title('Si: E-V Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('si_ev_curve.png', dpi=300)

    # 拟合Birch-Murnaghan状态方程
    from ase.eos import calculate_eos
    eos = calculate_eos(volumes, energies)
    v0, e0, B = eos.fit()  # 平衡体积、能量、体模量

    print(f"\n拟合结果:")
    print(f"平衡体积: {v0:.3f} Å³/atom")
    print(f"平衡能量: {e0:.4f} eV/atom")
    print(f"体模量: {B/1e9:.1f} GPa")

if __name__ == '__main__':
    calculate_energy_vs_volume()
```

---

## 8. 实用示例

### 示例1：构建掺杂结构

```python
from ase.build import bulk
import random

def create_doped_structure(host='Si', dopant='P', concentration=0.01, size=(4, 4, 4)):
    """
    创建掺杂结构

    Parameters:
    -----------
    host : str
        主体材料
    dopant : str
        掺杂元素
    concentration : float
        掺杂浓度
    size : tuple
        超胞大小
    """
    # 创建超胞
    atoms = bulk(host, 'diamond', a=5.43) * size

    # 计算要替换的原子数
    n_dopants = int(len(atoms) * concentration)

    # 随机选择原子进行替换
    indices = random.sample(range(len(atoms)), n_dopants)
    for i in indices:
        atoms[i].symbol = dopant

    formula = atoms.get_chemical_formula()
    print(f"创建掺杂结构: {formula}")
    print(f"总原子数: {len(atoms)}")
    print(f"掺杂浓度: {n_dopants/len(atoms):.2%}")

    return atoms

# 使用
doped_si = create_doped_structure('Si', 'P', 0.05, (3, 3, 3))
```

### 示例2：创建异质结

```python
from ase.build import bulk

def create_heterostructure(material1='Si', material2='Ge',
                          layers1=5, layers2=5):
    """创建异质结"""
    from ase.build import surface

    # 创建两种材料的表面
    slab1 = surface(material1, (0, 0, 1), layers=layers1)
    slab2 = surface(material2, (0, 0, 1), layers=layers2)

    # 组合（简化版）
    # 实际应该考虑晶格匹配等
    hetero = slab1 + slab2

    hetero.center(vacuum=10, axis=2)

    return hetero

# 使用
hetero = create_heterostructure('Si', 'Ge', 4, 4)
print(f"异质结原子数: {len(hetero)}")
```

---

## 9. 学习检查清单

- [ ] 理解Atoms对象的基本概念
- [ ] 能构建常见晶体结构
- [ ] 会创建分子和表面
- [ ] 掌握结构的几何操作
- [ ] 会读写各种格式的结构文件
- [ ] 能将ASE与GPAW结合使用
- [ ] 会创建复杂结构（掺杂、异质结等）

---

## 参考资源

- [ASE官方文档](https://wiki.fysik.dtu.dk/ase/)
- [ASE教程](https://wiki.fysik.dtu.dk/ase/tutorials/tutorials.html)
- [ASE数据库](https://wiki.fysik.dtu.dk/ase/ase/data.html)

**更多示例**：查看 `examples/` 目录 📁

**下一步**：GPAW第一性原理计算 →
