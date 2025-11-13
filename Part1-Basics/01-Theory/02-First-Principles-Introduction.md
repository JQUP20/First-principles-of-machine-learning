# 第一性原理计算介绍 | Introduction to First-Principles Calculations

## 1. 第一性原理计算的发展历程

### 1.1 从薛定谔方程到密度泛函理论

#### 1.1.1 量子力学的诞生（1920s）

**Schrödinger方程** (1926)

含时薛定谔方程：
```
iℏ ∂Ψ/∂t = ĤΨ
```

定态薛定谔方程：
```
ĤΨ = EΨ
```

其中：
- `Ψ`: 波函数
- `Ĥ`: 哈密顿算符
- `E`: 能量本征值
- `ℏ`: 约化普朗克常数

对于多电子体系，哈密顿算符包括：

```
Ĥ = T̂ₑ + T̂ₙ + V̂ₑₑ + V̂ₙₙ + V̂ₑₙ

T̂ₑ: 电子动能
T̂ₙ: 原子核动能
V̂ₑₑ: 电子-电子相互作用
V̂ₙₙ: 核-核相互作用
V̂ₑₙ: 电子-核相互作用
```

**挑战**：多电子薛定谔方程无法精确求解！

对于N个电子的体系，波函数是3N维空间的函数，计算复杂度随N指数增长。

#### 1.1.2 Born-Oppenheimer近似（1927）

**核心思想**：电子运动远快于原子核运动，可以分别处理

```
Ψₜₒₜₐₗ(r, R) ≈ Ψₑₗₑ(r; R) × Ψₙᵤc(R)
```

**结果**：
- 将问题简化为在固定核位置下求解电子结构
- 原子核提供外势场
- 极大简化了计算

#### 1.1.3 Hartree-Fock方法（1930s）

**平均场近似**：每个电子在其他所有电子的平均场中运动

**Hartree-Fock方程**：
```
[-½∇² + Vₑₓₜ(r) + VH(r) - Vₓ(r)]φᵢ(r) = εᵢφᵢ(r)
```

其中：
- `VH(r)`: Hartree势（电子-电子经典库仑排斥）
- `Vₓ(r)`: 交换势（泡利不相容原理）

**优点**：
- 考虑了电子的反对称性
- 包含交换作用

**缺点**：
- 忽略了电子关联效应
- 计算成本高（O(N⁴)）
- 对某些体系精度不够

#### 1.1.4 密度泛函理论的诞生

**Hohenberg-Kohn定理**（1964）

**定理1（存在性定理）**：
> 体系的基态性质由电子密度 n(r) 唯一确定

这意味着：能量是电子密度的泛函 E = E[n(r)]

**定理2（变分原理）**：
> 对于给定的外势，基态能量对应最小能量

```
E₀ = min E[n(r)]
     n(r)
```

**革命性意义**：
- 将3N维问题（波函数）简化为3维问题（电子密度）
- 为实用计算方法奠定理论基础

**Kohn-Sham方程**（1965）

Kohn和Sham提出了实用的计算方案：

```
[-½∇² + Vₑₓₜ(r) + VH(r) + Vₓc(r)]ψᵢ(r) = εᵢψᵢ(r)
```

电子密度：
```
n(r) = Σᵢ |ψᵢ(r)|²
```

总能量：
```
E[n] = Tₛ[n] + ∫Vₑₓₜ(r)n(r)dr + EH[n] + Eₓc[n]
```

其中：
- `Tₛ[n]`: 无相互作用电子的动能
- `EH[n]`: Hartree能（电子-电子经典库仑能）
- `Eₓc[n]`: 交换关联能（包含所有量子多体效应）

**诺贝尔奖** 🏆：Walter Kohn因发展密度泛函理论获得1998年诺贝尔化学奖

### 1.2 DFT的发展里程碑

```
1926: Schrödinger方程
  ↓
1927: Born-Oppenheimer近似
  ↓
1930s: Hartree-Fock方法
  ↓
1964: Hohenberg-Kohn定理
  ↓
1965: Kohn-Sham方程
  ↓
1980s: LDA/GGA泛函发展
  ↓
1990s: 杂化泛函（B3LYP等）
  ↓
2000s: 范德华修正
  ↓
2010s: 机器学习势函数
  ↓
2020s: 神经网络泛函
```

---

## 2. 密度泛函理论（DFT）详解

### 2.1 从波函数到电子密度

#### 2.1.1 波函数方法的困境

**N电子体系的波函数**：
```
Ψ(r₁, r₂, ..., rₙ)
```

- **自由度**：3N个空间坐标
- **内存需求**：假设每个坐标用100个网格点离散化
  - 2个电子：100⁶ = 10¹² 个点
  - 10个电子：100³⁰ 个点（天文数字！）

**结论**：直接求解波函数在实际体系中不可行

#### 2.1.2 电子密度的优势

**电子密度**：
```
n(r) = N∫|Ψ(r, r₂, ..., rₙ)|² dr₂...drₙ
```

- **自由度**：仅3个空间坐标（与电子数无关！）
- **物理意义**：r点附近单位体积内找到电子的概率
- **归一化**：∫n(r)dr = N（总电子数）

**示例**：氢原子

波函数（1s轨道）：
```
ψ₁ₛ(r) = (1/√π) (1/a₀)³/² exp(-r/a₀)
```

电子密度：
```
n(r) = |ψ₁ₛ(r)|² = (1/π) (1/a₀)³ exp(-2r/a₀)
```

### 2.2 Kohn-Sham自洽循环

DFT计算的核心是自洽求解Kohn-Sham方程：

```
步骤1: 初始化
  ├─ 输入：原子位置、初始电子密度猜测
  └─ 设置：k点、截断能等参数

步骤2: 自洽迭代
  │
  ├─ 计算有效势：
  │   Vₑff(r) = Vₑₓₜ(r) + VH(r) + Vₓc(r)
  │
  ├─ 求解Kohn-Sham方程：
  │   [-½∇² + Vₑff(r)]ψᵢ = εᵢψᵢ
  │
  ├─ 更新电子密度：
  │   nₙₑw(r) = Σᵢ |ψᵢ(r)|²
  │
  ├─ 混合密度（提高收敛）：
  │   n(r) = α·nₙₑw + (1-α)·nₒₗd
  │
  └─ 检查收敛：
      |Eₙₑw - Eₒₗd| < tolerance ?

步骤3: 收敛后
  ├─ 计算总能量
  ├─ 计算原子受力
  └─ 输出其他性质
```

**收敛判据**：
- 能量变化：ΔE < 10⁻⁴ eV
- 电子密度变化：Δn < 10⁻⁶ e/Å³
- 力的大小：F < 0.01 eV/Å

### 2.3 交换关联泛函

交换关联能 Eₓc 是DFT唯一的未知项，需要近似处理。

#### 2.3.1 局域密度近似（LDA）

**假设**：每一点的交换关联能只依赖于该点的电子密度

```
Eₓc^LDA[n] = ∫ n(r)εₓc(n(r))dr
```

其中εₓc(n)是均匀电子气的单位体积交换关联能。

**优点**：
- 简单、稳定
- 对金属和共价键体系效果好

**缺点**：
- 高估结合能
- 低估晶格常数
- 带隙通常偏小

#### 2.3.2 广义梯度近似（GGA）

**改进**：不仅考虑密度，还考虑密度梯度

```
Eₓc^GGA[n] = ∫ f(n(r), ∇n(r))dr
```

**常用GGA泛函**：
- **PBE**：适用于固体和表面
- **RPBE**：改进了吸附能计算
- **PW91**：早期广泛使用的GGA

**优点**：
- 改善了键长和能量
- 适用范围更广

**缺点**：
- 仍低估带隙
- 无法描述范德华力

#### 2.3.3 杂化泛函

**思想**：混合部分精确交换（来自Hartree-Fock）

```
Eₓc^hybrid = aEₓ^HF + (1-a)Eₓ^DFT + Ec^DFT
```

**常用杂化泛函**：
- **B3LYP**：化学中最流行
- **HSE06**：筛选库仑相互作用，适合固体
- **PBE0**：25%精确交换

**优点**：
- 显著改善带隙预测
- 更准确的能量

**缺点**：
- 计算成本高（约4-10倍）
- 可能影响金属体系

#### 2.3.4 范德华修正

标准DFT无法描述长程范德华相互作用，需要额外修正。

**方法**：
- **DFT-D3**：经验色散修正
- **vdW-DF**：非局域相关泛函
- **TS-vdW**：Tkatchenko-Scheffler方案

```
E_total = E_DFT + E_disp

E_disp = -½ ΣᵢΣⱼ (C₆ⁱʲ/r₆ⁱʲ) f_damp(rᵢⱼ)
```

### 2.4 泛函选择指南

| 体系类型 | 推荐泛函 | 说明 |
|---------|---------|------|
| 金属 | PBE, LDA | 快速，精度足够 |
| 半导体/绝缘体 | HSE06, PBE0 | 改善带隙 |
| 分子 | B3LYP, PBE | B3LYP更准确 |
| 表面吸附 | RPBE, PBE | RPBE改善吸附能 |
| 层状材料 | PBE+D3 | 必须包含vdW |
| 有机晶体 | PBE+D3, vdW-DF | 色散相互作用重要 |
| 强关联体系 | DFT+U, HSE06 | 标准DFT失效 |

---

## 3. 常用的原子建模环境软件

### 3.1 ASE (Atomic Simulation Environment)

#### 3.1.1 简介

**官网**：https://wiki.fysik.dtu.dk/ase/

**特点**：
- Python编写，易于使用和扩展
- 支持多种计算软件（VASP, GPAW, Quantum ESPRESSO等）
- 丰富的结构操作和分析工具
- 活跃的开发和社区

#### 3.1.2 核心功能

**1. 结构构建**
```python
from ase import Atoms
from ase.build import bulk, molecule, surface

# 创建晶体
si = bulk('Si', 'diamond', a=5.43)

# 创建分子
h2o = molecule('H2O')

# 创建表面
slab = surface('Au', (1,1,1), layers=4)
```

**2. 结构操作**
```python
# 超胞
si_supercell = si * (2, 2, 2)

# 添加真空层
slab.center(vacuum=10, axis=2)

# 结构优化
from ase.optimize import BFGS
opt = BFGS(atoms)
opt.run(fmax=0.01)
```

**3. 文件I/O**
```python
from ase.io import read, write

# 读取结构
atoms = read('structure.cif')
atoms = read('POSCAR')  # VASP格式

# 写入结构
write('output.xyz', atoms)
write('structure.cif', atoms)
```

**4. 可视化**
```python
from ase.visualize import view

# 图形界面查看
view(atoms)

# 保存图片
write('structure.png', atoms, rotation='10x,10y,10z')
```

#### 3.1.3 计算器接口

ASE提供统一的计算器接口连接不同软件：

```python
from ase.calculators.vasp import Vasp
from ase.calculators.gpaw import GPAW

# VASP计算器
calc_vasp = Vasp(xc='PBE', encut=400, kpts=(4,4,4))

# GPAW计算器
calc_gpaw = GPAW(mode='pw', xc='PBE', kpts=(4,4,4))

atoms.calc = calc_vasp
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### 3.2 Pymatgen (Python Materials Genomics)

#### 3.2.1 简介

**官网**：https://pymatgen.org/

**特点**：
- Materials Project官方库
- 强大的晶体学分析
- 丰富的材料数据接口
- 相图计算和分析

#### 3.2.2 核心功能

**1. 结构分析**
```python
from pymatgen.core import Structure, Lattice

# 创建结构
lattice = Lattice.cubic(4.2)
structure = Structure(lattice, ['Cs', 'Cl'],
                     [[0,0,0], [0.5,0.5,0.5]])

# 结构分析
print(structure.density)
print(structure.volume)
print(structure.get_space_group_info())
```

**2. Materials Project API**
```python
from pymatgen.ext.matproj import MPRester

# 从Materials Project下载数据
with MPRester("YOUR_API_KEY") as mpr:
    structure = mpr.get_structure_by_material_id("mp-149")
    data = mpr.get_data("Fe2O3")
```

**3. 相图计算**
```python
from pymatgen.analysis.phase_diagram import PhaseDiagram

# 构建相图
pd = PhaseDiagram(entries)
pd.get_decomposition_energy(entry)
```

**4. 电子结构分析**
```python
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.plotter import DosPlotter

# 读取VASP输出
vasprun = Vasprun("vasprun.xml")
dos = vasprun.complete_dos

# 绘制态密度
plotter = DosPlotter()
plotter.add_dos("Total DOS", dos)
plotter.show()
```

### 3.3 ASE vs Pymatgen

| 特性 | ASE | Pymatgen |
|------|-----|----------|
| **定位** | 原子模拟环境 | 材料基因组学 |
| **优势** | 计算接口、MD模拟 | 晶体学、相图分析 |
| **学习曲线** | 较平缓 | 较陡峭 |
| **可视化** | 内置简单工具 | 需外部库 |
| **数据库** | 有限 | Materials Project |
| **推荐用途** | 日常计算、快速建模 | 高通量筛选、数据分析 |

**实践建议**：两者结合使用
```python
# ASE建模 + Pymatgen分析
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

# ASE创建结构
ase_atoms = bulk('Si', 'diamond', a=5.43)

# 转换为Pymatgen
adaptor = AseAtomsAdaptor()
pmg_structure = adaptor.get_structure(ase_atoms)

# 使用Pymatgen分析
print(pmg_structure.get_space_group_info())
```

---

## 4. 常用的第一性原理计算软件

### 4.1 VASP (Vienna Ab initio Simulation Package)

#### 4.1.1 简介

**官网**：https://www.vasp.at/

**特点**：
- 业界标准，使用最广泛
- 基于平面波赝势方法
- 高度优化，计算效率高
- 支持各种高级功能（HSE、GW、DMFT等）

**许可**：商业软件，需购买许可证

#### 4.1.2 核心文件

**输入文件**：
- `POSCAR`: 结构文件
- `INCAR`: 计算参数
- `POTCAR`: 赝势文件
- `KPOINTS`: k点设置

**输出文件**：
- `OUTCAR`: 详细输出
- `vasprun.xml`: XML格式结果
- `CONTCAR`: 优化后结构
- `CHGCAR`: 电荷密度

#### 4.1.3 典型INCAR示例

```bash
# 基本设置
SYSTEM = Silicon bulk

# 电子结构
ENCUT = 400        # 截断能 (eV)
PREC = Accurate    # 精度
EDIFF = 1E-6       # 电子步收敛 (eV)

# 交换关联泛函
GGA = PE           # PBE泛函

# 结构优化
IBRION = 2         # 离子步算法(CG)
NSW = 100          # 最大离子步数
EDIFFG = -0.01     # 力收敛标准 (eV/Å)

# 并行
NCORE = 4          # 每个band的核数
```

#### 4.1.4 优势与局限

**优势**：
- 大量文献支持
- 成熟稳定
- 社区庞大
- 功能全面

**局限**：
- 商业软件，费用昂贵
- 闭源，无法自定义
- 学习曲线陡峭

### 4.2 GPAW

#### 4.2.1 简介

**官网**：https://wiki.fysik.dtu.dk/gpaw/

**特点**：
- 基于Python和C的混合编程
- 支持多种模式（平面波、实空间网格、原子轨道）
- 开源免费
- 与ASE深度集成
- 适合教学和方法开发

**许可**：GPL开源许可

#### 4.2.2 计算模式

**1. 平面波模式 (PW)**
```python
from gpaw import GPAW, PW

calc = GPAW(mode=PW(400),      # 截断能400 eV
            xc='PBE',
            kpts=(4,4,4),
            txt='output.txt')
```

**2. 实空间网格模式 (FD)**
```python
calc = GPAW(mode='fd',         # 有限差分
            h=0.2,             # 网格间距 0.2 Å
            xc='PBE',
            kpts=(4,4,4))
```

**3. 局域原子轨道模式 (LCAO)**
```python
calc = GPAW(mode='lcao',       # 线性组合原子轨道
            basis='dzp',       # 双zeta极化基组
            xc='PBE',
            kpts=(4,4,4))
```

#### 4.2.3 完整计算示例

```python
from ase.build import bulk
from gpaw import GPAW, PW
from ase.optimize import BFGS

# 创建结构
atoms = bulk('Si', 'diamond', a=5.43)

# 设置计算器
calc = GPAW(mode=PW(400),
            xc='PBE',
            kpts=(8,8,8),
            txt='si_relax.txt',
            symmetry={'point_group': True})

atoms.calc = calc

# 结构优化
opt = BFGS(atoms, trajectory='si.traj')
opt.run(fmax=0.01)

# 获取结果
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print(f'Total energy: {energy:.3f} eV')
print(f'Forces:\n{forces}')
```

#### 4.2.4 优势与局限

**优势**：
- 完全免费开源
- 易于扩展和自定义
- Python接口友好
- 适合学习和原型开发
- 文档详细

**局限**：
- 计算效率略低于VASP
- 社区相对较小
- 某些高级功能不如VASP完善

### 4.3 其他流行软件

#### Quantum ESPRESSO
- 开源、平面波
- 功能强大
- 适合固体物理

#### CP2K
- 混合高斯基组/平面波
- 适合大体系和MD
- 开源

#### CASTEP
- 商业软件
- 用户界面友好
- 集成Material Studio

#### Gaussian
- 量子化学标准
- 适合分子体系
- 商业软件

### 4.4 软件选择建议

| 应用场景 | 推荐软件 | 理由 |
|---------|---------|------|
| 学习入门 | GPAW | 开源、Python友好 |
| 科研发表 | VASP | 认可度高 |
| 大体系MD | CP2K | 效率高 |
| 分子计算 | Gaussian | 专业化 |
| 方法开发 | GPAW, QE | 开源可定制 |
| 工业应用 | VASP, CASTEP | 稳定可靠 |

---

## 5. 学习路线图

```
Level 1: 基础入门
├─ Linux基础操作
├─ Python编程
└─ 量子力学基本概念

Level 2: 理论学习
├─ Schrödinger方程
├─ DFT理论
└─ 交换关联泛函

Level 3: 软件使用
├─ ASE建模
├─ GPAW基础计算
└─ 结果分析

Level 4: 高级应用
├─ 收敛性测试
├─ 不同泛函比较
└─ 性质计算

Level 5: 研究实践
├─ 实际材料体系
├─ 方法验证
└─ 文献复现
```

---

## 6. 推荐资源

### 教材
1. **DFT基础**：
   - Sholl & Steckel: "Density Functional Theory: A Practical Introduction"
   - Martin: "Electronic Structure: Basic Theory and Practical Methods"

2. **ASE教程**：
   - https://wiki.fysik.dtu.dk/ase/tutorials/tutorials.html

3. **GPAW教程**：
   - https://wiki.fysik.dtu.dk/gpaw/tutorialsexercises/tutorialsexercises.html

### 在线课程
- Coursera: Introduction to Computational Materials Science
- edX: Atomistic Modeling of Materials

### 实践项目
- 计算常见材料的晶格常数
- 复现经典文献的DFT结果
- 构建材料性质数据库

---

**下一步**：让我们进入实践环节，从Linux基础开始！ 🚀
