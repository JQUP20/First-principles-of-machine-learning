# 晶体材料简介及其电子结构特点
# Introduction to Crystalline Materials and Electronic Structure

## 目录
- [晶体的基本概念](#晶体的基本概念)
- [布拉维格子](#布拉维格子)
- [空间群和对称性](#空间群和对称性)
- [晶体材料的电子结构](#晶体材料的电子结构)
- [态密度分析](#态密度分析)

---

## 晶体的基本概念

### 什么是晶体？

**定义**：原子、离子或分子在三维空间中周期性排列的固体。

**特征**：
- **长程有序性**：周期性重复的结构单元
- **各向异性**：不同方向物理性质可能不同
- **确定的熔点**：晶体结构破坏时对应明确的相变温度
- **解理性**：沿特定晶面易于破裂

**晶体 vs 非晶体**：

| 特性 | 晶体 | 非晶体（玻璃态） |
|-----|------|--------------|
| 原子排列 | 长程有序 | 短程有序 |
| 对称性 | 具有对称性 | 无对称性 |
| 衍射图样 | 尖锐衍射峰 | 弥散环 |
| 熔点 | 确定 | 软化温度范围 |
| 例子 | 金刚石、NaCl | 石英玻璃、非晶硅 |

### 晶胞 (Unit Cell)

**定义**：晶体结构的最小重复单元。

**原胞（Primitive Cell）**：
- 包含最少原子的晶胞
- 体积最小

**常规晶胞（Conventional Cell）**：
- 体现晶体对称性
- 可能包含多个原胞

**晶格参数（Lattice Parameters）**：
- 晶格常数：$a, b, c$（晶胞边长）
- 晶格角：$\alpha, \beta, \gamma$（晶胞夹角）

### 晶系 (Crystal System)

根据对称性，晶体分为7大晶系：

| 晶系 | 晶格参数关系 | 对称性 | 例子 |
|-----|------------|-------|------|
| 立方 (Cubic) | $a=b=c$, $\alpha=\beta=\gamma=90°$ | 最高 | NaCl, 金刚石 |
| 四方 (Tetragonal) | $a=b\neq c$, $\alpha=\beta=\gamma=90°$ | 4次旋转轴 | TiO₂, SnO₂ |
| 正交 (Orthorhombic) | $a\neq b\neq c$, $\alpha=\beta=\gamma=90°$ | 3个2次旋转轴 | α-S, 硫酸钡 |
| 六方 (Hexagonal) | $a=b\neq c$, $\alpha=\beta=90°, \gamma=120°$ | 6次旋转轴 | 石墨, ZnO |
| 三方 (Rhombohedral/Trigonal) | $a=b=c$, $\alpha=\beta=\gamma\neq 90°$ | 3次旋转轴 | 方解石, Bi |
| 单斜 (Monoclinic) | $a\neq b\neq c$, $\alpha=\gamma=90°\neq\beta$ | 1个2次旋转轴 | 石膏, β-S |
| 三斜 (Triclinic) | $a\neq b\neq c$, $\alpha\neq\beta\neq\gamma$ | 无对称性 | CuSO₄·5H₂O |

---

## 布拉维格子

### 定义

**布拉维格子（Bravais Lattice）**：通过平移操作可以互相重合的所有点的集合。

**数学表达**：
$$
\mathbf{R} = n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3
$$

其中：
- $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$：基矢
- $n_1, n_2, n_3$：整数

### 14种布拉维格子

通过组合7个晶系和晶胞中心类型，共有14种布拉维格子：

**晶胞中心类型**：
- **P (Primitive)**：简单格子，只有顶点有格点
- **I (Body-centered)**：体心格子，体心有额外格点
- **F (Face-centered)**：面心格子，6个面心有额外格点
- **C (Base-centered)**：底心格子，一对相对面心有额外格点

**立方晶系**（3种）：
1. **简单立方 (SC)**: Primitive
   - 配位数：6
   - 例子：Po（钋）

2. **体心立方 (BCC)**: Body-centered
   - 配位数：8
   - 例子：Fe, Cr, W

3. **面心立方 (FCC)**: Face-centered
   - 配位数：12
   - 例子：Al, Cu, Au, Ag

### 倒格子 (Reciprocal Lattice)

**定义**：与实空间布拉维格子对应的倒空间格子。

**倒格矢**：
$$
\mathbf{b}_1 = 2\pi \frac{\mathbf{a}_2 \times \mathbf{a}_3}{\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)}
$$

$$
\mathbf{b}_2 = 2\pi \frac{\mathbf{a}_3 \times \mathbf{a}_1}{\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)}
$$

$$
\mathbf{b}_3 = 2\pi \frac{\mathbf{a}_1 \times \mathbf{a}_2}{\mathbf{a}_1 \cdot (\mathbf{a}_2 \times \mathbf{a}_3)}
$$

**性质**：
$$
\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi \delta_{ij}
$$

### 布里渊区 (Brillouin Zone)

**第一布里渊区**：倒格子原点的Wigner-Seitz原胞。

**物理意义**：
- 包含所有独立的 $\mathbf{k}$ 点
- 电子能带在第一布里渊区内定义

**高对称点**（以FCC为例）：
- $\Gamma$：(0, 0, 0)
- $X$：(0, 1, 0) $\times \frac{2\pi}{a}$
- $L$：(1/2, 1/2, 1/2) $\times \frac{2\pi}{a}$
- $W$：(1/2, 1, 0) $\times \frac{2\pi}{a}$

---

## 空间群和对称性

### 点群 (Point Group)

**定义**：保持至少一点不动的对称操作集合。

**对称操作**：
1. **恒等操作 (E)**：不做任何变化
2. **旋转 (Cₙ)**：绕轴旋转 $\frac{360°}{n}$
3. **反演 (i)**：通过中心点反演
4. **镜面反射 (σ)**：通过平面反射
5. **旋转反演 (Sₙ)**：旋转后反演

**晶体学限制**：
只允许1, 2, 3, 4, 6次旋转对称（5次、7次及更高不允许）

**32个晶体学点群**：
根据对称性分为32种点群。

### 空间群 (Space Group)

**定义**：包含平移对称性的对称操作群。

**组成**：
- 点群操作（旋转、反射、反演）
- 平移操作
- 螺旋轴（旋转+平移）
- 滑移面（反射+平移）

**230个空间群**：
所有可能的晶体对称性分类。

**国际符号**（Hermann-Mauguin notation）：
例如：$Fm\overline{3}m$（FCC的空间群）
- F：面心
- m：镜面
- $\overline{3}$：3次旋转反演
- m：镜面

### 对称性在第一性原理计算中的应用

**1. 减少计算量**：
- 只需计算布里渊区的不可约部分
- 使用对称性生成等价原子位置

**2. 选择规则**：
- 确定哪些跃迁是允许的
- 简化矩阵元计算

**3. 能带标记**：
- 使用不可约表示标记能带

---

## 晶体材料的电子结构

### Bloch定理

在周期性势场中，电子波函数具有Bloch形式：

$$
\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}} u_{n\mathbf{k}}(\mathbf{r})
$$

其中：
- $\mathbf{k}$：波矢（在第一布里渊区内）
- $u_{n\mathbf{k}}(\mathbf{r})$：具有晶格周期性的函数
- $n$：能带指标

### 能带结构 (Band Structure)

**定义**：电子能量随波矢 $\mathbf{k}$ 的变化关系 $E_n(\mathbf{k})$。

**计算方法**：
1. 解Kohn-Sham方程：
$$
\left[-\frac{\hbar^2}{2m}\nabla^2 + V_{\text{eff}}(\mathbf{r})\right]\psi_{n\mathbf{k}} = E_{n\mathbf{k}}\psi_{n\mathbf{k}}
$$

2. 在高对称路径上计算能量

**高对称路径示例**（FCC）：
$$
\Gamma \rightarrow X \rightarrow W \rightarrow K \rightarrow \Gamma \rightarrow L
$$

### 能带分类

**1. 导体 (Conductor)**：
- 费米能级穿过能带
- 零带隙
- 例子：Cu, Al, Au

**2. 半导体 (Semiconductor)**：
- 小带隙（<3 eV）
- 直接带隙：导带底和价带顶在同一 $\mathbf{k}$ 点（如GaAs）
- 间接带隙：导带底和价带顶在不同 $\mathbf{k}$ 点（如Si）

**3. 绝缘体 (Insulator)**：
- 大带隙（>3 eV）
- 费米能级在带隙中
- 例子：金刚石(5.5 eV), NaCl(8.5 eV)

### BaTiO₃ 示例

**钛酸钡（BaTiO₃）**是典型的铁电材料。

**晶体结构**（室温以下）：
- 四方钙钛矿结构
- 空间群：P4mm
- Ti原子偏离中心位置 → 自发极化

**电子结构特点**：
1. **带隙**：~3.2 eV（间接带隙）
2. **价带顶**：主要是 O-2p 轨道
3. **导带底**：主要是 Ti-3d 轨道
4. **Born有效电荷**：比名义电荷大（动态电荷）

**能带计算流程**（GPAW/VASP）：

```python
from ase.build import bulk
from gpaw import GPAW, PW

# 构建BaTiO3结构
atoms = bulk('BaTiO3', 'perovskite', a=4.0)

# 自洽计算
calc = GPAW(mode=PW(500),
            kpts={'size': (8, 8, 8), 'gamma': True},
            xc='PBE')
atoms.calc = calc
atoms.get_potential_energy()

# 能带计算
calc.write('BaTiO3.gpw')

# 读取并计算能带
from gpaw import GPAW
calc = GPAW('BaTiO3.gpw')
calc.get_fermi_level()
```

---

## 态密度分析

### 态密度 (Density of States, DOS)

**定义**：单位能量间隔内的电子态数量。

$$
\text{DOS}(E) = \sum_n \int_{\text{BZ}} \delta(E - E_n(\mathbf{k})) \frac{d\mathbf{k}}{(2\pi)^3}
$$

**物理意义**：
- 描述能量分布
- 与光学、磁性等性质相关

**计算方法**：
1. 在布里渊区密集 $\mathbf{k}$ 点网格上计算能量
2. 使用四面体方法或高斯展宽

### 投影态密度 (Projected DOS, PDOS)

**定义**：特定原子或轨道对态密度的贡献。

$$
\text{PDOS}_{\mu}(E) = \sum_{n\mathbf{k}} |\langle \phi_{\mu} | \psi_{n\mathbf{k}} \rangle|^2 \delta(E - E_{n\mathbf{k}})
$$

其中 $\phi_{\mu}$ 是原子轨道。

**用途**：
- 识别能带的轨道成分
- 分析化学键
- 理解电子转移

### BaTiO₃的DOS分析

**典型特征**：

1. **价带区（-6 to 0 eV）**：
   - 主要来自 O-2p 轨道
   - 部分 Ti-3d 和 Ba-5p 贡献

2. **导带区（3 to 6 eV）**：
   - 主要是 Ti-3d 轨道
   - 分裂为 t₂g 和 eₓ 态（晶体场分裂）

3. **带隙**：
   - ~3.2 eV
   - 对应紫外光吸收

**PDOS示例代码**：

```python
from gpaw import GPAW

# 读取计算结果
calc = GPAW('BaTiO3.gpw')

# 计算DOS
from gpaw.dos import DOS
dos = DOS(calc, width=0.1)
energies = dos.get_energies()
weights = dos.get_dos()

# 计算PDOS
from gpaw.pdos import PDOS
pdos = PDOS(calc, width=0.1)

# 绘图
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(energies, weights, label='Total DOS')

# Ti-d PDOS
Ti_d = pdos.get_atom_dos(atom_index=Ti_index, l=2)
plt.plot(energies, Ti_d, label='Ti-d')

# O-p PDOS
O_p = pdos.get_atom_dos(atom_index=O_index, l=1)
plt.plot(energies, O_p, label='O-p')

plt.xlabel('Energy (eV)')
plt.ylabel('DOS (states/eV)')
plt.axvline(calc.get_fermi_level(), ls='--', color='k', label='Fermi')
plt.legend()
plt.show()
```

### 电荷密度分析

**总电荷密度**：
$$
\rho(\mathbf{r}) = \sum_{n\mathbf{k}} f_{n\mathbf{k}} |\psi_{n\mathbf{k}}(\mathbf{r})|^2
$$

其中 $f_{n\mathbf{k}}$ 是费米-狄拉克占据数。

**差分电荷密度**：
$$
\Delta\rho = \rho_{\text{molecule}} - \rho_{\text{atoms}}
$$

用于分析成键和电荷转移。

**Bader电荷分析**：
- 基于电荷密度梯度零点划分原子
- 给出每个原子的有效电荷

---

## 小结

1. **晶体结构**：周期性排列，由布拉维格子和基元描述
2. **对称性**：230个空间群分类所有晶体
3. **电子结构**：能带和态密度描述电子性质
4. **DFT计算**：可精确预测晶体电子结构

---

## 实验：BaTiO₃的完整计算

在实操部分，我们将：
1. 构建BaTiO₃结构
2. 优化晶格参数
3. 计算能带结构
4. 计算态密度和PDOS
5. 分析电荷密度
6. 使用AIMD生成训练数据

---

## 扩展阅读

1. Ashcroft, N. W., & Mermin, N. D. (1976). *Solid State Physics*. Holt-Saunders.
2. Kittel, C. (2004). *Introduction to Solid State Physics*. Wiley.
3. Cohen, R. E. (1992). Origin of ferroelectricity in perovskite oxides. *Nature*, 358(6382), 136-138.
4. [Bilbao Crystallographic Server](https://www.cryst.ehu.es/) - 空间群数据库

---

**上一节**: [神经网络势函数](./03-Neural-Network-Potentials.md)
**返回**: [课程主页](../../../README.md)
