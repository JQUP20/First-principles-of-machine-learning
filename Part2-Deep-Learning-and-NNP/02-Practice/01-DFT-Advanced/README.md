# 第一性原理计算进阶
# Advanced First-Principles Calculations

本目录包含第一性原理计算的进阶内容，重点关注BaTiO₃的电子结构计算和AIMD数据集构建。

## 学习目标

1. 掌握能带结构计算
2. 理解态密度(DOS)和投影态密度(PDOS)
3. 学习电荷密度分析
4. 掌握AIMD（从头算分子动力学）数据集构建
5. 使用t-SNE进行数据集可视化

## 内容概览

### 1. BaTiO₃能带结构计算

**文件**: `examples/01_batio3_bandstructure.py`

学习内容：
- 构建钙钛矿结构
- 自洽场(SCF)计算
- 能带结构计算
- 绘制能带图

### 2. 态密度和投影态密度

**文件**: `examples/02_batio3_dos.py`

学习内容：
- DOS计算方法
- PDOS分解（按原子和轨道）
- 分析能带成分
- 可视化DOS

### 3. 电荷密度分析

**文件**: `examples/03_batio3_charge_density.py`

学习内容：
- 计算电荷密度
- 可视化电荷分布
- Bader电荷分析（可选）

### 4. AIMD数据集构建

**文件**: `examples/04_aimd_dataset.py`

学习内容：
- 设置AIMD参数
- 运行分子动力学
- 提取能量和力数据
- 保存为训练数据集

### 5. t-SNE数据可视化

**文件**: `examples/05_tsne_visualization.py`

学习内容：
- 使用t-SNE降维
- 可视化结构多样性
- 分析数据集覆盖范围

## 前置要求

### 软件环境

```bash
# GPAW（推荐）
conda install -c conda-forge gpaw

# 或使用ASE + 其他DFT代码
pip install ase

# 数据处理和可视化
pip install numpy matplotlib scipy scikit-learn
```

### 基础知识

- Part 1的ASE和GPAW基础
- 密度泛函理论基本概念
- Python科学计算库使用

## 快速开始

### 1. 能带结构计算

```bash
# 运行能带结构计算（需要较长时间）
python examples/01_batio3_bandstructure.py
```

输出：
- `batio3_scf.gpw`: 自洽场结果
- `batio3_bands.json`: 能带数据
- `batio3_bandstructure.png`: 能带图

### 2. 态密度计算

```bash
python examples/02_batio3_dos.py
```

输出：
- `batio3_dos.png`: DOS和PDOS图

### 3. AIMD数据生成

```bash
# 注意：这个计算非常耗时，建议使用计算集群
python examples/04_aimd_dataset.py
```

输出：
- `aimd_trajectory.traj`: 轨迹文件
- `aimd_dataset.npz`: 训练数据（坐标、能量、力）

## 使用VASP的注意事项

如果使用VASP而非GPAW：

### INCAR示例（能带计算）

```
# SCF计算
SYSTEM = BaTiO3
ENCUT = 500
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.05
LWAVE = .TRUE.
LCHARG = .TRUE.
```

### INCAR示例（AIMD）

```
# AIMD设置
IBRION = 0         # 分子动力学
NSW = 1000         # MD步数
POTIM = 1.0        # 时间步长(fs)
SMASS = 0          # NVE系综
TEBEG = 300        # 初始温度
```

### 使用Python脚本处理VASP输出

```python
from ase.io import read

# 读取VASP输出
atoms = read('OUTCAR')
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## 数据集构建最佳实践

### 1. 温度范围

```python
temperatures = [300, 600, 900, 1200]  # K
for T in temperatures:
    run_aimd(temperature=T, steps=500)
```

### 2. 初始结构多样性

- 平衡结构
- 畸变结构（应变、原子位移）
- 不同超胞尺寸
- 表面和缺陷结构

### 3. 采样策略

```python
# 每隔N步保存一个结构（去相关）
save_interval = 10

# 或使用能量波动标准
if abs(E - E_mean) > threshold:
    save_structure()
```

## 练习题

### 练习1：能带分析
1. 计算BaTiO₃的能带结构
2. 确定带隙类型（直接/间接）
3. 找出价带顶和导带底的k点位置

### 练习2：DOS分析
1. 计算总DOS和PDOS
2. 确定哪些轨道贡献价带顶
3. 分析Ti-3d轨道的晶体场分裂

### 练习3：AIMD数据集
1. 运行300K的AIMD模拟
2. 提取100个构型
3. 检查能量和力的分布

### 练习4：数据可视化
1. 使用t-SNE可视化AIMD轨迹
2. 与平衡结构比较
3. 识别结构簇

## 常见问题

### Q1: GPAW计算太慢怎么办？

**A**:
- 减小截断能：`mode=PW(300)`
- 减少k点：`kpts=(4, 4, 4)`
- 使用更小的超胞
- 使用计算集群并行计算

### Q2: AIMD需要运行多久？

**A**:
- 至少1-2 ps达到平衡
- 数据收集：5-10 ps
- 总时间：取决于系统大小和精度要求

### Q3: 如何判断AIMD已平衡？

**A**:
```python
import numpy as np
import matplotlib.pyplot as plt

# 绘制能量和温度随时间变化
energies = [atoms.get_potential_energy() for atoms in traj]
plt.plot(energies)
plt.xlabel('Step')
plt.ylabel('Energy (eV)')
plt.show()
```

### Q4: t-SNE参数如何选择？

**A**:
- `perplexity`: 5-50（取决于数据量）
- `n_iter`: 1000-5000
- 尝试不同参数观察聚类效果

## 扩展资源

### 在线教程
- [GPAW Tutorials](https://wiki.fysik.dtu.dk/gpaw/tutorials/tutorials.html)
- [ASE Tutorials](https://wiki.fysik.dtu.dk/ase/tutorials/tutorials.html)
- [VASP Wiki](https://www.vasp.at/wiki/)

### 相关论文
1. Cohen, R. E. (1992). Origin of ferroelectricity in perovskite oxides. *Nature*.
2. Eshet, H., et al. (2010). Ab initio quality neural-network potential for sodium. *Phys. Rev. B*.

### 数据库
- [Materials Project](https://materialsproject.org/)
- [NOMAD Repository](https://nomad-lab.eu/)
- [AFLOW](http://www.aflowlib.org/)

---

## 下一步

完成本章后，进入：
- [PyTorch深度学习库](../02-PyTorch-Basics/)
- [ResNet手写数字识别](../03-ResNet-MNIST/)

---

**返回**: [Part 2主页](../../README.md)
