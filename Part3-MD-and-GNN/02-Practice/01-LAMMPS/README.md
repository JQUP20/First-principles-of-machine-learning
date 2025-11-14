# LAMMPS分子动力学软件
# LAMMPS Molecular Dynamics Software

本目录包含LAMMPS分子动力学软件的教程和示例脚本。

## 学习目标

1. 掌握LAMMPS的安装和基本使用
2. 学会编写LAMMPS输入脚本
3. 理解不同系综(NVE, NVT, NPT)的设置
4. 使用LAMMPS与神经网络势函数结合
5. 分析MD模拟结果

## 内容概览

### 1. LAMMPS基础
- LAMMPS安装
- 输入脚本语法
- 势函数设置
- 边界条件和系综

### 2. 示例脚本

#### `examples/01_lennard_jones.lmp`
- Lennard-Jones势基础示例
- NVE系综
- 能量守恒验证

#### `examples/02_metal_eam.lmp`
- 金属系统(Cu)
- EAM势函数
- NVT系综

#### `examples/03_water_tip3p.lmp`
- 水分子模拟
- TIP3P势函数
- NPT系综

#### `scripts/run_deepmd.lmp`
- 使用DeePMD神经网络势
- 与第一性原理精度对比

## LAMMPS安装

### 方法1：预编译二进制文件

```bash
# Ubuntu/Debian
sudo apt-get install lammps

# macOS (Homebrew)
brew install lammps
```

### 方法2：从源码编译

```bash
# 下载
git clone -b stable https://github.com/lammps/lammps.git
cd lammps

# 编译(MPI并行版本)
cd src
make yes-molecule yes-kspace yes-rigid yes-user-deepmd
make mpi -j4

# 测试
./lmp_mpi < in.example
```

### 方法3：Conda安装

```bash
conda install -c conda-forge lammps
```

## LAMMPS脚本基本结构

```lammps
# 1. 初始化
units metal              # 单位系统
atom_style atomic
dimension 3
boundary p p p           # 周期性边界条件

# 2. 创建体系
lattice fcc 3.615
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 box

# 3. 定义势函数
pair_style eam/alloy
pair_coeff * * Cu_u3.eam Cu
mass 1 63.546

# 4. 运行设置
velocity all create 300.0 12345
fix 1 all nvt temp 300.0 300.0 0.1
timestep 0.001

# 5. 输出
thermo 100
thermo_style custom step temp pe ke etotal press vol
dump 1 all custom 1000 dump.lammpstrj id type x y z

# 6. 运行
run 10000
```

## 常用命令详解

### 势函数类型

```lammps
# Lennard-Jones
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0

# EAM (嵌入原子势)
pair_style eam/alloy
pair_coeff * * Cu_u3.eam Cu

# Tersoff (共价材料)
pair_style tersoff
pair_coeff * * SiC.tersoff Si C

# 神经网络势(DeePMD)
pair_style deepmd model.pb
pair_coeff * *
```

### 系综设置

```lammps
# NVE (微正则系综)
fix 1 all nve

# NVT (正则系综 - Nosé-Hoover)
fix 1 all nvt temp 300.0 300.0 0.1

# NPT (等温等压系综)
fix 1 all npt temp 300.0 300.0 0.1 iso 1.0 1.0 1.0

# Langevin恒温器
fix 1 all langevin 300.0 300.0 0.1 12345
fix 2 all nve
```

### 输出控制

```lammps
# 热力学输出频率
thermo 100

# 自定义输出
thermo_style custom step temp pe ke etotal press vol density

# 轨迹文件
dump 1 all custom 1000 dump.lammpstrj id type x y z vx vy vz fx fy fz

# 重启文件
restart 10000 restart.*.lmp
```

## 使用DeePMD势函数

### 准备工作

1. 训练DeePMD模型(见`../02-DeePMD`)
2. 冻结模型得到`model.pb`
3. 编译支持DeePMD的LAMMPS

### LAMMPS脚本示例

```lammps
# DeePMD势函数示例
units metal
atom_style atomic
boundary p p p

# 读取结构
read_data data.lammps

# DeePMD势
pair_style deepmd model.pb
pair_coeff * *

# NVT模拟
velocity all create 300.0 12345
fix 1 all nvt temp 300.0 300.0 0.1

# 输出
thermo 100
dump 1 all custom 1000 traj.lammpstrj id type x y z fx fy fz
dump_modify 1 sort id

# 运行
timestep 0.001
run 100000  # 100 ps
```

## 结果分析

### 使用Python分析轨迹

```python
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# 读取LAMMPS轨迹
atoms_list = read('dump.lammpstrj', index=':', format='lammps-dump-text')

# 计算径向分布函数(RDF)
from ase.geometry.analysis import Analysis
ana = Analysis(atoms_list)
rdf = ana.get_rdf(rmax=10.0, nbins=100)

# 绘图
plt.figure()
plt.plot(rdf[0], rdf[1])
plt.xlabel('Distance (Å)')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function')
plt.savefig('rdf.png', dpi=300)
```

### 计算扩散系数

```python
# 均方位移(MSD)
from ase.md.analysis import DiffusionCoefficient

dc = DiffusionCoefficient(atoms_list, timestep=1.0)  # fs
dc.calculate()

# 扩散系数
D = dc.get_diffusion_coefficient()  # Å²/fs
print(f"Diffusion coefficient: {D * 1e4:.4f} cm²/s")
```

## 练习题

### 练习1：基础模拟
运行Lennard-Jones气体的NVE模拟，验证能量守恒。

### 练习2：熔化模拟
模拟Cu纳米粒子的熔化过程，确定熔点。

### 练习3：扩散计算
计算液态Cu中的自扩散系数。

### 练习4：DeePMD对比
使用EAM势和DeePMD势分别模拟同一体系，对比结果。

## 常见问题

### Q1: LAMMPS运行速度慢？

**优化方法**:
- 使用MPI并行：`mpirun -np 4 lmp_mpi < in.lammps`
- 启用GPU加速：编译GPU package
- 优化邻居列表：`neighbor 1.0 bin; neigh_modify delay 5 every 1`

### Q2: 能量不守恒？

**检查**:
- 时间步长是否太大？
- 是否正确使用NVE？
- 截断半径是否合适？

### Q3: 轨迹文件太大？

**解决**:
- 增大dump间隔
- 只输出需要的信息
- 压缩：`dump_modify 1 append yes` 或使用netcdf格式

## 扩展资源

- [LAMMPS官方文档](https://docs.lammps.org/)
- [LAMMPS教程集合](https://lammpstutorials.github.io/)
- [Material Studio + LAMMPS](https://www.3dsbiovia.com/products/collaborative-science/biovia-materials-studio/)

---

**返回**: [Part 3主页](../../README.md)
