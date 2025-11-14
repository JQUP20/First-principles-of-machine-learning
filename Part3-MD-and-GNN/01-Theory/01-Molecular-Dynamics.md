# 分子动力学模拟
# Molecular Dynamics Simulation

## 目录
- [第一性原理分子动力学——从微观到宏观](#第一性原理分子动力学从微观到宏观)
- [分子动力学的基本原理——牛顿运动方程](#分子动力学的基本原理牛顿运动方程)
- [牛顿运动方程的几种数值求解方法](#牛顿运动方程的几种数值求解方法)
- [统计力学系综概念](#统计力学系综概念)
- [LAMMPS分子动力学软件介绍](#lammps分子动力学软件介绍)
- [神经网络势函数和LAMMPS的结合](#神经网络势函数和lammps的结合)

---

## 第一性原理分子动力学——从微观到宏观

### 多尺度模拟方法

在材料科学中，我们需要跨越多个时间和空间尺度来理解材料的性质：

```
电子尺度 (Å, fs)         原子尺度 (nm, ps)        介观尺度 (μm, ns)       宏观尺度 (mm, s)
        ↓                        ↓                        ↓                      ↓
   DFT/量子化学    →    分子动力学(MD)    →    粗粒化模拟/相场    →    有限元分析
   (~100原子)           (~10^6原子)            (~10^9原子)           (连续介质)
```

### 为什么需要机器学习势函数？

**第一性原理分子动力学(AIMD)**：
- **优点**：准确描述化学键的形成和断裂
- **缺点**：计算成本极高，限制了系统大小(<1000原子)和模拟时间(<10 ps)

**经验势函数分子动力学**：
- **优点**：快速，可模拟百万原子和微秒时间尺度
- **缺点**：准确性差，难以描述复杂化学过程

**神经网络势函数**：
- ✅ **准确性**：接近DFT水平
- ✅ **速度**：比AIMD快10^4-10^6倍
- ✅ **可扩展性**：可模拟10^4-10^6原子
- ✅ **时间尺度**：纳秒到微秒

### 从微观到宏观的桥梁

**量子力学** → **统计力学** → **热力学**

$$
\text{薛定谔方程} \rightarrow \text{分子动力学轨迹} \rightarrow \text{系综平均} \rightarrow \text{宏观性质}
$$

**示例**：计算熔点
1. 使用神经网络势函数进行MD模拟
2. 在不同温度下模拟固相和液相
3. 通过自由能计算确定相变温度

---

## 分子动力学的基本原理——牛顿运动方程

### 牛顿第二定律

分子动力学的核心是求解牛顿运动方程：

$$
m_i \frac{d^2\mathbf{r}_i}{dt^2} = \mathbf{F}_i = -\nabla_{\mathbf{r}_i} U(\mathbf{r}_1, \mathbf{r}_2, ..., \mathbf{r}_N)
$$

其中：
- $m_i$：原子 $i$ 的质量
- $\mathbf{r}_i$：原子 $i$ 的位置
- $\mathbf{F}_i$：作用在原子 $i$ 上的力
- $U$：系统势能

### Born-Oppenheimer近似

**假设**：电子运动比原子核快得多，可以分离：

$$
\Psi_{\text{total}}(\mathbf{R}, \mathbf{r}) \approx \Psi_{\text{nuclear}}(\mathbf{R}) \Psi_{\text{electronic}}(\mathbf{r}; \mathbf{R})
$$

**结果**：
- 电子瞬时调整到基态
- 原子核在有效势能面上运动
- 势能面 $U(\mathbf{R})$ 由电子结构计算得到

### MD模拟的步骤

```
1. 初始化
   ├─ 设置初始构型（晶格、随机等）
   ├─ 分配初始速度（Maxwell-Boltzmann分布）
   └─ 定义模拟参数（时间步长、温度等）

2. 主循环（每个时间步）
   ├─ 计算力: F_i = -∇U
   ├─ 更新位置: r(t+Δt)
   ├─ 更新速度: v(t+Δt)
   ├─ 应用边界条件
   └─ 输出轨迹数据

3. 数据分析
   ├─ 径向分布函数(RDF)
   ├─ 均方位移(MSD) → 扩散系数
   ├─ 热力学性质（温度、压力、能量）
   └─ 结构分析
```

### 周期性边界条件(PBC)

为了模拟"无限"体系，使用周期性边界条件：

```
   ┌─────────┬─────────┬─────────┐
   │  镜像   │  镜像   │  镜像   │
   ├─────────┼─────────┼─────────┤
   │  镜像   │模拟盒子 │  镜像   │
   ├─────────┼─────────┼─────────┤
   │  镜像   │  镜像   │  镜像   │
   └─────────┴─────────┴─────────┘
```

**最小镜像约定**：
- 原子只与最近的镜像相互作用
- 截断半径 $r_c < L/2$（$L$ 是盒子边长）

---

## 牛顿运动方程的几种数值求解方法

### 1. Verlet算法

**基本Verlet**：

$$
\mathbf{r}(t + \Delta t) = 2\mathbf{r}(t) - \mathbf{r}(t - \Delta t) + \frac{\mathbf{F}(t)}{m}\Delta t^2
$$

**特点**：
- 时间可逆
- 能量守恒好
- 不需要显式计算速度

**速度计算**（事后）：
$$
\mathbf{v}(t) = \frac{\mathbf{r}(t + \Delta t) - \mathbf{r}(t - \Delta t)}{2\Delta t}
$$

### 2. Velocity Verlet算法

**最常用的算法**：

```python
# 半步更新速度
v(t + Δt/2) = v(t) + F(t)/(2m) * Δt

# 全步更新位置
r(t + Δt) = r(t) + v(t + Δt/2) * Δt

# 计算新的力
F(t + Δt) = -∇U(r(t + Δt))

# 完成速度更新
v(t + Δt) = v(t + Δt/2) + F(t + Δt)/(2m) * Δt
```

**数学表达**：
$$
\mathbf{v}(t + \frac{\Delta t}{2}) = \mathbf{v}(t) + \frac{\mathbf{F}(t)}{2m}\Delta t
$$
$$
\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t + \frac{\Delta t}{2})\Delta t
$$
$$
\mathbf{v}(t + \Delta t) = \mathbf{v}(t + \frac{\Delta t}{2}) + \frac{\mathbf{F}(t + \Delta t)}{2m}\Delta t
$$

**优点**：
- 同时计算位置和速度
- 数值稳定性好
- 易于实现恒温器

### 3. Leapfrog算法

位置和速度交错计算：

$$
\mathbf{v}(t + \frac{\Delta t}{2}) = \mathbf{v}(t - \frac{\Delta t}{2}) + \frac{\mathbf{F}(t)}{m}\Delta t
$$
$$
\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t + \frac{\Delta t}{2})\Delta t
$$

### 4. Predictor-Corrector方法

**预测步**（使用泰勒展开）：
$$
\mathbf{r}_{\text{pred}}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t)\Delta t + \frac{\mathbf{a}(t)}{2}\Delta t^2
$$

**校正步**（使用新的力）：
$$
\mathbf{r}_{\text{corr}}(t + \Delta t) = \mathbf{r}(t) + \frac{\mathbf{v}(t) + \mathbf{v}(t + \Delta t)}{2}\Delta t
$$

### 时间步长选择

**经验法则**：
- $\Delta t \approx \frac{1}{10}$ 至 $\frac{1}{20}$ 最高频率振动周期
- 典型值：
  - **AIMD**：0.5-1 fs（需要描述氢原子振动）
  - **经验势MD**：1-2 fs
  - **粗粒化MD**：5-10 fs

**能量守恒检查**：
$$
\frac{\Delta E}{E} = \frac{|E(t) - E(0)|}{E(0)} < 10^{-4}
$$

---

## 统计力学系综概念

### 系综(Ensemble)

系综是具有相同宏观约束但微观状态不同的大量系统的集合。

### 1. 微正则系综(NVE)

**约束条件**：
- $N$：粒子数固定
- $V$：体积固定
- $E$：能量固定（孤立系统）

**特点**：
- 最基本的MD系综
- 能量守恒
- 对应绝热过程

**实现**：
- 使用保守力场
- Velocity Verlet算法自然保持NVE

### 2. 正则系综(NVT)

**约束条件**：
- $N$：粒子数固定
- $V$：体积固定
- $T$：温度固定（恒温系统）

**温度定义**（能量均分定理）：
$$
\langle E_{\text{kin}} \rangle = \frac{3}{2}Nk_BT
$$

**实现方法——恒温器(Thermostat)**：

#### (a) Berendsen恒温器
$$
\frac{d\mathbf{v}}{dt} = \frac{\mathbf{F}}{m} + \frac{1}{2\tau_T}\left(\frac{T_0}{T(t)} - 1\right)\mathbf{v}
$$

- 简单、稳定
- 不产生正确的系综（非正则）
- 适合平衡阶段

#### (b) Nosé-Hoover恒温器
$$
\frac{d\mathbf{v}}{dt} = \frac{\mathbf{F}}{m} - \xi\mathbf{v}
$$
$$
\frac{d\xi}{dt} = \frac{1}{Q}\left(T(t) - T_0\right)
$$

- 产生正确的正则系综
- $Q$是"热浴质量"参数

#### (c) Langevin恒温器
$$
m\frac{d\mathbf{v}}{dt} = \mathbf{F} - \gamma m\mathbf{v} + \mathbf{R}(t)
$$

- $\gamma$：摩擦系数
- $\mathbf{R}(t)$：随机力（满足涨落-耗散定理）

### 3. 等温等压系综(NPT)

**约束条件**：
- $N$：粒子数固定
- $P$：压力固定
- $T$：温度固定

**应用**：最接近实验条件

**实现方法——控压器(Barostat)**：

#### Berendsen控压器
$$
\frac{d\mathbf{h}}{dt} = \frac{1}{\tau_P}(P_0 - P(t))\mathbf{h}
$$

其中 $\mathbf{h}$ 是盒子矩阵。

#### Parrinello-Rahman控压器
- 允许盒子形状和体积同时变化
- 产生正确的NPT系综

### 4. 巨正则系综(μVT)

**约束条件**：
- $\mu$：化学势固定
- $V$：体积固定
- $T$：温度固定

**应用**：
- 吸附研究
- 多相平衡
- 电化学系统

**实现**：
- Grand Canonical Monte Carlo (GCMC)
- 混合MD-MC方法

### 系综选择指南

| 系综 | 适用场景 | 优点 | 缺点 |
|-----|---------|------|------|
| NVE | 验证算法、测试势函数 | 简单、基本 | 不易控制温度 |
| NVT | 平衡态性质、结构优化 | 温度恒定 | 密度固定 |
| NPT | 模拟实验条件、相变 | 最接近实验 | 计算稍复杂 |
| μVT | 吸附、化学反应 | 粒子数可变 | 需要MC |

---

## LAMMPS分子动力学软件介绍

### LAMMPS简介

**LAMMPS** (Large-scale Atomic/Molecular Massively Parallel Simulator)

**特点**：
- 开源、高性能
- 支持多种势函数（经验势、神经网络势等）
- 强大的并行能力（MPI、GPU）
- 丰富的分析工具
- 灵活的脚本语言

**官网**：https://www.lammps.org/

### LAMMPS基本概念

#### 1. 原子样式(Atom Style)

```lammps
atom_style atomic      # 单原子（金属、惰性气体）
atom_style molecular   # 分子（有键连接）
atom_style full        # 完整（键、角、二面角、电荷）
atom_style charge      # 带电原子
```

#### 2. 边界条件

```lammps
boundary p p p    # 三个方向都是周期性(periodic)
boundary p p f    # x,y周期性，z固定(fixed)
boundary s s s    # 三个方向都是收缩(shrink-wrapped)
```

#### 3. 势函数(Pair Style)

```lammps
# Lennard-Jones势
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0

# EAM势(嵌入原子势)
pair_style eam/alloy
pair_coeff * * Cu_u3.eam Cu

# 神经网络势(DeePMD)
pair_style deepmd model.pb
pair_coeff * *
```

### LAMMPS脚本结构

```lammps
# 1. 初始化
units metal              # 单位系统
atom_style atomic
boundary p p p

# 2. 创建模拟盒子
lattice fcc 3.615        # FCC晶格，晶格常数3.615 Å
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 box

# 3. 定义势函数
pair_style eam/alloy
pair_coeff * * Cu_u3.eam Cu

# 4. 设置
mass 1 63.546            # Cu原子质量

# 5. 运行设置
velocity all create 300.0 12345  # 初始化速度
fix 1 all nvt temp 300.0 300.0 0.1  # NVT系综

# 6. 输出
thermo 100               # 每100步输出热力学信息
dump 1 all custom 1000 dump.lammpstrj id type x y z

# 7. 运行
timestep 0.001           # 时间步长 1 fs
run 10000                # 运行10000步
```

### LAMMPS输出文件

**1. 热力学输出**：
```
Step Temp E_pair E_mol TotEng Press
   0  300.0  -12345.6   0.0  -11234.5  12.34
 100  299.8  -12346.1   0.0  -11235.0  12.28
```

**2. 轨迹文件(dump file)**：
```
ITEM: TIMESTEP
1000
ITEM: NUMBER OF ATOMS
4000
ITEM: BOX BOUNDS pp pp pp
0.0 36.15
0.0 36.15
0.0 36.15
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 1.8 1.8 0.0
...
```

**3. 重启文件(restart)**：
- 保存完整状态
- 用于继续模拟或恢复

---

## 神经网络势函数和LAMMPS的结合

### DeePMD-kit + LAMMPS

**DeePMD-kit**是由张林峰团队开发的深度学习势函数工具包。

#### 工作流程

```
1. 数据准备
   ├─ AIMD生成训练数据
   ├─ 转换为DeePMD格式
   └─ 划分训练集/验证集/测试集

2. 模型训练
   ├─ 定义模型架构(input.json)
   ├─ 训练模型(dp train)
   ├─ 模型压缩(dp compress)
   └─ 冻结模型(dp freeze)

3. LAMMPS模拟
   ├─ 加载势函数模型
   ├─ 运行MD模拟
   └─ 分析结果
```

#### DeePMD训练示例

**input.json**（部分）：
```json
{
    "model": {
        "type_map": ["Cu", "O"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [60, 60],
            "rcut_smth": 0.50,
            "rcut": 6.00,
            "neuron": [25, 50, 100],
            "resnet_dt": false,
            "axis_neuron": 16
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": true
        }
    },
    "learning_rate": {
        "type": "exp",
        "start_lr": 0.001,
        "decay_steps": 5000,
        "decay_rate": 0.95
    },
    "loss": {
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1
    }
}
```

**训练命令**：
```bash
# 训练模型
dp train input.json

# 冻结模型
dp freeze -o model.pb

# 压缩模型（可选，减小模型大小）
dp compress -i model.pb -o model_compressed.pb
```

#### LAMMPS使用DeePMD势

```lammps
# LAMMPS脚本
units metal
atom_style atomic
boundary p p p

# 读取结构
read_data data.lammps

# 使用DeePMD势
pair_style deepmd model_compressed.pb
pair_coeff * *

# 设置NVT
velocity all create 300.0 12345
fix 1 all nvt temp 300.0 300.0 0.1

# 输出
thermo 100
dump 1 all custom 1000 dump.lammpstrj id type x y z fx fy fz

# 运行
timestep 0.001
run 100000
```

### 性能对比

| 方法 | 计算速度 | 准确性 | 系统大小 | 时间尺度 |
|------|---------|--------|---------|---------|
| AIMD (VASP) | 1× | 参考 | ~100原子 | ~10 ps |
| NNP+LAMMPS | 10^4×-10^6× | 接近AIMD | ~10^6原子 | ~μs |
| 经验势+LAMMPS | 10^6× | 较差 | >10^6原子 | >μs |

### 应用案例

**1. 液态金属结构**：
- 使用DeePMD势模拟液态铜
- 计算径向分布函数(RDF)
- 与实验X射线衍射对比

**2. 表面吸附**：
- 氧分子在Cu表面的吸附和扩散
- 计算吸附能和扩散势垒
- 需要描述化学键的形成和断裂

**3. 纳米材料**：
- 纳米粒子的结构稳定性
- 熔化温度预测
- 界面性质研究

**4. 离子导体**：
- 锂离子在固体电解质中的扩散
- 计算离子电导率
- 理解扩散机制

---

## 小结

1. **分子动力学**是连接微观和宏观的桥梁
2. **神经网络势函数**实现了DFT精度和经验势速度的结合
3. **LAMMPS**提供了强大的MD模拟平台
4. **DeePMD-kit**使神经网络势易于训练和使用

---

## 扩展阅读

1. Frenkel, D., & Smit, B. (2001). *Understanding Molecular Simulation*. Academic press.
2. Allen, M. P., & Tildesley, D. J. (2017). *Computer Simulation of Liquids*. Oxford university press.
3. Zhang, L., et al. (2018). End-to-end symmetry preserving inter-atomic potential energy model for finite and extended systems. *NeurIPS*.
4. [LAMMPS Documentation](https://docs.lammps.org/)
5. [DeePMD-kit Documentation](https://docs.deepmodeling.com/projects/deepmd/)

---

**下一节**: [图神经网络和MPNN消息传递神经网络](./02-Graph-Neural-Networks.md)
