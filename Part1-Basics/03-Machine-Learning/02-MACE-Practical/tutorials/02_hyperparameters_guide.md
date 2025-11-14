# MACE超参数完全指南 | Complete Guide to MACE Hyperparameters

## 概述 | Overview

MACE模型的性能高度依赖于超参数的选择。本指南详细解释每个超参数的作用，推荐值，以及调优策略。

MACE model performance heavily depends on hyperparameter selection. This guide explains each hyperparameter in detail, recommended values, and tuning strategies.

---

## 1. 模型架构超参数 | Model Architecture Hyperparameters

### 1.1 r_max (截断半径 | Cutoff Radius)

**含义**: 原子间相互作用的最大距离

**默认值**: 5.0 Å

**作用**:
- 定义了原子的"邻域"范围
- 影响计算成本（O(N × n_neighbors)）
- 影响模型的物理准确性

**选择策略**:

```python
# 不同系统的推荐值
材料类型          推荐r_max (Å)     原因
===============================================
金属              5.0 - 6.0        长程相互作用
共价晶体          4.0 - 5.0        近邻相互作用
分子              4.0 - 5.0        分子内相互作用
离子晶体          5.5 - 7.0        库仑相互作用
```

**调优建议**:
1. 从5.0Å开始
2. 逐步增加，观察能量收敛
3. 平衡精度和计算成本
4. 确保超胞大小 > 2 × r_max

**示例**:
```python
model = MACE(
    r_max=5.0,  # 5埃截断
    # ...
)
```

---

### 1.2 num_interactions (交互层数 | Number of Interaction Layers)

**含义**: 消息传递的层数，决定了信息传播的"深度"

**默认值**: 2

**作用**:
- 控制模型表达能力
- 影响长程相互作用的捕获
- 影响训练时间和内存消耗

**层数效果**:
```
1层: 只考虑一阶邻居
2层: 考虑到二阶邻居（推荐）
3层: 考虑到三阶邻居（高精度）
4层+: 通常不必要，可能过拟合
```

**选择策略**:
- **小分子**: 1-2层足够
- **晶体**: 2-3层
- **复杂系统**: 2-3层
- **大系统**: 1-2层（计算限制）

**计算成本**:
```
层数    相对训练时间    相对内存    推荐场景
===================================================
1       1×            1×         快速原型
2       2×            1.5×       大多数应用（推荐）
3       3×            2×         高精度需求
```

**示例**:
```python
# 标准配置
num_interactions=2

# 高精度配置
num_interactions=3
```

---

### 1.3 hidden_irreps (隐藏层不可约表示 | Hidden Irreducible Representations)

**含义**: 定义隐藏特征的球谐函数展开

**默认值**: "128x0e + 128x1o"

**格式**: "n1×l1p1 + n2×l2p2 + ..."
- n: 通道数 (multiplicity)
- l: 球谐阶数 (0, 1, 2, ...)
- p: 宇称 (e=even/偶, o=odd/奇)

**球谐阶数含义**:
```
l=0 (标量): 旋转不变量（能量等）
l=1 (矢量): 力、偶极矩等
l=2 (张量): 应力、四极矩等
l=3+: 高阶多极矩
```

**常用配置**:

```python
# 小模型（快速）
hidden_irreps="64x0e + 64x1o"

# 标准模型（推荐）
hidden_irreps="128x0e + 128x1o"

# 大模型（高精度）
hidden_irreps="256x0e + 256x1o"

# 包含l=2（更高精度，更慢）
hidden_irreps="128x0e + 128x1o + 128x2e"
```

**选择策略**:
1. 从"128x0e + 128x1o"开始
2. 如需更高精度，增加通道数到256
3. 如需计算应力，添加l=2项
4. 避免过大的l值（计算昂贵）

**性能影响**:
```
配置                      参数量    训练速度    推荐用途
================================================================
64x0e + 64x1o            ~0.5M     快        快速测试
128x0e + 128x1o          ~2M       中等      标准应用
256x0e + 256x1o          ~8M       慢        高精度
128x0e+128x1o+128x2e     ~4M       很慢      需要应力
```

---

### 1.4 correlation (ACE相关阶数 | Correlation Order)

**含义**: ACE展开的相关阶数，控制多体相互作用

**默认值**: 3

**作用**:
- correlation=2: 两体和三体相互作用
- correlation=3: 到四体相互作用（推荐）
- correlation=4: 到五体相互作用

**物理意义**:
```
阶数    描述                      典型用途
====================================================
2       两体 + 三体              简单系统
3       到四体（推荐）           大多数材料
4       到五体                   高精度，罕见
```

**选择策略**:
- **大多数情况**: 使用correlation=3
- **简单系统/快速训练**: correlation=2
- **极高精度需求**: correlation=4（计算昂贵）

**示例**:
```python
correlation=3  # 推荐
```

---

### 1.5 max_ell (最大球谐阶数 | Maximum Spherical Harmonic Order)

**含义**: 球谐展开的最大阶数

**默认值**: 3

**作用**: 控制角度信息的分辨率

**选择策略**:
```
max_ell    精度    计算成本    推荐场景
=================================================
2          较低    低          快速原型
3          标准    中等        大多数应用（推荐）
4          高      高          高精度需求
```

**注意**: max_ell应该 >= hidden_irreps中的最大l值

---

### 1.6 num_radial_basis (径向基函数数量 | Number of Radial Basis Functions)

**含义**: Bessel函数基的数量，用于径向部分的展开

**默认值**: 8

**作用**: 控制径向距离的分辨率

**选择策略**:
```python
# 标准配置
num_radial_basis=8

# 高精度配置
num_radial_basis=10-16

# 快速配置
num_radial_basis=6
```

**建议**: 8-10通常足够，更多不一定更好

---

### 1.7 MLP_irreps (MLP隐藏维度 | MLP Hidden Dimensions)

**含义**: 非线性MLP的隐藏层不可约表示

**默认值**: "16x0e"

**作用**: 增加模型非线性表达能力

**常用配置**:
```python
MLP_irreps="16x0e"   # 标准
MLP_irreps="32x0e"   # 更强表达力
MLP_irreps="8x0e"    # 快速
```

---

## 2. 训练超参数 | Training Hyperparameters

### 2.1 batch_size (批量大小)

**默认值**: 32

**作用**: 每次参数更新使用的样本数

**GPU内存影响**:
```
batch_size    GPU内存需求    训练速度    推荐场景
=========================================================
8             ~4GB          慢         小GPU
16            ~6GB          中等       8GB GPU
32            ~10GB         快         16GB GPU
64            ~18GB         很快       24GB+ GPU
```

**选择策略**:
1. 尽可能大（受GPU内存限制）
2. 2的幂次（8, 16, 32, 64）
3. 梯度累积可模拟更大batch

**示例**:
```python
# 根据GPU选择
batch_size=32  # RTX 3090 (24GB)
batch_size=16  # RTX 3080 (10GB)
batch_size=8   # GTX 1080 Ti (11GB)
```

---

### 2.2 learning_rate (学习率)

**默认值**: 0.01

**作用**: 控制参数更新步长

**推荐值**:
```python
# 从头训练
learning_rate=0.01

# 微调预训练模型
learning_rate=0.001  # 更小的学习率
```

**学习率调度**:
```python
# 推荐策略
1. 预热 (warmup): 前5-10个epoch线性增加
2. 余弦退火: 平滑衰减
3. ReduceLROnPlateau: 验证损失平台期时降低
```

**示例**:
```python
# 配置
"scheduler": "ReduceLROnPlateau",
"scheduler_patience": 50,
"lr_factor": 0.8,  # 降低到80%
```

---

### 2.3 max_num_epochs (最大训练轮数)

**默认值**: 1000

**作用**: 训练的最大迭代次数

**选择策略**:
```python
# 从头训练
max_num_epochs=1000-2000

# 微调
max_num_epochs=200-500

# 快速实验
max_num_epochs=100
```

**注意**: 配合早停 (early stopping) 使用

---

### 2.4 patience (早停耐心值)

**默认值**: 50

**作用**: 验证损失不改善时等待的epoch数

**选择策略**:
```python
# 大数据集
patience=50-100

# 小数据集（更容易过拟合）
patience=20-50

# 快速实验
patience=10-20
```

---

### 2.5 ema (指数移动平均 | Exponential Moving Average)

**默认值**: True

**作用**: 平滑模型参数，提高泛化性能

**推荐**: 始终启用

**EMA衰减率**:
```python
ema_decay=0.99  # 标准
ema_decay=0.999 # 更平滑
```

---

### 2.6 能量和力的损失权重 | Energy & Force Loss Weights

**参数**:
- `energy_weight`: 能量损失权重
- `forces_weight`: 力损失权重

**默认值**:
```python
energy_weight=1.0
forces_weight=100.0  # 力的权重更高
```

**为什么力的权重更高？**
- 力的数量是能量的3N倍（N个原子）
- 力的单位变化对总损失影响较小
- 需要平衡两者的贡献

**调整策略**:
```python
# 重视能量精度
energy_weight=10.0
forces_weight=100.0

# 重视力精度（MD模拟）
energy_weight=1.0
forces_weight=1000.0

# 平衡配置（推荐）
energy_weight=1.0
forces_weight=100.0
```

**单位转换**:
```python
# 确保单位一致
# 能量: eV 或 Hartree
# 力: eV/Å 或 Hartree/Bohr

# 权重公式建议
forces_weight = energy_weight * (typical_energy_range / typical_force_range) * (3 * avg_num_atoms)
```

---

## 3. 数据相关超参数 | Data Hyperparameters

### 3.1 train_file, valid_file, test_file

**格式**: XYZ或其他ASE支持的格式

**推荐比例**:
```
训练集: 70-80%
验证集: 10-15%
测试集: 10-15%
```

### 3.2 E0s (原子参考能量)

**含义**: 孤立原子的能量，用于标准化

**获取方式**:
1. DFT计算孤立原子能量
2. 从训练数据拟合
3. 使用MACE提供的值

**示例**:
```python
E0s="average"  # 自动从数据计算
# 或
E0s={1: -13.6, 6: -1029.5, 8: -2041.5}  # 手动指定（H, C, O）
```

---

## 4. 高级超参数 | Advanced Hyperparameters

### 4.1 gate (门控 | Gating)

**选项**: "silu", "tanh", "abs", "None"

**默认**: "silu"

**作用**: 非线性激活函数

**推荐**: "silu" (Swish activation)

### 4.2 radial_type (径向基类型)

**选项**: "bessel", "gaussian", "chebyshev"

**默认**: "bessel"

**推荐**: 使用默认的"bessel"

### 4.3 distance_transform (距离变换)

**选项**: "None", "Agnesi", "soft"

**默认**: "None"

**作用**: 平滑截断函数

---

## 5. 超参数调优策略 | Hyperparameter Tuning Strategy

### 5.1 三阶段调优方法

**阶段1: 架构搜索**
```python
# 固定训练参数，调整架构
grid_search = {
    'r_max': [4.5, 5.0, 5.5],
    'num_interactions': [1, 2, 3],
    'hidden_irreps': ['64x0e+64x1o', '128x0e+128x1o', '256x0e+256x1o']
}
```

**阶段2: 训练参数优化**
```python
# 固定最佳架构，调整训练参数
grid_search = {
    'learning_rate': [0.001, 0.005, 0.01],
    'batch_size': [16, 32, 64],
    'forces_weight': [50, 100, 200]
}
```

**阶段3: 精细调整**
```python
# 在最佳配置周围微调
fine_tune = {
    'num_radial_basis': [8, 10, 12],
    'correlation': [2, 3],
    'MLP_irreps': ['16x0e', '32x0e']
}
```

### 5.2 使用Optuna自动调优

```python
import optuna

def objective(trial):
    # 定义搜索空间
    r_max = trial.suggest_float('r_max', 4.0, 6.0)
    num_interactions = trial.suggest_int('num_interactions', 1, 3)
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])

    # 训练模型
    model = train_mace(r_max=r_max, num_interactions=num_interactions, ...)

    # 返回验证损失
    return validation_loss

# 运行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### 5.3 学习曲线分析

```python
# 绘制学习曲线判断：
1. 欠拟合: 训练和验证损失都高 → 增加模型容量
2. 过拟合: 训练损失低，验证损失高 → 减少模型容量或增加数据
3. 正常: 两者都低且接近 → 当前配置良好
```

---

## 6. 推荐配置 | Recommended Configurations

### 6.1 快速原型 (Fast Prototyping)
```python
{
    "r_max": 5.0,
    "num_interactions": 1,
    "hidden_irreps": "64x0e + 64x1o",
    "MLP_irreps": "16x0e",
    "correlation": 2,
    "max_ell": 2,
    "num_radial_basis": 8,
    "batch_size": 32,
    "learning_rate": 0.01,
    "max_num_epochs": 500
}
```

### 6.2 标准配置 (Standard)
```python
{
    "r_max": 5.0,
    "num_interactions": 2,
    "hidden_irreps": "128x0e + 128x1o",
    "MLP_irreps": "16x0e",
    "correlation": 3,
    "max_ell": 3,
    "num_radial_basis": 8,
    "batch_size": 32,
    "learning_rate": 0.01,
    "max_num_epochs": 1000,
    "patience": 50,
    "energy_weight": 1.0,
    "forces_weight": 100.0
}
```

### 6.3 高精度配置 (High Accuracy)
```python
{
    "r_max": 6.0,
    "num_interactions": 3,
    "hidden_irreps": "256x0e + 256x1o",
    "MLP_irreps": "32x0e",
    "correlation": 3,
    "max_ell": 3,
    "num_radial_basis": 10,
    "batch_size": 16,  # 更大模型需要更小batch
    "learning_rate": 0.005,
    "max_num_epochs": 2000,
    "patience": 100,
    "energy_weight": 1.0,
    "forces_weight": 200.0
}
```

### 6.4 微调配置 (Fine-tuning)
```python
{
    "r_max": 5.0,  # 与预训练模型一致
    "num_interactions": 2,
    "hidden_irreps": "128x0e + 128x1o",
    "learning_rate": 0.0005,  # 更小的学习率
    "max_num_epochs": 200,
    "patience": 30,
    "restart_latest": True,  # 从检查点继续
    "ema": True
}
```

---

## 7. 调试清单 | Debugging Checklist

### 训练损失不下降？
- [ ] 检查学习率（可能太小或太大）
- [ ] 检查数据标准化
- [ ] 检查E0s设置
- [ ] 验证数据质量（是否有异常值）
- [ ] 尝试更简单的模型

### 验证损失比训练损失高很多？
- [ ] 过拟合：减小模型，增加数据
- [ ] 检查数据划分（训练/验证集分布是否一致）
- [ ] 增加正则化
- [ ] 使用数据增强

### 训练很慢？
- [ ] 减少batch_size适应GPU内存
- [ ] 减少num_interactions
- [ ] 减少hidden_irreps通道数
- [ ] 使用混合精度训练
- [ ] 检查数据加载瓶颈

### 内存不足？
- [ ] 减小batch_size
- [ ] 减小模型大小
- [ ] 使用梯度累积
- [ ] 使用混合精度 (FP16)

---

## 8. 总结 | Summary

**核心超参数优先级**:
1. ⭐⭐⭐ `r_max`, `num_interactions`, `hidden_irreps` - 决定模型能力
2. ⭐⭐ `learning_rate`, `batch_size`, `forces_weight` - 影响训练效果
3. ⭐ `correlation`, `max_ell`, `num_radial_basis` - 微调精度

**通用建议**:
- 从推荐的标准配置开始
- 逐个调整超参数，记录结果
- 使用验证集选择超参数
- 在独立测试集上评估最终模型
- 记录所有实验配置

---

**下一步**: 查看 [MACE-MP-0模型指南](./03_mace_mp0_guide.md) 了解如何使用预训练模型。
