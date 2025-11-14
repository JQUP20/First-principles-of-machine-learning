# MACE实战教程 | MACE Practical Guide

## 简介 | Introduction

MACE (Multi Atomic Cluster Expansion) 是一个先进的等变神经网络原子间势能模型，在材料科学和分子动力学领域表现优异。本模块提供MACE的完整实战教程，包括超参数调优、预训练模型微调和分子动力学模拟。

MACE (Multi Atomic Cluster Expansion) is an advanced equivariant neural network interatomic potential that excels in materials science and molecular dynamics. This module provides a complete practical guide for MACE, including hyperparameter tuning, pre-trained model fine-tuning, and molecular dynamics simulations.

## 学习目标 | Learning Objectives

- 深入理解MACE模型架构和超参数
- 掌握MACE-MP-0预训练模型的加载和使用
- 学会在自定义数据集上微调MACE模型
- 应用MACE进行高精度分子动力学模拟
- 评估和对比MACE与传统力场的性能

## 目录 | Contents

### 1. MACE基础 | MACE Fundamentals

#### (1) MACE架构概述
- ACE (Atomic Cluster Expansion) 理论基础
- MACE与Allegro/NequIP的区别
- 等变性 (Equivariance) 的重要性
- 消息传递机制

📖 [MACE架构详解](./tutorials/01_mace_architecture.md)

#### (2) MACE超参数详解
- 模型超参数 (r_max, num_interactions, hidden_irreps等)
- 训练超参数 (learning rate, batch size, loss weights)
- 数据增强策略
- 超参数调优最佳实践

📖 [超参数完全指南](./tutorials/02_hyperparameters_guide.md)

#### (3) MACE-MP-0预训练模型
- Materials Project训练数据
- 模型规模和性能
- 适用范围和局限性
- 与其他预训练模型对比

📖 [MACE-MP-0模型指南](./tutorials/03_mace_mp0_guide.md)

### 2. 实战案例 | Practical Examples

#### 案例1: MACE-MP-0模型加载和使用

快速开始使用预训练的MACE-MP-0模型：

```python
examples/01_load_mace_mp0.py
```

**特点：**
- 下载和加载MACE-MP-0模型
- 单点能量和力计算
- 与DFT结果对比
- 推理速度测试

💻 [完整代码](./examples/01_load_mace_mp0.py)

#### 案例2: 在自定义数据上微调MACE

在小数据集上微调预训练模型：

```python
examples/02_finetune_mace.py
```

**特点：**
- 准备自定义训练数据
- 冻结/解冻层策略
- 学习率调度
- 验证和测试
- 与从头训练对比

💻 [完整代码](./examples/02_finetune_mace.py)

#### 案例3: MACE分子动力学模拟

使用MACE进行高精度MD模拟：

```python
examples/03_mace_md_simulation.py
```

**特点：**
- ASE集成
- NVT/NPT系综
- 轨迹分析
- 径向分布函数 (RDF)
- 能量守恒检查
- 性能基准测试

💻 [完整代码](./examples/03_mace_md_simulation.py)

#### 案例4: 超参数优化

系统地搜索最优超参数：

```python
examples/04_hyperparameter_optimization.py
```

**特点：**
- Optuna集成
- 贝叶斯优化
- 交叉验证
- 结果可视化
- 最佳配置导出

💻 [完整代码](./examples/04_hyperparameter_optimization.py)

### 3. 工具脚本 | Utility Scripts

```python
scripts/
├── prepare_data.py           # 数据准备和格式转换
├── evaluate_model.py         # 模型评估工具
├── compare_models.py         # 多模型性能对比
└── visualize_results.py      # 结果可视化
```

## 环境要求 | Requirements

### 必需软件 | Required Software

```bash
# MACE和依赖
pip install mace-torch

# ASE (原子模拟环境)
pip install ase

# PyTorch (根据CUDA版本)
pip install torch

# 数据处理和可视化
pip install numpy matplotlib pandas seaborn

# 分子动力学
pip install mdtraj  # 可选

# 超参数优化
pip install optuna  # 可选
```

### 推荐配置 | Recommended Setup

- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CPU**: 8+ cores for parallel training
- **RAM**: 16GB+
- **Storage**: 50GB+ for models and data

## 快速开始 | Quick Start

### 1. 安装MACE

```bash
# 创建环境
conda create -n mace python=3.10
conda activate mace

# 安装PyTorch (根据你的CUDA版本)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 安装MACE
pip install mace-torch

# 安装其他依赖
pip install ase numpy matplotlib pandas
```

### 2. 快速测试

```python
# 测试MACE安装
python -c "import mace; print(f'MACE version: {mace.__version__}')"

# 测试MACE计算器
from mace.calculators import MACECalculator
calc = MACECalculator(model_paths='small', device='cuda')
print("MACE calculator created successfully!")
```

### 3. 运行第一个示例

```bash
cd examples
python 01_load_mace_mp0.py
```

## MACE模型架构 | MACE Architecture

```
输入: 原子坐标和元素
    ↓
┌─────────────────────────────────┐
│  Node Features                  │
│  (原子特征)                      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Interaction Blocks (重复N次)    │
│                                 │
│  ┌───────────────────┐         │
│  │ Equivariant       │         │
│  │ Message Passing   │         │
│  └───────────────────┘         │
│           ↓                     │
│  ┌───────────────────┐         │
│  │ Self-Interaction  │         │
│  └───────────────────┘         │
│           ↓                     │
│  ┌───────────────────┐         │
│  │ Non-linearity     │         │
│  └───────────────────┘         │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Readout Block                  │
│  (能量和力的输出)                 │
└─────────────────────────────────┘
    ↓
输出: 总能量, 原子力, 应力 (可选)
```

## MACE超参数概览 | Hyperparameters Overview

### 核心超参数 | Core Hyperparameters

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `r_max` | 5.0 | 截断半径 (Å) | 4.0-6.0 |
| `num_interactions` | 2 | 交互层数 | 1-3 |
| `hidden_irreps` | "128x0e + 128x1o" | 隐藏层不可约表示 | 见详细指南 |
| `MLP_irreps` | "16x0e" | MLP隐藏维度 | 8-32 |
| `correlation` | 3 | ACE相关阶数 | 2-4 |
| `max_ell` | 3 | 最大球谐阶数 | 2-3 |
| `num_radial_basis` | 8 | 径向基函数数量 | 8-16 |

### 训练超参数 | Training Hyperparameters

| 参数 | 默认值 | 说明 | 推荐范围 |
|------|--------|------|----------|
| `batch_size` | 32 | 批量大小 | 8-64 |
| `learning_rate` | 0.01 | 初始学习率 | 0.001-0.01 |
| `max_num_epochs` | 1000 | 最大训练轮数 | 500-2000 |
| `patience` | 50 | 早停耐心值 | 20-100 |
| `energy_weight` | 1.0 | 能量损失权重 | 0.1-10 |
| `forces_weight` | 100.0 | 力损失权重 | 10-1000 |

详细说明请参考：📖 [超参数完全指南](./tutorials/02_hyperparameters_guide.md)

## MACE-MP-0模型信息 | MACE-MP-0 Model Info

### 模型变体 | Model Variants

| 模型 | 参数量 | 元素覆盖 | 训练数据 | 推荐用途 |
|------|--------|---------|---------|---------|
| `small` | ~1M | 89 elements | MP subset | 快速测试 |
| `medium` | ~5M | 89 elements | MP | 一般应用 |
| `large` | ~20M | 89 elements | MP | 高精度 |
| `MACE-MP-0` | ~5M | 89 elements | Full MP | 生产环境 |

### 性能基准 | Performance Benchmarks

基于Materials Project测试集：

| 指标 | MACE-MP-0 | Baseline |
|------|-----------|----------|
| Energy MAE (meV/atom) | **5-10** | 20-30 |
| Force MAE (meV/Å) | **20-40** | 50-100 |
| 推理速度 (atoms/s) | **10K-50K** | - |

## MACE vs 其他模型 | MACE vs Other Models

| 特性 | MACE | Allegro | NequIP | SchNet |
|------|------|---------|--------|--------|
| 等变性 | ✅ E(3) | ✅ E(3) | ✅ E(3) | ❌ |
| 消息传递 | ✅ | ✅ | ✅ | ✅ |
| ACE基础 | ✅ | ❌ | ❌ | ❌ |
| 预训练模型 | ✅ MACE-MP-0 | ❌ | ❌ | ❌ |
| 训练速度 | 中等 | 快 | 中等 | 快 |
| 精度 | **高** | 高 | 高 | 中 |
| 可解释性 | 较好 | 一般 | 一般 | 较差 |

## 应用场景 | Use Cases

### 1. 材料性质预测
- 晶体结构优化
- 弹性常数计算
- 相图预测
- 缺陷能量

### 2. 分子动力学
- 高温高压模拟
- 相变研究
- 扩散系数计算
- 热导率预测

### 3. 表面科学
- 吸附能计算
- 表面重构
- 催化反应路径
- 界面性质

### 4. 纳米材料
- 纳米团簇
- 二维材料
- 量子点
- 纳米线

## 常见问题 | FAQ

### Q1: MACE和Allegro有什么区别？

**A:** 主要区别：
- **理论基础**: MACE基于ACE (Atomic Cluster Expansion)，Allegro基于tensor products
- **预训练模型**: MACE有MACE-MP-0预训练模型，Allegro目前没有
- **训练效率**: Allegro通常训练更快，MACE精度可能略高
- **社区支持**: 两者都有活跃的社区

### Q2: 如何选择合适的r_max？

**A:** 考虑因素：
- **物理系统**: 金属(5-6Å)，分子(4-5Å)，离子晶体(6-7Å)
- **计算成本**: 更大的r_max需要更多计算
- **收敛性**: 测试不同r_max直到能量收敛
- **推荐**: 从5.0Å开始，根据需要调整

### Q3: 微调需要多少训练数据？

**A:** 取决于任务：
- **相似体系**: 10-100个结构可能足够
- **新化学空间**: 100-1000个结构
- **高精度要求**: 1000+个结构
- **建议**: 使用学习曲线判断是否需要更多数据

### Q4: MACE-MP-0可以用于分子吗？

**A:** 可以，但有限制：
- **适用**: C, H, O, N等常见元素的有机分子
- **不适用**: 训练数据主要是晶体，对某些分子可能精度不足
- **建议**: 对关键分子进行微调或使用专门的分子数据集训练

### Q5: 如何处理周期性边界条件？

**A:** MACE自动处理：
- 通过ASE的`pbc`参数指定周期性
- 计算器会自动考虑周期性镜像
- 确保超胞足够大（原子间距离 > r_max）

## 最佳实践 | Best Practices

### 数据准备
1. ✅ 使用多样化的训练数据
2. ✅ 包含不同温度和压力的配置
3. ✅ 确保能量和力的一致性
4. ✅ 标准化和归一化数据

### 模型训练
1. ✅ 从预训练模型开始（如果适用）
2. ✅ 使用学习率预热和衰减
3. ✅ 监控验证损失，使用早停
4. ✅ 保存最佳模型检查点

### 模型评估
1. ✅ 在独立测试集上评估
2. ✅ 检查能量和力的误差
3. ✅ 验证物理性质（如结构优化）
4. ✅ 与DFT或实验数据对比

### 生产部署
1. ✅ 测试推理速度
2. ✅ 验证能量守恒（MD模拟）
3. ✅ 监控异常值
4. ✅ 记录和版本化模型

## 参考资料 | References

### 论文 | Papers

1. **MACE原始论文**:
   - Batatia, I., et al. (2022). "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields." *NeurIPS 2022*.
   - https://arxiv.org/abs/2206.07697

2. **MACE-MP-0预训练模型**:
   - Batatia, I., et al. (2023). "A foundation model for atomistic materials chemistry." *arXiv preprint*.
   - https://arxiv.org/abs/2401.00096

3. **ACE理论**:
   - Drautz, R. (2019). "Atomic cluster expansion for accurate and transferable interatomic potentials." *Physical Review B*, 99(1), 014104.

### 代码和工具 | Code & Tools

- **MACE GitHub**: https://github.com/ACEsuit/mace
- **MACE文档**: https://mace-docs.readthedocs.io/
- **Materials Project**: https://materialsproject.org/
- **ASE**: https://wiki.fysik.dtu.dk/ase/

### 教程和资源 | Tutorials & Resources

- [MACE官方教程](https://github.com/ACEsuit/mace/tree/main/examples)
- [Materials Project API](https://docs.materialsproject.org/)
- [E(3)等变神经网络综述](https://arxiv.org/abs/2104.01963)

## 项目结构 | Project Structure

```
02-MACE-Practical/
├── README.md                          # 本文件
├── tutorials/                         # 教程文档
│   ├── 01_mace_architecture.md
│   ├── 02_hyperparameters_guide.md
│   └── 03_mace_mp0_guide.md
├── examples/                          # 实战案例
│   ├── 01_load_mace_mp0.py
│   ├── 02_finetune_mace.py
│   ├── 03_mace_md_simulation.py
│   └── 04_hyperparameter_optimization.py
├── scripts/                           # 工具脚本
│   ├── prepare_data.py
│   ├── evaluate_model.py
│   ├── compare_models.py
│   └── visualize_results.py
├── data/                             # 数据目录
├── models/                           # 模型保存
└── results/                          # 结果输出
```

## 贡献 | Contributing

欢迎贡献代码、报告bug或提出改进建议！

## 许可证 | License

本教程基于MIT许可证。MACE代码遵循其自身的许可证。

---

**开始你的MACE实战之旅！** 🚀⚛️🔬
