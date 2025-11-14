# 从不变性到等变性——具有等变性的消息传递神经网络
# From Invariance to Equivariance: Equivariant Message Passing Neural Networks

## 目录
- [等变的概念](#等变的概念)
- [等变性和不变性的区别](#等变性和不变性的区别)
- [理解等变性——群论的初步介绍](#理解等变性群论的初步介绍)
- [等变消息传递神经网络和不变消息传递神经网络的对比](#等变消息传递神经网络和不变消息传递神经网络的对比)

---

## 等变的概念

### 什么是等变性？

**等变性(Equivariance)**是指函数的输出随输入的变换而协变地变换。

**数学定义**：

对于函数 $f: X \rightarrow Y$ 和变换群 $G$，如果：
$$
f(g \cdot x) = g \cdot f(x), \quad \forall g \in G, x \in X
$$

则称 $f$ 对群 $G$ 是**等变的**。

### 物理中的等变性

**力是等变量**：

如果我们旋转整个原子系统，力也应该相应旋转：

```
原系统:     旋转R:      旋转后系统:
  ↑F          →         ↑RF
  ●                      ●
```

数学表达：
$$
\mathbf{F}(\mathbf{R}\mathbf{r}_1, \mathbf{R}\mathbf{r}_2, ...) = \mathbf{R}\mathbf{F}(\mathbf{r}_1, \mathbf{r}_2, ...)
$$

**能量是不变量**：

旋转不改变能量：
$$
E(\mathbf{R}\mathbf{r}_1, \mathbf{R}\mathbf{r}_2, ...) = E(\mathbf{r}_1, \mathbf{r}_2, ...)
$$

### 常见的等变性和不变性

| 物理量 | 类型 | 变换行为 | 数学表示 |
|-------|------|---------|---------|
| 能量 | 不变量(标量) | 不变 | $E' = E$ |
| 力 | 等变量(向量) | 旋转 | $\mathbf{F}' = \mathbf{R}\mathbf{F}$ |
| 偶极矩 | 等变量(向量) | 旋转 | $\boldsymbol{\mu}' = \mathbf{R}\boldsymbol{\mu}$ |
| 应力张量 | 等变量(张量) | 协变 | $\boldsymbol{\sigma}' = \mathbf{R}\boldsymbol{\sigma}\mathbf{R}^T$ |
| 距离 | 不变量(标量) | 不变 | $r' = r$ |
| 位移向量 | 等变量(向量) | 旋转 | $\mathbf{d}' = \mathbf{R}\mathbf{d}$ |

### 为什么需要等变性？

**1. 物理正确性**：
- 预测的力必须随坐标系旋转而旋转
- 违反等变性 = 违反物理定律

**2. 数据效率**：
- 等变模型自动利用对称性
- 相同数据下学习更好

**3. 泛化能力**：
- 在不同方向上的泛化
- 减少对数据增强的需求

**示例**：预测力

**不等变模型**：
```python
# 需要数据增强
for rotation in all_rotations:
    augmented_data.append(rotate(data, rotation))
```

**等变模型**：
```python
# 自动处理所有旋转
force = equivariant_model(coords)
# 无需数据增强!
```

---

## 等变性和不变性的区别

### 定义对比

**不变性(Invariance)**：
$$
f(g \cdot x) = f(x)
$$
输出**不随**输入变换而改变。

**等变性(Equivariance)**：
$$
f(g \cdot x) = g \cdot f(x)
$$
输出**协变地**随输入变换。

### 几何直观

**旋转不变性**：

```
输入:  ●→  旋转  →  ●↑
       ↓           ↓
输出:  5          5  (相同)
```

能量预测是不变的。

**旋转等变性**：

```
输入:  ●→  旋转  →  ●↑
       ↓           ↓
输出:  →          ↑  (跟随旋转)
```

力预测是等变的。

### 数学性质对比

| 性质 | 不变性 | 等变性 |
|-----|-------|--------|
| **定义** | $f(Tx) = f(x)$ | $f(Tx) = Tf(x)$ |
| **输出类型** | 标量 | 向量/张量 |
| **信息保留** | 丢失方向信息 | 保留方向信息 |
| **应用** | 分类、能量预测 | 力预测、向量场 |
| **复杂度** | 较低 | 较高 |

### 层次关系

**等变性 → 不变性**：

可以通过等变特征构造不变量：
$$
\text{不变量} = \|\text{等变量}\|
$$

例如：
- 位移向量（等变）→ 距离（不变）
- 力（等变）→ 力的大小（不变）

但反过来不行！

### 在神经网络中的实现

**不变神经网络**（SchNet）：

```python
# 节点特征（不变量）
h_i = scalar_features  # (num_nodes, hidden_dim)

# 输出（标量）
energy = sum(h_i)
```

**等变神经网络**（PaiNN, NequIP）：

```python
# 节点特征（等变量 + 不变量）
s_i = scalar_features    # (num_nodes, hidden_dim)
v_i = vector_features    # (num_nodes, hidden_dim, 3)

# 输出（向量）
forces = -grad(energy, coords)
```

### 表达能力

**等变网络 > 不变网络**

**原因**：
- 等变网络保留了更多信息（方向）
- 可以同时预测标量和向量
- 对于相同的准确度，需要更少的参数

**实验证据**：
- PaiNN vs SchNet: 参数减少50%，精度提升30%
- NequIP: 在MD17数据集上达到SOTA

---

## 理解等变性——群论的初步介绍

### 什么是群？

**群(Group)**是一个集合 $G$ 配备一个二元运算 $\cdot$，满足：

1. **封闭性**：$\forall g, h \in G, g \cdot h \in G$
2. **结合律**：$(g \cdot h) \cdot k = g \cdot (h \cdot k)$
3. **单位元**：$\exists e \in G, e \cdot g = g \cdot e = g$
4. **逆元**：$\forall g \in G, \exists g^{-1}, g \cdot g^{-1} = e$

### 与物理相关的群

#### 1. 旋转群 SO(3)

**定义**：三维空间的所有旋转构成的群。

**元素**：旋转矩阵 $\mathbf{R} \in \mathbb{R}^{3 \times 3}$，满足：
- $\mathbf{R}^T\mathbf{R} = \mathbf{I}$
- $\det(\mathbf{R}) = +1$

**应用**：旋转等变性

#### 2. 欧几里得群 E(3)

**定义**：三维空间的所有刚体变换（旋转+平移）。

**元素**：$(\mathbf{R}, \mathbf{t})$，作用为：
$$
\mathbf{x} \mapsto \mathbf{R}\mathbf{x} + \mathbf{t}
$$

**应用**：旋转和平移等变性

#### 3. 置换群 S_N

**定义**：$N$ 个元素的所有置换。

**应用**：原子交换不变性

### 群表示理论

**表示(Representation)**：群元素到矩阵的同态映射。

$$
\rho: G \rightarrow GL(V)
$$

满足：
$$
\rho(g \cdot h) = \rho(g) \rho(h)
$$

#### 不可约表示(Irreducible Representations)

**SO(3)的不可约表示**：

由角动量量子数 $l = 0, 1, 2, ...$ 标记：

- $l = 0$：标量（1维）
- $l = 1$：向量（3维）
- $l = 2$：秩2张量（5维，对称无迹）
- ...

**维度**：$2l + 1$

**球谐函数**：
$$
Y_l^m(\theta, \phi), \quad m = -l, ..., +l
$$
是 $l$ 阶不可约表示的基。

### 等变性的群论表述

**定义**：函数 $f: V \rightarrow W$ 等变于群 $G$，如果：

$$
\rho^W(g) \circ f = f \circ \rho^V(g), \quad \forall g \in G
$$

其中 $\rho^V$, $\rho^W$ 是 $G$ 在 $V$, $W$ 上的表示。

**例子**：

旋转等变的向量场：
$$
\mathbf{F}(\mathbf{R}\mathbf{r}) = \mathbf{R}\mathbf{F}(\mathbf{r})
$$

这里 $\rho^V(\mathbf{R}) = \mathbf{R}$（向量表示），$\rho^W(\mathbf{R}) = \mathbf{R}$（向量表示）。

### 张量积和Clebsch-Gordan系数

**张量积**：

两个不可约表示的张量积可以分解为不可约表示的直和：

$$
l_1 \otimes l_2 = \bigoplus_{l=|l_1-l_2|}^{l_1+l_2} l
$$

**例子**：
$$
\text{vector} \otimes \text{vector} = \text{scalar} \oplus \text{vector} \oplus \text{rank-2 tensor}
$$
$$
1 \otimes 1 = 0 \oplus 1 \oplus 2
$$

**Clebsch-Gordan系数**：用于计算张量积。

这在NequIP等模型中用于组合不同阶的特征。

### 等变卷积

**普通卷积**（CNN）：

$$
(f \star g)(x) = \int f(y) g(x - y) dy
$$

平移等变，但不旋转等变。

**等变卷积**（球面卷积）：

$$
(f \star g)(x) = \int_{SO(3)} f(R^{-1}x) g(R) dR
$$

旋转等变。

---

## 等变消息传递神经网络和不变消息传递神经网络的对比

### 架构对比

#### 不变MPNN（SchNet）

**节点特征**：标量
$$
\mathbf{h}_i \in \mathbb{R}^d
$$

**消息**：
$$
\mathbf{m}_{ij} = \mathbf{h}_j \odot \text{FilterNet}(r_{ij})
$$

**更新**：
$$
\mathbf{h}_i^{new} = \mathbf{h}_i + \text{Update}\left(\sum_j \mathbf{m}_{ij}\right)
$$

**输出**：
$$
E = \sum_i \text{OutputNet}(\mathbf{h}_i)
$$

#### 等变MPNN（PaiNN, NequIP）

**节点特征**：标量 + 向量
$$
\mathbf{s}_i \in \mathbb{R}^d, \quad \mathbf{V}_i \in \mathbb{R}^{d \times 3}
$$

**消息**：
$$
\begin{aligned}
\Delta \mathbf{s}_i &= f_s(\mathbf{s}_i, \mathbf{s}_j, r_{ij}) \\
\Delta \mathbf{V}_i &= f_V(\mathbf{s}_i, \mathbf{s}_j, \mathbf{V}_j, \mathbf{r}_{ij})
\end{aligned}
$$

**更新**（保持等变性）：
$$
\begin{aligned}
\mathbf{s}_i^{new} &= \mathbf{s}_i + \Delta \mathbf{s}_i \\
\mathbf{V}_i^{new} &= \mathbf{V}_i + \Delta \mathbf{V}_i
\end{aligned}
$$

### 信息传递对比

**不变MPNN**：

```
节点i: [s₁, s₂, ..., sₐ]  (标量)
         ↓
邻居信息: Σⱼ msg(sⱼ, rᵢⱼ)
         ↓
更新: [s₁', s₂', ..., sₐ']
```

只传递标量信息，**丢失方向**。

**等变MPNN**：

```
节点i: [s₁, s₂, ...]      (标量)
       [v₁, v₂, ...]      (向量, 每个3D)
         ↓
邻居信息: Σⱼ msg(sⱼ, vⱼ, rᵢⱼ, r̂ᵢⱼ)
         ↓
更新: [s₁', s₂', ...]
      [v₁', v₂', ...]
```

同时传递标量和向量，**保留方向**。

### 性能对比

#### 准确性

在MD17数据集上（力MAE，kcal/mol/Å）：

| 模型 | 阿司匹林 | 乙醇 | 马来酸 | 萘 | 水杨酸 |
|------|---------|------|--------|-----|--------|
| SchNet | 0.38 | 0.29 | 0.39 | 0.15 | 0.29 |
| PaiNN | 0.13 | 0.12 | 0.17 | 0.07 | 0.13 |
| NequIP | **0.08** | **0.05** | **0.10** | **0.04** | **0.08** |

等变模型显著更准确！

#### 数据效率

达到相同精度所需的训练样本数：

```
SchNet:   ████████████████████  (1000样本)
PaiNN:    ██████████            (500样本)
NequIP:   ██████                (300样本)
```

等变模型数据效率更高。

#### 参数效率

相同精度下的参数数量：

```
SchNet:   ████████████████  (1M参数)
PaiNN:    ████████          (500K参数)
NequIP:   ██████            (300K参数)
```

### 计算复杂度

| 模型 | 前向传播 | 反向传播 | 内存 |
|------|---------|---------|------|
| SchNet | O(N × E × d) | 2× | 基准 |
| PaiNN | O(N × E × d) | 2.5× | 1.5× |
| NequIP | O(N × E × d × l²) | 3× | 2× |

其中：
- $N$：原子数
- $E$：平均邻居数
- $d$：隐藏维度
- $l$：最大角动量

**权衡**：
- 等变模型计算稍慢，但更准确
- 总体上，由于需要更少的epoch，训练时间相当

### 设计选择

#### 何时使用不变MPNN？

- ✅ 只需预测标量（能量、带隙等）
- ✅ 计算资源受限
- ✅ 大规模系统（>10,000原子）

#### 何时使用等变MPNN？

- ✅ 需要预测向量（力、偶极矩等）
- ✅ 高精度要求
- ✅ 训练数据有限
- ✅ 需要更好的泛化

### 实现复杂度

**不变MPNN**：
```python
# 相对简单
h = embedding(z)
for layer in layers:
    messages = []
    for j in neighbors(i):
        m = h[j] * filter_net(r_ij)
        messages.append(m)
    h[i] = h[i] + sum(messages)
```

**等变MPNN**：
```python
# 更复杂
s, V = embedding(z)
for layer in layers:
    # 标量消息
    delta_s = scalar_message(s, r_ij)

    # 向量消息（保持等变性）
    delta_V = vector_message(s, V, r_ij, r_hat_ij)

    s = s + delta_s
    V = V + delta_V
```

需要仔细设计以保持等变性！

---

## 小结

1. **等变性**是输出协变地随输入变换的性质
2. **等变性 > 不变性**：保留更多信息，物理更正确
3. **群论**提供了理解和构造等变网络的数学框架
4. **等变MPNN**在准确性、数据效率和参数效率上都优于不变MPNN

---

## 扩展阅读

1. Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks. *ICML*.
2. Thomas, N., et al. (2018). Tensor field networks: Rotation-and translation-equivariant neural networks for 3D point clouds. *arXiv*.
3. Kondor, R. (2018). N-body networks: a covariant hierarchical neural network architecture for learning atomic potentials. *arXiv*.
4. Weiler, M., Geiger, M., et al. (2018). 3D steerable CNNs: Learning rotationally equivariant features in volumetric data. *NeurIPS*.

---

**下一节**: [常见的等变模型——PaiNN、NequIP和Allegro](./02-Equivariant-Models.md)
