# 深度学习基本理论
# Deep Learning Fundamentals

## 目录
- [人工神经网络与万能近似定理](#人工神经网络与万能近似定理)
- [神经元的基本结构与功能](#神经元的基本结构与功能)
- [常见的激活函数及其特点](#常见的激活函数及其特点)
- [前向传播与反向传播的基本原理](#前向传播与反向传播的基本原理)

---

## 人工神经网络与万能近似定理

### 什么是人工神经网络？

人工神经网络（Artificial Neural Network, ANN）是一种受生物神经系统启发的计算模型，通过模拟神经元之间的连接和信号传递来处理信息。神经网络由大量简单的处理单元（神经元）组成，这些单元通过加权连接相互作用。

**核心思想**：
- 神经网络通过学习调整连接权重来完成特定任务
- 网络可以从数据中自动提取特征和模式
- 具有强大的非线性映射能力

### 万能近似定理 (Universal Approximation Theorem)

**定理内容**：一个前馈神经网络，如果具有至少一个隐藏层，并且隐藏层包含足够多的神经元，使用非线性激活函数，那么它可以以任意精度近似任何连续函数。

**数学表述**：
对于任意连续函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$，和任意 $\epsilon > 0$，存在一个单隐藏层神经网络 $F$，使得：

$$
\sup_{x \in K} \|f(x) - F(x)\| < \epsilon
$$

其中 $K$ 是 $\mathbb{R}^n$ 的紧子集。

**物理意义**：
在第一性原理计算中，势能面（Potential Energy Surface, PES）是原子坐标的连续函数：

$$
E = f(\mathbf{R}_1, \mathbf{R}_2, ..., \mathbf{R}_N)
$$

万能近似定理保证了神经网络理论上可以学习任意复杂的势能面，这是神经网络势函数的理论基础。

**重要说明**：
- 定理只保证**存在性**，不保证**可学习性**
- 实际应用中需要合适的网络结构、训练算法和充足的数据
- 深度网络（多层）通常比宽网络（单层多神经元）更高效

---

## 神经元的基本结构与功能

### 生物神经元 vs 人工神经元

**生物神经元**：
- 树突（Dendrites）：接收信号
- 细胞体（Soma）：处理信号
- 轴突（Axon）：传递信号
- 突触（Synapse）：神经元间的连接

**人工神经元（感知机）**：

```
输入: x₁, x₂, ..., xₙ
       ↓
    加权求和
       ↓
  z = Σ wᵢxᵢ + b
       ↓
   激活函数 σ
       ↓
  输出: a = σ(z)
```

### 数学描述

单个神经元的数学表达式：

$$
a = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b)
$$

其中：
- $\mathbf{x} = [x_1, x_2, ..., x_n]^T$：输入向量
- $\mathbf{w} = [w_1, w_2, ..., w_n]^T$：权重向量
- $b$：偏置（bias）
- $\sigma$：激活函数
- $a$：输出（激活值）

### 全连接层（Fully Connected Layer）

对于包含 $m$ 个神经元的层：

$$
\mathbf{a} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})
$$

其中：
- $\mathbf{W} \in \mathbb{R}^{m \times n}$：权重矩阵
- $\mathbf{b} \in \mathbb{R}^m$：偏置向量
- $\mathbf{a} \in \mathbb{R}^m$：输出向量

**在第一性原理计算中的应用**：
- 输入：原子的局部化学环境描述符
- 输出：原子的能量贡献或嵌入向量
- 权重和偏置通过训练从DFT数据中学习

---

## 常见的激活函数及其特点

激活函数引入非线性，使神经网络能够学习复杂的函数关系。

### 1. Sigmoid 函数

**定义**：
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**特点**：
- 输出范围：(0, 1)
- 平滑、可导
- **缺点**：梯度消失问题、输出不以零为中心

**导数**：
$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

### 2. Tanh 函数

**定义**：
$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

**特点**：
- 输出范围：(-1, 1)
- 零中心化，优于 Sigmoid
- **缺点**：仍存在梯度消失问题

**导数**：
$$
\tanh'(z) = 1 - \tanh^2(z)
$$

### 3. ReLU (Rectified Linear Unit)

**定义**：
$$
\text{ReLU}(z) = \max(0, z) = \begin{cases}
z & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

**特点**：
- 计算简单、高效
- 缓解梯度消失问题
- 稀疏激活（约50%的神经元被激活）
- **缺点**："神经元死亡"问题（dying ReLU）

**导数**：
$$
\text{ReLU}'(z) = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

### 4. Leaky ReLU

**定义**：
$$
\text{LeakyReLU}(z) = \begin{cases}
z & \text{if } z > 0 \\
\alpha z & \text{if } z \leq 0
\end{cases}
$$

其中 $\alpha$ 通常取 0.01。

**特点**：
- 解决 ReLU 的"神经元死亡"问题
- 负值区域有小梯度

### 5. ELU (Exponential Linear Unit)

**定义**：
$$
\text{ELU}(z) = \begin{cases}
z & \text{if } z > 0 \\
\alpha(e^z - 1) & \text{if } z \leq 0
\end{cases}
$$

**特点**：
- 平滑、连续
- 输出均值接近零
- 计算成本比 ReLU 高

### 6. Softplus

**定义**：
$$
\text{Softplus}(z) = \ln(1 + e^z)
$$

**特点**：
- ReLU 的平滑版本
- 在神经网络势函数中常用（如 SchNet）
- 保证输出连续可导

### 激活函数选择建议

| 应用场景 | 推荐激活函数 |
|---------|------------|
| 隐藏层（一般） | ReLU, Leaky ReLU |
| 深度网络 | ReLU, ELU |
| 二分类输出层 | Sigmoid |
| 多分类输出层 | Softmax |
| 回归输出层 | 线性（无激活） |
| 神经网络势函数 | Softplus, Tanh |

---

## 前向传播与反向传播的基本原理

### 前向传播 (Forward Propagation)

前向传播是从输入层开始，逐层计算直到输出层的过程。

**多层网络示例**（3层网络）：

**第1层（输入层到隐藏层1）**：
$$
\mathbf{z}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}
$$
$$
\mathbf{a}^{[1]} = \sigma^{[1]}(\mathbf{z}^{[1]})
$$

**第2层（隐藏层1到隐藏层2）**：
$$
\mathbf{z}^{[2]} = \mathbf{W}^{[2]}\mathbf{a}^{[1]} + \mathbf{b}^{[2]}
$$
$$
\mathbf{a}^{[2]} = \sigma^{[2]}(\mathbf{z}^{[2]})
$$

**第3层（隐藏层2到输出层）**：
$$
\mathbf{z}^{[3]} = \mathbf{W}^{[3]}\mathbf{a}^{[2]} + \mathbf{b}^{[3]}
$$
$$
\hat{\mathbf{y}} = \mathbf{a}^{[3]} = \sigma^{[3]}(\mathbf{z}^{[3]})
$$

### 损失函数 (Loss Function)

**均方误差（MSE）** - 用于回归问题（如能量预测）：
$$
L(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|^2 = \frac{1}{2}\sum_{i}(y_i - \hat{y}_i)^2
$$

**交叉熵损失** - 用于分类问题：
$$
L(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i} y_i \log(\hat{y}_i)
$$

### 反向传播 (Backpropagation)

反向传播使用链式法则计算损失函数对每个参数的梯度。

**核心思想**：从输出层开始，逐层向输入层传播误差。

**链式法则**：
$$
\frac{\partial L}{\partial w_{ij}^{[l]}} = \frac{\partial L}{\partial a^{[l]}} \cdot \frac{\partial a^{[l]}}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial w_{ij}^{[l]}}
$$

**反向传播算法步骤**：

1. **输出层误差**：
$$
\delta^{[L]} = \frac{\partial L}{\partial \mathbf{z}^{[L]}} = (\mathbf{a}^{[L]} - \mathbf{y}) \odot \sigma'^{[L]}(\mathbf{z}^{[L]})
$$

2. **隐藏层误差**（从第 $L-1$ 层到第 1 层）：
$$
\delta^{[l]} = (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \odot \sigma'^{[l]}(\mathbf{z}^{[l]})
$$

3. **权重梯度**：
$$
\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \delta^{[l]} (\mathbf{a}^{[l-1]})^T
$$

4. **偏置梯度**：
$$
\frac{\partial L}{\partial \mathbf{b}^{[l]}} = \delta^{[l]}
$$

### 梯度下降优化

**参数更新**：
$$
\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \frac{\partial L}{\partial \mathbf{W}^{[l]}}
$$
$$
\mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \frac{\partial L}{\partial \mathbf{b}^{[l]}}
$$

其中 $\eta$ 是学习率（learning rate）。

### 在神经网络势函数中的应用

**目标**：最小化预测能量和DFT计算能量之间的差异

$$
L = \sum_{i=1}^{N} \left(E_i^{\text{DFT}} - E_i^{\text{NN}}\right)^2 + \lambda \sum_{i=1}^{N} \|\mathbf{F}_i^{\text{DFT}} - \mathbf{F}_i^{\text{NN}}\|^2
$$

其中：
- $E^{\text{DFT}}$, $E^{\text{NN}}$：DFT能量和神经网络预测能量
- $\mathbf{F}^{\text{DFT}}$, $\mathbf{F}^{\text{NN}}$：DFT力和神经网络预测力
- $\lambda$：力的权重系数

**力的计算**：
$$
\mathbf{F}_i^{\text{NN}} = -\frac{\partial E^{\text{NN}}}{\partial \mathbf{R}_i}
$$

通过自动微分（automatic differentiation）实现，PyTorch等框架自动处理。

---

## 小结

1. **万能近似定理**为神经网络学习复杂函数提供了理论保证
2. **神经元**是神经网络的基本单元，通过加权求和和激活函数处理信息
3. **激活函数**引入非线性，ReLU及其变体在现代深度学习中最常用
4. **前向传播**计算网络输出，**反向传播**计算梯度并更新参数

---

## 扩展阅读

1. Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural networks*, 2(5), 359-366.
2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

**下一节**: [深度学习的发展历程和优势](./02-Deep-Learning-Development.md)
