# 从描述符到图表示：图神经网络和MPNN消息传递神经网络
# From Descriptors to Graph Representations: Graph Neural Networks and MPNN

## 目录
- [具有不变性的消息传递神经网络](#具有不变性的消息传递神经网络)
- [晶体图卷积神经网络CGCNN](#晶体图卷积神经网络cgcnn)
- [消息传递神经网络的一般框架和组成](#消息传递神经网络的一般框架和组成)

---

## 具有不变性的消息传递神经网络

### 为什么需要图神经网络？

**原子结构的天然图表示**：

```
原子 = 节点(Nodes)
化学键/邻近关系 = 边(Edges)
```

**传统描述符的局限**：
- ACSF：手工设计，维度高
- SOAP：计算成本高
- 固定邻居数，不灵活

**图神经网络的优势**：
- ✅ 自动学习表征
- ✅ 端到端训练
- ✅ 灵活处理任意大小和结构
- ✅ 自然满足置换不变性

### 图的数学定义

**图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$**：
- $\mathcal{V} = \{v_1, v_2, ..., v_N\}$：节点集合
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$：边集合

**原子系统的图表示**：
- **节点特征** $\mathbf{h}_i^{(0)}$：原子类型（one-hot编码或嵌入）
- **边特征** $\mathbf{e}_{ij}$：原子间距离 $r_{ij}$、相对位置 $\mathbf{r}_{ij}$
- **全局特征** $\mathbf{u}$：晶胞参数、总电荷等

### 不变性和等变性

**置换不变性(Permutation Invariance)**：

对于任意置换 $\pi$：
$$
f(\{\mathbf{r}_{\pi(1)}, ..., \mathbf{r}_{\pi(N)}\}) = f(\{\mathbf{r}_1, ..., \mathbf{r}_N\})
$$

**旋转不变性(Rotational Invariance)**：

对于任意旋转矩阵 $\mathbf{R} \in SO(3)$：
$$
f(\{\mathbf{R}\mathbf{r}_1, ..., \mathbf{R}\mathbf{r}_N\}) = f(\{\mathbf{r}_1, ..., \mathbf{r}_N\})
$$

**等变性(Equivariance)**：

输出随输入变换：
$$
f(\mathbf{R}\mathbf{x}) = \mathbf{R}f(\mathbf{x})
$$

**例子**：
- **不变量**：距离 $r_{ij} = \|\mathbf{r}_j - \mathbf{r}_i\|$
- **等变量**：力 $\mathbf{F}_i$（向量）

### 消息传递的基本思想

**核心理念**：节点通过边传递信息，逐层更新特征。

**单层消息传递**：

```
   节点i            邻居j
    h_i  ←───msg──  h_j
         └─agg─┘
           ↓
         update
           ↓
        h_i^new
```

**数学表达**：

1. **消息构建**：
$$
\mathbf{m}_{ij}^{(l)} = \phi_{\text{msg}}(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ij})
$$

2. **消息聚合**：
$$
\mathbf{m}_i^{(l)} = \bigoplus_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(l)}
$$

其中 $\bigoplus$ 可以是求和、取最大值、平均等**置换不变**操作。

3. **节点更新**：
$$
\mathbf{h}_i^{(l+1)} = \phi_{\text{update}}(\mathbf{h}_i^{(l)}, \mathbf{m}_i^{(l)})
$$

### 保证不变性的设计

**置换不变性**：
- 通过**求和**或**平均**聚合消息
$$
\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}  \quad \text{或} \quad \mathbf{m}_i = \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}
$$

**旋转不变性**：
- 只使用**标量**特征（距离、角度）
- 或使用**球谐函数**等等变表示

**平移不变性**：
- 使用**相对坐标** $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$
- 而非绝对坐标 $\mathbf{r}_i$

---

## 晶体图卷积神经网络CGCNN

### CGCNN简介

**Crystal Graph Convolutional Neural Networks** (CGCNN) 由Tian Xie等人于2018年提出，专门用于晶体材料性质预测。

**论文**：*Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties*, PRL 2018

### CGCNN架构

```
晶体结构
    ↓
构建晶体图 (节点=原子, 边=化学键)
    ↓
节点特征初始化 (原子类型 → 嵌入向量)
边特征初始化 (键长 → 高斯展开)
    ↓
多层卷积层 (消息传递 + 更新)
    ↓
池化 (节点特征 → 图级别特征)
    ↓
全连接层
    ↓
输出 (形成能、带隙等)
```

### 晶体图构建

**节点**：晶体中的原子

**边**：
- 考虑截断半径 $r_c$（通常8-12 Å）内的所有原子对
- 包括周期性边界条件下的镜像原子

**边特征**：
$$
\mathbf{e}_{ij} = g(r_{ij}) = [g_1(r_{ij}), g_2(r_{ij}), ..., g_K(r_{ij})]
$$

其中 $g_k$ 是高斯基函数：
$$
g_k(r) = \exp\left(-\frac{(r - \mu_k)^2}{2\sigma^2}\right)
$$

$\mu_k$ 均匀分布在 $[0, r_c]$。

### CGCNN卷积层

**卷积操作**（单层）：

$$
\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \sigma\left(\mathbf{W}_f^{(l)} \mathbf{z}_{ij}^{(l)} + \mathbf{b}_f^{(l)}\right) \odot g\left(\mathbf{W}_s^{(l)} \mathbf{z}_{ij}^{(l)} + \mathbf{b}_s^{(l)}\right)
$$

其中：
- $\mathbf{z}_{ij}^{(l)} = [\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ij}]$：拼接特征
- $\sigma$：Sigmoid函数（门控）
- $g$：Softplus函数
- $\odot$：逐元素乘法
- 残差连接：$\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + ...$

**物理解释**：
- Sigmoid门控：控制信息流（类似LSTM）
- Softplus：确保平滑性
- 残差连接：帮助训练深层网络

### 池化层(Readout)

从节点特征得到图级别特征：

$$
\mathbf{h}_{\text{graph}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{h}_i^{(L)}
$$

平均池化保证了**尺寸可扩展性**(size extensivity)。

### CGCNN完整流程

**伪代码**：

```python
# 1. 初始化
h_i = Embedding(atom_type_i)  # 节点特征
e_ij = GaussianExpansion(r_ij)  # 边特征

# 2. 卷积层(多层)
for layer in range(num_layers):
    for i in nodes:
        messages = []
        for j in neighbors(i):
            z_ij = concat(h_i, h_j, e_ij)
            gate = sigmoid(W_f @ z_ij + b_f)
            filter = softplus(W_s @ z_ij + b_s)
            messages.append(gate * filter)

        h_i_new = h_i + sum(messages)

    h_i = h_i_new

# 3. 池化
h_graph = mean(h_i for i in nodes)

# 4. 预测
output = MLP(h_graph)
```

### CGCNN的应用

**1. 形成能预测**：
- 训练数据：Materials Project数据库
- 准确度：MAE ~0.04 eV/atom

**2. 带隙预测**：
- 分类：金属 vs 半导体
- 回归：带隙数值

**3. 体模量预测**：
- 力学性质
- 与实验吻合良好

**4. 新材料筛选**：
- 快速预测大量候选材料
- 发现新型电池材料、催化剂等

---

## 消息传递神经网络的一般框架和组成

### MPNN通用框架

Gilmer等人(2017)提出了消息传递神经网络(MPNN)的统一框架。

**核心组件**：

1. **消息函数** $M_t$
2. **更新函数** $U_t$
3. **读出函数** $R$

### 完整MPNN算法

**输入**：
- 初始节点特征 $\mathbf{h}_i^{(0)}$
- 边特征 $\mathbf{e}_{ij}$
- 图结构 $\mathcal{G}$

**消息传递阶段**（$T$ 层）：

For $t = 1$ to $T$:
1. **消息**：
$$
\mathbf{m}_i^{(t+1)} = \sum_{j \in \mathcal{N}(i)} M_t(\mathbf{h}_i^{(t)}, \mathbf{h}_j^{(t)}, \mathbf{e}_{ij})
$$

2. **更新**：
$$
\mathbf{h}_i^{(t+1)} = U_t(\mathbf{h}_i^{(t)}, \mathbf{m}_i^{(t+1)})
$$

**读出阶段**：

$$
\hat{y} = R(\{\mathbf{h}_i^{(T)} | i \in \mathcal{V}\})
$$

### 不同模型的MPNN实例化

| 模型 | 消息函数 $M_t$ | 更新函数 $U_t$ | 读出函数 $R$ |
|------|--------------|--------------|-------------|
| **GCN** | $\mathbf{W}\mathbf{h}_j$ | $\text{ReLU}(\sum_j \mathbf{m}_{ij})$ | $\text{Softmax}(\mathbf{h}_i)$ |
| **GraphSAGE** | $\mathbf{h}_j$ | $\mathbf{W}[\mathbf{h}_i, \text{mean}_j(\mathbf{m}_j)]$ | 池化 |
| **GAT** | $\alpha_{ij}\mathbf{W}\mathbf{h}_j$ | $\text{ELU}(\sum_j \mathbf{m}_{ij})$ | 平均 |
| **CGCNN** | Gate$(\mathbf{h}_i, \mathbf{h}_j, \mathbf{e}_{ij})$ | $\mathbf{h}_i + \sum_j \mathbf{m}_{ij}$ | 平均 |
| **SchNet** | $\phi_{\text{filter}}(r_{ij})\mathbf{h}_j$ | $\mathbf{h}_i + \text{Dense}(\sum_j \mathbf{m}_{ij})$ | $\sum_i \mathbf{h}_i$ |

### 关键设计选择

#### 1. 消息聚合方式

**求和**：
$$
\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}
$$
- 优点：保留信息、可区分不同度数节点
- 缺点：可能数值不稳定

**平均**：
$$
\mathbf{m}_i = \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}
$$
- 优点：归一化、稳定
- 缺点：丢失邻居数量信息

**最大值**：
$$
\mathbf{m}_i = \max_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}
$$
- 优点：关注最重要邻居
- 缺点：丢失大量信息

**注意力加权**（GAT）：
$$
\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{m}_{ij}
$$
其中 $\alpha_{ij}$ 是学习的注意力权重。

#### 2. 边特征的利用

**方法1**：直接拼接
$$
M(\mathbf{h}_i, \mathbf{h}_j, \mathbf{e}_{ij}) = \phi([\mathbf{h}_i, \mathbf{h}_j, \mathbf{e}_{ij}])
$$

**方法2**：边网络(Edge Network)
$$
\mathbf{W}_{ij} = \text{EdgeNet}(\mathbf{e}_{ij})
$$
$$
\mathbf{m}_{ij} = \mathbf{W}_{ij} \mathbf{h}_j
$$

**方法3**：连续卷积(SchNet)
$$
\mathbf{m}_{ij} = \mathbf{h}_j \odot \text{FilterNet}(\mathbf{e}_{ij})
$$

#### 3. 残差连接和归一化

**残差连接**（ResNet风格）：
$$
\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \text{Update}(\mathbf{h}_i^{(l)}, \mathbf{m}_i^{(l)})
$$

**层归一化**：
$$
\mathbf{h}_i^{(l+1)} = \text{LayerNorm}(\mathbf{h}_i^{(l+1)})
$$

**批归一化**：
$$
\mathbf{h}_i^{(l+1)} = \text{BatchNorm}(\mathbf{h}_i^{(l+1)})
$$

### MPNN的表达能力

**Weisfeiler-Lehman (WL) 测试**：

MPNN的表达能力最多等价于WL图同构测试。

**限制**：
- 无法区分某些非同构图（如正则图）
- 解决方案：
  - 更高阶的GNN（k-WL GNN）
  - 添加位置编码
  - 使用子图方法

### MPNN在材料科学中的优势

**1. 自然表示**：
- 原子 = 节点
- 化学键 = 边
- 无需手工设计描述符

**2. 端到端学习**：
- 从原子坐标到性质
- 自动提取特征

**3. 可解释性**：
- 可视化注意力权重
- 理解哪些原子/键重要

**4. 泛化能力**：
- 学到的表征可迁移
- 预训练-微调范式

---

## 小结

1. **图神经网络**是处理原子结构的自然选择
2. **CGCNN**成功应用于晶体性质预测
3. **MPNN框架**统一了各种图神经网络
4. **不变性设计**是构建物理正确模型的关键

---

## 扩展阅读

1. Gilmer, J., et al. (2017). Neural message passing for quantum chemistry. *ICML*.
2. Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. *Physical Review Letters*.
3. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
4. Veličković, P., et al. (2018). Graph attention networks. *ICLR*.
5. Battaglia, P. W., et al. (2018). Relational inductive biases, deep learning, and graph networks. *arXiv*.

---

**上一节**: [分子动力学模拟](./01-Molecular-Dynamics.md)
**下一节**: [SchNet和DimeNet++介绍](./03-SchNet-DimeNet.md)
