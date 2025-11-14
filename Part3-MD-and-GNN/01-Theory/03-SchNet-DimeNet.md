# SchNet和DimeNet++等不变消息传递神经网络
# SchNet, DimeNet++ and Invariant Message Passing Neural Networks

## 目录
- [SchNet和DimeNet++的特点](#schnet和dimenet的特点)
- [DimeNet++中角度信息的引入——球谐基函数](#dimenet中角度信息的引入球谐基函数)
- [深入理解消息传递过程——图卷积](#深入理解消息传递过程图卷积)
- [图卷积和图像卷积的区别](#图卷积和图像卷积的区别)

---

## SchNet和DimeNet++的特点

### SchNet: Continuous-filter Convolutional Neural Networks

**SchNet**由Schütt等人于2017年提出，是第一个成功的端到端深度学习势函数模型。

**论文**：*SchNet: A continuous-filter convolutional neural network for modeling quantum interactions*, NeurIPS 2017

#### SchNet架构

```
原子坐标和类型
    ↓
原子嵌入 (Atom Embedding)
    ↓
交互层 (Interaction Blocks) ×3-6
    ├─ 连续滤波器卷积
    ├─ 原子特征更新
    └─ 残差连接
    ↓
输出层 (Atom-wise Output)
    ↓
能量 = Σ 原子能量
```

#### 核心创新：连续滤波器卷积

**传统卷积**（图像）：
$$
h_i^{new} = \sum_j W_{ij} h_j
$$
权重 $W_{ij}$ 是离散的（固定）。

**连续滤波器卷积**（SchNet）：
$$
\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(l)} \odot \mathbf{W}(r_{ij})
$$

其中 $\mathbf{W}(r_{ij})$ 是距离的**连续函数**（通过神经网络学习）。

#### 滤波器生成网络

$$
\mathbf{W}(r) = \text{FilterNet}(r) = \text{Dense}(\text{RBF}(r))
$$

**径向基函数(RBF)展开**：
$$
\text{RBF}(r) = [\exp(-\gamma(r - \mu_1)^2), \exp(-\gamma(r - \mu_2)^2), ..., \exp(-\gamma(r - \mu_K)^2)]
$$

$\mu_k$ 均匀分布在 $[0, r_c]$，通常 $K=20-50$。

#### SchNet交互层

```python
class InteractionBlock(nn.Module):
    def __init__(self, n_atom_basis, n_filters):
        super().__init__()
        self.filter_net = nn.Sequential(
            Dense(n_rbf, n_filters),
            ShiftedSoftplus(),
            Dense(n_filters, n_filters)
        )
        self.cfconv = CFConv(n_atom_basis, n_filters)
        self.dense = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis),
            ShiftedSoftplus(),
            Dense(n_atom_basis, n_atom_basis)
        )

    def forward(self, h, r_ij, neighbors):
        # 1. 生成滤波器
        W = self.filter_net(rbf_expansion(r_ij))

        # 2. 连续滤波器卷积
        v = self.cfconv(h, W, neighbors)

        # 3. 原子更新
        v = self.dense(v)

        # 4. 残差连接
        h_new = h + v

        return h_new
```

#### SchNet输出层

**原子能量**：
$$
E_i = \text{OutputNet}(\mathbf{h}_i^{(L)})
$$

**总能量**：
$$
E = \sum_{i=1}^{N} E_i
$$

**力**（自动微分）：
$$
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}
$$

#### SchNet的优点和局限

**优点**：
- ✅ 端到端学习
- ✅ 连续、平滑、可导
- ✅ 自动满足平移、旋转、置换不变性
- ✅ 准确度高（接近DFT）

**局限**：
- ❌ 只使用距离信息（径向）
- ❌ 忽略角度信息
- ❌ 对某些体系（如水分子）精度不够

### DimeNet++: Directional Message Passing Neural Network

**DimeNet++**由Klicpera等人于2020年提出，引入角度信息以提高准确度。

**论文**：*Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules*, NeurIPS Workshop 2020

#### DimeNet++的关键改进

**1. 方向性消息传递**：

考虑三元组$(i, j, k)$：
```
    k
   /
  j
 /
i
```

**2. 角度编码**：

$$
\theta_{ijk} = \cos^{-1}\left(\frac{\mathbf{r}_{ij} \cdot \mathbf{r}_{ik}}{r_{ij} r_{ik}}\right)
$$

**3. 球谐基函数(Spherical Bessel Function)**：

用于编码角度信息（见下一节）。

#### DimeNet++架构

```
原子 {r_i, Z_i}
    ↓
嵌入层
    ├─ 原子嵌入 (基于原子序数)
    ├─ 距离嵌入 (RBF展开)
    └─ 角度嵌入 (球谐基函数)
    ↓
交互层 ×6
    ├─ 方向性消息 (考虑角度)
    ├─ 双线性层
    └─ 残差连接
    ↓
输出层
    ↓
能量
```

#### 方向性消息传递

**基于边的更新**（而非基于节点）：

$$
\mathbf{m}_{ij} = \sum_{k \in \mathcal{N}(i) \backslash j} \mathbf{W}(r_{ij}, r_{ik}, \theta_{ijk}) \mathbf{h}_k
$$

**三元组交互**：
- 考虑原子 $i$ 的两个邻居 $j$ 和 $k$ 之间的关系
- 捕获角度依赖性

#### DimeNet++的改进

相比原始DimeNet：
- **更快**：优化的实现，减少计算量
- **更准确**：改进的角度嵌入
- **更稳定**：更好的初始化和归一化

**性能**（QM9数据集）：
- 能量MAE: ~0.5 kcal/mol
- 力MAE: ~1.0 kcal/mol/Å
- 比SchNet准确30-40%

---

## DimeNet++中角度信息的引入——球谐基函数

### 为什么需要角度信息？

**示例**：水分子

```
    H
     \
      O---H
```

- 键长相同：O-H = 0.96 Å
- 键角不同：H-O-H ≈ 104.5°

**SchNet的问题**：
- 只使用距离 → 无法区分不同角度
- 对有方向性的化学键描述不准确

**解决方案**：
- 显式编码角度信息
- 使用球谐函数等数学工具

### 球谐基函数(Spherical Basis Functions)

#### 球谐函数$Y_l^m(\theta, \phi)$

**定义**：
$$
Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-|m|)!}{(l+|m|)!}} P_l^m(\cos\theta) e^{im\phi}
$$

其中：
- $l$：角动量量子数（$l = 0, 1, 2, ...$）
- $m$：磁量子数（$m = -l, ..., +l$）
- $P_l^m$：连带勒让德多项式

**性质**：
- 在球面上正交完备
- 旋转等变
- 广泛用于量子力学

#### 径向基函数 × 球谐函数

完整的三维基函数：
$$
\phi_{nlm}(\mathbf{r}) = R_{nl}(r) Y_l^m(\theta, \phi)
$$

其中 $R_{nl}(r)$ 是径向函数（如球贝塞尔函数）。

#### DimeNet++中的角度嵌入

**步骤1**：计算三元组角度
$$
\cos\theta_{ijk} = \frac{\mathbf{r}_{ij} \cdot \mathbf{r}_{ik}}{r_{ij} r_{ik}}
$$

**步骤2**：球谐展开
$$
\mathbf{a}_{ijk} = [Y_0^0(\theta_{ijk}), Y_1^{-1}(\theta_{ijk}), Y_1^0(\theta_{ijk}), Y_1^1(\theta_{ijk}), ...]
$$

实际中使用简化版本（只依赖$\theta$，不依赖$\phi$）。

**步骤3**：与距离信息结合
$$
\mathbf{e}_{ijk} = \text{MLP}([\text{RBF}(r_{ij}), \text{RBF}(r_{ik}), \text{SBF}(\theta_{ijk})])
$$

其中SBF是球面基函数。

### 球贝塞尔函数(Spherical Bessel Function)

**定义**：
$$
j_n(x) = (-x)^n \left(\frac{1}{x}\frac{d}{dx}\right)^n \frac{\sin x}{x}
$$

**前几项**：
$$
j_0(x) = \frac{\sin x}{x}
$$
$$
j_1(x) = \frac{\sin x}{x^2} - \frac{\cos x}{x}
$$
$$
j_2(x) = \left(\frac{3}{x^3} - \frac{1}{x}\right)\sin x - \frac{3\cos x}{x^2}
$$

**用途**：
- 角度展开的径向部分
- 在DimeNet++中用于角度嵌入

### 实现示例

```python
import torch
import numpy as np

def spherical_bessel_basis(angles, num_basis=7, cutoff=5.0):
    """
    计算球贝塞尔基函数

    Parameters:
    -----------
    angles : torch.Tensor
        角度值 (弧度)
    num_basis : int
        基函数数量
    """
    # 频率
    frequencies = torch.arange(1, num_basis + 1, dtype=torch.float32) * np.pi

    # 扩展维度
    angles = angles.unsqueeze(-1)  # (N, 1)
    frequencies = frequencies.unsqueeze(0)  # (1, K)

    # 计算 sin(n*pi*angle/cutoff) / angle
    x = frequencies * angles / cutoff
    sbf = torch.sin(x) / angles

    return sbf


def angle_embedding(r_ij, r_ik, r_jk, num_rbf=20, num_sbf=7):
    """
    DimeNet++风格的角度嵌入

    Parameters:
    -----------
    r_ij, r_ik, r_jk : torch.Tensor
        三条边的距离
    """
    # 1. 计算角度（余弦定理）
    cos_theta = (r_ij**2 + r_ik**2 - r_jk**2) / (2 * r_ij * r_ik + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1, 1)
    theta = torch.acos(cos_theta)

    # 2. 球贝塞尔基函数
    sbf = spherical_bessel_basis(theta, num_sbf)

    # 3. 距离的RBF
    rbf_ij = rbf_expansion(r_ij, num_rbf)
    rbf_ik = rbf_expansion(r_ik, num_rbf)

    # 4. 组合
    angle_emb = torch.cat([rbf_ij, rbf_ik, sbf], dim=-1)

    return angle_emb
```

---

## 深入理解消息传递过程——图卷积

### 图卷积的数学定义

**谱域图卷积**（基于图傅里叶变换）：
$$
\mathbf{h}^{(l+1)} = \sigma(\mathbf{U}g_\theta(\mathbf{\Lambda})\mathbf{U}^T\mathbf{h}^{(l)})
$$

其中：
- $\mathbf{L} = \mathbf{D} - \mathbf{A}$：图拉普拉斯矩阵
- $\mathbf{L} = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^T$：特征值分解
- $g_\theta(\mathbf{\Lambda})$：可学习的谱滤波器

**空域图卷积**（基于邻居聚合）：
$$
\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} \mathbf{W}^{(l)} \mathbf{h}_j^{(l)}\right)
$$

### 图卷积网络(GCN)

**简化的GCN层**：
$$
\mathbf{H}^{(l+1)} = \sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(l)}\mathbf{W}^{(l)})
$$

其中：
- $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$：添加自连接
- $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$：度矩阵

**归一化的意义**：
- $\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}$：对称归一化
- 防止不同度数节点的数值差异过大

### 消息传递的直观理解

**迭代聚合邻居信息**：

```
层0: h_i^(0) = 原子嵌入

层1: h_i^(1) = agg(h_j^(0) for j in N(i))
     → 包含1-hop邻居信息

层2: h_i^(2) = agg(h_j^(1) for j in N(i))
     → 包含2-hop邻居信息

...

层L: h_i^(L) = agg(h_j^(L-1) for j in N(i))
     → 包含L-hop邻居信息
```

**感受野(Receptive Field)**：
- L层网络 → 感受野半径 = L × (平均键长)
- 需要足够的层数来捕获长程相互作用

### 过平滑问题(Over-smoothing)

**问题**：随着层数增加，所有节点特征变得相似。

**数学解释**：
$$
\lim_{L \to \infty} \mathbf{h}_i^{(L)} = \mathbf{c} \quad \forall i
$$

所有节点收敛到相同的表示。

**解决方法**：
1. **残差连接**：
$$
\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + f(\mathbf{h}_i^{(l)}, \{\mathbf{h}_j^{(l)}\})
$$

2. **跳跃连接**(Jumping Knowledge):
$$
\mathbf{h}_i^{\text{final}} = \text{Aggregate}(\mathbf{h}_i^{(0)}, \mathbf{h}_i^{(1)}, ..., \mathbf{h}_i^{(L)})
$$

3. **层归一化**

4. **DropEdge**：随机丢弃边

---

## 图卷积和图像卷积的区别

### 图像卷积(Image Convolution)

**规则网格结构**：

```
输入特征图 (H × W × C)
    ↓
卷积核 (K × K × C)
    ↓
滑动窗口，逐位置计算
    ↓
输出特征图 (H' × W' × C')
```

**卷积操作**：
$$
y_{ij} = \sum_{m=-k}^{k} \sum_{n=-k}^{k} w_{mn} \cdot x_{i+m, j+n}
$$

**特点**：
- ✅ 固定邻居数量（9个，对于3×3卷积核）
- ✅ 有序的邻居（上、下、左、右等）
- ✅ 参数共享（同一个卷积核）
- ✅ 平移等变性

### 图卷积(Graph Convolution)

**不规则图结构**：

```
节点i的邻居: {j1, j2, ..., jn}
    ↓
n是可变的（每个节点度数不同）
    ↓
聚合邻居特征
    ↓
更新节点i的特征
```

**聚合操作**：
$$
\mathbf{h}_i^{new} = \text{Update}\left(\mathbf{h}_i, \text{Aggregate}(\{\mathbf{h}_j | j \in \mathcal{N}(i)\})\right)
$$

**特点**：
- ❌ 邻居数量可变
- ❌ 邻居无序（需要置换不变性）
- ✅ 参数共享（跨节点）
- ❌ 无平移概念

### 关键区别总结

| 方面 | 图像卷积 | 图卷积 |
|-----|---------|--------|
| **数据结构** | 规则网格 | 不规则图 |
| **邻居数量** | 固定 | 可变 |
| **邻居顺序** | 有序 | 无序 |
| **权重共享** | 卷积核固定 | 聚合函数固定 |
| **等变性** | 平移等变 | 置换不变 |
| **距离信息** | 隐式（位置） | 显式（边特征） |
| **实现** | 高度优化(GPU) | 需要scatter/gather |

### 为什么原子系统适合图表示？

**1. 自然的图结构**：
- 原子 = 节点
- 化学键/邻近关系 = 边
- 无规则网格约束

**2. 不变性匹配**：
- 物理规律要求旋转、平移、置换不变性
- 图卷积天然满足置换不变性

**3. 灵活性**：
- 可处理任意大小、形状的分子/晶体
- 可变配位数（不同原子邻居数不同）

**4. 效率**：
- 稀疏图 → 只计算存在的边
- 不浪费计算在不相邻的原子对上

### 从CNN到GNN的演化

```
图像识别 (CNN)
    → 平移不变性
    → 局部感受野
    → 参数共享

分子性质预测 (GNN)
    → 置换不变性
    → 局部邻域（化学键）
    → 聚合函数共享
```

**共同点**：
- 利用数据的结构性
- 局部性假设
- 层次化特征提取

**不同点**：
- 对称性要求不同
- 数据结构不同
- 归纳偏置不同

---

## 小结

1. **SchNet**使用连续滤波器卷积，实现端到端学习
2. **DimeNet++**引入角度信息，显著提高准确度
3. **球谐函数**是表示方向信息的数学工具
4. **图卷积**是专门为不规则图结构设计的卷积操作

---

## 扩展阅读

1. Schütt, K. T., et al. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *NeurIPS*.
2. Klicpera, J., et al. (2020). Directional message passing for molecular graphs. *ICLR*.
3. Gasteiger, J., et al. (2020). Fast and uncertainty-aware directional message passing for non-equilibrium molecules. *NeurIPS Workshop*.
4. Batzner, S., et al. (2022). E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*.

---

**上一节**: [图神经网络和MPNN](./02-Graph-Neural-Networks.md)
**返回**: [Part 3主页](../README.md)
