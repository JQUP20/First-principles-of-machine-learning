# 神经网络势函数
# Neural Network Potentials

## 目录
- [从高斯核回归到神经网络势函数](#从高斯核回归到神经网络势函数)
- [神经网络势函数的基本假设](#神经网络势函数的基本假设)
- [原子结构和周围化学环境的表征](#原子结构和周围化学环境的表征)
- [BPNN描述符和DP深度神经网络势函数](#bpnn描述符和dp深度神经网络势函数)

---

## 从高斯核回归到神经网络势函数

### 机器学习势函数的动机

**第一性原理计算（DFT）的局限**：
- **计算成本高**：O(N³) 复杂度，N是电子数
- **系统尺寸受限**：通常<1000原子
- **时间尺度受限**：皮秒级别的分子动力学

**经验势函数的局限**：
- Lennard-Jones势：$V(r) = 4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]$
- 嵌入原子势（EAM）
- **问题**：形式固定，迁移性差，无法描述复杂化学过程

**机器学习势函数的优势**：
- ✅ 接近DFT的精度（meV/atom）
- ✅ 接近经验势的速度（百万倍加速）
- ✅ 可迁移性和泛化能力

### 势能面拟合问题

**目标**：学习势能面（Potential Energy Surface, PES）

$$
E = f(\mathbf{R}_1, \mathbf{R}_2, ..., \mathbf{R}_N)
$$

其中 $\mathbf{R}_i$ 是第 $i$ 个原子的坐标。

**挑战**：
- 高维度：3N维空间
- 复杂性：非线性、多尺度
- 对称性：平移、旋转、置换不变性

### 核方法与高斯过程回归

**高斯过程回归（Gaussian Process Regression, GPR）**：

将能量视为高斯过程：

$$
E \sim \mathcal{GP}(m(\mathbf{R}), k(\mathbf{R}, \mathbf{R}'))
$$

**核函数**（例如高斯核）：
$$
k(\mathbf{R}, \mathbf{R}') = \sigma^2 \exp\left(-\frac{\|\mathbf{R} - \mathbf{R}'\|^2}{2l^2}\right)
$$

**优点**：
- 不确定性量化
- 理论基础扎实

**缺点**：
- 计算复杂度：O(N³)训练，O(N²)预测
- 难以扩展到大数据集
- 核函数设计依赖经验

### 从核方法到神经网络

**相似性**：
- 都是非线性映射
- 都依赖特征表示

**神经网络的优势**：
- 参数数量固定，不随训练集增长
- 预测速度快：O(1)
- 端到端学习特征表示
- 可处理大规模数据

**发展历程**：
- **2007年**：Behler-Parrinello Neural Network (BPNN)
- **2012年**：Atom-centered Symmetry Functions (ACSF)
- **2017年**：SchNet（连续卷积）
- **2018年**：DeepPot-SE（Deep Potential - Smooth Edition）
- **2019年**：PhysNet, DimeNet
- **2020年**：NequIP, MACE（等变神经网络）

---

## 神经网络势函数的基本假设

### 1. 局域性假设 (Locality)

**假设内容**：
原子的能量贡献主要由其**局部化学环境**决定，远距离原子的影响可忽略。

**数学表达**：
$$
E_{\text{total}} = \sum_{i=1}^{N} E_i(\{\mathbf{R}_j : \|\mathbf{R}_j - \mathbf{R}_i\| < r_c\})
$$

其中：
- $E_i$：原子 $i$ 的能量贡献
- $r_c$：截断半径（cutoff radius），通常 5-6 Å
- 只考虑截断半径内的原子

**物理依据**：
- 量子力学的指数衰减特性
- 原子轨道的局域性
- 屏蔽效应

**实际考虑**：
- **短程相互作用**：共价键、金属键
- **长程相互作用**：静电、范德华力需要特殊处理
  - 静电：Ewald求和、电荷平衡方法
  - 范德华：C₆项修正

### 2. 能量的可加性

**总能量分解**：
$$
E_{\text{total}} = \sum_{i=1}^{N} E_i^{\text{atomic}}
$$

**注意**：
- 不是简单的原子能量之和
- $E_i^{\text{atomic}}$ 取决于局部环境
- 不同环境下同种元素的能量贡献不同

**示例**：
- 表面的金原子能量 ≠ 体相金原子能量
- 配位数影响能量贡献

### 3. 对称性要求

神经网络势函数必须满足物理对称性：

#### (a) 平移不变性 (Translational Invariance)

$$
E(\mathbf{R}_1 + \mathbf{a}, \mathbf{R}_2 + \mathbf{a}, ...) = E(\mathbf{R}_1, \mathbf{R}_2, ...)
$$

**实现方法**：使用相对坐标或距离
$$
\mathbf{r}_{ij} = \mathbf{R}_j - \mathbf{R}_i
$$

#### (b) 旋转不变性 (Rotational Invariance)

$$
E(\mathbf{U}\mathbf{R}_1, \mathbf{U}\mathbf{R}_2, ...) = E(\mathbf{R}_1, \mathbf{R}_2, ...)
$$

其中 $\mathbf{U}$ 是旋转矩阵。

**实现方法**：
- **不变性方法**：使用距离、角度等不变量
  - 距离：$r_{ij} = \|\mathbf{R}_j - \mathbf{R}_i\|$
  - 角度：$\theta_{ijk} = \cos^{-1}\left(\frac{\mathbf{r}_{ij} \cdot \mathbf{r}_{ik}}{r_{ij}r_{ik}}\right)$

- **等变性方法**：使用球谐函数等
  - 输入/输出按SO(3)表示变换
  - e3nn, NequIP等框架

#### (c) 置换不变性 (Permutation Invariance)

对于同种元素的原子交换，能量不变：

$$
E(..., \mathbf{R}_i, ..., \mathbf{R}_j, ...) = E(..., \mathbf{R}_j, ..., \mathbf{R}_i, ...)
$$

**实现方法**：
- 对称函数求和
- 消息传递网络的聚合操作
- 注意力机制的归一化

#### (d) 尺度协变性 (Size Extensivity)

能量应与系统大小成正比：

$$
E(2 \times \text{system}) = 2 \times E(\text{system})
$$

通过原子能量求和自动满足。

### 4. 平滑性和可导性

**要求**：
- 能量关于坐标连续可导
- 力通过自动微分获得：$\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{R}_i}$

**实现**：
- 使用平滑截断函数
- 连续激活函数（tanh, softplus）

---

## 原子结构和周围化学环境的表征

### 描述符的设计原则

**好的描述符应该**：
1. **完备性**：唯一确定原子环境
2. **紧凑性**：维度尽可能低
3. **连续性**：原子微小移动 → 描述符微小变化
4. **对称性**：满足旋转、平移、置换不变性
5. **区分性**：不同环境 → 不同描述符

### 1. 径向分布函数 (Radial Distribution Function)

最简单的描述符，只考虑距离：

$$
G_i^1 = \sum_{j \neq i} e^{-\eta(r_{ij} - R_s)^2} f_c(r_{ij})
$$

其中：
- $\eta$：宽度参数
- $R_s$：中心位置
- $f_c(r_{ij})$：截断函数

**截断函数示例**：
$$
f_c(r) = \begin{cases}
0.5 \left[\cos\left(\frac{\pi r}{r_c}\right) + 1\right] & r \leq r_c \\
0 & r > r_c
\end{cases}
$$

### 2. 原子中心对称函数 (Atom-Centered Symmetry Functions, ACSF)

Behler-Parrinello方法的核心。

#### 径向对称函数（G2）：
$$
G_i^2 = \sum_{j \neq i} e^{-\eta(r_{ij} - R_s)^2} f_c(r_{ij})
$$

#### 角度对称函数（G4）：
$$
G_i^4 = 2^{1-\zeta} \sum_{j,k \neq i} (1 + \lambda \cos\theta_{ijk})^\zeta e^{-\eta(r_{ij}^2 + r_{ik}^2 + r_{jk}^2)} f_c(r_{ij})f_c(r_{ik})f_c(r_{jk})
$$

其中：
- $\theta_{ijk}$：$j$-$i$-$k$ 夹角
- $\zeta$：角度分辨率
- $\lambda \in \{-1, 1\}$

**特点**：
- ✅ 满足所有对称性
- ✅ 物理意义清晰
- ❌ 手工设计参数($\eta, R_s, \zeta$等)
- ❌ 维度高（通常50-200维）

### 3. 平滑重叠原子位置 (Smooth Overlap of Atomic Positions, SOAP)

基于原子密度的描述符：

**原子密度**：
$$
\rho_i(\mathbf{r}) = \sum_{j \neq i} e^{-\alpha\|\mathbf{r} - \mathbf{r}_{ij}\|^2} f_c(r_{ij})
$$

**功率谱**（旋转不变）：
$$
p_{nn'll'}^i = \pi \sqrt{\frac{8}{2l+1}} \sum_m c_{nlm}^i c_{n'l'm}^{i*}
$$

**特点**：
- ✅ 完备性有理论保证
- ✅ 连续、可导
- ❌ 计算成本高
- ❌ 维度非常高

### 4. 深度势方法（DeepPot描述符）

**核心思想**：端到端学习描述符

**局部环境矩阵**：
$$
\mathbf{R}_i = \left[\frac{\mathbf{R}_1 - \mathbf{R}_i}{r_{i1}}, \frac{\mathbf{R}_2 - \mathbf{R}_i}{r_{i2}}, ...\right]^T
$$

**嵌入网络**：
$$
\mathbf{D}_i = f_{\text{embed}}(\mathbf{R}_i)
$$

**特点**：
- ✅ 自动学习最优表示
- ✅ 可扩展性好
- ✅ 准确性高
- 需要大量数据

### 5. SchNet的连续卷积滤波器

**连续卷积**：
$$
\mathbf{x}_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j^{(l)} \odot W^{(l)}(r_{ij})
$$

其中 $W^{(l)}(r_{ij})$ 是可学习的径向滤波器：

$$
W(r) = \sum_{k=1}^{K} \mathbf{w}_k \phi_k(r)
$$

$\phi_k(r)$ 是径向基函数（如高斯基）。

**特点**：
- ✅ 端到端学习
- ✅ 连续、可导
- ✅ 物理直观

### 描述符对比

| 描述符 | 完备性 | 可学习 | 维度 | 计算成本 | 代表模型 |
|-------|-------|--------|------|---------|---------|
| ACSF | 近似完备 | ❌ | 高 | 低 | BPNN |
| SOAP | 完备 | ❌ | 很高 | 高 | GAP |
| DeepPot | 完备 | ✅ | 可调 | 中 | DeepMD |
| 连续卷积 | 完备 | ✅ | 可调 | 中 | SchNet |
| 等变描述符 | 完备 | ✅ | 可调 | 高 | NequIP |

---

## BPNN描述符和DP深度神经网络势函数

### Behler-Parrinello Neural Network (BPNN)

**架构**：

```
原子坐标 {Rᵢ}
    ↓
对称函数 {Gⁱ}
    ↓
原子神经网络 (每个元素一个)
    ↓
原子能量 Eᵢ
    ↓
总能量 E = ΣEᵢ
```

**详细流程**：

1. **计算对称函数**：
   - 对每个原子 $i$，计算ACSF描述符 $\mathbf{G}_i$
   - 维度通常50-100

2. **归一化**：
   $$
   \tilde{G}_i^k = \frac{G_i^k - \mu_k}{\sigma_k}
   $$

3. **前馈神经网络**（每种元素一个独立网络）：
   $$
   E_i = \text{NN}_{\text{element}(i)}(\tilde{\mathbf{G}}_i)
   $$

4. **总能量**：
   $$
   E_{\text{total}} = \sum_{i=1}^{N} E_i
   $$

5. **力**：
   $$
   \mathbf{F}_i = -\frac{\partial E_{\text{total}}}{\partial \mathbf{R}_i} = -\sum_j \frac{\partial E_j}{\partial \mathbf{G}_j} \frac{\partial \mathbf{G}_j}{\partial \mathbf{R}_i}
   $$

**训练**：

损失函数：
$$
L = \frac{1}{N_{\text{config}}} \sum_n \left[ w_E (E_n^{\text{DFT}} - E_n^{\text{NN}})^2 + w_F \frac{1}{N_n} \sum_i \|\mathbf{F}_{n,i}^{\text{DFT}} - \mathbf{F}_{n,i}^{\text{NN}}\|^2 \right]
$$

**优点**：
- 开创性工作，证明了神经网络势的可行性
- 物理意义清晰
- 训练相对简单

**缺点**：
- 手工设计对称函数
- 对称函数参数需要调优
- 扩展到新体系需要重新设计

### DeepPot (Deep Potential)

由张林峰团队开发，代表深度学习势函数的新一代方法。

#### DeepPot-SE (Smooth Edition)

**核心创新**：
1. 端到端学习描述符
2. 嵌入网络 + 拟合网络的两阶段架构
3. 平滑、连续的描述符

**架构**：

```
坐标 {Rᵢ}
    ↓
局部坐标矩阵 Rᵢ
    ↓
嵌入网络 (Embedding Net)
    ↓
描述符 Dᵢ
    ↓
拟合网络 (Fitting Net) - 每元素一个
    ↓
原子能量 Eᵢ
    ↓
总能量 E = ΣEᵢ
```

**详细步骤**：

**1. 构建局部坐标矩阵**：

对原子 $i$，邻居 $j$：
$$
\tilde{\mathbf{R}}_{ij} = \left(\frac{\mathbf{R}_j - \mathbf{R}_i}{r_{ij}}, \frac{1}{r_{ij}}\right) s(r_{ij})
$$

其中 $s(r)$ 是平滑截断函数：
$$
s(r) = \begin{cases}
\frac{1}{r} & r < r_s \\
\frac{1}{r} \left[ u^3(-6u^2 + 15u - 10) + 1 \right] & r_s \leq r < r_c \\
0 & r \geq r_c
\end{cases}
$$
$u = \frac{r - r_s}{r_c - r_s}$

**2. 嵌入网络**：

$$
\mathbf{G}_{ij} = \mathcal{N}_{\text{embed}}(\tilde{\mathbf{R}}_{ij})
$$

$\mathcal{N}_{\text{embed}}$ 是共享的神经网络（通常2-3层）。

**3. 描述符矩阵**：

$$
\mathbf{D}_i = \frac{1}{N_i} \sum_{j \in \mathcal{N}(i)} \mathbf{G}_{ij} \otimes \tilde{\mathbf{R}}_{ij}
$$

其中 $\otimes$ 是外积。

对 $\mathbf{D}_i$ 进行归一化处理。

**4. 拟合网络**：

$$
E_i = \mathcal{N}_{\text{fit}}^{\text{element}(i)}(\mathbf{D}_i)
$$

**对称性保证**：
- **平移不变性**：使用相对坐标
- **旋转不变性**：描述符矩阵构造方式保证（证明见论文）
- **置换不变性**：求和操作

**优点**：
- ✅ 端到端学习，无需手工设计
- ✅ 准确性高（接近DFT）
- ✅ 泛化能力强
- ✅ 可扩展到复杂体系

**应用**：
- DeePMD-kit软件包
- 与LAMMPS集成
- 广泛应用于材料模拟

#### 实现示例（伪代码）：

```python
class DeepPot(nn.Module):
    def __init__(self, n_elements, embed_dim, fit_dim):
        super().__init__()
        # 嵌入网络（共享）
        self.embed_net = nn.Sequential(
            nn.Linear(4, 25),
            nn.Tanh(),
            nn.Linear(25, 50),
            nn.Tanh(),
            nn.Linear(50, embed_dim)
        )

        # 拟合网络（每个元素一个）
        self.fit_nets = nn.ModuleDict({
            element: nn.Sequential(
                nn.Linear(embed_dim * 4, fit_dim),
                nn.Tanh(),
                nn.Linear(fit_dim, fit_dim),
                nn.Tanh(),
                nn.Linear(fit_dim, 1)
            ) for element in element_list
        })

    def forward(self, coords, elements, neighbor_list):
        total_energy = 0

        for i in range(len(coords)):
            # 1. 计算局部坐标
            R_i = coords[neighbor_list[i]] - coords[i]
            r_ij = torch.norm(R_i, dim=1, keepdim=True)
            R_ij_normalized = R_i / r_ij
            cutoff = smooth_cutoff(r_ij)

            # 2. 嵌入网络
            input_ij = torch.cat([R_ij_normalized, 1/r_ij], dim=1)
            G_ij = self.embed_net(input_ij) * cutoff

            # 3. 构建描述符
            D_i = torch.matmul(G_ij.T, R_ij_normalized) / len(neighbor_list[i])

            # 4. 拟合网络
            D_i_flat = D_i.flatten()
            element_type = elements[i]
            E_i = self.fit_nets[element_type](D_i_flat)

            total_energy += E_i

        return total_energy
```

### 训练细节

**数据集构建**：
1. AIMD（从头算分子动力学）生成轨迹
2. 不同温度、压力条件
3. 包含多种构型（平衡、畸变、表面等）

**损失函数**：
$$
L = w_E L_E + w_F L_F + w_V L_V
$$

其中：
- $L_E$：能量误差
- $L_F$：力误差
- $L_V$：维里张量误差（可选，用于应力）

**训练策略**：
1. **自适应权重**：根据误差分布动态调整
2. **课程学习**：先简单构型，后复杂构型
3. **主动学习**：迭代增加训练数据

---

## 小结

1. **机器学习势函数**是第一性原理和分子动力学的桥梁
2. **局域性、对称性**是势函数的核心物理约束
3. **描述符**的设计从手工到自动学习
4. **BPNN**开创了神经网络势，**DeepPot**代表深度学习势的最新进展

---

## 扩展阅读

1. Behler, J., & Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. *Physical Review Letters*, 98(14), 146401.
2. Zhang, L., Han, J., Wang, H., Car, R., & E, W. (2018). Deep potential molecular dynamics: a scalable model with the accuracy of quantum mechanics. *Physical Review Letters*, 120(14), 143001.
3. Schütt, K. T., Sauceda, H. E., Kindermans, P. J., Tkatchenko, A., & Müller, K. R. (2018). SchNet–A deep learning architecture for molecules and materials. *The Journal of Chemical Physics*, 148(24), 241722.
4. Unke, O. T., & Meuwly, M. (2019). PhysNet: A neural network for predicting energies, forces, dipole moments, and partial charges. *Journal of Chemical Theory and Computation*, 15(6), 3678-3693.

---

**上一节**: [深度学习的发展历程和优势](./02-Deep-Learning-Development.md)
**下一节**: [晶体材料简介及其电子结构特点](./04-Crystal-Materials.md)
