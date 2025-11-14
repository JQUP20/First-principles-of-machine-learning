# 深度学习的发展历程和优势
# Development History and Advantages of Deep Learning

## 目录
- [神经网络的发展历程](#神经网络的发展历程)
- [神经网络的常见分类](#神经网络的常见分类)
- [深度神经网络入门——ResNet残差网络](#深度神经网络入门resnet残差网络)
- [常用的深度学习库——PyTorch介绍](#常用的深度学习库pytorch介绍)

---

## 神经网络的发展历程

### 第一次浪潮：感知机时代 (1950s-1960s)

**1943年 - McCulloch-Pitts神经元模型**
- Warren McCulloch 和 Walter Pitts 提出第一个数学神经元模型
- 简单的二值逻辑单元

**1958年 - 感知机 (Perceptron)**
- Frank Rosenblatt 发明感知机
- 能够学习线性分类器
- Mark I Perceptron - 第一个神经网络硬件实现

**1969年 - 第一次AI寒冬**
- Minsky 和 Papert 的《Perceptrons》指出感知机的局限性
- 无法解决XOR问题（线性不可分）
- 导致神经网络研究停滞

### 第二次浪潮：反向传播与多层网络 (1980s-1990s)

**1986年 - 反向传播算法**
- Rumelhart, Hinton, Williams 推广反向传播算法
- 解决了多层网络的训练问题
- 开启了多层感知机（MLP）时代

**1989年 - 卷积神经网络 (CNN)**
- Yann LeCun 提出 LeNet
- 成功应用于手写数字识别（MNIST）
- 引入卷积和池化概念

**1997年 - LSTM**
- Hochreiter 和 Schmidhuber 提出长短期记忆网络
- 解决了循环神经网络的梯度消失问题

**1990s末 - 第二次AI寒冬**
- 训练深层网络困难（梯度消失/爆炸）
- 计算资源有限
- 支持向量机（SVM）等方法表现更好

### 第三次浪潮：深度学习革命 (2006-至今)

**2006年 - 深度信念网络 (DBN)**
- Geoffrey Hinton 提出逐层预训练方法
- "Deep Learning" 术语开始流行
- 深度学习复兴的起点

**2012年 - AlexNet革命**
- Alex Krizhevsky 在ImageNet竞赛中大幅领先
- 使用GPU加速训练
- 使用ReLU激活函数和Dropout
- 标志着深度学习时代的到来

**2014年 - VGGNet 和 GoogLeNet**
- VGG：更深的网络（16-19层）
- GoogLeNet：Inception模块，多尺度特征

**2015年 - ResNet残差网络**
- 何恺明等人提出残差连接
- 成功训练152层甚至更深的网络
- ImageNet Top-5错误率降至3.57%（超越人类）

**2017年 - Transformer架构**
- "Attention is All You Need"
- 摒弃CNN和RNN，完全基于注意力机制
- 开启自然语言处理新纪元

**2018-2020年 - 大模型时代**
- BERT, GPT-2, GPT-3
- 参数量从亿级到千亿级
- 预训练-微调范式

**2020-至今 - 基础模型与多模态**
- GPT-4, Claude, Gemini等大语言模型
- DALL-E, Stable Diffusion等图像生成模型
- 多模态融合（文本、图像、视频）

### 在材料科学和第一性原理计算中的应用

**2007年 - 第一代神经网络势函数**
- Behler-Parrinello Neural Network Potential (BPNN)
- 使用对称函数描述原子环境

**2017年 - 深度学习势函数**
- SchNet, DeepPot-SE等
- 端到端学习，自动提取特征

**2019-至今 - 图神经网络与Transformer**
- GemNet, DimeNet++
- Equiformer, NequIP（等变神经网络）
- 准确性接近DFT，速度提升百万倍

---

## 神经网络的常见分类

### 1. 前馈神经网络 (Feedforward Neural Network, FNN)

**结构特点**：
- 信息单向流动：输入层 → 隐藏层 → 输出层
- 没有循环或反馈连接
- 最基础的神经网络结构

**类型**：
- **单层感知机**：无隐藏层
- **多层感知机 (MLP)**：包含一个或多个隐藏层

**应用**：
- 分类和回归问题
- 神经网络势函数中的原子能量网络
- 简单的模式识别

**示例代码**：
```python
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 2. 卷积神经网络 (Convolutional Neural Network, CNN)

**核心概念**：
- **卷积层**：局部感受野，权重共享
- **池化层**：下采样，降低维度
- **全连接层**：最终分类或回归

**特点**：
- 平移不变性
- 局部连接
- 参数共享

**应用**：
- 图像识别和分类
- 物体检测
- 材料显微图像分析
- 晶体结构图像处理

**经典架构**：
- LeNet-5 (1998)
- AlexNet (2012)
- VGG (2014)
- ResNet (2015)
- EfficientNet (2019)

**示例代码**：
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 3. 循环神经网络 (Recurrent Neural Network, RNN)

**结构特点**：
- 包含循环连接
- 具有记忆能力
- 处理序列数据

**变体**：
- **标准RNN**：梯度消失问题严重
- **LSTM (Long Short-Term Memory)**：门控机制
- **GRU (Gated Recurrent Unit)**：简化版LSTM

**应用**：
- 自然语言处理
- 时间序列预测
- 分子动力学轨迹分析
- AIMD数据序列处理

**LSTM示例**：
```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 使用最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out
```

### 4. 图神经网络 (Graph Neural Network, GNN)

**核心思想**：
- 直接在图结构上操作
- 节点和边的特征传递
- 置换不变性

**主要类型**：
- **GCN (Graph Convolutional Network)**
- **GAT (Graph Attention Network)**
- **Message Passing Neural Network (MPNN)**

**在材料科学中的应用**：
- 原子结构天然是图（节点=原子，边=化学键）
- SchNet, DimeNet, PaiNN等势函数模型
- 晶体属性预测（CGCNN, MEGNet）

**消息传递框架**：
```python
# 伪代码示例
for layer in range(num_layers):
    # 1. 消息构建
    messages = message_function(node_features, edge_features)

    # 2. 消息聚合
    aggregated = aggregate_function(messages, graph_structure)

    # 3. 节点更新
    node_features = update_function(node_features, aggregated)

# 4. 图级别输出
graph_output = readout_function(node_features)
```

**对比表格**：

| 网络类型 | 数据结构 | 核心操作 | 主要应用 | 在材料科学中的应用 |
|---------|---------|---------|---------|------------------|
| FNN | 向量 | 矩阵乘法 | 通用分类/回归 | 原子能量预测 |
| CNN | 网格（图像） | 卷积 | 图像处理 | 显微图像分析 |
| RNN/LSTM | 序列 | 循环连接 | 时序数据 | MD轨迹分析 |
| GNN | 图 | 消息传递 | 图结构数据 | 分子/晶体性质预测 |

---

## 深度神经网络入门——ResNet残差网络

### 深度网络的退化问题

**问题**：理论上，更深的网络应该至少和浅层网络一样好（可以学习恒等映射），但实际中：
- 训练误差和测试误差都随深度增加而上升
- 这不是过拟合，而是优化困难

**原因**：
- 梯度消失/爆炸
- 深层网络难以学习恒等映射
- 优化景观复杂

### 残差学习 (Residual Learning)

**核心思想**：不直接学习期望的底层映射 $H(x)$，而是学习残差 $F(x) = H(x) - x$

**残差块 (Residual Block)**：
```
输入 x
  ↓
  ├─→ [卷积层] → [BN] → [ReLU] → [卷积层] → [BN] → + → [ReLU] → 输出
  │                                                  ↑
  └──────────────────────────────────────────────────┘
                    跳跃连接 (skip connection)
```

**数学表达**：
$$
\mathbf{y} = F(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

其中：
- $\mathbf{x}$：输入
- $F(\mathbf{x}, \{W_i\})$：残差映射
- $\mathbf{y}$：输出

### 为什么残差学习有效？

1. **恒等映射容易学习**：
   - 如果恒等映射是最优的，网络只需将 $F(x)$ 的权重推向零
   - 比直接学习 $H(x) = x$ 更容易

2. **梯度传播**：
$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \left(1 + \frac{\partial F}{\partial \mathbf{x}}\right)
$$
   - 梯度至少有 $\frac{\partial L}{\partial \mathbf{y}}$ 可以直接传播
   - 缓解梯度消失问题

3. **集成效应**：
   - ResNet可视为多个浅层网络的集成
   - $2^n$ 种路径（n是残差块数量）

### ResNet架构

**ResNet-34结构**：
```
Input (224×224×3)
  ↓
Conv1 (7×7, 64, stride 2)
  ↓
MaxPool (3×3, stride 2)
  ↓
Conv2_x (3层残差块, 64通道) ×3
  ↓
Conv3_x (4层残差块, 128通道) ×4
  ↓
Conv4_x (6层残差块, 256通道) ×6
  ↓
Conv5_x (3层残差块, 512通道) ×3
  ↓
Global Average Pooling
  ↓
Fully Connected (1000类)
  ↓
Softmax
```

**基础残差块（ResNet-18/34）**：
```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入输出维度不同，需要调整shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out
```

**瓶颈块（ResNet-50/101/152）**：
```python
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

### ResNet的影响

- **计算机视觉**：成为骨干网络标准
- **其他领域**：残差连接被广泛应用（Transformer、GNN等）
- **材料科学**：PhysNet等势函数模型使用残差连接

---

## 常用的深度学习库——PyTorch介绍

### 为什么选择PyTorch？

**优点**：
1. **动态计算图**：灵活、易调试
2. **Pythonic**：代码简洁、直观
3. **强大的自动微分**：`autograd`模块
4. **丰富的生态**：PyTorch Geometric, PyTorch Lightning等
5. **科研友好**：大多数论文代码使用PyTorch

**PyTorch vs TensorFlow**：

| 特性 | PyTorch | TensorFlow |
|-----|---------|-----------|
| 计算图 | 动态（define-by-run） | 静态+动态 |
| 调试 | 容易（Python调试器） | 较难 |
| 部署 | TorchScript, ONNX | TensorFlow Serving |
| 移动端 | PyTorch Mobile | TensorFlow Lite |
| 社区 | 学术界主导 | 工业界主导 |

### PyTorch核心概念

#### 1. Tensor（张量）

PyTorch的核心数据结构，类似NumPy数组但支持GPU加速。

```python
import torch

# 创建tensor
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 4)
z = torch.randn(2, 3)  # 正态分布随机数

# GPU加速
if torch.cuda.is_available():
    x = x.cuda()  # 或 x.to('cuda')

# 常用操作
a = torch.ones(2, 3)
b = torch.ones(2, 3)
c = a + b  # 逐元素加法
d = torch.mm(a, b.t())  # 矩阵乘法
e = a @ b.t()  # 等价写法
```

#### 2. Autograd（自动微分）

```python
# 需要梯度的tensor
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

# 反向传播
y.backward()

# 查看梯度
print(x.grad)  # dy/dx = 2x + 3 = 7
```

#### 3. nn.Module（神经网络模块）

所有神经网络的基类。

```python
import torch.nn as nn
import torch.nn.functional as F

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyNetwork()
```

#### 4. 训练循环

```python
import torch.optim as optim

# 定义模型、损失函数、优化器
model = MyNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数
```

### PyTorch生态系统

**核心库**：
- `torch`: 核心张量库
- `torch.nn`: 神经网络模块
- `torch.optim`: 优化器
- `torch.utils.data`: 数据加载工具

**扩展库**：
- **torchvision**: 计算机视觉（数据集、模型、变换）
- **torchaudio**: 音频处理
- **torchtext**: 自然语言处理
- **PyTorch Geometric**: 图神经网络
- **PyTorch Lightning**: 高层封装，简化训练

**材料科学相关**：
- **e3nn**: 等变神经网络
- **SchNetPack**: 量子化学和材料科学
- **LAMMPS-PyTorch**: 与分子动力学集成

### PyTorch最佳实践

1. **使用DataLoader**：
```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(data, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

2. **模型保存和加载**：
```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

3. **GPU加速**：
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

4. **混合精度训练**（加速训练）：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 小结

1. **发展历程**：从感知机到深度学习，经历三次浪潮
2. **网络分类**：FNN、CNN、RNN、GNN各有特点和应用场景
3. **ResNet**：残差连接解决深层网络训练难题，是深度学习的里程碑
4. **PyTorch**：灵活、直观的深度学习框架，科研首选

---

## 扩展阅读

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
3. [PyTorch官方文档](https://pytorch.org/docs/)
4. Schmidhuber, J. (2015). Deep learning in neural networks: An overview. *Neural networks*, 61, 85-117.

---

**上一节**: [深度学习基本理论](./01-Deep-Learning-Basics.md)
**下一节**: [神经网络势函数](./03-Neural-Network-Potentials.md)
