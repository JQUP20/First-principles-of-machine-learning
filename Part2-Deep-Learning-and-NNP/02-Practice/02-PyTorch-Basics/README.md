# PyTorch深度学习库
# PyTorch Deep Learning Library

本目录包含PyTorch深度学习库的基础教程和示例代码。

## 学习目标

1. 掌握PyTorch的安装和GPU配置
2. 理解Tensor操作和自动微分
3. 学会构建神经网络模型
4. 掌握完整的训练流程
5. 了解PyTorch的模块化设计

## 内容概览

### 教程 (Tutorials)

1. **PyTorch安装和环境配置** (`tutorials/01_installation.md`)
   - Conda环境创建
   - CPU vs GPU版本
   - 验证安装

2. **Tensor基础** (`tutorials/02_tensor_basics.md`)
   - Tensor创建和操作
   - GPU加速
   - 与NumPy的互操作

3. **自动微分** (`tutorials/03_autograd.md`)
   - requires_grad
   - 反向传播
   - 梯度计算

4. **神经网络模块** (`tutorials/04_nn_module.md`)
   - nn.Module基类
   - 常用层
   - 模型定义

5. **训练流程** (`tutorials/05_training_loop.md`)
   - DataLoader使用
   - 优化器选择
   - 损失函数

### 示例代码 (Examples)

1. **Tensor操作** (`examples/01_tensor_operations.py`)
2. **简单神经网络** (`examples/02_simple_nn.py`)
3. **线性回归** (`examples/03_linear_regression.py`)
4. **分类问题** (`examples/04_classification.py`)
5. **模型保存和加载** (`examples/05_save_load.py`)

## 快速开始

### 1. 安装PyTorch

```bash
# CPU版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# GPU版本（CUDA 11.8）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# GPU版本（CUDA 12.1）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. 验证安装

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 3. 第一个PyTorch程序

```python
import torch
import torch.nn as nn

# 创建简单的线性模型
model = nn.Linear(10, 1)

# 随机输入
x = torch.randn(5, 10)

# 前向传播
y = model(x)

print(f"输出形状: {y.shape}")
```

## 核心概念

### 1. Tensor

PyTorch的核心数据结构，类似NumPy数组但支持GPU加速。

```python
# 创建tensor
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 4)
z = torch.randn(2, 3)

# GPU加速
if torch.cuda.is_available():
    x = x.cuda()  # 或 x.to('cuda')
```

### 2. Autograd

自动微分系统，自动计算梯度。

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
```

### 3. nn.Module

所有神经网络的基类。

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 4. 训练循环

```python
# 定义模型、损失函数、优化器
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 常用模块

### 神经网络层

```python
import torch.nn as nn

# 全连接层
fc = nn.Linear(in_features, out_features)

# 卷积层
conv = nn.Conv2d(in_channels, out_channels, kernel_size)

# 池化层
pool = nn.MaxPool2d(kernel_size=2)

# 归一化层
bn = nn.BatchNorm2d(num_features)

# Dropout
dropout = nn.Dropout(p=0.5)
```

### 激活函数

```python
# ReLU
relu = nn.ReLU()

# Tanh
tanh = nn.Tanh()

# Sigmoid
sigmoid = nn.Sigmoid()

# Softmax
softmax = nn.Softmax(dim=1)
```

### 损失函数

```python
# 均方误差（回归）
mse_loss = nn.MSELoss()

# 交叉熵（分类）
ce_loss = nn.CrossEntropyLoss()

# 二元交叉熵
bce_loss = nn.BCELoss()
```

### 优化器

```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
```

## 最佳实践

### 1. 设备管理

```python
# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型和数据移到设备
model = model.to(device)
inputs = inputs.to(device)
```

### 2. 数据加载

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
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 3. 模型保存和加载

```python
# 保存模型参数
torch.save(model.state_dict(), 'model.pth')

# 加载模型参数
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### 4. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 练习题

### 练习1：Tensor操作
创建两个随机tensor，计算它们的矩阵乘法，并在GPU上执行（如果可用）。

### 练习2：自动微分
实现函数 $f(x, y) = x^2 + 2xy + y^2$，计算其在(1, 2)处的梯度。

### 练习3：简单神经网络
构建一个3层的全连接网络（输入784，隐藏层256、128，输出10）。

### 练习4：数据集
创建一个自定义Dataset类，加载MNIST数据（或任意数据）。

### 练习5：完整训练
使用练习3的模型和练习4的数据集，完成一个完整的训练循环。

## 常见问题

### Q1: RuntimeError: CUDA out of memory

**解决方案**:
- 减小batch size
- 使用梯度累积
- 使用混合精度训练
- 释放不需要的中间变量

### Q2: 梯度消失/爆炸

**解决方案**:
- 使用ReLU激活函数
- 批归一化(Batch Normalization)
- 梯度裁剪(Gradient Clipping)
- 残差连接(Residual Connections)

### Q3: 训练速度慢

**优化方法**:
- 使用GPU
- 增大batch size
- 使用DataLoader的num_workers
- 混合精度训练
- 使用更高效的模型架构

## 扩展资源

### 官方文档
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Forums](https://discuss.pytorch.org/)

### 推荐书籍
- *Deep Learning with PyTorch* - Eli Stevens, Luca Antiga
- *Programming PyTorch for Deep Learning* - Ian Pointer

### 在线课程
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)
- [PyTorch官方教程](https://pytorch.org/tutorials/)

## 下一步

完成本章后，进入：
- [ResNet手写数字识别实战](../03-ResNet-MNIST/)

---

**返回**: [Part 2主页](../../README.md)
