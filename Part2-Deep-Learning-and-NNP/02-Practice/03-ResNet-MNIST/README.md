# ResNet残差网络手写数字识别
# ResNet for Handwritten Digit Recognition (MNIST)

本项目使用ResNet残差网络实现手写数字识别，这是一个完整的深度学习项目示例。

## 项目简介

使用ResNet-18架构在MNIST数据集上训练手写数字分类模型。通过这个项目，你将学习：

- 如何构建ResNet残差网络
- 完整的深度学习训练流程
- 超参数调优技巧
- 模型评估和可视化
- 实验管理最佳实践

## 项目结构

```
03-ResNet-MNIST/
├── README.md              # 项目说明
├── requirements.txt       # 依赖包
├── train.py              # 训练脚本
├── test.py               # 测试脚本
├── models/
│   ├── __init__.py
│   ├── resnet.py         # ResNet模型定义
│   └── simple_cnn.py     # 简单CNN对比
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # 数据加载
│   ├── trainer.py        # 训练器
│   └── visualization.py  # 可视化工具
├── data/                 # 数据目录（自动下载）
├── checkpoints/          # 模型检查点
└── results/              # 结果和图表
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 使用默认参数训练
python train.py

# 自定义参数
python train.py --epochs 20 --batch-size 128 --lr 0.001
```

### 3. 测试模型

```bash
python test.py --model-path checkpoints/best_model.pth
```

## MNIST数据集

### 数据集介绍

- **训练集**: 60,000张28x28灰度图像
- **测试集**: 10,000张28x28灰度图像
- **类别**: 0-9十个数字
- **来源**: Yann LeCun等人收集

### 数据预处理

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),      # 调整为32x32适配ResNet
    transforms.ToTensor(),             # 转换为Tensor
    transforms.Normalize(              # 归一化
        mean=[0.1307],
        std=[0.3081]
    )
])
```

### 数据增强（可选）

```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),     # 随机旋转±10度
    transforms.RandomAffine(           # 随机仿射变换
        degrees=0,
        translate=(0.1, 0.1)
    ),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])
```

## ResNet架构

### ResNet-18结构

```
Input (1, 32, 32)
    ↓
Conv1 (7x7, stride=2) + BN + ReLU
    ↓
MaxPool (3x3, stride=2)
    ↓
Layer1 (BasicBlock x 2, 64 channels)
    ↓
Layer2 (BasicBlock x 2, 128 channels)
    ↓
Layer3 (BasicBlock x 2, 256 channels)
    ↓
Layer4 (BasicBlock x 2, 512 channels)
    ↓
Global Average Pooling
    ↓
Fully Connected (10 classes)
    ↓
Output
```

### 残差块(Basic Block)

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out
```

## 训练流程

### 超参数

```python
# 默认超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
WEIGHT_DECAY = 1e-4
```

### 训练循环

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc
```

### 学习率调度

```python
# 使用CosineAnnealingLR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# 或使用StepLR
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=7,
    gamma=0.1
)
```

## 模型评估

### 评估指标

1. **准确率 (Accuracy)**:
   $$\text{Accuracy} = \frac{\text{正确预测数}}{\text{总样本数}}$$

2. **精确率 (Precision)**:
   $$\text{Precision} = \frac{TP}{TP + FP}$$

3. **召回率 (Recall)**:
   $$\text{Recall} = \frac{TP}{TP + FN}$$

4. **F1分数**:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 混淆矩阵

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 绘制
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

## 实验结果

### 基准结果

| 模型 | 参数量 | 训练准确率 | 测试准确率 | 训练时间 |
|------|-------|----------|----------|---------|
| Simple CNN | 50K | 98.5% | 98.2% | 2 min |
| ResNet-18 | 11M | 99.8% | 99.4% | 5 min |
| ResNet-34 | 21M | 99.9% | 99.5% | 8 min |

### 训练曲线

训练脚本会自动生成以下图表：

1. **损失曲线**: `results/loss_curve.png`
2. **准确率曲线**: `results/accuracy_curve.png`
3. **学习率变化**: `results/lr_schedule.png`
4. **混淆矩阵**: `results/confusion_matrix.png`

## 超参数调优

### 学习率

```python
# 尝试不同的学习率
learning_rates = [0.1, 0.01, 0.001, 0.0001]

for lr in learning_rates:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 训练并记录结果
```

### 批次大小

```python
# 批次大小影响训练速度和模型性能
batch_sizes = [32, 64, 128, 256]
```

### 数据增强

```python
# 比较有无数据增强的效果
# 1. 无数据增强
# 2. 轻度数据增强（旋转、平移）
# 3. 重度数据增强（旋转、平移、缩放、弹性变形）
```

## 高级技巧

### 1. 早停 (Early Stopping)

```python
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
```

### 2. 模型集成

```python
# 训练多个模型并集成
models = [ResNet18(), ResNet34(), SimpleNet()]

# 预测时取平均
predictions = []
for model in models:
    pred = model(x)
    predictions.append(pred)

final_pred = torch.mean(torch.stack(predictions), dim=0)
```

### 3. 迁移学习

```python
# 使用预训练ResNet并微调
from torchvision.models import resnet18

model = resnet18(pretrained=True)

# 冻结前面的层
for param in model.parameters():
    param.requires_grad = False

# 只训练最后一层
model.fc = nn.Linear(512, 10)
```

## 命令行参数

```bash
python train.py \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.001 \
    --weight-decay 1e-4 \
    --device cuda \
    --save-dir checkpoints \
    --log-interval 100
```

## 练习题

### 练习1：基础训练
使用默认参数训练ResNet-18，观察训练过程和最终准确率。

### 练习2：超参数调优
尝试至少3组不同的超参数组合，比较结果。

### 练习3：模型比较
实现一个简单的CNN，与ResNet比较性能和训练时间。

### 练习4：错误分析
找出模型预测错误的样本，分析原因。

### 练习5：可视化
使用t-SNE可视化最后一层特征，观察不同数字的聚类情况。

## 常见问题

### Q1: 训练准确率很高但测试准确率低

**原因**: 过拟合

**解决方案**:
- 增加Dropout
- 数据增强
- 减小模型复杂度
- 增加训练数据

### Q2: 损失不下降

**可能原因**:
- 学习率太大或太小
- 梯度消失/爆炸
- 数据预处理问题

**解决方案**:
- 调整学习率
- 检查梯度范数
- 使用梯度裁剪
- 检查数据标准化

### Q3: 训练太慢

**优化方法**:
- 使用GPU训练
- 增大batch size
- 使用混合精度训练
- 减小模型大小

## 扩展实验

1. **Fashion-MNIST**: 尝试在Fashion-MNIST数据集上训练
2. **CIFAR-10**: 彩色图像分类（10类）
3. **迁移学习**: 使用ImageNet预训练权重
4. **模型压缩**: 使用知识蒸馏或剪枝
5. **可解释性**: 使用Grad-CAM可视化关注区域

## 参考文献

1. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
2. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.

---

**返回**: [Part 2主页](../../README.md)
