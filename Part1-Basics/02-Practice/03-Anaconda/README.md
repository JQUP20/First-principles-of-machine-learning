# Anaconda环境管理 | Anaconda Environment Management

## 目录 | Table of Contents

1. [Anaconda简介](#1-anaconda简介)
2. [Conda基本命令](#2-conda基本命令)
3. [环境管理](#3-环境管理)
4. [包管理](#4-包管理)
5. [PyTorch安装和GPU配置](#5-pytorch安装和gpu配置)

---

## 1. Anaconda简介

### 1.1 什么是Anaconda？

**Anaconda** 是一个开源的Python和R语言的发行版，专为科学计算设计。

**核心组件**：
- **Conda**: 包管理和环境管理系统
- **Python**: 预装Python及常用科学计算库
- **Anaconda Navigator**: 图形化管理界面

### 1.2 为什么使用Anaconda？

✅ **环境隔离**: 不同项目使用不同Python版本和包
✅ **依赖管理**: 自动处理包依赖关系
✅ **跨平台**: Linux、macOS、Windows统一体验
✅ **科学计算**: 预装NumPy、SciPy、Matplotlib等
✅ **易于安装**: 避免编译和依赖问题

### 1.3 安装Anaconda

**下载**：https://www.anaconda.com/download

**安装脚本 (Linux/macOS)**：
```bash
# 下载安装脚本
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

# 运行安装
bash Anaconda3-2023.09-0-Linux-x86_64.sh

# 按提示操作，建议：
# - 接受许可协议
# - 安装到默认位置 ~/anaconda3
# - 允许初始化conda
```

**初始化shell**：
```bash
# 初始化bash
conda init bash

# 初始化zsh
conda init zsh

# 重新加载shell配置
source ~/.bashrc  # 或 source ~/.zshrc
```

**验证安装**：
```bash
$ conda --version
conda 23.7.4

$ python --version
Python 3.11.5
```

---

## 2. Conda基本命令

### 2.1 获取帮助

```bash
# Conda帮助
$ conda --help

# 特定命令帮助
$ conda create --help
$ conda install --help
```

### 2.2 查看信息

```bash
# 查看conda信息
$ conda info

# 查看环境列表
$ conda env list
$ conda info --envs

# 查看当前环境的包
$ conda list

# 搜索包
$ conda search numpy
$ conda search "numpy>=1.20"
```

### 2.3 配置Conda

```bash
# 查看配置
$ conda config --show

# 添加清华镜像源（加速下载）
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

# 设置显示通道URL
$ conda config --set show_channel_urls yes

# 查看配置文件位置
$ conda config --show-sources
```

**配置文件** (`~/.condarc`):
```yaml
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - defaults
show_channel_urls: true
```

---

## 3. 环境管理

### 3.1 创建环境

```bash
# 创建基本环境
$ conda create -n myenv

# 指定Python版本
$ conda create -n myenv python=3.9

# 创建环境并安装包
$ conda create -n dft python=3.9 numpy scipy matplotlib

# 从YAML文件创建
$ conda env create -f environment.yml
```

**environment.yml 示例**：
```yaml
name: first-principles
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy
  - scipy
  - matplotlib
  - pandas
  - jupyter
  - ase
  - pip
  - pip:
    - some-pip-package
```

### 3.2 激活和切换环境

```bash
# 激活环境
$ conda activate myenv

# 查看当前环境
(myenv) $ which python
/home/user/anaconda3/envs/myenv/bin/python

# 返回base环境
(myenv) $ conda deactivate

# 直接切换到另一个环境
(base) $ conda activate another_env
```

### 3.3 查看和管理环境

```bash
# 列出所有环境
$ conda env list
# conda environments:
#
base                  *  /home/user/anaconda3
myenv                    /home/user/anaconda3/envs/myenv
dft                      /home/user/anaconda3/envs/dft

# 克隆环境
$ conda create -n myenv_copy --clone myenv

# 导出环境
$ conda env export > environment.yml
$ conda env export -n myenv > myenv.yml

# 导出跨平台环境（只包含显式安装的包）
$ conda env export --from-history > environment.yml

# 删除环境
$ conda env remove -n myenv
$ conda remove -n myenv --all
```

### 3.4 环境最佳实践

**1. 为每个项目创建独立环境**
```bash
# DFT计算项目
$ conda create -n dft-project python=3.9 ase gpaw

# 机器学习项目
$ conda create -n ml-project python=3.9 pytorch pandas scikit-learn

# 数据分析项目
$ conda create -n data-analysis python=3.9 pandas matplotlib seaborn jupyter
```

**2. 使用environment.yml管理依赖**
```yaml
name: my-dft-project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy>=1.20
  - scipy>=1.7
  - matplotlib>=3.4
  - ase>=3.22
  - jupyter
  - pip:
    - gpaw>=22.8.0
```

**3. 定期更新和清理**
```bash
# 更新conda
$ conda update conda

# 更新所有包
$ conda update --all

# 清理未使用的包和缓存
$ conda clean --all
```

---

## 4. 包管理

### 4.1 安装包

```bash
# 基本安装
$ conda install numpy

# 安装多个包
$ conda install numpy scipy matplotlib

# 指定版本
$ conda install numpy=1.24.0
$ conda install "numpy>=1.20,<1.25"

# 从特定channel安装
$ conda install -c conda-forge ase

# 使用pip安装（在conda环境中）
$ pip install gpaw
```

### 4.2 更新包

```bash
# 更新单个包
$ conda update numpy

# 更新多个包
$ conda update numpy scipy matplotlib

# 更新所有包
$ conda update --all

# 更新pip安装的包
$ pip install --upgrade package_name
```

### 4.3 删除包

```bash
# 删除包
$ conda remove numpy

# 删除多个包
$ conda remove numpy scipy matplotlib
```

### 4.4 查看包信息

```bash
# 列出已安装的包
$ conda list

# 搜索包
$ conda search ase

# 查看包详细信息
$ conda search --info numpy

# 查看包的依赖
$ conda info numpy
```

### 4.5 Conda vs Pip

| 特性 | Conda | Pip |
|------|-------|-----|
| **包类型** | 任何语言（Python, R, C++等） | 仅Python |
| **依赖处理** | 二进制依赖也处理 | 仅Python依赖 |
| **环境管理** | 内置 | 需要virtualenv |
| **包来源** | Anaconda仓库 | PyPI |
| **编译** | 预编译二进制 | 可能需要编译 |

**使用建议**：
1. 优先使用conda安装科学计算包（NumPy, SciPy等）
2. conda没有的包使用pip
3. 在conda环境中使用pip，不要在系统Python中使用

**正确顺序**：
```bash
# 1. 创建环境
$ conda create -n myenv python=3.9

# 2. 激活环境
$ conda activate myenv

# 3. 先用conda安装主要包
$ conda install numpy scipy matplotlib

# 4. 再用pip安装conda没有的包
$ pip install some-specialized-package
```

---

## 5. PyTorch安装和GPU配置

### 5.1 检查CUDA版本

```bash
# 检查NVIDIA驱动和CUDA版本
$ nvidia-smi

# 输出示例：
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
```

### 5.2 安装PyTorch (CPU版本)

```bash
# 创建环境
$ conda create -n pytorch-env python=3.9

# 激活环境
$ conda activate pytorch-env

# 安装PyTorch (CPU)
$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 5.3 安装PyTorch (GPU版本)

**CUDA 11.8**:
```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**CUDA 12.1**:
```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**从PyTorch官网获取命令**：
https://pytorch.org/get-started/locally/

### 5.4 验证PyTorch安装

创建 `test_pytorch.py`:
```python
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

    # 测试GPU计算
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"GPU计算测试成功！")
    print(f"结果形状: {z.shape}")
else:
    print("CUDA不可用，使用CPU")

    # CPU计算
    x = torch.rand(1000, 1000)
    y = torch.rand(1000, 1000)
    z = torch.matmul(x, y)
    print(f"CPU计算测试成功！")
```

运行测试：
```bash
$ python test_pytorch.py
PyTorch版本: 2.1.0
CUDA是否可用: True
CUDA版本: 11.8
GPU数量: 1
当前GPU: 0
GPU名称: NVIDIA GeForce RTX 3090
GPU计算测试成功！
结果形状: torch.Size([1000, 1000])
```

### 5.5 简单的神经网络示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型并移到GPU
model = SimpleNet().to(device)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 创建虚拟数据
X = torch.randn(100, 10).to(device)
y = torch.randn(100, 1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
print("\n开始训练...")
for epoch in range(100):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

print("训练完成！")
```

### 5.6 性能优化技巧

**1. 使用混合精度训练**：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(epochs):
    for data, target in dataloader:
        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**2. 使用DataLoader**：
```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

**3. 固定随机种子**：
```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## 6. 常见问题和解决方案

### 6.1 环境问题

**问题1**: conda activate不工作
```bash
# 解决方案：重新初始化
$ conda init bash
$ source ~/.bashrc
```

**问题2**: 包冲突
```bash
# 解决方案：创建新环境
$ conda create -n new_env python=3.9
$ conda activate new_env
# 重新安装包
```

**问题3**: 环境损坏
```bash
# 解决方案：删除并重建
$ conda env remove -n broken_env
$ conda env create -f environment.yml
```

### 6.2 GPU问题

**问题1**: PyTorch检测不到CUDA
```python
# 检查：
import torch
print(torch.version.cuda)  # 应该显示版本号
print(torch.cuda.is_available())  # 应该是True

# 如果是False，重新安装正确的PyTorch版本
```

**问题2**: CUDA out of memory
```python
# 解决方案：
# 1. 减小batch size
# 2. 使用梯度累积
# 3. 清理缓存
torch.cuda.empty_cache()
```

### 6.3 性能问题

**Conda速度慢**：
```bash
# 使用mamba（更快的conda替代）
$ conda install mamba -c conda-forge
$ mamba install numpy scipy  # 使用mamba代替conda
```

**包下载慢**：
```bash
# 使用国内镜像源（清华、阿里等）
# 见2.3节配置Conda
```

---

## 7. 完整工作流程示例

### 创建第一性原理计算环境

```bash
# 1. 创建环境配置文件
$ cat > first_principles_env.yml << EOF
name: first-principles
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy>=1.20
  - scipy>=1.7
  - matplotlib>=3.5
  - pandas
  - jupyter
  - jupyterlab
  - ase>=3.22
  - spglib
  - pymatgen
  - pip
  - pip:
    - gpaw>=22.8.0
EOF

# 2. 创建环境
$ conda env create -f first_principles_env.yml

# 3. 激活环境
$ conda activate first-principles

# 4. 验证安装
$ python -c "import ase; print(f'ASE version: {ase.__version__}')"
$ python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# 5. 启动Jupyter Lab
$ jupyter lab
```

---

## 8. 学习检查清单

- [ ] 理解Conda环境隔离的概念
- [ ] 能创建和管理Conda环境
- [ ] 会安装和更新包
- [ ] 理解Conda和Pip的区别
- [ ] 能安装和测试PyTorch
- [ ] 会使用GPU进行计算
- [ ] 能导出和分享环境配置

---

## 参考资源

- [Conda官方文档](https://docs.conda.io/)
- [PyTorch官网](https://pytorch.org/)
- [Conda速查表](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)

**练习和示例**：查看 `examples/` 目录 📁

**下一步**：ASE原子建模 →
