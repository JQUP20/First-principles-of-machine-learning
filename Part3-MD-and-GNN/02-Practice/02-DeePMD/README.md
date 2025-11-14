# DeePMD深势模型
# DeePMD: Deep Potential Molecular Dynamics

本目录包含DeePMD-kit的安装、训练和使用教程。

## 学习目标

1. 掌握DeePMD-kit的安装和配置
2. 学会准备训练数据集
3. 理解DeePMD模型架构和参数
4. 训练并验证深度势能模型
5. 在LAMMPS中使用训练好的模型进行MD模拟

## DeePMD-kit简介

**DeePMD-kit**是由张林峰团队开发的深度学习势函数工具包，基于TensorFlow/PyTorch实现。

**特点**：
- 端到端学习
- DFT级别准确度
- 百万倍加速
- 与LAMMPS无缝集成

**论文**：Zhang et al., *End-to-end Symmetry Preserving Inter-atomic Potential Energy Model for Finite and Extended Systems*, NeurIPS 2018

## 安装

### 方法1：Conda安装(推荐)

```bash
# 创建环境
conda create -n deepmd python=3.9
conda activate deepmd

# 安装DeePMD-kit
conda install deepmd-kit=*=*gpu lammps-dp=*=*gpu -c https://conda.deepmodeling.com

# 验证
dp --version
```

### 方法2：离线安装

```bash
# 下载安装包
wget https://conda.deepmodeling.com/deepmd-kit-2.x-gpu.tar.gz

# 安装
conda install --use-local deepmd-kit-2.x-gpu.tar.gz

# 测试
python -c "import deepmd; print(deepmd.__version__)"
```

## 数据准备

### 数据格式

DeePMD需要特定格式的数据：

```
dataset/
├── type.raw          # 原子类型
├── type_map.raw      # 类型映射
├── set.000/
│   ├── box.npy       # 晶胞矩阵
│   ├── coord.npy     # 原子坐标
│   ├── energy.npy    # 能量
│   └── force.npy     # 力
├── set.001/
│   └── ...
└── ...
```

### 从VASP/GPAW转换

```python
#!/usr/bin/env python3
"""
将AIMD轨迹转换为DeePMD格式
"""
import numpy as np
from ase.io import read
import os

def convert_to_deepmd(traj_file, output_dir='dataset'):
    """
    转换ASE轨迹到DeePMD格式
    """
    # 读取轨迹
    atoms_list = read(traj_file, index=':')

    os.makedirs(output_dir, exist_ok=True)

    # 获取元素类型
    elements = list(set(atoms_list[0].get_chemical_symbols()))
    elements.sort()

    # type_map.raw
    with open(os.path.join(output_dir, 'type_map.raw'), 'w') as f:
        f.write(' '.join(elements))

    # 分批处理(每1000个构型一个set)
    batch_size = 1000
    num_batches = (len(atoms_list) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(atoms_list))
        batch = atoms_list[start:end]

        set_dir = os.path.join(output_dir, f'set.{batch_idx:03d}')
        os.makedirs(set_dir, exist_ok=True)

        boxes = []
        coords = []
        energies = []
        forces = []

        for atoms in batch:
            # 晶胞
            boxes.append(atoms.cell[:])

            # 坐标
            coords.append(atoms.positions.flatten())

            # 能量
            energies.append(atoms.get_potential_energy())

            # 力
            forces.append(atoms.get_forces().flatten())

        # 保存
        np.save(os.path.join(set_dir, 'box.npy'), np.array(boxes))
        np.save(os.path.join(set_dir, 'coord.npy'), np.array(coords))
        np.save(os.path.join(set_dir, 'energy.npy'), np.array(energies))
        np.save(os.path.join(set_dir, 'force.npy'), np.array(forces))

    print(f"转换完成！共{len(atoms_list)}个构型，分为{num_batches}个set")

if __name__ == '__main__':
    convert_to_deepmd('aimd_trajectory.traj')
```

## 模型训练

### 输入文件配置(input.json)

```json
{
    "model": {
        "type_map": ["Cu", "O"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [60, 60],
            "rcut_smth": 0.50,
            "rcut": 6.00,
            "neuron": [25, 50, 100],
            "resnet_dt": false,
            "axis_neuron": 16,
            "seed": 1
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": true,
            "seed": 1
        }
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 5000,
        "start_lr": 0.001,
        "stop_lr": 3.51e-8
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1,
        "start_pref_f": 1000,
        "limit_pref_f": 1,
        "start_pref_v": 0,
        "limit_pref_v": 0
    },
    "training": {
        "training_data": {
            "systems": ["dataset/"],
            "batch_size": "auto"
        },
        "validation_data": {
            "systems": ["dataset_val/"],
            "batch_size": 1
        },
        "numb_steps": 1000000,
        "seed": 1,
        "disp_file": "lcurve.out",
        "disp_freq": 1000,
        "save_freq": 10000
    }
}
```

### 参数说明

**描述符(Descriptor)**:
- `type`: "se_e2_a"(平滑版DeepPot-SE)
- `sel`: 每种元素的最大邻居数
- `rcut`: 截断半径(Å)
- `neuron`: 嵌入网络的隐藏层神经元数

**拟合网络(Fitting Net)**:
- `neuron`: 拟合网络的隐藏层神经元数
- `resnet_dt`: 是否使用残差连接

**损失函数(Loss)**:
- `start_pref_e/limit_pref_e`: 能量权重(开始/极限)
- `start_pref_f/limit_pref_f`: 力权重(开始/极限)
- 通常前期重视力，后期重视能量

### 训练命令

```bash
# 开始训练
dp train input.json

# 多GPU训练
dp train input.json --gpus 0,1,2,3

# 从检查点继续
dp train input.json --restart model.ckpt
```

### 监控训练

```bash
# 查看学习曲线
tail -f lcurve.out

# 使用TensorBoard
tensorboard --logdir=./
```

## 模型评估

### 冻结模型

```bash
# 冻结最终模型
dp freeze -o model.pb

# 冻结特定检查点
dp freeze -c model.ckpt-100000 -o model_100k.pb
```

### 模型测试

```python
#!/usr/bin/env python3
"""
测试DeePMD模型
"""
from deepmd.infer import DeepPot
import numpy as np

# 加载模型
dp = DeepPot('model.pb')

# 准备输入
coord = np.array([...])  # 原子坐标 (N, 3)
cell = np.array([...])   # 晶胞 (3, 3)
atype = np.array([...])  # 原子类型 (N,)

# 预测
e, f, v = dp.eval(coord, cell, atype)

print(f"Energy: {e} eV")
print(f"Forces shape: {f.shape}")
```

### 与DFT对比

```python
import matplotlib.pyplot as plt

# 读取测试集
dft_energies = [...]
dp_energies = [...]

# 绘制对比图
plt.figure(figsize=(8, 8))
plt.scatter(dft_energies, dp_energies, alpha=0.5)
plt.plot([min(dft_energies), max(dft_energies)],
         [min(dft_energies), max(dft_energies)],
         'r--', label='Ideal')
plt.xlabel('DFT Energy (eV)')
plt.ylabel('DeePMD Energy (eV)')
plt.title('Energy Prediction Accuracy')
plt.legend()
plt.savefig('energy_parity.png', dpi=300)
```

## 在LAMMPS中使用

```lammps
# in.lammps
units metal
atom_style atomic
boundary p p p

# 读取结构
read_data data.lammps

# DeePMD势
pair_style deepmd model.pb
pair_coeff * *

# NVT模拟
velocity all create 300.0 12345
fix 1 all nvt temp 300.0 300.0 0.1

thermo 100
dump 1 all custom 1000 traj.lammpstrj id type x y z

timestep 0.001
run 100000
```

## 模型优化

### 压缩模型

```bash
# 减小模型大小(通常10-50倍)
dp compress -i model.pb -o model_compressed.pb
```

### 主动学习

迭代改进模型：

1. 使用当前模型进行MD
2. 识别不确定性高的构型
3. DFT计算这些构型
4. 添加到训练集
5. 重新训练模型
6. 重复1-5

## 练习题

### 练习1：数据准备
将Part2的AIMD数据转换为DeePMD格式。

### 练习2：模型训练
训练一个Cu的DeePMD模型，监控学习曲线。

### 练习3：模型评估
在测试集上评估模型，计算MAE和RMSE。

### 练习4：MD模拟
使用训练好的模型在LAMMPS中进行液态Cu的MD模拟，计算RDF。

## 常见问题

### Q1: 训练loss不下降？

**检查**:
- 学习率是否合适？
- 数据是否归一化？
- 训练集是否有代表性？

### Q2: 力的误差很大？

**解决**:
- 增加`start_pref_f`权重
- 增加训练数据
- 检查数据质量

### Q3: 模型在新体系上表现差？

**原因**: 训练集覆盖不足

**解决**: 主动学习，添加新体系的数据

## 扩展资源

- [DeePMD-kit官方文档](https://docs.deepmodeling.com/projects/deepmd/)
- [DeePMD-kit GitHub](https://github.com/deepmodeling/deepmd-kit)
- [DeePMD论文集](https://deepmodeling.com/papers)

---

**返回**: [Part 3主页](../../README.md)
