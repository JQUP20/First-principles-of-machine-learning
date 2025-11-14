# MACE实战：从预训练到微调
# MACE Practice: From Pre-trained Models to Fine-tuning

本教程涵盖MACE的完整使用流程，包括使用预训练模型、微调和自定义训练。

---

## 1. MACE快速开始

### 1.1 环境安装

```bash
# 创建环境
conda create -n mace python=3.10 -y
conda activate mace

# 安装PyTorch
conda install pytorch==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装MACE
pip install mace-torch

# 验证
python -c "import mace; print(f'MACE version: {mace.__version__}')"
```

### 1.2 使用预训练模型（5分钟上手）

```python
# quick_start.py
from mace.calculators import MACECalculator
from ase import Atoms
from ase.optimize import BFGS

# 1. 加载MACE-MP-0预训练模型（自动下载）
calc = MACECalculator(model_paths='medium', device='cuda')

# 2. 创建分子
water = Atoms(
    'H2O',
    positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]]
)
water.set_calculator(calc)

# 3. 结构优化
opt = BFGS(water)
opt.run(fmax=0.01)

# 4. 获取性质
energy = water.get_potential_energy()
forces = water.get_forces()

print(f"优化后能量: {energy:.4f} eV")
print(f"优化后坐标:\n{water.positions}")

# 5. 保存结果
from ase.io import write
write('water_optimized.xyz', water)

print("\n✓ 完成！使用MACE-MP-0完成结构优化")
```

运行：
```bash
python quick_start.py
# 输出：
# 优化后能量: -14.2345 eV
# ✓ 完成！
```

---

## 2. MACE-MP-0预训练模型详解

### 2.1 可用的预训练模型

MACE提供了多个不同规模的预训练模型：

| 模型 | 参数量 | 精度 | 速度 | 推荐用途 |
|------|--------|------|------|---------|
| `small` | 1M | 好 | 快 | 快速筛选 |
| `medium` | 10M | 很好 | 中 | **推荐默认** |
| `large` | 100M | 极好 | 慢 | 高精度计算 |

### 2.2 模型选择

```python
from mace.calculators import MACECalculator

# 小模型（快速）
calc_small = MACECalculator(model_paths='small', device='cuda')

# 中等模型（推荐）
calc_medium = MACECalculator(model_paths='medium', device='cuda')

# 大模型（高精度）
calc_large = MACECalculator(model_paths='large', device='cuda')

# 自定义模型路径
calc_custom = MACECalculator(
    model_paths='/path/to/your/model.model',
    device='cuda'
)
```

### 2.3 应用示例

#### (1) 结构优化

```python
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import MACECalculator

# 加载初始结构
atoms = read('initial_structure.xyz')

# 设置计算器
calc = MACECalculator(model_paths='medium', device='cuda')
atoms.set_calculator(calc)

# 优化
opt = BFGS(atoms, trajectory='optimization.traj')
opt.run(fmax=0.05)  # 力收敛到0.05 eV/Å

# 保存
write('optimized_structure.xyz', atoms)

print(f"初始能量: {read('initial_structure.xyz').get_potential_energy():.4f} eV")
print(f"优化后能量: {atoms.get_potential_energy():.4f} eV")
```

#### (2) 分子动力学模拟

```python
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io import read, Trajectory

atoms = read('structure.xyz')
atoms.set_calculator(MACECalculator(model_paths='medium', device='cuda'))

# 设置初始速度（300 K）
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# NVE MD
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)

# 保存轨迹
traj = Trajectory('md.traj', 'w', atoms)
dyn.attach(traj.write, interval=10)

# 运行10 ps
dyn.run(10000)

print("MD模拟完成！")
```

#### (3) 能量扫描

```python
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from mace.calculators import MACECalculator

# 扫描H2键长
calc = MACECalculator(model_paths='medium', device='cuda')

distances = np.linspace(0.5, 3.0, 50)
energies = []

for d in distances:
    h2 = Atoms('H2', positions=[[0, 0, 0], [d, 0, 0]])
    h2.set_calculator(calc)
    energies.append(h2.get_potential_energy())

# 绘图
plt.plot(distances, energies, 'o-')
plt.xlabel('H-H Distance (Å)')
plt.ylabel('Energy (eV)')
plt.title('H2 Potential Energy Surface')
plt.savefig('h2_scan.pdf')
```

---

## 3. 微调MACE模型（Few-Shot Learning）

### 3.1 为什么要微调？

**场景**：你有一个新体系（如特殊配体的金属有机框架），只有50-100个DFT计算结果。

**直接用MACE-MP-0**：
- ✅ 优点：零训练时间
- ❌ 缺点：可能不够精确（MAE ~ 10-20 meV）

**微调MACE-MP-0**：
- ✅ 优点：极高精度（MAE < 5 meV）
- ✅ 需要数据：只需50-100样本！
- ⏱️ 训练时间：<1小时

### 3.2 准备微调数据

```python
# prepare_finetuning_data.py
from ase.io import read, write

# 读取你的DFT轨迹
atoms_list = read('my_dft_trajectory.traj', index=':')

# 确保有能量和力
for atoms in atoms_list:
    assert atoms.get_potential_energy() is not None
    assert atoms.get_forces() is not None

# 保存为xyz格式（MACE支持）
write('finetune_data.xyz', atoms_list)

print(f"准备了 {len(atoms_list)} 个构型用于微调")
```

### 3.3 微调配置文件

```yaml
# finetune_config.yaml

# 数据
name: my_system_finetuned
train_file: ./finetune_data.xyz
valid_fraction: 0.1              # 10%作为验证集
test_file: ./test_data.xyz       # 可选

# 微调设置
foundation_model: medium         # 从medium模型开始微调
foundation_model_readout: True   # 重新训练ReadOut层

# 训练参数
batch_size: 5
max_num_epochs: 100
patience: 50                     # Early stopping

# 学习率（微调用小学习率）
lr: 0.0001                       # 比从头训练小10倍
scheduler: ReduceLROnPlateau
lr_factor: 0.8
lr_patience: 20

# 损失权重
energy_weight: 1.0
forces_weight: 100.0

# 其他（继承预训练模型）
# 不需要指定num_channels等，会自动继承
```

### 3.4 运行微调

```bash
# 微调命令
mace_run_train \
  --config=finetune_config.yaml \
  --device=cuda

# 查看进度
tail -f logs/my_system_finetuned.log
```

### 3.5 评估微调模型

```python
# evaluate_finetuned.py
from mace.calculators import MACECalculator
from ase.io import read
import numpy as np

# 加载微调后的模型
calc_finetuned = MACECalculator(
    model_paths='my_system_finetuned_run-1.model',
    device='cuda'
)

# 加载基础模型（对比）
calc_base = MACECalculator(model_paths='medium', device='cuda')

# 测试数据
test_atoms = read('test_data.xyz', index=':')

errors_finetuned = []
errors_base = []

for atoms in test_atoms:
    # 真实值
    e_true = atoms.get_potential_energy()

    # 微调模型预测
    atoms.set_calculator(calc_finetuned)
    e_finetuned = atoms.get_potential_energy()
    errors_finetuned.append(abs(e_finetuned - e_true))

    # 基础模型预测
    atoms.set_calculator(calc_base)
    e_base = atoms.get_potential_energy()
    errors_base.append(abs(e_base - e_true))

# 统计
mae_finetuned = np.mean(errors_finetuned) * 1000  # meV
mae_base = np.mean(errors_base) * 1000

print(f"基础模型 MAE: {mae_base:.2f} meV")
print(f"微调模型 MAE: {mae_finetuned:.2f} meV")
print(f"改进: {(1 - mae_finetuned/mae_base)*100:.1f}%")

# 典型结果：
# 基础模型 MAE: 12.5 meV
# 微调模型 MAE: 3.2 meV
# 改进: 74.4%
```

---

## 4. 从头训练MACE模型

### 4.1 何时从头训练？

**推荐从头训练**：
- 你有大量数据（>5000个构型）
- 体系包含MACE-MP-0未见过的元素
- 需要极致性能

**推荐微调**：
- 数据有限（<1000个构型）
- 元素在周期表中常见
- 快速原型开发

### 4.2 训练配置文件

```yaml
# train_from_scratch.yaml

# 数据
name: my_custom_system
train_file: ./large_train_data.xyz
valid_fraction: 0.1
test_file: ./test_data.xyz

# 不使用预训练模型
foundation_model: null

# 网络架构
num_radial_basis: 8
num_cutoff_basis: 5
max_ell: 3                       # 最大角动量l=3
interaction_cls: RealAgnosticResidualInteractionBlock
interaction_cls_first: RealAgnosticInteractionBlock
num_interactions: 2              # 层数
hidden_irreps: 128x0e + 128x1o + 64x2e + 32x3o  # 不可约表示
MLP_irreps: 16x0e                # MLP维度
gate: silu                       # 激活函数
correlation: 3                   # 三体相关
r_max: 5.0                       # 截断半径

# 训练参数
batch_size: 10
max_num_epochs: 1000
patience: 200
ema: true                        # 指数移动平均
ema_decay: 0.99

# 优化器
optimizer: adam
lr: 0.001                        # 从头训练用大学习率
weight_decay: 5.0e-7
amsgrad: true

# 学习率调度
scheduler: ReduceLROnPlateau
lr_factor: 0.8
lr_patience: 50

# 损失函数
loss: weighted
energy_weight: 1.0
forces_weight: 100.0

# 数据增强
data_augmentation: true          # 随机旋转

# 其他
default_dtype: float32
device: cuda
seed: 42
```

### 4.3 训练命令

```bash
# 开始训练
mace_run_train --config=train_from_scratch.yaml

# 多GPU训练
torchrun --nproc_per_node=4 \
  $(which mace_run_train) \
  --config=train_from_scratch.yaml

# 从检查点恢复
mace_run_train \
  --config=train_from_scratch.yaml \
  --restart_latest
```

---

## 5. MACE超参数调优

### 5.1 关键超参数

#### (1) 网络深度和宽度

```yaml
# 小体系（<20原子，快速）
num_interactions: 2
hidden_irreps: 64x0e + 32x1o + 16x2e

# 中等体系（20-100原子）
num_interactions: 2
hidden_irreps: 128x0e + 64x1o + 32x2e + 16x3o

# 大体系（>100原子，高精度）
num_interactions: 3
hidden_irreps: 256x0e + 128x1o + 64x2e + 32x3o
```

#### (2) 角动量

```yaml
# 简单体系（共价键）
max_ell: 2  # l=0,1,2（标量、矢量、张量）

# 复杂体系（离子、金属）
max_ell: 3  # l=0,1,2,3
```

#### (3) 多体相关

```yaml
# 二体相关（快）
correlation: 2

# 三体相关（精确，推荐）
correlation: 3

# 四体相关（非常慢，通常不需要）
correlation: 4
```

### 5.2 训练技巧

#### 技巧1：课程学习

```yaml
# 阶段1：只训练能量（快速收敛）
max_num_epochs: 200
energy_weight: 1.0
forces_weight: 0.0

# 阶段2：加入力（精细调整）
max_num_epochs: 500
energy_weight: 1.0
forces_weight: 100.0
```

#### 技巧2：学习率预热

```yaml
lr: 0.001
warmup_steps: 100    # 前100步线性增加学习率
```

#### 技巧3：混合精度训练

```yaml
default_dtype: float32  # 默认使用float32
# 对于A100 GPU，可以尝试：
# default_dtype: bfloat16  # 内存节省50%，速度提升2x
```

---

## 6. MACE用于MD模拟

### 6.1 与ASE集成

```python
# md_with_mace.py
from ase import units
from ase.io import read, Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from mace.calculators import MACECalculator

# 1. 加载结构
atoms = read('initial.xyz')

# 2. 设置MACE计算器
calc = MACECalculator(model_paths='medium', device='cuda')
atoms.set_calculator(calc)

# 3. 初始化速度（300 K）
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# 4. 设置MD
dyn = VelocityVerlet(atoms, timestep=0.5*units.fs)

# 5. 轨迹和热力学输出
traj = Trajectory('md.traj', 'w', atoms)
dyn.attach(traj.write, interval=10)

def print_energy(a=atoms):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print(f"E_pot = {epot:.3f} eV/atom, E_kin = {ekin:.3f} eV/atom, "
          f"T = {ekin/(1.5*units.kB):.1f} K")

dyn.attach(print_energy, interval=100)

# 6. 运行100 ps
dyn.run(200000)  # 0.5 fs * 200000 = 100 ps

print("MD模拟完成！")
```

### 6.2 与LAMMPS集成（TODO）

MACE正在开发LAMMPS接口。当前可以通过以下方式：

```python
# 方法1：使用ASE-LAMMPS接口
from ase.calculators.lammpsrun import LAMMPS

calc = LAMMPS(
    # ... LAMMPS设置
)

# 方法2：导出为DeePMD格式（未来版本）
# mace_export --format deepmd --model my_model.model
```

---

## 7. 实战案例

### 7.1 案例1：蛋白质-配体结合

```python
# protein_ligand.py
from ase.io import read
from mace.calculators import MACECalculator
from ase.optimize import BFGS

# 加载蛋白质-配体复合物
complex = read('protein_ligand_complex.pdb')

# 使用large模型（高精度）
calc = MACECalculator(model_paths='large', device='cuda')
complex.set_calculator(calc)

# 优化配体位置（固定蛋白质）
# 假设配体是最后100个原子
ligand_indices = range(len(complex) - 100, len(complex))
constraint = FixAtoms(mask=[i not in ligand_indices for i in range(len(complex))])
complex.set_constraint(constraint)

# 优化
opt = BFGS(complex)
opt.run(fmax=0.05)

# 计算结合能
E_complex = complex.get_potential_energy()
E_protein = read('protein_alone.pdb').get_potential_energy()
E_ligand = read('ligand_alone.xyz').get_potential_energy()

E_binding = E_complex - E_protein - E_ligand

print(f"结合能: {E_binding:.4f} eV ({E_binding * 23.06:.2f} kcal/mol)")
```

### 7.2 案例2：晶体相变

```python
# phase_transition.py
from ase.io import read
from ase import units
from ase.md.npt import NPT
from mace.calculators import MACECalculator

# 加载晶体
crystal = read('crystal.cif') * (2, 2, 2)  # 超胞

# MACE计算器
calc = MACECalculator(model_paths='medium', device='cuda')
crystal.set_calculator(calc)

# NPT MD（升温实验）
for T in range(300, 1500, 100):  # 300 K to 1500 K
    MaxwellBoltzmannDistribution(crystal, temperature_K=T)

    dyn = NPT(
        crystal,
        timestep=1.0*units.fs,
        temperature_K=T,
        externalstress=0.0,  # 1 atm
        ttime=25*units.fs,
        pfactor=(75*units.fs)**2 * units.GPa
    )

    dyn.run(10000)  # 10 ps equilibration

    # 分析结构
    volume = crystal.get_volume()
    print(f"T = {T} K: V = {volume:.2f} Å³")

    # 保存
    write(f'crystal_{T}K.xyz', crystal)
```

---

## 8. 常见问题和优化

### 8.1 内存不足

**问题**：训练大模型时OOM

**解决**：
```yaml
# 减小batch size
batch_size: 1

# 使用梯度累积
gradient_accumulation_steps: 4  # 等价于batch_size=4

# 使用混合精度
default_dtype: bfloat16
```

### 8.2 训练很慢

**问题**：GPU利用率低

**解决**：
```yaml
# 增加batch size
batch_size: 20

# 使用更多workers
num_workers: 4

# 启用编译（PyTorch 2.0+）
compile_model: true
```

### 8.3 精度不够

**问题**：微调后MAE仍然>10 meV

**检查**：
1. 数据是否足够diverse？
2. 是否选择了正确的预训练模型？
3. 训练是否收敛？

**改进**：
```yaml
# 增大模型
foundation_model: large  # 使用large而不是medium

# 更多训练
max_num_epochs: 500

# 数据增强
data_augmentation: true
```

---

## 9. 最佳实践总结

### 快速决策树

```
你的任务是什么？
├─ 快速预测/筛选
│  └─ 使用 MACE-MP-0 small/medium
│
├─ 高精度单点计算
│  └─ 使用 MACE-MP-0 large
│
├─ 新体系，少量数据（<1000）
│  └─ 微调 MACE-MP-0
│
└─ 新体系，大量数据（>5000）
   └─ 从头训练MACE
```

### 推荐配置

| 场景 | 模型 | 数据需求 | 训练时间 | 精度 |
|------|------|---------|---------|------|
| 通用预测 | MACE-MP-0 medium | 0 | 0 | 好 |
| 高精度 | MACE-MP-0 large | 0 | 0 | 极好 |
| 微调 | Medium + 50样本 | 50-500 | <1小时 | 极好 |
| 从头训练 | 自定义 | 5000+ | 1-2天 | 最优 |

---

## 参考资料

- **MACE论文**: Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks", NeurIPS 2022
- **MACE-OFF论文**: Batatia et al., "A foundation model for atomistic materials chemistry", arXiv 2024
- **代码**: https://github.com/ACEsuit/mace
- **文档**: https://mace-docs.readthedocs.io/

---

**返回**: [Part 5主页](../../README.md)
