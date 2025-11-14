#!/usr/bin/env python3
"""
MACE-MP-0模型加载和微调
Loading and Fine-tuning MACE-MP-0 Model

本示例展示如何:
1. 加载预训练的MACE-MP-0模型
2. 在自定义数据集上微调
3. 评估微调后的模型性能
4. 与从头训练进行对比

This example demonstrates how to:
1. Load pre-trained MACE-MP-0 model
2. Fine-tune on custom dataset
3. Evaluate fine-tuned model performance
4. Compare with training from scratch

作者 | Author: First-Principles ML Course
"""

import os
import sys
import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from ase.build import bulk
from ase.calculators.emt import EMT
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

print("=" * 80)
print("MACE-MP-0模型加载和微调示例")
print("MACE-MP-0 Model Loading and Fine-tuning Example")
print("=" * 80)
print()

# ============================================================================
# 检查MACE安装
# Check MACE Installation
# ============================================================================

try:
    import mace
    from mace.calculators import MACECalculator, mace_mp
    print(f"✓ MACE版本: {mace.__version__}")
    print(f"✓ MACE version: {mace.__version__}")
except ImportError:
    print("错误: MACE未安装")
    print("Error: MACE not installed")
    print("请运行: pip install mace-torch")
    print("Please run: pip install mace-torch")
    sys.exit(1)

print()

# ============================================================================
# 第一步: 生成自定义训练数据
# Step 1: Generate Custom Training Data
# ============================================================================

print("第一步: 生成自定义训练数据")
print("Step 1: Generating custom training data")
print("-" * 80)

def generate_training_data(
    n_configs: int = 100,
    crystal_type: str = 'fcc',
    element: str = 'Cu',
    data_dir: str = '../data'
) -> str:
    """
    生成用于微调的训练数据

    Args:
        n_configs: 配置数量
        crystal_type: 晶体类型
        element: 元素
        data_dir: 数据保存目录

    Returns:
        训练数据文件路径
    """
    os.makedirs(data_dir, exist_ok=True)

    print(f"生成{n_configs}个{element}晶体配置...")
    print(f"Generating {n_configs} {element} crystal configurations...")

    # 创建基础结构
    atoms_list = []

    for i in range(n_configs):
        # 在平衡晶格常数附近随机扰动
        if element == 'Cu':
            a = 3.6 + np.random.uniform(-0.2, 0.2)
        elif element == 'Al':
            a = 4.05 + np.random.uniform(-0.2, 0.2)
        else:
            a = 4.0 + np.random.uniform(-0.2, 0.2)

        # 创建晶体
        atoms = bulk(element, crystal_type, a=a)

        # 创建2x2x2超胞
        atoms = atoms * (2, 2, 2)

        # 添加随机位移模拟热振动
        displacement = np.random.normal(0, 0.05, size=atoms.positions.shape)
        atoms.positions += displacement

        # 使用EMT计算器计算能量和力（真实应用中应使用DFT）
        # In real applications, use DFT instead of EMT
        calc = EMT()
        atoms.calc = calc

        # 计算能量和力
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        # 保存信息到atoms对象
        atoms.info['energy'] = energy
        atoms.arrays['forces'] = forces

        atoms_list.append(atoms)

        if (i + 1) % 20 == 0:
            print(f"  已生成 {i+1}/{n_configs} 个配置")
            print(f"  Generated {i+1}/{n_configs} configurations")

    # 保存为XYZ文件
    train_file = os.path.join(data_dir, f'{element}_training.xyz')
    write(train_file, atoms_list)

    print(f"✓ 训练数据已保存到: {train_file}")
    print(f"✓ Training data saved to: {train_file}")
    print()

    return train_file


# 生成训练数据
train_file = generate_training_data(n_configs=100, element='Cu')

# 划分训练集和验证集
print("划分训练集和验证集...")
print("Splitting training and validation sets...")

all_data = read(train_file, ':')
n_total = len(all_data)
n_train = int(0.8 * n_total)

train_data = all_data[:n_train]
val_data = all_data[n_train:]

train_file = '../data/Cu_train.xyz'
val_file = '../data/Cu_val.xyz'

write(train_file, train_data)
write(val_file, val_data)

print(f"  训练集: {len(train_data)} 个配置")
print(f"  Training set: {len(train_data)} configurations")
print(f"  验证集: {len(val_data)} 个配置")
print(f"  Validation set: {len(val_data)} configurations")
print()

# ============================================================================
# 第二步: 加载预训练的MACE-MP-0模型
# Step 2: Load Pre-trained MACE-MP-0 Model
# ============================================================================

print("第二步: 加载预训练的MACE-MP-0模型")
print("Step 2: Loading pre-trained MACE-MP-0 model")
print("-" * 80)

# 检查CUDA可用性
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")
print(f"Using device: {device}")
print()

try:
    # 加载MACE-MP-0模型（small版本用于演示）
    print("加载MACE-MP-0模型...")
    print("Loading MACE-MP-0 model...")

    # 方式1: 使用mace_mp函数（推荐）
    calc_pretrained = mace_mp(
        model="small",  # 可选: "small", "medium", "large"
        dispersion=False,
        default_dtype="float32",
        device=device
    )

    print("✓ MACE-MP-0模型加载成功")
    print("✓ MACE-MP-0 model loaded successfully")
    print()

    # 测试预训练模型
    print("测试预训练模型...")
    print("Testing pre-trained model...")

    test_atoms = train_data[0].copy()
    test_atoms.calc = calc_pretrained

    energy_pretrained = test_atoms.get_potential_energy()
    forces_pretrained = test_atoms.get_forces()

    print(f"  预训练模型预测能量: {energy_pretrained:.4f} eV")
    print(f"  Pre-trained model energy: {energy_pretrained:.4f} eV")
    print(f"  预训练模型力范数: {np.linalg.norm(forces_pretrained):.4f} eV/Å")
    print(f"  Pre-trained model force norm: {np.linalg.norm(forces_pretrained):.4f} eV/Å")
    print()

except Exception as e:
    print(f"警告: 无法加载MACE-MP-0模型")
    print(f"Warning: Could not load MACE-MP-0 model")
    print(f"错误信息: {e}")
    print(f"Error: {e}")
    print("将跳过预训练模型的使用")
    print("Will skip using pre-trained model")
    calc_pretrained = None
    print()

# ============================================================================
# 第三步: 微调MACE模型
# Step 3: Fine-tune MACE Model
# ============================================================================

print("第三步: 在自定义数据上微调MACE模型")
print("Step 3: Fine-tuning MACE model on custom data")
print("-" * 80)

# 注意: MACE的微调需要使用命令行工具或直接调用训练脚本
# Note: MACE fine-tuning requires using command-line tools or training scripts

print("MACE微调配置示例:")
print("Example MACE fine-tuning configuration:")
print()

finetuning_config = """
# MACE微调配置文件 | MACE Fine-tuning Configuration

# 数据文件
train_file: ../data/Cu_train.xyz
valid_file: ../data/Cu_val.xyz
test_file: ../data/Cu_val.xyz  # 使用验证集作为测试集

# 预训练模型
foundation_model: small  # 使用MACE-MP-0 small模型作为基础

# 模型架构（应与预训练模型匹配）
r_max: 5.0
num_interactions: 2
hidden_irreps: 128x0e + 128x1o
MLP_irreps: 16x0e
correlation: 3
max_ell: 3

# 微调策略
restart_latest: true
loss: huber  # 对异常值更鲁棒
default_dtype: float32

# 训练超参数（微调时使用更小的学习率）
learning_rate: 0.0005  # 比从头训练小10倍
batch_size: 16
max_num_epochs: 200
patience: 50
ema: true

# 损失权重
energy_weight: 1.0
forces_weight: 100.0

# 输出
model_dir: ../models/mace_finetuned
log_dir: ../results/logs

# 评估
eval_interval: 10
save_cpu: true
"""

config_file = '../models/finetune_config.yaml'
os.makedirs(os.path.dirname(config_file), exist_ok=True)

with open(config_file, 'w') as f:
    f.write(finetuning_config)

print(finetuning_config)
print()
print(f"配置文件已保存到: {config_file}")
print(f"Configuration saved to: {config_file}")
print()

# 运行微调的命令
finetune_command = f"""
要运行微调，请执行以下命令:
To run fine-tuning, execute the following command:

mace_run_train \\
    --config={config_file} \\
    --device={device}

或者使用Python API (如果可用):
Or use Python API (if available):

python -m mace.cli.run_train \\
    --config={config_file} \\
    --device={device}
"""

print(finetune_command)
print()

# ============================================================================
# 第四步: 评估微调后的模型
# Step 4: Evaluate Fine-tuned Model
# ============================================================================

print("第四步: 评估模型性能")
print("Step 4: Evaluating model performance")
print("-" * 80)

def evaluate_model(
    calculator,
    test_data: List[Atoms],
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    评估模型性能

    Args:
        calculator: MACE计算器
        test_data: 测试数据
        model_name: 模型名称

    Returns:
        包含误差指标的字典
    """
    print(f"评估 {model_name}...")
    print(f"Evaluating {model_name}...")

    energy_errors = []
    force_errors = []

    for atoms in test_data:
        # 获取参考值
        energy_ref = atoms.info['energy']
        forces_ref = atoms.arrays['forces']

        # 使用模型预测
        atoms_copy = atoms.copy()
        atoms_copy.calc = calculator

        try:
            energy_pred = atoms_copy.get_potential_energy()
            forces_pred = atoms_copy.get_forces()

            # 计算误差
            energy_error = abs(energy_pred - energy_ref)
            force_error = np.mean(np.abs(forces_pred - forces_ref))

            energy_errors.append(energy_error)
            force_errors.append(force_error)

        except Exception as e:
            print(f"  警告: 计算失败 - {e}")
            continue

    # 计算统计量
    results = {
        'energy_mae': np.mean(energy_errors),
        'energy_rmse': np.sqrt(np.mean(np.array(energy_errors)**2)),
        'force_mae': np.mean(force_errors),
        'force_rmse': np.sqrt(np.mean(np.array(force_errors)**2))
    }

    print(f"  能量MAE: {results['energy_mae']:.4f} eV")
    print(f"  能量RMSE: {results['energy_rmse']:.4f} eV")
    print(f"  力MAE: {results['force_mae']:.4f} eV/Å")
    print(f"  力RMSE: {results['force_rmse']:.4f} eV/Å")
    print()

    return results

# 评估预训练模型（如果可用）
if calc_pretrained is not None:
    results_pretrained = evaluate_model(
        calc_pretrained,
        val_data,
        "MACE-MP-0 (未微调)"
    )

# 注意: 要评估微调后的模型，需要先完成微调训练
print("注意: 要评估微调后的模型，请先运行上述微调命令")
print("Note: To evaluate fine-tuned model, first run the fine-tuning command above")
print()

# 如果微调模型存在，加载并评估
finetuned_model_path = '../models/mace_finetuned/MACE_model_run-123.model'
if os.path.exists(finetuned_model_path):
    print("找到微调后的模型，正在加载...")
    print("Found fine-tuned model, loading...")

    calc_finetuned = MACECalculator(
        model_paths=finetuned_model_path,
        device=device
    )

    results_finetuned = evaluate_model(
        calc_finetuned,
        val_data,
        "MACE微调后"
    )

    # 对比结果
    print("性能对比 | Performance Comparison")
    print("-" * 80)
    print(f"{'指标':<20} {'未微调':<15} {'微调后':<15} {'提升':<15}")
    print(f"{'Metric':<20} {'Pre-trained':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 80)

    for metric in ['energy_mae', 'force_mae']:
        before = results_pretrained[metric]
        after = results_finetuned[metric]
        improvement = (before - after) / before * 100

        print(f"{metric:<20} {before:<15.4f} {after:<15.4f} {improvement:<15.2f}%")

    print()

# ============================================================================
# 第五步: 可视化结果
# Step 5: Visualize Results
# ============================================================================

print("第五步: 可视化结果")
print("Step 5: Visualizing results")
print("-" * 80)

def visualize_predictions(
    calculator,
    test_data: List[Atoms],
    title: str = "Model Predictions"
):
    """可视化预测vs真实值"""

    energies_ref = []
    energies_pred = []
    forces_ref_norm = []
    forces_pred_norm = []

    for atoms in test_data[:50]:  # 只可视化前50个
        energy_ref = atoms.info['energy']
        forces_ref = atoms.arrays['forces']

        atoms_copy = atoms.copy()
        atoms_copy.calc = calculator

        try:
            energy_pred = atoms_copy.get_potential_energy()
            forces_pred = atoms_copy.get_forces()

            energies_ref.append(energy_ref)
            energies_pred.append(energy_pred)
            forces_ref_norm.append(np.linalg.norm(forces_ref))
            forces_pred_norm.append(np.linalg.norm(forces_pred))
        except:
            continue

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 能量预测
    ax1.scatter(energies_ref, energies_pred, alpha=0.6)
    ax1.plot([min(energies_ref), max(energies_ref)],
             [min(energies_ref), max(energies_ref)],
             'r--', label='Perfect prediction')
    ax1.set_xlabel('Reference Energy (eV)')
    ax1.set_ylabel('Predicted Energy (eV)')
    ax1.set_title(f'{title} - Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 力预测
    ax2.scatter(forces_ref_norm, forces_pred_norm, alpha=0.6)
    ax2.plot([min(forces_ref_norm), max(forces_ref_norm)],
             [min(forces_ref_norm), max(forces_ref_norm)],
             'r--', label='Perfect prediction')
    ax2.set_xlabel('Reference Force Norm (eV/Å)')
    ax2.set_ylabel('Predicted Force Norm (eV/Å)')
    ax2.set_title(f'{title} - Forces')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs('../results', exist_ok=True)
    output_file = f'../results/predictions_{title.replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_file}")
    print(f"Visualization saved to: {output_file}")

    plt.close()

# 可视化预训练模型结果
if calc_pretrained is not None:
    visualize_predictions(calc_pretrained, val_data, "MACE-MP-0")

print()

# ============================================================================
# 总结
# Summary
# ============================================================================

print("=" * 80)
print("示例完成! | Example completed!")
print("=" * 80)
print()

print("总结 | Summary:")
print("-" * 80)
print("✓ 生成了自定义训练数据")
print("✓ Generated custom training data")
print("✓ 加载了预训练的MACE-MP-0模型")
print("✓ Loaded pre-trained MACE-MP-0 model")
print("✓ 创建了微调配置文件")
print("✓ Created fine-tuning configuration")
print("✓ 评估了模型性能")
print("✓ Evaluated model performance")
print()

print("下一步 | Next Steps:")
print("-" * 80)
print("1. 运行微调命令训练模型")
print("   Run fine-tuning command to train model")
print()
print("2. 评估微调后的模型性能")
print("   Evaluate fine-tuned model performance")
print()
print("3. 将模型用于分子动力学模拟")
print("   Use model for molecular dynamics simulations")
print()
print("4. 查看 03_mace_md_simulation.py 了解MD模拟示例")
print("   See 03_mace_md_simulation.py for MD simulation example")
print()

print("微调的关键点 | Key Points for Fine-tuning:")
print("-" * 80)
print("• 使用更小的学习率 (0.0005 vs 0.01)")
print("  Use smaller learning rate (0.0005 vs 0.01)")
print("• 更少的训练轮数 (200 vs 1000)")
print("  Fewer training epochs (200 vs 1000)")
print("• 冻结部分层可能提高效率")
print("  Freezing some layers may improve efficiency")
print("• 监控验证损失避免过拟合")
print("  Monitor validation loss to avoid overfitting")
print()
