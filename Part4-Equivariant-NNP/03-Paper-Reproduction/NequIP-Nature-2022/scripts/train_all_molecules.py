#!/usr/bin/env python3
"""
自动化训练所有MD17分子
Automated Training for All MD17 Molecules

This script automatically trains NequIP models on all MD17 molecules
with the exact hyperparameters from the Nature Communications paper.

Usage:
    python train_all_molecules.py --molecules all --gpus 0,1,2,3
    python train_all_molecules.py --molecules aspirin ethanol --gpus 0
"""

import os
import sys
import argparse
import subprocess
import yaml
import time
from pathlib import Path
import json

# 论文中的超参数
PAPER_HYPERPARAMETERS = {
    'num_layers': 5,
    'l_max': 2,
    'parity': True,
    'num_features': 64,
    'invariant_layers': 2,
    'invariant_neurons': 64,
    'num_basis': 8,
    'BesselBasis_trainable': True,
    'PolynomialCutoff_p': 6,
    'r_max': 4.0,
    'learning_rate': 0.005,
    'batch_size': 5,
    'max_epochs': 10000,
    'force_weight': 100
}

# 每个分子的元素组成
MOLECULE_ELEMENTS = {
    'aspirin': ['H', 'C', 'O'],
    'benzene': ['H', 'C'],
    'ethanol': ['H', 'C', 'O'],
    'malonaldehyde': ['H', 'C', 'O'],
    'naphthalene': ['H', 'C'],
    'salicylic_acid': ['H', 'C', 'O'],
    'toluene': ['H', 'C'],
    'uracil': ['H', 'C', 'N', 'O']
}

def create_config(molecule, n_train=1000, n_val=100, output_dir='../configs'):
    """
    为指定分子创建NequIP配置文件

    Args:
        molecule: 分子名称
        n_train: 训练集大小
        n_val: 验证集大小
        output_dir: 配置文件输出目录

    Returns:
        config_path: 配置文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    config = {
        # 数据集
        'root': f'../data/md17_{molecule}',
        'dataset': 'ase',
        'dataset_file_name': f'{molecule}.xyz',
        'chemical_symbols': MOLECULE_ELEMENTS[molecule],

        # 数据划分
        'n_train': n_train,
        'n_val': n_val,
        'dataset_statistics_stride': 1,

        # 网络架构（论文设置）
        'num_layers': PAPER_HYPERPARAMETERS['num_layers'],
        'l_max': PAPER_HYPERPARAMETERS['l_max'],
        'parity': PAPER_HYPERPARAMETERS['parity'],
        'num_features': PAPER_HYPERPARAMETERS['num_features'],
        'invariant_layers': PAPER_HYPERPARAMETERS['invariant_layers'],
        'invariant_neurons': PAPER_HYPERPARAMETERS['invariant_neurons'],

        # 径向网络
        'num_basis': PAPER_HYPERPARAMETERS['num_basis'],
        'BesselBasis_trainable': PAPER_HYPERPARAMETERS['BesselBasis_trainable'],
        'PolynomialCutoff_p': PAPER_HYPERPARAMETERS['PolynomialCutoff_p'],
        'r_max': PAPER_HYPERPARAMETERS['r_max'],

        # 归一化
        'avg_num_neighbors': 'auto',
        'use_sc': True,

        # 优化器
        'optimizer_name': 'Adam',
        'optimizer_params': {
            'amsgrad': False,
            'betas': [0.9, 0.999],
            'eps': 1.0e-8
        },

        # 学习率
        'learning_rate': PAPER_HYPERPARAMETERS['learning_rate'],
        'lr_scheduler_name': 'ReduceLROnPlateau',
        'lr_scheduler_patience': 50,
        'lr_scheduler_factor': 0.5,
        'lr_scheduler_min_lr': 1.0e-6,

        # 训练参数
        'batch_size': PAPER_HYPERPARAMETERS['batch_size'],
        'max_epochs': PAPER_HYPERPARAMETERS['max_epochs'],
        'train_val_split': 'random',
        'shuffle': True,
        'metrics_key': 'validation_loss',

        # Early stopping
        'early_stopping_patiences': {
            'validation_loss': 1000
        },

        # 梯度
        'gradient_clip_val': 10.0,

        # 损失函数
        'loss_coeffs': {
            'forces': [PAPER_HYPERPARAMETERS['force_weight'], 'PerSpeciesL1Loss'],
            'total_energy': [1, 'PerAtomMSELoss']
        },

        # 正则化
        'weight_decay': 0.0,

        # 输出
        'model_save_dir': f'../results/{molecule}',
        'wandb': False,
        'verbose': 'info',
        'log_batch_freq': 10,
        'log_epoch_freq': 1,
        'save_checkpoint_freq': -1,
        'save_ema': True,
        'save_ema_decay': 0.99,

        # 硬件
        'device': 'cuda',
        'default_dtype': 'float32',
        'allow_tf32': False
    }

    config_path = os.path.join(output_dir, f'nequip_paper_{molecule}.yaml')

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created config for {molecule}: {config_path}")

    return config_path

def train_molecule(molecule, config_path, gpu_id=0):
    """
    训练单个分子

    Args:
        molecule: 分子名称
        config_path: 配置文件路径
        gpu_id: GPU ID

    Returns:
        success: 是否成功
    """
    print(f"\n{'='*60}")
    print(f"Training {molecule} on GPU {gpu_id}")
    print(f"{'='*60}\n")

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 训练命令
    cmd = ['nequip-train', config_path]

    # 日志文件
    log_dir = f'../results/{molecule}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')

    # 运行训练
    start_time = time.time()

    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            # 实时输出
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()

            process.wait()

        elapsed_time = time.time() - start_time

        if process.returncode == 0:
            print(f"\n✓ Training completed for {molecule}")
            print(f"  Time: {elapsed_time/3600:.2f} hours")
            return True
        else:
            print(f"\n✗ Training failed for {molecule}")
            return False

    except Exception as e:
        print(f"\n✗ Error training {molecule}: {e}")
        return False

def parallel_train(molecules, gpus, n_train=1000, n_val=100):
    """
    并行训练多个分子

    Args:
        molecules: 分子列表
        gpus: GPU ID列表
        n_train: 训练集大小
        n_val: 验证集大小
    """
    import multiprocessing
    from multiprocessing import Pool

    # 创建所有配置
    configs = {}
    for molecule in molecules:
        config_path = create_config(molecule, n_train, n_val)
        configs[molecule] = config_path

    # 分配GPU
    tasks = []
    for i, molecule in enumerate(molecules):
        gpu_id = gpus[i % len(gpus)]
        tasks.append((molecule, configs[molecule], gpu_id))

    print(f"\n{'='*60}")
    print(f"Parallel Training Plan")
    print(f"{'='*60}")
    print(f"Molecules: {len(molecules)}")
    print(f"GPUs: {gpus}")
    print(f"Tasks:")
    for mol, _, gpu in tasks:
        print(f"  {mol} → GPU {gpu}")
    print()

    # 并行执行
    results = {}
    for mol, config, gpu in tasks:
        success = train_molecule(mol, config, gpu)
        results[mol] = success

    # 总结
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")

    successful = [m for m, s in results.items() if s]
    failed = [m for m, s in results.items() if not s]

    print(f"Successful: {len(successful)}/{len(molecules)}")
    if successful:
        print(f"  {', '.join(successful)}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(molecules)}")
        print(f"  {', '.join(failed)}")

    # 保存结果
    results_file = '../results/training_summary.json'
    with open(results_file, 'w') as f:
        json.dump({
            'successful': successful,
            'failed': failed,
            'config': {
                'n_train': n_train,
                'n_val': n_val,
                'hyperparameters': PAPER_HYPERPARAMETERS
            }
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Train NequIP on MD17 molecules')

    parser.add_argument('--molecules', nargs='+', default=['all'],
                        help='Molecules to train (default: all)')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU IDs (comma-separated, e.g., 0,1,2,3)')
    parser.add_argument('--n-train', type=int, default=1000,
                        help='Training set size (default: 1000)')
    parser.add_argument('--n-val', type=int, default=100,
                        help='Validation set size (default: 100)')
    parser.add_argument('--sequential', action='store_true',
                        help='Train sequentially instead of in parallel')

    args = parser.parse_args()

    # 解析GPU
    gpus = [int(g) for g in args.gpus.split(',')]

    # 解析分子
    all_molecules = list(MOLECULE_ELEMENTS.keys())
    if args.molecules == ['all']:
        molecules = all_molecules
    else:
        molecules = [m for m in args.molecules if m in MOLECULE_ELEMENTS]
        if len(molecules) != len(args.molecules):
            unknown = set(args.molecules) - set(molecules)
            print(f"Warning: Unknown molecules: {unknown}")

    if not molecules:
        print("Error: No valid molecules specified")
        return

    print("=" * 60)
    print("NequIP Training - Nature Communications Paper Reproduction")
    print("=" * 60)
    print(f"Molecules: {', '.join(molecules)}")
    print(f"GPUs: {gpus}")
    print(f"Training set size: {args.n_train}")
    print(f"Validation set size: {args.n_val}")
    print(f"Mode: {'Sequential' if args.sequential else 'Parallel'}")
    print()

    # 训练
    if args.sequential:
        # 顺序训练
        for i, molecule in enumerate(molecules):
            gpu_id = gpus[i % len(gpus)]
            config_path = create_config(molecule, args.n_train, args.n_val)
            train_molecule(molecule, config_path, gpu_id)
    else:
        # 并行训练
        parallel_train(molecules, gpus, args.n_train, args.n_val)

if __name__ == '__main__':
    main()
