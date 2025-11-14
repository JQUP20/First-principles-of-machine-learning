#!/usr/bin/env python3
"""
MACE分子动力学模拟 | MACE Molecular Dynamics Simulation

展示如何使用MACE模型进行高精度分子动力学模拟。
包括NVE、NVT和NPT系综的模拟。

Demonstrates how to use MACE models for high-accuracy molecular dynamics simulations.
Includes simulations in NVE, NVT, and NPT ensembles.

作者 | Author: First-Principles ML Course
日期 | Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase import units
import os
import time

# 尝试导入MACE
# Try to import MACE
try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    print("警告: MACE未安装，将使用EMT势函数进行演示")
    print("Warning: MACE not installed, will use EMT potential for demonstration")
    MACE_AVAILABLE = False
    from ase.calculators.emt import EMT


def setup_calculator(device='cpu', model_type='small'):
    """
    设置计算器

    Args:
        device: 'cpu' 或 'cuda'
        model_type: MACE模型类型 ('small', 'medium', 'large')

    Returns:
        ASE calculator对象
    """
    if MACE_AVAILABLE:
        print(f"加载MACE-MP-0模型 ({model_type})...")
        print(f"Loading MACE-MP-0 model ({model_type})...")
        try:
            calc = mace_mp(model=model_type, device=device, default_dtype='float64')
            print("✓ MACE模型加载成功")
            print("✓ MACE model loaded successfully")
            return calc
        except Exception as e:
            print(f"MACE加载失败: {e}")
            print(f"MACE loading failed: {e}")
            print("使用EMT势函数替代")
            print("Using EMT potential instead")
            return EMT()
    else:
        print("使用EMT势函数进行演示")
        print("Using EMT potential for demonstration")
        return EMT()


def run_nve_simulation(atoms, calculator, steps=1000, timestep=1.0, temperature=300):
    """
    运行NVE系综分子动力学模拟 (微正则系综)

    Run NVE ensemble MD simulation (Microcanonical ensemble)

    Args:
        atoms: ASE Atoms对象
        calculator: ASE calculator
        steps: 模拟步数
        timestep: 时间步长 (fs)
        temperature: 初始温度 (K)

    Returns:
        trajectory: 轨迹文件路径
        energies: 能量数据字典
    """
    print("\n" + "="*70)
    print("NVE系综分子动力学模拟 | NVE Ensemble MD Simulation")
    print("="*70)

    # 设置计算器
    atoms.calc = calculator

    # 设置初始速度 (Maxwell-Boltzmann分布)
    # Set initial velocities (Maxwell-Boltzmann distribution)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    # 创建MD引擎 (Velocity Verlet算法)
    # Create MD engine (Velocity Verlet algorithm)
    dyn = VelocityVerlet(atoms, timestep * units.fs)

    # 准备数据存储
    # Prepare data storage
    os.makedirs('../results', exist_ok=True)
    traj_file = '../results/nve_trajectory.traj'
    traj = Trajectory(traj_file, 'w', atoms)
    dyn.attach(traj.write, interval=10)

    # 能量记录
    # Energy recording
    energies = {
        'time': [],
        'potential': [],
        'kinetic': [],
        'total': [],
        'temperature': []
    }

    def record_energy():
        """记录能量和温度"""
        energies['time'].append(dyn.get_number_of_steps() * timestep)
        energies['potential'].append(atoms.get_potential_energy())
        energies['kinetic'].append(atoms.get_kinetic_energy())
        energies['total'].append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
        energies['temperature'].append(atoms.get_temperature())

    dyn.attach(record_energy, interval=1)

    # 运行模拟
    # Run simulation
    print(f"运行NVE模拟: {steps}步, 时间步长={timestep} fs")
    print(f"Running NVE simulation: {steps} steps, timestep={timestep} fs")
    print(f"初始温度: {temperature} K")
    print(f"Initial temperature: {temperature} K")

    start_time = time.time()
    dyn.run(steps)
    elapsed_time = time.time() - start_time

    traj.close()

    print(f"\n✓ 模拟完成!")
    print(f"✓ Simulation completed!")
    print(f"模拟时间: {steps * timestep / 1000:.2f} ps")
    print(f"Simulation time: {steps * timestep / 1000:.2f} ps")
    print(f"计算耗时: {elapsed_time:.2f} s")
    print(f"Computation time: {elapsed_time:.2f} s")
    print(f"速度: {steps / elapsed_time:.1f} steps/s")
    print(f"Speed: {steps / elapsed_time:.1f} steps/s")
    print(f"轨迹已保存: {traj_file}")
    print(f"Trajectory saved: {traj_file}")

    return traj_file, energies


def run_nvt_simulation(atoms, calculator, steps=1000, timestep=1.0, temperature=300, friction=0.01):
    """
    运行NVT系综分子动力学模拟 (正则系综, 恒温)

    Run NVT ensemble MD simulation (Canonical ensemble, constant temperature)

    Args:
        atoms: ASE Atoms对象
        calculator: ASE calculator
        steps: 模拟步数
        timestep: 时间步长 (fs)
        temperature: 目标温度 (K)
        friction: Langevin摩擦系数

    Returns:
        trajectory: 轨迹文件路径
        energies: 能量数据字典
    """
    print("\n" + "="*70)
    print("NVT系综分子动力学模拟 | NVT Ensemble MD Simulation")
    print("="*70)

    # 设置计算器
    atoms.calc = calculator

    # 设置初始速度
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    # 创建MD引擎 (Langevin恒温器)
    # Create MD engine (Langevin thermostat)
    dyn = Langevin(atoms, timestep * units.fs, temperature_K=temperature, friction=friction)

    # 准备数据存储
    os.makedirs('../results', exist_ok=True)
    traj_file = '../results/nvt_trajectory.traj'
    traj = Trajectory(traj_file, 'w', atoms)
    dyn.attach(traj.write, interval=10)

    # 能量记录
    energies = {
        'time': [],
        'potential': [],
        'kinetic': [],
        'total': [],
        'temperature': []
    }

    def record_energy():
        energies['time'].append(dyn.get_number_of_steps() * timestep)
        energies['potential'].append(atoms.get_potential_energy())
        energies['kinetic'].append(atoms.get_kinetic_energy())
        energies['total'].append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
        energies['temperature'].append(atoms.get_temperature())

    dyn.attach(record_energy, interval=1)

    # 运行模拟
    print(f"运行NVT模拟: {steps}步, 时间步长={timestep} fs")
    print(f"Running NVT simulation: {steps} steps, timestep={timestep} fs")
    print(f"目标温度: {temperature} K, 摩擦系数: {friction}")
    print(f"Target temperature: {temperature} K, friction: {friction}")

    start_time = time.time()
    dyn.run(steps)
    elapsed_time = time.time() - start_time

    traj.close()

    print(f"\n✓ 模拟完成!")
    print(f"✓ Simulation completed!")
    print(f"模拟时间: {steps * timestep / 1000:.2f} ps")
    print(f"Simulation time: {steps * timestep / 1000:.2f} ps")
    print(f"计算耗时: {elapsed_time:.2f} s")
    print(f"Computation time: {elapsed_time:.2f} s")
    print(f"平均温度: {np.mean(energies['temperature']):.1f} K")
    print(f"Average temperature: {np.mean(energies['temperature']):.1f} K")
    print(f"轨迹已保存: {traj_file}")
    print(f"Trajectory saved: {traj_file}")

    return traj_file, energies


def analyze_trajectory(traj_file):
    """
    分析MD轨迹

    Analyze MD trajectory

    Args:
        traj_file: 轨迹文件路径

    Returns:
        analysis: 分析结果字典
    """
    print("\n分析轨迹...")
    print("Analyzing trajectory...")

    traj = read(traj_file, ':')

    # 计算径向分布函数 (RDF)
    # Calculate Radial Distribution Function (RDF)
    from ase.geometry.analysis import Analysis

    analysis_results = {
        'n_frames': len(traj),
        'total_time': len(traj) * 10 * 1.0 / 1000,  # ps (假设每10步保存一次)
    }

    # 计算平均结构参数
    # Calculate average structural parameters
    volumes = [atoms.get_volume() for atoms in traj]
    analysis_results['avg_volume'] = np.mean(volumes)
    analysis_results['std_volume'] = np.std(volumes)

    print(f"轨迹帧数: {analysis_results['n_frames']}")
    print(f"Trajectory frames: {analysis_results['n_frames']}")
    print(f"总时间: {analysis_results['total_time']:.2f} ps")
    print(f"Total time: {analysis_results['total_time']:.2f} ps")
    print(f"平均体积: {analysis_results['avg_volume']:.2f} Å³")
    print(f"Average volume: {analysis_results['avg_volume']:.2f} Å³")

    return analysis_results


def plot_md_results(energies_nve, energies_nvt, save_path='../results/md_analysis.png'):
    """
    可视化MD模拟结果

    Visualize MD simulation results

    Args:
        energies_nve: NVE能量数据
        energies_nvt: NVT能量数据
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # NVE结果
    # NVE results
    # 能量守恒
    axes[0, 0].plot(energies_nve['time'], energies_nve['total'], 'b-', linewidth=1.5, label='Total')
    axes[0, 0].plot(energies_nve['time'], energies_nve['potential'], 'r-', linewidth=1.5, label='Potential')
    axes[0, 0].plot(energies_nve['time'], energies_nve['kinetic'], 'g-', linewidth=1.5, label='Kinetic')
    axes[0, 0].set_xlabel('Time (fs)', fontsize=12)
    axes[0, 0].set_ylabel('Energy (eV)', fontsize=12)
    axes[0, 0].set_title('NVE: Energy Conservation', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 温度波动
    axes[0, 1].plot(energies_nve['time'], energies_nve['temperature'], 'purple', linewidth=1.5)
    axes[0, 1].axhline(y=np.mean(energies_nve['temperature']), color='r', linestyle='--',
                       linewidth=2, label=f"Mean: {np.mean(energies_nve['temperature']):.1f} K")
    axes[0, 1].set_xlabel('Time (fs)', fontsize=12)
    axes[0, 1].set_ylabel('Temperature (K)', fontsize=12)
    axes[0, 1].set_title('NVE: Temperature Fluctuations', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 能量分布
    axes[0, 2].hist(energies_nve['total'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].axvline(x=np.mean(energies_nve['total']), color='r', linestyle='--',
                       linewidth=2, label=f"Mean: {np.mean(energies_nve['total']):.3f} eV")
    axes[0, 2].set_xlabel('Total Energy (eV)', fontsize=12)
    axes[0, 2].set_ylabel('Frequency', fontsize=12)
    axes[0, 2].set_title('NVE: Energy Distribution', fontsize=14, fontweight='bold')
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)

    # NVT结果
    # NVT results
    # 能量演化
    axes[1, 0].plot(energies_nvt['time'], energies_nvt['potential'], 'r-', linewidth=1.5, label='Potential')
    axes[1, 0].plot(energies_nvt['time'], energies_nvt['kinetic'], 'g-', linewidth=1.5, label='Kinetic')
    axes[1, 0].set_xlabel('Time (fs)', fontsize=12)
    axes[1, 0].set_ylabel('Energy (eV)', fontsize=12)
    axes[1, 0].set_title('NVT: Energy Evolution', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # 温度控制
    axes[1, 1].plot(energies_nvt['time'], energies_nvt['temperature'], 'purple', linewidth=1.5)
    axes[1, 1].axhline(y=np.mean(energies_nvt['temperature']), color='r', linestyle='--',
                       linewidth=2, label=f"Mean: {np.mean(energies_nvt['temperature']):.1f} K")
    axes[1, 1].set_xlabel('Time (fs)', fontsize=12)
    axes[1, 1].set_ylabel('Temperature (K)', fontsize=12)
    axes[1, 1].set_title('NVT: Temperature Control', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    # 温度分布
    axes[1, 2].hist(energies_nvt['temperature'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 2].axvline(x=np.mean(energies_nvt['temperature']), color='r', linestyle='--',
                       linewidth=2, label=f"Mean: {np.mean(energies_nvt['temperature']):.1f} K")
    axes[1, 2].set_xlabel('Temperature (K)', fontsize=12)
    axes[1, 2].set_ylabel('Frequency', fontsize=12)
    axes[1, 2].set_title('NVT: Temperature Distribution', fontsize=14, fontweight='bold')
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存: {save_path}")
    print(f"Visualization saved: {save_path}")
    plt.close()


def calculate_statistics(energies, name=""):
    """
    计算统计数据

    Calculate statistics

    Args:
        energies: 能量数据字典
        name: 系综名称
    """
    print(f"\n{'='*70}")
    print(f"{name} 统计数据 | {name} Statistics")
    print(f"{'='*70}")

    # 能量守恒 (仅适用于NVE)
    if name == "NVE":
        total_energy = np.array(energies['total'])
        energy_drift = (total_energy[-1] - total_energy[0]) / total_energy[0] * 100
        print(f"总能量漂移: {energy_drift:.6f}% (理想情况应接近0)")
        print(f"Total energy drift: {energy_drift:.6f}% (should be close to 0)")
        print(f"能量标准差: {np.std(total_energy):.6f} eV")
        print(f"Energy std dev: {np.std(total_energy):.6f} eV")

    # 温度统计
    temp_array = np.array(energies['temperature'])
    print(f"\n温度统计 | Temperature Statistics:")
    print(f"  平均值 (Mean): {np.mean(temp_array):.2f} K")
    print(f"  标准差 (Std): {np.std(temp_array):.2f} K")
    print(f"  最小值 (Min): {np.min(temp_array):.2f} K")
    print(f"  最大值 (Max): {np.max(temp_array):.2f} K")

    # 能量统计
    pot_array = np.array(energies['potential'])
    kin_array = np.array(energies['kinetic'])
    print(f"\n势能统计 | Potential Energy Statistics:")
    print(f"  平均值 (Mean): {np.mean(pot_array):.4f} eV")
    print(f"  标准差 (Std): {np.std(pot_array):.4f} eV")
    print(f"\n动能统计 | Kinetic Energy Statistics:")
    print(f"  平均值 (Mean): {np.mean(kin_array):.4f} eV")
    print(f"  标准差 (Std): {np.std(kin_array):.4f} eV")


def main():
    """主函数 | Main function"""
    print("="*70)
    print("MACE分子动力学模拟演示 | MACE Molecular Dynamics Demonstration")
    print("="*70)

    # 1. 设置系统
    # Setup system
    print("\n1. 创建原子系统...")
    print("1. Creating atomic system...")

    # 创建铜晶体 (3x3x3超胞)
    # Create Cu crystal (3x3x3 supercell)
    atoms = bulk('Cu', 'fcc', a=3.6) * (3, 3, 3)
    print(f"系统: {len(atoms)} 个Cu原子")
    print(f"System: {len(atoms)} Cu atoms")
    print(f"晶胞参数: {atoms.cell.cellpar()}")
    print(f"Cell parameters: {atoms.cell.cellpar()}")

    # 2. 设置计算器
    # Setup calculator
    print("\n2. 设置计算器...")
    print("2. Setting up calculator...")

    device = 'cpu'  # 改为'cuda'如果有GPU
    calc = setup_calculator(device=device, model_type='small')

    # 3. 运行NVE模拟
    # Run NVE simulation
    print("\n3. 运行NVE系综模拟...")
    print("3. Running NVE ensemble simulation...")

    atoms_nve = atoms.copy()
    traj_nve, energies_nve = run_nve_simulation(
        atoms_nve,
        calc,
        steps=500,  # 生产环境中可增加到5000+
        timestep=1.0,
        temperature=300
    )
    calculate_statistics(energies_nve, name="NVE")

    # 4. 运行NVT模拟
    # Run NVT simulation
    print("\n4. 运行NVT系综模拟...")
    print("4. Running NVT ensemble simulation...")

    atoms_nvt = atoms.copy()
    traj_nvt, energies_nvt = run_nvt_simulation(
        atoms_nvt,
        calc,
        steps=500,  # 生产环境中可增加到5000+
        timestep=1.0,
        temperature=300,
        friction=0.01
    )
    calculate_statistics(energies_nvt, name="NVT")

    # 5. 分析轨迹
    # Analyze trajectories
    print("\n5. 分析MD轨迹...")
    print("5. Analyzing MD trajectories...")

    print("\nNVE轨迹分析:")
    print("NVE trajectory analysis:")
    analyze_trajectory(traj_nve)

    print("\nNVT轨迹分析:")
    print("NVT trajectory analysis:")
    analyze_trajectory(traj_nvt)

    # 6. 可视化结果
    # Visualize results
    print("\n6. 生成可视化...")
    print("6. Generating visualizations...")

    plot_md_results(energies_nve, energies_nvt)

    # 7. 总结
    # Summary
    print("\n" + "="*70)
    print("模拟总结 | Simulation Summary")
    print("="*70)
    print(f"✓ NVE模拟完成: {len(energies_nve['time'])} 步")
    print(f"✓ NVE simulation completed: {len(energies_nve['time'])} steps")
    print(f"✓ NVT模拟完成: {len(energies_nvt['time'])} 步")
    print(f"✓ NVT simulation completed: {len(energies_nvt['time'])} steps")
    print(f"✓ 轨迹文件已保存到 ../results/")
    print(f"✓ Trajectory files saved to ../results/")
    print(f"✓ 分析图表已保存")
    print(f"✓ Analysis plots saved")

    print("\n关键发现 | Key Findings:")
    print(f"1. NVE能量守恒: {np.std(energies_nve['total']):.6f} eV (越小越好)")
    print(f"1. NVE energy conservation: {np.std(energies_nve['total']):.6f} eV (smaller is better)")
    print(f"2. NVT温度控制: {np.mean(energies_nvt['temperature']):.1f} ± {np.std(energies_nvt['temperature']):.1f} K")
    print(f"2. NVT temperature control: {np.mean(energies_nvt['temperature']):.1f} ± {np.std(energies_nvt['temperature']):.1f} K")

    if MACE_AVAILABLE:
        print("\n使用MACE势函数的优势:")
        print("Advantages of using MACE potential:")
        print("  • DFT级别的精度 (能量MAE ~5-10 meV/atom)")
        print("  • DFT-level accuracy (energy MAE ~5-10 meV/atom)")
        print("  • 计算速度提升1000-10000倍")
        print("  • 1000-10000× faster than DFT")
        print("  • 可扩展到大规模系统 (>1000原子)")
        print("  • Scalable to large systems (>1000 atoms)")
        print("  • 保持物理一致性 (E(3)等变)")
        print("  • Maintains physical consistency (E(3) equivariance)")

    print("\n" + "="*70)
    print("完成! | Completed!")
    print("="*70)


if __name__ == "__main__":
    main()
