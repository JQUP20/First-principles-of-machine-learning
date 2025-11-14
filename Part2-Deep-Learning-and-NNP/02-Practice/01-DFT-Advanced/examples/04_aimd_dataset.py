#!/usr/bin/env python3
"""
AIMD数据集构建示例
AIMD Dataset Construction Example

本脚本演示如何使用从头算分子动力学(AIMD)生成训练数据集。
This script demonstrates how to generate training datasets using Ab Initio Molecular Dynamics.
"""

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units
from ase.io import write, read, Trajectory
from gpaw import GPAW, PW, FermiDirac
import time


def create_supercell(primitive_atoms, size=(2, 2, 2)):
    """
    创建超胞
    Create supercell

    Parameters:
    -----------
    primitive_atoms : ASE Atoms
        原胞
    size : tuple
        超胞尺寸
    """
    from ase.build import make_supercell

    supercell_matrix = np.diag(size)
    supercell = make_supercell(primitive_atoms, supercell_matrix)

    print(f"超胞尺寸: {size}")
    print(f"Supercell size: {size}")
    print(f"原子总数: {len(supercell)}")
    print(f"Total atoms: {len(supercell)}")

    return supercell


def setup_calculator(atoms, mode='fast'):
    """
    设置DFT计算器
    Setup DFT calculator

    Parameters:
    -----------
    atoms : ASE Atoms
        原子结构
    mode : str
        'fast' 或 'accurate'
    """
    if mode == 'fast':
        # 快速设置（用于测试）
        calc = GPAW(
            mode=PW(300),  # 较低的截断能
            xc='PBE',
            kpts=(2, 2, 2),  # 稀疏k点
            occupations=FermiDirac(0.1),
            txt='aimd.txt'
        )
    else:
        # 精确设置（用于生产）
        calc = GPAW(
            mode=PW(500),
            xc='PBE',
            kpts=(4, 4, 4),
            occupations=FermiDirac(0.1),
            convergence={'energy': 1e-5},
            txt='aimd.txt'
        )

    atoms.calc = calc
    return calc


def run_aimd(atoms, temperature=300, timestep=1.0, steps=100,
             trajectory_file='aimd_trajectory.traj',
             interval=1):
    """
    运行AIMD模拟
    Run AIMD simulation

    Parameters:
    -----------
    atoms : ASE Atoms
        原子结构
    temperature : float
        温度 (K)
    timestep : float
        时间步长 (fs)
    steps : int
        MD步数
    trajectory_file : str
        轨迹文件名
    interval : int
        保存间隔
    """
    print("\n" + "=" * 60)
    print(f"开始AIMD模拟: T={temperature}K, 步数={steps}")
    print(f"Starting AIMD: T={temperature}K, steps={steps}")
    print("=" * 60)

    # 初始化速度（Maxwell-Boltzmann分布）
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    # 移除系统总动量
    momentum = atoms.get_momenta().sum(axis=0)
    atoms.set_momenta(atoms.get_momenta() - momentum / len(atoms))

    # 设置Langevin恒温器
    # Langevin thermostat (NVT ensemble)
    dyn = Langevin(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature,
        friction=0.01  # 摩擦系数
    )

    # 或使用Velocity Verlet (NVE ensemble)
    # dyn = VelocityVerlet(atoms, timestep=timestep * units.fs)

    # 打开轨迹文件
    traj = Trajectory(trajectory_file, 'w', atoms)
    dyn.attach(traj.write, interval=interval)

    # 附加打印函数
    def print_status():
        """打印当前状态"""
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * len(atoms))

        print(f"Step {dyn.nsteps:4d}: "
              f"Epot = {epot:10.4f} eV, "
              f"Ekin = {ekin:10.4f} eV, "
              f"T = {temp:6.1f} K")

    dyn.attach(print_status, interval=interval)

    # 运行MD
    start_time = time.time()
    dyn.run(steps)
    end_time = time.time()

    print(f"\nAIMD完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"AIMD completed! Time: {end_time - start_time:.2f} s")
    print(f"轨迹保存在: {trajectory_file}")
    print(f"Trajectory saved in: {trajectory_file}")

    return traj


def extract_dataset(trajectory_file='aimd_trajectory.traj',
                   output_file='aimd_dataset.npz',
                   skip=1):
    """
    从轨迹提取训练数据
    Extract training data from trajectory

    Parameters:
    -----------
    trajectory_file : str
        轨迹文件
    output_file : str
        输出数据文件
    skip : int
        采样间隔（去相关）
    """
    print("\n" + "=" * 60)
    print("提取训练数据...")
    print("Extracting training data...")
    print("=" * 60)

    # 读取轨迹
    traj = Trajectory(trajectory_file, 'r')

    # 存储数据
    positions_list = []
    cells_list = []
    energies_list = []
    forces_list = []
    atomic_numbers_list = []

    for i, atoms in enumerate(traj[::skip]):
        print(f"处理构型 {i+1}/{len(traj[::skip])}")

        # 提取数据
        positions_list.append(atoms.get_positions())
        cells_list.append(atoms.get_cell())
        energies_list.append(atoms.get_potential_energy())
        forces_list.append(atoms.get_forces())
        atomic_numbers_list.append(atoms.get_atomic_numbers())

    # 转换为numpy数组
    positions = np.array(positions_list)
    cells = np.array(cells_list)
    energies = np.array(energies_list)
    forces = np.array(forces_list)
    atomic_numbers = atomic_numbers_list[0]  # 原子序数相同

    # 保存数据
    np.savez(
        output_file,
        positions=positions,
        cells=cells,
        energies=energies,
        forces=forces,
        atomic_numbers=atomic_numbers
    )

    print(f"\n数据集统计:")
    print(f"Dataset statistics:")
    print(f"  构型数量: {len(positions)}")
    print(f"  Configurations: {len(positions)}")
    print(f"  每个构型原子数: {len(atomic_numbers)}")
    print(f"  Atoms per config: {len(atomic_numbers)}")
    print(f"  能量范围: {energies.min():.4f} - {energies.max():.4f} eV")
    print(f"  Energy range: {energies.min():.4f} - {energies.max():.4f} eV")
    print(f"  力的最大值: {np.abs(forces).max():.4f} eV/Å")
    print(f"  Max force: {np.abs(forces).max():.4f} eV/Å")

    print(f"\n数据已保存到: {output_file}")
    print(f"Data saved to: {output_file}")

    return positions, cells, energies, forces, atomic_numbers


def visualize_trajectory(trajectory_file='aimd_trajectory.traj'):
    """
    可视化轨迹（能量、温度）
    Visualize trajectory (energy, temperature)

    Parameters:
    -----------
    trajectory_file : str
        轨迹文件
    """
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("可视化轨迹...")
    print("Visualizing trajectory...")
    print("=" * 60)

    # 读取轨迹
    traj = Trajectory(trajectory_file, 'r')

    # 提取数据
    times = []
    energies = []
    temperatures = []

    for i, atoms in enumerate(traj):
        times.append(i)
        energies.append(atoms.get_potential_energy())
        ekin = atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * len(atoms))
        temperatures.append(temp)

    times = np.array(times)
    energies = np.array(energies)
    temperatures = np.array(temperatures)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 能量
    ax1.plot(times, energies, 'b-', linewidth=1)
    ax1.set_xlabel('MD Step', fontsize=12)
    ax1.set_ylabel('Potential Energy (eV)', fontsize=12)
    ax1.set_title('Energy Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 温度
    ax2.plot(times, temperatures, 'r-', linewidth=1)
    ax2.set_xlabel('MD Step', fontsize=12)
    ax2.set_ylabel('Temperature (K)', fontsize=12)
    ax2.set_title('Temperature Evolution', fontsize=14, fontweight='bold')
    ax2.axhline(temperatures.mean(), color='k', linestyle='--',
                label=f'Mean: {temperatures.mean():.1f} K')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aimd_analysis.png', dpi=300)
    print("轨迹分析图已保存: aimd_analysis.png")
    print("Trajectory analysis saved: aimd_analysis.png")
    plt.show()


def main():
    """
    主函数
    Main function
    """
    print("\n" + "=" * 60)
    print("AIMD数据集构建")
    print("AIMD Dataset Construction")
    print("=" * 60)

    # 参数设置
    TEMPERATURE = 300  # K
    TIMESTEP = 1.0     # fs
    STEPS = 50         # 示例：使用较少步数
    SKIP = 5           # 采样间隔

    print("\n警告：完整的AIMD计算非常耗时！")
    print("Warning: Full AIMD calculation is very time-consuming!")
    print(f"\n当前设置（测试用）:")
    print(f"Current settings (for testing):")
    print(f"  温度 Temperature: {TEMPERATURE} K")
    print(f"  步数 Steps: {STEPS}")
    print(f"  时间步长 Timestep: {TIMESTEP} fs")
    print(f"  总时间 Total time: {STEPS * TIMESTEP / 1000:.2f} ps")

    user_input = input("\n是否继续? [y/N]: ")

    if user_input.lower() != 'y':
        print("已取消。")
        print("Cancelled.")
        return

    # 1. 创建结构
    print("\n步骤 1: 创建BaTiO₃超胞")
    print("Step 1: Create BaTiO₃ supercell")

    # 创建原胞
    # 这里使用简化的立方钙钛矿结构
    a = 4.0
    cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    positions = [
        [0.0, 0.0, 0.0],     # Ba
        [0.5, 0.5, 0.5],     # Ti
        [0.5, 0.5, 0.0],     # O
        [0.5, 0.0, 0.5],     # O
        [0.0, 0.5, 0.5],     # O
    ]
    atoms = Atoms('BaTiO3', scaled_positions=positions, cell=cell, pbc=True)

    # 创建2x2x2超胞
    supercell = create_supercell(atoms, size=(2, 2, 2))

    # 2. 设置计算器
    print("\n步骤 2: 设置DFT计算器")
    print("Step 2: Setup DFT calculator")
    calc = setup_calculator(supercell, mode='fast')

    # 3. 运行AIMD
    print("\n步骤 3: 运行AIMD")
    print("Step 3: Run AIMD")
    traj = run_aimd(
        supercell,
        temperature=TEMPERATURE,
        timestep=TIMESTEP,
        steps=STEPS,
        trajectory_file='aimd_trajectory.traj',
        interval=1
    )

    # 4. 可视化轨迹
    print("\n步骤 4: 可视化轨迹")
    print("Step 4: Visualize trajectory")
    visualize_trajectory('aimd_trajectory.traj')

    # 5. 提取数据集
    print("\n步骤 5: 提取训练数据集")
    print("Step 5: Extract training dataset")
    extract_dataset(
        trajectory_file='aimd_trajectory.traj',
        output_file='aimd_dataset.npz',
        skip=SKIP
    )

    print("\n" + "=" * 60)
    print("完成！")
    print("Completed!")
    print("=" * 60)
    print("\n生成的文件:")
    print("Generated files:")
    print("  - aimd_trajectory.traj  (完整轨迹)")
    print("  - aimd_dataset.npz      (训练数据)")
    print("  - aimd_analysis.png     (可视化)")


if __name__ == '__main__':
    main()
