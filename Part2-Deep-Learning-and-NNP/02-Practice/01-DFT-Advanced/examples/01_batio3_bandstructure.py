#!/usr/bin/env python3
"""
BaTiO3能带结构计算示例
Band Structure Calculation for BaTiO3

本脚本演示如何使用GPAW计算钛酸钡(BaTiO3)的能带结构。
This script demonstrates how to calculate the band structure of BaTiO3 using GPAW.
"""

import numpy as np
from ase import Atoms
from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
from gpaw.band_structure import calculate_band_structure
import matplotlib.pyplot as plt


def create_batio3_structure():
    """
    创建BaTiO3钙钛矿结构
    Create BaTiO3 perovskite structure
    """
    # 晶格常数 (单位: Angstrom)
    a = 4.0  # 立方相近似值

    # 创建晶胞
    # Ba在(0,0,0), Ti在(0.5,0.5,0.5), O在(0.5,0.5,0)等位置
    cell = [[a, 0, 0],
            [0, a, 0],
            [0, 0, a]]

    positions = [
        [0.0, 0.0, 0.0],     # Ba
        [0.5, 0.5, 0.5],     # Ti
        [0.5, 0.5, 0.0],     # O1
        [0.5, 0.0, 0.5],     # O2
        [0.0, 0.5, 0.5],     # O3
    ]

    atoms = Atoms('BaTiO3',
                  scaled_positions=positions,
                  cell=cell,
                  pbc=True)

    return atoms


def run_scf_calculation(atoms, gpw_file='batio3_scf.gpw'):
    """
    运行自洽场(SCF)计算
    Run self-consistent field calculation

    Parameters:
    -----------
    atoms : ASE Atoms object
        原子结构
    gpw_file : str
        保存结果的文件名
    """
    print("=" * 60)
    print("开始自洽场(SCF)计算...")
    print("Starting SCF calculation...")
    print("=" * 60)

    # 设置计算参数
    calc = GPAW(
        mode=PW(500),  # 平面波截断能 500 eV
        xc='PBE',      # 交换关联泛函
        kpts={'size': (6, 6, 6), 'gamma': True},  # k点网格
        occupations=FermiDirac(0.1),  # Fermi-Dirac展宽
        txt='batio3_scf.txt',  # 输出日志
        symmetry='off'  # 关闭对称性（简化示例）
    )

    atoms.calc = calc

    # 计算总能量
    energy = atoms.get_potential_energy()
    print(f"\n总能量: {energy:.4f} eV")
    print(f"Total energy: {energy:.4f} eV")

    # 保存结果
    calc.write(gpw_file)
    print(f"\n结果已保存到: {gpw_file}")
    print(f"Results saved to: {gpw_file}")

    return calc


def calculate_bands(gpw_file='batio3_scf.gpw',
                   bands_file='batio3_bands.json'):
    """
    计算能带结构
    Calculate band structure

    Parameters:
    -----------
    gpw_file : str
        SCF结果文件
    bands_file : str
        能带数据保存文件
    """
    print("\n" + "=" * 60)
    print("计算能带结构...")
    print("Calculating band structure...")
    print("=" * 60)

    # 读取SCF结果
    calc = GPAW(gpw_file, txt=None)
    atoms = calc.get_atoms()

    # 定义高对称路径 (简单立方近似)
    # Γ-X-M-Γ-R-X
    path_points = [
        'G',   # Γ点 (0, 0, 0)
        'X',   # X点 (0.5, 0, 0)
        'M',   # M点 (0.5, 0.5, 0)
        'G',   # Γ点
        'R',   # R点 (0.5, 0.5, 0.5)
        'X'    # X点
    ]

    # 计算能带
    bs = calculate_band_structure(atoms, path_points, npoints=100)

    # 保存能带数据
    bs.write(bands_file)
    print(f"能带数据已保存到: {bands_file}")
    print(f"Band structure data saved to: {bands_file}")

    return bs


def plot_band_structure(bands_file='batio3_bands.json',
                        output_file='batio3_bandstructure.png'):
    """
    绘制能带图
    Plot band structure

    Parameters:
    -----------
    bands_file : str
        能带数据文件
    output_file : str
        输出图片文件
    """
    print("\n" + "=" * 60)
    print("绘制能带图...")
    print("Plotting band structure...")
    print("=" * 60)

    from ase.io import read
    from ase.spectrum.band_structure import BandStructure

    # 读取能带数据
    bs = BandStructure(bands_file)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制能带
    bs.plot(ax=ax, emin=-10, emax=10)

    # 设置标题和标签
    ax.set_title('BaTiO₃ Band Structure', fontsize=16, fontweight='bold')
    ax.set_ylabel('Energy (eV)', fontsize=14)
    ax.set_xlabel('k-path', fontsize=14)

    # 在费米能级处画水平线
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(0.02, 0.5, 'Fermi Level', transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom')

    # 设置y轴范围
    ax.set_ylim(-8, 8)

    # 网格
    ax.grid(True, alpha=0.3)

    # 保存图片
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n能带图已保存到: {output_file}")
    print(f"Band structure plot saved to: {output_file}")

    plt.show()


def analyze_band_gap(bands_file='batio3_bands.json'):
    """
    分析带隙
    Analyze band gap

    Parameters:
    -----------
    bands_file : str
        能带数据文件
    """
    print("\n" + "=" * 60)
    print("分析带隙...")
    print("Analyzing band gap...")
    print("=" * 60)

    from ase.spectrum.band_structure import BandStructure

    bs = BandStructure(bands_file)

    # 获取带隙信息
    # 注意：这里需要根据实际情况调整
    print("\n提示：带隙分析需要手动检查能带图")
    print("Tip: Band gap analysis requires manual inspection of the band diagram")
    print("\n查看能带图，找出：")
    print("Check the band diagram to find:")
    print("1. 价带顶(VBM)的位置和能量")
    print("   Valence band maximum (VBM) position and energy")
    print("2. 导带底(CBM)的位置和能量")
    print("   Conduction band minimum (CBM) position and energy")
    print("3. 带隙类型（直接/间接）")
    print("   Band gap type (direct/indirect)")


def main():
    """
    主函数
    Main function
    """
    print("\n" + "=" * 60)
    print("BaTiO₃能带结构计算")
    print("BaTiO₃ Band Structure Calculation")
    print("=" * 60)

    # 1. 创建结构
    print("\n步骤 1: 创建BaTiO₃结构")
    print("Step 1: Create BaTiO₃ structure")
    atoms = create_batio3_structure()
    print(f"晶胞参数: {atoms.cell.cellpar()}")
    print(f"Cell parameters: {atoms.cell.cellpar()}")
    print(f"原子数: {len(atoms)}")
    print(f"Number of atoms: {len(atoms)}")

    # 2. 运行SCF计算
    # 注意：这步计算可能需要较长时间！
    # Note: This step may take a long time!
    print("\n步骤 2: 自洽场计算")
    print("Step 2: SCF calculation")
    user_input = input("是否运行SCF计算？(需要较长时间) [y/N]: ")

    if user_input.lower() == 'y':
        calc = run_scf_calculation(atoms)

        # 3. 计算能带
        print("\n步骤 3: 计算能带结构")
        print("Step 3: Calculate band structure")
        bs = calculate_bands()

        # 4. 绘制能带图
        print("\n步骤 4: 绘制能带图")
        print("Step 4: Plot band structure")
        plot_band_structure()

        # 5. 分析带隙
        print("\n步骤 5: 分析带隙")
        print("Step 5: Analyze band gap")
        analyze_band_gap()
    else:
        print("\n跳过计算。请先运行SCF计算。")
        print("Calculation skipped. Please run SCF calculation first.")
        print("\n如果已有计算结果，可以只绘图：")
        print("If you have calculation results, you can plot directly:")
        print("  from gpaw.band_structure import BandStructure")
        print("  bs = BandStructure('batio3_bands.json')")
        print("  bs.plot()")

    print("\n" + "=" * 60)
    print("计算完成！")
    print("Calculation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
