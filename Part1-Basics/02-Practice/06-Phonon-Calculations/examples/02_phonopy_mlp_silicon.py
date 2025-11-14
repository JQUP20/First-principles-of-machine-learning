#!/usr/bin/env python3
"""
使用Phonopy和机器学习势函数(Allegro/NequIP)计算硅晶体的声子谱
Calculate phonon spectrum of silicon using Phonopy and ML potential

相比DFT方法，ML势函数可以:
- 速度提升100-1000倍
- 使用更大的超胞
- 保持接近DFT的精度

作者 | Author: First-Principles ML Course
日期 | Date: 2025-11-14
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import os
import time

# ============================================================================
# 配置
# Configuration
# ============================================================================

# 设置是否使用预训练的ML势函数
USE_PRETRAINED_MODEL = False  # 设置为True如果有预训练模型

# 预训练模型路径（如果有）
PRETRAINED_MODEL_PATH = "silicon_allegro.pth"

print("=" * 70)
print("硅晶体声子谱计算 | Silicon Phonon Spectrum Calculation")
print("方法 | Method: Phonopy + ML Potential (Allegro/NequIP)")
print("=" * 70)
print()

if not USE_PRETRAINED_MODEL:
    print("注意: 此示例需要预训练的ML势函数模型")
    print("Note: This example requires a pre-trained ML potential model")
    print()
    print("获取模型的方法:")
    print("Methods to obtain a model:")
    print("  1. 训练自己的模型（参考Part2-Allegro-Reproduction）")
    print("     Train your own model (see Part2-Allegro-Reproduction)")
    print("  2. 下载预训练模型")
    print("     Download pre-trained model")
    print("  3. 使用模拟数据进行演示（将使用简化方法）")
    print("     Use simulated data for demonstration (will use simplified method)")
    print()
    use_simulation = input("是否使用模拟模式继续？(y/n) | Continue with simulation mode? (y/n): ")

    if use_simulation.lower() != 'y':
        print("退出程序。请准备好ML势函数模型后重试。")
        print("Exiting. Please prepare ML potential model and try again.")
        exit(0)

    print()
    print("使用模拟模式...（演示流程，不使用真实ML势函数）")
    print("Using simulation mode... (demonstrating workflow without real ML potential)")
    print()

# ============================================================================
# 第一步：设置晶体结构
# Step 1: Set up crystal structure
# ============================================================================

print("构建硅晶体结构...")
print("Building silicon crystal structure...")

# 构建硅晶体原胞
atoms = bulk('Si', 'diamond', a=5.43)

print(f"  原胞原子数: {len(atoms)}")
print(f"  Number of atoms in primitive cell: {len(atoms)}")
print()

# 转换为Phonopy格式
def ase_to_phonopy(ase_atoms):
    """将ASE Atoms对象转换为PhonopyAtoms"""
    return PhonopyAtoms(
        symbols=ase_atoms.get_chemical_symbols(),
        cell=ase_atoms.cell,
        scaled_positions=ase_atoms.get_scaled_positions()
    )

phonopy_atoms = ase_to_phonopy(atoms)

# ============================================================================
# 第二步：创建超胞和位移构型
# Step 2: Create supercell and displaced configurations
# ============================================================================

print("创建超胞...")
print("Creating supercell...")

# 使用更大的超胞（ML势函数的优势！）
# Use larger supercell (advantage of ML potential!)
supercell_matrix = [[3, 0, 0],
                    [0, 3, 0],
                    [0, 0, 3]]  # 3×3×3 (vs 2×2×2 for DFT)

# 创建Phonopy对象
phonon = Phonopy(phonopy_atoms, supercell_matrix)

print(f"  超胞矩阵: {supercell_matrix}")
print(f"  Supercell matrix: {supercell_matrix}")
print(f"  超胞原子数: {len(phonon.supercell)}")
print(f"  Number of atoms in supercell: {len(phonon.supercell)}")
print("  (使用3×3×3超胞，比DFT方法的2×2×2更大！)")
print("  (Using 3×3×3 supercell, larger than 2×2×2 in DFT method!)")
print()

# 生成位移构型
print("生成位移构型...")
print("Generating displaced configurations...")

phonon.generate_displacements(distance=0.01)
displaced_supercells = phonon.supercells_with_displacements

print(f"  位移构型数量: {len(displaced_supercells)}")
print(f"  Number of displacements: {len(displaced_supercells)}")
print()

# ============================================================================
# 第三步：使用ML势函数计算受力
# Step 3: Calculate forces using ML potential
# ============================================================================

print("使用ML势函数计算受力...")
print("Calculating forces using ML potential...")
print()

start_time = time.time()

if USE_PRETRAINED_MODEL:
    # ========== 使用真实ML势函数 ==========
    # Use real ML potential
    try:
        from nequip.ase import NequIPCalculator

        print("加载ML势函数模型...")
        print("Loading ML potential model...")

        # 加载已部署的模型
        calc = NequIPCalculator.from_deployed_model(
            PRETRAINED_MODEL_PATH,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        print(f"  模型已加载: {PRETRAINED_MODEL_PATH}")
        print(f"  Model loaded: {PRETRAINED_MODEL_PATH}")
        print()

        # 计算力
        forces_list = []
        for i, disp_supercell in enumerate(displaced_supercells):
            print(f"计算位移构型 {i+1}/{len(displaced_supercells)}...")

            # 转换为ASE格式
            from ase import Atoms
            disp_atoms = Atoms(
                symbols=disp_supercell.symbols,
                cell=disp_supercell.cell,
                scaled_positions=disp_supercell.scaled_positions,
                pbc=True
            )

            # 使用ML势函数计算
            disp_atoms.calc = calc
            forces = disp_atoms.get_forces()
            forces_list.append(forces)

            print(f"  最大力: {np.max(np.abs(forces)):.6f} eV/Å")

    except ImportError:
        print("错误: 未安装NequIP/Allegro")
        print("Error: NequIP/Allegro not installed")
        print("请运行: pip install nequip-allegro")
        print("Please run: pip install nequip-allegro")
        exit(1)

else:
    # ========== 模拟模式：使用简化的力场 ==========
    # Simulation mode: use simplified force field

    print("模拟模式: 使用Lennard-Jones势函数作为演示")
    print("Simulation mode: Using Lennard-Jones potential for demonstration")
    print()

    from ase.calculators.lj import LennardJones

    # 使用Lennard-Jones势函数（这不是ML势函数，仅用于演示流程）
    lj_calc = LennardJones(sigma=2.0, epsilon=0.01)

    forces_list = []
    for i, disp_supercell in enumerate(displaced_supercells):
        print(f"计算位移构型 {i+1}/{len(displaced_supercells)}...")

        from ase import Atoms
        disp_atoms = Atoms(
            symbols=disp_supercell.symbols,
            cell=disp_supercell.cell,
            scaled_positions=disp_supercell.scaled_positions,
            pbc=True
        )

        disp_atoms.calc = lj_calc
        forces = disp_atoms.get_forces()
        forces_list.append(forces)

        if i == 0:
            print(f"  最大力: {np.max(np.abs(forces)):.6f} eV/Å")

calc_time = time.time() - start_time

print()
print(f"力计算完成！耗时: {calc_time:.2f} 秒")
print(f"Force calculation completed! Time: {calc_time:.2f} seconds")
print()

if USE_PRETRAINED_MODEL:
    print("性能对比:")
    print("Performance comparison:")
    print(f"  ML势函数: ~{calc_time:.1f} 秒")
    print(f"  ML potential: ~{calc_time:.1f} seconds")
    print(f"  DFT (估计): ~2-4 小时")
    print(f"  DFT (estimated): ~2-4 hours")
    print(f"  加速比: ~{7200/calc_time:.0f}×")
    print(f"  Speedup: ~{7200/calc_time:.0f}×")
    print()

# 设置力
phonon.forces = np.array(forces_list)

# ============================================================================
# 第四步：计算力常数
# Step 4: Calculate force constants
# ============================================================================

print("计算力常数矩阵...")
print("Calculating force constants...")

phonon.produce_force_constants()
phonon.save('force_constants_mlp.hdf5')

print("  力常数已保存")
print("  Force constants saved")
print()

# ============================================================================
# 第五步：计算声子谱
# Step 5: Calculate phonon spectrum
# ============================================================================

print("计算声子带结构...")
print("Calculating phonon band structure...")

# 高对称路径
path = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5],
    [0.5, 0.25, 0.75],
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
]
labels = ['$\\Gamma$', 'X', 'K', '$\\Gamma$', 'L']

phonon.run_band_structure(path, npoints=101, labels=labels)
band_dict = phonon.get_band_structure_dict()

# 声子DOS
print("计算声子态密度...")
print("Calculating phonon DOS...")
phonon.run_mesh([30, 30, 30])  # 更密的网格（ML势函数可以承受）
phonon.run_total_dos()
dos_dict = phonon.get_total_dos_dict()

print("  声子计算完成")
print("  Phonon calculation completed")
print()

# ============================================================================
# 第六步：可视化
# Step 6: Visualization
# ============================================================================

print("绘制声子谱...")
print("Plotting phonon spectrum...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 带结构
distances = band_dict['distances']
frequencies = band_dict['frequencies']

ax1.set_xlabel('Wave Vector')
ax1.set_ylabel('Frequency (THz)')
ax1.set_title('Phonon Band Structure (ML Potential)')

for i in range(frequencies.shape[1]):
    ax1.plot(distances, frequencies[:, i], 'b-', linewidth=1)

special_points = [0]
for i in range(len(path) - 1):
    special_points.append(
        special_points[-1] + np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
    )

ax1.set_xticks(special_points)
ax1.set_xticklabels(labels)

for x in special_points:
    ax1.axvline(x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# DOS
dos_frequencies = dos_dict['frequency_points']
dos = dos_dict['total_dos']

ax2.set_xlabel('Density of States')
ax2.set_ylabel('Frequency (THz)')
ax2.set_title('Phonon DOS (ML Potential)')

ax2.plot(dos, dos_frequencies, 'r-', linewidth=2)
ax2.fill_betweenx(dos_frequencies, 0, dos, alpha=0.3, color='red')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

plt.tight_layout()

os.makedirs('../results', exist_ok=True)
output_file = '../results/silicon_phonon_mlp.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')

print(f"  声子谱已保存: {output_file}")
print(f"  Phonon spectrum saved: {output_file}")

plt.show()

# ============================================================================
# 第七步：热性质
# Step 7: Thermal properties
# ============================================================================

print()
print("计算热性质...")
print("Calculating thermal properties...")

phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)
tp_dict = phonon.get_thermal_properties_dict()

temps = tp_dict['temperatures']
free_energy = tp_dict['free_energy']
entropy = tp_dict['entropy']
heat_capacity = tp_dict['heat_capacity']

idx_300 = np.argmin(np.abs(temps - 300))
print()
print("室温(300K)性质:")
print("Properties at 300K:")
print(f"  自由能: {free_energy[idx_300]:.3f} kJ/mol")
print(f"  Free energy: {free_energy[idx_300]:.3f} kJ/mol")
print(f"  熵: {entropy[idx_300]:.3f} J/(mol·K)")
print(f"  Entropy: {entropy[idx_300]:.3f} J/(mol·K)")
print(f"  比热容: {heat_capacity[idx_300]:.3f} J/(mol·K)")
print(f"  Heat capacity: {heat_capacity[idx_300]:.3f} J/(mol·K)")
print()

# ============================================================================
# 总结
# Summary
# ============================================================================

print("=" * 70)
print("计算完成! | Calculation Complete!")
print("=" * 70)
print()
print("ML势函数的优势:")
print("Advantages of ML potential:")
print("  ✓ 计算速度快（100-1000倍于DFT）")
print("    Fast calculation (100-1000× faster than DFT)")
print("  ✓ 可以使用更大的超胞（3×3×3 vs 2×2×2）")
print("    Can use larger supercell (3×3×3 vs 2×2×2)")
print("  ✓ 接近DFT的精度")
print("    Near DFT accuracy")
print()
print("注意事项:")
print("Considerations:")
print("  - 需要高质量的训练数据")
print("    Requires high-quality training data")
print("  - 训练数据应包含声子位移构型")
print("    Training data should include phonon displacements")
print("  - 外推能力有限")
print("    Limited extrapolation capability")
print()
print("下一步:")
print("Next steps:")
print("  - 与DFT结果对比（运行03_compare_phonon_results.py）")
print("    Compare with DFT results (run 03_compare_phonon_results.py)")
print("  - 分析误差来源")
print("    Analyze error sources")
print("  - 优化ML势函数训练")
print("    Optimize ML potential training")
print()
