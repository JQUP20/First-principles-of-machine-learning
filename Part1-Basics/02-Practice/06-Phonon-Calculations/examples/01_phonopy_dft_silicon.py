#!/usr/bin/env python3
"""
使用Phonopy和GPAW计算硅晶体的声子谱
Calculate phonon spectrum of silicon crystal using Phonopy and GPAW

作者 | Author: First-Principles ML Course
日期 | Date: 2025-11-14
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.calculators.gpaw import GPAW, PW
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import os

# ============================================================================
# 第一步：设置晶体结构
# Step 1: Set up crystal structure
# ============================================================================

print("=" * 70)
print("硅晶体声子谱计算 | Silicon Phonon Spectrum Calculation")
print("方法 | Method: Phonopy + GPAW (DFT)")
print("=" * 70)
print()

# 构建硅晶体原胞（金刚石结构）
# Build silicon primitive cell (diamond structure)
print("构建硅晶体结构...")
print("Building silicon crystal structure...")

# 使用ASE构建
atoms = bulk('Si', 'diamond', a=5.43)  # 晶格常数 a = 5.43 Å

print(f"  原胞原子数: {len(atoms)}")
print(f"  Number of atoms in primitive cell: {len(atoms)}")
print(f"  晶格常数: a = {atoms.cell.cellpar()[0]:.3f} Å")
print(f"  Lattice parameter: a = {atoms.cell.cellpar()[0]:.3f} Å")
print()

# 转换为Phonopy格式
# Convert to Phonopy format
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

# 定义超胞大小（2×2×2）
# Define supercell size (2×2×2)
supercell_matrix = [[2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2]]

# 创建Phonopy对象
# Create Phonopy object
phonon = Phonopy(phonopy_atoms, supercell_matrix)

print(f"  超胞矩阵: {supercell_matrix}")
print(f"  Supercell matrix: {supercell_matrix}")
print(f"  超胞原子数: {len(phonon.supercell)}")
print(f"  Number of atoms in supercell: {len(phonon.supercell)}")
print()

# 生成位移构型（利用对称性）
# Generate displaced configurations (using symmetry)
print("生成位移构型...")
print("Generating displaced configurations...")

phonon.generate_displacements(distance=0.01)  # 位移大小 0.01 Å

displaced_supercells = phonon.supercells_with_displacements

print(f"  位移构型数量: {len(displaced_supercells)}")
print(f"  Number of displacements: {len(displaced_supercells)}")
print()

# ============================================================================
# 第三步：使用GPAW计算每个位移构型的受力
# Step 3: Calculate forces for each displaced configuration using GPAW
# ============================================================================

print("使用GPAW计算受力...")
print("Calculating forces using GPAW (DFT)...")
print("警告: 这可能需要较长时间！")
print("Warning: This may take a while!")
print()

# GPAW计算器设置
# GPAW calculator setup
calc_params = {
    'mode': PW(300),           # 平面波截断能 300 eV
    'kpts': (4, 4, 4),        # k点网格
    'xc': 'PBE',              # 交换关联泛函
    'txt': 'gpaw_output.txt', # 输出文件
    'symmetry': 'off'         # 关闭对称性（Phonopy需要）
}

print("GPAW计算参数:")
print("GPAW calculation parameters:")
print(f"  截断能 | Cutoff: {calc_params['mode'].ecut} eV")
print(f"  k点 | k-points: {calc_params['kpts']}")
print(f"  泛函 | XC functional: {calc_params['xc']}")
print()

# 计算每个位移构型的力
# Calculate forces for each displacement
forces_list = []

for i, disp_supercell in enumerate(displaced_supercells):
    print(f"计算位移构型 {i+1}/{len(displaced_supercells)}...")
    print(f"Calculating displacement {i+1}/{len(displaced_supercells)}...")

    # 转换回ASE格式
    # Convert back to ASE format
    from ase import Atoms
    disp_atoms = Atoms(
        symbols=disp_supercell.symbols,
        cell=disp_supercell.cell,
        scaled_positions=disp_supercell.scaled_positions,
        pbc=True
    )

    # 设置计算器
    # Set calculator
    calc = GPAW(**calc_params)
    disp_atoms.calc = calc

    # 计算力
    # Calculate forces
    forces = disp_atoms.get_forces()
    forces_list.append(forces)

    print(f"  最大力: {np.max(np.abs(forces)):.6f} eV/Å")
    print(f"  Max force: {np.max(np.abs(forces)):.6f} eV/Å")
    print()

# 设置计算的力
# Set calculated forces
phonon.forces = np.array(forces_list)

# ============================================================================
# 第四步：计算力常数矩阵
# Step 4: Calculate force constants
# ============================================================================

print("计算力常数矩阵...")
print("Calculating force constants...")

phonon.produce_force_constants()

# 保存力常数
# Save force constants
phonon.save('force_constants.hdf5')
print("  力常数已保存到: force_constants.hdf5")
print("  Force constants saved to: force_constants.hdf5")
print()

# ============================================================================
# 第五步：计算声子谱
# Step 5: Calculate phonon band structure
# ============================================================================

print("计算声子带结构...")
print("Calculating phonon band structure...")

# 定义高对称路径（金刚石结构）
# Define high-symmetry path (diamond structure)
# Γ-X-K-Γ-L
path = [
    [0.0, 0.0, 0.0],  # Γ
    [0.5, 0.0, 0.5],  # X
    [0.5, 0.25, 0.75],  # K (or W)
    [0.0, 0.0, 0.0],  # Γ
    [0.5, 0.5, 0.5],  # L
]

labels = ['$\\Gamma$', 'X', 'K', '$\\Gamma$', 'L']

# 设置路径
# Set path
phonon.run_band_structure(
    path,
    npoints=101,
    with_eigenvectors=True,
    labels=labels
)

# 获取带结构数据
# Get band structure data
band_dict = phonon.get_band_structure_dict()
distances = band_dict['distances']
frequencies = band_dict['frequencies']
eigenvectors = band_dict['eigenvectors']

print(f"  计算了 {frequencies.shape[1]} 个声子支")
print(f"  Calculated {frequencies.shape[1]} phonon branches")
print(f"  q点数量: {frequencies.shape[0]}")
print(f"  Number of q-points: {frequencies.shape[0]}")
print()

# ============================================================================
# 第六步：计算声子态密度
# Step 6: Calculate phonon density of states
# ============================================================================

print("计算声子态密度...")
print("Calculating phonon density of states...")

# 设置网格
# Set mesh
phonon.run_mesh([20, 20, 20])
phonon.run_total_dos()

# 获取DOS数据
# Get DOS data
dos_dict = phonon.get_total_dos_dict()
dos_frequencies = dos_dict['frequency_points']
dos = dos_dict['total_dos']

print("  声子态密度计算完成")
print("  Phonon DOS calculation completed")
print()

# ============================================================================
# 第七步：可视化结果
# Step 7: Visualize results
# ============================================================================

print("绘制声子谱...")
print("Plotting phonon spectrum...")

# 创建图形
# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ========== 声子带结构 | Phonon Band Structure ==========
ax1.set_xlabel('Wave Vector')
ax1.set_ylabel('Frequency (THz)')
ax1.set_title('Phonon Band Structure of Silicon')

# 绘制所有支
# Plot all branches
for i in range(frequencies.shape[1]):
    ax1.plot(distances, frequencies[:, i], 'b-', linewidth=1)

# 添加高对称点标签
# Add high-symmetry point labels
special_points = [0]
for i in range(len(path) - 1):
    special_points.append(
        special_points[-1] + np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
    )

ax1.set_xticks(special_points)
ax1.set_xticklabels(labels)

# 添加竖线
# Add vertical lines
for x in special_points:
    ax1.axvline(x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# 添加零线
# Add zero line
ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)

ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# ========== 声子态密度 | Phonon DOS ==========
ax2.set_xlabel('Density of States (states/THz)')
ax2.set_ylabel('Frequency (THz)')
ax2.set_title('Phonon Density of States')

ax2.plot(dos, dos_frequencies, 'r-', linewidth=2)
ax2.fill_betweenx(dos_frequencies, 0, dos, alpha=0.3, color='red')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

plt.tight_layout()

# 保存图像
# Save figure
os.makedirs('../results', exist_ok=True)
output_file = '../results/silicon_phonon_dft.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  声子谱已保存到: {output_file}")
print(f"  Phonon spectrum saved to: {output_file}")

plt.show()

# ============================================================================
# 第八步：输出热力学性质
# Step 8: Output thermodynamic properties
# ============================================================================

print()
print("计算热力学性质...")
print("Calculating thermodynamic properties...")

# 在室温（300K）计算性质
# Calculate properties at room temperature (300K)
from phonopy import PhonopyQHA

# 自由能（简化计算）
# Free energy (simplified calculation)
temperatures = np.linspace(0, 1000, 101)
phonon.run_mesh([20, 20, 20], with_eigenvectors=False)
phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)

# 获取热性质
# Get thermal properties
tp_dict = phonon.get_thermal_properties_dict()
temps = tp_dict['temperatures']
free_energy = tp_dict['free_energy']
entropy = tp_dict['entropy']
heat_capacity = tp_dict['heat_capacity']

print()
print("室温(300K)性质:")
print("Properties at 300K:")
idx_300 = np.argmin(np.abs(temps - 300))
print(f"  自由能: {free_energy[idx_300]:.3f} kJ/mol")
print(f"  Free energy: {free_energy[idx_300]:.3f} kJ/mol")
print(f"  熵: {entropy[idx_300]:.3f} J/(mol·K)")
print(f"  Entropy: {entropy[idx_300]:.3f} J/(mol·K)")
print(f"  比热容: {heat_capacity[idx_300]:.3f} J/(mol·K)")
print(f"  Heat capacity: {heat_capacity[idx_300]:.3f} J/(mol·K)")
print()

# 保存热性质数据
# Save thermal properties
thermal_output = '../results/silicon_thermal_properties_dft.txt'
with open(thermal_output, 'w') as f:
    f.write("# Temperature (K), Free Energy (kJ/mol), Entropy (J/mol·K), Heat Capacity (J/mol·K)\n")
    for T, F, S, Cv in zip(temps, free_energy, entropy, heat_capacity):
        f.write(f"{T:.1f} {F:.6f} {S:.6f} {Cv:.6f}\n")

print(f"热性质数据已保存到: {thermal_output}")
print(f"Thermal properties saved to: {thermal_output}")

# ============================================================================
# 总结
# Summary
# ============================================================================

print()
print("=" * 70)
print("计算完成! | Calculation Complete!")
print("=" * 70)
print()
print("输出文件:")
print("Output files:")
print(f"  1. 声子谱图: {output_file}")
print(f"     Phonon spectrum: {output_file}")
print(f"  2. 力常数: force_constants.hdf5")
print(f"     Force constants: force_constants.hdf5")
print(f"  3. 热性质: {thermal_output}")
print(f"     Thermal properties: {thermal_output}")
print()
print("下一步:")
print("Next steps:")
print("  - 使用ML势函数重复计算以对比性能")
print("    Repeat calculation with ML potential to compare performance")
print("  - 分析声子模式的对称性")
print("    Analyze symmetry of phonon modes")
print("  - 计算热导率（需要phono3py）")
print("    Calculate thermal conductivity (requires phono3py)")
print()
