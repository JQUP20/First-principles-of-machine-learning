#!/usr/bin/env python3
"""
ASE基本结构构建示例
演示如何使用ASE构建常见的晶体结构
"""

from ase import Atoms
from ase.build import bulk, molecule, surface
from ase.io import write
import os

def create_output_dir():
    """创建输出目录"""
    os.makedirs('structures', exist_ok=True)
    print("输出目录: ./structures/")

def build_crystals():
    """构建各种晶体结构"""
    print("\n" + "="*60)
    print("1. 构建晶体结构")
    print("="*60)

    # FCC结构 - Cu
    cu = bulk('Cu', 'fcc', a=3.61)
    write('structures/Cu_fcc.cif', cu)
    write('structures/Cu_fcc.xyz', cu)
    print(f"✓ Cu (FCC): {len(cu)} atoms, a = {cu.cell[0,0]:.3f} Å")

    # BCC结构 - Fe
    fe = bulk('Fe', 'bcc', a=2.87)
    write('structures/Fe_bcc.cif', fe)
    print(f"✓ Fe (BCC): {len(fe)} atoms, a = {fe.cell[0,0]:.3f} Å")

    # Diamond结构 - Si
    si = bulk('Si', 'diamond', a=5.43)
    write('structures/Si_diamond.cif', si)
    print(f"✓ Si (Diamond): {len(si)} atoms, a = {si.cell[0,0]:.3f} Å")

    # HCP结构 - Mg
    mg = bulk('Mg', 'hcp', a=3.21, c=5.21)
    write('structures/Mg_hcp.cif', mg)
    print(f"✓ Mg (HCP): {len(mg)} atoms, a = {mg.cell[0,0]:.3f} Å, c = {mg.cell[2,2]:.3f} Å")

    # Rocksalt结构 - NaCl
    nacl = bulk('NaCl', 'rocksalt', a=5.64)
    write('structures/NaCl_rocksalt.cif', nacl)
    print(f"✓ NaCl (Rocksalt): {len(nacl)} atoms, formula = {nacl.get_chemical_formula()}")

    # Perovskite结构 - SrTiO3
    sto = bulk('SrTiO3', 'perovskite', a=3.905)
    write('structures/SrTiO3_perovskite.cif', sto)
    print(f"✓ SrTiO3 (Perovskite): {len(sto)} atoms, formula = {sto.get_chemical_formula()}")

def build_supercells():
    """构建超胞"""
    print("\n" + "="*60)
    print("2. 构建超胞")
    print("="*60)

    # Si原胞
    si_unit = bulk('Si', 'diamond', a=5.43)
    print(f"Si原胞: {len(si_unit)} atoms")

    # 2x2x2超胞
    si_222 = si_unit * (2, 2, 2)
    write('structures/Si_2x2x2.cif', si_222)
    print(f"✓ Si 2×2×2超胞: {len(si_222)} atoms")

    # 3x3x3超胞
    si_333 = si_unit * (3, 3, 3)
    write('structures/Si_3x3x3.cif', si_333)
    print(f"✓ Si 3×3×3超胞: {len(si_333)} atoms")

    # 非对称超胞
    si_234 = si_unit * (2, 3, 4)
    write('structures/Si_2x3x4.cif', si_234)
    print(f"✓ Si 2×3×4超胞: {len(si_234)} atoms")

def build_molecules():
    """构建分子"""
    print("\n" + "="*60)
    print("3. 构建分子")
    print("="*60)

    molecules_to_build = [
        'H2O', 'CO2', 'CH4', 'NH3', 'C6H6',
        'CO', 'O2', 'N2', 'H2'
    ]

    for mol_name in molecules_to_build:
        mol = molecule(mol_name)
        # 放入盒子中
        mol.set_cell([10, 10, 10])
        mol.center()

        write(f'structures/{mol_name}_molecule.xyz', mol)
        formula = mol.get_chemical_formula()
        print(f"✓ {mol_name}: {len(mol)} atoms, formula = {formula}")

def build_surfaces():
    """构建表面"""
    print("\n" + "="*60)
    print("4. 构建表面")
    print("="*60)

    # Au(111)表面
    au111 = surface('Au', (1, 1, 1), layers=4)
    au111.center(vacuum=10.0, axis=2)
    write('structures/Au_111_surface.cif', au111)
    print(f"✓ Au(111)表面: {len(au111)} atoms, 真空层 = 10 Å")

    # Cu(100)表面
    cu100 = surface('Cu', (1, 0, 0), layers=4)
    cu100.center(vacuum=10.0, axis=2)
    write('structures/Cu_100_surface.cif', cu100)
    print(f"✓ Cu(100)表面: {len(cu100)} atoms")

    # Pt(111)表面
    pt111 = surface('Pt', (1, 1, 1), layers=4)
    pt111.center(vacuum=10.0, axis=2)
    write('structures/Pt_111_surface.cif', pt111)
    print(f"✓ Pt(111)表面: {len(pt111)} atoms")

def analyze_structures():
    """分析结构性质"""
    print("\n" + "="*60)
    print("5. 结构分析")
    print("="*60)

    # 创建Si晶体
    si = bulk('Si', 'diamond', a=5.43)

    print(f"化学式: {si.get_chemical_formula()}")
    print(f"原子数: {len(si)}")
    print(f"原子序数: {si.get_atomic_numbers()}")
    print(f"质量: {si.get_masses()} amu")
    print(f"体积: {si.get_volume():.3f} Å³")
    print(f"密度: {si.get_volume()/len(si):.3f} Å³/atom")
    print(f"周期性: {si.get_pbc()}")
    print(f"\n晶胞矩阵:\n{si.get_cell()}")
    print(f"\n原子位置:\n{si.get_positions()}")
    print(f"\n缩放坐标:\n{si.get_scaled_positions()}")

def create_custom_structure():
    """手动创建自定义结构"""
    print("\n" + "="*60)
    print("6. 创建自定义结构")
    print("="*60)

    # 创建简单立方Na
    cell = [[3.0, 0, 0],
            [0, 3.0, 0],
            [0, 0, 3.0]]

    atoms = Atoms('Na',
                  positions=[[0, 0, 0]],
                  cell=cell,
                  pbc=True)

    write('structures/Na_sc_custom.cif', atoms)
    print(f"✓ 自定义Na简单立方: {len(atoms)} atoms")

    # 创建FCC结构（手动）
    cell = [[4.0, 0, 0],
            [0, 4.0, 0],
            [0, 0, 4.0]]

    # FCC位置（分数坐标）
    scaled_positions = [[0.0, 0.0, 0.0],
                       [0.5, 0.5, 0.0],
                       [0.5, 0.0, 0.5],
                       [0.0, 0.5, 0.5]]

    fcc_custom = Atoms('Cu4',
                       scaled_positions=scaled_positions,
                       cell=cell,
                       pbc=True)

    write('structures/Cu_fcc_custom.cif', fcc_custom)
    print(f"✓ 自定义Cu FCC结构: {len(fcc_custom)} atoms")

def main():
    """主函数"""
    print("\n" + "#"*60)
    print("  ASE基本结构构建示例")
    print("#"*60)

    # 创建输出目录
    create_output_dir()

    # 构建各种结构
    build_crystals()
    build_supercells()
    build_molecules()
    build_surfaces()

    # 分析结构
    analyze_structures()

    # 创建自定义结构
    create_custom_structure()

    print("\n" + "#"*60)
    print("  所有结构已创建！")
    print("  查看 ./structures/ 目录")
    print("#"*60 + "\n")

if __name__ == '__main__':
    main()
