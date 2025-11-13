#!/usr/bin/env python3
"""
GPAW简单计算示例
演示如何使用GPAW进行基本的第一性原理计算

注意：这个脚本需要安装GPAW才能运行
如果没有安装GPAW，请参考教程中的安装说明
"""

def example_1_h2_molecule():
    """
    示例1: 计算H2分子的能量
    这是最简单的GPAW计算示例
    """
    print("\n" + "="*60)
    print("示例1: H2分子能量计算")
    print("="*60)

    from ase import Atoms
    from ase.io import write

    # 创建H2分子
    d = 0.74  # H-H键长 (Å)
    h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, d]])

    # 放入盒子中（分子需要足够的真空层）
    h2.set_cell([6, 6, 6])
    h2.center()

    # 保存结构
    write('h2_molecule.xyz', h2)
    print(f"✓ H2分子已创建: 键长 = {d} Å")

    try:
        from gpaw import GPAW, PW

        # 设置计算器
        calc = GPAW(mode=PW(400),  # 平面波，截断能400 eV
                    xc='PBE',       # PBE泛函
                    txt='h2_calculation.txt')

        h2.calc = calc

        # 计算能量
        energy = h2.get_potential_energy()
        forces = h2.get_forces()

        print(f"\n结果:")
        print(f"  总能量: {energy:.4f} eV")
        print(f"  力:\n{forces}")
        print(f"  最大力: {abs(forces).max():.4f} eV/Å")

        # 保存计算结果
        calc.write('h2_calculation.gpw')
        print(f"\n✓ 计算结果已保存到 h2_calculation.gpw")

    except ImportError:
        print("\n⚠ GPAW未安装，跳过计算部分")
        print("  请运行: conda install -c conda-forge gpaw")


def example_2_si_crystal():
    """
    示例2: 计算硅晶体的性质
    演示晶体结构的计算
    """
    print("\n" + "="*60)
    print("示例2: Si晶体性质计算")
    print("="*60)

    from ase.build import bulk
    from ase.io import write

    # 创建Si晶体
    si = bulk('Si', 'diamond', a=5.43)
    write('si_crystal.cif', si)
    print(f"✓ Si晶体已创建:")
    print(f"  原子数: {len(si)}")
    print(f"  晶格常数: {si.cell[0,0]:.3f} Å")
    print(f"  化学式: {si.get_chemical_formula()}")

    try:
        from gpaw import GPAW, PW

        # 设置计算器（需要k点采样）
        calc = GPAW(mode=PW(400),
                    xc='PBE',
                    kpts=(4, 4, 4),  # k点网格
                    txt='si_calculation.txt')

        si.calc = calc

        # 计算
        print("\n开始计算...")
        energy = si.get_potential_energy()
        forces = si.get_forces()
        stress = si.get_stress()

        print(f"\n结果:")
        print(f"  总能量: {energy:.4f} eV")
        print(f"  每原子能量: {energy/len(si):.4f} eV")
        print(f"  力:\n{forces}")
        print(f"  应力 (eV/Å³):\n{stress}")

        # 费米能级
        fermi = calc.get_fermi_level()
        print(f"  费米能级: {fermi:.3f} eV")

        # 保存
        calc.write('si_calculation.gpw')
        print(f"\n✓ 计算完成！")

    except ImportError:
        print("\n⚠ GPAW未安装，跳过计算部分")


def example_3_convergence_test():
    """
    示例3: 截断能收敛性测试
    演示如何测试计算参数的收敛性
    """
    print("\n" + "="*60)
    print("示例3: 截断能收敛性测试")
    print("="*60)

    from ase.build import bulk
    import numpy as np

    si = bulk('Si', 'diamond', a=5.43)

    # 测试不同的截断能
    encuts = [200, 300, 400, 500, 600]
    energies = []

    print(f"\n测试截断能: {encuts}")

    try:
        from gpaw import GPAW, PW

        for encut in encuts:
            print(f"\n计算 ENCUT = {encut} eV...")

            calc = GPAW(mode=PW(encut),
                       xc='PBE',
                       kpts=(8, 8, 8),
                       txt=f'si_encut_{encut}.txt')

            si.calc = calc
            energy = si.get_potential_energy()
            energies.append(energy / len(si))

            print(f"  能量/原子: {energy/len(si):.6f} eV")

        # 分析收敛性
        energies = np.array(energies)
        ref_energy = energies[-1]  # 最高截断能作为参考
        diff = (energies - ref_energy) * 1000  # meV

        print(f"\n收敛性分析:")
        print(f"{'ENCUT (eV)':>12} {'E/atom (eV)':>15} {'ΔE (meV)':>12}")
        print("-" * 42)
        for encut, e, d in zip(encuts, energies, diff):
            converged = " ✓" if abs(d) < 1.0 else ""
            print(f"{encut:>12} {e:>15.6f} {d:>12.3f}{converged}")

        print(f"\n建议截断能: ≥{encuts[np.argmax(abs(diff) < 1.0)]} eV")

    except ImportError:
        print("\n⚠ GPAW未安装，跳过计算部分")


def example_4_structure_optimization():
    """
    示例4: 结构优化
    演示如何优化原子结构
    """
    print("\n" + "="*60)
    print("示例4: 结构优化")
    print("="*60)

    from ase.build import bulk
    from ase.io import write

    # 创建稍微偏离平衡的Si结构
    si = bulk('Si', 'diamond', a=5.5)  # 比平衡值5.43大
    write('si_initial.cif', si)

    print(f"初始结构:")
    print(f"  晶格常数: {si.cell[0,0]:.3f} Å")

    try:
        from gpaw import GPAW, PW
        from ase.optimize import BFGS

        # 设置计算器
        calc = GPAW(mode=PW(400),
                    xc='PBE',
                    kpts=(8, 8, 8),
                    txt='si_optimization.txt')

        si.calc = calc

        # 结构优化
        print(f"\n开始优化...")
        opt = BFGS(si, trajectory='si_opt.traj', logfile='opt.log')
        opt.run(fmax=0.01)

        # 结果
        final_a = si.cell[0, 0]
        final_energy = si.get_potential_energy()

        print(f"\n优化结果:")
        print(f"  最终晶格常数: {final_a:.3f} Å")
        print(f"  最终能量: {final_energy:.4f} eV")
        print(f"  与实验值(5.43Å)差异: {abs(final_a - 5.43):.4f} Å")

        write('si_optimized.cif', si)
        print(f"\n✓ 优化完成！")

    except ImportError:
        print("\n⚠ GPAW未安装，跳过计算部分")


def demonstrate_without_gpaw():
    """
    不需要GPAW的演示
    展示如何准备结构和分析
    """
    print("\n" + "="*60)
    print("结构准备示例（不需要GPAW）")
    print("="*60)

    from ase.build import bulk, molecule, surface
    from ase.io import write
    import os

    os.makedirs('demo_structures', exist_ok=True)

    # 1. 晶体
    si = bulk('Si', 'diamond', a=5.43)
    write('demo_structures/si.cif', si)
    print(f"✓ Si晶体: {len(si)} atoms")

    # 2. 分子
    h2o = molecule('H2O')
    h2o.set_cell([10, 10, 10])
    h2o.center()
    write('demo_structures/h2o.xyz', h2o)
    print(f"✓ H2O分子: {len(h2o)} atoms")

    # 3. 表面
    au111 = surface('Au', (1, 1, 1), layers=4)
    au111.center(vacuum=10.0, axis=2)
    write('demo_structures/au111.cif', au111)
    print(f"✓ Au(111)表面: {len(au111)} atoms")

    print(f"\n所有结构已保存到 demo_structures/ 目录")


def main():
    """主函数"""
    print("\n" + "#"*60)
    print("  GPAW计算示例")
    print("#"*60)

    # 首先展示不需要GPAW的部分
    demonstrate_without_gpaw()

    # 检查GPAW是否安装
    try:
        import gpaw
        print(f"\n✓ GPAW已安装 (版本: {gpaw.__version__})")
        print("  将运行完整计算示例\n")

        # 运行所有示例
        example_1_h2_molecule()
        example_2_si_crystal()

        # 收敛性测试和优化比较耗时，可以选择性运行
        print("\n" + "-"*60)
        response = input("是否运行收敛性测试？(较耗时) [y/N]: ")
        if response.lower() == 'y':
            example_3_convergence_test()

        print("\n" + "-"*60)
        response = input("是否运行结构优化？(较耗时) [y/N]: ")
        if response.lower() == 'y':
            example_4_structure_optimization()

    except ImportError:
        print("\n⚠ GPAW未安装")
        print("  已展示结构准备部分")
        print("  要运行完整计算，请安装GPAW:")
        print("    conda install -c conda-forge gpaw")

    print("\n" + "#"*60)
    print("  示例演示完成！")
    print("#"*60 + "\n")


if __name__ == '__main__':
    main()
