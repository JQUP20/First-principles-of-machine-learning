#!/usr/bin/env python3
"""
晶格常数优化示例
演示如何使用NumPy和Matplotlib分析晶格常数与能量的关系
"""

import numpy as np
import matplotlib.pyplot as plt

def morse_potential(r, D=1.0, a=1.0, r0=1.0):
    """
    Morse势能函数
    E(r) = D * [1 - exp(-a(r-r0))]^2

    Parameters:
    -----------
    r : float or array
        原子间距
    D : float
        势阱深度
    a : float
        势阱宽度参数
    r0 : float
        平衡距离

    Returns:
    --------
    float or array
        势能
    """
    return D * (1 - np.exp(-a * (r - r0)))**2


def simple_crystal_energy(a, a0=5.43, E0=-100.0):
    """
    简化的晶体能量模型（抛物线近似）

    Parameters:
    -----------
    a : float or array
        晶格常数 (Å)
    a0 : float
        平衡晶格常数 (Å)
    E0 : float
        最小能量 (eV)

    Returns:
    --------
    float or array
        能量 (eV)
    """
    # 简单的二次型: E = E0 + k*(a-a0)^2
    k = 10.0  # 弹性常数
    return E0 + k * (a - a0)**2


def find_minimum(a_values, energies):
    """
    找到能量最小值及对应的晶格常数

    Parameters:
    -----------
    a_values : array
        晶格常数数组
    energies : array
        能量数组

    Returns:
    --------
    tuple
        (最优晶格常数, 最小能量, 最小值索引)
    """
    min_idx = np.argmin(energies)
    a_min = a_values[min_idx]
    e_min = energies[min_idx]

    return a_min, e_min, min_idx


def fit_parabola(a_values, energies):
    """
    用抛物线拟合E-V曲线

    Parameters:
    -----------
    a_values : array
        晶格常数
    energies : array
        能量

    Returns:
    --------
    tuple
        拟合参数 (a, b, c) for E = a*x^2 + b*x + c
    """
    # 多项式拟合（2次）
    coeffs = np.polyfit(a_values, energies, 2)
    return coeffs


def plot_energy_vs_lattice(a_values, energies, a_min, e_min):
    """
    绘制能量-晶格常数曲线

    Parameters:
    -----------
    a_values : array
        晶格常数数组
    energies : array
        能量数组
    a_min : float
        最优晶格常数
    e_min : float
        最小能量
    """
    plt.figure(figsize=(10, 6))

    # 原始数据点
    plt.plot(a_values, energies, 'bo-', linewidth=2, markersize=8, label='Calculated')

    # 标记最小值
    plt.plot(a_min, e_min, 'r*', markersize=20, label=f'Minimum: a={a_min:.3f} Å')

    # 拟合曲线
    coeffs = fit_parabola(a_values, energies)
    a_fit = np.linspace(a_values.min(), a_values.max(), 100)
    e_fit = np.polyval(coeffs, a_fit)
    plt.plot(a_fit, e_fit, 'r--', linewidth=1.5, alpha=0.7, label='Parabolic fit')

    plt.xlabel('Lattice Constant (Å)', fontsize=14)
    plt.ylabel('Energy (eV)', fontsize=14)
    plt.title('Crystal Energy vs. Lattice Constant', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图片
    plt.savefig('lattice_optimization.png', dpi=300, bbox_inches='tight')
    print("图片已保存: lattice_optimization.png")
    plt.show()


def calculate_bulk_modulus(a_values, energies, a0, V0):
    """
    计算体模量

    B = V * d²E/dV² |_{V=V0}

    Parameters:
    -----------
    a_values : array
        晶格常数
    energies : array
        能量
    a0 : float
        平衡晶格常数
    V0 : float
        平衡体积

    Returns:
    --------
    float
        体模量 (GPa)
    """
    # 简化计算：使用二阶导数
    # 对于立方晶系：V = a³
    # dE/dV = (dE/da) * (da/dV) = (dE/da) / (3a²)

    # 数值微分计算二阶导数
    coeffs = fit_parabola(a_values, energies)
    # E = c0*a² + c1*a + c2
    # dE/da = 2*c0*a + c1
    # d²E/da² = 2*c0

    d2E_da2 = 2 * coeffs[0]

    # 转换为体模量
    # B = V * d²E/dV² = V * (d²E/da²) / (9*a0²)
    B = V0 * d2E_da2 / (9 * a0**2)

    # 转换单位: eV/Å³ → GPa
    # 1 eV/Å³ = 160.21766208 GPa
    B_GPa = B * 160.21766208

    return B_GPa


def main():
    """主函数"""
    print("=" * 60)
    print("  晶格常数优化示例")
    print("=" * 60)

    # 参数设置
    a0_true = 5.43  # Si的实验晶格常数
    a_values = np.linspace(5.1, 5.7, 13)  # 扫描范围

    print(f"\n扫描晶格常数范围: {a_values[0]:.2f} - {a_values[-1]:.2f} Å")
    print(f"步长: {a_values[1] - a_values[0]:.3f} Å")
    print(f"计算点数: {len(a_values)}\n")

    # 计算能量
    print("计算能量...")
    energies = simple_crystal_energy(a_values, a0=a0_true, E0=-100.0)

    # 显示结果表格
    print("\n" + "-" * 60)
    print(f"{'晶格常数 (Å)':^20} {'能量 (eV)':^20} {'相对能量 (meV)':^20}")
    print("-" * 60)

    e_ref = energies.min()
    for a, e in zip(a_values, energies):
        delta_e = (e - e_ref) * 1000  # 转换为meV
        marker = " ← 最小值" if abs(e - e_ref) < 1e-6 else ""
        print(f"{a:^20.3f} {e:^20.4f} {delta_e:^20.2f}{marker}")

    print("-" * 60)

    # 找到最小值
    a_min, e_min, min_idx = find_minimum(a_values, energies)

    print(f"\n最优化结果:")
    print(f"  最优晶格常数: {a_min:.3f} Å")
    print(f"  最小能量: {e_min:.4f} eV")
    print(f"  与真实值差异: {abs(a_min - a0_true):.4f} Å")

    # 计算体模量
    V0 = a_min**3  # 对于简单立方
    B = calculate_bulk_modulus(a_values, energies, a_min, V0)
    print(f"  体模量: {B:.1f} GPa")

    # 拟合分析
    coeffs = fit_parabola(a_values, energies)
    print(f"\n抛物线拟合: E = {coeffs[0]:.3f}*a² + {coeffs[1]:.3f}*a + {coeffs[2]:.3f}")

    # 从拟合求最小值
    a_fit_min = -coeffs[1] / (2 * coeffs[0])
    e_fit_min = np.polyval(coeffs, a_fit_min)
    print(f"拟合得到的最小值:")
    print(f"  a_min = {a_fit_min:.3f} Å")
    print(f"  E_min = {e_fit_min:.4f} eV")

    # 绘图
    print("\n绘制能量曲线...")
    plot_energy_vs_lattice(a_values, energies, a_min, e_min)

    print("\n" + "=" * 60)
    print("  计算完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
