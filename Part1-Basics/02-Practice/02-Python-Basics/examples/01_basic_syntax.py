#!/usr/bin/env python3
"""
Python基本语法示例
演示数据类型、控制流、函数等
"""

def demonstrate_data_types():
    """演示基本数据类型"""
    print("=" * 50)
    print("1. 数据类型演示")
    print("=" * 50)

    # 数值类型
    atomic_number = 14  # 硅的原子序数
    lattice_constant = 5.43  # 埃
    energy = -123.456  # eV

    print(f"原子序数: {atomic_number} (类型: {type(atomic_number).__name__})")
    print(f"晶格常数: {lattice_constant} Å (类型: {type(lattice_constant).__name__})")
    print(f"能量: {energy:.2f} eV")

    # 字符串
    element = "Silicon"
    formula = "Si"
    print(f"\n元素: {element}, 化学式: {formula}")

    # 列表
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O']
    print(f"\n前8个元素: {elements}")
    print(f"第一个元素: {elements[0]}")
    print(f"最后一个元素: {elements[-1]}")

    # 字典
    atom_info = {
        'symbol': 'Si',
        'atomic_number': 14,
        'mass': 28.0855,
        'group': 14
    }
    print(f"\n原子信息:")
    for key, value in atom_info.items():
        print(f"  {key}: {value}")


def demonstrate_control_flow():
    """演示控制流"""
    print("\n" + "=" * 50)
    print("2. 控制流演示")
    print("=" * 50)

    # if语句
    energy = -100.5
    if energy < -100.0:
        status = "低能量，结构稳定"
    elif energy < -50.0:
        status = "中等能量"
    else:
        status = "高能量"
    print(f"\n能量: {energy} eV")
    print(f"状态: {status}")

    # for循环
    print("\n计算前5个数的平方:")
    for i in range(5):
        print(f"  {i}² = {i**2}")

    # 列表推导式
    squares = [x**2 for x in range(10)]
    print(f"\n前10个平方数: {squares}")

    # 实用例子：找到收敛的迭代步
    print("\n自洽迭代模拟:")
    energies = [-100.0, -100.3, -100.45, -100.48, -100.49, -100.495, -100.497]
    tolerance = 0.01

    for i in range(1, len(energies)):
        energy_diff = abs(energies[i] - energies[i-1])
        print(f"  步骤 {i}: E = {energies[i]:.3f} eV, ΔE = {energy_diff:.4f} eV")

        if energy_diff < tolerance:
            print(f"  ✓ 在第{i}步收敛！")
            break
    else:
        print("  ✗ 未收敛")


def calculate_distance(pos1, pos2):
    """
    计算两个原子之间的距离

    Parameters:
    -----------
    pos1, pos2 : list
        原子位置 [x, y, z]

    Returns:
    --------
    float
        距离（埃）
    """
    import math
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def calculate_bond_angles(positions):
    """
    计算分子的键角

    Parameters:
    -----------
    positions : list of lists
        原子位置列表

    Returns:
    --------
    list
        键角列表
    """
    import math

    if len(positions) < 3:
        return []

    angles = []
    # 简化版：只计算第一个键角
    # 这里应该用向量运算，简化处理
    return angles


def demonstrate_functions():
    """演示函数使用"""
    print("\n" + "=" * 50)
    print("3. 函数演示")
    print("=" * 50)

    # 计算原子间距离
    atom1_pos = [0.0, 0.0, 0.0]
    atom2_pos = [1.5, 1.5, 1.5]

    distance = calculate_distance(atom1_pos, atom2_pos)
    print(f"\n原子1位置: {atom1_pos}")
    print(f"原子2位置: {atom2_pos}")
    print(f"距离: {distance:.3f} Å")

    # Lambda函数
    square = lambda x: x ** 2
    print(f"\n使用lambda函数: 5² = {square(5)}")

    # 列表操作
    numbers = [1, 2, 3, 4, 5]
    squared = list(map(lambda x: x**2, numbers))
    print(f"原列表: {numbers}")
    print(f"平方后: {squared}")


class Atom:
    """原子类"""

    def __init__(self, symbol, position):
        self.symbol = symbol
        self.position = position
        self.mass = self._get_mass()

    def _get_mass(self):
        """获取原子质量"""
        masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007,
            'O': 15.999, 'Si': 28.086
        }
        return masses.get(self.symbol, 0.0)

    def move(self, displacement):
        """移动原子"""
        self.position = [p + d for p, d in zip(self.position, displacement)]

    def __str__(self):
        return f"{self.symbol} at ({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})"


def demonstrate_classes():
    """演示类和对象"""
    print("\n" + "=" * 50)
    print("4. 类和对象演示")
    print("=" * 50)

    # 创建原子对象
    si_atom = Atom('Si', [0.0, 0.0, 0.0])
    o_atom = Atom('O', [1.5, 1.5, 1.5])

    print(f"\n创建原子:")
    print(f"  {si_atom}")
    print(f"  质量: {si_atom.mass} amu")

    print(f"\n  {o_atom}")
    print(f"  质量: {o_atom.mass} amu")

    # 移动原子
    print(f"\n移动Si原子 [0.1, 0.0, 0.0]")
    si_atom.move([0.1, 0.0, 0.0])
    print(f"  新位置: {si_atom}")


def demonstrate_numpy():
    """演示NumPy使用"""
    print("\n" + "=" * 50)
    print("5. NumPy演示")
    print("=" * 50)

    try:
        import numpy as np

        # 创建数组
        energies = np.array([-100.5, -100.3, -100.4, -100.6, -100.2])
        print(f"\n能量数组: {energies}")
        print(f"平均能量: {np.mean(energies):.3f} eV")
        print(f"标准差: {np.std(energies):.3f} eV")
        print(f"最小能量: {np.min(energies):.3f} eV")
        print(f"最大能量: {np.max(energies):.3f} eV")

        # 数组操作
        forces = np.array([
            [0.01, -0.02, 0.00],
            [0.00,  0.01, -0.01],
            [-0.01, 0.01, 0.01]
        ])

        print(f"\n力矩阵:")
        print(forces)
        print(f"最大力: {np.max(np.abs(forces)):.4f} eV/Å")
        print(f"力的范数: {np.linalg.norm(forces, axis=1)}")

    except ImportError:
        print("\nNumPy未安装，请运行: pip install numpy")


def main():
    """主函数"""
    print("\n")
    print("*" * 50)
    print("  Python基础语法演示")
    print("*" * 50)

    demonstrate_data_types()
    demonstrate_control_flow()
    demonstrate_functions()
    demonstrate_classes()
    demonstrate_numpy()

    print("\n" + "*" * 50)
    print("  演示完成！")
    print("*" * 50)


if __name__ == '__main__':
    main()
