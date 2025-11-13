# Python编程基础 | Python Programming Basics

## 目录 | Table of Contents

1. [Python简介](#1-python简介)
2. [数据类型](#2-数据类型)
3. [控制流程](#3-控制流程)
4. [函数](#4-函数)
5. [类和对象](#5-类和对象)
6. [模块和包](#6-模块和包)
7. [科学计算库](#7-科学计算库)
8. [PyCharm使用](#8-pycharm使用)

---

## 1. Python简介

### 1.1 为什么选择Python？

在科学计算和第一性原理计算领域，Python已成为首选语言：

**优势**：
- 🎯 **语法简洁**：易于学习和使用
- 📦 **丰富的库**：NumPy, SciPy, ASE, Pymatgen等
- 🔬 **科学计算**：强大的数值计算能力
- 🤝 **社区支持**：活跃的科研社区
- 🔗 **接口友好**：易于与C/Fortran代码集成

### 1.2 Python版本

**推荐使用 Python 3.8+**

检查Python版本：
```bash
$ python --version
Python 3.9.7

$ python3 --version
Python 3.9.7
```

### 1.3 第一个Python程序

创建 `hello.py`:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
我的第一个Python程序
"""

print("Hello, Python!")
print("欢迎来到第一性原理计算的世界！")
```

运行：
```bash
$ python hello.py
Hello, Python!
欢迎来到第一性原理计算的世界！
```

---

## 2. 数据类型

### 2.1 基本数据类型

#### 数值类型

```python
# 整数 (int)
n = 42
atomic_number = 14  # Si的原子序数

# 浮点数 (float)
lattice_constant = 5.43  # 埃
energy = -123.456  # eV

# 复数 (complex)
wave = 1 + 2j

# 类型查询
print(type(n))  # <class 'int'>
print(type(lattice_constant))  # <class 'float'>
```

#### 字符串 (str)

```python
# 单引号或双引号
element = 'Silicon'
formula = "SiO2"

# 三引号（多行字符串）
description = """
这是一个硅晶体结构
晶格常数为 5.43 埃
"""

# 字符串操作
print(element.upper())  # SILICON
print(element.lower())  # silicon
print(len(formula))  # 4

# 字符串拼接
message = element + " " + formula
message = f"{element} {formula}"  # f-string (推荐)

# 字符串格式化
energy = -123.456
print(f"Energy = {energy:.2f} eV")  # Energy = -123.46 eV
print(f"Energy = {energy:.4e} eV")  # Energy = -1.2346e+02 eV
```

#### 布尔值 (bool)

```python
is_converged = True
has_error = False

# 逻辑运算
result = is_converged and (not has_error)

# 比较运算
x = 5
print(x > 3)  # True
print(x == 5)  # True
print(x != 5)  # False
```

### 2.2 容器类型

#### 列表 (List)

```python
# 创建列表
elements = ['Si', 'O', 'C', 'N']
energies = [-100.5, -100.3, -100.4]
mixed = [1, 'atom', 3.14, True]

# 访问元素（索引从0开始）
print(elements[0])  # Si
print(elements[-1])  # N (最后一个)

# 切片
print(elements[1:3])  # ['O', 'C']
print(elements[:2])   # ['Si', 'O']
print(elements[2:])   # ['C', 'N']

# 修改元素
elements[0] = 'Silicon'

# 添加元素
elements.append('H')
elements.extend(['He', 'Li'])
elements.insert(0, 'First')

# 删除元素
elements.remove('First')
last = elements.pop()  # 删除并返回最后一个
del elements[0]

# 列表操作
print(len(elements))  # 长度
print('Si' in elements)  # 检查是否存在
elements.sort()  # 排序
elements.reverse()  # 反转

# 列表推导式（重要！）
squares = [x**2 for x in range(10)]
even_numbers = [x for x in range(20) if x % 2 == 0]
```

#### 元组 (Tuple)

```python
# 不可变的序列
position = (0.0, 0.0, 0.0)
lattice_params = (5.43, 5.43, 5.43, 90, 90, 90)

# 访问元素
x, y, z = position  # 解包

# 元组用于函数返回多个值
def get_energy_and_forces():
    return -100.5, [0.01, -0.02, 0.00]

energy, forces = get_energy_and_forces()
```

#### 字典 (Dictionary)

```python
# 键值对
atom = {
    'symbol': 'Si',
    'atomic_number': 14,
    'mass': 28.0855,
    'position': [0.0, 0.0, 0.0]
}

# 访问
print(atom['symbol'])  # Si
print(atom.get('symbol'))  # Si
print(atom.get('charge', 0))  # 默认值

# 修改和添加
atom['position'] = [0.1, 0.1, 0.1]
atom['charge'] = 0

# 删除
del atom['charge']
value = atom.pop('mass')

# 遍历
for key in atom:
    print(f"{key}: {atom[key]}")

for key, value in atom.items():
    print(f"{key}: {value}")

# 字典推导式
squared = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

#### 集合 (Set)

```python
# 无序不重复元素集
elements = {'Si', 'O', 'C', 'Si'}  # 重复的Si只保留一个
print(elements)  # {'Si', 'O', 'C'}

# 集合操作
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)  # 并集 {1, 2, 3, 4, 5, 6}
print(a & b)  # 交集 {3, 4}
print(a - b)  # 差集 {1, 2}
print(a ^ b)  # 对称差 {1, 2, 5, 6}
```

---

## 3. 控制流程

### 3.1 条件语句

```python
# if语句
energy = -100.5

if energy < -100.0:
    print("能量较低，结构稳定")
elif energy < -50.0:
    print("能量中等")
else:
    print("能量较高")

# 三元运算符
status = "收敛" if abs(energy) < 0.001 else "未收敛"

# 实用例子：检查收敛性
def check_convergence(energy_diff, force_max):
    """检查计算是否收敛"""
    if energy_diff < 1e-4 and force_max < 0.01:
        return True, "已收敛"
    elif energy_diff < 1e-3:
        return False, "能量接近收敛"
    else:
        return False, "未收敛"
```

### 3.2 循环语句

#### for循环

```python
# 遍历列表
elements = ['Si', 'O', 'C']
for element in elements:
    print(f"元素: {element}")

# range函数
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 10, 2):  # 1, 3, 5, 7, 9
    print(i)

# enumerate（获取索引和值）
for i, element in enumerate(elements):
    print(f"{i}: {element}")

# 遍历字典
atom = {'symbol': 'Si', 'number': 14}
for key, value in atom.items():
    print(f"{key} = {value}")

# 实用例子：计算平均能量
energies = [-100.5, -100.3, -100.4, -100.6]
total = 0.0
for energy in energies:
    total += energy
average = total / len(energies)
print(f"平均能量: {average:.3f} eV")

# 更Pythonic的方式
average = sum(energies) / len(energies)
```

#### while循环

```python
# while循环
count = 0
while count < 5:
    print(count)
    count += 1

# 实用例子：自洽迭代
max_iter = 100
tolerance = 1e-6
energy_old = 0.0

for iteration in range(max_iter):
    energy_new = calculate_energy()  # 假设的函数

    if abs(energy_new - energy_old) < tolerance:
        print(f"在第{iteration}步收敛")
        break

    energy_old = energy_new
else:
    print("未在最大迭代次数内收敛")
```

#### 循环控制

```python
# break: 跳出循环
for i in range(10):
    if i == 5:
        break
    print(i)  # 0, 1, 2, 3, 4

# continue: 跳过当前迭代
for i in range(5):
    if i == 2:
        continue
    print(i)  # 0, 1, 3, 4

# pass: 占位符
for i in range(5):
    if i == 2:
        pass  # 以后实现
    print(i)
```

---

## 4. 函数

### 4.1 定义函数

```python
def calculate_energy(lattice_constant):
    """
    计算晶体能量

    Parameters:
    -----------
    lattice_constant : float
        晶格常数（埃）

    Returns:
    --------
    float
        能量（eV）
    """
    # 简单的示例公式
    energy = -100.0 * (5.43 / lattice_constant)**2
    return energy

# 调用函数
e = calculate_energy(5.43)
print(f"能量: {e:.2f} eV")
```

### 4.2 参数类型

```python
# 位置参数
def add(a, b):
    return a + b

# 默认参数
def calculate(encut=400, kpts=4, xc='PBE'):
    print(f"ENCUT: {encut}, KPTS: {kpts}, XC: {xc}")

calculate()  # 使用默认值
calculate(encut=500)  # 指定部分参数
calculate(500, 6, 'LDA')  # 位置参数

# 可变参数
def sum_all(*args):
    """接受任意数量的参数"""
    return sum(args)

print(sum_all(1, 2, 3, 4))  # 10

# 关键字参数
def set_parameters(**kwargs):
    """接受任意关键字参数"""
    for key, value in kwargs.items():
        print(f"{key} = {value}")

set_parameters(encut=400, kpts=4, ismear=1)
```

### 4.3 返回值

```python
# 单个返回值
def square(x):
    return x ** 2

# 多个返回值（实际返回元组）
def get_structure_info():
    n_atoms = 8
    volume = 160.32
    formula = "Si8"
    return n_atoms, volume, formula

n, v, f = get_structure_info()

# 返回None（隐式）
def print_info(message):
    print(message)
    # 没有return语句，返回None
```

### 4.4 Lambda函数

```python
# 匿名函数
square = lambda x: x ** 2
print(square(5))  # 25

# 常用于排序
atoms = [
    {'symbol': 'O', 'number': 8},
    {'symbol': 'C', 'number': 6},
    {'symbol': 'N', 'number': 7}
]

# 按原子序数排序
sorted_atoms = sorted(atoms, key=lambda x: x['number'])
```

### 4.5 实用示例

```python
def optimize_structure(atoms, fmax=0.01, max_steps=100):
    """
    优化原子结构

    Parameters:
    -----------
    atoms : object
        原子结构对象
    fmax : float
        力的收敛标准 (eV/Å)
    max_steps : int
        最大优化步数

    Returns:
    --------
    dict
        包含能量、力和步数的字典
    """
    from random import random

    for step in range(max_steps):
        # 模拟计算
        energy = -100.0 - random() * 0.1
        forces_max = 0.1 * (1 - step / max_steps)

        print(f"Step {step}: E = {energy:.4f} eV, "
              f"F_max = {forces_max:.4f} eV/Å")

        if forces_max < fmax:
            return {
                'converged': True,
                'energy': energy,
                'forces_max': forces_max,
                'steps': step
            }

    return {
        'converged': False,
        'energy': energy,
        'forces_max': forces_max,
        'steps': max_steps
    }

# 使用函数
result = optimize_structure(None, fmax=0.01, max_steps=50)
print(f"\n收敛状态: {result['converged']}")
print(f"最终能量: {result['energy']:.4f} eV")
```

---

## 5. 类和对象

### 5.1 定义类

```python
class Atom:
    """原子类"""

    def __init__(self, symbol, position):
        """
        初始化原子

        Parameters:
        -----------
        symbol : str
            元素符号
        position : list
            原子位置 [x, y, z]
        """
        self.symbol = symbol
        self.position = position
        self.mass = self._get_mass()

    def _get_mass(self):
        """获取原子质量（简化版）"""
        masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007,
            'O': 15.999, 'Si': 28.086
        }
        return masses.get(self.symbol, 0.0)

    def move(self, displacement):
        """移动原子"""
        self.position = [
            p + d for p, d in zip(self.position, displacement)
        ]

    def distance_to(self, other):
        """计算到另一个原子的距离"""
        import math
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        dz = self.position[2] - other.position[2]
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def __str__(self):
        """字符串表示"""
        return f"{self.symbol} at {self.position}"

    def __repr__(self):
        """调试表示"""
        return f"Atom('{self.symbol}', {self.position})"

# 使用类
atom1 = Atom('Si', [0.0, 0.0, 0.0])
atom2 = Atom('O', [1.5, 1.5, 1.5])

print(atom1)  # Si at [0.0, 0.0, 0.0]
print(f"质量: {atom1.mass} amu")

atom1.move([0.1, 0.0, 0.0])
print(atom1)  # Si at [0.1, 0.0, 0.0]

distance = atom1.distance_to(atom2)
print(f"距离: {distance:.3f} Å")
```

### 5.2 继承

```python
class Molecule:
    """分子类（基类）"""

    def __init__(self, atoms):
        self.atoms = atoms

    def get_n_atoms(self):
        """获取原子数"""
        return len(self.atoms)

    def get_formula(self):
        """获取化学式"""
        from collections import Counter
        symbols = [atom.symbol for atom in self.atoms]
        counts = Counter(symbols)
        formula = ''.join(f"{s}{n if n > 1 else ''}"
                         for s, n in sorted(counts.items()))
        return formula

class WaterMolecule(Molecule):
    """水分子类（派生类）"""

    def __init__(self):
        # 创建H2O结构
        atoms = [
            Atom('O', [0.0, 0.0, 0.0]),
            Atom('H', [0.96, 0.0, 0.0]),
            Atom('H', [-0.24, 0.93, 0.0])
        ]
        super().__init__(atoms)

    def get_dipole_moment(self):
        """计算偶极矩（简化）"""
        return 1.85  # Debye

# 使用
water = WaterMolecule()
print(f"化学式: {water.get_formula()}")  # H2O
print(f"原子数: {water.get_n_atoms()}")  # 3
print(f"偶极矩: {water.get_dipole_moment()} D")
```

### 5.3 特殊方法

```python
class Vector3D:
    """3D向量类"""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        """向量加法: v1 + v2"""
        return Vector3D(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __mul__(self, scalar):
        """标量乘法: v * scalar"""
        return Vector3D(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __len__(self):
        """向量长度"""
        import math
        return int(math.sqrt(self.x**2 + self.y**2 + self.z**2))

# 使用
v1 = Vector3D(1, 2, 3)
v2 = Vector3D(4, 5, 6)
v3 = v1 + v2
v4 = v1 * 2

print(v3)  # (5, 7, 9)
print(v4)  # (2, 4, 6)
```

---

## 6. 模块和包

### 6.1 导入模块

```python
# 导入整个模块
import math
print(math.pi)
print(math.sqrt(16))

# 导入特定函数
from math import pi, sqrt
print(pi)
print(sqrt(16))

# 导入所有（不推荐）
from math import *

# 使用别名
import numpy as np
import matplotlib.pyplot as plt

# 导入子模块
from os.path import join, exists
```

### 6.2 创建模块

创建 `mymodule.py`:
```python
"""
我的第一性原理计算工具模块
"""

# 常量
HARTREE_TO_EV = 27.211386
BOHR_TO_ANGSTROM = 0.529177

# 函数
def hartree_to_ev(energy_hartree):
    """将Hartree转换为eV"""
    return energy_hartree * HARTREE_TO_EV

def bohr_to_angstrom(length_bohr):
    """将Bohr转换为Angstrom"""
    return length_bohr * BOHR_TO_ANGSTROM

# 类
class Calculator:
    """简单的计算器类"""

    def __init__(self, name):
        self.name = name

    def calculate(self):
        return f"{self.name} calculating..."

# 测试代码（仅在直接运行时执行）
if __name__ == '__main__':
    print("测试模块功能")
    print(f"1 Hartree = {hartree_to_ev(1)} eV")
    print(f"1 Bohr = {bohr_to_angstrom(1)} Å")
```

使用模块：
```python
import mymodule

energy_ev = mymodule.hartree_to_ev(1.0)
print(f"能量: {energy_ev} eV")

calc = mymodule.Calculator("GPAW")
print(calc.calculate())
```

### 6.3 包结构

```
mypackage/
├── __init__.py
├── core.py
├── utils.py
└── io/
    ├── __init__.py
    ├── read.py
    └── write.py
```

`__init__.py`:
```python
"""
My DFT Package
"""

__version__ = '0.1.0'

from .core import Calculator
from .utils import hartree_to_ev

__all__ = ['Calculator', 'hartree_to_ev']
```

使用包：
```python
import mypackage
from mypackage import Calculator
from mypackage.io import read_structure
```

---

## 7. 科学计算库

### 7.1 NumPy基础

```python
import numpy as np

# 创建数组
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 3))  # 3x3零矩阵
c = np.ones((2, 4))   # 2x4全1矩阵
d = np.eye(3)         # 3x3单位矩阵
e = np.linspace(0, 10, 11)  # 0到10的11个点

# 数组操作
print(a + 10)  # [11 12 13 14 15]
print(a * 2)   # [ 2  4  6  8 10]
print(a ** 2)  # [ 1  4  9 16 25]

# 数学函数
print(np.sin(a))
print(np.exp(a))
print(np.sqrt(a))

# 统计
print(np.mean(a))  # 平均值
print(np.std(a))   # 标准差
print(np.max(a))   # 最大值
print(np.argmax(a))  # 最大值索引

# 线性代数
matrix = np.array([[1, 2], [3, 4]])
print(np.linalg.det(matrix))  # 行列式
print(np.linalg.inv(matrix))  # 逆矩阵

# 实用例子：计算平均力
forces = np.array([
    [0.01, -0.02, 0.00],
    [0.00,  0.01, -0.01],
    [-0.01, 0.01, 0.01]
])

max_force = np.max(np.abs(forces))
mean_force = np.mean(np.abs(forces))
print(f"最大力: {max_force:.4f} eV/Å")
print(f"平均力: {mean_force:.4f} eV/Å")
```

### 7.2 Matplotlib绘图

```python
import matplotlib.pyplot as plt
import numpy as np

# 简单折线图
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Function')
plt.legend()
plt.grid(True)
plt.savefig('sine.png', dpi=300)
plt.show()

# 实用例子：能量收敛图
iterations = np.arange(1, 21)
energies = -100.0 + 0.5 * np.exp(-iterations / 5)

plt.figure(figsize=(10, 6))
plt.plot(iterations, energies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Energy (eV)', fontsize=12)
plt.title('SCF Convergence', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 8. PyCharm使用

### 8.1 安装PyCharm

**下载**：https://www.jetbrains.com/pycharm/

**版本选择**：
- **Community Edition**: 免费，适合学习
- **Professional Edition**: 付费，功能更全

### 8.2 基本功能

#### 创建项目
1. File → New Project
2. 选择位置和Python解释器
3. Create

#### 运行代码
- **快捷键**: `Ctrl+Shift+F10` (首次运行)
- **快捷键**: `Shift+F10` (运行)
- **右键**: Run 'filename'

#### 调试
- **设置断点**: 点击行号旁边
- **调试运行**: `Shift+F9`
- **步过**: `F8`
- **步入**: `F7`
- **继续**: `F9`

### 8.3 常用快捷键

```
Ctrl + Space      # 代码补全
Ctrl + /          # 注释/取消注释
Ctrl + D          # 复制行
Ctrl + Y          # 删除行
Ctrl + F          # 查找
Ctrl + R          # 替换
Ctrl + Alt + L    # 格式化代码
Shift + Shift     # 搜索任何内容
Ctrl + Click      # 跳转到定义
```

### 8.4 实用配置

**设置Python解释器**:
- File → Settings → Project → Python Interpreter
- 添加Conda环境或虚拟环境

**代码风格**:
- File → Settings → Editor → Code Style → Python
- 设置缩进为4个空格
- 行长度限制为79或88

**快速文档**:
- 将光标放在函数上
- 按 `Ctrl+Q` 查看文档

---

## 9. 实践项目

### 项目1：晶格常数优化

```python
"""
简单的晶格常数优化示例
"""
import numpy as np
import matplotlib.pyplot as plt

def calculate_energy(lattice_constant, a0=5.43):
    """
    计算能量（使用简单的抛物线模型）

    Parameters:
    -----------
    lattice_constant : float
        晶格常数
    a0 : float
        平衡晶格常数

    Returns:
    --------
    float
        能量
    """
    return 10.0 * (lattice_constant - a0)**2 - 100.0

def optimize_lattice():
    """优化晶格常数"""
    # 扫描范围
    a_values = np.linspace(5.0, 5.8, 20)
    energies = [calculate_energy(a) for a in a_values]

    # 找最小值
    min_idx = np.argmin(energies)
    a_opt = a_values[min_idx]
    e_min = energies[min_idx]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(a_values, energies, 'b-', linewidth=2, label='E(a)')
    plt.plot(a_opt, e_min, 'ro', markersize=10, label=f'Minimum at a={a_opt:.3f} Å')
    plt.xlabel('Lattice Constant (Å)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title('Lattice Constant Optimization', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lattice_optimization.png', dpi=300)
    plt.show()

    print(f"最优晶格常数: {a_opt:.3f} Å")
    print(f"最小能量: {e_min:.3f} eV")

if __name__ == '__main__':
    optimize_lattice()
```

---

## 10. 学习检查清单

- [ ] 理解Python基本数据类型
- [ ] 掌握列表、字典等容器类型
- [ ] 能编写函数和类
- [ ] 了解模块和包的使用
- [ ] 熟悉NumPy数组操作
- [ ] 能使用Matplotlib绘制基本图形
- [ ] 会使用PyCharm进行开发

---

## 参考资源

- [Python官方文档](https://docs.python.org/3/)
- [NumPy文档](https://numpy.org/doc/)
- [Matplotlib文档](https://matplotlib.org/)
- [Real Python教程](https://realpython.com/)

**练习答案和更多示例**：查看 `examples/` 目录 📁

**下一步**：Anaconda环境管理 →
