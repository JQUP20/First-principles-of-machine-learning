# 第一性原理与机器学习课程
# First-Principles Calculations and Machine Learning Course

![Course Banner](https://img.shields.io/badge/Course-First--Principles--ML-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 课程简介 | Course Introduction

本课程系统地介绍了第一性原理计算的基础理论和实践应用，以及如何将深度学习技术与第一性原理计算相结合。课程涵盖从基础的Linux操作、Python编程到高级的密度泛函理论（DFT）计算和原子建模。

This course systematically introduces the fundamental theories and practical applications of first-principles calculations, and how to combine deep learning techniques with first-principles methods. The curriculum covers everything from basic Linux operations and Python programming to advanced Density Functional Theory (DFT) calculations and atomic modeling.

## 课程特色 | Course Features

- **理论与实践结合**: 每个理论章节都配有实际操作示例
- **循序渐进**: 从Linux基础到高级DFT计算的完整学习路径
- **实用工具**: 涵盖ASE、GPAW、VASP等主流计算工具
- **机器学习整合**: 展示如何将深度学习应用于第一性原理计算

## 课程结构 | Course Structure

### 第一部分：第一性原理基础和Python编程
### Part 1: First-Principles Foundations and Python Programming

#### 1. 理论内容 | Theoretical Content

##### (1) 课程引言 | Course Introduction
- 深度学习在第一性原理的应用和优势
- 课程内容安排

📖 [查看详细内容](./Part1-Basics/01-Theory/01-Course-Introduction.md)

##### (2) 第一性原理计算介绍 | Introduction to First-Principles Calculations
- 第一性原理计算的发展历程——从薛定谔方程到密度泛函理论
- 密度泛函理论（DFT）——从波函数到电子密度
- 常用的原子建模环境软件——ASE和pymatgen
- 常用的第一性原理计算软件——VASP和GPAW

📖 [查看详细内容](./Part1-Basics/01-Theory/02-First-Principles-Introduction.md)

#### 2. 实操内容 | Practical Content

##### (1) Linux系统的常用命令和超算服务器的使用
- 命令行终端软件——iTerm和Xshell
- ls/ll/cd/cp/mv/cat/pwd/less/tail/mkdir/touch等命令行操作
- vim文本编辑

📂 [教程](./Part1-Basics/02-Practice/01-Linux-Basics/) | 💻 [练习](./Part1-Basics/02-Practice/01-Linux-Basics/exercises/)

##### (2) Python编程语言基础和集成开发环境(IDE)的介绍
- 数据类型、函数、类和对象、模块
- PyCharm软件的使用和常见用法

📂 [教程](./Part1-Basics/02-Practice/02-Python-Basics/) | 💻 [示例代码](./Part1-Basics/02-Practice/02-Python-Basics/examples/)

##### (3) Python环境管理软件Anaconda的使用
- 使用Conda命令创建环境、安装Python库
- 使用Conda命令管理环境和环境的回溯
- PyTorch的安装和调用GPU训练模型

📂 [教程](./Part1-Basics/02-Practice/03-Anaconda/) | 💻 [示例](./Part1-Basics/02-Practice/03-Anaconda/examples/)

##### (4) 原子建模环境软件ASE的使用
- 使用ASE对体系结构进行建模，得到cif文件
- ASE和GPAW软件结合使用

📂 [教程](./Part1-Basics/02-Practice/04-ASE-Modeling/) | 💻 [示例](./Part1-Basics/02-Practice/04-ASE-Modeling/examples/)

##### (5) 第一性原理计算软件GPAW的使用
- 第一性原理计算软件的参数设置和结果收敛性检查
- 以晶体材料为例，使用GPAW进行第一性原理计算
- 体系能量、原子受力和极化等性质的计算

📂 [教程](./Part1-Basics/02-Practice/05-GPAW-Calculations/) | 💻 [示例](./Part1-Basics/02-Practice/05-GPAW-Calculations/examples/)

### 第二部分：前沿研究论文复现
### Part 2: Cutting-Edge Research Paper Reproduction

#### Allegro Nature论文复现 | Allegro Nature Paper Reproduction

复现Nature Communications 2023年发表的Allegro论文成果，该论文提出了一种高精度、高效率的深度学习原子间势能模型。

Reproduce the results from the Allegro paper published in Nature Communications (2023), which introduces a highly accurate and efficient deep learning interatomic potential.

**论文信息 | Paper Information:**
- **标题 | Title:** Learning local equivariant representations for large-scale atomistic dynamics
- **作者 | Authors:** Albert Musaelian, Simon Batzner, Anders Johansson, et al.
- **期刊 | Journal:** Nature Communications 14, 579 (2023)
- **链接 | Link:** [https://www.nature.com/articles/s41467-023-36329-y](https://www.nature.com/articles/s41467-023-36329-y)

**复现目标 | Reproduction Targets:**
- QM9数据集基准测试 (4-5 meV MAE) | QM9 benchmark (4-5 meV MAE)
- revMD17数据集基准测试 (能量: 3.84 meV, 力: 12.98 meV/Å) | revMD17 benchmark (Energy: 3.84 meV, Force: 12.98 meV/Å)
- 大规模分子动力学模拟演示 | Large-scale MD simulation demonstrations

📂 [完整文档](./Part2-Allegro-Reproduction/) | 🔬 [复现指南](./Part2-Allegro-Reproduction/docs/reproduction-guide.md) | 📊 [论文总结](./Part2-Allegro-Reproduction/docs/paper-summary.md)

## 环境要求 | Environment Requirements

### 基础环境 | Basic Environment
- **操作系统**: Linux / macOS / Windows (WSL)
- **Python版本**: 3.8 或更高
- **内存**: 建议 8GB 以上
- **存储**: 建议 20GB 可用空间

### 必需软件包 | Required Packages
```bash
# 创建conda环境
conda create -n first-principles python=3.9

# 激活环境
conda activate first-principles

# 安装基础包
conda install numpy scipy matplotlib pandas jupyter

# 安装ASE
pip install ase

# 安装GPAW (需要编译，或使用conda)
conda install -c conda-forge gpaw

# 安装PyTorch (根据你的CUDA版本)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他工具
pip install pymatgen
```

## 快速开始 | Quick Start

1. **克隆仓库**
```bash
git clone https://github.com/yourusername/First-principles-of-machine-learning.git
cd First-principles-of-machine-learning
```

2. **设置环境**
```bash
conda env create -f environment.yml
conda activate first-principles
```

3. **开始学习**
```bash
# 启动Jupyter Notebook
jupyter notebook

# 或使用Jupyter Lab
jupyter lab
```

## 学习路径建议 | Recommended Learning Path

### 初学者 | Beginners
1. 从Linux基础开始 → Python基础 → Anaconda环境管理
2. 学习ASE基础建模
3. 了解第一性原理理论基础
4. 进行简单的GPAW计算

### 有经验者 | Experienced Users
1. 复习第一性原理理论
2. 深入学习ASE高级功能
3. 掌握GPAW参数优化
4. 探索机器学习与第一性原理的结合

## 教学资源 | Teaching Resources

- 📚 **理论文档**: 详细的理论推导和说明
- 💻 **代码示例**: 每个主题都有完整的代码示例
- 📓 **Jupyter Notebooks**: 交互式学习材料
- 🎯 **练习题**: 巩固知识的实践练习
- 📊 **案例研究**: 真实材料计算案例

## 参考文献 | References

### 书籍 | Books
1. Martin, R. M. (2004). *Electronic Structure: Basic Theory and Practical Methods*
2. Sholl, D., & Steckel, J. A. (2009). *Density Functional Theory: A Practical Introduction*
3. Kohanoff, J. (2006). *Electronic Structure Calculations for Solids and Molecules*

### 在线资源 | Online Resources
- [ASE Documentation](https://wiki.fysik.dtu.dk/ase/)
- [GPAW Documentation](https://wiki.fysik.dtu.dk/gpaw/)
- [VASP Wiki](https://www.vasp.at/wiki/)
- [Materials Project](https://materialsproject.org/)

## 贡献指南 | Contributing

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 许可证 | License

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 联系方式 | Contact

如有问题或建议，请提交 Issue 或 Pull Request。

For questions or suggestions, please submit an Issue or Pull Request.

---

**祝学习愉快！Happy Learning! 🚀**
