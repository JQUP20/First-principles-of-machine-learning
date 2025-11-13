# 贡献指南 | Contributing Guidelines

感谢您对本项目的兴趣！我们欢迎所有形式的贡献。

Thank you for your interest in contributing to this project! We welcome all forms of contributions.

## 如何贡献 | How to Contribute

### 报告问题 | Reporting Issues

如果您发现了bug或有功能建议，请：

If you find a bug or have a feature suggestion, please:

1. 检查是否已有相关issue
2. 创建新的issue，详细描述问题或建议
3. 如果可能，提供复现步骤

1. Check if there's already a related issue
2. Create a new issue with a detailed description
3. If possible, provide steps to reproduce

### 提交代码 | Submitting Code

1. **Fork 本仓库**
   ```bash
   # 克隆您fork的仓库
   git clone https://github.com/your-username/First-principles-of-machine-learning.git
   cd First-principles-of-machine-learning
   ```

2. **创建新分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **进行修改**
   - 遵循项目的代码风格
   - 添加必要的注释
   - 更新相关文档

4. **测试您的修改**
   ```bash
   # 运行测试（如果有）
   pytest tests/

   # 确保示例代码能运行
   python examples/your_example.py
   ```

5. **提交更改**
   ```bash
   git add .
   git commit -m "描述您的更改"
   ```

6. **推送到GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建Pull Request**
   - 在GitHub上创建PR
   - 详细描述您的更改
   - 链接相关的issue

## 代码规范 | Code Style

### Python代码

- 遵循 PEP 8 风格指南
- 使用有意义的变量名
- 添加文档字符串（docstrings）
- 函数应该简短且功能单一

**示例**：
```python
def calculate_energy(atoms, calculator='GPAW'):
    """
    计算原子体系的总能量

    Parameters:
    -----------
    atoms : ase.Atoms
        原子结构对象
    calculator : str
        计算器名称，默认为'GPAW'

    Returns:
    --------
    float
        总能量（eV）
    """
    # 实现代码
    pass
```

### 文档

- 使用清晰的Markdown格式
- 提供代码示例
- 中英文双语（如适用）
- 包含必要的图片和图表

### 提交信息

提交信息应该清晰描述更改内容：

**好的示例**：
```
添加ASE建模教程中的表面构建部分
修复GPAW计算示例中的k点设置错误
更新README中的安装说明
```

**不好的示例**：
```
update
fix bug
修改
```

## 贡献类型 | Types of Contributions

### 📚 教程和文档

- 改进现有教程
- 添加新的示例
- 修正错别字
- 翻译内容

### 💻 代码

- 添加新的示例脚本
- 优化现有代码
- 修复bug
- 添加测试

### 🎨 可视化

- 改进图表质量
- 添加新的可视化示例
- 优化图表样式

### 🐛 Bug修复

- 修复代码错误
- 修正文档错误
- 改进错误处理

## 审查流程 | Review Process

1. 提交PR后，维护者会审查您的代码
2. 可能会要求进行修改
3. 通过审查后，PR会被合并
4. 您的贡献会被记录在贡献者列表中

## 行为准则 | Code of Conduct

- 尊重所有贡献者
- 保持友好和专业
- 接受建设性批评
- 关注项目最佳利益

## 许可 | License

贡献的代码将采用与项目相同的MIT许可证。

By contributing, you agree that your contributions will be licensed under the MIT License.

## 需要帮助？| Need Help?

如果您有任何问题，可以：

If you have any questions, you can:

- 创建issue询问
- 在discussion中讨论
- 查看现有的文档和示例

---

**再次感谢您的贡献！| Thank you again for your contribution!** 🙏
