# Linux基础练习1：文件和目录操作

## 练习目标
掌握基本的Linux文件和目录操作命令

## 任务列表

### 任务1：创建项目结构
创建以下目录结构：
```
my_dft_project/
├── calculations/
│   ├── bulk/
│   └── surface/
├── scripts/
└── results/
```

**提示**：使用 `mkdir -p` 命令

### 任务2：创建文件
在相应目录中创建以下文件：
- `calculations/bulk/input.txt`
- `calculations/surface/input.txt`
- `scripts/run_calculation.sh`
- `results/README.md`

**提示**：使用 `touch` 命令

### 任务3：写入内容
使用 `echo` 命令将以下内容写入文件：

1. 在 `calculations/bulk/input.txt` 中写入：
```
System = Silicon bulk
ENCUT = 400
```

2. 在 `results/README.md` 中写入：
```
# 计算结果

本目录存储第一性原理计算结果。
```

**提示**：使用 `echo "content" > file` 和 `echo "content" >> file`

### 任务4：复制和移动
1. 复制 `calculations/bulk/input.txt` 到 `calculations/surface/`
2. 创建 `calculations/bulk/input_backup.txt` 备份

**提示**：使用 `cp` 命令

### 任务5：查看和搜索
1. 使用 `ls -R` 查看整个项目结构
2. 使用 `cat` 查看所有input.txt文件
3. 使用 `find` 查找所有.txt文件
4. 使用 `grep` 搜索包含"ENCUT"的文件

### 任务6：文件统计
1. 统计项目中有多少个文件
2. 统计 `input.txt` 文件的行数
3. 查看项目占用的磁盘空间

**提示**：使用 `wc -l`, `du -sh`

## 参考答案

<details>
<summary>点击查看答案</summary>

```bash
# 任务1
mkdir -p my_dft_project/{calculations/{bulk,surface},scripts,results}

# 任务2
cd my_dft_project
touch calculations/bulk/input.txt
touch calculations/surface/input.txt
touch scripts/run_calculation.sh
touch results/README.md

# 任务3
echo "System = Silicon bulk" > calculations/bulk/input.txt
echo "ENCUT = 400" >> calculations/bulk/input.txt
echo "# 计算结果" > results/README.md
echo "" >> results/README.md
echo "本目录存储第一性原理计算结果。" >> results/README.md

# 任务4
cp calculations/bulk/input.txt calculations/surface/
cp calculations/bulk/input.txt calculations/bulk/input_backup.txt

# 任务5
ls -R
cat calculations/bulk/input.txt
cat calculations/surface/input.txt
find . -name "*.txt"
grep -r "ENCUT" .

# 任务6
find . -type f | wc -l
wc -l calculations/bulk/input.txt
du -sh my_dft_project/
```

</details>

## 挑战任务

### 高级任务1：批量创建文件
创建10个编号的输入文件：`input_001.txt`, `input_002.txt`, ..., `input_010.txt`

**提示**：使用for循环
```bash
for i in {001..010}; do
    touch input_$i.txt
done
```

### 高级任务2：查找和处理
1. 找出所有大于1KB的文件
2. 找出最近24小时内修改的文件
3. 统计每个子目录的文件数量

```bash
# 查找大于1KB的文件
find . -size +1k

# 最近24小时修改的文件
find . -mtime 0

# 统计每个目录的文件数
find . -type d -exec sh -c 'echo "{}:"; find "{}" -maxdepth 1 -type f | wc -l' \;
```

## 学习检查清单

- [ ] 能够创建目录结构
- [ ] 能够创建和编辑文件
- [ ] 理解绝对路径和相对路径
- [ ] 掌握文件复制和移动
- [ ] 能够搜索文件和内容
- [ ] 会使用基本的文件统计命令

完成所有任务后，请进入下一个练习！
