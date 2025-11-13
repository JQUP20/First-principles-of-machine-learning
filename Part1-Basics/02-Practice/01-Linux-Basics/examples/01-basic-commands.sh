#!/bin/bash
# Linux基本命令示例脚本
# 演示常用的文件和目录操作

echo "=== Linux基本命令演示 ==="
echo ""

# 1. 显示当前目录
echo "1. 当前工作目录:"
pwd
echo ""

# 2. 创建测试目录结构
echo "2. 创建测试目录结构..."
mkdir -p test_dir/subdir1/subsubdir
mkdir -p test_dir/subdir2
echo "目录创建完成！"
echo ""

# 3. 创建测试文件
echo "3. 创建测试文件..."
touch test_dir/file1.txt
touch test_dir/file2.txt
touch test_dir/subdir1/data.txt
echo "测试文件创建完成！"
echo ""

# 4. 写入内容到文件
echo "4. 写入测试数据..."
echo "This is file1" > test_dir/file1.txt
echo "This is file2" > test_dir/file2.txt
echo "Energy: -100.5 eV" > test_dir/subdir1/data.txt
echo "Force: 0.01 eV/Ang" >> test_dir/subdir1/data.txt
echo ""

# 5. 显示目录结构
echo "5. 显示目录树:"
ls -R test_dir/
echo ""

# 6. 查看文件内容
echo "6. 查看文件内容:"
echo "--- file1.txt ---"
cat test_dir/file1.txt
echo ""
echo "--- data.txt ---"
cat test_dir/subdir1/data.txt
echo ""

# 7. 复制文件
echo "7. 复制文件..."
cp test_dir/file1.txt test_dir/file1_backup.txt
echo "已创建备份: file1_backup.txt"
echo ""

# 8. 移动文件
echo "8. 移动文件到subdir2..."
cp test_dir/file2.txt test_dir/subdir2/
echo "file2.txt已复制到subdir2/"
echo ""

# 9. 搜索文件内容
echo "9. 搜索包含'Energy'的行:"
grep "Energy" test_dir/subdir1/data.txt
echo ""

# 10. 统计文件
echo "10. 统计文件数量:"
find test_dir -type f | wc -l
echo ""

# 11. 查看文件详细信息
echo "11. 文件详细信息:"
ls -lh test_dir/
echo ""

# 12. 清理
echo "12. 是否清理测试文件? (y/n)"
read -p "输入选择: " choice
if [ "$choice" = "y" ]; then
    rm -rf test_dir
    echo "清理完成！"
else
    echo "保留测试文件在 test_dir/"
fi

echo ""
echo "=== 演示完成 ==="
