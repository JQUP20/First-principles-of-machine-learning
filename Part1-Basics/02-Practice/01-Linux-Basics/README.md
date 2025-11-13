# Linux基础与超算服务器使用 | Linux Basics and HPC Server Usage

## 目录 | Table of Contents

1. [终端软件介绍](#1-终端软件介绍)
2. [基本命令](#2-基本命令)
3. [文本编辑器Vim](#3-文本编辑器vim)
4. [超算服务器使用](#4-超算服务器使用)
5. [实践练习](#5-实践练习)

---

## 1. 终端软件介绍

### 1.1 为什么需要命令行？

在科学计算和第一性原理计算中，我们主要使用Linux系统和超算服务器，这些环境通常没有图形界面，需要通过命令行进行操作。

**优势**：
- 远程访问服务器
- 批量处理文件
- 自动化脚本
- 资源管理
- 高效的文件操作

### 1.2 常用终端软件

#### macOS用户: iTerm2

**官网**：https://iterm2.com/

**特点**：
- 分屏功能
- 搜索功能强大
- 自动补全
- 主题定制

**安装**：
```bash
# 使用Homebrew安装
brew install --cask iterm2
```

**常用快捷键**：
- `Cmd + D`: 垂直分屏
- `Cmd + Shift + D`: 水平分屏
- `Cmd + T`: 新标签页
- `Cmd + F`: 搜索

#### Windows用户: Xshell / Windows Terminal

**Xshell**
- 商业软件（个人版免费）
- 功能强大
- 会话管理方便

**Windows Terminal** (推荐)
- 微软官方
- 开源免费
- 支持多种Shell
- 美观现代

**安装Windows Terminal**：
```powershell
# 从Microsoft Store安装
# 或使用winget
winget install Microsoft.WindowsTerminal
```

#### Linux用户: 系统自带终端

- Gnome Terminal
- Konsole
- Terminator (支持分屏)

---

## 2. 基本命令

### 2.1 导航命令

#### `pwd` - 显示当前目录

```bash
$ pwd
/home/username/projects
```

**含义**：Print Working Directory（打印工作目录）

#### `ls` - 列出文件

```bash
# 基本用法
$ ls
file1.txt  file2.py  folder1

# 详细列表
$ ls -l
total 12
-rw-r--r-- 1 user group 1234 Nov 13 10:00 file1.txt
-rw-r--r-- 1 user group 5678 Nov 13 10:01 file2.py
drwxr-xr-x 2 user group 4096 Nov 13 10:02 folder1

# 显示隐藏文件
$ ls -a
.  ..  .hidden  file1.txt  file2.py  folder1

# 人类可读的文件大小
$ ls -lh
total 12K
-rw-r--r-- 1 user group 1.2K Nov 13 10:00 file1.txt
-rw-r--r-- 1 user group 5.5K Nov 13 10:01 file2.py
drwxr-xr-x 2 user group 4.0K Nov 13 10:02 folder1

# 组合参数
$ ls -lah
```

**`ll` 命令**：通常是 `ls -l` 的别名

```bash
$ ll
# 等同于 ls -l
```

**文件权限说明**：
```
-rw-r--r--
│││││││││└─ 其他用户权限 (r--)
││││││└──── 组权限 (r--)
│││└─────── 所有者权限 (rw-)
│└────────── 文件类型 (-=普通文件, d=目录, l=链接)

r = read (读)
w = write (写)
x = execute (执行)
```

#### `cd` - 切换目录

```bash
# 进入指定目录
$ cd /home/username/projects

# 返回上一级
$ cd ..

# 返回上两级
$ cd ../..

# 返回主目录
$ cd ~
# 或简单地
$ cd

# 返回上一个目录
$ cd -

# 使用Tab键自动补全
$ cd /home/us[Tab]
```

**相对路径 vs 绝对路径**：
```bash
# 绝对路径（从根目录开始）
$ cd /home/username/projects/dft

# 相对路径（从当前目录开始）
$ cd dft
$ cd ../other_project
```

### 2.2 文件操作命令

#### `mkdir` - 创建目录

```bash
# 创建单个目录
$ mkdir new_folder

# 创建多个目录
$ mkdir folder1 folder2 folder3

# 创建嵌套目录（-p参数）
$ mkdir -p projects/dft/calculations

# 创建带权限的目录
$ mkdir -m 755 public_folder
```

#### `touch` - 创建空文件或更新时间戳

```bash
# 创建空文件
$ touch newfile.txt

# 创建多个文件
$ touch file1.txt file2.py file3.sh

# 更新文件时间戳
$ touch existing_file.txt
```

#### `cp` - 复制文件

```bash
# 复制文件
$ cp source.txt destination.txt

# 复制文件到目录
$ cp file.txt /path/to/directory/

# 复制目录（-r递归）
$ cp -r folder1 folder2

# 保留文件属性（-p）
$ cp -p original.txt copy.txt

# 交互式复制（-i覆盖前询问）
$ cp -i source.txt destination.txt

# 复制多个文件
$ cp file1.txt file2.txt file3.txt /destination/
```

**实用示例**：
```bash
# 备份文件
$ cp important.txt important.txt.bak

# 复制并重命名
$ cp POSCAR POSCAR_original
```

#### `mv` - 移动或重命名

```bash
# 重命名文件
$ mv oldname.txt newname.txt

# 移动文件
$ mv file.txt /path/to/directory/

# 移动并重命名
$ mv old.txt /new/path/new.txt

# 移动目录
$ mv old_folder new_location/

# 交互式移动
$ mv -i source.txt destination.txt
```

#### `rm` - 删除文件

```bash
# 删除文件
$ rm file.txt

# 删除多个文件
$ rm file1.txt file2.txt file3.txt

# 删除目录（-r递归）
$ rm -r folder/

# 强制删除（-f无需确认）
$ rm -f file.txt

# 组合使用（危险！）
$ rm -rf folder/

# 交互式删除（-i确认）
$ rm -i important.txt

# 删除匹配的文件
$ rm *.tmp
```

**⚠️ 警告**：
```bash
# 极度危险！不要执行！
$ rm -rf /
$ rm -rf *

# 使用rm前，先用ls确认
$ ls *.tmp
$ rm *.tmp
```

### 2.3 查看文件内容

#### `cat` - 查看整个文件

```bash
# 显示文件内容
$ cat file.txt

# 显示多个文件
$ cat file1.txt file2.txt

# 显示行号
$ cat -n file.txt

# 合并文件
$ cat file1.txt file2.txt > combined.txt
```

#### `less` - 分页查看文件

```bash
# 打开文件
$ less large_file.txt
```

**less中的操作**：
- `空格键`: 下一页
- `b`: 上一页
- `g`: 跳到开头
- `G`: 跳到结尾
- `/pattern`: 向下搜索
- `?pattern`: 向上搜索
- `n`: 下一个匹配
- `q`: 退出

**与more的区别**：less功能更强大，可以向上翻页

#### `head` - 查看文件开头

```bash
# 默认显示前10行
$ head file.txt

# 指定行数
$ head -n 20 file.txt
$ head -20 file.txt  # 简写

# 查看多个文件
$ head *.txt
```

#### `tail` - 查看文件末尾

```bash
# 默认显示后10行
$ tail file.txt

# 指定行数
$ tail -n 20 file.txt

# 实时查看（监控日志）
$ tail -f output.log

# 从第N行开始显示
$ tail -n +50 file.txt
```

**实用示例**：
```bash
# 监控计算任务输出
$ tail -f gpaw_output.txt

# 查看VASP计算能量收敛
$ tail -20 OUTCAR
```

### 2.4 其他常用命令

#### `grep` - 文本搜索

```bash
# 搜索文件中的文本
$ grep "pattern" file.txt

# 忽略大小写
$ grep -i "pattern" file.txt

# 显示行号
$ grep -n "pattern" file.txt

# 递归搜索目录
$ grep -r "pattern" directory/

# 反向匹配（不包含）
$ grep -v "pattern" file.txt

# 统计匹配行数
$ grep -c "pattern" file.txt
```

**实用示例**：
```bash
# 搜索能量
$ grep "energy" OUTCAR

# 搜索所有Python文件中的import
$ grep -r "import numpy" *.py

# 查找包含"error"的行
$ grep -i "error" output.log
```

#### `find` - 查找文件

```bash
# 查找文件名
$ find . -name "*.py"

# 查找目录
$ find . -type d -name "results"

# 查找大于10MB的文件
$ find . -size +10M

# 查找并删除
$ find . -name "*.tmp" -delete

# 查找最近修改的文件
$ find . -mtime -7  # 7天内
```

#### `wc` - 统计

```bash
# 统计行数、单词数、字节数
$ wc file.txt
  100  500 3000 file.txt
  行数 单词 字节

# 只统计行数
$ wc -l file.txt

# 统计多个文件
$ wc -l *.txt
```

#### `chmod` - 修改权限

```bash
# 数字方式
$ chmod 755 script.sh
# 7=rwx, 5=r-x

# 符号方式
$ chmod +x script.sh        # 添加执行权限
$ chmod -w file.txt         # 移除写权限
$ chmod u+x script.sh       # 用户添加执行权限
$ chmod g-w file.txt        # 组移除写权限
$ chmod o+r file.txt        # 其他添加读权限

# 递归修改
$ chmod -R 755 directory/
```

**权限数字对照**：
```
0 = ---
1 = --x
2 = -w-
3 = -wx
4 = r--
5 = r-x
6 = rw-
7 = rwx
```

---

## 3. 文本编辑器Vim

### 3.1 为什么学Vim？

- 服务器上最常见的编辑器
- 无需图形界面
- 强大高效
- 几乎所有Linux系统都预装

### 3.2 Vim模式

Vim有三种主要模式：

1. **普通模式** (Normal Mode)：默认模式，用于导航和操作
2. **插入模式** (Insert Mode)：编辑文本
3. **命令模式** (Command Mode)：执行命令

```
普通模式 --i/a/o--> 插入模式
   ↑                    │
   └────────ESC─────────┘
   │
   :
   ↓
命令模式
```

### 3.3 基本操作

#### 打开文件

```bash
# 打开文件
$ vim filename.txt

# 打开文件并跳到第N行
$ vim +10 filename.txt

# 打开多个文件
$ vim file1.txt file2.txt
```

#### 退出Vim

```
:q          # 退出（未修改）
:q!         # 强制退出（不保存）
:w          # 保存
:wq         # 保存并退出
:x          # 保存并退出（同:wq）
ZZ          # 保存并退出（普通模式）
```

### 3.4 普通模式操作

#### 光标移动

```
h  # 左
j  # 下
k  # 上
l  # 右

w  # 下一个单词开头
b  # 上一个单词开头
e  # 单词结尾

0  # 行首
$  # 行尾
^  # 行首第一个非空字符

gg # 文件开头
G  # 文件结尾
:10 # 跳到第10行
10G # 跳到第10行
```

#### 编辑操作

```
i  # 当前位置插入
a  # 光标后插入
o  # 下一行插入
O  # 上一行插入

x  # 删除字符
dd # 删除行
D  # 删除到行尾
dw # 删除单词

yy # 复制行
yw # 复制单词
p  # 粘贴

u  # 撤销
Ctrl+r  # 重做

r  # 替换字符
cw # 修改单词
```

#### 搜索和替换

```
/pattern    # 向下搜索
?pattern    # 向上搜索
n           # 下一个匹配
N           # 上一个匹配

:s/old/new/      # 替换当前行第一个
:s/old/new/g     # 替换当前行所有
:%s/old/new/g    # 替换全文
:%s/old/new/gc   # 替换全文（确认）
```

### 3.5 实用配置

创建 `~/.vimrc` 文件：

```vim
" 显示行号
set number

" 语法高亮
syntax on

" 自动缩进
set autoindent
set smartindent

" Tab设置
set tabstop=4
set shiftwidth=4
set expandtab

" 搜索高亮
set hlsearch
set incsearch

" 显示匹配括号
set showmatch

" 启用鼠标
set mouse=a

" 显示状态栏
set laststatus=2

" 文件编码
set encoding=utf-8
```

### 3.6 快速入门练习

**第1步**：创建文件
```bash
$ vim practice.txt
```

**第2步**：进入插入模式（按 `i`）

**第3步**：输入文本
```
Hello Vim!
This is my first vim file.
```

**第4步**：返回普通模式（按 `ESC`）

**第5步**：保存并退出（输入 `:wq` 然后回车）

---

## 4. 超算服务器使用

### 4.1 SSH连接

#### 基本连接

```bash
# 连接服务器
$ ssh username@server.address.edu

# 指定端口
$ ssh -p 2222 username@server.address.edu

# 使用密钥
$ ssh -i ~/.ssh/id_rsa username@server.address.edu
```

#### 配置SSH快捷登录

编辑 `~/.ssh/config`:

```bash
Host myserver
    HostName server.address.edu
    User username
    Port 22
    IdentityFile ~/.ssh/id_rsa
```

使用：
```bash
$ ssh myserver
```

#### 生成SSH密钥

```bash
# 生成密钥对
$ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 复制公钥到服务器
$ ssh-copy-id username@server.address.edu
```

### 4.2 文件传输

#### scp - 安全复制

```bash
# 本地到远程
$ scp local_file.txt username@server:/remote/path/

# 远程到本地
$ scp username@server:/remote/file.txt ./

# 复制目录
$ scp -r local_folder username@server:/remote/path/

# 指定端口
$ scp -P 2222 file.txt username@server:/path/
```

#### rsync - 同步工具

```bash
# 同步文件夹（更高效）
$ rsync -avz local_folder/ username@server:/remote/path/

# 显示进度
$ rsync -avz --progress large_file username@server:/path/

# 排除文件
$ rsync -avz --exclude='*.tmp' folder/ username@server:/path/
```

**参数说明**：
- `-a`: 归档模式（保留权限等）
- `-v`: 详细输出
- `-z`: 压缩传输

### 4.3 作业调度系统

超算系统通常使用作业调度系统管理计算资源。

#### SLURM系统

**提交作业**：

创建提交脚本 `job.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

# 加载模块
module load vasp/5.4.4

# 运行计算
mpirun vasp_std
```

**常用命令**：
```bash
# 提交作业
$ sbatch job.sh

# 查看队列
$ squeue

# 查看自己的作业
$ squeue -u username

# 取消作业
$ scancel job_id

# 查看作业详情
$ scontrol show job job_id
```

#### PBS/Torque系统

提交脚本示例：
```bash
#!/bin/bash
#PBS -N my_job
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -q batch

cd $PBS_O_WORKDIR
mpirun vasp_std
```

**常用命令**：
```bash
# 提交作业
$ qsub job.sh

# 查看队列
$ qstat

# 取消作业
$ qdel job_id
```

### 4.4 环境模块系统

#### Module命令

```bash
# 查看可用模块
$ module avail

# 搜索模块
$ module avail vasp

# 加载模块
$ module load vasp/5.4.4

# 查看已加载模块
$ module list

# 卸载模块
$ module unload vasp

# 清除所有模块
$ module purge
```

### 4.5 监控和管理

#### 查看资源使用

```bash
# CPU和内存使用
$ top
$ htop  # 更友好的界面

# 磁盘使用
$ df -h  # 查看磁盘空间
$ du -sh *  # 查看当前目录各文件夹大小
$ du -h --max-depth=1

# 查看进程
$ ps aux | grep username
```

#### 后台运行

```bash
# 后台运行
$ python script.py &

# nohup（断开SSH后继续运行）
$ nohup python script.py > output.log 2>&1 &

# screen会话
$ screen -S mysession
$ screen -r mysession  # 重新连接
$ screen -ls           # 列出会话

# tmux（推荐）
$ tmux new -s mysession
$ tmux attach -t mysession
$ tmux ls
```

---

## 5. 实践练习

### 练习1: 基本操作

```bash
# 1. 创建工作目录
mkdir -p ~/dft_tutorial/day1
cd ~/dft_tutorial/day1

# 2. 创建几个测试文件
touch file1.txt file2.txt file3.txt
echo "Hello DFT" > file1.txt

# 3. 查看文件内容
cat file1.txt

# 4. 复制文件
cp file1.txt file1_backup.txt

# 5. 创建子目录并移动文件
mkdir backup
mv file1_backup.txt backup/

# 6. 列出所有文件
ls -lR
```

### 练习2: Vim编辑

```bash
# 1. 创建Python脚本
vim hello.py

# 2. 在Vim中输入以下内容（按i进入插入模式）:
"""
#!/usr/bin/env python3
print("Hello from Vim!")
"""

# 3. 保存退出（ESC然后:wq）

# 4. 运行脚本
python3 hello.py
```

### 练习3: 文本处理

```bash
# 创建测试数据
cat > data.txt << EOF
Energy: -123.45 eV
Energy: -123.46 eV
Energy: -123.44 eV
Energy: -123.45 eV
Converged: True
EOF

# 提取能量值
grep "Energy" data.txt

# 统计行数
wc -l data.txt

# 查看最后3行
tail -3 data.txt
```

### 练习4: 管道和重定向

```bash
# 创建文件列表
ls -l > file_list.txt

# 追加内容
echo "End of list" >> file_list.txt

# 管道使用
ls -l | grep "\.txt$"

# 统计txt文件数量
ls | grep "\.txt$" | wc -l

# 查找并排序
find . -name "*.txt" | sort
```

---

## 6. 常见问题和技巧

### 6.1 快捷键

```bash
Ctrl + C  # 中断当前命令
Ctrl + Z  # 暂停当前程序
Ctrl + D  # EOF/退出
Ctrl + L  # 清屏（同clear命令）
Ctrl + A  # 光标到行首
Ctrl + E  # 光标到行尾
Ctrl + U  # 删除到行首
Ctrl + K  # 删除到行尾
Ctrl + R  # 搜索历史命令
```

### 6.2 通配符

```bash
*       # 任意字符
?       # 单个字符
[abc]   # a, b, 或 c
[0-9]   # 0到9的数字

# 示例
ls *.txt        # 所有txt文件
ls file?.txt    # file1.txt, file2.txt等
ls [A-Z]*.py    # 大写字母开头的py文件
```

### 6.3 别名设置

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
alias ll='ls -lh'
alias la='ls -lAh'
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
alias ..='cd ..'
alias ...='cd ../..'
```

重新加载配置：
```bash
$ source ~/.bashrc
```

---

## 总结

本节我们学习了：

✅ Linux命令行基础操作
✅ 文件和目录管理
✅ Vim文本编辑器
✅ 超算服务器连接和使用
✅ 作业调度系统

**下一步**：Python编程基础 →

---

## 参考资料

- [Linux命令大全](https://man.linuxde.net/)
- [Vim入门教程](https://www.openvim.com/)
- [SSH教程](https://www.ssh.com/academy/ssh)
- [SLURM文档](https://slurm.schedmd.com/)

**练习答案和更多示例**：查看 `exercises/` 目录 📁
