# 字体数据集生成与Demons插值工具

一个用于生成字体字符图像数据集并进行Demons形变插值的自动化工具。适用于字体风格迁移、字体生成等深度学习研究。

##  目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [目录结构](#目录结构)

---

##  功能特性

- ✅ **自动字体渲染**：批量生成字体字符PNG图像（256×256）
- ✅ **智能对齐**：基于样本的统一居中对齐策略
- ✅ **Demons插值**：自动生成字体间的平滑形变序列（12帧）
- ✅ **增量处理**：自动跳过已完成部分，支持断点续传
- ✅ **完整性验证**：检测并自动修复损坏文件
- ✅ **并行加速**：支持多进程并行处理

---

##  快速开始

### 1. 安装依赖

**Python依赖：**

```bash
pip install Pillow numpy SimpleITK tqdm
```

**系统依赖（Linux）：**

```bash
# Ubuntu/Debian
sudo apt-get install libinsighttoolkit4.13

# CentOS/RHEL  
sudo yum install InsightToolkit-devel
```

### 2. 准备字体文件

将字体文件放入 `font/` 目录（支持 .ttf / .otf / .ttc 格式）：

```
font/
├── 思源黑体.otf
├── 宋体.ttf
├── 楷体.ttf
└── ...
```

### 3. 添加可执行权限（Linux/macOS）

```bash
chmod +x LogDomainDemonsRegistration
```

### 4. 运行

```bash
python run.py
```

1. 生成所有字体的字符图像
2. 以"思源黑体"为源进行Demons插值
3. 将结果保存到 `out/` 目录

---

##  详细使用

### 基础命令

#### **一键生成全部数据**（推荐）

```bash
python run.py
```

自动完成数据集生成 + Demons插值，跳过已完成部分。

---

#### **仅生成数据集**（不做插值）

```bash
python run.py --skip-demons
```

或：

```bash
python dataset_only.py
```

**输出示例：**
```
out/
├── 思源黑体/
│   ├── train/
│   │   ├── 一.png
│   │   ├── 二.png
│   │   └── ... (1000+字符)
│   └── test/
│       └── ... (108字符)
└── 宋体/
    └── ...
```

---

#### **仅做Demons插值**（跳过数据集）

```bash
python run.py --skip-dataset
```

适用于已有数据集，只需要生成插值的情况。

**输出示例：**
```
out/
└── 宋体/
    ├── train/
    │   ├── 一/
    │   │   ├── 000.png  # 接近源字体
    │   │   ├── 001.png
    │   │   ├── ...
    │   │   └── 011.png  # 接近目标字体
    │   ├── 二/
    │   │   └── ...
    │   └── 一.png (原始图像)
    └── test/
        └── ...
```

---

### 进阶参数

#### **处理指定字体**

```bash
# 只处理宋体和楷体
python run.py --fonts 宋体 楷体

# 支持模糊匹配
python run.py --fonts 宋 楷
```

---

#### **强制重新生成**

```bash
# 强制重建数据集
python run.py --force-dataset

# 强制重建插值
python run.py --force-demons

# 全部重建
python run.py --force-dataset --force-demons
```

---

#### **更换源字体**

默认使用"思源黑体"作为插值源，可自定义：

```bash
python run.py --source-font 宋体
```

所有其他字体将以"宋体"为起点进行形变。

---

#### **处理特定子集**

```bash
# 只处理训练集
python run.py --demons-subset train

# 只处理测试集
python run.py --demons-subset test

# 处理全部（默认）
python run.py --demons-subset all
```

---

#### **调整性能**

```bash
# 调整并行进程数（默认8）
python run.py --demons-workers 4   # 低配机器
python run.py --demons-workers 16  # 高配机器
```

---

### 维护与调试

#### **验证数据完整性**

```bash
python run.py --verify
```

检查所有PNG文件是否有效，插值序列是否完整。

**输出示例：**
```
  进入详细验证模式...
  发现损坏文件: 宋体 - train(3), test(1)

  发现需要修复的内容:
  - 需要重新生成数据集的字体: 宋体
  - train集中损坏的字符: 一, 二, 三

  建议运行以下命令修复:
  python run.py --fonts 宋体 --force-dataset
  python run.py --skip-dataset --force-demons
```

---

#### **自动修复**

```bash
python run.py --auto-fix
```

自动检测并重新生成损坏的文件。

---

#### **调试模式**

```bash
# 只处理前10个字符 + 详细日志
python run.py --demons-max-chars 10 --demons-verbose
```

适合测试参数或排查问题。

---

##   目录结构

### 项目结构

```
数据集/
├── run.py                    # 主入口：一键执行
├── dataset_only.py           # 仅生成数据集
├── Demons.py                 # Demons配准核心模块
│
├── scripts/
│   ├── generate_dataset.py       # 数据集生成脚本
│   └── generate_demons_interp.py # Demons插值脚本
│
├── font/                     # 输入：字体文件
│   ├── 思源黑体.otf
│   └── ...
│
├── out/                      # 输出：生成的数据
│   └── <字体名>/
│       ├── train/
│       └── test/
│
├── metadata/                 # 状态文件
│   ├── font_status.json
│   ├── demons_status.json
│   └── missing_chars.csv
│
├── train.txt                 # 训练集字符列表
├── test.txt                  # 测试集字符列表
└── LogDomainDemonsRegistration  # Demons可执行文件
```

---

### 输出数据结构

```
out/
└── 宋体/                      # 目标字体
    ├── train/
    │   ├── 一/                # 插值序列文件夹
    │   │   ├── 000.png       # 第0帧（接近源字体）
    │   │   ├── 001.png       # 第1帧
    │   │   ├── 002.png
    │   │   ├── ...
    │   │   ├── 005.png       # 第5帧（中点）
    │   │   ├── 006.png
    │   │   ├── ...
    │   │   └── 011.png       # 第11帧（接近目标字体）
    │   │
    │   ├── 二/
    │   │   └── 000.png ~ 011.png
    │   │
    │   ├── 一.png             # 原始字符图像（256×256）
    │   ├── 二.png
    │   └── ...
    │
    └── test/
        └── ...
```

**说明：**
- 每个字符有两种形式：
  - **PNG文件**（如 `一.png`）：原始渲染图像
  - **文件夹**（如 `一/`）：包含12帧插值序列

---

##   生成规则

### 字符图像渲染

- **尺寸**：256×256 像素
- **背景**：白色 (255)
- **前景**：黑色 (0)
- **字符占比**：约60%画布
- **对齐方式**：基于样本的统一居中对齐

### 对齐逻辑

1. 从字符列表中随机抽取10个样本
2. 为每个样本计算最优字号（使字符占画布60%）
3. 计算所有样本的平均字号和偏移量
4. 所有字符使用统一参数渲染（保证一致性）

### Demons插值

- **帧数**：12帧
- **算法**：LogDomainDemonsRegistration
- **插值方式**：双向插值（forward + backward）
- **默认源字体**：思源黑体

---

##   使用场景

### 场景1：生成训练数据集

```bash
# 生成全部数据
python run.py

# 数据位于 out/*/train 和 out/*/test
```

---

### 场景2：添加新字体

```bash
# 1. 将新字体放入 font/ 目录
# 2. 只处理新字体
python run.py --fonts 新字体名称
```

---

### 场景3：更换源字体重新插值

```bash
# 使用宋体作为新源
python run.py --source-font 宋体 --skip-dataset --force-demons
```

---

### 场景4：定期维护

```bash
# 每周验证数据完整性
python run.py --verify

# 发现问题自动修复
python run.py --auto-fix
```

---

##   完整参数列表

运行查看所有参数：

```bash
python run.py --help
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source-font` | 源字体名称 | 思源黑体 |
| `--skip-dataset` | 跳过数据集生成 | False |
| `--skip-demons` | 跳过Demons插值 | False |
| `--force-dataset` | 强制重建数据集 | False |
| `--force-demons` | 强制重建插值 | False |
| `--demons-subset` | 处理子集 (train/test/all) | all |
| `--demons-max-chars` | 仅处理前N个字符 | None |
| `--demons-verbose` | 详细日志 | False |
| `--demons-workers` | 并行进程数 | 8 |
| `--fonts` | 仅处理指定字体 | 全部 |
| `--verify` | 验证模式 | False |
| `--auto-fix` | 自动修复 | False |

