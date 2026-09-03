# Safe Game Flow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

基于 Flow Matching 的安全博弈流生成框架，用于生成满足安全约束的轨迹。

## 📋 目录

- [功能特性](#功能特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [模块说明](#模块说明)
- [命令行工具](#命令行工具)
- [示例](#示例)
- [文档](#文档)
- [开发](#开发)

## ✨ 功能特性

- 🎯 **Flow Matching 训练**: 基于连续归一化流的轨迹生成模型
- 🔒 **安全约束**: 支持 Control Barrier Function (CBF) 约束
- 📊 **可视化工具**: 丰富的训练过程和结果可视化功能
- 🚀 **命令行工具**: 简单易用的命令行接口
- 📦 **模块化设计**: 清晰的代码结构，易于扩展

## 🔧 安装

### 从源码安装

```bash
# 克隆仓库
git clone https://gitee.com/karlmaji/safe-game-flow.git
cd safe-game-flow

# 安装包（开发模式）
pip install -e .

# 或安装包及开发依赖
pip install -e ".[dev]"
```

### 依赖要求

- Python >= 3.9
- PyTorch >= 1.9
- NumPy >= 1.20
- Matplotlib >= 3.3

## 🚀 快速开始

### 1. 准备数据

准备轨迹数据，格式为 `.npy` 文件，形状为 `(N, C, H)`：
- `N`: 样本数量
- `C`: 坐标通道数（二维为 2，三维为 3）
- `H`: 时间步数

```python
import numpy as np

# 示例：生成随机轨迹数据
trajectories = np.random.randn(100, 3, 100)  # 三维轨迹示例
np.save('datasets/2Dtrajectories.npy', trajectories)
```

### 2. 训练模型

使用 `Mtrain` 命令训练 Flow Matching 模型：

```bash
Mtrain --data datasets/2Dtrajectories.npy \
       --save_dir runs/exp1 \
       --epochs 100 \
       --batch_size 128 \
       --lr 2e-4
```

### 3. 可视化结果

使用 `Mvisualize` 命令可视化训练结果：

```bash
# 查看最新 epoch 的轨迹
Mvisualize --run-dir runs/exp1

# 对比多个 epochs
Mvisualize --run-dir runs/exp1 --epochs 1 10 50 100 --save comparison.png

# 创建训练过程动画
Mvisualize --run-dir runs/exp1 --animation --save training.gif
```

## 📦 模块说明

### `safe_game_flow.flow_matching`

Flow Matching 训练和采样模块：

- `model.py`: Flow Matching 模型实现
- `dataloader.py`: 数据加载器
- `train.py`: 训练脚本和采样函数

### `safe_game_flow.cbf`

Control Barrier Function 相关实现（开发中）

### `safe_game_flow.sim`

仿真环境（开发中）

### `safe_game_flow.utils`

实用工具模块：

- `visualize.py`: 轨迹可视化工具

## 🛠️ 命令行工具

### Mtrain - 模型训练

训练 Flow Matching 模型生成固定通道数的二维或三维轨迹。

```bash
Mtrain --data <数据文件> [选项]
```

**主要参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data` | str | 必需 | 训练数据文件路径 (.npy) |
| `--save_dir` | str | runs/exp1 | 保存目录 |
| `--epochs` | int | 200 | 训练轮数 |
| `--batch_size` | int | 128 | 批次大小 |
| `--lr` | float | 2e-4 | 学习率 |
| `--hidden` | int | 256 | 隐藏层维度 |
| `--blocks` | int | 6 | 网络块数量 |
| `--ode_steps` | int | 200 | ODE 求解步数 |
| `--amp` | flag | False | 启用混合精度训练 |
| `--resume` | str | - | 恢复训练的检查点路径 |

**仅采样模式：**

```bash
# 从已训练模型采样
Mtrain --data datasets/2Dtrajectories.npy \
       --resume runs/exp1/checkpoint.pt \
       --sample_only \
       --num_samples 100
```

**详细示例：**

```bash
# 基础训练
Mtrain --data datasets/2Dtrajectories.npy --epochs 100

# 自定义参数训练
Mtrain --data datasets/2Dtrajectories.npy \
       --save_dir runs/custom_exp \
       --epochs 300 \
       --batch_size 256 \
       --lr 1e-4 \
       --hidden 512 \
       --blocks 8 \
       --amp

# 恢复训练
Mtrain --data datasets/2Dtrajectories.npy \
       --resume runs/exp1/checkpoint.pt \
       --epochs 300
```

### Mvisualize - 结果可视化

可视化训练过程中生成的轨迹样本。

```bash
Mvisualize [选项]
```

**主要参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--run-dir` | str | runs/exp1 | 运行目录路径 |
| `--file` | str | - | 单个样本文件路径 |
| `--epochs` | int[] | - | 要对比的 epoch 列表 |
| `--animation` | flag | False | 创建训练动画 |
| `--max-epochs` | int | - | 动画最大 epoch 数 |
| `--save` | str | - | 保存文件路径 |
| `--fps` | int | 5 | 动画帧率 |

**详细示例：**

```bash
# 查看单个 epoch 的轨迹
Mvisualize --file runs/exp1/samples_epoch50.npy --save epoch50.png

# 对比训练进度（多个 epochs）
Mvisualize --run-dir runs/exp1 --epochs 1 10 30 50 100 --save progress.png

# 创建完整训练动画
Mvisualize --run-dir runs/exp1 --animation --save training.gif

# 创建前 30 个 epochs 的动画（更快）
Mvisualize --run-dir runs/exp1 --animation --max-epochs 30 --fps 10

# 查看最新 epoch（无参数默认行为）
Mvisualize --run-dir runs/exp1
```

## 📚 示例

### 完整训练流程示例

```bash
# 1. 训练模型
Mtrain --data datasets/2Dtrajectories.npy \
       --save_dir runs/my_experiment \
       --epochs 100 \
       --batch_size 128 \
       --hidden 256 \
       --amp

# 2. 查看训练进度
Mvisualize --run-dir runs/my_experiment --epochs 1 25 50 75 100

# 3. 创建训练动画
Mvisualize --run-dir runs/my_experiment --animation

# 4. 从训练好的模型采样更多轨迹
Mtrain --data datasets/2Dtrajectories.npy \
       --resume runs/my_experiment/checkpoint.pt \
       --sample_only \
       --num_samples 200
```

### Python API 示例

```python
from safe_game_flow.flow_matching.model import FlowMatching1D
from safe_game_flow.flow_matching.train import sample_ode
from safe_game_flow.utils import visualize
import torch

# 加载模型
model = FlowMatching1D(in_channels=2, hidden_channels=256, num_blocks=6)
checkpoint = torch.load('runs/exp1/checkpoint.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

# 生成样本
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
samples = sample_ode(model, num_samples=16, H=100, ode_steps=200, device=device)

# 可视化
import numpy as np
samples_np = samples.cpu().numpy()
visualize.plot_single_epoch(samples_np, epoch_num=0, save_path='generated.png')
```

## 📖 文档

详细文档请参见 [docs/](docs/) 目录：

- [训练指南](docs/training_guide.md) - 详细的训练参数说明和最佳实践
- [可视化指南](docs/visualization_guide.md) - 可视化工具的详细使用方法
- [API 参考](docs/api_reference.md) - Python API 文档

## 👨‍💻 开发

### 开发环境设置

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 安装 pre-commit hooks
pre-commit install

# 运行测试
pytest tests/

# 代码格式化
black src/

# 类型检查
mypy src/
```

### 代码规范

- 使用 Black 进行代码格式化（行长度 100）
- 使用 Ruff 进行代码检查
- 使用 MyPy 进行类型检查
- 遵循 PEP 8 命名规范

### 项目结构

```
safe-game-flow/
├── src/safe_game_flow/       # 源代码
│   ├── flow_matching/         # Flow Matching 模块
│   ├── cbf/                   # CBF 约束模块
│   ├── sim/                   # 仿真模块
│   └── utils/                 # 工具模块
├── datasets/                  # 数据集
├── runs/                      # 实验结果
├── tests/                     # 测试
├── docs/                      # 文档
├── pyproject.toml             # 项目配置
└── README.md                  # 本文件
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎贡献！请随时提交 Issue 或 Pull Request。

## 📧 联系

- 仓库: https://gitee.com/karlmaji/safe-game-flow.git
- 问题反馈: 请使用 GitHub Issues

## 🙏 致谢

本项目使用了以下开源项目：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Matplotlib](https://matplotlib.org/) - 数据可视化
- [NumPy](https://numpy.org/) - 数值计算

## 📝 更新日志

### v0.1.0 (当前版本)

- ✅ Flow Matching 训练框架
- ✅ 2D/3D 固定长度轨迹生成
- ✅ 可视化工具
- ✅ 命令行工具（Mtrain, Mvisualize）
- 🚧 CBF 约束集成（开发中）
- 🚧 仿真环境（开发中）
