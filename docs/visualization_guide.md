# 可视化指南

本文档详细介绍如何使用 `Mvisualize` 命令可视化训练结果。

## 目录

- [快速开始](#快速开始)
- [命令参数详解](#命令参数详解)
- [使用场景](#使用场景)
- [Python API](#python-api)
- [自定义可视化](#自定义可视化)
- [常见问题](#常见问题)

## 快速开始

### 查看最新结果

```bash
# 显示最新 epoch 的轨迹
Mvisualize --run-dir runs/exp1
```

### 对比训练进度

```bash
# 对比多个 epochs
Mvisualize --run-dir runs/exp1 --epochs 1 10 50 100
```

### 创建训练动画

```bash
# 创建 GIF 动画
Mvisualize --run-dir runs/exp1 --animation
```

## 命令参数详解

### 基本参数

#### `--run-dir` (默认: `runs/exp1`)

运行目录，包含训练输出的样本文件。

```bash
--run-dir runs/exp1
--run-dir runs/my_experiment
```

目录应该包含 `samples_epoch*.npy` 文件。

#### `--save`

保存图像的路径。如果不指定，则显示图像。

```bash
--save output.png       # 保存为 PNG
--save result.pdf       # 保存为 PDF
--save animation.gif    # 保存动画为 GIF
```

### 查看单个文件

#### `--file`

指定单个样本文件。

```bash
Mvisualize --file runs/exp1/samples_epoch50.npy
Mvisualize --file runs/exp1/samples_epoch50.npy --save epoch50.png
```

### 对比多个 Epochs

#### `--epochs`

要对比的 epoch 列表。

```bash
# 对比 4 个 epochs
Mvisualize --run-dir runs/exp1 --epochs 1 10 50 100

# 保存对比图
Mvisualize --run-dir runs/exp1 --epochs 1 10 50 100 --save comparison.png

# 对比更多 epochs
Mvisualize --run-dir runs/exp1 --epochs 1 5 10 20 30 40 50 --save progress.png
```

**显示布局:**
- 自动排列为网格布局（默认每行 3 个子图）
- 每个子图显示一个 epoch 的所有轨迹
- 绿色圆点标记起点，红色方块标记终点

### 创建动画

#### `--animation`

创建训练过程的动画。

```bash
# 创建完整动画
Mvisualize --run-dir runs/exp1 --animation

# 保存到指定路径
Mvisualize --run-dir runs/exp1 --animation --save training.gif
```

#### `--max-epochs`

限制动画中包含的最大 epoch 数。

```bash
# 只使用前 30 个 epochs（更快）
Mvisualize --run-dir runs/exp1 --animation --max-epochs 30

# 前 50 个 epochs
Mvisualize --run-dir runs/exp1 --animation --max-epochs 50 --save quick_preview.gif
```

#### `--fps`

动画帧率（每秒帧数）。

```bash
# 慢速动画（2 fps）
Mvisualize --run-dir runs/exp1 --animation --fps 2

# 快速动画（10 fps）
Mvisualize --run-dir runs/exp1 --animation --fps 10

# 默认速度（5 fps）
Mvisualize --run-dir runs/exp1 --animation --fps 5
```

## 使用场景

### 场景 1: 训练期间快速检查

```bash
# 查看最新结果
Mvisualize --run-dir runs/exp1

# 或查看特定 epoch
Mvisualize --file runs/exp1/samples_epoch10.npy
```

**用途:** 快速检查当前训练进度。

### 场景 2: 训练完成后分析

```bash
# 创建完整的进度对比图
Mvisualize --run-dir runs/exp1 \
           --epochs 1 20 40 60 80 100 \
           --save final_progress.png

# 创建训练动画
Mvisualize --run-dir runs/exp1 --animation --save training.gif
```

**用途:** 分析训练过程，制作展示材料。

### 场景 3: 对比不同实验

```bash
# 实验 1 的最终结果
Mvisualize --file runs/exp1/samples_epoch100.npy --save exp1_final.png

# 实验 2 的最终结果
Mvisualize --file runs/exp2/samples_epoch100.npy --save exp2_final.png

# 实验 3 的最终结果
Mvisualize --file runs/exp3/samples_epoch100.npy --save exp3_final.png
```

**用途:** 对比不同超参数设置的效果。

### 场景 4: 制作论文图表

```bash
# 高质量图像（适合论文）
Mvisualize --run-dir runs/exp1 \
           --epochs 1 25 50 75 100 \
           --save paper_figure.pdf  # PDF 格式，矢量图
```

**用途:** 生成用于论文或报告的高质量图表。

### 场景 5: 快速动画预览

```bash
# 只用前 20 个 epochs，快速生成
Mvisualize --run-dir runs/exp1 \
           --animation \
           --max-epochs 20 \
           --fps 10 \
           --save quick_preview.gif
```

**用途:** 快速检查训练趋势，不需要等待完整动画生成。

## Python API

### 基本使用

```python
from safe_game_flow.utils import visualize
import numpy as np

# 加载样本
data = visualize.load_samples('runs/exp1/samples_epoch50.npy')
print(f"数据形状: {data.shape}")

# 可视化单个 epoch
visualize.plot_single_epoch(
    data, 
    epoch_num=50, 
    save_path='epoch50.png'
)
```

### 对比多个 Epochs

```python
from safe_game_flow.utils import visualize

# 准备文件列表
file_paths = [
    'runs/exp1/samples_epoch1.npy',
    'runs/exp1/samples_epoch10.npy',
    'runs/exp1/samples_epoch50.npy',
    'runs/exp1/samples_epoch100.npy',
]
epoch_nums = [1, 10, 50, 100]

# 创建对比图
visualize.compare_epochs(
    file_paths, 
    epoch_nums, 
    save_path='comparison.png'
)
```

### 创建动画

```python
from safe_game_flow.utils import visualize

# 创建动画
visualize.create_animation(
    run_dir='runs/exp1',
    save_path='training.gif',
    max_epochs=50,  # 可选：限制 epochs 数量
    fps=5           # 帧率
)
```

### 高级用法：批量处理

```python
from safe_game_flow.utils import visualize
from pathlib import Path

# 查找所有实验目录
run_dirs = list(Path('runs').glob('exp*'))

for run_dir in run_dirs:
    print(f"处理 {run_dir}...")
    
    # 为每个实验创建动画
    visualize.create_animation(
        str(run_dir),
        save_path=f'{run_dir}/training_animation.gif',
        max_epochs=30,
        fps=10
    )
```

## 自定义可视化

### 自定义单个 Epoch 的图

```python
from safe_game_flow.utils import visualize
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
data = visualize.load_samples('runs/exp1/samples_epoch50.npy')

# 创建自定义图
fig, ax = plt.subplots(figsize=(12, 12))

n_samples = data.shape[0]
colors = plt.cm.viridis(np.linspace(0, 1, n_samples))

for i in range(n_samples):
    x = data[i, 0, :]
    y = data[i, 1, :]
    
    # 使用渐变颜色显示时间进度
    for j in range(len(x) - 1):
        ax.plot(x[j:j+2], y[j:j+2], 
                color=colors[i], 
                alpha=j/len(x),  # 透明度随时间增加
                linewidth=2)
    
    # 标记起点和终点
    ax.plot(x[0], y[0], 'o', color='green', markersize=12)
    ax.plot(x[-1], y[-1], 's', color='red', markersize=12)

ax.set_xlabel('X', fontsize=16)
ax.set_ylabel('Y', fontsize=16)
ax.set_title('Epoch 50 - 自定义可视化', fontsize=18, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_aspect('equal')

plt.savefig('custom_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
```

### 添加统计信息

```python
from safe_game_flow.utils import visualize
import matplotlib.pyplot as plt
import numpy as np

data = visualize.load_samples('runs/exp1/samples_epoch50.npy')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左图：轨迹可视化
ax1 = axes[0]
for i in range(data.shape[0]):
    x, y = data[i, 0, :], data[i, 1, :]
    ax1.plot(x, y, '-', alpha=0.6, linewidth=2)
    ax1.plot(x[0], y[0], 'o', color='green', markersize=8)
    ax1.plot(x[-1], y[-1], 's', color='red', markersize=8)

ax1.set_title('生成的轨迹', fontsize=14, fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 右图：统计信息
ax2 = axes[1]

# 计算轨迹长度
lengths = []
for i in range(data.shape[0]):
    x, y = data[i, 0, :], data[i, 1, :]
    length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    lengths.append(length)

# 绘制轨迹长度分布
ax2.hist(lengths, bins=20, alpha=0.7, edgecolor='black')
ax2.set_title('轨迹长度分布', fontsize=14, fontweight='bold')
ax2.set_xlabel('轨迹长度')
ax2.set_ylabel('频数')
ax2.axvline(np.mean(lengths), color='red', linestyle='--', 
            label=f'平均值: {np.mean(lengths):.2f}')
ax2.legend()

plt.tight_layout()
plt.savefig('trajectories_with_stats.png', dpi=200, bbox_inches='tight')
plt.close()
```

### 3D 可视化（添加时间维度）

```python
from safe_game_flow.utils import visualize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = visualize.load_samples('runs/exp1/samples_epoch50.npy')

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

n_samples = min(5, data.shape[0])  # 只显示前 5 条轨迹
colors = plt.cm.tab10(range(n_samples))

for i in range(n_samples):
    x = data[i, 0, :]
    y = data[i, 1, :]
    t = np.linspace(0, 1, len(x))  # 时间轴
    
    ax.plot(x, y, t, color=colors[i], linewidth=2, alpha=0.7)
    ax.scatter(x[0], y[0], t[0], color='green', s=100, marker='o')
    ax.scatter(x[-1], y[-1], t[-1], color='red', s=100, marker='s')

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Time', fontsize=12)
ax.set_title('轨迹 3D 可视化（X, Y, Time）', fontsize=14, fontweight='bold')

plt.savefig('trajectories_3d.png', dpi=200, bbox_inches='tight')
plt.close()
```

### 比较真实数据和生成数据

```python
from safe_game_flow.utils import visualize
import matplotlib.pyplot as plt
import numpy as np

# 加载真实数据和生成数据
real_data = np.load('datasets/2Dtrajectories.npy')
gen_data = visualize.load_samples('runs/exp1/samples_epoch100.npy')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 真实数据
ax1 = axes[0]
n_show = min(8, real_data.shape[0])
for i in range(n_show):
    x, y = real_data[i, 0, :], real_data[i, 1, :]
    ax1.plot(x, y, '-', alpha=0.6, linewidth=2)

ax1.set_title('真实数据', fontsize=14, fontweight='bold')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# 生成数据
ax2 = axes[1]
for i in range(gen_data.shape[0]):
    x, y = gen_data[i, 0, :], gen_data[i, 1, :]
    ax2.plot(x, y, '-', alpha=0.6, linewidth=2)

ax2.set_title('生成数据 (Epoch 100)', fontsize=14, fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('real_vs_generated.png', dpi=200, bbox_inches='tight')
plt.close()
```

## 常见问题

### Q1: 图像中的中文显示为方块怎么办？

这是字体问题。可以在代码中设置字体：

```python
import matplotlib.pyplot as plt

# 使用英文标签
plt.rcParams['font.family'] = 'DejaVu Sans'

# 或安装中文字体（Linux）
# sudo apt-get install fonts-wqy-microhei
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
```

### Q2: 如何调整图像大小？

使用 `figsize` 参数：

```python
from safe_game_flow.utils import visualize

data = visualize.load_samples('runs/exp1/samples_epoch50.npy')
visualize.plot_single_epoch(
    data, 
    epoch_num=50, 
    figsize=(14, 14),  # 更大的图
    save_path='large_figure.png'
)
```

### Q3: 动画文件太大怎么办？

**解决方案：**

1. 减少帧数：
```bash
Mvisualize --run-dir runs/exp1 --animation --max-epochs 30
```

2. 降低帧率：
```bash
Mvisualize --run-dir runs/exp1 --animation --fps 3
```

3. 使用视频格式（需要 ffmpeg）：
```python
from matplotlib.animation import FFMpegWriter

# 在 create_animation 函数中
writer = FFMpegWriter(fps=5, bitrate=1800)
anim.save('training.mp4', writer=writer)
```

### Q4: 如何只显示部分轨迹？

修改源代码或直接在 Python 中：

```python
from safe_game_flow.utils import visualize
import matplotlib.pyplot as plt

data = visualize.load_samples('runs/exp1/samples_epoch50.npy')

# 只显示前 3 条轨迹
n_show = 3
fig, ax = plt.subplots(figsize=(10, 10))

for i in range(n_show):
    x, y = data[i, 0, :], data[i, 1, :]
    ax.plot(x, y, '-', alpha=0.7, linewidth=2, label=f'轨迹 {i+1}')
    ax.plot(x[0], y[0], 'o', color='green', markersize=10)
    ax.plot(x[-1], y[-1], 's', color='red', markersize=10)

ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.savefig('selected_trajectories.png', dpi=150, bbox_inches='tight')
```

### Q5: 如何在 Jupyter Notebook 中使用？

```python
%matplotlib inline
from safe_game_flow.utils import visualize

# 加载并显示（不保存）
data = visualize.load_samples('runs/exp1/samples_epoch50.npy')
visualize.plot_single_epoch(data, epoch_num=50, save_path=None)
```

### Q6: 如何批量生成所有 epochs 的图像？

```bash
#!/bin/bash
# 保存为 visualize_all.sh

for i in {1..100}; do
    if [ -f "runs/exp1/samples_epoch$i.npy" ]; then
        Mvisualize --file runs/exp1/samples_epoch$i.npy \
                   --save figures/epoch_$i.png
    fi
done
```

或使用 Python：

```python
from safe_game_flow.utils import visualize
from pathlib import Path

run_dir = Path('runs/exp1')
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

# 查找所有样本文件
sample_files = sorted(run_dir.glob('samples_epoch*.npy'))

for file_path in sample_files:
    epoch_num = int(file_path.stem.split('epoch')[1])
    data = visualize.load_samples(str(file_path))
    
    output_path = output_dir / f'epoch_{epoch_num}.png'
    visualize.plot_single_epoch(
        data, 
        epoch_num=epoch_num, 
        save_path=str(output_path)
    )
    print(f"保存: {output_path}")
```

## 下一步

- 阅读 [训练指南](training_guide.md) 了解如何训练模型
- 阅读 [API 参考](api_reference.md) 了解完整的 API 文档

