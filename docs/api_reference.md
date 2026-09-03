# API 参考文档

本文档提供 `safe-game-flow` 包的完整 API 参考。

## 目录

- [flow_matching 模块](#flow_matching-模块)
- [utils 模块](#utils-模块)
- [数据结构](#数据结构)

## flow_matching 模块

### `safe_game_flow.flow_matching.train`

训练和采样相关函数。

#### `fm_loss(model, x1, rng)`

计算 Flow Matching 损失函数。

**参数：**
- `model` (nn.Module): Flow Matching 模型
- `x1` (torch.Tensor): 真实数据样本，形状 (B, C, H)
- `rng` (torch.Generator): 随机数生成器

**返回：**
- `torch.Tensor`: 标量损失值

**示例：**
```python
import torch
from safe_game_flow.flow_matching.model import FlowMatching1D
from safe_game_flow.flow_matching.train import fm_loss

model = FlowMatching1D(in_channels=2, hidden_channels=256, num_blocks=6)
x1 = torch.randn(32, 2, 100)  # batch=32, 2D, length=100
rng = torch.Generator().manual_seed(42)

loss = fm_loss(model, x1, rng)
print(f"损失: {loss.item():.4f}")
```

---

#### `sample_ode(model, num_samples, H, ode_steps, device, dtype=torch.float32)`

使用 Heun 方法从训练好的模型采样新轨迹。

**参数：**
- `model` (nn.Module): 训练好的 Flow Matching 模型
- `num_samples` (int): 要生成的样本数量
- `H` (int): 序列长度（轨迹的时间步数）
- `ode_steps` (int): ODE 求解的步数
- `device` (torch.device): 计算设备
- `dtype` (torch.dtype, 可选): 数据类型，默认 torch.float32

**返回：**
- `torch.Tensor`: 生成的轨迹样本，形状 (num_samples, 2, H)

**示例：**
```python
import torch
from safe_game_flow.flow_matching.model import FlowMatching1D
from safe_game_flow.flow_matching.train import sample_ode

# 加载模型
model = FlowMatching1D(in_channels=2, hidden_channels=256, num_blocks=6)
checkpoint = torch.load('runs/exp1/checkpoint.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

# 采样
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

samples = sample_ode(
    model=model,
    num_samples=16,
    H=100,
    ode_steps=200,
    device=device
)

print(f"生成样本形状: {samples.shape}")  # (16, 2, 100)
```

---

#### `train(args)`

Flow Matching 模型训练主函数。

**参数：**
- `args` (argparse.Namespace): 命令行参数对象

**返回：**
- None

**示例：**
```python
from safe_game_flow.flow_matching.train import train, parse_args
import sys

# 设置命令行参数
sys.argv = [
    'train.py',
    '--data', 'datasets/2Dtrajectories.npy',
    '--epochs', '100',
    '--save_dir', 'runs/my_exp'
]

args = parse_args()
train(args)
```

---

### `safe_game_flow.flow_matching.model`

模型定义。

#### `FlowMatching1D`

Flow Matching 1D 卷积模型。

**参数：**
- `in_channels` (int): 输入通道数，对于 2D 轨迹为 2
- `hidden_channels` (int, 默认 256): 隐藏层通道数
- `num_blocks` (int, 默认 6): ResNet 块数量
- `time_emb_dim` (int, 默认 128): 时间嵌入维度
- `kernel_size` (int, 默认 3): 卷积核大小

**方法：**
- `forward(x, t)`: 前向传播
  - `x` (torch.Tensor): 输入，形状 (B, C, H)
  - `t` (torch.Tensor): 时间，形状 (B,)
  - 返回: (B, C, H) 预测的速度场

**示例：**
```python
import torch
from safe_game_flow.flow_matching.model import FlowMatching1D

# 创建模型
model = FlowMatching1D(
    in_channels=2,
    hidden_channels=256,
    num_blocks=6,
    time_emb_dim=128,
    kernel_size=3
)

# 前向传播
x = torch.randn(32, 2, 100)  # batch, channels, length
t = torch.rand(32)            # time in [0, 1]

v = model(x, t)
print(f"输出形状: {v.shape}")  # (32, 2, 100)
```

---

### `safe_game_flow.flow_matching.dataloader`

数据加载器。

#### `TrajDataset`

轨迹数据集类。

**参数：**
- `path` (str): .npy 文件路径
- `normalize` (bool, 默认 True): 是否归一化数据

**属性：**
- `data` (np.ndarray): 轨迹数据
- `H` (int): 序列长度
- `mean` (torch.Tensor): 数据均值，形状 (2, 1)
- `std` (torch.Tensor): 数据标准差，形状 (2, 1)

**方法：**
- `__len__()`: 返回数据集大小
- `__getitem__(idx)`: 返回第 idx 个样本

**示例：**
```python
from safe_game_flow.flow_matching.dataloader import TrajDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = TrajDataset('datasets/2Dtrajectories.npy', normalize=True)

print(f"数据集大小: {len(dataset)}")
print(f"序列长度: {dataset.H}")
print(f"均值: {dataset.mean.squeeze()}")
print(f"标准差: {dataset.std.squeeze()}")

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    print(f"批次形状: {batch.shape}")  # (32, 2, H)
    break
```

---

## utils 模块

### `safe_game_flow.utils.visualize`

可视化工具。

#### `load_samples(file_path)`

加载样本数据。

**参数：**
- `file_path` (str): .npy 文件路径

**返回：**
- `np.ndarray`: 轨迹数据，形状 (n_samples, 2, n_steps)

**异常：**
- `FileNotFoundError`: 文件不存在
- `ValueError`: 数据格式不正确

**示例：**
```python
from safe_game_flow.utils.visualize import load_samples

data = load_samples('runs/exp1/samples_epoch50.npy')
print(f"数据形状: {data.shape}")
print(f"样本数量: {data.shape[0]}")
print(f"时间步数: {data.shape[2]}")
```

---

#### `plot_single_epoch(data, epoch_num, save_path=None, figsize=(10, 10), show_legend=True)`

绘制单个 epoch 的所有轨迹。

**参数：**
- `data` (np.ndarray): 轨迹数据，形状 (n_samples, 2, n_steps)
- `epoch_num` (int): epoch 编号
- `save_path` (str, 可选): 保存路径，None 表示显示图像
- `figsize` (tuple, 默认 (10, 10)): 图像大小
- `show_legend` (bool, 默认 True): 是否显示图例

**返回：**
- None

**示例：**
```python
from safe_game_flow.utils.visualize import load_samples, plot_single_epoch

data = load_samples('runs/exp1/samples_epoch50.npy')
plot_single_epoch(
    data, 
    epoch_num=50, 
    save_path='epoch50.png',
    figsize=(12, 12),
    show_legend=True
)
```

---

#### `compare_epochs(file_paths, epoch_nums, save_path=None, ncols=3)`

比较多个 epoch 的轨迹。

**参数：**
- `file_paths` (List[str]): 文件路径列表
- `epoch_nums` (List[int]): epoch 编号列表
- `save_path` (str, 可选): 保存路径，None 表示显示图像
- `ncols` (int, 默认 3): 子图列数

**返回：**
- None

**示例：**
```python
from safe_game_flow.utils.visualize import compare_epochs

file_paths = [
    'runs/exp1/samples_epoch1.npy',
    'runs/exp1/samples_epoch10.npy',
    'runs/exp1/samples_epoch50.npy',
    'runs/exp1/samples_epoch100.npy',
]
epoch_nums = [1, 10, 50, 100]

compare_epochs(
    file_paths, 
    epoch_nums, 
    save_path='comparison.png',
    ncols=2  # 每行 2 个子图
)
```

---

#### `create_animation(run_dir, save_path='training_animation.gif', max_epochs=None, fps=5, interval=200)`

创建训练过程的动画。

**参数：**
- `run_dir` (str): 运行目录，包含所有 samples_epoch*.npy 文件
- `save_path` (str, 默认 'training_animation.gif'): 保存动画的路径
- `max_epochs` (int, 可选): 最大 epoch 数，None 表示使用所有
- `fps` (int, 默认 5): 动画帧率
- `interval` (int, 默认 200): 帧间隔（毫秒）

**返回：**
- None

**异常：**
- `ValueError`: 运行目录中没有找到样本文件

**示例：**
```python
from safe_game_flow.utils.visualize import create_animation

# 创建完整动画
create_animation(
    run_dir='runs/exp1',
    save_path='training.gif',
    fps=5
)

# 只使用前 30 个 epochs
create_animation(
    run_dir='runs/exp1',
    save_path='quick_preview.gif',
    max_epochs=30,
    fps=10
)
```

---

## 数据结构

### 轨迹数据格式

所有轨迹数据使用 NumPy 数组，形状为 `(N, C, H)`；二维轨迹 `C=2`，三维轨迹 `C=3`：

- `N`: 样本数量（轨迹数量）
- `2`: 2D 坐标 (x, y)
- `H`: 时间步数（每条轨迹的长度）

**示例：**
```python
import numpy as np

# 创建轨迹数据
N = 100  # 100 条轨迹
H = 100  # 每条 100 个时间步

trajectories = np.random.randn(N, 2, H)

# 访问第一条轨迹
traj_0 = trajectories[0]  # shape: (2, H)
x_coords = trajectories[0, 0, :]  # x 坐标
y_coords = trajectories[0, 1, :]  # y 坐标

# 保存
np.save('my_trajectories.npy', trajectories)
```

### 检查点格式

训练检查点是包含以下键的字典：

```python
checkpoint = {
    'model': model.state_dict(),      # 模型权重
    'opt': optimizer.state_dict(),    # 优化器状态
    'epoch': current_epoch,           # 当前 epoch
    'H': sequence_length,             # 序列长度
    'mean': data_mean,                # 数据均值 (2, 1)
    'std': data_std,                  # 数据标准差 (2, 1)
}
```

**加载检查点：**
```python
import torch

checkpoint = torch.load('runs/exp1/checkpoint.pt')

# 恢复模型
model.load_state_dict(checkpoint['model'])

# 恢复优化器（用于继续训练）
optimizer.load_state_dict(checkpoint['opt'])

# 获取其他信息
start_epoch = checkpoint['epoch']
H = checkpoint['H']
mean = checkpoint['mean']
std = checkpoint['std']
```

---

## 完整示例

### 示例 1: 完整的训练和采样流程

```python
import torch
import numpy as np
from safe_game_flow.flow_matching.model import FlowMatching1D
from safe_game_flow.flow_matching.train import train, sample_ode, parse_args
from safe_game_flow.flow_matching.dataloader import TrajDataset
from safe_game_flow.utils.visualize import plot_single_epoch
import sys

# 1. 准备数据
print("准备数据...")
N, H = 1000, 100
t = np.linspace(0, 2*np.pi, H)
trajectories = np.zeros((N, 2, H))
for i in range(N):
    A = np.random.rand() * 2 + 0.5
    freq = np.random.rand() * 2 + 1
    phase = np.random.rand() * 2 * np.pi
    trajectories[i, 0, :] = t / (2*np.pi)
    trajectories[i, 1, :] = A * np.sin(freq * t + phase)
np.save('temp_data.npy', trajectories)

# 2. 训练模型
print("\n训练模型...")
sys.argv = [
    'train.py',
    '--data', 'temp_data.npy',
    '--save_dir', 'temp_run',
    '--epochs', '50',
    '--batch_size', '64',
    '--hidden', '128',
]
args = parse_args()
train(args)

# 3. 加载训练好的模型
print("\n加载模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowMatching1D(in_channels=2, hidden_channels=128, num_blocks=6)
checkpoint = torch.load('temp_run/checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

# 4. 采样
print("\n采样新轨迹...")
mean = checkpoint['mean'].to(device)
std = checkpoint['std'].to(device)
H = checkpoint['H']

samples_norm = sample_ode(model, num_samples=16, H=H, ode_steps=200, device=device)
samples = samples_norm * std.unsqueeze(0) + mean.unsqueeze(0)

# 5. 可视化
print("\n可视化结果...")
samples_np = samples.cpu().numpy()
plot_single_epoch(samples_np, epoch_num=50, save_path='final_samples.png')

print("\n完成！结果保存在 final_samples.png")
```

### 示例 2: 批量采样和分析

```python
import torch
import numpy as np
from safe_game_flow.flow_matching.model import FlowMatching1D
from safe_game_flow.flow_matching.train import sample_ode
import matplotlib.pyplot as plt

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowMatching1D(in_channels=2, hidden_channels=256, num_blocks=6)
checkpoint = torch.load('runs/exp1/checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

mean = checkpoint['mean'].to(device)
std = checkpoint['std'].to(device)
H = checkpoint['H']

# 生成大量样本
print("生成 1000 条轨迹...")
all_samples = []
for _ in range(10):  # 批量生成，每次 100 条
    samples_norm = sample_ode(model, num_samples=100, H=H, ode_steps=200, device=device)
    samples = samples_norm * std.unsqueeze(0) + mean.unsqueeze(0)
    all_samples.append(samples.cpu().numpy())

all_samples = np.concatenate(all_samples, axis=0)  # (1000, 2, H)

# 分析
print(f"\n生成样本统计:")
print(f"形状: {all_samples.shape}")
print(f"均值: {all_samples.mean():.4f}")
print(f"标准差: {all_samples.std():.4f}")
print(f"最小值: {all_samples.min():.4f}")
print(f"最大值: {all_samples.max():.4f}")

# 计算轨迹长度分布
lengths = []
for i in range(all_samples.shape[0]):
    x, y = all_samples[i, 0, :], all_samples[i, 1, :]
    length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    lengths.append(length)

plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('轨迹长度')
plt.ylabel('频数')
plt.title('生成轨迹的长度分布')
plt.axvline(np.mean(lengths), color='red', linestyle='--', 
            label=f'平均: {np.mean(lengths):.2f}')
plt.legend()
plt.savefig('trajectory_length_distribution.png', dpi=150, bbox_inches='tight')
print("\n长度分布图保存到 trajectory_length_distribution.png")

# 保存所有样本
np.save('generated_samples_1000.npy', all_samples)
print("所有样本保存到 generated_samples_1000.npy")
```

### 示例 3: 自定义训练循环

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from safe_game_flow.flow_matching.model import FlowMatching1D
from safe_game_flow.flow_matching.train import fm_loss, sample_ode
from safe_game_flow.flow_matching.dataloader import TrajDataset

# 设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# 加载数据
dataset = TrajDataset('datasets/2Dtrajectories.npy', normalize=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 创建模型
model = FlowMatching1D(
    in_channels=2,
    hidden_channels=256,
    num_blocks=6
).to(device)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 随机数生成器
rng = torch.Generator(device=device).manual_seed(42)

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_idx, x1 in enumerate(dataloader):
        x1 = x1.to(device)
        
        # 计算损失
        loss = fm_loss(model, x1, rng)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # 更新学习率
    scheduler.step()
    
    # 打印信息
    avg_loss = epoch_loss / len(dataloader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
    
    # 每 10 个 epoch 采样
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            samples = sample_ode(model, num_samples=8, H=dataset.H, 
                                ode_steps=200, device=device)
            samples_denorm = samples * dataset.std.unsqueeze(0).to(device) + \
                           dataset.mean.unsqueeze(0).to(device)
            np.save(f'custom_samples_epoch{epoch+1}.npy', 
                   samples_denorm.cpu().numpy())
        model.train()

# 保存最终模型
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': num_epochs,
    'H': dataset.H,
    'mean': dataset.mean,
    'std': dataset.std,
}, 'custom_checkpoint.pt')

print("\n训练完成！")
```

---

## 最佳实践

### 1. 数据预处理

```python
import numpy as np

# 检查数据
data = np.load('my_data.npy')
assert data.ndim == 3 and data.shape[1] in (2, 3), "数据形状应该是 (N, C, H)"
assert not np.isnan(data).any(), "数据不应包含 NaN"
assert not np.isinf(data).any(), "数据不应包含 Inf"

# 可选：手动归一化
data_mean = data.mean(axis=(0, 2), keepdims=True)
data_std = data.std(axis=(0, 2), keepdims=True)
data_normalized = (data - data_mean) / (data_std + 1e-8)
```

### 2. 模型保存和加载

```python
import torch

# 保存
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    # 保存数据归一化参数很重要！
    'mean': mean,
    'std': std,
    'H': H,
}, 'checkpoint.pt')

# 加载
checkpoint = torch.load('checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint['model'])

# 采样时使用相同的归一化参数
mean = checkpoint['mean']
std = checkpoint['std']
```

### 3. 错误处理

```python
import torch
from pathlib import Path

def safe_load_checkpoint(checkpoint_path, model, device):
    """安全地加载检查点"""
    try:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 检查必需的键
        required_keys = ['model', 'mean', 'std', 'H']
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"检查点缺少必需的键: {key}")
        
        # 加载模型
        model.load_state_dict(checkpoint['model'])
        
        return checkpoint
        
    except Exception as e:
        print(f"加载检查点失败: {e}")
        raise

# 使用
checkpoint = safe_load_checkpoint('runs/exp1/checkpoint.pt', model, device)
```

---

这份文档涵盖了主要的 API。如需更多详情，请查看源代码或提交 Issue。
