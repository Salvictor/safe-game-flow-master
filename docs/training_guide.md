# 训练指南

本文档详细介绍如何使用 `Mtrain` 命令训练 Flow Matching 模型。

## 目录

- [快速开始](#快速开始)
- [数据准备](#数据准备)
- [训练参数详解](#训练参数详解)
- [训练技巧](#训练技巧)
- [常见问题](#常见问题)

## 快速开始

### 最简单的训练命令

```bash
Mtrain --data datasets/2Dtrajectories.npy --epochs 100
```

这将使用默认参数训练 100 个 epoch，结果保存在 `runs/exp1` 目录。

### 推荐的训练配置

```bash
Mtrain --data datasets/2Dtrajectories.npy \
       --save_dir runs/my_exp \
       --epochs 200 \
       --batch_size 128 \
       --lr 2e-4 \
       --hidden 256 \
       --blocks 6 \
       --ode_steps 200 \
       --amp
```

## 数据准备

### 数据格式要求

训练数据必须是 `.npy` 文件，形状为 `(N, C, H)`，其中 `C=2` 表示二维、`C=3` 表示三维：

- `N`: 样本数量（轨迹数量）
- `2`: 2D 坐标 (x, y)
- `H`: 时间步数（每条轨迹的长度）

### 数据准备示例

```python
import numpy as np

# 方法1: 从现有轨迹数据准备
trajectories = []  # 收集你的轨迹数据
trajectories = np.array(trajectories)  # shape: (N, 2, H)
np.save('datasets/my_trajectories.npy', trajectories)

# 方法2: 生成示例数据
N = 1000  # 1000 条轨迹
H = 100   # 每条轨迹 100 个时间步

# 生成正弦轨迹作为示例
t = np.linspace(0, 2*np.pi, H)
trajectories = np.zeros((N, 2, H))
for i in range(N):
    # 随机参数
    A = np.random.rand() * 2 + 0.5  # 振幅
    freq = np.random.rand() * 2 + 1  # 频率
    phase = np.random.rand() * 2 * np.pi  # 相位
    
    trajectories[i, 0, :] = t / (2*np.pi)  # x: 归一化时间
    trajectories[i, 1, :] = A * np.sin(freq * t + phase)  # y: 正弦波

np.save('datasets/my_trajectories.npy', trajectories)
```

### 数据验证

加载数据前先检查：

```python
import numpy as np

data = np.load('datasets/my_trajectories.npy')
print(f"数据形状: {data.shape}")
print(f"数据范围: [{data.min():.3f}, {data.max():.3f}]")
print(f"数据均值: {data.mean():.3f}")
print(f"数据标准差: {data.std():.3f}")

# 检查是否有 NaN 或 Inf
assert not np.isnan(data).any(), "数据包含 NaN"
assert not np.isinf(data).any(), "数据包含 Inf"
```

## 训练参数详解

### 数据相关参数

#### `--data` (必需)

训练数据文件路径。

```bash
--data datasets/2Dtrajectories.npy
```

#### `--save_dir` (默认: `runs/exp1`)

保存检查点和样本的目录。

```bash
--save_dir runs/my_experiment
```

目录结构：
```
runs/my_experiment/
├── checkpoint.pt              # 最新的模型检查点
├── samples_epoch1.npy         # 第 1 个 epoch 的样本
├── samples_epoch2.npy         # 第 2 个 epoch 的样本
└── ...
```

### 训练超参数

#### `--epochs` (默认: 200)

训练轮数。

```bash
--epochs 100  # 快速实验
--epochs 300  # 更充分的训练
```

**建议:**
- 小数据集 (N < 1000): 100-200 epochs
- 中等数据集 (1000 ≤ N < 10000): 200-500 epochs
- 大数据集 (N ≥ 10000): 500+ epochs

#### `--batch_size` (默认: 128)

批次大小。

```bash
--batch_size 64   # 小显存
--batch_size 128  # 推荐
--batch_size 256  # 大显存
```

**建议:**
- GPU 显存 < 8GB: 32-64
- GPU 显存 8-16GB: 128-256
- GPU 显存 > 16GB: 256-512

#### `--lr` (默认: 2e-4)

学习率。

```bash
--lr 1e-4  # 保守学习率
--lr 2e-4  # 推荐学习率
--lr 5e-4  # 激进学习率
```

**建议:**
- 从 2e-4 开始
- 如果训练不稳定，降低到 1e-4
- 如果收敛太慢，提高到 5e-4

#### `--weight_decay` (默认: 1e-4)

权重衰减（L2 正则化）。

```bash
--weight_decay 1e-4  # 默认值
--weight_decay 1e-5  # 更弱的正则化
```

#### `--grad_clip` (默认: 1.0)

梯度裁剪阈值。设为 0 禁用。

```bash
--grad_clip 1.0  # 默认值
--grad_clip 0.5  # 更强的裁剪
--grad_clip 0    # 禁用梯度裁剪
```

### 模型架构参数

#### `--hidden` (默认: 256)

隐藏层通道数。

```bash
--hidden 128  # 小模型，快速训练
--hidden 256  # 推荐配置
--hidden 512  # 大模型，更强表达能力
```

**模型大小对比:**
- 128: ~1M 参数
- 256: ~3M 参数
- 512: ~10M 参数

#### `--blocks` (默认: 6)

ResNet 块数量。

```bash
--blocks 4   # 浅层网络
--blocks 6   # 推荐深度
--blocks 8   # 深层网络
```

#### `--time_emb_dim` (默认: 128)

时间嵌入维度。

```bash
--time_emb_dim 128  # 默认值
--time_emb_dim 256  # 更丰富的时间表示
```

#### `--kernel_size` (默认: 3)

卷积核大小。

```bash
--kernel_size 3  # 小感受野
--kernel_size 5  # 大感受野
```

### 采样参数

#### `--ode_steps` (默认: 200)

ODE 求解步数。用于训练期间生成预览样本。

```bash
--ode_steps 50   # 快速预览（质量较低）
--ode_steps 200  # 推荐配置
--ode_steps 500  # 高质量采样（较慢）
```

**采样质量 vs 速度:**
- 50 steps: ~0.1s/样本，适合快速检查
- 200 steps: ~0.5s/样本，推荐用于训练
- 500 steps: ~1s/样本，适合最终评估

### 系统设置

#### `--amp`

启用混合精度训练（需要 CUDA）。

```bash
--amp  # 使用混合精度
```

**优势:**
- 减少显存占用（约 40-50%）
- 加速训练（约 2-3x）
- 通常不影响精度

**建议:** 始终启用（如果使用 GPU）

#### `--cpu`

强制使用 CPU。

```bash
--cpu  # 使用 CPU 训练
```

**注意:** CPU 训练会非常慢，仅用于调试。

#### `--seed` (默认: 42)

随机种子，用于可复现性。

```bash
--seed 42    # 默认种子
--seed 12345  # 自定义种子
```

#### `--workers` (默认: 2)

DataLoader 工作进程数。

```bash
--workers 0   # 单进程（调试用）
--workers 2   # 默认值
--workers 4   # 更快的数据加载
```

### 恢复训练

#### `--resume`

从检查点恢复训练。

```bash
Mtrain --data datasets/2Dtrajectories.npy \
       --resume runs/exp1/checkpoint.pt \
       --epochs 300
```

恢复的内容：
- 模型权重
- 优化器状态
- 当前 epoch 数
- 数据归一化参数

### 仅采样模式

#### `--sample_only`

从已训练模型采样，不进行训练。

```bash
Mtrain --data datasets/2Dtrajectories.npy \
       --resume runs/exp1/checkpoint.pt \
       --sample_only \
       --num_samples 100 \
       --ode_steps 500
```

#### `--num_samples` (默认: 16)

采样数量（仅采样模式）。

```bash
--num_samples 100  # 生成 100 条轨迹
```

## 训练技巧

### 1. 监控训练过程

训练时会显示损失：

```
Epoch 1/100 | step 10/78 | loss 0.523456
Epoch 1/100 | step 20/78 | loss 0.412345
...
```

**正常的训练曲线:**
- 损失快速下降（前 10-20 epochs）
- 损失逐渐平稳（20-50 epochs）
- 损失缓慢优化（50+ epochs）

**异常情况:**
- 损失不下降：学习率太低
- 损失震荡：学习率太高或批次太小
- 损失突然上升：梯度爆炸，增加 grad_clip

### 2. 检查生成样本

每个 epoch 会保存样本到 `runs/exp1/samples_epoch{N}.npy`。

使用可视化工具检查：

```bash
# 查看第 10 个 epoch 的样本
Mvisualize --file runs/exp1/samples_epoch10.npy

# 对比多个 epochs
Mvisualize --run-dir runs/exp1 --epochs 1 10 50 100
```

**好的迹象:**
- 轨迹逐渐变得光滑
- 轨迹形状接近训练数据
- 没有异常的尖刺或不连续

### 3. 超参数调优策略

**阶段 1: 快速实验**
```bash
Mtrain --data datasets/2Dtrajectories.npy \
       --epochs 50 \
       --hidden 128 \
       --blocks 4
```

**阶段 2: 中等模型**
```bash
Mtrain --data datasets/2Dtrajectories.npy \
       --epochs 200 \
       --hidden 256 \
       --blocks 6 \
       --amp
```

**阶段 3: 最优模型**
```bash
Mtrain --data datasets/2Dtrajectories.npy \
       --epochs 500 \
       --hidden 512 \
       --blocks 8 \
       --lr 1e-4 \
       --amp
```

### 4. 数据归一化

模型会自动归一化数据（zero mean, unit std）。

如果你的数据有特殊范围，可以预先归一化：

```python
import numpy as np

data = np.load('datasets/my_data.npy')

# 归一化到 [-1, 1]
data_min = data.min()
data_max = data.max()
data_norm = 2 * (data - data_min) / (data_max - data_min) - 1

np.save('datasets/my_data_normalized.npy', data_norm)
```

### 5. GPU 优化

**使用混合精度训练:**
```bash
Mtrain --data datasets/2Dtrajectories.npy --amp
```

**调整批次大小以充分利用 GPU:**
```bash
# 检查 GPU 利用率
nvidia-smi -l 1

# 如果利用率低，增加批次大小
Mtrain --data datasets/2Dtrajectories.npy --batch_size 256 --amp
```

## 常见问题

### Q1: 训练很慢怎么办？

**解决方案:**
1. 启用混合精度训练 `--amp`
2. 增加批次大小 `--batch_size 256`
3. 减少模型大小 `--hidden 128 --blocks 4`
4. 使用 GPU 而非 CPU

### Q2: GPU 显存不足怎么办？

**解决方案:**
1. 减小批次大小 `--batch_size 64`
2. 减小模型大小 `--hidden 128`
3. 启用混合精度 `--amp`（减少约 50% 显存）

### Q3: 生成的轨迹质量不好怎么办？

**解决方案:**
1. 训练更多 epochs `--epochs 500`
2. 增大模型 `--hidden 512 --blocks 8`
3. 增加 ODE 步数 `--ode_steps 500`
4. 检查训练数据质量

### Q4: 训练不稳定怎么办？

**解决方案:**
1. 降低学习率 `--lr 1e-4`
2. 增加梯度裁剪 `--grad_clip 0.5`
3. 增加批次大小 `--batch_size 256`
4. 增加权重衰减 `--weight_decay 1e-3`

### Q5: 如何恢复中断的训练？

```bash
# 训练会自动保存检查点到 checkpoint.pt
# 使用 --resume 恢复
Mtrain --data datasets/2Dtrajectories.npy \
       --resume runs/exp1/checkpoint.pt \
       --epochs 300  # 新的总 epoch 数
```

### Q6: 如何在多个 GPU 上训练？

目前不支持多 GPU 训练。单 GPU 训练通常足够快。

### Q7: 训练数据需要多少样本？

**建议:**
- 最少: 100 条轨迹（可能欠拟合）
- 推荐: 1000+ 条轨迹
- 理想: 10000+ 条轨迹

### Q8: 如何评估模型质量？

1. **可视化检查:**
   ```bash
   Mvisualize --run-dir runs/exp1 --epochs 1 50 100 200
   ```

2. **定量评估（需自己实现）:**
   - 与真实轨迹的分布距离（Wasserstein distance）
   - 轨迹平滑度
   - 特定任务指标

## 完整示例

### 示例 1: 标准训练流程

```bash
# 1. 准备数据
python prepare_data.py --output datasets/my_trajectories.npy

# 2. 快速实验（检查是否能正常运行）
Mtrain --data datasets/my_trajectories.npy \
       --save_dir runs/quick_test \
       --epochs 10 \
       --hidden 128

# 3. 可视化快速实验结果
Mvisualize --run-dir runs/quick_test --epochs 1 5 10

# 4. 正式训练
Mtrain --data datasets/my_trajectories.npy \
       --save_dir runs/final_model \
       --epochs 300 \
       --batch_size 128 \
       --lr 2e-4 \
       --hidden 256 \
       --blocks 6 \
       --amp

# 5. 可视化训练过程
Mvisualize --run-dir runs/final_model \
           --epochs 1 50 100 150 200 250 300 \
           --save progress.png

# 6. 创建训练动画
Mvisualize --run-dir runs/final_model --animation

# 7. 从最佳模型采样
Mtrain --data datasets/my_trajectories.npy \
       --resume runs/final_model/checkpoint.pt \
       --sample_only \
       --num_samples 1000 \
       --ode_steps 500
```

### 示例 2: 超参数搜索

```bash
# 尝试不同的隐藏层大小
for hidden in 128 256 512; do
    Mtrain --data datasets/my_trajectories.npy \
           --save_dir runs/hidden_$hidden \
           --epochs 200 \
           --hidden $hidden \
           --amp
done

# 对比结果
Mvisualize --file runs/hidden_128/samples_epoch200.npy --save h128.png
Mvisualize --file runs/hidden_256/samples_epoch200.npy --save h256.png
Mvisualize --file runs/hidden_512/samples_epoch200.npy --save h512.png
```

## 下一步

- 阅读 [可视化指南](visualization_guide.md) 了解如何分析结果
- 阅读 [API 参考](api_reference.md) 了解如何在代码中使用模型
