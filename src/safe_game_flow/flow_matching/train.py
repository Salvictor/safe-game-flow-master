"""
Flow Matching 训练模块

该模块实现了基于 Flow Matching 的轨迹生成模型的训练和采样功能。

Flow Matching 是一种连续归一化流方法，通过学习从简单分布（如高斯分布）到复杂数据分布
的连续变换来生成样本。本模块支持任意固定通道数的轨迹数据，例如 2D 或 3D 轨迹。

主要功能:
    - fm_loss: Flow Matching 损失函数
    - sample_ode: 使用训练好的模型采样新轨迹
    - train: 完整的训练流程
    - main: 命令行入口

使用示例:
    命令行训练:
        $ Mtrain --data datasets/trajectories.npy --epochs 100
    
    Python API:
        >>> from safe_game_flow.flow_matching.train import train
        >>> args = parse_args()
        >>> train(args)

参考文献:
    Flow Matching for Generative Modeling
    https://arxiv.org/abs/2210.02747
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from safe_game_flow.flow_matching.dataloader import TrajDataset
from safe_game_flow.flow_matching.model import FlowMatching1D, model_config_from_checkpoint
from safe_game_flow.flow_matching.normalization import denormalize_trajectory


# ----------------------------
# Flow Matching 训练/采样
# ----------------------------
def fm_loss(
    model: nn.Module,
    x1: torch.Tensor,
    rng: torch.Generator,
    condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    计算 Flow Matching 损失函数
    
    使用线性概率路径的 Flow Matching 损失：
        - 概率路径: x_t = (1-t) * x0 + t * x1
        - 其中 x0 ~ N(0, I) (标准高斯噪声)
        - t ~ U(0,1) (均匀采样时间)
        - 目标速度场: u_t = d/dt x_t = x1 - x0
    
    损失函数:
        Loss = E[ || v_theta(x_t, t) - (x1 - x0) ||^2 ]
    
    Args:
        model: Flow Matching 模型，输入 (x_t, t)，输出预测的速度场 v_theta
        x1: 真实数据样本，形状 (B, C, H)
            - B: batch size
            - C: 通道数（二维轨迹 C=2，三维轨迹 C=3）
            - H: 序列长度（时间步数）
        rng: PyTorch 随机数生成器，用于可重复的随机采样
        
    Returns:
        标量损失值
        
    Examples:
        >>> model = FlowMatching1D(in_channels=2, hidden_channels=256, num_blocks=6)
        >>> x1 = torch.randn(32, 2, 100)  # batch=32, 2D trajectories, length=100
        >>> rng = torch.Generator().manual_seed(42)
        >>> loss = fm_loss(model, x1, rng)
        >>> loss.backward()
    
    Note:
        - 为了数值稳定性，时间 t 被限制在 [eps, 1-eps] 范围内
        - 使用 MSE 损失来度量预测速度场和真实速度场的差异
    """
    device = x1.device
    B, C, H = x1.shape
    x0 = torch.randn(B, C, H, device=device, generator=rng)
    t = torch.rand(B, device=device, generator=rng)  # [0,1)
    # 可选：避免端点
    eps = 1e-3
    t = t.clamp(eps, 1.0 - eps)

    t_bc = t.view(B, 1, 1)
    x_t = (1.0 - t_bc) * x0 + t_bc * x1
    u_t = x1 - x0  # 因为 phi(t)=t, phi'(t)=1

    v = model(x_t, t, condition)  # (B,C,H)
    if v.shape != u_t.shape:
        raise ValueError(
            f"Model output shape {tuple(v.shape)} does not match target {tuple(u_t.shape)}"
        )
    loss = F.mse_loss(v, u_t)
    return loss


@torch.no_grad()
def sample_ode(
    model: nn.Module,
    num_samples: int,
    H: int,
    ode_steps: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    channels: int | None = None,
    generator: torch.Generator | None = None,
    condition: torch.Tensor | None = None,
    velocity_projector: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]
    | None = None,
    state_projector: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    使用 Heun 方法从训练好的模型采样新轨迹
    
    该函数实现了从 t=0 到 t=1 的 ODE 积分过程：
        dx/dt = v_theta(x, t),  初始条件: x(0) ~ N(0, I)
    
    使用 Heun 方法（二阶 Runge-Kutta 方法，也称为改进的 Euler 方法）
    进行数值积分，比标准 Euler 方法更精确。
    
    Heun 方法步骤:
        1. Euler 预测: x_pred = x + v(x, t) * dt
        2. 计算预测点的速度: v_pred = v(x_pred, t+dt)
        3. Heun 修正: x_new = x + 0.5 * (v + v_pred) * dt
    
    Args:
        model: 训练好的 Flow Matching 模型
        num_samples: 要生成的样本数量
        H: 序列长度（轨迹的时间步数）
        ode_steps: ODE 求解的步数，越大越精确但速度越慢
            - 推荐值: 100-200 for good quality
            - 50 for fast preview
        device: 计算设备 (cpu 或 cuda)
        dtype: 数据类型，默认 torch.float32
        channels: 轨迹通道数。默认从模型配置推断
        generator: 可选随机数生成器，用于可重复采样
        
    Returns:
        生成的标准化轨迹样本，形状 (num_samples, C, H)
        
    Examples:
        >>> model = FlowMatching1D(in_channels=2, hidden_channels=256, num_blocks=6)
        >>> model.eval()
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> samples = sample_ode(model, num_samples=16, H=100, ode_steps=200, device=device)
        >>> print(samples.shape)  # torch.Size([16, 2, 100])
    
    Note:
        - 该函数使用 @torch.no_grad() 装饰器，不计算梯度
        - 初始状态从标准高斯分布采样
        - 采样质量与 ode_steps 正相关，但计算时间也会增加
        - 对于归一化的数据，记得在采样后反归一化
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if H < 1:
        raise ValueError("H must be at least 1")
    if ode_steps < 1:
        raise ValueError("ode_steps must be at least 1")
    if channels is None:
        channels = getattr(model, "in_channels", None)
    if channels is None and hasattr(model, "in_proj"):
        channels = model.in_proj.in_channels
    if channels is None or channels < 1:
        raise ValueError("Unable to infer a valid channel count from model")

    condition_dim = int(getattr(model, "condition_dim", 0))
    if condition_dim > 0:
        if condition is None:
            raise ValueError(
                f"Conditional model requires condition with last dimension {condition_dim}"
            )
        condition = torch.as_tensor(condition, device=device, dtype=dtype)
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2 or condition.shape[1] != condition_dim:
            raise ValueError(
                f"Expected condition with shape (B,{condition_dim}), got {tuple(condition.shape)}"
            )
        if condition.shape[0] == 1 and num_samples > 1:
            condition = condition.expand(num_samples, -1)
        elif condition.shape[0] != num_samples:
            raise ValueError(
                f"Condition batch has {condition.shape[0]} rows for {num_samples} samples"
            )
    elif condition is not None:
        raise ValueError("An unconditional model does not accept condition input")

    model.eval()
    if initial_state is None:
        x = torch.randn(
            num_samples,
            channels,
            H,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        x = torch.as_tensor(initial_state, device=device, dtype=dtype)
        expected_shape = (num_samples, channels, H)
        if tuple(x.shape) != expected_shape:
            raise ValueError(
                f"initial_state must have shape {expected_shape}, got {tuple(x.shape)}"
            )
        x = x.clone()
    dt = 1.0 / ode_steps

    for k in range(ode_steps):
        t0 = torch.full((num_samples,), k / ode_steps, device=device, dtype=dtype)
        t1 = torch.full((num_samples,), (k + 1) / ode_steps, device=device, dtype=dtype)

        v0 = model(x, t0, condition)
        if velocity_projector is not None:
            v0 = velocity_projector(x, t0, v0)
        x_pred = x + v0 * dt  # Euler 预测
        v1 = model(x_pred, t1, condition)
        if velocity_projector is not None:
            v1 = velocity_projector(x_pred, t1, v1)
        x = x + 0.5 * (v0 + v1) * dt  # Heun 修正
        if state_projector is not None:
            x = state_projector(x, t1)

    return x


# ----------------------------
# 训练主循环
# ----------------------------
def train(args):
    """
    Flow Matching 模型训练主函数
    
    实现完整的训练流程，包括：
        - 数据加载和预处理（归一化）
        - 模型初始化
        - 训练循环
        - 定期保存检查点
        - 每个 epoch 生成预览样本
    
    训练流程:
        1. 加载并归一化训练数据
        2. 初始化模型、优化器和学习率调度器
        3. 每个 epoch:
            - 在训练集上训练
            - 保存模型检查点
            - 生成预览样本用于可视化
        4. 可选：仅采样模式（--sample_only）
    
    Args:
        args: 命令行参数对象，包含以下主要字段:
            - data: 训练数据路径
            - save_dir: 保存目录
            - epochs: 训练轮数
            - batch_size: 批次大小
            - lr: 学习率
            - hidden: 隐藏层维度
            - blocks: 网络块数
            - ode_steps: ODE 求解步数
            - amp: 是否使用混合精度训练
            - resume: 恢复训练的检查点路径
            - sample_only: 仅采样模式
            
    Examples:
        命令行使用:
            $ Mtrain --data datasets/trajectories.npy --epochs 100
        
        Python API:
            >>> args = parse_args()
            >>> train(args)
    
    文件输出:
        在 save_dir 中会生成以下文件:
            - checkpoint.pt: 模型检查点（每个 epoch 更新）
            - samples_epoch{N}.npy: 第 N 个 epoch 的预览样本
    
    Note:
        - 数据会自动归一化（zero mean, unit std）
        - 支持从检查点恢复训练
        - 支持混合精度训练（--amp）以加速训练
        - 采样时会自动反归一化
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = None
    resume_path = Path(args.resume) if args.resume else None
    if resume_path and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device)

    # 加载数据。恢复/采样时必须使用训练检查点的统计量，避免对新场景重新拟合归一化。
    condition_path = args.conditions or None
    ds = TrajDataset(
        args.data,
        normalize=True,
        condition_path=condition_path,
        trajectory_mean=checkpoint.get("mean") if checkpoint is not None else None,
        trajectory_std=checkpoint.get("std") if checkpoint is not None else None,
        condition_mean=checkpoint.get("condition_mean") if checkpoint is not None else None,
        condition_std=checkpoint.get("condition_std") if checkpoint is not None else None,
    )
    H = ds.H
    channels = ds.channels
    mean = ds.mean.to(device)  # (C,1)
    std = ds.std.to(device)    # (C,1)
    condition_dim = ds.condition_dim
    condition_mean = (
        ds.condition_mean.to(device) if ds.condition_mean is not None else None
    )
    condition_std = ds.condition_std.to(device) if ds.condition_std is not None else None

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # 初始化模型
    if checkpoint is not None:
        model_config = model_config_from_checkpoint(checkpoint)
    else:
        model_config = {
            "in_channels": channels,
            "hidden_channels": args.hidden,
            "num_blocks": args.blocks,
            "time_emb_dim": args.time_emb_dim,
            "kernel_size": args.kernel_size,
            "condition_dim": condition_dim,
        }
    if model_config["in_channels"] != channels:
        raise ValueError(
            "Dataset/checkpoint channel mismatch: "
            f"dataset has {channels}, checkpoint expects {model_config['in_channels']}"
        )
    checkpoint_condition_dim = int(model_config.get("condition_dim", 0))
    if checkpoint_condition_dim != condition_dim:
        raise ValueError(
            "Dataset/checkpoint condition mismatch: "
            f"dataset has D={condition_dim}, checkpoint expects D={checkpoint_condition_dim}. "
            "Pass --conditions for a conditional checkpoint."
        )
    model = FlowMatching1D(**model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 1234)

    start_epoch = 0
    training_history = []
    os.makedirs(args.save_dir, exist_ok=True)

    # 恢复
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        if "opt" in checkpoint and not args.sample_only:
            optimizer.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint.get("epoch", 0)
        training_history = list(checkpoint.get("training_history", []))
        checkpoint_h = int(checkpoint.get("H", H))
        if checkpoint_h != H:
            raise ValueError(
                f"Dataset/checkpoint horizon mismatch: dataset H={H}, checkpoint H={checkpoint_h}"
            )
        mean = torch.as_tensor(checkpoint.get("mean", mean), device=device, dtype=torch.float32)
        std = torch.as_tensor(checkpoint.get("std", std), device=device, dtype=torch.float32)
        expected_stat_shape = (channels, 1)
        if tuple(mean.shape) != expected_stat_shape or tuple(std.shape) != expected_stat_shape:
            raise ValueError(
                "Checkpoint normalization statistics have invalid shape: "
                f"mean={tuple(mean.shape)}, std={tuple(std.shape)}, expected={expected_stat_shape}"
            )
        if condition_dim > 0:
            condition_mean = torch.as_tensor(
                checkpoint.get("condition_mean", condition_mean),
                device=device,
                dtype=torch.float32,
            )
            condition_std = torch.as_tensor(
                checkpoint.get("condition_std", condition_std),
                device=device,
                dtype=torch.float32,
            )
            expected_condition_shape = (condition_dim,)
            if (
                tuple(condition_mean.shape) != expected_condition_shape
                or tuple(condition_std.shape) != expected_condition_shape
            ):
                raise ValueError(
                    "Checkpoint condition statistics have invalid shape: "
                    f"mean={tuple(condition_mean.shape)}, std={tuple(condition_std.shape)}, "
                    f"expected={expected_condition_shape}"
                )
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # 仅采样模式
    if args.sample_only:
        with torch.no_grad():
            sample_rng = torch.Generator(device=device).manual_seed(args.seed + 5678)
            sample_condition = None
            if condition_dim > 0:
                indices = torch.arange(args.num_samples) % len(ds)
                sample_condition = ds.condition_data[indices].to(device)
            x_norm = sample_ode(
                model,
                args.num_samples,
                H,
                args.ode_steps,
                device,
                generator=sample_rng,
                condition=sample_condition,
            )
            x_denorm = denormalize_trajectory(x_norm, mean, std)  # (B,C,H)
            out_path = Path(args.save_dir) / f"samples_{args.num_samples}x{H}.npy"
            np.save(out_path.as_posix(), x_denorm.cpu().numpy())
            print(f"Saved samples to {out_path}")
        return

    # 训练
    log_interval = max(1, len(dl))
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for i, batch in enumerate(dl):
            if condition_dim > 0:
                x1, condition = batch
                condition = condition.to(device, non_blocking=True)
            else:
                x1 = batch
                condition = None
            x1 = x1.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = fm_loss(model, x1, rng, condition)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            epoch_loss_sum += loss.item()
            epoch_batches += 1
            if (i + 1) % log_interval == 0:
                avg_loss = running / log_interval
                print(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"step {i+1}/{len(dl)} | "
                    f"loss {avg_loss:.6f}"
                )
                running = 0.0

        epoch_record = {
            "epoch": epoch + 1,
            "mean_loss": epoch_loss_sum / max(epoch_batches, 1),
            "duration_seconds": time.perf_counter() - epoch_start,
        }
        training_history.append(epoch_record)
        (Path(args.save_dir) / "training_history.json").write_text(
            json.dumps(training_history, indent=2), encoding="utf-8"
        )

        # 每个 epoch 保存一次权重
        ckpt_path = Path(args.save_dir) / "checkpoint.pt"
        torch.save({
            "format_version": 2,
            "model": model.state_dict(),
            "model_config": model.get_config(),
            "opt": optimizer.state_dict(),
            "epoch": epoch + 1,
            "H": H,
            "channels": channels,
            "mean": mean,
            "std": std,
            "condition_dim": condition_dim,
            "condition_mean": condition_mean,
            "condition_std": condition_std,
            "condition_schema": ds.condition_schema,
            "trajectory_representation": getattr(
                args, "trajectory_representation", "position"
            ),
            "training_history": training_history,
        }, ckpt_path.as_posix())
        print(f"Saved checkpoint to {ckpt_path}")

        # 按指定间隔生成少量预览；正式批量实验可禁用以节省训练时间。
        preview_interval = int(getattr(args, "preview_interval", 1))
        if preview_interval > 0 and (
            (epoch + 1) % preview_interval == 0 or epoch + 1 == args.epochs
        ):
            with torch.no_grad():
                preview_count = min(8, args.batch_size)
                preview_condition = (
                    ds.condition_data[:preview_count].to(device) if condition_dim > 0 else None
                )
                x_norm = sample_ode(
                    model,
                    num_samples=preview_count,
                    H=H,
                    ode_steps=args.ode_steps,
                    device=device,
                    condition=preview_condition,
                )
                x_denorm = denormalize_trajectory(x_norm, mean, std)
                sample_path = Path(args.save_dir) / f"samples_epoch{epoch+1}.npy"
                np.save(sample_path.as_posix(), x_denorm.cpu().numpy())
                print(f"Saved preview samples to {sample_path}")


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有命令行参数的对象
        
    参数说明:
        数据相关:
            --data: 训练数据文件路径，必需参数
                   数据格式: .npy 文件，形状 (N, C, H)
            --save_dir: 保存检查点和样本的目录
            
        训练超参数:
            --epochs: 训练轮数
            --batch_size: 批次大小
            --lr: 学习率 (AdamW 优化器)
            --weight_decay: 权重衰减系数
            --grad_clip: 梯度裁剪阈值，0 表示不裁剪
            
        模型架构:
            --hidden: 隐藏层通道数
            --blocks: ResNet 块数量
            --time_emb_dim: 时间嵌入维度
            --kernel_size: 卷积核大小
            
        采样参数:
            --ode_steps: ODE 求解步数（用于生成样本）
            --num_samples: 采样数量（仅采样模式）
            --sample_only: 仅执行采样（需指定 --resume）
            
        系统设置:
            --amp: 启用混合精度训练（需要 CUDA）
            --cpu: 强制使用 CPU
            --seed: 随机种子
            --workers: DataLoader 工作进程数
            --resume: 恢复训练的检查点路径
    """
    p = argparse.ArgumentParser(
        description="Flow Matching 训练 CxH 轨迹生成网络",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  基础训练:
    %(prog)s --data datasets/trajectories.npy --epochs 100
  
  自定义模型和超参数:
    %(prog)s --data datasets/trajectories.npy \\
             --hidden 512 --blocks 8 --lr 1e-4 \\
             --epochs 200 --batch_size 256
  
  启用混合精度训练:
    %(prog)s --data datasets/trajectories.npy --amp
  
  恢复训练:
    %(prog)s --data datasets/trajectories.npy \\
             --resume runs/exp1/checkpoint.pt \\
             --epochs 300
  
  仅采样（从已训练模型）:
    %(prog)s --data datasets/trajectories.npy \\
             --resume runs/exp1/checkpoint.pt \\
             --sample_only --num_samples 100

更多信息请参见文档: docs/training_guide.md
        """,
    )
    p.add_argument("--data", type=str, required=True, help="npy 文件路径，形状 (N,C,H)")
    p.add_argument(
        "--conditions",
        type=str,
        default="",
        help="可选条件文件：形状(N,D)的.npy，或含values/schema的.npz",
    )
    p.add_argument("--save_dir", type=str, default="runs/exp1", help="保存目录")
    p.add_argument("--resume", type=str, default="", help="恢复训练的 checkpoint 路径")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--time_emb_dim", type=int, default=128)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="开启混合精度")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    p.add_argument("--seed", type=int, default=42)

    # 采样相关
    p.add_argument("--sample_only", action="store_true", help="仅执行采样（需 --resume）")
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--ode_steps", type=int, default=200)
    p.add_argument(
        "--preview_interval",
        type=int,
        default=1,
        help="每隔多少轮生成预览；0表示禁用",
    )
    return p.parse_args()


def main():
    """
    命令行入口函数
    
    该函数是 Mtrain 命令的入口点，在 pyproject.toml 中配置:
        [project.scripts]
        Mtrain = "safe_game_flow.flow_matching.train:main"
    
    执行流程:
        1. 解析命令行参数
        2. 调用 train() 函数开始训练
        
    Examples:
        $ Mtrain --data datasets/trajectories.npy --epochs 100
        $ Mtrain --help  # 查看所有可用参数
    """
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
