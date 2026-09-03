#!/usr/bin/env python3
"""
可视化训练过程中生成的轨迹样本

该模块提供了多种可视化工具来分析 Flow Matching 训练过程中生成的轨迹样本：
- 单个 epoch 的轨迹可视化
- 多个 epochs 的对比可视化
- 训练过程动画生成

主要功能:
    load_samples: 加载 .npy 格式的轨迹样本
    plot_single_epoch: 绘制单个 epoch 的所有轨迹
    compare_epochs: 对比多个 epochs 的轨迹变化
    create_animation: 创建训练过程的 GIF 动画

示例:
    >>> from safe_game_flow.utils import visualize
    >>> # 加载并可视化单个 epoch
    >>> data = visualize.load_samples('runs/exp1/samples_epoch10.npy')
    >>> visualize.plot_single_epoch(data, epoch_num=10, save_path='epoch10.png')
    
    >>> # 对比多个 epochs
    >>> files = ['samples_epoch1.npy', 'samples_epoch10.npy', 'samples_epoch50.npy']
    >>> visualize.compare_epochs(files, [1, 10, 50], save_path='comparison.png')
    
    >>> # 创建训练动画
    >>> visualize.create_animation('runs/exp1', save_path='training.gif')
"""

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# 设置中文字体支持
try:
    # 尝试使用系统中的中文字体
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    # 如果没有中文字体，使用默认字体
    pass


def load_samples(file_path: str) -> np.ndarray:
    """
    加载样本数据
    
    Args:
        file_path: .npy 文件路径
        
    Returns:
        轨迹数据，形状为 (n_samples, 2, n_steps)
        
    Raises:
        FileNotFoundError: 当文件不存在时
        ValueError: 当数据格式不正确时
    """
    data = np.load(file_path)
    print(f"加载文件: {file_path}")
    print(f"数据形状: {data.shape}")
    
    if data.ndim != 3 or data.shape[1] != 2:
        raise ValueError(
            f"期望数据形状为 (n_samples, 2, n_steps)，但得到 {data.shape}"
        )
    
    return data


def plot_single_epoch(
    data: np.ndarray,
    epoch_num: int,
    save_path: str | None = None,
    figsize: tuple = (10, 10),
    show_legend: bool = True,
) -> None:
    """
    绘制单个 epoch 的所有轨迹
    
    该函数会在同一张图上绘制所有轨迹，用绿色圆点标记起点，
    红色方块标记终点，每条轨迹使用不同颜色。
    
    Args:
        data: 轨迹数据，形状为 (n_samples, 2, n_steps)
        epoch_num: epoch 编号
        save_path: 保存路径，如果为 None 则显示图像
        figsize: 图像大小，默认 (10, 10)
        show_legend: 是否显示图例，默认 True
        
    Examples:
        >>> data = load_samples('samples_epoch10.npy')
        >>> plot_single_epoch(data, 10, save_path='output.png')
    """
    n_samples = data.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制每条轨迹
    for i in range(n_samples):
        x = data[i, 0, :]
        y = data[i, 1, :]

        # 绘制轨迹线
        ax.plot(x, y, "-", alpha=0.6, linewidth=2, label=f"轨迹 {i+1}")

        # 标记起点和终点
        ax.plot(x[0], y[0], "o", markersize=10, color="green", alpha=0.7)
        ax.plot(x[-1], y[-1], "s", markersize=10, color="red", alpha=0.7)

    # 添加起点和终点的图例
    ax.plot([], [], "o", markersize=10, color="green", label="起点")
    ax.plot([], [], "s", markersize=10, color="red", label="终点")

    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_title(f"Epoch {epoch_num} - 生成的轨迹", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    if show_legend:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def compare_epochs(
    file_paths: list[str],
    epoch_nums: list[int],
    save_path: str | None = None,
    ncols: int = 3,
) -> None:
    """
    比较多个 epoch 的轨迹
    
    在同一张图中以子图形式展示多个 epochs 的轨迹，
    便于观察训练过程中轨迹的演化。
    
    Args:
        file_paths: 文件路径列表
        epoch_nums: epoch 编号列表
        save_path: 保存路径，如果为 None 则显示图像
        ncols: 子图列数，默认 3
        
    Examples:
        >>> files = ['samples_epoch1.npy', 'samples_epoch10.npy', 'samples_epoch50.npy']
        >>> epochs = [1, 10, 50]
        >>> compare_epochs(files, epochs, save_path='comparison.png')
    """
    n_epochs = len(file_paths)

    # 计算子图布局
    cols = min(ncols, n_epochs)
    rows = (n_epochs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = [axes] if n_epochs == 1 else axes.flatten()

    for idx, (file_path, epoch_num) in enumerate(zip(file_paths, epoch_nums, strict=False)):
        data = load_samples(file_path)
        ax = axes[idx]

        n_samples = data.shape[0]
        for i in range(n_samples):
            x = data[i, 0, :]
            y = data[i, 1, :]
            ax.plot(x, y, "-", alpha=0.6, linewidth=1.5)
            ax.plot(x[0], y[0], "o", markersize=6, color="green", alpha=0.7)
            ax.plot(x[-1], y[-1], "s", markersize=6, color="red", alpha=0.7)

        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title(f"Epoch {epoch_num}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    # 隐藏多余的子图
    for idx in range(n_epochs, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"对比图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def create_animation(
    run_dir: str,
    save_path: str = "training_animation.gif",
    max_epochs: int | None = None,
    fps: int = 5,
    interval: int = 200,
) -> None:
    """
    创建训练过程的动画
    
    将训练过程中各个 epoch 的轨迹样本制作成 GIF 动画，
    直观展示模型训练的收敛过程。
    
    Args:
        run_dir: 运行目录，包含所有 samples_epoch*.npy 文件
        save_path: 保存动画的路径，默认 'training_animation.gif'
        max_epochs: 最大 epoch 数，None 表示使用所有可用的
        fps: 动画帧率，默认 5
        interval: 帧间隔（毫秒），默认 200
        
    Examples:
        >>> # 创建包含所有 epochs 的动画
        >>> create_animation('runs/exp1', save_path='training.gif')
        
        >>> # 只使用前 20 个 epochs
        >>> create_animation('runs/exp1', max_epochs=20, fps=10)
        
    Raises:
        ValueError: 当运行目录中没有找到样本文件时
    """
    run_path = Path(run_dir)

    # 查找所有样本文件
    sample_files = sorted(
        run_path.glob("samples_epoch*.npy"),
        key=lambda x: int(re.search(r"epoch(\d+)", x.name).group(1)),
    )

    if max_epochs:
        sample_files = sample_files[:max_epochs]

    if not sample_files:
        raise ValueError(f"在 {run_dir} 中未找到样本文件")

    print(f"找到 {len(sample_files)} 个样本文件")

    # 预加载所有数据以确定坐标范围
    all_data = []
    epoch_nums = []
    for file_path in sample_files:
        data = np.load(file_path)
        all_data.append(data)
        epoch_num = int(re.search(r"epoch(\d+)", file_path.name).group(1))
        epoch_nums.append(epoch_num)

    # 计算全局坐标范围
    all_x = np.concatenate([data[:, 0, :].flatten() for data in all_data])
    all_y = np.concatenate([data[:, 1, :].flatten() for data in all_data])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    # 添加边距
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range

    # 创建动画
    fig, ax = plt.subplots(figsize=(10, 10))

    def update(frame_idx):
        """更新动画帧"""
        ax.clear()
        data = all_data[frame_idx]
        epoch_num = epoch_nums[frame_idx]

        n_samples = data.shape[0]
        for i in range(n_samples):
            x = data[i, 0, :]
            y = data[i, 1, :]
            ax.plot(x, y, "-", alpha=0.6, linewidth=2)
            ax.plot(x[0], y[0], "o", markersize=10, color="green", alpha=0.7)
            ax.plot(x[-1], y[-1], "s", markersize=10, color="red", alpha=0.7)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X", fontsize=14)
        ax.set_ylabel("Y", fontsize=14)
        ax.set_title(f"训练过程 - Epoch {epoch_num}", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # 添加进度文本
        ax.text(
            0.02,
            0.98,
            f"Epoch: {epoch_num}/{epoch_nums[-1]}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    anim = FuncAnimation(fig, update, frames=len(all_data), interval=interval, repeat=True)

    # 保存动画
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    print(f"动画已保存到: {save_path}")

    plt.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="可视化训练样本轨迹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看单个 epoch:
  %(prog)s --file runs/exp1/samples_epoch10.npy
  
  # 对比多个 epochs:
  %(prog)s --epochs 1 10 20 50
  
  # 创建训练动画:
  %(prog)s --animation --max-epochs 30
  
  # 指定运行目录:
  %(prog)s --run-dir runs/exp2 --animation
        """,
    )
    parser.add_argument("--file", type=str, help="单个样本文件路径")
    parser.add_argument(
        "--run-dir", type=str, default="runs/exp1", help="运行目录路径 (默认: runs/exp1)"
    )
    parser.add_argument(
        "--epochs", type=int, nargs="+", help="要对比的 epoch 列表，例如: --epochs 1 10 20 50"
    )
    parser.add_argument("--animation", action="store_true", help="创建训练过程动画")
    parser.add_argument("--max-epochs", type=int, help="动画中包含的最大 epoch 数")
    parser.add_argument("--save", type=str, help="保存图像的路径")
    parser.add_argument("--fps", type=int, default=5, help="动画帧率 (默认: 5)")

    return parser.parse_args()


def main():
    """命令行入口函数"""
    args = parse_args()

    if args.file:
        # 单个文件可视化
        data = load_samples(args.file)
        epoch_num = (
            int(re.search(r"epoch(\d+)", args.file).group(1)) if "epoch" in args.file else 0
        )
        plot_single_epoch(data, epoch_num, args.save)

    elif args.animation:
        # 创建动画
        save_path = args.save if args.save else "training_animation.gif"
        create_animation(args.run_dir, save_path, args.max_epochs, fps=args.fps)

    elif args.epochs:
        # 对比多个 epoch
        run_path = Path(args.run_dir)
        file_paths = []
        for epoch in args.epochs:
            file_path = run_path / f"samples_epoch{epoch}.npy"
            if file_path.exists():
                file_paths.append(file_path)
            else:
                print(f"警告: 文件不存在 {file_path}")

        if file_paths:
            compare_epochs(file_paths, args.epochs, args.save)
        else:
            print("错误: 未找到任何有效的样本文件")

    else:
        # 默认: 显示运行目录中的最后一个 epoch
        run_path = Path(args.run_dir)
        sample_files = sorted(
            run_path.glob("samples_epoch*.npy"),
            key=lambda x: int(re.search(r"epoch(\d+)", x.name).group(1)),
        )

        if sample_files:
            latest_file = sample_files[-1]
            data = load_samples(latest_file)
            epoch_num = int(re.search(r"epoch(\d+)", latest_file.name).group(1))
            plot_single_epoch(data, epoch_num, args.save)
        else:
            print(f"在 {args.run_dir} 中未找到样本文件")
            print("\n使用说明:")
            print("  查看单个 epoch:     Mvisualize --file runs/exp1/samples_epoch10.npy")
            print("  对比多个 epochs:    Mvisualize --epochs 1 10 20 50")
            print("  创建训练动画:       Mvisualize --animation")
            print("  指定运行目录:       Mvisualize --run-dir runs/exp2")


if __name__ == "__main__":
    main()

