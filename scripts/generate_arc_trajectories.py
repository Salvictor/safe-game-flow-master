#!/usr/bin/env python3
"""
生成平滑的弧形轨迹数据

该脚本用于生成从一个点到另一个点的平滑弧形轨迹，轨迹分布呈椭圆形。
生成的轨迹可以用于训练 Flow Matching 模型。

主要功能:
    - 生成平滑的弧形轨迹（使用贝塞尔曲线或椭圆弧）
    - 轨迹分布呈椭圆形（通过控制弧度和高度参数）
    - 支持自定义起点、终点、轨迹数量和点数
    - 自动可视化生成的轨迹
    - 保存为 .npy 格式，兼容现有的训练流程

使用示例:
    $ python scripts/generate_arc_trajectories.py --n_trajectories 100 --visualize
    $ python scripts/generate_arc_trajectories.py --start -50 0 --end 50 0 --n_points 100 --n_trajectories 200
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_arc_trajectory(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    n_points: int,
    arc_height: float,
    direction: int = 1,
) -> np.ndarray:
    """
    生成单条平滑的弧形轨迹
    
    使用椭圆弧参数化方法生成从起点到终点的平滑轨迹。
    轨迹为椭圆的一部分，通过控制弧高来调整曲率。
    
    Args:
        start_point: 起点坐标 (x0, y0)
        end_point: 终点坐标 (x1, y1)
        n_points: 轨迹点数
        arc_height: 弧的最大高度（相对于连线的垂直距离）
        direction: 弧的方向，1 表示向上，-1 表示向下
        
    Returns:
        轨迹数组，形状为 (2, n_points)，第一行为 x 坐标，第二行为 y 坐标
    """
    x0, y0 = start_point
    x1, y1 = end_point
    
    # 计算起点和终点之间的距离和角度
    dx = x1 - x0
    dy = y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # 在局部坐标系中生成弧线（水平方向）
    t = np.linspace(0, 1, n_points)
    
    # 使用抛物线生成弧形
    local_x = t * dist
    local_y = direction * 4 * arc_height * t * (1 - t)  # 抛物线公式
    
    # 旋转到实际坐标系
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    x = x0 + local_x * cos_a - local_y * sin_a
    y = y0 + local_x * sin_a + local_y * cos_a
    
    trajectory = np.stack([x, y], axis=0)  # shape: (2, n_points)
    
    return trajectory


def generate_elliptical_arc_distribution(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    n_trajectories: int,
    n_points: int,
    height_range: tuple[float, float] = (0, 40),
    both_directions: bool = True,
) -> np.ndarray:
    """
    生成呈椭圆分布的多条弧形轨迹
    
    生成多条从起点到终点的弧形轨迹，这些轨迹的高度在指定范围内均匀分布，
    形成填满整个椭圆区域的分布。包括接近直线的轨迹和大弧度的轨迹。
    
    Args:
        start_point: 起点坐标 (x0, y0)
        end_point: 终点坐标 (x1, y1)
        n_trajectories: 要生成的轨迹数量
        n_points: 每条轨迹的点数
        height_range: 弧高度范围 (min_height, max_height)，min_height=0 表示包含直线
        both_directions: 是否生成双向弧（上下都有），默认为 True
        
    Returns:
        轨迹数组，形状为 (n_trajectories, 2, n_points)
    """
    trajectories = []
    
    min_height, max_height = height_range
    
    if both_directions:
        # 生成上下两个方向的弧，均匀填充整个椭圆区域
        n_upper = n_trajectories // 2
        n_lower = n_trajectories - n_upper
        
        # 上方的弧：从接近直线（高度接近0）到最大弧度
        for i in range(n_upper):
            # 均匀分布高度，从 min_height 到 max_height
            height = min_height + (max_height - min_height) * (i / max(n_upper - 1, 1))
            
            trajectory = generate_arc_trajectory(
                start_point, end_point, n_points, height, direction=1
            )
            trajectories.append(trajectory)
        
        # 下方的弧：从接近直线到最大弧度
        for i in range(n_lower):
            height = min_height + (max_height - min_height) * (i / max(n_lower - 1, 1))
            
            trajectory = generate_arc_trajectory(
                start_point, end_point, n_points, height, direction=-1
            )
            trajectories.append(trajectory)
    else:
        # 只生成单方向的弧，均匀分布
        for i in range(n_trajectories):
            height = min_height + (max_height - min_height) * (i / max(n_trajectories - 1, 1))
            
            trajectory = generate_arc_trajectory(
                start_point, end_point, n_points, height, direction=1
            )
            trajectories.append(trajectory)
    
    trajectories = np.stack(trajectories, axis=0)  # shape: (n_trajectories, 2, n_points)
    
    return trajectories


def visualize_trajectories(
    trajectories: np.ndarray,
    save_path: str | None = None,
    title: str = "生成的弧形轨迹",
    show: bool = True,
) -> None:
    """
    可视化生成的轨迹
    
    Args:
        trajectories: 轨迹数据，形状为 (n_trajectories, 2, n_points)
        save_path: 保存路径，如果为 None 则不保存
        title: 图像标题
        show: 是否显示图像
    """
    n_trajectories = trajectories.shape[0]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用渐变颜色
    colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
    
    for i in range(n_trajectories):
        x = trajectories[i, 0, :]
        y = trajectories[i, 1, :]
        
        ax.plot(x, y, "-", alpha=0.6, linewidth=1.5, color=colors[i])
        
        # 标记起点和终点
        if i == 0:
            ax.plot(x[0], y[0], "o", markersize=12, color="green", 
                   alpha=0.8, label="起点", zorder=10)
            ax.plot(x[-1], y[-1], "s", markersize=12, color="red", 
                   alpha=0.8, label="终点", zorder=10)
        else:
            ax.plot(x[0], y[0], "o", markersize=8, color="green", alpha=0.5)
            ax.plot(x[-1], y[-1], "s", markersize=8, color="red", alpha=0.5)
    
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=12, loc="best")
    ax.set_aspect("equal", adjustable="box")
    
    # 添加统计信息
    info_text = f"轨迹数量: {n_trajectories}\n每条轨迹点数: {trajectories.shape[2]}"
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ 可视化图像已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成平滑的弧形轨迹数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成默认配置的轨迹（100条，从 (-50,0) 到 (50,0)）
  %(prog)s
  
  # 生成 200 条轨迹并可视化
  %(prog)s --n_trajectories 200 --visualize
  
  # 自定义起点和终点
  %(prog)s --start -100 0 --end 100 0 --n_trajectories 150
  
  # 调整弧高度范围（从直线到大弧度）
  %(prog)s --height_range 0 50 --n_trajectories 100
  
  # 调整弧高度范围（只有弧度较大的轨迹）
  %(prog)s --height_range 20 50 --n_trajectories 100
  
  # 只生成单向弧（上方）
  %(prog)s --no_both_directions --n_trajectories 50
        """,
    )
    
    parser.add_argument(
        "--start", type=float, nargs=2, default=[-50, 0],
        help="起点坐标 (x y)，默认: -50 0",
    )
    parser.add_argument(
        "--end", type=float, nargs=2, default=[50, 0],
        help="终点坐标 (x y)，默认: 50 0",
    )
    parser.add_argument(
        "--n_trajectories", type=int, default=100,
        help="要生成的轨迹数量，默认: 100",
    )
    parser.add_argument(
        "--n_points", type=int, default=100,
        help="每条轨迹的点数，默认: 100",
    )
    parser.add_argument(
        "--height_range", type=float, nargs=2, default=[0, 10],
        help="弧高度范围 (min max)，min=0 表示包含直线，默认: 0 10",
    )
    parser.add_argument(
        "--no_both_directions", action="store_true",
        help="只生成单向弧（默认生成上下双向）",
    )
    parser.add_argument(
        "--output", type=str, default="datasets/arc_trajectories.npy",
        help="输出文件路径，默认: datasets/arc_trajectories.npy",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="生成可视化图像",
    )
    parser.add_argument(
        "--viz_output", type=str, default="datasets/arc_trajectories_viz.png",
        help="可视化图像保存路径，默认: datasets/arc_trajectories_viz.png",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子，用于可重复性",
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"✓ 随机种子设置为: {args.seed}")
    
    # 生成轨迹
    print("\n" + "="*60)
    print("生成弧形轨迹数据")
    print("="*60)
    print(f"起点: {tuple(args.start)}")
    print(f"终点: {tuple(args.end)}")
    print(f"轨迹数量: {args.n_trajectories}")
    print(f"每条轨迹点数: {args.n_points}")
    print(f"弧高度范围: {tuple(args.height_range)}")
    print(f"双向弧: {'是' if not args.no_both_directions else '否'}")
    print("-"*60)
    
    trajectories = generate_elliptical_arc_distribution(
        start_point=tuple(args.start),
        end_point=tuple(args.end),
        n_trajectories=args.n_trajectories,
        n_points=args.n_points,
        height_range=tuple(args.height_range),
        both_directions=not args.no_both_directions,
    )
    
    print(f"✓ 成功生成轨迹数据，形状: {trajectories.shape}")
    print(f"  - 数据范围:")
    print(f"    X: [{trajectories[:, 0, :].min():.2f}, {trajectories[:, 0, :].max():.2f}]")
    print(f"    Y: [{trajectories[:, 1, :].min():.2f}, {trajectories[:, 1, :].max():.2f}]")
    
    # 保存数据
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, trajectories)
    print(f"✓ 数据已保存到: {output_path}")
    print(f"  文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    
    # 可视化
    if args.visualize:
        print("-"*60)
        print("生成可视化图像...")
        viz_path = Path(args.viz_output)
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        
        visualize_trajectories(
            trajectories,
            save_path=str(viz_path),
            title=f"弧形轨迹: {tuple(args.start)} → {tuple(args.end)}",
            show=False,
        )
    
    print("="*60)
    print("✓ 完成！")
    print("="*60)
    
    # 使用提示
    print("\n使用生成的数据:")
    print(f"  在 Python 中加载: data = np.load('{args.output}')")
    print(f"  可视化: python -m safe_game_flow.utils.visualize --file {args.output}")
    print()


if __name__ == "__main__":
    main()
