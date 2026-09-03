# 实验4：四机钻环集群博弈（PyBullet）

## 场景

4 架四旋翼各对应 **1 个环**，从四角出发，穿过中心区域的环后飞往对角，路径交叉，需 **集群博弈 CBF** 避碰。

| 无人机 | 起点 | 环心 | 终点 |
|--------|------|------|------|
| UAV-1 红 | (-7,-7,2.5) | (0,-2.8,3.0) | (7,7,2.5) |
| UAV-2 蓝 | (7,-7,3.0) | (2.8,0,3.5) | (-7,7,3.0) |
| UAV-3 绿 | (7,7,3.5) | (0,2.8,4.0) | (-7,-7,3.5) |
| UAV-4 橙 | (-7,7,2.8) | (-2.8,0,3.2) | (7,-7,2.8) |

- 机间安全距离 `d_safe = 1.6 m`
- 中心静态柱障碍 1 个

## 方法

1. **名义流场**：航点插值参考轨迹 + 流匹配式收缩速度（`reference_paths.py`）
2. **集群博弈 CBF-QP**：集中式 v-GNE，`min Σ||u_i||²` s.t. 机间 + 静态障碍 CBF（扩展 exp2 至 4 机 3D）
3. **PyBullet**：四旋翼 URDF + 环可视化 + 同步飞行 GIF

## 运行

```bash
pip install osqp pybullet
cd safe-game-flow-master
python experiments/exp4_swarm_ring_game_pybullet/run_experiment.py
python experiments/exp4_swarm_ring_game_pybullet/run_experiment.py --gui
```

## 输出

```
results/
├── trajectories_3d.png       # 3D 轨迹对比
├── safety_metrics.png        # 机间距离 & 规划时间
├── swarm_ring_flight.gif     # PyBullet 四机钻环动画
├── swarm_ring_snapshot.png   # 截图
├── traj_safe_agent*.npy
└── metrics.json
```

## 目录

```
exp4_swarm_ring_game_pybullet/
├── config.py
├── reference_paths.py
├── game_cbf_qp.py
├── safeflow_swarm.py
├── pybullet_sim.py
├── run_experiment.py
└── assets/quadrotor.urdf
```
