# Experiment 3 Metrics Explained（指标定义与意义）

本实验在十字路口两车场景中，对三种方法进行多次随机试验（Pure FM / Non‑Game CBF / Game CBF），并统计 **安全性、分布保持、效率** 指标。

本文件说明这些指标 **如何计算**、以及它们 **反映的意义**。对应实现主要在：
- `experiments/exp3_statistical_comparison/run_experiments.py`
- `experiments/exp3_statistical_comparison/visualize_results.py`

---

## 记号与数据结构

- 一次试验中，每个方法会生成两条最终轨迹：
  - Agent 1: `final_trajectory_1`，形状为 \((H, 2)\)
  - Agent 2: `final_trajectory_2`，形状为 \((H, 2)\)
- 轨迹点数：\(H\)（脚本默认 50）
- 积分步数：\(K\)（脚本默认 100）
- 离散时间步：\(t_k = k/K\)，\(k=0,\dots,K-1\)
- 安全距离阈值：\(d_{\text{safe}} = \texttt{D_SAFE}\)
- 每一步都有名义速度场（Flow Matching 输出）：
  - \(v_1(t_k)\in\mathbb{R}^{H\times 2}\), \(v_2(t_k)\in\mathbb{R}^{H\times 2}\)
- CBF-QP 输出的修正项：
  - \(u_1(t_k)\in\mathbb{R}^{H\times 2}\), \(u_2(t_k)\in\mathbb{R}^{H\times 2}\)
- 轨迹更新（在 `run_experiments.py` 中）：
  \[
  x_1 \leftarrow x_1 + \Delta t\,(v_1 + u_1),\quad
  x_2 \leftarrow x_2 + \Delta t\,(v_2 + u_2)
  \]

---

## A. 安全性（Safety）

### A1. `min_distance_final`（最终轨迹最小距离）

**定义**：在最终生成的两条轨迹上，对齐同一索引 \(i\) 的点，计算两车距离并取最小值：
\[
d_{\min}^{\text{final}} = \min_{i=0,\dots,H-1}\|x_{1,i}^{\text{final}} - x_{2,i}^{\text{final}}\|_2
\]

**代码对应**（`run_experiments.py`）：
- `final_dist = np.linalg.norm(traj_1 - traj_2, axis=1)`
- `min_dist_final = final_dist.min()`

**意义**：
- 这是你定义的“最终是否相撞”的核心指标（也是 `collision_rate` 的依据）。
- 反映最终生成轨迹是否满足安全距离约束。

---

### A2. `collision` / `collision_rate`（最终碰撞判定/碰撞率）

**定义**：
\[
\text{collision} = \mathbb{1}\left[d_{\min}^{\text{final}} < d_{\text{safe}}\right]
\]

**碰撞率**（跨多次试验）：
\[
\text{collision\_rate} = \frac{1}{N}\sum_{n=1}^{N}\text{collision}_n \times 100\%
\]

**意义**：
- 直接衡量安全性能（越低越好）。
- 对论文而言是最直观的安全统计结果。

---

### A3. `min_distance_over_time`（过程最小距离，诊断项）

**定义**：在积分过程中每一步都会形成一个“中间轨迹对”（因为我们在轨迹空间上更新），在每个时间步计算一次最小距离并取最小：
\[
d_{\min}^{\text{rollout}}=\min_{k=0,\dots,K-1}\left(\min_{i=0,\dots,H-1}\|x_{1,i}(t_k)-x_{2,i}(t_k)\|_2\right)
\]

**代码对应**：
- 每步 `dist = np.linalg.norm(traj_1 - traj_2, axis=1)`
- `min_dists_over_time.append(dist.min())`
- 最后 `min_dist_over_time = min(min_dists_over_time)`

**意义**：
- **不是主碰撞定义**，但能反映“过程中是否贴得很近/是否出现过危险时刻”。
- 常见现象：最终轨迹 safe，但中间过程更“贴边”，这能体现方法的保守性/协调性差异。

---

## B. 分布保持（Distribution Preservation / Deviation）

核心思想：CBF 会修改名义速度场（Flow Matching 的行为）。我们希望在保持安全的同时，尽量少改动 learned behavior（分布保持更好）。

### B1. `total_correction`（总修正量，Σ||u||）

**定义**：把每一步每个点上的修正向量范数累加（两车相加）：
\[
\text{TotalCorrection} =
\sum_{k=0}^{K-1}\sum_{i=0}^{H-1}\|u_{1,i}(t_k)\|_2
\;+\;
\sum_{k=0}^{K-1}\sum_{i=0}^{H-1}\|u_{2,i}(t_k)\|_2
\]

**代码对应**（`run_experiments.py`）：
- `u*_norm_per_point = np.linalg.norm(u*, axis=1)`
- `total_u* += u*_norm_per_point.sum()`
- `total_correction = total_u1 + total_u2`

**意义**：
- 越小表示越接近 Pure FM 的生成行为（分布保持更好）。
- 通常 Non‑Game CBF 会更保守（更大修正），Game CBF 在协调下可能用更小的总修正达成安全。

---

### B2. `peak_mean_u`（最大平均修正强度）

**定义**：每一步先计算两车修正的“点均值”，再取时间上的最大值：
\[
\text{meanU}(t_k)=\frac{1}{2}\left(\frac{1}{H}\sum_i\|u_{1,i}(t_k)\|_2+\frac{1}{H}\sum_i\|u_{2,i}(t_k)\|_2\right)
\]
\[
\text{PeakMeanU}=\max_{k}\text{meanU}(t_k)
\]

**意义**：
- 衡量“最激进/最保守的一瞬间”修正强度有多大。
- 当两种方法都 safe 时，这个指标往往更能体现 **博弈协调让避让更平滑、更不极端**。

---

### B3. 图7：到 Pure FM 的“生成分布距离”（Wasserstein）

你要求：图7 不再比较“与真实训练数据的距离”，而是比较 **其他算法生成分布 vs Pure FM 生成分布**。

**做法**（`visualize_results.py`）：
- 对每个方法，提取多次试验的终点集合（每个 agent 各一组）：
  - Agent 1 endpoints: \(\{x_{1,\text{end}}^{(n)}\}_{n=1}^N\)
  - Agent 2 endpoints: \(\{x_{2,\text{end}}^{(n)}\}_{n=1}^N\)
- 以 Pure FM 的终点分布作为基准，计算 1D Wasserstein 距离并相加（x/y，两个 agent）：
\[
W_{\text{to Pure}}=
W(x_{1,\text{end}}^x,\;x_{1,\text{end,Pure}}^x)+
W(x_{1,\text{end}}^y,\;x_{1,\text{end,Pure}}^y)+
W(x_{2,\text{end}}^x,\;x_{2,\text{end,Pure}}^x)+
W(x_{2,\text{end}}^y,\;x_{2,\text{end,Pure}}^y)
\]

**意义**：
- Pure FM 的 “to Pure” 距离应接近 0。
- CBF 方法越接近 Pure FM 的生成分布，该距离越小 → **分布保持更好**。
- 这是“相对保持”指标：把 Pure FM 当作 learned behavior 的参考，不受训练数据本身噪声/覆盖范围影响。

> 说明：这里用的是“终点分布”作为一个轻量的分布表征。更严格可以扩展为：
> - 对整条轨迹特征（曲率、长度、最大偏离等）做分布距离
> - 或对轨迹点云/速度点云做 Wasserstein/MMD

---

## C. 效率（Efficiency）

### C1. `mean_comp_time_ms`（平均计算时间）

**定义**：每个积分步对“求解修正 u”的耗时，取平均后换算毫秒：
\[
\text{TimePerStep} = \frac{1}{K}\sum_{k=0}^{K-1}\Delta t_k \times 1000\;\text{ms}
\]

**意义**：
- 反映方法的实时性。
- 通常 Non‑Game 是两个小QP，Game 是一个大QP，时间可能不同；也可能因为迭代次数不同而变化。

---

### C2. `solver_success_rate`（QP 求解成功率）

**定义**：统计 OSQP 是否返回 `'solved'`（脚本里通过 `solver_successes` 记录），再取平均：
\[
\text{SuccessRate}=\frac{\#\{\text{solved}\}}{\text{total steps}}\times 100\%
\]

**意义**：
- 衡量数值稳定性/鲁棒性。
- 若 success 率低，可能频繁触发 fallback，从而影响安全与分布保持。

---

## Trade-off（如何读这些图）

当两种方法都安全时，博弈方法的优势通常体现在：
- **更小 `total_correction`**：更接近 learned behavior（分布保持更好）
- **更小 `peak_mean_u`**：避让更平滑，不需要“猛打方向盘”
- **更小的 to‑Pure Wasserstein 距离**：整体生成分布更接近 Pure FM（更少扭曲）

这些指标合在一起，就能定量证明 “Game‑Theoretic CBF 在保证安全的同时，更好地保持 learned distribution，并且更不保守”。

