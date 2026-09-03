# Experiment 3: Statistical Comparison of Safe Multi-Agent Methods

## 实验目标

在十字路口场景下，通过大量随机实验统计比较不同安全控制方法的性能。

## 实验设置

### 场景
- 十字路口交叉场景
- Agent 1: 西→东方向（随机起点和终点）
- Agent 2: 南→北方向（随机起点和终点）

### 随机化
```python
# Agent 1 (West-East)
start_1_x ∈ [-40, -20]  # 起点x坐标随机
start_1_y ∈ [-3, 3]      # 起点y坐标小范围随机（道路宽度）
goal_1_x ∈ [20, 40]      # 终点x坐标随机
goal_1_y ∈ [-3, 3]       # 终点y坐标小范围随机

# Agent 2 (South-North)
start_2_x ∈ [-3, 3]      # 起点x坐标小范围随机
start_2_y ∈ [-40, -20]   # 起点y坐标随机
goal_2_x ∈ [-3, 3]       # 终点x坐标小范围随机
goal_2_y ∈ [20, 40]      # 终点y坐标随机
```

### 实验次数
- N = 100-200 次随机实验
- 每次实验随机生成新的起点和终点

---

## 对比算法（3个方法）

### 方法1: **Pure Flow Matching (Baseline)** 
**描述**: 纯粹的Flow Matching，没有任何安全约束

```python
# 直接使用learned velocity field
v1 = velocity_field_1(traj_1, t)
v2 = velocity_field_2(traj_2, t)
traj_1 = traj_1 + dt * v1
traj_2 = traj_2 + dt * v2
# No safety correction: u1 = 0, u2 = 0
```

**预期结果**: 
- ❌ 高碰撞率
- ✓ 最接近训练数据分布（因为没有修正）
- ✓ 计算最快

**作用**: 证明安全约束的必要性

---

### 方法2: **Non-Game-Theoretic CBF (Independent Optimization)**
**描述**: 每个agent独立优化，把其他agent当作移动障碍物

```python
# Agent 1独立求解QP：
# min ||u1||^2
# s.t. 
#   - Agent 1的静态障碍约束
#   - Agent 2当作动态障碍：h(x1, x2) >= 0

# Agent 2独立求解QP：
# min ||u2||^2
# s.t. 
#   - Agent 2的静态障碍约束
#   - Agent 1当作动态障碍：h(x2, x1) >= 0

# 两个QP独立求解，不协调
```

**关键区别**:
- **独立决策**: 每个agent只考虑自己的correction，不考虑对方的correction
- **把对方当障碍**: Agent 2在Agent 1的约束中只是一个"障碍物"参数
- **可能冲突**: 两个agent可能做出相互抵消的决策（例如都向同一方向躲避）

**预期结果**:
- ⚠️ 碰撞率比Pure FM低，但仍可能有碰撞
- ⚠️ 偏离训练分布较多（可能过度反应或不协调）
- ✓ 计算快（两个小QP而非一个大QP）

**作用**: 证明博弈论协调的重要性

---

### 方法3: **Ours - Game-Theoretic CBF + Flow Matching** ⭐
**描述**: 我们提出的方法 - 游戏论联合优化

```python
# 联合求解QP：
# min ||u1||^2 + ||u2||^2
# s.t. 
#   - Agent 1的静态障碍约束: a1^T u1 >= b1
#   - Agent 2的静态障碍约束: a2^T u2 >= b2
#   - 双方协商的安全约束: a^T u1 - a^T u2 >= b12

# 一个大QP，联合优化u1和u2
```

**关键特点**:
- **协同决策**: 两个agent的correction在一个QP中同时优化
- **博弈论约束**: 安全约束考虑双方的correction (u1, u2)
- **最小总correction**: 目标是最小化总的偏离
- **轨迹级约束**: 考虑整个H点轨迹的安全

**优势**: 
- ✓ 保持learned behavior (最小总correction)
- ✓ 分布式可实现（虽然需要通信）
- ✓ 协调决策，避免冲突

**预期结果**:
- ✓ 低碰撞率（理想情况下零碰撞）
- ✓ 较好地保持训练分布
- ⚠️ 计算时间适中（一个大QP）

---

## 三个方法的层次关系

```
Pure FM (无约束)
    ↓ +安全约束
Non-Game CBF (独立优化)
    ↓ +博弈论协调
Game-Theoretic CBF (联合优化) ← Ours
```

每一层都解决了上一层的问题：
1. Pure FM → Non-Game CBF: 需要安全约束
2. Non-Game CBF → Game CBF: 需要协调避免冲突

---

## 评估指标（3大类）

### 1. 安全性指标 🛡️

#### 1.1 碰撞率
```python
# 定义碰撞：任意时刻最小距离 < d_safe
collision_rate = (num_collisions / total_experiments) * 100
```

#### 1.2 最小距离统计
```python
min_distance_mean = np.mean(min_distances_all_experiments)
min_distance_std = np.std(min_distances_all_experiments)
min_distance_min = np.min(min_distances_all_experiments)
min_distance_25th = np.percentile(min_distances_all_experiments, 25)
min_distance_75th = np.percentile(min_distances_all_experiments, 75)
```

#### 1.3 安全违反严重程度
```python
# 违反程度 = d_safe - actual_distance (when < d_safe)
violation_severity = np.mean(d_safe - min_distances[min_distances < d_safe])
```

---

### 2. 分布保持指标 📊

#### 2.1 Wasserstein Distance (推荐)
```python
from scipy.stats import wasserstein_distance

# 对每个agent分别计算
# 比较生成轨迹的终点分布与训练数据的终点分布
generated_endpoints = [traj[-1] for traj in generated_trajectories]
training_endpoints = training_data[:, :, -1]  # (N, 2)

# X方向和Y方向分别计算
W_x = wasserstein_distance(generated_endpoints[:, 0], training_endpoints[:, 0])
W_y = wasserstein_distance(generated_endpoints[:, 1], training_endpoints[:, 1])
W_total = W_x + W_y
```

#### 2.2 轨迹形状相似度
```python
# 比较轨迹特征的统计分布
features = {
    'path_length': np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)),
    'mean_curvature': ...,  # 曲率的平均值
    'max_deviation': ...,   # 距离直线的最大偏离
}

# 比较generated vs training的特征分布
# 使用KL散度或Wasserstein距离
```

#### 2.3 总Correction量（偏离程度）
```python
# 越小越好，说明越接近原始learned behavior
total_correction = np.mean([
    np.linalg.norm(u1_all, axis=1).sum(),
    np.linalg.norm(u2_all, axis=1).sum()
])
```

---

### 3. 效率指标 ⚡

#### 3.1 计算时间
```python
# 每步平均计算时间（毫秒）
time_per_step_ms = total_computation_time / K * 1000

# 每个实验总时间
time_per_experiment_s = total_experiment_time
```

#### 3.2 求解成功率
```python
# QP求解器成功率（特别是OSQP）
solver_success_rate = (num_successful_solves / total_solves) * 100
```

---

### 指标汇总表格

| 指标类别 | 指标名称 | Pure FM | Non-Game CBF | Ours (Game CBF) |
|---------|---------|---------|--------------|-----------------|
| **安全性** | 碰撞率 (%) | ? | ? | ? |
| | 最小距离 (mean±std) | ? | ? | ? |
| | 违反严重度 | ? | ? | ? |
| **分布保持** | Wasserstein距离 | ? | ? | ? |
| | 总Correction量 | 0 | ? | ? |
| | 路径长度偏差 | ? | ? | ? |
| **效率** | 时间/步 (ms) | ? | ? | ? |
| | 求解成功率 (%) | 100 | ? | ? |

---

## 实现建议

### 统一接口设计

```python
class SafetyMethod:
    """Base class for all safety methods"""
    def __init__(self, name):
        self.name = name
    
    def compute_correction(self, traj_1, traj_2, v1, v2, t, obstacles, d_safe):
        """
        计算安全correction
        
        Args:
            traj_1, traj_2: (H, 2) 当前轨迹
            v1, v2: (H, 2) 名义速度场
            t: float, 当前时间
            obstacles: list, 静态障碍物
            d_safe: float, 安全距离
            
        Returns:
            u1, u2: (H, 2) correction速度
        """
        raise NotImplementedError
```

### 三个方法的实现

```python
class PureFlowMatching(SafetyMethod):
    """方法1: 纯Flow Matching，无安全约束"""
    def compute_correction(self, traj_1, traj_2, v1, v2, t, obstacles, d_safe):
        H = traj_1.shape[0]
        return np.zeros((H, 2)), np.zeros((H, 2))


class NonGameCBF(SafetyMethod):
    """方法2: 非博弈CBF，独立优化"""
    def compute_correction(self, traj_1, traj_2, v1, v2, t, obstacles, d_safe):
        # Agent 1: 独立求解，把Agent 2当障碍
        u1 = self._solve_single_agent_qp(traj_1, v1, traj_2, obstacles, d_safe, t)
        
        # Agent 2: 独立求解，把Agent 1当障碍
        u2 = self._solve_single_agent_qp(traj_2, v2, traj_1, obstacles, d_safe, t)
        
        return u1, u2
    
    def _solve_single_agent_qp(self, my_traj, my_v, other_traj, obstacles, d_safe, t):
        """
        单个agent的QP:
        min ||u||^2
        s.t. 
          - 静态障碍约束
          - 把other_traj当作静态障碍
        """
        # 实现独立QP求解
        ...


class GameTheoreticCBF(SafetyMethod):
    """方法3: 博弈论CBF，联合优化（Ours）"""
    def compute_correction(self, traj_1, traj_2, v1, v2, t, obstacles, d_safe):
        # 联合求解大QP
        return solve_min_norm_qp_two_agents(
            traj_1, v1, traj_2, v2, t, obstacles, obstacles, d_safe
        )
```

### 统一实验框架

```python
def run_single_experiment(method, start_1, goal_1, start_2, goal_2, 
                         model_1, model_2, config):
    """
    运行单次实验
    
    Returns:
        metrics: dict with safety, distribution, and efficiency metrics
    """
    H, K, dt = config['H'], config['K'], config['dt']
    d_safe = config['d_safe']
    
    # 初始化
    traj_1 = np.linspace(start_1, goal_1, H)
    traj_2 = np.linspace(start_2, goal_2, H)
    
    # 存储指标
    min_dists = []
    corrections = []
    comp_times = []
    
    # 主循环
    for k in range(K):
        t = k / K
        
        # 计算名义速度
        v1 = velocity_field_1(traj_1, t, model_1)
        v2 = velocity_field_2(traj_2, t, model_2)
        
        # 计算correction（计时）
        start_time = time.time()
        u1, u2 = method.compute_correction(traj_1, traj_2, v1, v2, t, 
                                          config['obstacles'], d_safe)
        comp_time = time.time() - start_time
        
        # 更新轨迹
        traj_1 = traj_1 + dt * (v1 + u1)
        traj_2 = traj_2 + dt * (v2 + u2)
        
        # 记录指标
        dist = np.linalg.norm(traj_1 - traj_2, axis=1)
        min_dists.append(dist.min())
        corrections.append((np.linalg.norm(u1), np.linalg.norm(u2)))
        comp_times.append(comp_time)
    
    # 计算metrics
    metrics = {
        'collision': min(min_dists) < d_safe,
        'min_distance': min(min_dists),
        'mean_min_distance': np.mean(min_dists),
        'mean_correction': np.mean(corrections),
        'mean_comp_time': np.mean(comp_times),
        'final_trajectory_1': traj_1,
        'final_trajectory_2': traj_2,
    }
    
    return metrics


def run_batch_experiments(methods, n_experiments, config):
    """
    运行批量实验
    
    Args:
        methods: dict of {method_name: SafetyMethod}
        n_experiments: int
        config: dict with experiment configuration
        
    Returns:
        results: dict of {method_name: list of metrics}
    """
    results = {name: [] for name in methods.keys()}
    
    for i in range(n_experiments):
        # 随机生成起点和终点
        start_1, goal_1 = sample_random_start_goal_1()
        start_2, goal_2 = sample_random_start_goal_2()
        
        # 对每种方法运行
        for method_name, method in methods.items():
            metrics = run_single_experiment(
                method, start_1, goal_1, start_2, goal_2,
                config['model_1'], config['model_2'], config
            )
            results[method_name].append(metrics)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{n_experiments} experiments")
    
    return results
```

---

## 预期结果

### 假设（待实验验证）

| 方法 | 碰撞率 | 分布保持 | 计算时间 |
|------|--------|---------|---------|
| Pure FM | ~40-60% | ⭐⭐⭐⭐⭐ (最好) | ⭐⭐⭐⭐⭐ (最快) |
| Non-Game CBF | ~10-20% | ⭐⭐⭐ (中等) | ⭐⭐⭐⭐ (快) |
| Game CBF (Ours) | ~0-5% | ⭐⭐⭐⭐ (好) | ⭐⭐⭐ (中等) |

### 论文可用的结论

1. **Pure FM**: 证明learned model本身不保证安全
2. **Non-Game CBF**: 独立优化可以提高安全性，但不够协调
3. **Game CBF**: 博弈论协调在保持learned behavior的同时实现高安全性
