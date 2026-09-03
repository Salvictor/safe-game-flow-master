import matplotlib.pyplot as plt
import numpy as np
import torch
import osqp
from scipy import sparse

from safe_game_flow.flow_matching.model import FlowMatching1D

# 车1目标与起点 (西到东)
x0_1   = np.array([-30, 0])   # 起点在西侧
goal_1 = np.array([10, -0])    # 终点在东侧

# 车2目标与起点 (南到北)  
x0_2   = np.array([0.0, -20])   # 起点在南侧
goal_2 = np.array([0, 20])    # 终点在北侧

# 离散轨迹长度（H个点）
H = 50
target_traj_1 = np.linspace(x0_1, goal_1, H)
target_traj_2 = np.linspace(x0_2, goal_2, H)
# target_traj_1 = np.random.randn(H,2)
# target_traj_2 = np.random.randn(H,2)
d_safe = 8
import torch
from safe_game_flow.flow_matching.model import FlowMatching1D

# 模型参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型1: 西到东轨迹 (车1)
model_1 = FlowMatching1D(
    in_channels=2,
    hidden_channels=256,
    num_blocks=6,
    time_emb_dim=128,
    kernel_size=3
).to(device)

ckpt_1 = torch.load("/home/karl/safe-game-flow/runs/model_west_to_east/checkpoint.pt", map_location=device)
model_1.load_state_dict(ckpt_1["model"])
model_1.eval()

# 初始化模型2: 南到北轨迹 (车2)
model_2 = FlowMatching1D(
    in_channels=2,
    hidden_channels=256,
    num_blocks=6,
    time_emb_dim=128,
    kernel_size=3
).to(device)

ckpt_2 = torch.load("/home/karl/safe-game-flow/runs/model_south_to_north/checkpoint.pt", map_location=device)
model_2.load_state_dict(ckpt_2["model"])
model_2.eval()

@torch.no_grad()
def vfield_fun_1(traj, t):
    """车1的速度场 (西到东模型)"""
    t_input = torch.tensor(t, device=device, dtype=torch.float32)
    traj_input = torch.tensor(traj.T, device=device, dtype=torch.float32)
    t_input = t_input.unsqueeze(0)
    traj_input = traj_input.unsqueeze(0)
    v = model_1(traj_input, t_input).detach().cpu().numpy().squeeze(0).T
    return v

@torch.no_grad()
def vfield_fun_2(traj, t):
    """车2的速度场 (南到北模型)"""
    t_input = torch.tensor(t, device=device, dtype=torch.float32)
    traj_input = torch.tensor(traj.T, device=device, dtype=torch.float32)
    t_input = t_input.unsqueeze(0)
    traj_input = traj_input.unsqueeze(0)
    v = model_2(traj_input, t_input).detach().cpu().numpy().squeeze(0).T
    return v
# 时间与积分
T = 1.0
K = 100
dt = T / K

# 多障碍（可加多个），半径带安全裕度
# 在十字路口中心放置障碍，测试两车避障和互相避碰
obstacles = [
    # {'c': np.array([0.0, 0.0]), 'r': 5, 'margin': 0.2},  # 中心障碍
]
for obs in obstacles:
    obs['r_eff'] = obs['r'] + obs['margin']

# φ(t,h)（论文：安全区常数，不安全区随 t->1 发散）
phi0 = 1.0
phi1_scale = 1.0
def phi_fun(t, h_val):
    if h_val >= 0:
        return phi0
    t_clip = min(t, 1.0 - 1e-6)
    return phi1_scale / (1.0 - t_clip)

# h 与 ∇h（对某个障碍）
def h_of(x, c, r_eff):
    dx = x - c
    return np.sum(dx**2, axis=1) - r_eff**2  # (H,)

def grad_h_of(x, c):
    return 2.0 * (x - c)  # (H,2)


# ========== 两车最小范数QP：min 0.5 * (||u1||^2 + ||u2||^2) s.t. 线性不等式 ==========
pair_radius = 0.5   # 每车的等效半径（或包络半径）
pair_margin = 0.2   # 安全裕度
d_pair_eff = pair_radius + pair_margin  # 彼此安全距离

def solve_min_norm_qp_two_agents(traj1, v1, traj2, v2, t, obstacles1, obstacles2, d_safe):
    """
    输入:
      traj1, v1: 车1的离散轨迹与名义速度场 (H,2)
      traj2, v2: 车2的离散轨迹与名义速度场 (H,2)
      t: 归一化时间 ∈ [0,1)
      obstacles1, obstacles2: 各自的静态圆障碍列表（可为空）
      d_safe: 彼此最小安全距离（标量）
    返回:
      u1, u2: (H,2) 两车的最小范数修正解
      h12: (H,) 两车彼此的 h 值（可用于记录 min/plot）
    """
    H = traj1.shape[0]
    # 组装 A_big ∈ R^{m × 4H}, b_big ∈ R^{m}
    A_rows = []
    b_rows = []

    # 1) 车1对静态障碍
    for obs in obstacles1:
        c = obs['c']; r_eff = obs['r_eff']
        h1 = h_of(traj1, c, r_eff)                   # (H,)
        a1 = grad_h_of(traj1, c)                     # (H,2)
        phi1 = np.array([phi_fun(t, h) for h in h1]) # (H,)
        b1 = -np.sum(a1 * v1, axis=1) - phi1 * h1    # (H,)
        for i in range(H):
            row = np.zeros(4*H, dtype=float)
            row[2*i:2*i+2] = a1[i]       # 只作用于 u1_i
            A_rows.append(row)
        b_rows.append(b1)

    # 2) 车2对静态障碍
    for obs in obstacles2:
        c = obs['c']; r_eff = obs['r_eff']
        h2 = h_of(traj2, c, r_eff)
        a2 = grad_h_of(traj2, c)
        phi2 = np.array([phi_fun(t, h) for h in h2])
        b2 = -np.sum(a2 * v2, axis=1) - phi2 * h2
        for i in range(H):
            row = np.zeros(4*H, dtype=float)
            row[2*H + 2*i: 2*H + 2*i + 2] = a2[i]  # 只作用于 u2_i
            A_rows.append(row)
        b_rows.append(b2)

    # 3) 两车之间的互相避碰
    d = traj1 - traj2
    h12 = np.sum(d*d, axis=1) - d_safe**2           # (H,)
    a12 = 2.0 * d                                   # (H,2) = ∂h/∂p1
    phi12 = np.array([phi_fun(t, h) for h in h12])
    # 右端：-a12^T v1 + a12^T v2 - φ h12
    b12 = -np.sum(a12 * v1, axis=1) + np.sum(a12 * v2, axis=1) - phi12 * h12

    for i in range(H):
        row = np.zeros(4*H, dtype=float)
        row[2*i:2*i+2] = a12[i]               # 对 u1_i 的系数
        row[2*H + 2*i: 2*H + 2*i + 2] = -a12[i]  # 对 u2_i 的系数
        A_rows.append(row)
    b_rows.append(b12)

    if len(A_rows) == 0:
        # 没有任何约束（不太可能），u=0
        return np.zeros((H,2)), np.zeros((H,2))

    A_big = np.vstack(A_rows)              # (m, 4H)
    b_big = np.concatenate(b_rows, axis=0) # (m,)

    # 使用OSQP求解器
    # minimize    (1/2) u^T P u + q^T u
    # subject to  l <= A u <= u_bound
    n_vars = 4 * H
    P = sparse.eye(n_vars, format='csc')
    q = np.zeros(n_vars)
    
    A_sparse = sparse.csc_matrix(A_big)
    l = b_big
    u_bound = np.inf * np.ones(len(b_big))
    
    prob = osqp.OSQP()
    prob.setup(P, q, A_sparse, l, u_bound, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    res = prob.solve()
    
    if res.info.status != 'solved':
        print(f"two-agent OSQP solver warning: {res.info.status}")
        # 失败时回退到简易策略
        u1 = np.zeros((H,2)); u2 = np.zeros((H,2))
        for i in range(H):
            a = a12[i]; a_n2 = np.dot(a, a) + 1e-12
            b = b12[i]
            if b > 0:
                u1[i] = ( b / (2*a_n2) ) * a
                u2[i] = (-b / (2*a_n2) ) * a
        return u1, u2

    u_flat = res.x
    u1 = u_flat[:2*H].reshape(H,2)
    u2 = u_flat[2*H:].reshape(H,2)
    
    return u1,u2
# 构造并求解QP：min 0.5*||u||^2 s.t. A_big u >= b_big
def solve_min_norm_qp(traj, v, t, obstacles):
    """
    traj: (H,2), v: (H,2), t ∈ [0,1)
    返回: u: (H,2), h_min_over_obs: (H,) 所有障碍上的最小 h
    """
    # 逐障碍计算 h、∇h、φ，并构造分块约束 Ai ui >= bi
    A_rows = []
    b_rows = []
    h_all  = []

    for obs in obstacles:
        c = obs['c']; r_eff = obs['r_eff']
        h_val = h_of(traj, c, r_eff)                 # (H,)
        a     = grad_h_of(traj, c)                   # (H,2)
        phi   = np.array([phi_fun(t, h_i) for h_i in h_val])  # (H,)
        bi    = -np.sum(a * v, axis=1) - phi * h_val         # (H,)

        # 形成块对角矩阵行：每行只作用于对应 ui 的两个分量
        # Ai_row = [0... a_i^T ...0] ∈ R^{1×(2H)}
        for i in range(H):
            row = np.zeros(2*H, dtype=float)
            row[2*i:2*i+2] = a[i]
            A_rows.append(row)
        b_rows.append(bi)
        h_all.append(h_val)

    A_big = np.vstack(A_rows)                    # shape ((H * M), 2H)
    b_big = np.concatenate(b_rows, axis=0)       # shape ((H * M),)

    # 目标函数：0.5 * u^T u
    def objective(u_flat):
        return 0.5 * np.dot(u_flat, u_flat)

    # 约束：A_big u_flat - b_big >= 0（SLSQP支持向量不等式）
    cons = {'type': 'ineq', 'fun': lambda u: A_big @ u - b_big}

    u0 = np.zeros(2*H, dtype=float)

    res = minimize(objective, u0, method='SLSQP', constraints=cons,
                   options={'ftol': 1e-9, 'maxiter': 500, 'disp': False})
    if not res.success:
        print("solver error")
        # 失败时回退到逐点闭式解（单障碍情形等价；多障碍取最紧约束）
        u = np.zeros((H,2))
        # 对每个点，把多个障碍的约束取“最要求修正”的一个（最大 b_i/||a_i||^2 投影）
        for i in range(H):
            best_u = np.zeros(2)
            best_gain = 0.0
            for obs in obstacles:
                c = obs['c']; r_eff = obs['r_eff']
                a_i = 2.0 * (traj[i] - c)
                a_n2 = np.dot(a_i, a_i) + 1e-12
                h_i = np.dot(traj[i]-c, traj[i]-c) - r_eff**2
                b_i = - np.dot(a_i, v[i]) - phi_fun(t, h_i) * h_i
                if b_i > 0:
                    u_i = (b_i / a_n2) * a_i
                    gain = np.dot(u_i, u_i)
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u_i
            u[i] = best_u
        h_min_over_obs = np.min(np.vstack(h_all), axis=0)
        return u, h_min_over_obs

    u_flat = res.x
    u = u_flat.reshape(H, 2)
    h_min_over_obs = np.min(np.vstack(h_all), axis=0)
    return u, h_min_over_obs

# 初始化一条离散轨迹（可用先验采样；这里直接线性插值起点到终点）
traj_1 = target_traj_1
traj0_1 = traj_1

traj_2 = target_traj_2
traj0_2 = traj_2
# 记录
mean_v1, mean_v2, mean_u1, mean_u2, min_h1, min_h2 = [], [], [], [], [], []
min_dist_between_agents = []  # 记录两车之间的最小距离
mean_dist_between_agents = []  # 记录两车之间的平均距离

for k in range(K):
    t = k / K
    # 使用两个不同的模型生成速度场
    v1 = vfield_fun_1(traj_1, t)  # 车1: 西到东模型
    v2 = vfield_fun_2(traj_2, t)  # 车2: 南到北模型
    

    # v2 = (target_traj_2 - traj_2)   # 名义速度场
    u1,u2 = solve_min_norm_qp_two_agents(traj_1,v1,traj_2,v2,t,obstacles,obstacles,d_safe)
    # u, h_min = solve_min_norm_qp(traj, v, t, obstacles)
    
    # u1 = np.zeros((H,2))
    # u2 = np.zeros((H,2))

    traj_1 = traj_1 + dt * (v1 + u1)
    traj_2 = traj_2 + dt * (v2 + u2)
    
    # 计算两车之间的距离
    dist_between = np.linalg.norm(traj_1 - traj_2, axis=1)  # (H,) 每个离散点的距离
    min_dist_between_agents.append(dist_between.min())
    mean_dist_between_agents.append(dist_between.mean())

    mean_v1.append(np.linalg.norm(v1, axis=1).mean())
    mean_v2.append(np.linalg.norm(v2, axis=1).mean())
    mean_u1.append(np.linalg.norm(u1, axis=1).mean())
    mean_u2.append(np.linalg.norm(u2, axis=1).mean())
    # min_h.append(h_min.min())

# traj_history = np.array(traj_history)  # (K, H, 2)
mean_v1 = np.array(mean_v1)
mean_v2 = np.array(mean_v2)
mean_u1 = np.array(mean_u1)
mean_u2 = np.array(mean_u2)
min_h1 = np.array(min_h1)
min_h2 = np.array(min_h2)
min_dist_between_agents = np.array(min_dist_between_agents)
mean_dist_between_agents = np.array(mean_dist_between_agents)

# 打印统计信息
print("=" * 60)
print("两车安全距离统计:")
print("=" * 60)
print(f"安全距离阈值 (d_safe): {d_safe}")
print(f"两车最小距离 (全局): {min_dist_between_agents.min():.4f}")
print(f"两车最小距离 (平均): {min_dist_between_agents.mean():.4f}")
print(f"两车平均距离 (全局平均): {mean_dist_between_agents.mean():.4f}")
print(f"违反安全约束的时间步数: {np.sum(min_dist_between_agents < d_safe)} / {K}")
if np.sum(min_dist_between_agents < d_safe) > 0:
    print(f"最小违反量: {(d_safe - min_dist_between_agents[min_dist_between_agents < d_safe]).max():.4f}")
print("=" * 60)

# 取最终生成的轨迹（t=1）
trajectory_1 = traj_1
trajectory_2 = traj_2

# ========== 可视化 ==========
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 轨迹 + 障碍（用颜色表示时间）
ax = axs[0, 0]

# 车1轨迹 (西到东) - 使用蓝色系 (Blues)
t_colors = np.linspace(0, 1, trajectory_1.shape[0])
sc1 = ax.scatter(trajectory_1[:, 0], trajectory_1[:, 1], c=np.linspace(0,1,H), 
                 cmap='Blues', s=12, alpha=0.8, label='车1 (西→东)', vmin=0.2, vmax=1.0)
ax.plot(trajectory_1[:, 0], trajectory_1[:, 1], color='blue', linestyle='-', alpha=0.5, linewidth=2)

# 车2轨迹 (南到北) - 使用红色系 (Reds)
t_colors = np.linspace(0, 1, trajectory_2.shape[0])
sc2 = ax.scatter(trajectory_2[:, 0], trajectory_2[:, 1], c=np.linspace(0,1,H), 
                 cmap='Reds', s=12, alpha=0.8, label='车2 (南→北)', vmin=0.2, vmax=1.0)
ax.plot(trajectory_2[:, 0], trajectory_2[:, 1], color='red', linestyle='-', alpha=0.5, linewidth=2)

# 障碍+裕度可视化
th = np.linspace(0, 2*np.pi, 200)
for j, obs in enumerate(obstacles):
    c = obs['c']; r_eff = obs['r_eff']
    ax.plot(c[0] + r_eff*np.cos(th), c[1] + r_eff*np.sin(th), 'r-', lw=2, label='obstacle+margin' if j==0 else None)

ax.plot([x0_1[0]], [x0_1[1]], 'bo', label='start1')
ax.plot([goal_1[0]], [goal_1[1]], 'rx', label='goal1')
ax.plot([x0_2[0]], [x0_2[1]], 'bo', label='start2')
ax.plot([goal_2[0]], [goal_2[1]], 'rx', label='goal2')

ax.axis('equal'); ax.grid(True); ax.legend()
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Safe generation with FMBF (QP over H points)')

# 信号：均值速度/修正范数 + 全轨迹最小h
ax2 = axs[0, 1]
tt = np.arange(K) * dt
ax2.plot(tt, mean_v1, label='mean |v1|')
ax2.plot(tt, mean_v2, label='mean |v2|')
ax2.plot(tt, mean_u1, label='mean |u1| (CBF correction)')
ax2.plot(tt, mean_u2, label='mean |u2| (CBF correction)')
# ax2.plot(tt, min_h1, label='min h1 over points')
# ax2.plot(tt, min_h2, label='min h2 over points')
ax2.axhline(0.0, color='k', lw=0.8)
ax2.grid(True); ax2.legend(); ax2.set_xlabel('t')
ax2.set_title('Signals (mean norms, min h)')

# 两车之间的距离随时间变化
ax3 = axs[1, 0]
ax3.plot(tt, min_dist_between_agents, label='Min distance between agents', linewidth=2)
ax3.plot(tt, mean_dist_between_agents, label='Mean distance between agents', linewidth=2, alpha=0.7)
ax3.axhline(d_safe, color='r', linestyle='--', linewidth=2, label='Safety threshold (d_safe=d_safe)')
ax3.fill_between(tt, 0, d_safe, color='red', alpha=0.1, label='Unsafe region')
ax3.grid(True); ax3.legend(); ax3.set_xlabel('t'); ax3.set_ylabel('Distance')
ax3.set_title('Distance between two agents over time')
ax3.set_ylim(bottom=0)

# 最终轨迹上每个离散点之间的距离
ax4 = axs[1, 1]
final_dist = np.linalg.norm(trajectory_1 - trajectory_2, axis=1)
point_indices = np.arange(H)
ax4.plot(point_indices, final_dist, 'o-', linewidth=2, markersize=4)
ax4.axhline(d_safe, color='r', linestyle='--', linewidth=2, label='Safety threshold (d_safe=d_safe)')
ax4.fill_between(point_indices, 0, d_safe, color='red', alpha=0.1, label='Unsafe region')
ax4.grid(True); ax4.legend(); ax4.set_xlabel('Discrete point index')
ax4.set_ylabel('Distance')
ax4.set_title(f'Final trajectory inter-agent distance (min={final_dist.min():.3f})')
ax4.set_ylim(bottom=0)

plt.tight_layout(); plt.show()