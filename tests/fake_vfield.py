import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch
from safe_game_flow.flow_matching.model import FlowMatching1D

# 模型参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 初始化模型
model = FlowMatching1D(
    in_channels=2,
    hidden_channels=256,
    num_blocks=6,
    time_emb_dim=128,
    kernel_size=3
).to(device)

ckpt = torch.load("/home/karl/safe-game-flow/runs/exp1/checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model"])
@torch.no_grad()
def vfield_fun(traj, t):
    global model
    t_input = torch.tensor(t,device=device,dtype=torch.float32)
    traj_input = torch.tensor(traj.T,device=device,dtype=torch.float32)
    t_input = t_input.unsqueeze(0)
    traj_input = traj_input.unsqueeze(0)
    # print(traj_input.shape)
    # print(t_input.shape)
    v = model(traj_input,t_input).detach().cpu().numpy().squeeze(0).T  # 模型预测速度场
    return v
# 目标与初始
goal = np.array([2.5, 2.0])
x0   = np.array([-2.5, -2.0])

# 离散轨迹长度（H个点）
H = 100
target_traj = np.linspace(x0, goal, H)

# 时间与积分
T = 1.0
K = 100
dt = T / K

# 多障碍（可加多个），半径带安全裕度
obstacles = [
    {'c': np.array([-0.5, 0.0]), 'r': 0.5, 'margin': 0.2},
    # 示例：再加一个障碍可按下行注释
    {'c': np.array([1.0, 0.5]), 'r': 0.4, 'margin': 0.2},
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
traj = np.linspace(x0, goal, H).copy()
# traj = np.random.randn(H,2)
traj0 = traj



# 记录
mean_v, mean_u, min_h = [], [], []
traj_history = []

for k in range(K):
    t = k / K
    # v = (target_traj - traj)   # 名义速度场
    v = vfield_fun(traj,t)

    u, h_min = solve_min_norm_qp(traj, v, t, obstacles)
    traj = traj + dt * (v + u)

    traj_history.append(traj.copy())
    mean_v.append(np.linalg.norm(v, axis=1).mean())
    mean_u.append(np.linalg.norm(u, axis=1).mean())
    min_h.append(h_min.min())

traj_history = np.array(traj_history)  # (K, H, 2)
mean_v = np.array(mean_v)
mean_u = np.array(mean_u)
min_h = np.array(min_h)

# 取最终生成的轨迹（t=1）
trajectory = traj_history[-1]

# ========== 可视化 ==========
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

# 轨迹 + 障碍（用颜色表示时间）
ax = axs[0]
# 把离散点按时间绘制，最后一条是最终的 T1
t_colors = np.linspace(0, 1, trajectory.shape[0])
sc = ax.scatter(trajectory[:, 0], trajectory[:, 1], c=np.linspace(0,1,H), cmap='viridis', s=12, label='T1 (end)')
ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', alpha=0.4)

# 障碍+裕度可视化
th = np.linspace(0, 2*np.pi, 200)
for j, obs in enumerate(obstacles):
    c = obs['c']; r_eff = obs['r_eff']
    ax.plot(c[0] + r_eff*np.cos(th), c[1] + r_eff*np.sin(th), 'r-', lw=2, label='obstacle+margin' if j==0 else None)

ax.plot([x0[0]], [x0[1]], 'bo', label='start')
ax.plot([goal[0]], [goal[1]], 'rx', label='goal')

ax.axis('equal'); ax.grid(True); ax.legend()
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Safe generation with FMBF (QP over H points)')

# 信号：均值速度/修正范数 + 全轨迹最小h
ax2 = axs[1]
tt = np.arange(K) * dt
ax2.plot(tt, mean_v, label='mean |v|')
ax2.plot(tt, mean_u, label='mean |u| (CBF correction)')
ax2.plot(tt, min_h, label='min h over points')
ax2.axhline(0.0, color='k', lw=0.8)
ax2.grid(True); ax2.legend(); ax2.set_xlabel('t')
ax2.set_title('Signals (mean norms, min h)')

plt.tight_layout(); plt.show()