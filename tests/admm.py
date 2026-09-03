import matplotlib.pyplot as plt
import numpy as np

# ================== 场景与参数 ==================
# 两车的起止目标
goal_1 = np.array([ 2.5,  2.0])
x0_1   = np.array([-2.5, -2.0])
goal_2 = np.array([ 2.5, -2.0])
x0_2   = np.array([-2.5,  2.0])

# 离散轨迹长度（每车H个路标点）
H = 30
target_traj_1 = np.linspace(x0_1, goal_1, H)
target_traj_2 = np.linspace(x0_2, goal_2, H)

# 仿真时间与步数（外层）
T  = 1.0
K  = 100
dt = T / K

# 静态圆障碍（可多个）
obstacles = [
    {'c': np.array([0.5, 0.0]), 'r': 0.5, 'margin': 0.2},
    {'c': np.array([1.0, 0.5]), 'r': 0.4, 'margin': 0.2},
]
for obs in obstacles:
    obs['r_eff'] = obs['r'] + obs['margin']

# 两车互相避让的有效安全距离 d_safe
pair_radius = 0.2
pair_margin = 0.2
d_pair_eff  = pair_radius + pair_margin

# φ(t,h)（安全区常数，不安全区随 t->1 发散）
phi0 = 1.0
phi1_scale = 1.0
def phi_fun(t, h_val):
    if h_val >= 0:
        return phi0
    t_clip = min(t, 1.0 - 1e-6)
    return phi1_scale / (1.0 - t_clip)

# 屏障与梯度（圆）
def h_of(x, c, r_eff):
    dx = x - c
    return np.sum(dx*dx, axis=1) - r_eff**2  # (H,)

def grad_h_of(x, c):
    return 2.0 * (x - c)  # (H,2)

# ================== 名义速度场（更接近FM风格） ==================
eps = 1e-3
k_goal = 0.8       # 目标吸引增益
k_smooth = 3.0     # 平滑/弹性带增益
k_tan = 0.0        # 可选：绕障切向偏置（0关）
def tangent_bias(traj, obstacles, influence=0.5):
    bias = np.zeros_like(traj)
    for obs in obstacles:
        c = obs['c']; r_eff = obs['r_eff']
        d = traj - c
        dist = np.linalg.norm(d, axis=1) + 1e-12
        h_val = dist**2 - r_eff**2
        mask = h_val < influence
        if np.any(mask):
            n = d[mask] / dist[mask, None]
            tvec = np.stack([-n[:,1], n[:,0]], axis=1)
            w = np.clip((influence - h_val[mask]) / max(influence, 1e-6), 0, 1)
            bias[mask] += (w[:,None] * tvec)
    return bias

def nominal_velocity(traj, target_traj, t):
    lam = k_goal / (1.0 - t + eps)
    v_goal = lam * (target_traj - traj)
    v_smooth = np.zeros_like(traj)
    if H >= 3:
        v_smooth[1:-1] = (traj[:-2] - 2*traj[1:-1] + traj[2:])
    v_tan = k_tan * tangent_bias(traj, obstacles, influence=0.5)
    return v_goal + k_smooth * v_smooth + v_tan

# ================== 约束构造：A1, A2, b ==================
def build_constraints_two_agents(traj1, v1, traj2, v2, t, obstacles1, obstacles2, d_safe):
    """
    约束形式：A1 u1 + A2 u2 >= b
    - 静态障碍（车1）： a1_i^T u1_i >= -a1_i^T v1_i - φ h1_i
    - 静态障碍（车2）： a2_i^T u2_i >= -a2_i^T v2_i - φ h2_i
    - 互相避让： a12_i^T u1_i - a12_i^T u2_i >= -a12_i^T v1_i + a12_i^T v2_i - φ h12_i
    返回：
      A1: (m, 2H), A2: (m, 2H), b: (m,), h12: (H,)
    """
    rows_A1, rows_A2, rows_b = [], [], []

    # 1) 车1-静态障碍
    for obs in obstacles1:
        c = obs['c']; r_eff = obs['r_eff']
        h1 = h_of(traj1, c, r_eff)                           # (H,)
        a1 = grad_h_of(traj1, c)                             # (H,2)
        phi1 = np.array([phi_fun(t, hi) for hi in h1])       # (H,)
        b1 = -np.sum(a1 * v1, axis=1) - phi1 * h1            # (H,)
        for i in range(H):
            row1 = np.zeros(2*H); row2 = np.zeros(2*H)
            row1[2*i:2*i+2] = a1[i]                          # 仅作用于 u1_i
            rows_A1.append(row1); rows_A2.append(row2)
        rows_b.append(b1)

    # 2) 车2-静态障碍
    for obs in obstacles2:
        c = obs['c']; r_eff = obs['r_eff']
        h2 = h_of(traj2, c, r_eff)
        a2 = grad_h_of(traj2, c)
        phi2 = np.array([phi_fun(t, hi) for hi in h2])
        b2 = -np.sum(a2 * v2, axis=1) - phi2 * h2
        for i in range(H):
            row1 = np.zeros(2*H); row2 = np.zeros(2*H)
            row2[2*i:2*i+2] = a2[i]                          # 仅作用于 u2_i
            rows_A1.append(row1); rows_A2.append(row2)
        rows_b.append(b2)

    # 3) 两车互相避让
    d = traj1 - traj2
    h12 = np.sum(d*d, axis=1) - d_safe**2                   # (H,)
    a12 = 2.0 * d                                           # (H,2)=∂h/∂p1
    phi12 = np.array([phi_fun(t, hi) for hi in h12])
    b12 = -np.sum(a12 * v1, axis=1) + np.sum(a12 * v2, axis=1) - phi12 * h12
    for i in range(H):
        row1 = np.zeros(2*H); row2 = np.zeros(2*H)
        row1[2*i:2*i+2] =  a12[i]                           # 对 u1_i
        row2[2*i:2*i+2] = -a12[i]                           # 对 u2_i
        rows_A1.append(row1); rows_A2.append(row2)
    rows_b.append(b12)

    A1 = np.vstack(rows_A1) if rows_A1 else np.zeros((0,2*H))
    A2 = np.vstack(rows_A2) if rows_A2 else np.zeros((0,2*H))
    b  = np.concatenate(rows_b, axis=0) if rows_b else np.zeros(0,)
    return A1, A2, b, h12

# ================== ADMM 求解（v-GNE / 共享乘子） ==================
def admm_two_agent(A1, A2, b, rho=10.0, max_admm=50, tol=1e-4,
                   u1_init=None, u2_init=None):
    """
    目标：min (1/2)||u1||^2 + (1/2)||u2||^2
    约束：A1 u1 + A2 u2 - b - s = 0,  s >= 0
          Σ := A1 u1 + A2 u2,  v = Σ
    三块ADMM：u并行、v闭式、s投影；乘子 y, μ。
    返回：u1,u2, 收敛信息
    """
    m = b.shape[0]
    n = 2*H
    u1 = np.zeros(n) if u1_init is None else u1_init.copy()
    u2 = np.zeros(n) if u2_init is None else u2_init.copy()
    v  = np.zeros(m)
    s  = np.zeros(m)
    y  = np.zeros(m)   # 乘子 for Σ - v = 0
    mu = np.zeros(m)   # 乘子 for v - b - s = 0
    eps_reg = 1e-8
    # 预分解（可提升效率）
    M1 = np.eye(n) + rho * (A1.T @ A1) + eps_reg * np.eye(n)
    M2 = np.eye(n) + rho * (A2.T @ A2) + eps_reg * np.eye(n)
    # 简单用np.linalg.solve；实际可用Cholesky
    for it in range(max_admm):
        # Σ^k
        Sigma = A1 @ u1 + A2 @ u2

        # u1-步、u2-步（并行可行）： (I + ρA_i^T A_i) u_i = ρ A_i^T r_i
        # r1 = v - y/ρ - A2 u2
        r1 = v - y/rho - (A2 @ u2)
        rhs1 = rho * (A1.T @ r1)
        u1 = np.linalg.solve(M1, rhs1)

        # r2 = v - y/ρ - A1 u1
        r2 = v - y/rho - (A1 @ u1)
        rhs2 = rho * (A2.T @ r2)
        u2 = np.linalg.solve(M2, rhs2)

        # 重新计算 Σ^{k+1}
        Sigma = A1 @ u1 + A2 @ u2

        # v-步（闭式）：v = 0.5 * ( Σ - y/ρ + b + s - μ/ρ )
        v_prev = v.copy()
        v = 0.5 * ( Sigma - y/rho + b + s - mu/rho )

        # s-步（投影）：s = [ v - b + μ/ρ ]_+
        s_prev = s.copy()
        s = np.maximum(0.0, v - b + mu/rho)

        # 乘子更新
        # y ← y + ρ (Σ - v)
        # μ ← μ + ρ (v - b - s)
        y  = y  + rho * (Sigma - v)
        mu = mu + rho * (v - b - s)

        # 残差与停止
        r_pri1 = np.linalg.norm(Sigma - v)
        r_pri2 = np.linalg.norm(v - b - s)
        r_pri = np.sqrt(r_pri1**2 + r_pri2**2)
        r_dual = rho * np.sqrt(np.linalg.norm(v - v_prev)**2 + np.linalg.norm(s - s_prev)**2)
        if r_pri < tol and r_dual < tol:
            break

    info = {'it': it+1, 'r_pri': r_pri, 'r_dual': r_dual}
    return u1.reshape(H,2), u2.reshape(H,2), info

# ================== 主仿真循环 ==================
traj_1 = np.linspace(x0_1, goal_1, H).copy()
traj_2 = np.linspace(x0_2, goal_2, H).copy()

mean_v1, mean_v2 = [], []
mean_u1, mean_u2 = [], []
min_h12_list = []
admm_it_hist = []

for k in range(K):
    t = k / K
    # 名义速度场（FM风格）
    v1 = nominal_velocity(traj_1, target_traj_1, t)
    v2 = nominal_velocity(traj_2, target_traj_2, t)
    # v1 = (target_traj_1 - traj_1)   # 名义速度场
    # v2 = (target_traj_2 - traj_2)   # 名义速度场


    # 构造约束 A1,A2,b（含静态障碍与互让）
    A1, A2, b, h12 = build_constraints_two_agents(traj_1, v1, traj_2, v2,
                                                  t, [], [], d_pair_eff)
    print(A1.shape, A2.shape, b.shape)
    # 若无约束则 u=0
    if b.size == 0:
        u1 = np.zeros_like(traj_1)
        u2 = np.zeros_like(traj_2)
        info = {'it': 0, 'r_pri': 0.0, 'r_dual': 0.0}
    else:
        # ADMM 求 v-GNE 的最小范数修正
        u1, u2, info = admm_two_agent(A1, A2, b, rho=0.1, max_admm=5, tol=1e-4)
        print(u1,u2)

    # 前向积分
    traj_1 = traj_1 + dt * (v1 + u1)
    traj_2 = traj_2 + dt * (v2 + u2)

    # 日志
    mean_v1.append(np.linalg.norm(v1, axis=1).mean())
    mean_v2.append(np.linalg.norm(v2, axis=1).mean())
    mean_u1.append(np.linalg.norm(u1, axis=1).mean())
    mean_u2.append(np.linalg.norm(u2, axis=1).mean())
    min_h12_list.append(h12.min())
    admm_it_hist.append(info['it'])

print(traj_1)
# 结果
trajectory_1 = traj_1
trajectory_2 = traj_2
mean_v1 = np.array(mean_v1); mean_v2 = np.array(mean_v2)
mean_u1 = np.array(mean_u1); mean_u2 = np.array(mean_u2)
min_h12_list = np.array(min_h12_list)
admm_it_hist  = np.array(admm_it_hist)

# ================== 可视化 ==================
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axs[0]
ax.plot(trajectory_1[:,0], trajectory_1[:,1], 'b.-', label='car 1')
ax.plot(trajectory_2[:,0], trajectory_2[:,1], 'g.-', label='car 2')
ax.plot([x0_1[0]], [x0_1[1]], 'bo', label='start1')
ax.plot([goal_1[0]], [goal_1[1]], 'bx', label='goal1')
ax.plot([x0_2[0]], [x0_2[1]], 'go', label='start2')
ax.plot([goal_2[0]], [goal_2[1]], 'gx', label='goal2')

# 画静态障碍+裕度
th = np.linspace(0, 2*np.pi, 200)
for j, obs in enumerate(obstacles):
    c = obs['c']; r_eff = obs['r_eff']
    ax.plot(c[0] + r_eff*np.cos(th), c[1] + r_eff*np.sin(th), 'r-', lw=2,
            label='obstacle+margin' if j==0 else None)

ax.axis('equal'); ax.grid(True); ax.legend()
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Two-car Safe Game Flow (ADMM v-GNE)')
plt.tight_layout(); plt.show()
# ax2 = axs[1]
# tt = np.arange(K) * dt
# ax2.plot(tt, mean_v1, label='mean |v1|')
# ax2.plot(tt, mean_v2, label='mean |v2|')
# ax2.plot(tt, mean_u1, label='mean |u1| (ADMM)')
# ax2.plot(tt, mean_u2, label='mean |u2| (ADMM)')
# ax2.plot(tt, min_h12_list, label='min h12')
# ax2.plot(tt, admm_it_hist/np.max(admm_it_hist +