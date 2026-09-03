import matplotlib.pyplot as plt
import numpy as np

"""
SafeFlow/FMBF 最小例子（2D导航，单圆障碍）
- 构造手工速度场 v(x)：指向目标的单位方向
- 在每个积分步上，求最小范数修正 u，使得
    ∇h(x)^T (v + u) + φ(t, h(x)) h(x) >= 0
  成立（这是论文里 FMBF/CBF 线性不等式的特例）。
- 对单个状态的最小范数解有闭式：若 b <= 0 则 u*=0；否则 u* = (b / ||a||^2) a
  其中 a=∇h(x)，b = -a^T v - φ h(x)。

运行环境：numpy, matplotlib
"""

# ========== 场景参数 ==========
goal = np.array([2.5, 2.0])      # 目标位置
x0   = np.array([-2.5, -2.0])    # 初始位置
c    = np.array([1, 0.0])      # 障碍圆心
r    = 0.8                       # 障碍半径
margin = 0.2                     # 安全裕度
r_eff = r + margin               # 有效安全半径

# ========== 时间与φ函数参数 ==========
T = 1.0                          # 归一化时间区间 [0,1]
K = 100                          # 积分步数
dt = T / K
phi0 = 1.0                       # 安全区的 φ0（常数）
phi1_scale = 1.0                 # 不安全区 blow-up 基数
# 注意：φ1(t) 需要随 t->1 发散，且积分发散；这里用 1/(1-t)

# ========== 速度场参数 ==========
v_mag = 1.0                      # 速度场标称模长
v_clip = 1.0                     # 速度场限幅（可选）

# ========== 屏障函数及其梯度 ==========
def h(x):
    # h(x) = ||x - c||^2 - r_eff^2
    dx = x - c
    return dx @ dx - r_eff**2

def grad_h(x):
    # ∇h = 2(x - c)
    return 2.0 * (x - c)

def phi(t, h_val):
    """
    论文式的 φ 分段：
    - 若 h>=0（安全区）：φ = 常数 φ0
    - 若 h<0（不安全）：φ = φ1(t)，需要随 t->1 发散，且 ∫ φ1 = ∞
      这里取 φ1(t) = phi1_scale / (1 - t)，并做数值裁剪
    """
    if h_val >= 0:
        return phi0
    else:
        t_clip = min(t, 1.0 - 1e-6)
        return phi1_scale / (1.0 - t_clip)

# ========== 手工速度场 ==========
def v_field(x):
    """
    手工速度场：指向目标方向的单位向量，乘以常数模长 v_mag（再限幅）。
    你可以按需改成更“智能”的引导场。
    """
    d = goal - x
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.zeros(2)
    v = v_mag * (d / n)
    nv = np.linalg.norm(v)
    if nv > v_clip:
        v = v * (v_clip / nv)
    return v

# ========== FMBF 最小范数修正（闭式） ==========
def cbf_correction(x, t, v):
    """
    求解最小范数 u：
      a^T (v + u) + φ h >= 0
    令 a = ∇h(x)，b = -a^T v - φ h(x)，约束 a^T u >= b。
    最小范数解：
      若 b <= 0，则 u*=0（v 已满足约束）
      否则 u* = (b / ||a||^2) a
    """
    a = grad_h(x)
    a_norm2 = a @ a
    if a_norm2 < 1e-12:
        # 在圆心上这种退化情况（极少见），不修正
        return np.zeros(2)
    b = - a.dot(v) - phi(t, h(x)) * h(x)
    if b <= 0:
        return np.zeros(2)
    return (b / a_norm2) * a

# ========== 仿真主循环（SafeFlow-like 生成） ==========
xs = [x0.copy()]
us = []
vs = []
hs = [h(x0)]
ts = []

x = x0.copy()
for k in range(K):
    t = k / K  # 归一化“生成时间” t ∈ [0,1)

    v = v_field(x)
    u = cbf_correction(x, t, v)

    # 前向欧拉积分（也可换成更高阶积分器）
    x = x + dt * (v + u)

    xs.append(x.copy())
    us.append(u.copy())
    vs.append(v.copy())
    hs.append(h(x))
    ts.append(t)

xs = np.array(xs)
us = np.array(us)
vs = np.array(vs)
hs = np.array(hs)
ts = np.array(ts)

# ========== 结果可视化 ==========
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

# 轨迹 + 障碍
ax = axs[0]
ax.plot(xs[:,0], xs[:,1], 'k.-', label='trajectory')
ax.plot([x0[0]], [x0[1]], 'bo', label='start')
ax.plot([goal[0]], [goal[1]], 'rx', label='goal')

th = np.linspace(0, 2*np.pi, 200)
ax.plot(c[0] + r_eff*np.cos(th), c[1] + r_eff*np.sin(th), 'r-', lw=2, label='obstacle+margin')
ax.axis('equal')
ax.grid(True)
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Safe generation with FMBF (min-norm correction)')

# 信号（|v|, |u|, h）
ax2 = axs[1]
tt = np.arange(len(us)) * dt
ax2.plot(tt, np.linalg.norm(vs, axis=1), label='|v|')
ax2.plot(tt, np.linalg.norm(us, axis=1), label='|u| (CBF correction)')
ax2.plot(np.arange(len(hs))*dt, hs, label='h(x)')
ax2.axhline(0.0, color='k', lw=0.8)
ax2.grid(True)
ax2.legend()
ax2.set_xlabel('t')
ax2.set_title('Norms and barrier h')

plt.tight_layout()
plt.show()