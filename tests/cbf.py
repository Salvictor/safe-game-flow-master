import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


class CBFMPCBicycle:
    def __init__(self, dt=0.1, N=20, L=2.5,
                 Q=np.diag([2.0, 2.0, 0.2, 0.2]),
                 R=np.diag([0.1, 0.1]),
                 QN=np.diag([4.0, 4.0, 0.5, 0.5]),
                 v_bounds=(0.0, 5.0),
                 a_bounds=(-2.0, 2.0),
                 delta_bounds=(-0.5, 0.5),  # radians
                 cbf_slack_weight=1000.0,
                 obs_center=(0.0, 0.0), obs_radius=0.7, safety_margin=0.2,
                 gamma=0.3):
        """
        状态: [x, y, v, theta], 控制: [a, w] 其中 w = tan(delta)
        动力学: RK4 离散
        CBF: 离散时间一步约束 h_{k+1} - (1-gamma) h_k + s_k >= 0
        """
        self.dt = dt
        self.N = N
        self.L = L
        self.Q = Q
        self.R = R
        self.QN = QN
        self.v_bounds = v_bounds
        self.a_bounds = a_bounds
        self.delta_bounds = delta_bounds
        self.w_bounds = (np.tan(delta_bounds[0]), np.tan(delta_bounds[1]))  # w = tan(delta)
        self.cbf_slack_weight = cbf_slack_weight
        self.obs = np.array(obs_center, dtype=float)
        self.r_eff = obs_radius + safety_margin
        self.gamma = gamma

        self._build_solver()

    def _rk4(self, x, u):
        # x = [x, y, v, theta], u = [a, w], w=tan(delta)
        def f(x, u):
            x_pos, y_pos, v, th = x[0], x[1], x[2], x[3]
            a, w = u[0], u[1]
            dx = v * ca.cos(th)
            dy = v * ca.sin(th)
            dv = a
            dth = (v / self.L) * w
            return ca.vertcat(dx, dy, dv, dth)

        k1 = f(x, u)
        k2 = f(x + 0.5 * self.dt * k1, u)
        k3 = f(x + 0.5 * self.dt * k2, u)
        k4 = f(x + self.dt * k3, u)
        x_next = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next

    def _h_circle(self, x):
        dx = x[0] - self.obs[0]
        dy = x[1] - self.obs[1]
        return dx*dx + dy*dy - (self.r_eff**2)

    def _build_solver(self):
        nx, nu = 4, 2
        N = self.N

        X = ca.SX.sym('X', nx, N+1)
        U = ca.SX.sym('U', nu, N)  # [a, w]
        S = ca.SX.sym('S', N)      # slack for CBF

        # Parameters: initial state and reference states over horizon
        X0 = ca.SX.sym('X0', nx)
        Xref = ca.SX.sym('Xref', nx, N+1)
        Uref = ca.DM.zeros(nu, N)

        cost = 0
        g_list = []
        lbx = []
        ubx = []

        z_list = []

        # bounds for X
        for k in range(N+1):
            for i in range(nx):
                z_list.append(X[i, k])
                if i == 2:  # v bounds
                    lbx.append(self.v_bounds[0])
                    ubx.append(self.v_bounds[1])
                else:
                    lbx.append(-np.inf)
                    ubx.append(np.inf)

        # bounds for U
        for k in range(N):
            # a bounds
            z_list.append(U[0, k])
            lbx.append(self.a_bounds[0])
            ubx.append(self.a_bounds[1])
            # w bounds
            z_list.append(U[1, k])
            lbx.append(self.w_bounds[0])
            ubx.append(self.w_bounds[1])

        # bounds for Slack S (>= 0)
        for k in range(N):
            z_list.append(S[k])
            lbx.append(0.0)
            ubx.append(np.inf)

        z = ca.vertcat(*z_list)
        self.lbx = np.array(lbx, dtype=float)
        self.ubx = np.array([np.inf if np.isinf(v) else v for v in ubx], dtype=float)

        # Initial condition constraint
        g_list.append(X[:, 0] - X0)
        lbg = [0.0]*nx
        ubg = [0.0]*nx

        # Dynamics constraints, costs, and CBF constraints
        for k in range(N):
            # stage cost
            e = X[:, k] - Xref[:, k]
            du = U[:, k] - Uref[:, k]
            cost += ca.mtimes([e.T, self.Q, e]) + ca.mtimes([du.T, self.R, du]) + self.cbf_slack_weight * (S[k]**2)

            # dynamics via RK4
            X_next = self._rk4(X[:, k], U[:, k])
            g_list.append(X[:, k+1] - X_next)
            lbg += [0.0]*nx
            ubg += [0.0]*nx

            # Discrete CBF: h_{k+1} - (1-gamma) h_k + S_k >= 0
            h_k = self._h_circle(X[:, k])
            h_k1 = self._h_circle(X[:, k+1])
            g_list.append(h_k1 - (1.0 - self.gamma) * h_k + S[k])
            lbg += [0.0]
            ubg += [np.inf]

        # terminal cost
        eN = X[:, N] - Xref[:, N]
        cost += ca.mtimes([eN.T, self.QN, eN])

        g = ca.vertcat(*g_list)
        p = ca.vertcat(X0, ca.reshape(Xref, -1, 1))
        self.lbg = np.array(lbg, dtype=float)
        self.ubg = np.array([np.inf if np.isinf(v) else v for v in ubg], dtype=float)

        nlp = {'x': z, 'f': cost, 'g': g, 'p': p}
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 400,
            'ipopt.tol': 1e-4,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # cache sizes for unpacking
        self._nx = nx
        self._nu = nu
        self._nZ = int(z.numel())

    def solve(self, x0, xref_traj, z_init=None):
        # xref_traj: shape (nx, N+1)
        p = ca.vertcat(ca.DM(x0), ca.reshape(ca.DM(xref_traj), -1, 1))

        # initial guess
        if z_init is None:
            z0 = np.zeros(self._nZ)
            # set initial X guess to Xref, U and S stay zeros
            idx = 0
            for k in range(self.N + 1):
                z0[idx:idx + self._nx] = np.array(xref_traj[:, k]).flatten()
                idx += self._nx
        else:
            z0 = z_init

        sol = self.solver(x0=z0, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg, p=p)

        z_opt = np.array(sol['x']).squeeze()

        # Unpack
        idx = 0
        X_opt = np.zeros((self._nx, self.N + 1))
        for k in range(self.N + 1):
            X_opt[:, k] = z_opt[idx:idx + self._nx]
            idx += self._nx

        U_opt = np.zeros((self._nu, self.N))
        for k in range(self.N):
            U_opt[:, k] = z_opt[idx:idx + self._nu]
            idx += self._nu

        S_opt = z_opt[idx:idx + self.N]

        return X_opt, U_opt, S_opt, z_opt

    def simulate_step(self, x, u):
        # u = [a, w]
        x_ca = ca.DM(x)
        u_ca = ca.DM(u)
        x_next = np.array(self._rk4(x_ca, u_ca)).squeeze()
        return x_next

def build_reference(current_state, goal, v_ref, N):
    # 位置参考=goal，速度=v_ref，朝向=指向goal（简化）
    x, y, v, th = current_state
    dx = goal[0] - x
    dy = goal[1] - y
    th_ref = np.arctan2(dy, dx)
    Xref = np.zeros((4, N+1))
    Xref[0, :] = goal[0]
    Xref[1, :] = goal[1]
    Xref[2, :] = v_ref
    Xref[3, :] = th_ref
    return Xref

if __name__ == "__main__":
    # 控制器参数
    mpc = CBFMPCBicycle(
        dt=0.1, N=20, L=2.5,
        Q=np.diag([4.0, 4.0, 0.2, 0.2]),
        R=np.diag([0.2, 0.2]),
        QN=np.diag([6.0, 6.0, 0.3, 0.3]),
        v_bounds=(0.0, 4.0),
        a_bounds=(-1.5, 1.5),
        delta_bounds=(-0.6, 0.6),         # delta bounds (rad)
        cbf_slack_weight=1000.0,
        obs_center=(0.0, 0.0), obs_radius=0.8, safety_margin=0.2,
        gamma=0.3
    )

    # 初始与目标
    x0 = np.array([-5.0, -4.0, 1.0, 0.0])
    goal = np.array([1.5, 1.5])
    v_ref = 1.2

    # 仿真循环（闭环）
    x = x0.copy()
    traj = [x.copy()]
    ulog = []
    dlog = []   # delta = atan(w)
    slog = []
    z_init = None
    T_sim = 120

    openloop_list = []  # 保存每次优化得到的开环轨迹（用于画图）

    for t in range(T_sim):
        Xref = build_reference(x, goal, v_ref, mpc.N)
        X_opt, U_opt, S_opt, z_init = mpc.solve(x, Xref, z_init=z_init)

        # 记录开环
        openloop_list.append((X_opt.copy(), U_opt.copy()))

        # 应用第一步控制
        a0, w0 = U_opt[:, 0]
        delta0 = np.arctan(w0)
        ulog.append([a0, w0])
        dlog.append(delta0)
        slog.append(S_opt[0])

        x = mpc.simulate_step(x, [a0, w0])
        traj.append(x.copy())

        if np.linalg.norm(x[:2] - goal) < 0.2 and abs(x[2] - v_ref) < 0.3:
            break

    traj = np.array(traj)
    ulog = np.array(ulog)
    dlog = np.array(dlog)
    slog = np.array(slog)

    # 画图：轨迹与障碍
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axs[0]
    ax.plot(traj[:, 0], traj[:, 1], 'k.-', label='closed-loop traj')
    ax.plot([x0[0]], [x0[1]], 'bo', label='start')
    ax.plot([goal[0]], [goal[1]], 'rx', label='goal')

    # 画若干次开环预测（稀疏显示，避免太密）
    skip = max(1, len(openloop_list)//6)
    for i, (Xol, Uol) in enumerate(openloop_list[::skip]):
        ax.plot(Xol[0, :], Xol[1, :], 'g--', alpha=0.4)

    # 障碍 + 安全裕度
    th = np.linspace(0, 2*np.pi, 200)
    r_eff = mpc.r_eff
    obs = mpc.obs
    ax.plot(obs[0] + r_eff*np.cos(th), obs[1] + r_eff*np.sin(th), 'r-', lw=2, label='obstacle+margin')

    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('CBF-MPC Trajectory (discrete CBF)')

    # 画控制与松弛
    ax2 = axs[1]
    tgrid = np.arange(len(ulog)) * mpc.dt
    ax2.plot(tgrid, ulog[:, 0], label='a [m/s^2]')
    ax2.plot(tgrid, dlog, label='delta [rad]')
    ax2.plot(tgrid, slog, label='CBF slack')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlabel('time [s]')
    ax2.set_title('Inputs and CBF slack')

    plt.tight_layout()
    plt.show()