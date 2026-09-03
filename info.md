# 用 ADMM 求解带共享 CBF 约束的广义纳什均衡（GNE）

以下给出用 ADMM 求解"带共享 CBF 约束的广义纳什均衡（GNE）"的一套从定义到算法的完整推导与实现配方。设场景是多车/多智能体安全流匹配：每个玩家 $i$ 在每个生成时刻 $t$ 最小化自己的正则控制范数，同时满足由 CBF 线性化得到的耦合安全不等式。

---

## 一、问题与均衡定义

**玩家集合**：$i = 1, \ldots, N$，决策 $u_i \in \mathbb{R}^{n_i}$（可为拼接的时域控制/速度修正变量）。

**目标函数（强凸且可分）**：$J_i(u_i)$（最常见即 $J_i(u_i) = \frac{1}{2}\|u_i\|^2$ 或加上本地约束 $U_i$ 的指示函数）。

**共享线性不等式约束**（由 CBF/FMBF 离线或在线线性化得到）：

$$Au \geq b$$

其中 $A = [A_1 \ \ldots \ A_N] \in \mathbb{R}^{m \times (\sum n_i)}$, $b \in \mathbb{R}^m$，$A_i$ 只对 $u_i$ 的分量起作用。

**广义纳什均衡（GNE）非正式定义**：不可单边改进且满足共享约束的策略剖分 $u^* = (u_1^*, \ldots, u_N^*)$。

**变分广义纳什均衡（v-GNE）定义（共享乘子 $\lambda$）**：存在 $\lambda \in \mathbb{R}_+^m$ 使对每个 $i$：

$$
\begin{cases}
0 \in \partial J_i(u_i^*) + A_i^T \lambda + N_{U_i}(u_i^*) \\
\lambda \geq 0, \quad Au^* - b \geq 0 \\
\lambda^T (Au^* - b) = 0
\end{cases}
$$

这正是"集中式凸规划 $\min \sum J_i$ s.t. $Au \geq b$"的 KKT 条件，与 v-GNE 等价（在凸假设下）。

---

## 二、由 CBF 到线性耦合约束 $Au \geq b$

以两车 $i, j$ 在每个离散点 $k$ 的互不碰撞为例（拼到整条轨迹即 $m = H \times$ 约束数）：

**障碍函数**：

$$h_{ij,k} = \|p_i(k) - p_j(k)\|^2 - d_{\text{safe}}^2$$

**线性化的 CBF/FMBF 约束**（名义速度 $v_i$，修正 $u_i$）：

$$\nabla h_{ij,k}^T (v_i(k) + u_i(k)) + \nabla h_{ij,k}^T(-v_j(k) - u_j(k)) + \varphi \cdot h_{ij,k} \geq 0$$

把所有 $i, j, k$ 叠起来，就得到 $Au \geq b$ 的仿射不等式（对静态圆障碍同理，$A_i$ 仅出现在其本车不等式中）。

---

## 三、引入松弛变量把不等式转为等式+锥约束

令 $s \in \mathbb{R}^m$，$s \geq 0$，且：

$$Au - b - s = 0$$

于是约束集合变为**等式 + 非负正交锥**，方便 ADMM 处理。

---

## 四、增广拉格朗日与共识分裂（并行友好）

为了让各玩家更新尽量并行，我们再引入共识变量 $v \approx Au$，使耦合拆分为两条等式：

$$
\begin{cases}
v = \sum_i A_i u_i \\
v - b - s = 0, \quad s \geq 0
\end{cases}
$$

**增广拉格朗日**（未缩放形式，$\rho > 0$）：

$$
\begin{aligned}
\mathcal{L}_\rho(\{u_i\}, v, s; y, \mu) = &\ \sum_i \left[ J_i(u_i) + I_{U_i}(u_i) \right] + I_{\mathbb{R}_+^m}(s) \\
&+ y^T \left( \sum_i A_i u_i - v \right) + \frac{\rho}{2} \left\| \sum_i A_i u_i - v \right\|^2 \\
&+ \mu^T (v - b - s) + \frac{\rho}{2} \|v - b - s\|^2
\end{aligned}
$$

其中 $y, \mu$ 为两条等式的乘子；$I_C$ 为集合的指示函数。

---

## 五、ADMM 迭代（2-等式-1-锥的三块体）

**初始化**：$u_i^0$、$v^0$、$s^0 \geq 0$、$y^0$、$\mu^0$。对 $k = 0, 1, 2, \ldots$ 迭代：

### 5.1 每个玩家的 $u_i$-步（并行）

给定 $\{u_j^k\}_{j \neq i}$, $v^k$, $y^k$，求解：

$$u_i^{k+1} = \underset{u_i \in U_i}{\arg\min} \ J_i(u_i) + \frac{\rho}{2} \|A_i u_i - r_i^k\|^2$$

其中：

$$r_i^k := v^k - \frac{y^k}{\rho} - \sum_{j \neq i} A_j u_j^k$$

**说明**：

这是强凸子问题；若 $J_i(u_i) = \frac{1}{2}\|u_i\|^2$ 且 $U_i = \mathbb{R}^{n_i}$，则一阶条件给出闭式：

$$(I + \rho A_i^T A_i) u_i^{k+1} = \rho A_i^T r_i^k$$

$$\Rightarrow \quad u_i^{k+1} = (I + \rho A_i^T A_i)^{-1} \rho A_i^T r_i^k$$

若有盒约束/范数约束，可在上式外再做投影或用近端步解决该强凸子问题。

**实现要点**：

$r_i^k$ 需要 $\sum_{j \neq i} A_j u_j^k$，可由"协调器/环形通信"广播：

$$\Sigma^k := \sum_j A_j u_j^k$$

后，本地用：

$$r_i^k = v^k - \frac{y^k}{\rho} - (\Sigma^k - A_i u_i^k)$$

得到。这样每个玩家只需一条标量向量 $\Sigma^k$ 的广播即可并行更新。

### 5.2 $v$-步（闭式）

给定 $u^{k+1}$, $s^k$, $y^k$, $\mu^k$，求解：

$$v^{k+1} = \underset{v}{\arg\min} \ \frac{\rho}{2}\left\|v - \Sigma^{k+1} + \frac{y^k}{\rho}\right\|^2 + \frac{\rho}{2}\left\|v - \left(b + s^k - \frac{\mu^k}{\rho}\right)\right\|^2$$

其中：

$$\Sigma^{k+1} := \sum_i A_i u_i^{k+1}$$

**闭式解（加权平均）**：

$$v^{k+1} = \frac{1}{2} \left[ \Sigma^{k+1} - \frac{y^k}{\rho} + b + s^k - \frac{\mu^k}{\rho} \right]$$

### 5.3 $s$-步（投影到非负正交锥）

$$s^{k+1} = \underset{s}{\arg\min} \ I_{\mathbb{R}_+^m}(s) + \frac{\rho}{2}\left\|v^{k+1} - b - s + \frac{\mu^k}{\rho}\right\|^2$$

$$\Rightarrow \quad s^{k+1} = \Pi_{\mathbb{R}_+^m} \left( v^{k+1} - b + \frac{\mu^k}{\rho} \right)$$

即**分量非负投影**：

$$s^{k+1} = \max\left(0, v^{k+1} - b + \frac{\mu^k}{\rho}\right)$$

### 5.4 乘子更新（原-对偶残差）

$$y^{k+1} = y^k + \rho \left( \Sigma^{k+1} - v^{k+1} \right)$$

$$\mu^{k+1} = \mu^k + \rho \left( v^{k+1} - b - s^{k+1} \right)$$

### 5.5 停止准则

**原始残差**：

$$r_p^k = \begin{bmatrix} \Sigma^k - v^k \\ v^k - b - s^k \end{bmatrix}$$

**对偶残差**：

$$r_d^k = \rho \begin{bmatrix} v^k - v^{k-1} \\ s^k - s^{k-1} \end{bmatrix}$$

当 $\|r_p^k\|$ 和 $\|r_d^k\|$ 足够小则停止。

---

## 六、与 v-GNE/KKT 的关系与正确性

- $s$-步的最优性条件是 $0 \in \partial I_{\mathbb{R}_+}(s^*) - \mu^*$，等价于 $\mu^* \in N_{\mathbb{R}_+}(s^*)$，即 $\mu^* \geq 0$ 且 $\mu^* \perp s^*$。

- 等式 $v^* = \Sigma^*$ 与 $v^* - b - s^* = 0$ 联立给出 $s^* = \Sigma^* - b = Au^* - b$。

- 因此 $\mu^* \perp s^*$ 等价于 $\mu^* \perp (Au^* - b)$，且 $\mu^* \geq 0$，这正是 v-GNE 的互补条件（$\mu^*$ 扮演共享乘子 $\lambda^*$）。

- 每个 $u_i$-步的一阶条件在极限点给出：

$$\nabla J_i(u_i^*) + A_i^T \mu^* + \xi_i = 0, \quad \xi_i \in N_{U_i}(u_i^*)$$

与 v-GNE 驻点条件一致。

- 在标准凸假设（$J_i$ 强凸，$U_i$ 闭凸，$A$ 满足常规正则性）与合适的 $\rho$ 下，ADMM 收敛到 v-GNE（等价集中式 KKT 解）。

---

## 七、与 SafeFlow/CBF 的对接

在 SafeFlow 的每个生成时间 $t$，你会得到一批线性化 CBF/FMBF 约束（对每条轨迹片段/每个障碍/每对车辆）。

把这些约束拼成 $Au \geq b$，即可直接套上面的 ADMM 内循环做"**博弈式安全滤波**"。

每个 ADMM 外层迭代内做少量（如 5–20 次）$u/v/s/y/\mu$ 的更新，一旦原-对偶残差足够小，即得到此时刻的 v-GNE 修正（分布式实现），再推进轨迹状态。

---

## 八、二次目标的闭式本地步（常用特例）

若 $J_i(u_i) = \frac{1}{2}\|u_i\|^2$ 且无本地约束 $U_i = \mathbb{R}^{n_i}$，则：

$$u_i^{k+1} = (I + \rho A_i^T A_i)^{-1} \rho A_i^T r_i^k$$

$$r_i^k = v^k - \frac{y^k}{\rho} - (\Sigma^k - A_i u_i^k)$$

若有盒约束 $U_i = [l_i, u_i]$，则可先算上述无约束解，再投影到盒子；或在 $u_i$-步里用近端/坐标下降。

---

## 九、实现提示与调参

- **$\rho$ 的选择**：
  - 过小：收敛慢
  - 过大：原始残差快、对偶残差振荡
  - 经验上与 $\|A_i\|$ 缩放相匹配

- **并行/通信**：每轮只需要广播 $\Sigma^k$ 和两个乘子向量 $y^k$, $\mu^k$（或其缩放版），通信开销低。

- **不等式激活**：ADMM 自动处理激活/非激活（通过 $s$ 的投影与 $\mu$ 的非负性），无需显式主动集。

- **数值稳定**：若 $A_i^T A_i$ 条件数大，可预先做列尺度化；或在 $u_i$-步用共轭梯度解线性系统。

---

## 十、小结（从定义到步骤）

1. **定义**：v-GNE = 集中式 KKT（共享 $\lambda$）。CBF 线性化给出 $Au \geq b$。

2. **变换**：引入 $s \geq 0$ 与 $v$，使耦合不等式化为两个等式与一组锥约束。

3. **算法**：三块 ADMM（$u$ 并行、$v$ 闭式、$s$ 投影），乘子 $y, \mu$ 做原-对偶更新。

4. **收敛**：凸/强凸设定下收敛到 v-GNE。对 SafeFlow，可作为每个生成时间步的分布式"安全滤波器"。

---

## 附注

如果你愿意，我可以基于你两车的那段代码，直接把上述 ADMM 的每步更新实现为可运行的 Python（并提供每步的 $\Sigma$、$v$、$s$、$y$、$\mu$ 更新与终止判据）。
