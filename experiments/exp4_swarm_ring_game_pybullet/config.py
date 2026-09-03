"""四机钻环集群博弈：同高 XY 方阵环 + 并列出发 + 穿环奖励"""

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
OUTPUT_DIR = EXP_DIR / "results"

NUM_AGENTS = 4
H = 72
K = 90
T_FLOW = 1.0
DT_FLOW = T_FLOW / K

D_SAFE = 1.2
PHI0 = 1.0
PHI1_SCALE = 1.0

# 环：同高 FLIGHT_Z，中心在 XY 平面 2×2 正方形，法向 +X
FLIGHT_Z = 3.0
RING_HALF = 2.8          # 环心距原点 XY 半间距 (m)
RING_RADIUS = 1.15
RING_TUBE = 0.10
RING_NORMAL = (1.0, 0.0, 0.0)
RING_PASS_TOL = 0.35     # 判定穿环：到环心距离阈值 (m)

RINGS = [
    {"id": 0, "name": "Ring-SW", "center": (-RING_HALF, -RING_HALF, FLIGHT_Z), "color": [0.92, 0.18, 0.18]},
    {"id": 1, "name": "Ring-SE", "center": (RING_HALF, -RING_HALF, FLIGHT_Z), "color": [0.18, 0.45, 0.95]},
    {"id": 2, "name": "Ring-NW", "center": (-RING_HALF, RING_HALF, FLIGHT_Z), "color": [0.12, 0.78, 0.32]},
    {"id": 3, "name": "Ring-NE", "center": (RING_HALF, RING_HALF, FLIGHT_Z), "color": [0.95, 0.62, 0.08]},
]

# 并列出发：环阵前方同一条线
START_X = -9.0
START_Y_SPACING = 3.5    # 四机 Y 方向等间距（并列）
GOAL_X = 11.0

_AGENT_COLORS = [
    [0.92, 0.18, 0.18],
    [0.18, 0.45, 0.95],
    [0.12, 0.78, 0.32],
    [0.95, 0.62, 0.08],
]
_START_YS = [
    -1.5 * START_Y_SPACING,
    -0.5 * START_Y_SPACING,
    0.5 * START_Y_SPACING,
    1.5 * START_Y_SPACING,
]

AGENTS = [
    {
        "name": f"UAV-{i + 1}",
        "color": _AGENT_COLORS[i],
        "start": (START_X, _START_YS[i], FLIGHT_Z),
        "goal": (GOAL_X, _START_YS[i], FLIGHT_Z),
        "ring_priority_offset": i,  # 博弈：各机优先环顺序错开
    }
    for i in range(NUM_AGENTS)
]

STATIC_OBSTACLES: list[dict] = []

# 奖励机制
REWARD_PER_RING = 100.0
REWARD_ALL_RINGS_BONUS = 80.0      # 四环全穿额外奖励
PENALTY_PATH_LENGTH = 0.35         # 每米路径代价
PENALTY_CONTROL = 2.0              # 流匹配修正强度惩罚（博弈成本）
PENALTY_UNSAFE = 200.0             # 违反 d_safe 惩罚

# 极简点质心无人机（尺寸远小于环直径 2.3m）
DRONE_RADIUS = 0.07
DRONE_VIS_RADIUS = 0.10            # 可视化略放大，仍 << 环
DRONE_MASS = 0.05

# PyBullet 一阶 PD 跟踪（极简动力学）
SIM_DT = 1.0 / 240.0
SIM_SUBSTEPS = 1
QUAD_MASS = DRONE_MASS
QUAD_KP = 22.0
QUAD_KD = 6.0
QUAD_MAX_FORCE = 8.0
GIF_FPS = 20
GIF_WIDTH = 1280
GIF_HEIGHT = 720
GIF_FRAME_STRIDE = 2
SIM_PLAYBACK_HZ = 22
SIM_TOTAL_TIME = 11.0
CAMERA_DISTANCE = 28.0
CAMERA_YAW = 55.0
CAMERA_PITCH = -35.0
CAMERA_TARGET = (0.0, 0.0, FLIGHT_Z)
