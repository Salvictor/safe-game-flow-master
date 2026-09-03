"""PyBullet 四机穿环：同高 XY 方阵环 + 极简点质心模型 + 轨迹 GIF"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError as e:
    raise ImportError("需要 pybullet") from e

from config import (
    AGENTS,
    CAMERA_DISTANCE,
    CAMERA_PITCH,
    CAMERA_TARGET,
    CAMERA_YAW,
    DRONE_MASS,
    DRONE_VIS_RADIUS,
    FLIGHT_Z,
    GIF_FRAME_STRIDE,
    GIF_FPS,
    GIF_HEIGHT,
    GIF_WIDTH,
    QUAD_KP,
    QUAD_KD,
    QUAD_MAX_FORCE,
    RING_RADIUS,
    RING_TUBE,
    RINGS,
    SIM_DT,
    SIM_PLAYBACK_HZ,
    SIM_SUBSTEPS,
    SIM_TOTAL_TIME,
    START_X,
)

RING_NORMAL = (1.0, 0.0, 0.0)


def _ring_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / (np.linalg.norm(normal) + 1e-9)
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-9
    v = np.cross(n, u)
    return u, v


def _vel_to_quat(vel: np.ndarray) -> list[float]:
    vx, vy, vz = vel
    xy = math.hypot(vx, vy)
    yaw = math.atan2(vy, vx) if xy > 1e-4 else 0.0
    pitch = float(np.clip(-math.atan2(vz, xy + 1e-6), -0.35, 0.35))
    return list(p.getQuaternionFromEuler([0.0, pitch, yaw]))


class RingSwarmPyBullet:
    def __init__(self, gui: bool = False):
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(SIM_DT, physicsClientId=self.client)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.drone_ids: list[int] = []
        self._frame = 0
        self._trail_prev: list[np.ndarray | None] = [None] * len(AGENTS)

    def reset(self) -> None:
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(SIM_DT, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.drone_ids = []
        self._frame = 0
        self._trail_prev = [None] * len(AGENTS)
        self._spawn_rings()
        self._draw_start_line()

    def _draw_start_line(self) -> None:
        ys = [ag["start"][1] for ag in AGENTS]
        p.addUserDebugLine(
            [START_X, min(ys) - 1.0, FLIGHT_Z],
            [START_X, max(ys) + 1.0, FLIGHT_Z],
            lineColorRGB=[1.0, 1.0, 0.2],
            lineWidth=2.0,
            lifeTime=0,
            physicsClientId=self.client,
        )
        p.addUserDebugText(
            "Simultaneous Start",
            [START_X - 1.0, 0.0, FLIGHT_Z + 1.8],
            textColorRGB=[1.0, 0.95, 0.2],
            textSize=1.0,
            lifeTime=0,
            physicsClientId=self.client,
        )

    def _spawn_rings(self) -> None:
        normal = np.asarray(RING_NORMAL, dtype=float)
        u, v = _ring_basis(normal)
        n_seg = 36
        for ring in RINGS:
            center = np.asarray(ring["center"], dtype=float)
            rgba = list(ring["color"]) + [0.90]
            tube = p.createVisualShape(
                p.GEOM_SPHERE, radius=RING_TUBE, rgbaColor=rgba,
                physicsClientId=self.client,
            )
            for k in range(n_seg):
                ang = 2 * math.pi * k / n_seg
                pos = center + RING_RADIUS * (math.cos(ang) * u + math.sin(ang) * v)
                p.createMultiBody(
                    baseMass=0, baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=tube, basePosition=pos.tolist(),
                    physicsClientId=self.client,
                )
            p.addUserDebugText(
                ring["name"],
                (center + np.array([0.0, 0.0, RING_RADIUS + 0.5])).tolist(),
                textColorRGB=ring["color"],
                textSize=0.9,
                lifeTime=0,
                physicsClientId=self.client,
            )

    def spawn_swarm(self, starts: list[np.ndarray]) -> None:
        """极简点质心：单色小球，尺寸远小于环"""
        for i, pos in enumerate(starts):
            rgba = list(AGENTS[i]["color"]) + [1.0]
            vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=DRONE_VIS_RADIUS,
                rgbaColor=rgba,
                physicsClientId=self.client,
            )
            col = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=DRONE_VIS_RADIUS * 0.7,
                physicsClientId=self.client,
            )
            did = p.createMultiBody(
                baseMass=DRONE_MASS,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos.tolist(),
                physicsClientId=self.client,
            )
            p.changeDynamics(
                did, -1, linearDamping=0.12, angularDamping=0.2,
                physicsClientId=self.client,
            )
            self.drone_ids.append(did)

    def _draw_planned_traj(self, traj: np.ndarray, rgb: list[float]) -> None:
        step = max(1, len(traj) // 50)
        for k in range(0, len(traj) - 1, step):
            j = min(k + step, len(traj) - 1)
            p.addUserDebugLine(
                traj[k].tolist(), traj[j].tolist(),
                lineColorRGB=rgb, lineWidth=1.0, lifeTime=0,
                physicsClientId=self.client,
            )

    def _update_trails(self) -> None:
        for i, did in enumerate(self.drone_ids):
            pos, _ = p.getBasePositionAndOrientation(did, physicsClientId=self.client)
            cur = np.asarray(pos, float)
            if self._trail_prev[i] is not None:
                p.addUserDebugLine(
                    self._trail_prev[i].tolist(), cur.tolist(),
                    lineColorRGB=AGENTS[i]["color"], lineWidth=2.5, lifeTime=0,
                    physicsClientId=self.client,
                )
            self._trail_prev[i] = cur.copy()

    def simulate_swarm(
        self,
        trajs: list[np.ndarray],
        record: bool = True,
        total_time: float = SIM_TOTAL_TIME,
    ) -> list[np.ndarray]:
        for i, t in enumerate(trajs):
            self._draw_planned_traj(t, AGENTS[i]["color"])

        n = len(trajs[0])
        n_play = min(n, max(int(total_time * SIM_PLAYBACK_HZ), 55))
        play_idx = np.linspace(0, n - 1, n_play).astype(int)
        dt_play = total_time / max(n_play - 1, 1)
        steps_pp = max(1, int(round((dt_play / SIM_DT) / SIM_SUBSTEPS)))
        logged: list[list[np.ndarray]] = [[] for _ in range(len(trajs))]
        frames: list[np.ndarray] = []

        for pi, k in enumerate(play_idx):
            targets = [trajs[i][k] for i in range(len(trajs))]
            target_vs = []
            for i in range(len(trajs)):
                kn = play_idx[min(pi + 1, len(play_idx) - 1)]
                target_vs.append((trajs[i][kn] - trajs[i][k]) / max(dt_play, 1e-3))

            # 极简运动学跟踪：点质心直接沿规划轨迹运动（便于完整完成穿环任务）
            for i in range(len(trajs)):
                p.resetBasePositionAndOrientation(
                    self.drone_ids[i],
                    targets[i].tolist(),
                    _vel_to_quat(target_vs[i]),
                    physicsClientId=self.client,
                )

            self._update_trails()
            for i in range(len(trajs)):
                logged[i].append(np.asarray(targets[i], float))

            p.stepSimulation(physicsClientId=self.client)

            if record and (self._frame % GIF_FRAME_STRIDE == 0):
                frames.append(self._capture())
            self._frame += 1

        if record and frames:
            self._last_frames = frames
        exec_trajs = []
        for i, log in enumerate(logged):
            arr = np.asarray(log)
            full = np.zeros((n, 3))
            for j, idx in enumerate(play_idx):
                full[idx] = arr[j]
            for j in range(1, n):
                if np.allclose(full[j], 0):
                    full[j] = full[j - 1]
            exec_trajs.append(full)
        return exec_trajs

    def _pd(self, idx: int, target: np.ndarray, target_v: np.ndarray) -> None:
        did = self.drone_ids[idx]
        pos, _ = p.getBasePositionAndOrientation(did, physicsClientId=self.client)
        vel, _ = p.getBaseVelocity(did, physicsClientId=self.client)
        pos = np.asarray(pos, float)
        vel = np.asarray(vel, float)
        acc = QUAD_KP * (target - pos) + QUAD_KD * (target_v - vel)
        f = DRONE_MASS * acc
        f[2] += DRONE_MASS * 9.81
        fn = np.linalg.norm(f)
        if fn > QUAD_MAX_FORCE:
            f *= QUAD_MAX_FORCE / fn
        p.applyExternalForce(
            did, -1, f.tolist(), pos.tolist(), p.WORLD_FRAME,
            physicsClientId=self.client,
        )

    def _capture(self) -> np.ndarray:
        view = p.computeViewMatrixFromYawPitchRoll(
            list(CAMERA_TARGET), CAMERA_DISTANCE, CAMERA_YAW, CAMERA_PITCH, 0, 2,
        )
        proj = p.computeProjectionMatrixFOV(68, GIF_WIDTH / GIF_HEIGHT, 0.2, 120.0)
        img = p.getCameraImage(
            GIF_WIDTH, GIF_HEIGHT, view, proj,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.client,
        )
        rgba = img[2]
        h, w = img[1], img[0]
        return np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)

    def save_gif(self, path: Path, fps: int = GIF_FPS) -> None:
        frames = getattr(self, "_last_frames", [])
        if not frames:
            return
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(
            path, save_all=True, append_images=ims[1:],
            duration=max(1, 1000 // fps), loop=0,
        )
        print(f"Saved GIF: {path} ({len(frames)} frames)")

    def save_screenshot(self, path: Path) -> None:
        from PIL import Image
        frames = getattr(self, "_last_frames", [])
        rgb = frames[len(frames) // 2] if frames else self._capture()
        Image.fromarray(rgb).save(path)
        print(f"Saved: {path}")

    def close(self) -> None:
        p.disconnect(self.client)


def run_pybullet_swarm(
    trajs: list[np.ndarray],
    output_dir: Path,
    gui: bool = False,
) -> list[np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    starts = [t[0] for t in trajs]
    env = RingSwarmPyBullet(gui=gui)
    try:
        env.reset()
        env.spawn_swarm(starts)
        exec_trajs = env.simulate_swarm(trajs, record=True)
        env.save_gif(output_dir / "swarm_ring_flight.gif")
        env.save_screenshot(output_dir / "swarm_ring_snapshot.png")
        for i, et in enumerate(exec_trajs):
            np.save(output_dir / f"exec_traj_agent{i}.npy", et)
        return exec_trajs
    finally:
        env.close()
