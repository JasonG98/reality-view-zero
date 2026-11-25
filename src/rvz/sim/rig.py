"""小车与无人机相机 rig；世界位姿均由实际 MuJoCo 状态读出。"""

import math
from dataclasses import dataclass
from itertools import product

import mujoco
import numpy as np

from rvz.config import SimConfig
from rvz.geometry import CAM_ORDER, MARKER_LOCAL, gl_cam_to_cv_pose, invert, project, relative
from rvz.sim.layout import Layout

ROBOT_BODY_Z = 0.1
ROBOT_HALF_EXTENTS = (0.30, 0.20, 0.10)
LOOKAT_HEIGHT_OFFSET_M = 0.15  # 无人机看向点相对小车中心抬高，落到车身中部


def robot_poses(path: np.ndarray, n_steps: int, dt: float, speed: float) -> np.ndarray:
    """按路径弧长匀速采样 ``T_W_B`` (T,4,4)，z 向上、yaw 跟随路径切线。"""
    points = np.asarray(path, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 2 and len(points) >= 1
    if n_steps == 1:
        pose = np.eye(4, dtype=np.float32)[None]
        pose[0, :2, 3] = points[0]
        pose[0, 2, 3] = ROBOT_BODY_Z
        return pose
    if len(points) < 2:
        raise ValueError("多步轨迹的路径至少需要两个点")
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-10):
        raise ValueError("路径不能含重复相邻点")
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    distances = np.arange(n_steps, dtype=np.float64) * speed * dt
    if distances[-1] > cumulative[-1] + 1e-6:
        raise ValueError(f"路径总长 {cumulative[-1]:.3f} m 小于轨迹所需 {distances[-1]:.3f} m")
    indices = np.clip(np.searchsorted(cumulative, distances, side="right") - 1, 0, len(segments) - 1)
    ratios = (distances - cumulative[indices]) / lengths[indices]
    positions = points[indices] + segments[indices] * ratios[:, None]
    tangents = segments[indices]
    yaws = np.arctan2(tangents[:, 1], tangents[:, 0])
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], n_steps, axis=0)
    poses[:, 0, 0] = np.cos(yaws)
    poses[:, 0, 1] = -np.sin(yaws)
    poses[:, 1, 0] = np.sin(yaws)
    poses[:, 1, 1] = np.cos(yaws)
    poses[:, :2, 3] = positions
    poses[:, 2, 3] = ROBOT_BODY_Z
    return poses.astype(np.float32)


@dataclass(frozen=True)
class DroneTrajectory:
    """内部角单位为弧度；仅在构造 MjvCamera 时转换为度。"""

    lookat: np.ndarray
    distance: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray

    def camera(self, step: int) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = self.lookat[step]
        camera.distance = float(self.distance[step])
        camera.azimuth = math.degrees(float(self.azimuth[step]))
        camera.elevation = math.degrees(float(self.elevation[step]))
        return camera


def drone_trajectories(
    rng: np.random.Generator, layout: Layout, robot_world_poses: np.ndarray, cfg: SimConfig
) -> list[DroneTrajectory]:
    """生成独立平滑轨道；位置与位姿有效性由渲染后的真实 GL 相机检查。"""
    times = np.arange(cfg.steps, dtype=np.float64) * cfg.dt
    target_headings = np.unwrap(np.arctan2(robot_world_poses[:, 1, 0], robot_world_poses[:, 0, 0]).astype(np.float64))
    trajectories = []
    for _ in range(cfg.n_drones):
        azimuth = target_headings

        distance_base = rng.uniform(cfg.drone_distance_min, cfg.drone_distance_max)
        distance = distance_base + 0.12 * np.sin(times * 0.8 + rng.uniform(0.0, 2.0 * math.pi))
        distance = np.clip(distance, cfg.drone_distance_min, cfg.drone_distance_max)
        elevation_base = math.radians(rng.uniform(cfg.drone_elevation_min_deg, cfg.drone_elevation_max_deg))
        elevation = elevation_base + 0.025 * np.sin(times * 0.6 + rng.uniform(0.0, 2.0 * math.pi))
        elevation = np.clip(
            elevation, math.radians(cfg.drone_elevation_min_deg), math.radians(cfg.drone_elevation_max_deg)
        )

        lookat = robot_world_poses[:, :3, 3].astype(np.float64).copy()
        lookat[:, 2] += LOOKAT_HEIGHT_OFFSET_M
        lookat[:, 0] = np.clip(lookat[:, 0], -layout.width / 2.0, layout.width / 2.0)
        lookat[:, 1] = np.clip(lookat[:, 1], -layout.depth / 2.0, layout.depth / 2.0)
        trajectories.append(
            DroneTrajectory(
                lookat=lookat.astype(np.float32),
                distance=distance.astype(np.float32),
                azimuth=azimuth.astype(np.float32),
                elevation=elevation.astype(np.float32),
            )
        )
    return trajectories


def set_robot_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: np.ndarray) -> None:
    """设置 free joint 后仅做运动学求值；四元数不离开 MuJoCo 内部状态。"""
    assert pose.shape == (4, 4) and model.nq == 7
    data.qpos[:3] = pose[:3, 3]
    quaternion = np.empty(4, dtype=np.float64)
    rotation = np.ascontiguousarray(pose[:3, :3], dtype=np.float64).reshape(9)
    mujoco.mju_mat2Quat(quaternion, rotation)
    data.qpos[3:7] = quaternion
    mujoco.mj_forward(model, data)


def read_gl_camera(renderer: mujoco.Renderer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取单目实际使用的双目平均相机，消除默认 IPD 造成的半眼距横移。"""
    left, right = renderer.scene.camera
    return (
        (np.asarray(left.pos, dtype=np.float64) + np.asarray(right.pos, dtype=np.float64)) / 2.0,
        (np.asarray(left.forward, dtype=np.float64) + np.asarray(right.forward, dtype=np.float64)) / 2.0,
        (np.asarray(left.up, dtype=np.float64) + np.asarray(right.up, dtype=np.float64)) / 2.0,
    )


def robot_camera_extrinsics(model: mujoco.MjModel, data: mujoco.MjData, T_W_B: np.ndarray) -> np.ndarray:
    """读取 4 路车体固定相机并返回 OpenCV 约定的 ``T_B_Crobot``。"""
    poses = []
    for name in CAM_ORDER:
        camera_id = model.camera(name).id
        rotation = np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)
        pose = gl_cam_to_cv_pose(data.cam_xpos[camera_id], -rotation[:, 2], rotation[:, 1])
        poses.append(relative(T_W_B, pose))
    return np.asarray(poses, dtype=np.float32)


def drone_view_error(
    layout: Layout,
    T_W_C: np.ndarray,
    K: np.ndarray,
    T_W_B: np.ndarray,
    width: int,
    height: int,
    cfg: SimConfig,
) -> str | None:
    """用实际世界位姿检查高度、AABB 穿模及完整小车和标记的视锥可见性。"""
    position = T_W_C[:3, 3]
    if not cfg.drone_height_min <= position[2] <= cfg.drone_height_max:
        return f"无人机高度 {position[2]:.3f} 超出范围"
    for box in layout.boxes:
        if np.all(np.abs(position - box.center) < box.size / 2.0):
            return f"无人机位于几何体 {box.name} 内"
    corners = [
        (sx * ROBOT_HALF_EXTENTS[0], sy * ROBOT_HALF_EXTENTS[1], sz * ROBOT_HALF_EXTENTS[2])
        for sx, sy, sz in product((-1.0, 1.0), repeat=3)
    ]
    local_points = np.asarray([*corners, MARKER_LOCAL], dtype=np.float64)
    homogeneous = np.column_stack([local_points, np.ones(len(local_points))])
    points_world = (T_W_B @ homogeneous.T).T[:, :3]
    uv, depth = project(K, invert(T_W_C), points_world)
    if np.any(depth <= 0.05):
        return "小车或朝向标记位于相机后方/近裁剪面"
    if (
        np.any(uv[:, 0] < 0.0)
        or np.any(uv[:, 0] > width - 1)
        or np.any(uv[:, 1] < 0.0)
        or np.any(uv[:, 1] > height - 1)
    ):
        return "小车或朝向标记超出视野"
    return None
