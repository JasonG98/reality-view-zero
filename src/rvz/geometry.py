"""位姿、OpenCV 相机约定与投影工具（纯 numpy）。"""

import math

import numpy as np

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
CAM_ORDER = ("front", "left", "right", "back")
SIDE_CAMERAS = ("left", "right")
MARKER_LOCAL = (0.20, 0.0, 0.18)


def gl_cam_to_cv_pose(pos: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """从实际 GL 相机的位置、前向和上向量构造 4×4 ``T_W_Ccv``。"""
    pos = np.array(pos, dtype=np.float64, copy=True)
    forward = np.array(forward, dtype=np.float64, copy=True)
    up = np.array(up, dtype=np.float64, copy=True)
    assert pos.shape == (3,) and forward.shape == (3,) and up.shape == (3,)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(-forward, right)
    up /= np.linalg.norm(up)
    rotation_gl = np.column_stack([right, up, -forward])
    pose_gl = np.eye(4, dtype=np.float64)
    pose_gl[:3, :3] = rotation_gl
    pose_gl[:3, 3] = pos
    return (pose_gl @ GL_TO_CV).astype(np.float32)


def invert(T: np.ndarray) -> np.ndarray:
    """求刚体齐次变换的逆。"""
    assert T.shape == (4, 4)
    rotation = T[:3, :3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ T[:3, 3])
    return result.astype(np.float32)


def relative(T_W_a: np.ndarray, T_W_b: np.ndarray) -> np.ndarray:
    """返回 ``T_a_b = inv(T_W_a) @ T_W_b``。"""
    return (invert(T_W_a) @ T_W_b).astype(np.float32)


def marker_world(T_W_B: np.ndarray) -> np.ndarray:
    """车体朝向标记在世界系中的坐标 (3,)。"""
    return (T_W_B @ np.array([*MARKER_LOCAL, 1.0], dtype=np.float64))[:3]


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """按 MuJoCo 的垂直视场和像素中心约定构造 OpenCV 内参。"""
    focal = height / 2.0 / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def project(K: np.ndarray, T_C_W: np.ndarray, pts_W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将 ``pts_W`` (N,3) 投影为像素坐标 (N,2) 和正前方深度 (N,)。"""
    assert K.shape == (3, 3) and T_C_W.shape == (4, 4)
    points = np.asarray(pts_W, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 3
    homogeneous = np.column_stack([points, np.ones(len(points))])
    points_camera = (np.asarray(T_C_W, dtype=np.float64) @ homogeneous.T).T[:, :3]
    depth = points_camera[:, 2]
    safe_depth = np.where(np.abs(depth) > 1e-12, depth, np.nan)
    projected = points_camera @ np.asarray(K, dtype=np.float64).T
    return projected[:, :2] / safe_depth[:, None], depth


def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """图像缩放时同步缩放焦距和主点。"""
    assert K.shape == (3, 3)
    scaled = np.asarray(K, dtype=np.float32).copy()
    scaled[0] *= sx
    scaled[1] *= sy
    return scaled
