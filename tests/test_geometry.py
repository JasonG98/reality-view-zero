"""几何约定、相对位姿与投影的契约测试。"""

import numpy as np

from rvz.geometry import gl_cam_to_cv_pose, intrinsics_from_fovy, invert, project, relative, scale_K


def test_gl_camera_to_cv_identity() -> None:
    pos = np.zeros(3)
    forward = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, -1.0, 0.0])
    original_forward = forward.copy()
    pose = gl_cam_to_cv_pose(pos, forward, up)
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-6)
    np.testing.assert_array_equal(forward, original_forward)
    assert pose.dtype == np.float32


def test_gl_camera_to_cv_axes_and_translation() -> None:
    pose = gl_cam_to_cv_pose(
        pos=np.array([1.0, 2.0, 3.0]),
        forward=np.array([1.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0]),
    )
    expected_rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    np.testing.assert_allclose(pose[:3, :3], expected_rotation, atol=1e-6)
    np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 3.0])


def test_invert_and_relative_roundtrip() -> None:
    T_W_a = gl_cam_to_cv_pose([1.0, -2.0, 3.0], [0.0, 1.0, -0.2], [0.0, 0.0, 1.0])
    T_W_b = gl_cam_to_cv_pose([-1.0, 0.5, 2.0], [1.0, 0.0, -0.3], [0.0, 0.0, 1.0])
    T_a_b = relative(T_W_a, T_W_b)
    np.testing.assert_allclose(invert(T_W_a) @ T_W_b, T_a_b, atol=1e-6)
    np.testing.assert_allclose(invert(T_a_b) @ T_a_b, np.eye(4), atol=1e-6)


def test_intrinsics_and_project_known_points() -> None:
    K = intrinsics_from_fovy(90.0, 128, 128)
    points = np.array([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0], [0.0, -0.5, 2.0]])
    uv, depth = project(K, np.eye(4, dtype=np.float32), points)
    np.testing.assert_allclose(depth, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(uv, [[63.5, 63.5], [79.5, 63.5], [63.5, 47.5]], atol=1e-6)


def test_scale_K_scales_principal_point_and_focal_lengths() -> None:
    K = intrinsics_from_fovy(90.0, 128, 128)
    scaled = scale_K(K, 0.5, 0.25)
    np.testing.assert_allclose(scaled, K * np.array([0.5, 0.25, 1.0], dtype=np.float32)[:, None])
    np.testing.assert_allclose(scaled[2], [0.0, 0.0, 1.0])
