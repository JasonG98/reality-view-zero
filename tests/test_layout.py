"""小型仓库布局有效性、A* 平滑路径与匀速运动学轨迹测试。"""

import numpy as np

from rvz.config import SimConfig
from rvz.sim.layout import path_is_free, plan_path, validate_layout
from rvz.sim.rig import robot_poses
from rvz.sim.scene import build_scene


def test_100_seeds_have_valid_layouts_and_collision_free_paths() -> None:
    cfg = SimConfig()
    layout, _ = build_scene(0, cfg)
    validate_layout(layout)
    for seed in range(100):
        path = plan_path(layout, np.random.default_rng(seed + 10_000))
        assert path.ndim == 2 and path.shape[1] == 2
        assert path_is_free(layout, path)
        assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() >= layout.min_path_length


def test_layout_and_path_are_deterministic() -> None:
    first, _ = build_scene(7, SimConfig())
    second, _ = build_scene(7, SimConfig())
    assert first.width == second.width
    assert first.depth == second.depth
    assert len(first.boxes) == len(second.boxes)
    np.testing.assert_array_equal(first.occupied, second.occupied)
    for first_box, second_box in zip(first.boxes, second.boxes, strict=True):
        assert first_box.name == second_box.name
        np.testing.assert_array_equal(first_box.center, second_box.center)
        np.testing.assert_array_equal(first_box.size, second_box.size)
    first_path = plan_path(first, np.random.default_rng(7))
    second_path = plan_path(second, np.random.default_rng(7))
    np.testing.assert_array_equal(first_path, second_path)


def test_robot_poses_move_at_constant_speed_and_follow_free_path() -> None:
    cfg = SimConfig(steps=12, dt=0.1, robot_speed=0.5)
    layout, _ = build_scene(3, cfg)
    path = plan_path(layout, np.random.default_rng(3))
    poses = robot_poses(path, cfg.steps, cfg.dt, cfg.robot_speed)
    assert poses.shape == (cfg.steps, 4, 4)
    assert poses.dtype == np.float32
    distances = np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1)
    np.testing.assert_allclose(distances, cfg.robot_speed * cfg.dt, atol=1e-5)
    assert path_is_free(layout, poses[:, :2, 3])
    np.testing.assert_allclose(poses[:, 3, :], np.tile([0.0, 0.0, 0.0, 1.0], (cfg.steps, 1)), atol=1e-6)


def test_planned_paths_have_smooth_turns_without_large_yaw_jumps() -> None:
    cfg = SimConfig(steps=40)
    layout, _ = build_scene(0, cfg)
    turning_episodes = 0
    for seed in range(50):
        path = plan_path(layout, np.random.default_rng(seed + 1_000_003))
        poses = robot_poses(path, cfg.steps, cfg.dt, cfg.robot_speed)
        step_distances = np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1)
        np.testing.assert_allclose(step_distances, cfg.robot_speed * cfg.dt, atol=1e-5)
        assert path_is_free(layout, poses[:, :2, 3])
        yaws = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
        dyaws = np.abs(np.arctan2(np.sin(np.diff(yaws)), np.cos(np.diff(yaws))))
        assert float(np.rad2deg(dyaws.max())) <= 35.0
        if float(np.rad2deg(dyaws.sum())) >= 15.0:
            turning_episodes += 1
    assert turning_episodes >= 30
