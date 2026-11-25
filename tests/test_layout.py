"""布局有效性、A* 路径与匀速运动学轨迹测试。"""

import numpy as np

from rvz.config import SimConfig
from rvz.sim.layout import path_is_free, plan_path, sample_layout, validate_layout
from rvz.sim.rig import robot_poses


def test_100_seeds_have_valid_layouts_and_collision_free_paths() -> None:
    cfg = SimConfig()
    for seed in range(100):
        layout, _ = sample_layout(seed, cfg)
        validate_layout(layout)
        path = plan_path(layout, np.random.default_rng(seed + 10_000))
        assert path.ndim == 2 and path.shape[1] == 2
        assert path_is_free(layout, path)
        assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() >= layout.min_path_length


def test_layout_is_deterministic() -> None:
    first, first_colors = sample_layout(7, SimConfig())
    second, second_colors = sample_layout(7, SimConfig())
    assert first.width == second.width
    assert first.depth == second.depth
    assert len(first.boxes) == len(second.boxes)
    np.testing.assert_array_equal(first.occupied, second.occupied)
    np.testing.assert_array_equal(first_colors.floor, second_colors.floor)
    np.testing.assert_array_equal(first_colors.wall, second_colors.wall)
    np.testing.assert_array_equal(first_colors.sky, second_colors.sky)
    for first_box, second_box in zip(first.boxes, second.boxes, strict=True):
        assert first_box.name == second_box.name
        np.testing.assert_array_equal(first_box.center, second_box.center)
        np.testing.assert_array_equal(first_box.size, second_box.size)
        np.testing.assert_array_equal(first_box.rgba, second_box.rgba)


def test_robot_poses_move_at_constant_speed_and_follow_free_path() -> None:
    cfg = SimConfig(steps=12, dt=0.1, robot_speed=0.5)
    layout, _ = sample_layout(3, cfg)
    path = plan_path(layout, np.random.default_rng(3))
    poses = robot_poses(path, cfg.steps, cfg.dt, cfg.robot_speed)
    assert poses.shape == (cfg.steps, 4, 4)
    assert poses.dtype == np.float32
    distances = np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1)
    np.testing.assert_allclose(distances, cfg.robot_speed * cfg.dt, atol=1e-5)
    assert path_is_free(layout, poses[:, :2, 3])
    np.testing.assert_allclose(poses[:, 3, :], np.tile([0.0, 0.0, 0.0, 1.0], (cfg.steps, 1)), atol=1e-6)
