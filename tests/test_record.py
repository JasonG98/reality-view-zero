"""真实 MuJoCo 渲染、episode 录制与多进程落盘的集成测试。"""

import json
from pathlib import Path
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from rvz.config import SimConfig
from rvz.data.schema import SCHEMA_VERSION, validate_episode
from rvz.geometry import intrinsics_from_fovy, invert, project
from rvz.sim.layout import plan_path, sample_layout
from rvz.sim.record import _marker_centroid, generate_episode, generate_many
from rvz.sim.rig import ROBOT_BODY_Z, drone_trajectories, robot_camera_extrinsics, robot_poses, set_robot_pose
from rvz.sim.scene import layout_to_mjcf


def tiny_config() -> SimConfig:
    return SimConfig(
        robot_res=48,
        drone_res=48,
        steps=2,
        n_drones=3,
        warehouse_min_size=12.0,
        warehouse_max_size=12.0,
        min_aisles=3,
        max_aisles=3,
        max_resamples=30,
    )


def test_mjcf_contains_required_cameras_and_no_ceiling() -> None:
    cfg = tiny_config()
    layout = sample_layout(0, cfg)
    xml = layout_to_mjcf(layout, cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.ncam == 4
    assert [model.camera(index).name for index in range(4)] == ["front", "left", "right", "back"]
    assert model.vis.global_.fovy == cfg.fovy_deg
    np.testing.assert_allclose(
        model.cam_fovy,
        [cfg.fovy_deg, cfg.robot_side_fovy_deg, cfg.robot_side_fovy_deg, cfg.fovy_deg],
    )
    data = mujoco.MjData(model)
    robot_pose = np.eye(4, dtype=np.float32)
    robot_pose[2, 3] = ROBOT_BODY_Z
    set_robot_pose(model, data, robot_pose)
    camera_poses = robot_camera_extrinsics(model, data, robot_pose)
    tilt = np.deg2rad(cfg.robot_side_down_deg)
    np.testing.assert_allclose(camera_poses[1, :3, 2], [0.0, np.cos(tilt), -np.sin(tilt)], atol=1e-6)
    np.testing.assert_allclose(camera_poses[2, :3, 2], [0.0, -np.cos(tilt), -np.sin(tilt)], atol=1e-6)
    side_K = intrinsics_from_fovy(cfg.robot_side_fovy_deg, cfg.robot_res, cfg.robot_res)
    ground_points = np.array([[0.0, 1.0, -ROBOT_BODY_Z], [0.0, -1.0, -ROBOT_BODY_Z]])
    for camera_index, point in zip((1, 2), ground_points, strict=True):
        uv, depth = project(side_K, invert(camera_poses[camera_index]), point[None])
        assert depth[0] > 0.0
        assert np.all(uv[0] >= 0.0) and np.all(uv[0] <= cfg.robot_res - 1)
    assert model.vis.global_.offwidth >= cfg.drone_res
    assert model.vis.global_.offheight >= cfg.drone_res
    assert all(model.geom(index).name != "ceiling" for index in range(model.ngeom))
    assert 1 <= len(layout.lights) <= 3


def test_drone_trajectories_are_strictly_behind() -> None:
    cfg = SimConfig(steps=8)
    layout = sample_layout(4, cfg)
    path = plan_path(layout, np.random.default_rng(4))
    poses = robot_poses(path, cfg.steps, cfg.dt, cfg.robot_speed)
    trajectories = drone_trajectories(np.random.default_rng(8), layout, poses, cfg)
    assert len(trajectories) == cfg.n_drones
    assert len({float(trajectory.distance[0]) for trajectory in trajectories}) > 1
    expected_azimuth = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    for trajectory in trajectories:
        np.testing.assert_allclose(trajectory.azimuth, expected_azimuth, rtol=0.0, atol=1e-6)
        assert np.all(np.isfinite(trajectory.azimuth))
        assert np.all(trajectory.distance >= cfg.drone_distance_min)
        assert np.all(trajectory.distance <= cfg.drone_distance_max)
        assert np.all(np.rad2deg(trajectory.elevation) >= cfg.drone_elevation_min_deg)
        assert np.all(np.rad2deg(trajectory.elevation) <= cfg.drone_elevation_max_deg)


def test_generate_episode_schema_determinism_and_reprojection(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tiny_config()
    step = Mock(side_effect=AssertionError("运动学录制不得调用 mj_step"))
    monkeypatch.setattr(mujoco, "mj_step", step)
    first = generate_episode(5, cfg)
    validate_episode(first)
    second = generate_episode(5, cfg)
    assert step.call_count == 0
    for field in first:
        if field == "meta_json":
            assert first[field] == second[field]
        else:
            np.testing.assert_array_equal(first[field], second[field])
    meta = json.loads(str(first["meta_json"]))
    assert meta["schema_version"] == SCHEMA_VERSION
    expected_side_K = intrinsics_from_fovy(cfg.robot_side_fovy_deg, cfg.robot_res, cfg.robot_res)
    np.testing.assert_allclose(first["K_robot"][1], expected_side_K)
    np.testing.assert_allclose(first["K_robot"][2], expected_side_K)
    np.testing.assert_allclose(first["K_robot"][0], intrinsics_from_fovy(cfg.fovy_deg, cfg.robot_res, cfg.robot_res))
    assert meta["seed"] == 5
    assert meta["max_reprojection_error_px"] <= 2.0
    assert len(meta["drone_resamples"]) == cfg.n_drones
    assert all(0 <= count < cfg.max_resamples for count in meta["drone_resamples"])
    actual_poses = first["T_W_Cdrone"]
    assert np.all((actual_poses[..., 2, 3] >= cfg.drone_height_min) & (actual_poses[..., 2, 3] <= cfg.drone_height_max))
    downward_angles = np.rad2deg(np.arcsin(-actual_poses[..., 2, 2]))
    assert np.all(
        (downward_angles >= -cfg.drone_elevation_max_deg - 1.0)
        & (downward_angles <= -cfg.drone_elevation_min_deg + 1.0)
    )


def test_marker_centroid_rejects_bright_floor_pixels() -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[8:16, 14:20] = [190, 100, 190]  # 亮粉色地面，旧阈值会把它误并入质心。
    image[9:12, 9:12] = [255, 12, 255]
    centroid = _marker_centroid(image, np.array([10.0, 10.0]))
    assert centroid is not None
    assert np.linalg.norm(centroid - [10.0, 10.0]) <= 1.0


def test_seed_13_marker_centroid_rejects_bright_floor() -> None:
    # 旧阈值把 ep_000013 的亮粉色地面混入标记质心，最大误差曾达 5.175 px。
    cfg = SimConfig(robot_res=128, drone_res=64, steps=30)
    episode = generate_episode(13, cfg)
    meta = json.loads(str(episode["meta_json"]))
    assert meta["max_reprojection_error_px"] <= 2.0


def test_seed_6_hides_occluder_from_strict_rear_view() -> None:
    # 正后方机位在第 11 步被货架挡住，必须隐藏遮挡物而不是转向斜后方。
    episode = generate_episode(6, SimConfig(robot_res=128, drone_res=64, steps=30))
    meta = json.loads(str(episode["meta_json"]))
    assert meta["max_reprojection_error_px"] <= 2.0
    relative_xy = episode["T_W_Cdrone"][..., :2, 3] - episode["T_W_B"][:, None, :2, 3]
    forward_xy = episode["T_W_B"][:, None, :2, 0]
    cross = relative_xy[..., 0] * forward_xy[..., 1] - relative_xy[..., 1] * forward_xy[..., 0]
    along = np.sum(relative_xy * forward_xy, axis=-1)
    np.testing.assert_allclose(cross, 0.0, rtol=0.0, atol=2e-5)
    assert np.all(along < 0.0)


def test_seed_5_spring_arm_avoids_rear_obstacle() -> None:
    # 近低机位下固定距离会穿货架，弹簧臂必须提前平滑缩短且保持正后方。
    cfg = SimConfig(robot_res=128, drone_res=64, steps=30)
    episode = generate_episode(5, cfg)
    lookat = episode["T_W_B"][:, None, :3, 3] + np.array([0.0, 0.0, 0.15], dtype=np.float32)
    distances = np.linalg.norm(episode["T_W_Cdrone"][..., :3, 3] - lookat, axis=-1)
    assert distances.min() < cfg.drone_distance_min
    assert np.max(np.abs(np.diff(distances, axis=0))) <= 0.120001
    assert json.loads(str(episode["meta_json"]))["max_reprojection_error_px"] <= 2.0


def test_generate_many_uses_independent_workers(tmp_path: Path) -> None:
    cfg = SimConfig(
        robot_res=40,
        drone_res=40,
        steps=1,
        n_drones=1,
        warehouse_min_size=12.0,
        warehouse_max_size=12.0,
        min_aisles=3,
        max_aisles=3,
        max_resamples=30,
    )
    seeds = [101, 102]
    generate_many(seeds, cfg, tmp_path, workers=2)
    for seed in seeds:
        path = tmp_path / f"ep_{seed:06d}.npz"
        assert path.is_file()
        with np.load(path, allow_pickle=False) as episode:
            validate_episode(episode)
            meta = json.loads(str(episode["meta_json"].item()))
            assert meta["seed"] == seed


def test_generate_many_refuses_to_overwrite(tmp_path: Path) -> None:
    cfg = tiny_config()
    path = tmp_path / "ep_000005.npz"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="已存在"):
        generate_many([5], cfg, tmp_path, workers=1)
    assert path.read_bytes() == b"existing"
