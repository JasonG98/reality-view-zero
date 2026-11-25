"""真实 MuJoCo 渲染、episode 录制与多进程落盘的集成测试。"""

import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from rvz.config import SimConfig
from rvz.data.schema import SCHEMA_VERSION, validate_episode
from rvz.geometry import intrinsics_from_fovy, invert, project
from rvz.sim.layout import plan_path, sample_layout
from rvz.sim.record import generate_episode, generate_many, live_camera_frames, marker_centroid
from rvz.sim.rig import (
    LOOKAT_HEIGHT_OFFSET_M,
    ROBOT_BODY_Z,
    drone_trajectories,
    robot_camera_extrinsics,
    robot_poses,
    set_robot_pose,
)
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
    layout, colors = sample_layout(0, cfg)
    xml = layout_to_mjcf(layout, colors, cfg)
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
    near_clip = model.vis.map.znear * model.stat.extent
    for camera_index, side in ((1, 1), (2, -1)):
        ground_points = np.array([[0.0, side * 0.30, -ROBOT_BODY_Z], [0.0, side * 1.0, -ROBOT_BODY_Z]])
        uv, depth = project(side_K, invert(camera_poses[camera_index]), ground_points)
        assert np.all(depth > near_clip)
        assert np.all(uv >= 0.0) and np.all(uv <= cfg.robot_res - 1)
    assert model.vis.global_.offwidth >= cfg.drone_res
    assert model.vis.global_.offheight >= cfg.drone_res
    assert all(model.geom(index).name != "ceiling" for index in range(model.ngeom))
    assert 1 <= len(layout.lights) <= 3


def test_drone_trajectories_are_strictly_behind() -> None:
    cfg = SimConfig(steps=8)
    layout, _ = sample_layout(4, cfg)
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


@pytest.mark.parametrize("drop_first_marker", [False, True])
def test_live_camera_frames_follow_episode_order_and_loop(
    monkeypatch: pytest.MonkeyPatch, drop_first_marker: bool
) -> None:
    cfg = replace(tiny_config(), drone_res=64)
    render = mujoco.Renderer.render
    drop_marker = drop_first_marker

    def render_with_marker_dropout(renderer: mujoco.Renderer, *, out: np.ndarray | None = None) -> np.ndarray:
        nonlocal drop_marker
        image = render(renderer, out=out)
        if drop_marker and renderer.width == cfg.drone_res:
            image.fill(0)
            drop_marker = False
        return image

    monkeypatch.setattr(mujoco.Renderer, "render", render_with_marker_dropout)
    episode = generate_episode(5, cfg)
    if drop_first_marker:
        assert json.loads(str(episode["meta_json"]))["drone_resamples"][0] == 1
    drop_marker = drop_first_marker
    drone_index = 1
    with closing(live_camera_frames(5, cfg, drone_index)) as frames:
        for step in (0, 1, 0):
            robot_rgb, drone_rgb = next(frames)
            np.testing.assert_array_equal(robot_rgb, episode["robot_rgb"][step])
            np.testing.assert_array_equal(drone_rgb, episode["drone_rgb"][step, drone_index])
    assert np.any(episode["robot_rgb"][0] != episode["robot_rgb"][1])
    assert np.any(episode["drone_rgb"][0, drone_index] != episode["drone_rgb"][1, drone_index])


def test_small_warehouse_renders_at_preview_resolution() -> None:
    cfg = replace(tiny_config(), scene="small_warehouse", robot_res=512, drone_res=768, n_drones=1)

    episode = generate_episode(0, cfg)

    validate_episode(episode)
    assert episode["robot_rgb"].shape == (2, 4, 512, 512, 3)
    assert episode["drone_rgb"].shape == (2, 1, 768, 768, 3)
    assert np.any(episode["robot_rgb"][0] != episode["robot_rgb"][1])
    meta = json.loads(str(episode["meta_json"]))
    assert meta["layout"]["depth"] == pytest.approx(20.907, abs=0.01)
    assert meta["max_reprojection_error_px"] <= 2.0


def test_small_warehouse_limits_rear_camera_at_open_shelving() -> None:
    cfg = replace(tiny_config(), scene="small_warehouse", drone_res=128, n_drones=1, max_resamples=4)

    episode = generate_episode(7, cfg)

    lookat = episode["T_W_B"][:, None, :3, 3] + [0.0, 0.0, LOOKAT_HEIGHT_OFFSET_M]
    distances = np.linalg.norm(episode["T_W_Cdrone"][..., :3, 3] - lookat, axis=-1)
    assert np.all(distances < cfg.drone_distance_min)
    validate_episode(episode)


def test_marker_centroid_rejects_bright_floor_pixels() -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[8:16, 14:20] = [190, 100, 190]  # 亮粉色地面，旧阈值会把它误并入质心。
    image[9:12, 9:12] = [255, 12, 255]
    centroid = marker_centroid(image, np.array([10.0, 10.0]))
    assert centroid is not None
    assert np.linalg.norm(centroid - [10.0, 10.0]) <= 1.0


def test_marker_centroid_detects_antialiased_beacon() -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[9:11, 11:13] = [143, 24, 157]
    image[8:16, 14:20] = [190, 100, 190]

    centroid = marker_centroid(image, np.array([11.5, 9.5]))

    np.testing.assert_array_equal(centroid, [11.5, 9.5])


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
    lookat = episode["T_W_B"][:, None, :3, 3] + np.array([0.0, 0.0, LOOKAT_HEIGHT_OFFSET_M], dtype=np.float32)
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
