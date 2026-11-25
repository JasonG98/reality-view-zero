"""episode 渲染、重采样、schema 字段组装与多进程落盘。"""

import json
import logging
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from rvz.config import SCHEMA_VERSION, SimConfig
from rvz.geometry import CAM_ORDER, MARKER_LOCAL, SIDE_CAMERAS, gl_cam_to_cv_pose, intrinsics_from_fovy, invert, project
from rvz.sim.layout import Layout, path_is_free, plan_path, sample_layout
from rvz.sim.rig import (
    DroneTrajectory,
    drone_trajectories,
    drone_view_error,
    read_gl_camera,
    robot_camera_extrinsics,
    robot_poses,
    set_robot_pose,
)
from rvz.sim.scene import layout_to_mjcf

logger = logging.getLogger(__name__)
_MARKER_SEARCH_RADIUS_PX = 8.0
_HIDDEN_GEOM_GROUP = mujoco.mjNGROUP - 1
_RAY_EPSILON_M = 1e-5
_MIN_CAMERA_DISTANCE_M = 0.45
_CAMERA_COLLISION_MARGIN_M = 0.05
_MAX_DISTANCE_STEP_M = 0.12


def _git_hash() -> str | None:
    process = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else None


def _marker_centroid(image: np.ndarray, expected_uv: np.ndarray) -> np.ndarray | None:
    """在投影邻域内找发光标记，避免其他彩色箱体污染质心。"""
    rgb = image.astype(np.int32)
    mask = (rgb[..., 0] > 160) & (rgb[..., 2] > 160) & (rgb[..., 1] < 80)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    distances = np.hypot(xs - expected_uv[0], ys - expected_uv[1])
    nearby = distances <= _MARKER_SEARCH_RADIUS_PX
    if not nearby.any():
        return None
    return np.array([xs[nearby].mean(), ys[nearby].mean()])


def _occluding_geoms(model: mujoco.MjModel, data: mujoco.MjData, origin: np.ndarray, targets: np.ndarray) -> list[int]:
    """用射线找出相机到小车目标点之间的环境几何体，排除小车自身。"""
    robot_body_id = model.body("robot").id
    occluders: list[int] = []
    for target in targets:
        start = np.asarray(origin, dtype=np.float64).copy()
        vector = np.asarray(target, dtype=np.float64) - start
        remaining = float(np.linalg.norm(vector))
        if remaining <= _RAY_EPSILON_M:
            continue
        direction = vector / remaining
        for _ in range(2 * model.ngeom):
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, start, direction, None, robot_body_id, 1, geom_id, None)
            if distance < 0.0 or distance >= remaining - _RAY_EPSILON_M:
                break
            occluder = int(geom_id[0])
            if occluder not in occluders:
                occluders.append(occluder)
            advance = max(distance + _RAY_EPSILON_M, 10.0 * _RAY_EPSILON_M)
            start += direction * advance
            remaining -= advance
        else:
            raise RuntimeError("无人机遮挡射线检测未收敛")
    return occluders


def _render_drone(
    trajectory_index: int,
    trajectory: DroneTrajectory,
    layout: Layout,
    robot_world_poses: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    K_drone: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """按正后方轨迹渲染，平滑缩短避障，并仅对当前画面隐藏挡视线的几何体。"""
    images = np.empty((cfg.steps, cfg.drone_res, cfg.drone_res, 3), dtype=np.uint8)
    poses = np.empty((cfg.steps, 4, 4), dtype=np.float32)
    max_error = 0.0
    marker_local = np.array([*MARKER_LOCAL, 1.0])
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.geomgroup[_HIDDEN_GEOM_GROUP] = 0

    distances = trajectory.distance.astype(np.float64).copy()
    robot_body_id = model.body("robot").id
    for step in range(cfg.steps):
        set_robot_pose(model, data, robot_world_poses[step])
        camera = trajectory.camera(step)
        renderer.update_scene(data, camera=camera)
        position = read_gl_camera(renderer)[0]
        origin = np.asarray(camera.lookat, dtype=np.float64).copy()
        vector = position - origin
        camera_distance = float(np.linalg.norm(vector))
        geom_id = np.array([-1], dtype=np.int32)
        hit_distance = mujoco.mj_ray(
            model, data, origin, vector / camera_distance, None, robot_body_id, 1, geom_id, None
        )
        if 0.0 < hit_distance < camera_distance:
            distances[step] = min(distances[step], hit_distance - _CAMERA_COLLISION_MARGIN_M)
    if np.any(distances < _MIN_CAMERA_DISTANCE_M):
        logger.debug("无人机 %d 正后方没有足够近的安全机位", trajectory_index)
        return None
    for step in range(1, cfg.steps):
        distances[step] = min(distances[step], distances[step - 1] + _MAX_DISTANCE_STEP_M)
    for step in range(cfg.steps - 2, -1, -1):
        distances[step] = min(distances[step], distances[step + 1] + _MAX_DISTANCE_STEP_M)

    for step in range(cfg.steps):
        T_W_B = robot_world_poses[step]
        set_robot_pose(model, data, T_W_B)
        camera = trajectory.camera(step)
        camera.distance = float(distances[step])
        renderer.update_scene(data, camera=camera)
        position = read_gl_camera(renderer)[0]
        marker_world = (T_W_B @ marker_local)[:3]
        targets = np.asarray([marker_world, T_W_B[:3, 3] + [0.0, 0.0, 0.15]])
        occluders = _occluding_geoms(model, data, position, targets)
        original_groups = {geom_id: int(model.geom_group[geom_id]) for geom_id in occluders}
        try:
            for geom_id in occluders:
                model.geom_group[geom_id] = _HIDDEN_GEOM_GROUP
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            image = renderer.render()
            position, forward, up = read_gl_camera(renderer)
        finally:
            for geom_id, group in original_groups.items():
                model.geom_group[geom_id] = group
        T_W_C = gl_cam_to_cv_pose(position, forward, up)
        error = drone_view_error(layout, T_W_C, K_drone, T_W_B, cfg.drone_res, cfg.drone_res, cfg)
        if error is not None:
            logger.debug("无人机 %d 第 %d 步无效：%s", trajectory_index, step, error)
            return None
        expected_uv, depth = project(K_drone, invert(T_W_C), marker_world[None])
        centroid_uv = _marker_centroid(image, expected_uv[0]) if depth[0] > 0.0 else None
        if centroid_uv is None:
            logger.debug("无人机 %d 第 %d 步隐藏遮挡物后仍看不到发光标记", trajectory_index, step)
            return None
        reprojection_error = float(np.linalg.norm(centroid_uv - expected_uv[0]))
        if not math.isfinite(reprojection_error):
            return None
        max_error = max(max_error, reprojection_error)
        images[step] = image
        poses[step] = T_W_C
    return images, poses, max_error


def generate_episode(seed: int, cfg: SimConfig) -> dict[str, np.ndarray | str]:
    """生成一个确定性 episode，返回 SPEC §3.1 的完整字段。"""
    layout = sample_layout(seed, cfg)
    path = plan_path(layout, np.random.default_rng(seed + 1_000_003))
    assert path_is_free(layout, path)
    robot_world_poses = robot_poses(path, cfg.steps, cfg.dt, cfg.robot_speed)
    model = mujoco.MjModel.from_xml_string(layout_to_mjcf(layout, cfg))
    data = mujoco.MjData(model)
    K_robot = np.stack(
        [
            intrinsics_from_fovy(
                cfg.robot_side_fovy_deg if name in SIDE_CAMERAS else cfg.fovy_deg, cfg.robot_res, cfg.robot_res
            )
            for name in CAM_ORDER
        ]
    )
    K_drone = intrinsics_from_fovy(cfg.fovy_deg, cfg.drone_res, cfg.drone_res)
    robot_rgb = np.empty((cfg.steps, 4, cfg.robot_res, cfg.robot_res, 3), dtype=np.uint8)
    drone_rgb = np.empty((cfg.steps, cfg.n_drones, cfg.drone_res, cfg.drone_res, 3), dtype=np.uint8)
    drone_poses = np.empty((cfg.steps, cfg.n_drones, 4, 4), dtype=np.float32)
    resamples = [0] * cfg.n_drones
    max_reprojection_error = 0.0
    with (
        mujoco.Renderer(model, cfg.robot_res, cfg.robot_res) as robot_renderer,
        mujoco.Renderer(model, cfg.drone_res, cfg.drone_res) as drone_renderer,
    ):
        set_robot_pose(model, data, robot_world_poses[0])
        T_B_Crobot = robot_camera_extrinsics(model, data, robot_world_poses[0])
        for step in range(cfg.steps):
            set_robot_pose(model, data, robot_world_poses[step])
            for camera_index, camera_name in enumerate(CAM_ORDER):
                robot_renderer.update_scene(data, camera=camera_name)
                robot_rgb[step, camera_index] = robot_renderer.render()
        rng = np.random.default_rng(seed + 2_000_033)
        trajectories = drone_trajectories(rng, layout, robot_world_poses, cfg)
        for drone_index in range(cfg.n_drones):
            result = None
            for attempt in range(cfg.max_resamples):
                if attempt > 0:
                    trajectories = drone_trajectories(rng, layout, robot_world_poses, cfg)
                result = _render_drone(
                    drone_index,
                    trajectories[drone_index],
                    layout,
                    robot_world_poses,
                    model,
                    data,
                    drone_renderer,
                    K_drone,
                    cfg,
                )
                if result is not None:
                    resamples[drone_index] = attempt
                    break
            if result is None:
                raise RuntimeError(f"无人机 {drone_index} 重采样 {cfg.max_resamples} 次后仍无有效轨迹")
            drone_rgb[:, drone_index], drone_poses[:, drone_index], drone_error = result
            max_reprojection_error = max(max_reprojection_error, drone_error)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "dt": cfg.dt,
        "fovy": cfg.fovy_deg,
        "mujoco_version": mujoco.__version__,
        "platform": sys.platform,
        "git_hash": _git_hash(),
        "sim_config": asdict(cfg),
        "layout": {
            "width": layout.width,
            "depth": layout.depth,
            "box_count": len(layout.boxes),
            "light_count": len(layout.lights),
        },
        "drone_resamples": resamples,
        "max_reprojection_error_px": max_reprojection_error,
    }
    episode: dict[str, np.ndarray | str] = {
        "robot_rgb": robot_rgb,
        "T_W_B": robot_world_poses,
        "T_B_Crobot": T_B_Crobot,
        "K_robot": K_robot,
        "drone_rgb": drone_rgb,
        "T_W_Cdrone": drone_poses,
        "K_drone": K_drone,
        "meta_json": json.dumps(meta, ensure_ascii=False),
    }
    return episode


def _generate_worker(payload: tuple[int, SimConfig, str]) -> str:
    seed, cfg, out_dir = payload
    start = time.perf_counter()
    episode = generate_episode(seed, cfg)
    path = Path(out_dir) / f"ep_{seed:06d}.npz"
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **episode)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    elapsed = time.perf_counter() - start
    frame_count = cfg.steps * (4 + cfg.n_drones)
    logger.info(
        "seed=%d 已写入 %s，用时 %.2fs，渲染 %.1f 帧/s，最大重投影误差 %.3f px",
        seed,
        path,
        elapsed,
        frame_count / elapsed,
        json.loads(str(episode["meta_json"]))["max_reprojection_error_px"],
    )
    return str(path)


def generate_many(seeds: list[int], cfg: SimConfig, out_dir: str | Path, workers: int) -> None:
    """逐 episode 原子落盘；多进程使用 spawn 和独立渲染器。"""
    if workers < 1:
        raise ValueError("workers 必须 >= 1")
    if not seeds:
        raise ValueError("seeds 不能为空")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds 不能重复")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    existing = [seed for seed in seeds if (output / f"ep_{seed:06d}.npz").exists()]
    if existing:
        raise FileExistsError(f"episode 已存在，拒绝覆盖：{existing}")
    payloads = [(seed, cfg, str(output)) for seed in seeds]
    start = time.perf_counter()
    if workers == 1:
        for payload in payloads:
            _generate_worker(payload)
    else:
        context = mp.get_context("spawn")
        with context.Pool(min(workers, len(payloads))) as pool:
            list(pool.map(_generate_worker, payloads))
    elapsed = time.perf_counter() - start
    total_frames = len(seeds) * cfg.steps * (4 + cfg.n_drones)
    logger.info(
        "共生成 %d 个 episode，%d 帧，用时 %.2fs，整体 %.1f 帧/s",
        len(seeds),
        total_frames,
        elapsed,
        total_frames / elapsed,
    )
