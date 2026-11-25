"""episode 渲染、重采样、schema 字段组装与多进程落盘。"""

import json
import logging
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections.abc import Generator
from dataclasses import asdict, dataclass, replace
from itertools import islice
from pathlib import Path

import mujoco
import numpy as np

from rvz.config import SCHEMA_VERSION, SimConfig
from rvz.geometry import (
    CAM_ORDER,
    SIDE_CAMERAS,
    gl_cam_to_cv_pose,
    intrinsics_from_fovy,
    invert,
    marker_world,
    project,
)
from rvz.sim.layout import Layout, path_is_free, plan_path
from rvz.sim.rig import (
    LOOKAT_HEIGHT_OFFSET_M,
    DroneTrajectory,
    drone_trajectories,
    drone_view_error,
    read_gl_camera,
    robot_camera_extrinsics,
    robot_poses,
    set_robot_pose,
)
from rvz.sim.scene import build_scene

logger = logging.getLogger(__name__)
_MARKER_SEARCH_RADIUS_PX = 8.0
_MARKER_SEARCH_MARGIN_PX = 2.0
_HIDDEN_GEOM_GROUP = mujoco.mjNGROUP - 1
_RAY_EPSILON_M = 1e-5
_MIN_CAMERA_DISTANCE_M = 0.45
_CAMERA_COLLISION_MARGIN_M = 0.05
_MAX_DISTANCE_STEP_M = 0.12
_PATH_SEED_OFFSET = 1_000_003
_DRONE_SEED_OFFSET = 2_000_033


@dataclass(frozen=True)
class _EpisodeScene:
    layout: Layout
    model: mujoco.MjModel
    simulation: mujoco.MjData
    robot_world_poses: np.ndarray
    drone_scene_option: mujoco.MjvOption


def _make_episode_scene(seed: int, cfg: SimConfig) -> _EpisodeScene:
    layout, model = build_scene(seed, cfg)
    path = plan_path(layout, np.random.default_rng(seed + _PATH_SEED_OFFSET))
    assert path_is_free(layout, path)
    robot_world_poses = robot_poses(path, cfg.steps, cfg.dt, cfg.robot_speed)
    drone_scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(drone_scene_option)
    drone_scene_option.geomgroup[_HIDDEN_GEOM_GROUP] = 0
    return _EpisodeScene(layout, model, mujoco.MjData(model), robot_world_poses, drone_scene_option)


def _git_hash() -> str | None:
    process = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else None


def marker_centroid(image: np.ndarray, expected_uv: np.ndarray, marker_radius_px: float = 0.0) -> np.ndarray | None:
    """在投影邻域内找发光标记，避免其他彩色箱体污染质心。"""
    radius = max(_MARKER_SEARCH_RADIUS_PX, marker_radius_px + _MARKER_SEARCH_MARGIN_PX)
    x, y = expected_uv
    start_x, stop_x = max(0, math.floor(x - radius)), min(image.shape[1], math.ceil(x + radius) + 1)
    start_y, stop_y = max(0, math.floor(y - radius)), min(image.shape[0], math.ceil(y + radius) + 1)
    rgb = image[start_y:stop_y, start_x:stop_x]
    # 小尺寸画面中的灯标可能只有部分像素覆盖，保留与周围明显不同的品红色。
    magenta = np.minimum(rgb[..., 0], rgb[..., 2]).astype(np.int16)
    green = rgb[..., 1]
    mask = (magenta > 120) & (green < 80) & (magenta - green > 64)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    xs, ys = xs + start_x, ys + start_y
    nearby = np.hypot(xs - x, ys - y) <= radius
    if not nearby.any():
        return None
    return np.array([xs[nearby].mean(), ys[nearby].mean()])


def _occluding_geoms(scene: _EpisodeScene, origin: np.ndarray, targets: np.ndarray) -> list[int]:
    """用射线找出相机到小车目标点之间可见的环境几何体。"""
    model, data = scene.model, scene.simulation
    visible_groups = scene.drone_scene_option.geomgroup
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
            distance = mujoco.mj_ray(model, data, start, direction, visible_groups, robot_body_id, 1, geom_id, None)
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


def _render_robot_frame(scene: _EpisodeScene, renderer: mujoco.Renderer, robot_rgb: np.ndarray) -> None:
    assert robot_rgb.shape == (len(CAM_ORDER), renderer.height, renderer.width, 3)
    for camera_name, image in zip(CAM_ORDER, robot_rgb, strict=True):
        renderer.update_scene(scene.simulation, camera=camera_name)
        renderer.render(out=image)


def _fit_drone_trajectory(
    scene: _EpisodeScene, renderer: mujoco.Renderer, trajectory: DroneTrajectory
) -> DroneTrajectory | None:
    """双向限幅使无人机在遇到障碍前便开始平滑缩短跟随距离。"""
    distances = trajectory.distance.astype(np.float64).copy()
    robot_body_id = scene.model.body("robot").id
    for step, T_W_B in enumerate(scene.robot_world_poses):
        set_robot_pose(scene.model, scene.simulation, T_W_B)
        camera = trajectory.camera(step)
        renderer.update_scene(scene.simulation, camera=camera)
        position = read_gl_camera(renderer)[0]
        origin = np.asarray(camera.lookat, dtype=np.float64).copy()
        vector = position - origin
        camera_distance = float(np.linalg.norm(vector))
        geom_id = np.array([-1], dtype=np.int32)
        hit_distance = mujoco.mj_ray(
            scene.model, scene.simulation, origin, vector / camera_distance, None, robot_body_id, 1, geom_id, None
        )
        if 0.0 < hit_distance < camera_distance:
            distances[step] = min(distances[step], hit_distance - _CAMERA_COLLISION_MARGIN_M)
    if np.any(distances < _MIN_CAMERA_DISTANCE_M):
        return None
    for step in range(1, len(distances)):
        distances[step] = min(distances[step], distances[step - 1] + _MAX_DISTANCE_STEP_M)
    for step in range(len(distances) - 2, -1, -1):
        distances[step] = min(distances[step], distances[step + 1] + _MAX_DISTANCE_STEP_M)
    return replace(trajectory, distance=distances)


def _render_drone_frame(
    scene: _EpisodeScene,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    T_W_B: np.ndarray,
    marker_world_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """遮挡物只对本次无人机渲染隐藏。"""
    renderer.update_scene(scene.simulation, camera=camera)
    position = read_gl_camera(renderer)[0]
    targets = np.asarray([marker_world_position, T_W_B[:3, 3] + [0.0, 0.0, LOOKAT_HEIGHT_OFFSET_M]])
    occluders = _occluding_geoms(scene, position, targets)
    original_groups = {geom_id: int(scene.model.geom_group[geom_id]) for geom_id in occluders}
    try:
        for geom_id in occluders:
            scene.model.geom_group[geom_id] = _HIDDEN_GEOM_GROUP
        renderer.update_scene(scene.simulation, camera=camera, scene_option=scene.drone_scene_option)
        image = renderer.render()
        T_W_C = gl_cam_to_cv_pose(*read_gl_camera(renderer))
    finally:
        for geom_id, group in original_groups.items():
            scene.model.geom_group[geom_id] = group
    return image, T_W_C


def _validate_drone_frame(
    frame: tuple[np.ndarray, np.ndarray],
    scene: _EpisodeScene,
    K_drone: np.ndarray,
    T_W_B: np.ndarray,
    marker_position: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """返回画面、实际位姿和有限的重投影误差，校验失败时误差为 None。"""
    image, T_W_C = frame
    height, width = image.shape[:2]
    view_error = drone_view_error(scene.layout, T_W_C, K_drone, T_W_B, width, height, cfg)
    if view_error is not None:
        logger.debug("无人机几何校验失败：%s", view_error)
        return image, T_W_C, None

    expected_uv, depth = project(K_drone, invert(T_W_C), marker_position[None])
    centroid_uv = None
    if depth[0] > 0.0:
        marker_radius_m = scene.model.geom("heading_marker").size[0]
        radius_px = K_drone[0, 0] * marker_radius_m / depth[0]
        centroid_uv = marker_centroid(image, expected_uv[0], radius_px)
    if centroid_uv is None:
        logger.debug("无人机隐藏遮挡物后仍看不到发光标记")
        return image, T_W_C, None

    reprojection_error = float(np.linalg.norm(centroid_uv - expected_uv[0]))
    return image, T_W_C, reprojection_error if math.isfinite(reprojection_error) else None


def _render_drone(
    drone_index: int,
    trajectory: DroneTrajectory,
    scene: _EpisodeScene,
    renderer: mujoco.Renderer,
    K_drone: np.ndarray,
    cfg: SimConfig,
    drone_buffers: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[DroneTrajectory, float] | None:
    trajectory = _fit_drone_trajectory(scene, renderer, trajectory)
    if trajectory is None:
        logger.debug("无人机 %d 正后方没有足够近的安全机位", drone_index)
        return None

    max_error = 0.0
    for step, T_W_B in enumerate(scene.robot_world_poses):
        set_robot_pose(scene.model, scene.simulation, T_W_B)
        marker_position = marker_world(T_W_B)
        frame = _render_drone_frame(scene, renderer, trajectory.camera(step), T_W_B, marker_position)
        image, T_W_C, reprojection_error = _validate_drone_frame(frame, scene, K_drone, T_W_B, marker_position, cfg)
        if reprojection_error is None:
            logger.debug("无人机 %d 第 %d 步校验失败", drone_index, step)
            return None
        max_error = max(max_error, reprojection_error)
        if drone_buffers is not None:
            images, poses = drone_buffers
            images[step] = image
            poses[step] = T_W_C
    return trajectory, max_error


def _validated_drone_trajectories(
    seed: int,
    scene: _EpisodeScene,
    renderer: mujoco.Renderer,
    K_drone: np.ndarray,
    cfg: SimConfig,
    *,
    episode_buffers: tuple[np.ndarray, np.ndarray] | None = None,
) -> Generator[tuple[DroneTrajectory, int, float], None, None]:
    rng = np.random.default_rng(seed + _DRONE_SEED_OFFSET)
    trajectories = drone_trajectories(rng, scene.layout, scene.robot_world_poses, cfg)
    for drone_index in range(cfg.n_drones):
        drone_buffers = (
            (episode_buffers[0][:, drone_index], episode_buffers[1][:, drone_index])
            if episode_buffers is not None
            else None
        )
        for attempt in range(cfg.max_resamples):
            if attempt > 0:
                # 后续无人机沿用最近一次抽取的候选批次，以保持录制的采样顺序。
                trajectories = drone_trajectories(rng, scene.layout, scene.robot_world_poses, cfg)
            result = _render_drone(drone_index, trajectories[drone_index], scene, renderer, K_drone, cfg, drone_buffers)
            if result is not None:
                trajectory, max_error = result
                yield trajectory, attempt, max_error
                break
        else:
            raise RuntimeError(f"无人机 {drone_index} 重采样 {cfg.max_resamples} 次后仍无有效轨迹")


def generate_episode(seed: int, cfg: SimConfig) -> dict[str, np.ndarray | str]:
    """生成一个确定性 episode，返回 SPEC §3.1 的完整字段。"""
    scene = _make_episode_scene(seed, cfg)
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
    resamples: list[int] = []
    max_reprojection_error = 0.0
    with (
        mujoco.Renderer(scene.model, cfg.robot_res, cfg.robot_res) as robot_renderer,
        mujoco.Renderer(scene.model, cfg.drone_res, cfg.drone_res) as drone_renderer,
    ):
        set_robot_pose(scene.model, scene.simulation, scene.robot_world_poses[0])
        T_B_Crobot = robot_camera_extrinsics(scene.model, scene.simulation, scene.robot_world_poses[0])
        for step, T_W_B in enumerate(scene.robot_world_poses):
            set_robot_pose(scene.model, scene.simulation, T_W_B)
            _render_robot_frame(scene, robot_renderer, robot_rgb[step])
        for _, attempt, drone_error in _validated_drone_trajectories(
            seed, scene, drone_renderer, K_drone, cfg, episode_buffers=(drone_rgb, drone_poses)
        ):
            resamples.append(attempt)
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
            "width": scene.layout.width,
            "depth": scene.layout.depth,
            "box_count": len(scene.layout.boxes),
            "light_count": len(scene.layout.lights),
        },
        "drone_resamples": resamples,
        "max_reprojection_error_px": max_reprojection_error,
    }
    episode: dict[str, np.ndarray | str] = {
        "robot_rgb": robot_rgb,
        "T_W_B": scene.robot_world_poses,
        "T_B_Crobot": T_B_Crobot,
        "K_robot": K_robot,
        "drone_rgb": drone_rgb,
        "T_W_Cdrone": drone_poses,
        "K_drone": K_drone,
        "meta_json": json.dumps(meta, ensure_ascii=False),
    }
    return episode


def live_camera_frames(
    seed: int, cfg: SimConfig, drone_index: int = 0
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """在同一仿真时刻渲染小车 (4,H,W,3) 与无人机 (Hd,Wd,3)，循环运行路径。"""
    if seed < 0:
        raise ValueError(f"seed 必须为非负整数：{seed}")
    if not 0 <= drone_index < cfg.n_drones:
        raise ValueError(f"drone_index 超出范围 [0, {cfg.n_drones})：{drone_index}")

    scene = _make_episode_scene(seed, cfg)
    K_drone = intrinsics_from_fovy(cfg.fovy_deg, cfg.drone_res, cfg.drone_res)
    with (
        mujoco.Renderer(scene.model, cfg.robot_res, cfg.robot_res) as robot_renderer,
        mujoco.Renderer(scene.model, cfg.drone_res, cfg.drone_res) as drone_renderer,
    ):
        # 先检查前序机位，重现录制时的重采样顺序。
        trajectories = _validated_drone_trajectories(seed, scene, drone_renderer, K_drone, cfg)
        trajectory, _, _ = next(islice(trajectories, drone_index, None))

        while True:
            for step, T_W_B in enumerate(scene.robot_world_poses):
                set_robot_pose(scene.model, scene.simulation, T_W_B)
                robot_rgb = np.empty((len(CAM_ORDER), cfg.robot_res, cfg.robot_res, 3), dtype=np.uint8)
                _render_robot_frame(scene, robot_renderer, robot_rgb)
                marker_position = marker_world(T_W_B)
                drone_rgb, _ = _render_drone_frame(
                    scene, drone_renderer, trajectory.camera(step), T_W_B, marker_position
                )
                yield robot_rgb, drone_rgb


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
