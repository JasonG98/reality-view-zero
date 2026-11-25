"""episode 图片接触表与位姿重投影检查图。"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rvz.data.schema import CAM_ORDER, validate_episode
from rvz.geometry import MARKER_LOCAL, invert, project


def _cross(draw: ImageDraw.ImageDraw, uv: np.ndarray, color: str) -> None:
    x, y = float(uv[0]), float(uv[1])
    draw.line((x - 5, y, x + 5, y), fill=color, width=1)
    draw.line((x, y - 5, x, y + 5), fill=color, width=1)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=1)


def viz_episode(
    episode_path: str | Path, out_dir: str | Path, step: int = 0, drone_index: int = 0
) -> dict[str, Path]:
    """导出一个时刻的 4 路上下文、无人机图和重投影叠加图。"""
    with np.load(episode_path, allow_pickle=False) as loaded:
        episode = {name: loaded[name] for name in loaded.files}
    validate_episode(episode)
    robot_rgb = episode["robot_rgb"]
    drone_rgb = episode["drone_rgb"]
    if not 0 <= step < robot_rgb.shape[0]:
        raise ValueError(f"step 超出范围 [0, {robot_rgb.shape[0]})：{step}")
    if not 0 <= drone_index < drone_rgb.shape[1]:
        raise ValueError(f"drone_index 超出范围 [0, {drone_rgb.shape[1]})：{drone_index}")

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    height, width = robot_rgb.shape[-3:-1]
    tile_size = max(128, width, height)
    context = Image.new("RGB", (tile_size * 2, tile_size * 2), "black")
    for camera_index, camera_name in enumerate(CAM_ORDER):
        image = Image.fromarray(robot_rgb[step, camera_index])
        if image.size != (tile_size, tile_size):
            image = image.resize((tile_size, tile_size), Image.Resampling.NEAREST)
        x, y = (camera_index % 2) * tile_size, (camera_index // 2) * tile_size
        context.paste(image, (x, y))
        ImageDraw.Draw(context).text((x + 5, y + 5), camera_name, fill="white", stroke_width=1, stroke_fill="black")

    drone_array = drone_rgb[step, drone_index]
    drone = Image.fromarray(drone_array)
    display_size = max(256, context.height, drone.width, drone.height)
    scale = display_size / drone.width
    drone_display = drone.resize((display_size, display_size), Image.Resampling.NEAREST)
    reprojection = drone_display.copy()
    draw = ImageDraw.Draw(reprojection)
    T_W_B = episode["T_W_B"][step]
    T_W_C = episode["T_W_Cdrone"][step, drone_index]
    marker_world = (T_W_B @ np.array([*MARKER_LOCAL, 1.0]))[:3]
    marker_uv, _ = project(episode["K_drone"], invert(T_W_C), marker_world[None])
    robot_uv, _ = project(episode["K_drone"], invert(T_W_C), T_W_B[:3, 3][None])
    _cross(draw, marker_uv[0] * scale, "cyan")
    _cross(draw, robot_uv[0] * scale, "yellow")
    meta = json.loads(str(episode["meta_json"].item()))
    max_error = meta.get("max_reprojection_error_px")
    error_text = "n/a" if max_error is None else f"{max_error:.3f}px"
    draw.rectangle((0, display_size - 30, display_size, display_size), fill="black")
    draw.text((5, display_size - 27), "cyan=heading marker, yellow=robot center", fill="white")
    draw.text((5, display_size - 14), f"max reprojection error={error_text} (native px)", fill="white")

    header_height = 24
    overview = Image.new("RGB", (display_size * 3, display_size + header_height), (24, 24, 24))
    panels = (
        ("1. Robot cameras", context),
        (f"2. Drone ({drone.width}x{drone.height} -> {display_size})", drone_display),
        ("3. Pose check", reprojection),
    )
    overview_draw = ImageDraw.Draw(overview)
    for panel_index, (title, image) in enumerate(panels):
        x = panel_index * display_size
        overview_draw.text((x + 6, 6), title, fill="white")
        overview.paste(image, (x, header_height))
    outputs = {
        "context": output / "context.png",
        "drone": output / "drone.png",
        "reprojection": output / "reprojection.png",
        "overview": output / "overview.png",
    }
    context.save(outputs["context"])
    drone.save(outputs["drone"])
    reprojection.save(outputs["reprojection"])
    overview.save(outputs["overview"])
    return outputs
