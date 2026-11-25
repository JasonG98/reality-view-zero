"""布局到 MJCF 的转换；只使用 MuJoCo 基本体和程序纹理。"""

import math
from collections.abc import Iterable

from rvz.config import SimConfig
from rvz.geometry import CAM_ORDER, MARKER_LOCAL, SIDE_CAMERAS
from rvz.sim.layout import Layout
from rvz.sim.rig import ROBOT_BODY_Z, ROBOT_HALF_EXTENTS

MIN_OFFSCREEN_BUFFER = 640
_ROBOT_CAMERA_POSES = {
    "front": ((0.30, 0.0, 0.20), (0.0, -1.0, 0.0, 0.0, 0.0, 1.0)),
    "left": ((0.0, 0.20, 0.20), (1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    "right": ((0.0, -0.20, 0.20), (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    "back": ((-0.30, 0.0, 0.20), (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
}


def _numbers(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def layout_to_mjcf(layout: Layout, cfg: SimConfig) -> str:
    """生成含 4 路车体相机、自由相机参数和无天花板仓库的 MJCF。"""
    buffer_size = max(MIN_OFFSCREEN_BUFFER, cfg.robot_res, cfg.drone_res)
    sky_top, sky_bottom = _numbers(layout.sky_colors[0]), _numbers(layout.sky_colors[1])
    floor_dark, floor_light = _numbers(layout.floor_colors[0]), _numbers(layout.floor_colors[1])
    wall_dark, wall_light = _numbers(layout.wall_colors[0]), _numbers(layout.wall_colors[1])
    lines = [
        '<mujoco model="warehouse">',
        "  <visual>",
        f'    <global fovy="{cfg.fovy_deg:.6g}" offwidth="{buffer_size}" offheight="{buffer_size}"/>',
        '    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>',
        "  </visual>",
        "  <asset>",
        f'    <texture name="sky" type="skybox" builtin="gradient" rgb1="{sky_top}" '
        f'rgb2="{sky_bottom}" width="128" height="128"/>',
        f'    <texture name="floor_texture" type="2d" builtin="checker" rgb1="{floor_dark}" '
        f'rgb2="{floor_light}" width="128" height="128"/>',
        '    <material name="floor_material" texture="floor_texture" texrepeat="8 8"/>',
        f'    <texture name="wall_texture" type="cube" builtin="checker" rgb1="{wall_dark}" '
        f'rgb2="{wall_light}" width="64" height="64"/>',
        '    <material name="wall_material" texture="wall_texture"/>',
        '    <texture name="box_texture" type="cube" builtin="checker" '
        'rgb1="0.95 0.95 0.95" rgb2="0.55 0.55 0.55" width="32" height="32"/>',
    ]
    for box in layout.boxes:
        if box.kind != "wall":
            lines.append(f'    <material name="mat_{box.name}" texture="box_texture" rgba="{_numbers(box.rgba)}"/>')
    lines.extend(
        [
            '    <material name="robot_material" rgba="0.12 0.14 0.18 1"/>',
            '    <material name="marker_material" rgba="1 0.02 1 1" emission="1"/>',
            "  </asset>",
            "  <worldbody>",
            f'    <geom name="floor" type="plane" pos="0 0 0" '
            f'size="{layout.width / 2.0:.6g} {layout.depth / 2.0:.6g} 0.1" '
            'material="floor_material" contype="0" conaffinity="0"/>',
        ]
    )
    for index, light in enumerate(layout.lights):
        lines.append(
            f'    <light name="light_{index}" pos="{_numbers(light.position)}" '
            f'dir="{_numbers(light.direction)}" diffuse="{_numbers(light.diffuse)}"/>'
        )
    for box in layout.boxes:
        material = "wall_material" if box.kind == "wall" else f"mat_{box.name}"
        lines.append(
            f'    <geom name="{box.name}" type="box" pos="{_numbers(box.center)}" '
            f'size="{_numbers(box.size / 2.0)}" material="{material}" contype="0" conaffinity="0"/>'
        )
    lines.extend(
        [
            f'    <body name="robot" pos="0 0 {ROBOT_BODY_Z}">',
            "      <freejoint/>",
            f'      <geom name="robot_body" type="box" pos="0 0 0" size="{_numbers(ROBOT_HALF_EXTENTS)}" '
            'material="robot_material" contype="0" conaffinity="0"/>',
            f'      <geom name="heading_marker" type="sphere" pos="{_numbers(MARKER_LOCAL)}" size="0.14" '
            'material="marker_material" contype="0" conaffinity="0"/>',
        ]
    )
    for name in CAM_ORDER:
        position, xyaxes = _ROBOT_CAMERA_POSES[name]
        fovy = cfg.fovy_deg
        if name in SIDE_CAMERAS:
            tilt = math.radians(cfg.robot_side_down_deg) * (1.0 if name == "left" else -1.0)
            xyaxes = (*xyaxes[:3], 0.0, math.sin(tilt), math.cos(tilt))
            fovy = cfg.robot_side_fovy_deg
        lines.append(
            f'      <camera name="{name}" pos="{_numbers(position)}" xyaxes="{_numbers(xyaxes)}" fovy="{fovy:.6g}"/>'
        )
    lines.extend(["    </body>", "  </worldbody>", "</mujoco>"])
    return "\n".join(lines)
