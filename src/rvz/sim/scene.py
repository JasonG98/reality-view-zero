"""仓库场景与车载相机的 MJCF 组装。"""

import json
import math
from collections.abc import Iterable
from dataclasses import replace
from functools import cache
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import mujoco
import numpy as np

from rvz.config import SimConfig
from rvz.geometry import CAM_ORDER, MARKER_LOCAL, SIDE_CAMERAS
from rvz.sim.layout import Box, Layout, Light, ProceduralColors, occupied_grid, required_path_length, sample_layout
from rvz.sim.rig import ROBOT_BODY_Z

MIN_OFFSCREEN_BUFFER = 640
_CAMERA_NEAR_SCALE = 0.001
_WAREHOUSE_CLEARANCE_M = 0.85
_WAREHOUSE_BOUNDS_GROUP = 3
_WAREHOUSE_ASSET = Path(__file__).with_name("assets") / "small_warehouse.zip"
_ROBOT_CAMERA_POSES = {
    "front": ((0.30, 0.0, 0.20), (0.0, -1.0, 0.0, 0.0, 0.0, 1.0)),
    "left": ((0.0, 0.20, 0.20), (1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    "right": ((0.0, -0.20, 0.20), (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    "back": ((-0.30, 0.0, 0.20), (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
}
_ROBOT_MATERIALS = """
<asset>
  <material name="robot_shell" rgba="0.82 0.86 0.87 1" specular="0.4" shininess="0.45"/>
  <material name="robot_trim" rgba="0.075 0.09 0.105 1" specular="0.22"/>
  <material name="robot_rubber" rgba="0.034 0.039 0.044 1" specular="0.04"/>
  <material name="robot_alloy" rgba="0.31 0.36 0.41 1" specular="0.45"/>
  <material name="robot_accent" rgba="0.94 0.32 0.055 1" specular="0.28"/>
  <material name="robot_glass" rgba="0.027 0.048 0.083 1" specular="0.65" shininess="0.6"/>
  <material name="robot_headlight" rgba="0.67 0.86 1 1" emission="0.45"/>
  <material name="robot_taillight" rgba="0.91 0.075 0.045 1" emission="0.45"/>
  <material name="marker_material" rgba="1 0.02 0.86 1" emission="1"/>
</asset>
"""
_ROBOT_DETAILS = """
<body>
  <freejoint/>
  <geom name="robot_bumper" type="mesh" mesh="robot_bumper_mesh" pos="0 0 -0.028" material="robot_trim"/>
  <geom name="robot_body" type="mesh" mesh="robot_body_mesh" pos="0 0 0.034" material="robot_shell"/>
  <geom name="robot_top_deck" type="mesh" mesh="robot_deck_mesh" pos="-0.024 0 0.091" material="robot_trim"/>
  <geom name="robot_front_trim" type="box" pos="0.284 0 0.018" size="0.006 0.106 0.012" material="robot_accent"/>
  <geom name="robot_rear_trim" type="box" pos="-0.284 0 0.018" size="0.006 0.106 0.012" material="robot_trim"/>
  <geom name="robot_front_camera_pod" type="box" pos="0.253 0 0.102" size="0.035 0.041 0.027" material="robot_trim"/>
  <geom name="robot_front_lens" type="cylinder" pos="0.29 0 0.108" size="0.017 0.005" xyaxes="0 1 0 0 0 1"
        material="robot_glass"/>
  <geom name="robot_rear_camera_pod" type="box" pos="-0.254 0 0.103" size="0.033 0.035 0.024" material="robot_trim"/>
  <geom name="robot_rear_lens" type="cylinder" pos="-0.288 0 0.109" size="0.013 0.005" xyaxes="0 1 0 0 0 1"
        material="robot_glass"/>
  <geom name="robot_beacon_base" type="cylinder" pos="0.2 0 0.105" size="0.049 0.023" material="robot_trim"/>
</body>
"""


def _numbers(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def _chamfered_box_mesh() -> tuple[str, str]:
    """生成带倒角的长方体顶点与面索引，三种车壳按比例复用同一份网格。"""
    corners = ((1, 0.8), (0.8, 1), (-0.8, 1), (-1, 0.8), (-1, -0.8), (-0.8, -1), (0.8, -1), (1, -0.8))
    rings = ((-1.0, 0.93), (-0.68, 1.0), (0.62, 1.0), (1.0, 0.90))
    vertices = _numbers(
        coordinate for height, profile in rings for x, y in corners for coordinate in (x * profile, y * profile, height)
    )
    faces = [(0, index + 1, index) for index in range(1, len(corners) - 1)]
    for ring in range(len(rings) - 1):
        first = ring * len(corners)
        for index in range(len(corners)):
            current = first + index
            following = first + (index + 1) % len(corners)
            faces.extend(
                (
                    (current, following, following + len(corners)),
                    (current, following + len(corners), current + len(corners)),
                )
            )
    last = (len(rings) - 1) * len(corners)
    faces.extend((last, last + index, last + index + 1) for index in range(1, len(corners) - 1))
    return vertices, _numbers(index for face in faces for index in face)


_ROBOT_MESH_VERTICES, _ROBOT_MESH_FACES = _chamfered_box_mesh()


def _with_robot(root: ElementTree.Element, cfg: SimConfig) -> str:
    visual, assets, worldbody = root.find("visual"), root.find("asset"), root.find("worldbody")
    assert visual is not None and assets is not None and worldbody is not None
    buffer_size = str(max(MIN_OFFSCREEN_BUFFER, cfg.robot_res, cfg.drone_res))
    ElementTree.SubElement(
        visual, "global", {"fovy": f"{cfg.fovy_deg:.6g}", "offwidth": buffer_size, "offheight": buffer_size}
    )
    # 近裁剪面按场景范围缩放，低机位广角需要保留紧邻车轮的地面。
    ElementTree.SubElement(visual, "map", {"znear": str(_CAMERA_NEAR_SCALE)})
    assets.extend(ElementTree.fromstring(_ROBOT_MATERIALS))
    for name, size in (
        ("robot_bumper_mesh", (0.297, 0.171, 0.046)),
        ("robot_body_mesh", (0.285, 0.185, 0.05)),
        ("robot_deck_mesh", (0.187, 0.131, 0.01)),
    ):
        ElementTree.SubElement(
            assets,
            "mesh",
            {"name": name, "vertex": _ROBOT_MESH_VERTICES, "face": _ROBOT_MESH_FACES, "scale": _numbers(size)},
        )

    robot = ElementTree.fromstring(_ROBOT_DETAILS)
    robot.attrib.update(name="robot", pos=f"0 0 {ROBOT_BODY_Z}")
    worldbody.append(robot)
    ElementTree.SubElement(
        robot,
        "geom",
        {
            "name": "heading_marker",
            "type": "sphere",
            "pos": _numbers(MARKER_LOCAL),
            "size": "0.055",
            "material": "marker_material",
        },
    )
    for side in (-1, 1):
        for end, x in (("front", 0.176), ("back", -0.176)):
            for part, offset, radius, half_width, material in (
                ("tire", 0.167, 0.066, 0.026, "robot_rubber"),
                ("rim", 0.192, 0.042, 0.004, "robot_alloy"),
                ("hub", 0.196, 0.015, 0.003, "robot_trim"),
            ):
                ElementTree.SubElement(
                    robot,
                    "geom",
                    {
                        "name": f"robot_{end}_{side}_{part}",
                        "type": "cylinder",
                        "pos": _numbers((x, side * offset, -0.034)),
                        "size": _numbers((radius, half_width)),
                        "xyaxes": "1 0 0 0 0 1",
                        "material": material,
                    },
                )
        for name, x, height, material in (
            ("headlight", 0.287, 0.050, "robot_headlight"),
            ("taillight", -0.287, 0.050, "robot_taillight"),
        ):
            ElementTree.SubElement(
                robot,
                "geom",
                {
                    "name": f"robot_{name}_{side}",
                    "type": "box",
                    "pos": _numbers((x, side * 0.116, height)),
                    "size": "0.006 0.035 0.007",
                    "material": material,
                },
            )
        ElementTree.SubElement(
            robot,
            "geom",
            {
                "name": f"robot_side_camera_{side}",
                "type": "box",
                "pos": _numbers((0, side * 0.163, 0.097)),
                "size": "0.035 0.022 0.020",
                "material": "robot_trim",
            },
        )
    for geom in robot.iter("geom"):
        geom.attrib.update(contype="0", conaffinity="0")
    for name in CAM_ORDER:
        position, xyaxes = _ROBOT_CAMERA_POSES[name]
        fovy = cfg.fovy_deg
        if name in SIDE_CAMERAS:
            tilt = math.radians(cfg.robot_side_down_deg) * (1.0 if name == "left" else -1.0)
            xyaxes = (*xyaxes[:3], 0.0, math.sin(tilt), math.cos(tilt))
            fovy = cfg.robot_side_fovy_deg
        ElementTree.SubElement(
            robot,
            "camera",
            {"name": name, "pos": _numbers(position), "xyaxes": _numbers(xyaxes), "fovy": f"{fovy:.6g}"},
        )
    return ElementTree.tostring(root, encoding="unicode")


def layout_to_mjcf(layout: Layout, colors: ProceduralColors, cfg: SimConfig) -> str:
    """生成含 4 路车体相机、自由相机参数和无天花板仓库的 MJCF。"""
    sky_top, sky_bottom = _numbers(colors.sky[0]), _numbers(colors.sky[1])
    floor_dark, floor_light = _numbers(colors.floor[0]), _numbers(colors.floor[1])
    wall_dark, wall_light = _numbers(colors.wall[0]), _numbers(colors.wall[1])
    lines = [
        '<mujoco model="warehouse">',
        "  <visual>",
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
    lines.extend(["  </worldbody>", "</mujoco>"])
    return _with_robot(ElementTree.fromstring("\n".join(lines)), cfg)


def build_scene(seed: int, cfg: SimConfig) -> tuple[Layout, mujoco.MjModel]:
    """按 cfg.scene 构造布局与 MJCF 模型。"""
    if cfg.scene == "small_warehouse":
        template, model = _warehouse_template(cfg)
        return replace(template, seed=seed), model
    layout, colors = sample_layout(seed, cfg)
    return layout, mujoco.MjModel.from_xml_string(layout_to_mjcf(layout, colors, cfg))


@cache
def _warehouse_template(cfg: SimConfig) -> tuple[Layout, mujoco.MjModel]:
    """仓库模型与布局只随 cfg 变化，按 cfg 缓存避免逐 episode 重复解压与编译。"""
    with ZipFile(_WAREHOUSE_ASSET) as archive:
        root = ElementTree.fromstring(archive.read("warehouse.xml"))
        navigation = json.loads(archive.read("navigation.json"))
        assets = {name: archive.read(name) for name in archive.namelist() if name.endswith((".obj", ".png"))}
    width, depth = navigation["size"]
    boxes = tuple(
        Box(
            name=name,
            kind="wall" if name.startswith("wall_") else "obstacle",
            center=np.asarray(bounds["center"], dtype=np.float32),
            size=np.asarray(bounds["size"], dtype=np.float32),
            rgba=np.ones(4, dtype=np.float32),
        )
        for name, bounds in navigation["boxes"].items()
    )
    worldbody = root.find("worldbody")
    assert worldbody is not None
    # 弹簧臂检测与位姿校验共用货架边界，保留货架空隙的真实可见性。
    for box in boxes:
        ElementTree.SubElement(
            worldbody,
            "geom",
            {
                "name": f"nav_{box.name}",
                "type": "box",
                "pos": _numbers(box.center),
                "size": _numbers(box.size / 2.0),
                "group": str(_WAREHOUSE_BOUNDS_GROUP),
                "contype": "0",
                "conaffinity": "0",
            },
        )
    model = mujoco.MjModel.from_xml_string(_with_robot(root, cfg), assets)
    clearance = max(cfg.robot_radius, _WAREHOUSE_CLEARANCE_M)
    layout = Layout(
        seed=0,
        width=width,
        depth=depth,
        boxes=boxes,
        lights=tuple(
            Light(model.light_pos[index], model.light_dir[index], model.light_diffuse[index])
            for index in range(model.nlight)
        ),
        grid_resolution=cfg.grid_resolution,
        robot_radius=clearance,
        occupied=occupied_grid(width, depth, cfg.grid_resolution, clearance, boxes),
        min_path_length=required_path_length(cfg),
    )
    return layout, model
