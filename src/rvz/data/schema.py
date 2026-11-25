"""SPEC §3.1 的 episode schema v2 与严格校验。"""

import json
from collections.abc import Mapping

import numpy as np

from rvz.config import SCHEMA_VERSION
from rvz.geometry import CAM_ORDER as CAM_ORDER

_REQUIRED_FIELDS = {
    "robot_rgb",
    "T_W_B",
    "T_B_Crobot",
    "K_robot",
    "drone_rgb",
    "T_W_Cdrone",
    "K_drone",
    "meta_json",
}
_LAST_ROW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _check_pose(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} 含非有限值")
    if not np.allclose(value[..., 3, :], _LAST_ROW, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} 的最后一行必须为 [0,0,0,1]")
    rotation = value[..., :3, :3].astype(np.float64)
    identity = np.eye(3)
    if not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, identity, rtol=0.0, atol=1e-4):
        raise ValueError(f"{name} 的旋转矩阵不正交（容差 1e-4）")


def _check_array(name: str, value: np.ndarray | str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        actual = getattr(value, "shape", None)
        raise ValueError(f"{name} 形状应为 {shape}，实际为 {actual}")
    if value.dtype != dtype:
        raise ValueError(f"{name} dtype 应为 {dtype}，实际为 {value.dtype}")
    return value


def validate_episode(ep: Mapping[str, np.ndarray | str]) -> None:
    """校验字段、形状、dtype、meta 和齐次位姿；失败时抛出 ValueError。"""
    missing = sorted(_REQUIRED_FIELDS - set(ep))
    unexpected = sorted(set(ep) - _REQUIRED_FIELDS)
    if missing:
        raise ValueError(f"episode 缺少字段：{', '.join(missing)}")
    if unexpected:
        raise ValueError(f"episode 含未知字段：{', '.join(unexpected)}")

    robot_rgb = ep["robot_rgb"]
    if not isinstance(robot_rgb, np.ndarray) or robot_rgb.ndim != 5 or robot_rgb.shape[1:2] != (4,):
        raise ValueError("robot_rgb 形状应为 [T,4,H,W,3]")
    if robot_rgb.shape[-1] != 3 or robot_rgb.shape[0] < 1:
        raise ValueError("robot_rgb 形状应为 [T,4,H,W,3]，且 T>=1")
    if robot_rgb.dtype != np.uint8:
        raise ValueError(f"robot_rgb dtype 应为 uint8，实际为 {robot_rgb.dtype}")
    T = robot_rgb.shape[0]

    drone_rgb = ep["drone_rgb"]
    if not isinstance(drone_rgb, np.ndarray) or drone_rgb.ndim != 5 or drone_rgb.shape[0] != T:
        raise ValueError("drone_rgb 形状应为 [T,M,Hd,Wd,3]，且 T 与 robot_rgb 一致")
    if drone_rgb.shape[1] < 1 or drone_rgb.shape[-1] != 3:
        raise ValueError("drone_rgb 形状应为 [T,M,Hd,Wd,3]，且 M>=1")
    if drone_rgb.dtype != np.uint8:
        raise ValueError(f"drone_rgb dtype 应为 uint8，实际为 {drone_rgb.dtype}")
    M = drone_rgb.shape[1]

    _check_pose("T_W_B", _check_array("T_W_B", ep["T_W_B"], (T, 4, 4), np.dtype(np.float32)))
    _check_pose("T_B_Crobot", _check_array("T_B_Crobot", ep["T_B_Crobot"], (4, 4, 4), np.dtype(np.float32)))
    _check_pose("T_W_Cdrone", _check_array("T_W_Cdrone", ep["T_W_Cdrone"], (T, M, 4, 4), np.dtype(np.float32)))
    _check_array("K_robot", ep["K_robot"], (4, 3, 3), np.dtype(np.float32))
    _check_array("K_drone", ep["K_drone"], (3, 3), np.dtype(np.float32))

    meta_value = ep["meta_json"]
    if isinstance(meta_value, np.ndarray):
        if meta_value.shape != () or meta_value.dtype.kind != "U":
            raise ValueError("meta_json 必须是标量字符串")
        meta_text = str(meta_value.item())
    elif isinstance(meta_value, str):
        meta_text = meta_value
    else:
        raise ValueError("meta_json 必须是标量字符串")
    try:
        meta = json.loads(meta_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"meta_json 不是有效 JSON：{exc.msg}") from exc
    if not isinstance(meta, dict) or meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"meta_json.schema_version 必须为 {SCHEMA_VERSION}")
