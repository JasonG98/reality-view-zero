"""episode schema v2 的字段、形状、dtype 与位姿校验测试。"""

import json
from pathlib import Path

import numpy as np
import pytest

from rvz.data.schema import SCHEMA_VERSION, validate_episode


def valid_episode() -> dict[str, np.ndarray | str]:
    T, M, H, W, Hd, Wd = 2, 3, 8, 10, 6, 4
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], T, axis=0)
    drone_poses = np.repeat(np.eye(4, dtype=np.float32)[None, None], T, axis=0)
    drone_poses = np.repeat(drone_poses, M, axis=1)
    return {
        "robot_rgb": np.zeros((T, 4, H, W, 3), dtype=np.uint8),
        "T_W_B": poses,
        "T_B_Crobot": np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0),
        "K_robot": np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0),
        "drone_rgb": np.zeros((T, M, Hd, Wd, 3), dtype=np.uint8),
        "T_W_Cdrone": drone_poses,
        "K_drone": np.eye(3, dtype=np.float32),
        "meta_json": json.dumps({"schema_version": SCHEMA_VERSION, "seed": 0}),
    }


def test_valid_episode_and_npz_roundtrip(tmp_path: Path) -> None:
    episode = valid_episode()
    validate_episode(episode)
    path = tmp_path / "episode.npz"
    np.savez(path, **episode)
    with np.load(path, allow_pickle=False) as loaded:
        validate_episode(loaded)


def test_missing_and_unexpected_fields_rejected() -> None:
    episode = valid_episode()
    episode.pop("K_robot")
    with pytest.raises(ValueError, match="缺少字段"):
        validate_episode(episode)
    episode = valid_episode()
    episode["extra"] = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError, match="未知字段"):
        validate_episode(episode)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("robot_rgb", np.zeros((2, 4, 8, 10, 3), dtype=np.float32), "robot_rgb.*uint8"),
        ("T_W_B", np.zeros((1, 4, 4), dtype=np.float32), "T_W_B.*形状"),
        ("T_B_Crobot", np.eye(4, dtype=np.float64), "T_B_Crobot.*形状"),
        ("K_drone", np.eye(3, dtype=np.float64), "K_drone.*float32"),
        ("K_robot", np.eye(3, dtype=np.float32), "K_robot.*形状"),
        ("drone_rgb", np.zeros((1, 3, 6, 4, 3), dtype=np.uint8), "drone_rgb.*形状"),
        ("meta_json", np.array(["not scalar"]), "meta_json.*标量"),
    ],
)
def test_invalid_fields_rejected(field: str, value: np.ndarray, match: str) -> None:
    episode = valid_episode()
    episode[field] = value
    with pytest.raises(ValueError, match=match):
        validate_episode(episode)


def test_rotation_and_homogeneous_last_row_rejected() -> None:
    episode = valid_episode()
    episode["T_W_B"][0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="旋转矩阵"):
        validate_episode(episode)
    episode = valid_episode()
    episode["T_W_Cdrone"][0, 0, 3] = [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="最后一行"):
        validate_episode(episode)


def test_schema_version_mismatch_rejected() -> None:
    episode = valid_episode()
    episode["meta_json"] = json.dumps({"schema_version": SCHEMA_VERSION + 1})
    with pytest.raises(ValueError, match="schema_version"):
        validate_episode(episode)
