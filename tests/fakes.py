"""测试共用的离线假编码器与合成 episode 构造器（不联网、不依赖 MuJoCo）。"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from rvz.data.schema import SCHEMA_VERSION
from rvz.geometry import intrinsics_from_fovy


class DeterministicFakeEncoder:
    """与 FrozenEncoder 同接口的确定性离线假编码器（ARCH §8）。"""

    def __init__(
        self,
        *,
        encoder_id: str = "fake/dinov3-tiny",
        revision: str = "rev-001",
        input_res: int = 32,
        patch_size: int = 16,
        embed_dim: int = 12,
        pool_2x2: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.encoder_id = encoder_id
        self.revision = revision
        self.input_res = input_res
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pool_2x2 = pool_2x2
        self.mean = mean
        self.std = std
        self.calls = 0

    @property
    def num_patches(self) -> int:
        grid = self.input_res // self.patch_size
        if self.pool_2x2:
            grid //= 2
        return grid * grid

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        base = rgb.mean(dim=(2, 3))
        idx = torch.arange(self.num_patches, device=rgb.device, dtype=torch.float32).view(1, -1, 1)
        ch = torch.arange(self.embed_dim, device=rgb.device, dtype=torch.float32).view(1, 1, -1)
        return base[:, None, :1] + 0.01 * idx + 0.001 * ch


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    mat = np.eye(4, dtype=np.float32)
    mat[0, 0], mat[0, 1] = c, -s
    mat[1, 0], mat[1, 1] = s, c
    return mat


def write_synthetic_episode(
    raw_dir: Path,
    seed: int,
    *,
    num_steps: int = 4,
    num_drones: int = 2,
    robot_res: int = 64,
    drone_res: int = 32,
) -> dict[str, np.ndarray]:
    """写入符合 schema v2 的确定性合成 episode npz。"""
    rng = np.random.default_rng(seed)
    T_W_B = np.zeros((num_steps, 4, 4), dtype=np.float32)
    for t in range(num_steps):
        pose = _rot_z(0.2 * t + 0.1 * (seed % 5))
        pose[:3, 3] = [0.5 * t, -0.3 * t, 0.2]
        T_W_B[t] = pose

    T_B_Crobot = np.zeros((4, 4, 4), dtype=np.float32)
    for c, yaw in enumerate((0.0, math.pi / 2, -math.pi / 2, math.pi)):
        cam = _rot_z(yaw)
        cam[:3, 3] = [0.1 * c, -0.05 * c, 0.3]
        T_B_Crobot[c] = cam

    T_W_Cdrone = np.zeros((num_steps, num_drones, 4, 4), dtype=np.float32)
    for t in range(num_steps):
        for m in range(num_drones):
            drone_rel = _rot_z(0.15 * m)
            drone_rel[:3, 3] = [-1.5 - 0.2 * m, 0.0, 0.8 + 0.1 * m]
            T_W_Cdrone[t, m] = (T_W_B[t] @ drone_rel).astype(np.float32)

    K_robot = np.stack(
        [intrinsics_from_fovy(90.0 if c in (0, 3) else 120.0, robot_res, robot_res) for c in range(4)],
        axis=0,
    )
    K_drone = intrinsics_from_fovy(90.0, drone_res, drone_res)
    robot_rgb = rng.integers(0, 256, size=(num_steps, 4, robot_res, robot_res, 3), dtype=np.uint8)
    drone_rgb = rng.integers(0, 256, size=(num_steps, num_drones, drone_res, drone_res, 3), dtype=np.uint8)
    episode = {
        "robot_rgb": robot_rgb,
        "T_W_B": T_W_B,
        "T_B_Crobot": T_B_Crobot,
        "K_robot": K_robot,
        "drone_rgb": drone_rgb,
        "T_W_Cdrone": T_W_Cdrone,
        "K_drone": K_drone,
        "meta_json": np.array(json.dumps({"schema_version": SCHEMA_VERSION, "seed": seed})),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez(raw_dir / f"ep_{seed:06d}.npz", **episode)
    return episode
