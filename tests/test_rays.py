"""Plücker 射线嵌入测试：验证 torch 与 numpy 参考一致（≤1e-5）及几何不变量（SPEC §2.5）。"""

import math

import numpy as np
import pytest
import torch

from rvz.geometry import intrinsics_from_fovy
from rvz.models.rays import plucker_rays, plucker_rays_numpy


def _random_rigid_poses(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    count = math.prod(shape) if shape else 1
    poses = np.zeros((count, 4, 4), dtype=np.float32)
    for i in range(count):
        mat = rng.standard_normal((3, 3))
        q, _ = np.linalg.qr(mat)
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = q.astype(np.float32)
        pose[:3, 3] = rng.uniform(-3.0, 3.0, size=3).astype(np.float32)
        poses[i] = pose
    return poses.reshape((*shape, 4, 4))


@pytest.mark.parametrize(
    ("batch_shape", "h", "w", "patch"),
    [
        ((), 128, 128, 8),
        ((3,), 128, 128, 16),
        ((2, 2, 4), 64, 64, 16),
    ],
)
def test_plucker_rays_torch_matches_numpy_reference(batch_shape: tuple[int, ...], h: int, w: int, patch: int) -> None:
    rng = np.random.default_rng(42 + len(batch_shape))
    poses = _random_rigid_poses(rng, batch_shape)
    count = math.prod(batch_shape) if batch_shape else 1
    Ks = np.stack(
        [intrinsics_from_fovy(float(rng.uniform(70.0, 125.0)), w, h) for _ in range(count)],
        axis=0,
    ).reshape((*batch_shape, 3, 3))

    ref = plucker_rays_numpy(Ks, poses, h, w, patch)
    actual = plucker_rays(torch.from_numpy(Ks), torch.from_numpy(poses), h, w, patch)

    expected_n = (h // patch) * (w // patch)
    assert actual.shape == (*batch_shape, expected_n, 6)
    assert actual.dtype == torch.float32
    np.testing.assert_allclose(actual.numpy(), ref, atol=1e-5, rtol=0.0)

    # Plücker 几何不变量：方向为单位向量且与力矩正交 d · (o × d) == 0。
    d = actual[..., :3]
    m = actual[..., 3:]
    torch.testing.assert_close(d.norm(dim=-1), torch.ones_like(d[..., 0]), atol=1e-5, rtol=0.0)
    torch.testing.assert_close((d * m).sum(dim=-1), torch.zeros_like(d[..., 0]), atol=1e-5, rtol=0.0)


def test_plucker_rays_canonical_camera_directions_and_validation() -> None:
    K = torch.from_numpy(intrinsics_from_fovy(90.0, 16, 16))
    T = torch.eye(4, dtype=torch.float32)
    rays = plucker_rays(K, T, 16, 16, patch=8)  # 2x2 grid: (0,0), (0,1), (1,0), (1,1)
    assert rays.shape == (4, 6)
    # 相机原点在 Base 原点时，力矩 o × d 全为 0。
    torch.testing.assert_close(rays[:, 3:], torch.zeros(4, 3), atol=1e-6, rtol=0.0)
    # OpenCV 约定：x 右、y 下、z 前；左上 patch (0,0) 应指向 x<0, y<0, z>0，右下 patch (1,1) 指向 x>0, y>0, z>0。
    assert rays[0, 0].item() < 0.0 and rays[0, 1].item() < 0.0 and rays[0, 2].item() > 0.0
    assert rays[3, 0].item() > 0.0 and rays[3, 1].item() > 0.0 and rays[3, 2].item() > 0.0
    # 四个对称 patch 方向均值严格沿 +z 光轴。
    mean_dir = rays[:, :3].mean(dim=0)
    assert abs(mean_dir[0].item()) < 1e-6 and abs(mean_dir[1].item()) < 1e-6 and mean_dir[2].item() > 0.0

    with pytest.raises(ValueError, match="整除"):
        plucker_rays(K, T, 16, 16, patch=6)
