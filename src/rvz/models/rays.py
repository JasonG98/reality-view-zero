"""Plücker 射线嵌入：torch 实现与纯 numpy 参考实现（SPEC §2.5、ARCH §5.4）。"""

import numpy as np
import torch


def _validate_grid(h: int, w: int, patch: int) -> tuple[int, int]:
    if h <= 0 or w <= 0 or patch <= 0:
        raise ValueError(f"h、w、patch 必须为正整数：h={h}, w={w}, patch={patch}")
    if h % patch != 0 or w % patch != 0:
        raise ValueError(f"图像尺寸 ({h}, {w}) 必须能被 patch={patch} 整除")
    return h // patch, w // patch


def plucker_rays_numpy(K: np.ndarray, T_B_C: np.ndarray, h: int, w: int, patch: int) -> np.ndarray:
    """SPEC §2.5 纯 numpy 参考实现，返回 `(..., N_patch, 6)` float32 Plücker 射线。"""
    grid_h, grid_w = _validate_grid(h, w, patch)
    K_arr = np.asarray(K, dtype=np.float32)
    T_arr = np.asarray(T_B_C, dtype=np.float32)
    assert K_arr.ndim >= 2 and K_arr.shape[-2:] == (3, 3), f"K 形状应为 (...,3,3)，实际 {K_arr.shape}"
    assert T_arr.ndim >= 2 and T_arr.shape[-2:] == (4, 4), f"T_B_C 形状应为 (...,4,4)，实际 {T_arr.shape}"
    assert K_arr.shape[:-2] == T_arr.shape[:-2], f"K 与 T_B_C 批维度不匹配：{K_arr.shape[:-2]} vs {T_arr.shape[:-2]}"

    # 像素中心按 geometry.intrinsics_from_fovy 的 [0, W-1] 约定对齐，每个 patch 取其覆盖像素的均值中心。
    us = np.arange(grid_w, dtype=np.float32) * patch + (patch - 1) * 0.5
    vs = np.arange(grid_h, dtype=np.float32) * patch + (patch - 1) * 0.5
    v_grid, u_grid = np.meshgrid(vs, us, indexing="ij")
    ones = np.ones_like(u_grid)
    pix = np.stack([u_grid, v_grid, ones], axis=-1).reshape(-1, 3)  # (N_patch, 3)

    K_inv = np.linalg.inv(K_arr.astype(np.float64)).astype(np.float32)
    d_cam = np.einsum("...ij,nj->...ni", K_inv, pix)
    d_cam = d_cam / np.linalg.norm(d_cam, axis=-1, keepdims=True)

    rotation = T_arr[..., :3, :3]
    origin = T_arr[..., :3, 3]
    d_base = np.einsum("...ij,...nj->...ni", rotation, d_cam)
    d_base = d_base / np.linalg.norm(d_base, axis=-1, keepdims=True)
    origin_expanded = np.broadcast_to(origin[..., None, :], d_base.shape)
    moment = np.cross(origin_expanded, d_base, axis=-1)
    return np.concatenate([d_base, moment], axis=-1).astype(np.float32)


def plucker_rays(K: torch.Tensor, T_B_C: torch.Tensor, h: int, w: int, patch: int) -> torch.Tensor:
    """SPEC §2.5 torch 实现，按 patch 中心像素计算 Base 系下 `(..., N_patch, 6)` Plücker 射线。"""
    grid_h, grid_w = _validate_grid(h, w, patch)
    assert K.ndim >= 2 and K.shape[-2:] == (3, 3), f"K 形状应为 (...,3,3)，实际 {tuple(K.shape)}"
    assert T_B_C.ndim >= 2 and T_B_C.shape[-2:] == (4, 4), f"T_B_C 形状应为 (...,4,4)，实际 {tuple(T_B_C.shape)}"
    assert K.shape[:-2] == T_B_C.shape[:-2], (
        f"K 与 T_B_C 批维度不匹配：{tuple(K.shape[:-2])} vs {tuple(T_B_C.shape[:-2])}"
    )

    device = T_B_C.device
    K_f = K.to(device=device, dtype=torch.float32)
    T_f = T_B_C.to(device=device, dtype=torch.float32)

    us = torch.arange(grid_w, device=device, dtype=torch.float32) * patch + (patch - 1) * 0.5
    vs = torch.arange(grid_h, device=device, dtype=torch.float32) * patch + (patch - 1) * 0.5
    v_grid, u_grid = torch.meshgrid(vs, us, indexing="ij")
    ones = torch.ones_like(u_grid)
    pix = torch.stack([u_grid, v_grid, ones], dim=-1).reshape(-1, 3)  # (N_patch, 3)

    K_inv = torch.linalg.inv(K_f)
    d_cam = torch.einsum("...ij,nj->...ni", K_inv, pix)
    d_cam = d_cam / torch.linalg.norm(d_cam, dim=-1, keepdim=True)

    rotation = T_f[..., :3, :3]
    origin = T_f[..., :3, 3]
    d_base = torch.einsum("...ij,...nj->...ni", rotation, d_cam)
    d_base = d_base / torch.linalg.norm(d_base, dim=-1, keepdim=True)
    origin_expanded = origin.unsqueeze(-2).expand_as(d_base)
    moment = torch.linalg.cross(origin_expanded, d_base, dim=-1)
    return torch.cat([d_base, moment], dim=-1)
