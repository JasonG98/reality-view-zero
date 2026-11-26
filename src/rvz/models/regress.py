"""M2 直接回归基线网络（SPEC §6.1）：从 {t-1, t} 四相机特征与位姿直接回归无人机图像。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossAttentionBlock(nn.Module):
    """目标空间查询对上下文特征做多头 Cross-Attention（统一走 SDPA）。"""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) 必须能被 heads ({heads}) 整除")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, num_q, dim = query.shape
        num_kv = context.shape[1]
        q = self.q_proj(self.norm_q(query)).view(batch, num_q, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(self.norm_kv(context)).view(batch, num_kv, 2, self.heads, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, num_q, dim)
        query = query + self.out_proj(attn)
        return query + self.mlp(self.norm_mlp(query))


class RegressBaseline(nn.Module):
    """无生成模型的像素空间回归基线，用作 G2/G3 下限对照。"""

    def __init__(
        self,
        feat_dim: int,
        num_patches: int,
        drone_res: int,
        *,
        dim: int = 256,
        heads: int = 4,
        num_layers: int = 2,
        query_grid: int = 8,
    ) -> None:
        super().__init__()
        if feat_dim <= 0 or num_patches <= 0 or drone_res <= 0:
            raise ValueError("feat_dim、num_patches、drone_res 必须为正整数")
        if dim <= 0 or heads <= 0 or num_layers <= 0 or query_grid <= 0:
            raise ValueError("dim、heads、num_layers、query_grid 必须为正整数")

        self.feat_dim = feat_dim
        self.num_patches = num_patches
        self.drone_res = drone_res
        self.dim = dim
        self.heads = heads
        self.num_layers = num_layers
        self.query_grid = query_grid

        self.ctx_proj = nn.Linear(feat_dim, dim)
        self.ctx_pose_proj = nn.Linear(12, dim)
        self.time_embed = nn.Parameter(torch.randn(1, 2, 1, 1, dim) * 0.02)
        self.cam_embed = nn.Parameter(torch.randn(1, 1, 4, 1, dim) * 0.02)
        self.patch_embed = nn.Parameter(torch.randn(1, 1, 1, num_patches, dim) * 0.02)

        self.query_embed = nn.Parameter(torch.randn(1, query_grid * query_grid, dim) * 0.02)
        self.tgt_cond_mlp = nn.Sequential(
            nn.Linear(16, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([_CrossAttentionBlock(dim, heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)

        # 按 2 倍逐级上采样解码到接近 drone_res，避免大核反卷积棋盘伪影。
        stages: list[nn.Module] = []
        channels = dim
        current_res = query_grid
        while current_res < drone_res:
            next_channels = max(32, channels // 2) if current_res * 2 >= drone_res // 2 else channels
            groups = math.gcd(8, next_channels)
            stages.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(channels, next_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(groups, next_channels),
                    nn.SiLU(),
                )
            )
            channels = next_channels
            current_res *= 2
        self.upsample_stages = nn.ModuleList(stages)
        self.to_rgb = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def model_meta(self) -> dict[str, int]:
        return {
            "feat_dim": self.feat_dim,
            "num_patches": self.num_patches,
            "drone_res": self.drone_res,
            "dim": self.dim,
            "heads": self.heads,
            "num_layers": self.num_layers,
            "query_grid": self.query_grid,
        }

    def forward(
        self,
        ctx_feats: torch.Tensor,
        ctx_T_tgt: torch.Tensor,
        tgt_T: torch.Tensor,
        K_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """从上下文特征和位姿回归 `[-1, 1]` 范围的无人机 RGB `(B, 3, Hd, Wd)`。"""
        assert ctx_feats.ndim == 5 and ctx_feats.shape[1:] == (2, 4, self.num_patches, self.feat_dim), (
            f"ctx_feats 形状应为 (B,2,4,{self.num_patches},{self.feat_dim})，实际 {tuple(ctx_feats.shape)}"
        )
        batch = ctx_feats.shape[0]
        assert ctx_T_tgt.shape == (batch, 2, 4, 4, 4), f"ctx_T_tgt 形状异常：{tuple(ctx_T_tgt.shape)}"
        assert tgt_T.shape == (batch, 4, 4), f"tgt_T 形状异常：{tuple(tgt_T.shape)}"
        assert K_tgt.shape == (batch, 3, 3), f"K_tgt 形状异常：{tuple(K_tgt.shape)}"

        ctx_pose = self.ctx_pose_proj(ctx_T_tgt[:, :, :, :3, :].reshape(batch, 2, 4, 12)).unsqueeze(3)
        ctx_tokens = (
            self.ctx_proj(ctx_feats) + ctx_pose + self.time_embed + self.cam_embed + self.patch_embed
        ).reshape(batch, 2 * 4 * self.num_patches, self.dim)

        scale = float(self.drone_res)
        intrinsics_vec = torch.stack(
            [K_tgt[:, 0, 0] / scale, K_tgt[:, 1, 1] / scale, K_tgt[:, 0, 2] / scale, K_tgt[:, 1, 2] / scale],
            dim=-1,
        )
        tgt_vec = torch.cat([tgt_T[:, :3, :].reshape(batch, 12), intrinsics_vec], dim=-1)
        query = self.query_embed.expand(batch, -1, -1) + self.tgt_cond_mlp(tgt_vec).unsqueeze(1)

        for block in self.blocks:
            query = block(query, ctx_tokens)
        spatial = self.final_norm(query).transpose(1, 2).reshape(batch, self.dim, self.query_grid, self.query_grid)
        for stage in self.upsample_stages:
            spatial = stage(spatial)
        if spatial.shape[-2:] != (self.drone_res, self.drone_res):
            spatial = F.interpolate(
                spatial,
                size=(self.drone_res, self.drone_res),
                mode="bilinear",
                align_corners=False,
            )
        return torch.tanh(self.to_rgb(spatial))
