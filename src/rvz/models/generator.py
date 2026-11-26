"""像素空间 DiT Transformer 生成器（x-prediction，SPEC §5.2–5.3、ARCH §5.4）。"""

import contextlib
import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from rvz.models.rays import plucker_rays

# MeanFlow 的 JVP 路径与原生 Flash Attention 不兼容，通过该开关切到纯数学注意力（ARCH ADR-4）。
_FORCE_MATH_ATTENTION = False


@contextlib.contextmanager
def math_attention_ctx(enabled: bool = True) -> Iterator[None]:
    """在上下文内强制生成器注意力走纯数学实现（供 M5 JVP 复用）。"""
    global _FORCE_MATH_ATTENTION
    previous = _FORCE_MATH_ATTENTION
    _FORCE_MATH_ATTENTION = enabled
    try:
        yield
    finally:
        _FORCE_MATH_ATTENTION = previous


def _sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    use_math_attn: bool | None = None,
) -> torch.Tensor:
    use_math = _FORCE_MATH_ATTENTION if use_math_attn is None else use_math_attn
    if use_math:
        scale = 1.0 / math.sqrt(query.shape[-1])
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        return torch.matmul(weights, value)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)


def _modulate(tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return tokens * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """将 `[0, 1]` 标量时间步编码为正弦特征，缩放 1000 倍以覆盖全频段。"""
    assert values.ndim == 1
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=values.device, dtype=torch.float32) / max(half, 1))
    angles = values.float().unsqueeze(-1) * 1000.0 * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _build_2d_sincos_pos_embed(grid_size: int, dim: int) -> torch.Tensor:
    """构造 2D 正弦位置编码初值 `(1, grid_size * grid_size, dim)`。"""
    assert dim % 4 == 0, f"2D 位置编码要求 dim ({dim}) 能被 4 整除"
    quarter = dim // 4
    omega = torch.exp(-math.log(10000.0) * torch.arange(quarter, dtype=torch.float32) / quarter)
    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    y_angles = grid_y.reshape(-1, 1) * omega.unsqueeze(0)
    x_angles = grid_x.reshape(-1, 1) * omega.unsqueeze(0)
    pos = torch.cat(
        [torch.sin(y_angles), torch.cos(y_angles), torch.sin(x_angles), torch.cos(x_angles)],
        dim=-1,
    )
    return pos.unsqueeze(0)


@dataclass(frozen=True)
class GeneratorCondition:
    """采样与训练复用的预计算几何与特征条件，避免多步采样重复算 Plücker 射线。"""

    ctx_feats: torch.Tensor  # (B, 2, 4, N_ctx, feat_dim)
    rays_ctx: torch.Tensor  # (B, 2, 4, N_ctx, 6)
    rays_tgt: torch.Tensor  # (B, N_tgt, 6)
    prev: torch.Tensor | None = None  # (B, 3, Hd, Wd)，M4 时间递归预留

    def __post_init__(self) -> None:
        assert self.ctx_feats.ndim == 5 and self.ctx_feats.shape[1:3] == (2, 4), (
            f"ctx_feats 形状应为 (B,2,4,N_ctx,C)，实际 {tuple(self.ctx_feats.shape)}"
        )
        batch, _, _, num_ctx, _ = self.ctx_feats.shape
        assert self.rays_ctx.shape == (batch, 2, 4, num_ctx, 6), (
            f"rays_ctx 形状应为 {(batch, 2, 4, num_ctx, 6)}，实际 {tuple(self.rays_ctx.shape)}"
        )
        assert self.rays_tgt.ndim == 3 and self.rays_tgt.shape[0] == batch and self.rays_tgt.shape[2] == 6, (
            f"rays_tgt 形状应为 (B,N_tgt,6)，实际 {tuple(self.rays_tgt.shape)}"
        )
        if self.prev is not None:
            assert self.prev.ndim == 4 and self.prev.shape[:2] == (batch, 3), (
                f"prev 形状应为 (B,3,Hd,Wd)，实际 {tuple(self.prev.shape)}"
            )


def build_condition(
    ctx_feats: torch.Tensor,
    ctx_T_tgt: torch.Tensor,
    K_ctx: torch.Tensor,
    tgt_T: torch.Tensor,
    K_tgt: torch.Tensor,
    *,
    encoder_res: int,
    drone_res: int,
    tgt_patch: int,
    prev: torch.Tensor | None = None,
) -> GeneratorCondition:
    """由目标 Base 系位姿与内参构造 `GeneratorCondition`（纯张量，不依赖 data 模块）。"""
    assert ctx_feats.ndim == 5 and ctx_feats.shape[1:3] == (2, 4)
    batch, _, _, num_ctx, _ = ctx_feats.shape
    assert ctx_T_tgt.shape == (batch, 2, 4, 4, 4)
    assert K_ctx.shape == (batch, 4, 3, 3)
    assert tgt_T.shape == (batch, 4, 4)
    assert K_tgt.shape == (batch, 3, 3)

    grid_ctx = math.isqrt(num_ctx)
    if grid_ctx * grid_ctx != num_ctx or encoder_res % grid_ctx != 0:
        raise ValueError(f"num_ctx_patches={num_ctx} 与 encoder_res={encoder_res} 不构成正方形 patch 网格")
    ctx_patch = encoder_res // grid_ctx

    K_ctx_expanded = K_ctx.unsqueeze(1).expand(batch, 2, 4, 3, 3)
    rays_ctx = plucker_rays(K_ctx_expanded, ctx_T_tgt, encoder_res, encoder_res, ctx_patch)
    rays_tgt = plucker_rays(K_tgt, tgt_T, drone_res, drone_res, tgt_patch)
    return GeneratorCondition(ctx_feats=ctx_feats, rays_ctx=rays_ctx, rays_tgt=rays_tgt, prev=prev)


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, *, use_math_attn: bool | None = None) -> torch.Tensor:
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).view(batch, length, 3, self.heads, self.head_dim)
        query = self.q_norm(qkv[:, :, 0]).transpose(1, 2)
        key = self.k_norm(qkv[:, :, 1]).transpose(1, 2)
        value = qkv[:, :, 2].transpose(1, 2)
        out = _sdpa(query, key, value, use_math_attn=use_math_attn)
        return self.proj(out.transpose(1, 2).reshape(batch, length, dim))


class _CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        use_math_attn: bool | None = None,
    ) -> torch.Tensor:
        batch, num_q, dim = tokens.shape
        num_kv = context.shape[1]
        query = self.q_norm(self.q_proj(tokens).view(batch, num_q, self.heads, self.head_dim)).transpose(1, 2)
        kv = self.kv_proj(context).view(batch, num_kv, 2, self.heads, self.head_dim)
        key = self.k_norm(kv[:, :, 0]).transpose(1, 2)
        value = kv[:, :, 1].transpose(1, 2)
        out = _sdpa(query, key, value, attn_mask=attn_mask, use_math_attn=use_math_attn)
        return self.proj(out.transpose(1, 2).reshape(batch, num_q, dim))


class _DiTBlock(nn.Module):
    """DiT Block：AdaLN-Zero 调制 Self-Attention 与 MLP，Cross-Attention 注入多视角上下文。"""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = _SelfAttention(dim, heads)
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = _CrossAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        time_cond: torch.Tensor,
        *,
        ctx_mask: torch.Tensor | None = None,
        use_math_attn: bool | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time_cond).chunk(
            6, dim=-1
        )
        tokens = tokens + gate_msa.unsqueeze(1) * self.self_attn(
            _modulate(self.norm1(tokens), shift_msa, scale_msa),
            use_math_attn=use_math_attn,
        )
        tokens = tokens + self.cross_attn(
            self.norm_q(tokens),
            self.norm_kv(context),
            attn_mask=ctx_mask,
            use_math_attn=use_math_attn,
        )
        tokens = tokens + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(tokens), shift_mlp, scale_mlp))
        return tokens


class _FinalLayer(nn.Module):
    """AdaLN-Zero 输出头，将目标 token 线性投影回像素 patch（x-prediction）。"""

    def __init__(self, dim: int, patch_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, patch_dim)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, tokens: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(time_cond).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(tokens), shift, scale))


class Generator(nn.Module):
    """像素空间 DiT 生成器：输入带噪图 `x_t` 与多视角几何条件，直接输出去噪图 `x_hat`（x-prediction）。"""

    def __init__(
        self,
        feat_dim: int,
        num_ctx_patches: int,
        encoder_res: int,
        drone_res: int,
        *,
        patch: int = 8,
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        p_drop: float = 0.15,
    ) -> None:
        super().__init__()
        if feat_dim <= 0 or num_ctx_patches <= 0 or encoder_res <= 0 or drone_res <= 0:
            raise ValueError("feat_dim、num_ctx_patches、encoder_res、drone_res 必须为正整数")
        if patch <= 0 or drone_res % patch != 0:
            raise ValueError(f"drone_res ({drone_res}) 必须能被 patch ({patch}) 整除")
        if dim <= 0 or depth <= 0 or heads <= 0 or dim % heads != 0 or dim % 4 != 0:
            raise ValueError(f"dim ({dim}) 必须为正整数且同时能被 heads ({heads}) 和 4 整除")
        if not 0.0 <= p_drop < 1.0:
            raise ValueError(f"p_drop 必须在 [0, 1) 内：{p_drop}")

        self.feat_dim = feat_dim
        self.num_ctx_patches = num_ctx_patches
        self.encoder_res = encoder_res
        self.drone_res = drone_res
        self.patch = patch
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.p_drop = p_drop
        self.tgt_grid = drone_res // patch
        self.num_tgt_patches = self.tgt_grid * self.tgt_grid
        self.patch_dim = 3 * patch * patch

        self.x_embed = nn.Linear(self.patch_dim, dim)
        self.tgt_pos_embed = nn.Parameter(_build_2d_sincos_pos_embed(self.tgt_grid, dim))

        # 上下文与目标射线均在同一目标 Base 系下，共用同一 Plücker 嵌入 MLP 便于跨视角几何对齐。
        self.ray_embed = nn.Sequential(
            nn.Linear(6, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.ctx_proj = nn.Linear(feat_dim, dim)
        self.ctx_time_embed = nn.Parameter(torch.randn(1, 2, 1, 1, dim) * 0.02)
        self.ctx_cam_embed = nn.Parameter(torch.randn(1, 1, 4, 1, dim) * 0.02)
        self.ctx_norm = nn.LayerNorm(dim, eps=1e-6)

        self.time_freq_dim = min(256, dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(self.time_freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # 预留 (r, t) 时间间隔嵌入（dt = t - r），零初始化末层以保证 M3 (r=t) 与 M5 平滑兼容。
        self.dt_mlp = nn.Sequential(
            nn.Linear(self.time_freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.dt_mlp[2].weight)
        nn.init.zeros_(self.dt_mlp[2].bias)

        self.blocks = nn.ModuleList([_DiTBlock(dim, heads) for _ in range(depth)])
        self.final_layer = _FinalLayer(dim, self.patch_dim)

    def model_meta(self) -> dict[str, int | float]:
        return {
            "feat_dim": self.feat_dim,
            "num_ctx_patches": self.num_ctx_patches,
            "encoder_res": self.encoder_res,
            "drone_res": self.drone_res,
            "patch": self.patch,
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "p_drop": self.p_drop,
        }

    def _sample_camera_mask(
        self, batch: int, device: torch.device, dropout_generator: torch.Generator | None
    ) -> torch.Tensor | None:
        if not self.training or self.p_drop <= 0.0:
            return None
        if dropout_generator is not None and dropout_generator.device != device:
            probs = torch.rand((batch, 4), generator=dropout_generator, device=dropout_generator.device).to(device)
        else:
            probs = torch.rand((batch, 4), generator=dropout_generator, device=device)
        keep_cam = probs >= self.p_drop  # (B, 4)
        # 至少保留 1 路相机，避免全丢弃导致 cross-attention softmax 产生 NaN。
        all_dropped = ~keep_cam.any(dim=1)
        if all_dropped.any():
            fallback_idx = probs[all_dropped].argmax(dim=1)
            keep_cam[all_dropped, fallback_idx] = True
        keep_tokens = (
            keep_cam[:, None, :, None]
            .expand(batch, 2, 4, self.num_ctx_patches)
            .reshape(batch, 1, 1, 2 * 4 * self.num_ctx_patches)
        )
        return keep_tokens

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ctx_feats: torch.Tensor,
        rays_ctx: torch.Tensor,
        rays_tgt: torch.Tensor,
        *,
        r: torch.Tensor | None = None,
        prev: torch.Tensor | None = None,
        use_math_attn: bool | None = None,
        dropout_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """从带噪图像 `x_t`、时间步 `t`（及可选 `r`）与多视角几何条件预测去噪图像 `x_hat`。"""
        assert x_t.ndim == 4 and x_t.shape[1:] == (3, self.drone_res, self.drone_res), (
            f"x_t 形状应为 (B,3,{self.drone_res},{self.drone_res})，实际 {tuple(x_t.shape)}"
        )
        batch = x_t.shape[0]
        assert t.shape == (batch,), f"t 形状应为 ({batch},)，实际 {tuple(t.shape)}"
        if r is not None:
            assert r.shape == (batch,), f"r 形状应为 ({batch},)，实际 {tuple(r.shape)}"
        assert ctx_feats.shape == (batch, 2, 4, self.num_ctx_patches, self.feat_dim), (
            f"ctx_feats 形状应为 {(batch, 2, 4, self.num_ctx_patches, self.feat_dim)}，实际 {tuple(ctx_feats.shape)}"
        )
        assert rays_ctx.shape == (batch, 2, 4, self.num_ctx_patches, 6), (
            f"rays_ctx 形状应为 {(batch, 2, 4, self.num_ctx_patches, 6)}，实际 {tuple(rays_ctx.shape)}"
        )
        assert rays_tgt.shape == (batch, self.num_tgt_patches, 6), (
            f"rays_tgt 形状应为 {(batch, self.num_tgt_patches, 6)}，实际 {tuple(rays_tgt.shape)}"
        )
        if prev is not None:
            raise NotImplementedError("时间递归条件 prev 将在 M4 (fm_rec) 实现")

        patches = rearrange(x_t, "b c (gh p1) (gw p2) -> b (gh gw) (p1 p2 c)", p1=self.patch, p2=self.patch)
        tokens = self.x_embed(patches) + self.tgt_pos_embed + self.ray_embed(rays_tgt)

        ctx_tokens = (
            self.ctx_proj(ctx_feats) + self.ray_embed(rays_ctx) + self.ctx_time_embed + self.ctx_cam_embed
        ).reshape(batch, 2 * 4 * self.num_ctx_patches, self.dim)
        ctx_tokens = self.ctx_norm(ctx_tokens)

        dt = torch.zeros_like(t) if r is None else (t - r)
        time_cond = self.t_mlp(_sinusoidal_embedding(t, self.time_freq_dim)) + self.dt_mlp(
            _sinusoidal_embedding(dt, self.time_freq_dim)
        )
        ctx_mask = self._sample_camera_mask(batch, x_t.device, dropout_generator)

        for block in self.blocks:
            tokens = block(tokens, ctx_tokens, time_cond, ctx_mask=ctx_mask, use_math_attn=use_math_attn)

        out_patches = self.final_layer(tokens, time_cond)
        return rearrange(
            out_patches,
            "b (gh gw) (p1 p2 c) -> b c (gh p1) (gw p2)",
            gh=self.tgt_grid,
            gw=self.tgt_grid,
            p1=self.patch,
            p2=self.patch,
            c=3,
        )
