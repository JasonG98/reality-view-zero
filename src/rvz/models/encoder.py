"""冻结视觉特征提取器：默认 DINOv3-B（patch 16），支持可选 2×2 池化与 LoRA（SPEC §5.1、ARCH §5.4）。"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import try_to_load_from_cache
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor, AutoModel

from rvz.config import EncoderConfig
from rvz.device import autocast_ctx

LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _resolve_hf_revision(model_id: str, backbone: nn.Module) -> str:
    """从本地 HF 缓存目录提取 snapshot commit hash，用于写入特征缓存 manifest。"""
    cached = try_to_load_from_cache(model_id, "config.json")
    if isinstance(cached, str):
        return Path(cached).parent.name
    commit_hash = getattr(backbone.config, "_commit_hash", None)
    return str(commit_hash) if commit_hash else "unknown"


def _load_pretrained_components(
    model_id: str, local_files_only: bool | None
) -> tuple[nn.Module, tuple[float, float, float], tuple[float, float, float]]:
    """优先命中本地缓存，未缓存时自动联网下载。"""
    attempts = (local_files_only,) if local_files_only is not None else (True, False)
    last_error: OSError | None = None
    for offline in attempts:
        try:
            processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=offline)
            backbone = AutoModel.from_pretrained(model_id, local_files_only=offline)
            mean_seq = tuple(float(v) for v in processor.image_mean)
            std_seq = tuple(float(v) for v in processor.image_std)
            if len(mean_seq) != 3 or len(std_seq) != 3:
                raise ValueError(f"编码器归一化参数长度应为 3：mean={mean_seq}, std={std_seq}")
            return backbone, (mean_seq[0], mean_seq[1], mean_seq[2]), (std_seq[0], std_seq[1], std_seq[2])
        except OSError as exc:
            last_error = exc
            if offline is False or local_files_only is not None:
                raise
    assert last_error is not None
    raise last_error


class FrozenEncoder(nn.Module):
    """将 `[0, 1]` 的 RGB 批量编码为 patch token `[B, N, C]`；默认全量冻结。"""

    _mean: torch.Tensor
    _std: torch.Tensor

    def __init__(
        self,
        backbone: nn.Module,
        *,
        encoder_id: str,
        revision: str,
        input_res: int,
        patch_size: int,
        num_register_tokens: int,
        embed_dim: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        pool_2x2: bool = False,
    ) -> None:
        super().__init__()
        if input_res <= 0 or patch_size <= 0 or input_res % patch_size != 0:
            raise ValueError(f"input_res ({input_res}) 必须能被 patch_size ({patch_size}) 整除")
        raw_grid = input_res // patch_size
        if pool_2x2 and raw_grid % 2 != 0:
            raise ValueError(f"启用 pool_2x2 时 patch 网格边长 ({raw_grid}) 必须为偶数")
        if num_register_tokens < 0 or embed_dim <= 0:
            raise ValueError("num_register_tokens 必须非负且 embed_dim 必须为正")
        if any(s <= 0.0 for s in std):
            raise ValueError(f"归一化标准差必须为正：{std}")

        self.backbone: nn.Module = backbone
        self.encoder_id = encoder_id
        self.revision = revision
        self.input_res = input_res
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.embed_dim = embed_dim
        self.mean = mean
        self.std = std
        self.pool_2x2 = pool_2x2
        self.lora_enabled = False

        self.register_buffer("_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    @classmethod
    def from_config(cls, cfg: EncoderConfig, *, local_files_only: bool | None = None) -> "FrozenEncoder":
        """按 EncoderConfig 加载预训练骨干与归一化参数。"""
        backbone, mean, std = _load_pretrained_components(cfg.id, local_files_only)
        model_cfg = backbone.config
        revision = _resolve_hf_revision(cfg.id, backbone)
        return cls(
            backbone,
            encoder_id=cfg.id,
            revision=revision,
            input_res=cfg.res,
            patch_size=int(model_cfg.patch_size),
            num_register_tokens=int(model_cfg.num_register_tokens),
            embed_dim=int(model_cfg.hidden_size),
            mean=mean,
            std=std,
            pool_2x2=cfg.pool_2x2,
        )

    @property
    def grid_size(self) -> int:
        raw_grid = self.input_res // self.patch_size
        return raw_grid // 2 if self.pool_2x2 else raw_grid

    @property
    def effective_patch_size(self) -> int:
        return self.patch_size * (2 if self.pool_2x2 else 1)

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size

    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        """将 `(B, 3, H, W)` 的 `[0, 1]` 浮点图像缩放并按预训练均值/方差归一化。"""
        assert rgb.ndim == 4 and rgb.shape[1] == 3, f"期望 (B,3,H,W)，实际 {tuple(rgb.shape)}"
        assert rgb.is_floating_point(), f"期望浮点张量，实际 {rgb.dtype}"
        pixels = rgb
        if pixels.shape[-2:] != (self.input_res, self.input_res):
            pixels = F.interpolate(
                pixels,
                size=(self.input_res, self.input_res),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        mean = self._mean.to(device=pixels.device, dtype=pixels.dtype)
        std = self._std.to(device=pixels.device, dtype=pixels.dtype)
        return (pixels - mean) / std

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """提取 patch token `(B, N, C)`；启用 LoRA 时保留梯度。"""
        pixels = self.preprocess(rgb)
        output = self.backbone(pixel_values=pixels)
        tokens: torch.Tensor = output.last_hidden_state
        prefix_tokens = 1 + self.num_register_tokens
        patches = tokens[:, prefix_tokens:, :]
        raw_grid = self.input_res // self.patch_size
        expected_raw = raw_grid * raw_grid
        assert patches.shape == (rgb.shape[0], expected_raw, self.embed_dim), (
            f"patch token 形状异常：期望 {(rgb.shape[0], expected_raw, self.embed_dim)}，实际 {tuple(patches.shape)}"
        )
        if self.pool_2x2:
            spatial = patches.view(rgb.shape[0], raw_grid, raw_grid, self.embed_dim).permute(0, 3, 1, 2)
            pooled = F.avg_pool2d(spatial, kernel_size=2, stride=2)
            patches = pooled.permute(0, 2, 3, 1).reshape(rgb.shape[0], self.num_patches, self.embed_dim)
        return patches

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """冻结前向：关闭梯度并经由平台 autocast 上下文输出 float32 `(B, N, C)`。"""
        self.backbone.eval()
        with torch.no_grad(), autocast_ctx(rgb.device):
            return self.forward(rgb).float()

    def enable_lora(self, r: int, alpha: int) -> None:
        """仅解冻 LoRA 适配层参数，保留主干冻结（SPEC §5.1）。"""
        if self.lora_enabled:
            raise RuntimeError("LoRA 已启用，禁止重复注入")
        if r <= 0 or alpha <= 0:
            raise ValueError(f"LoRA 参数必须为正：r={r}, alpha={alpha}")
        lora_cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=list(LORA_TARGET_MODULES))
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        self.lora_enabled = True
