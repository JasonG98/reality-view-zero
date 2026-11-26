"""FrozenEncoder 离线单元测试（本地随机小 DINOv3-ViT）与真权重慢测。"""

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from rvz.config import EncoderConfig
from rvz.device import get_device, sync
from rvz.models.encoder import FrozenEncoder


def _make_tiny_encoder(*, input_res: int = 32, pool_2x2: bool = False) -> FrozenEncoder:
    cfg = DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
        num_register_tokens=2,
    )
    backbone = DINOv3ViTModel(cfg)
    return FrozenEncoder(
        backbone,
        encoder_id="test/tiny-dinov3",
        revision="unit-rev-1",
        input_res=input_res,
        patch_size=16,
        num_register_tokens=2,
        embed_dim=32,
        mean=(0.5, 0.25, 0.75),
        std=(0.25, 0.5, 0.25),
        pool_2x2=pool_2x2,
    )


def test_preprocess_resizes_and_normalizes_exact_values() -> None:
    encoder = _make_tiny_encoder(input_res=32)
    rgb = torch.full((2, 3, 64, 48), 0.5, dtype=torch.float32)
    pixels = encoder.preprocess(rgb)
    assert pixels.shape == (2, 3, 32, 32)
    torch.testing.assert_close(pixels[:, 0], torch.zeros(2, 32, 32))
    torch.testing.assert_close(pixels[:, 1], torch.full((2, 32, 32), 0.5))
    torch.testing.assert_close(pixels[:, 2], torch.full((2, 32, 32), -1.0))


def test_encode_returns_frozen_patch_tokens_and_supports_pool_2x2() -> None:
    encoder = _make_tiny_encoder(input_res=32, pool_2x2=False)
    assert all(not p.requires_grad for p in encoder.parameters())
    rgb = torch.rand(2, 3, 48, 48)
    tokens = encoder.encode(rgb)
    assert tokens.shape == (2, 4, 32)
    assert tokens.dtype == torch.float32
    assert not tokens.requires_grad
    assert torch.isfinite(tokens).all()

    pooled = FrozenEncoder(
        encoder.backbone,
        encoder_id=encoder.encoder_id,
        revision=encoder.revision,
        input_res=32,
        patch_size=16,
        num_register_tokens=2,
        embed_dim=32,
        mean=encoder.mean,
        std=encoder.std,
        pool_2x2=True,
    )
    pooled_tokens = pooled.encode(rgb)
    assert pooled_tokens.shape == (2, 1, 32)
    torch.testing.assert_close(pooled_tokens[:, 0], tokens.mean(dim=1), atol=1e-5, rtol=1e-5)


def test_enable_lora_unfreezes_only_adapter_weights() -> None:
    encoder = _make_tiny_encoder(input_res=32)
    encoder.enable_lora(r=4, alpha=8)
    assert encoder.lora_enabled
    trainable_names = [name for name, param in encoder.named_parameters() if param.requires_grad]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    out = encoder(torch.rand(1, 3, 32, 32))
    out.sum().backward()
    assert any(param.grad is not None for param in encoder.parameters() if param.requires_grad)
    with pytest.raises(RuntimeError, match="禁止重复注入"):
        encoder.enable_lora(r=4, alpha=8)


def test_invalid_encoder_grid_rejected() -> None:
    cfg = DINOv3ViTConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, patch_size=16
    )
    backbone = DINOv3ViTModel(cfg)
    with pytest.raises(ValueError, match="整除"):
        FrozenEncoder(
            backbone,
            encoder_id="x",
            revision="r",
            input_res=30,
            patch_size=16,
            num_register_tokens=0,
            embed_dim=16,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
    with pytest.raises(ValueError, match="偶数"):
        FrozenEncoder(
            backbone,
            encoder_id="x",
            revision="r",
            input_res=48,
            patch_size=16,
            num_register_tokens=0,
            embed_dim=16,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            pool_2x2=True,
        )


@pytest.mark.slow
@pytest.mark.gpu
def test_real_dinov3_cached_forward_on_accelerator() -> None:
    """SPEC §12.2-7：在当前加速设备（MPS/CUDA）上验证真实 DINOv3-B 权重前向。"""
    device = get_device()
    cfg = EncoderConfig(res=128)
    encoder = FrozenEncoder.from_config(cfg, local_files_only=True).to(device)
    rgb = torch.rand(2, 3, 256, 256, device=device)
    tokens = encoder.encode(rgb)
    sync(device)
    assert tokens.shape == (2, 64, 768)
    assert torch.isfinite(tokens).all()
