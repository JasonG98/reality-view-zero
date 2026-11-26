"""各训练阶段损失函数（ARCH §3 规则 4：train 自行引用 lpips，不依赖 evaluate）。"""

from typing import Protocol

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from rvz.models.generator import Generator, GeneratorCondition


class LpipsFn(Protocol):
    """感知损失调用协议，支持在默认离线测试中注入轻量替身。"""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


class LpipsLoss(nn.Module):
    """封装 VGG LPIPS；显式关闭 autocast 在 fp32 下计算感知距离，骨干参数保持冻结。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = lpips.LPIPS(net="vgg", verbose=False)
        self.net.requires_grad_(False)
        self.net.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape and pred.ndim == 4 and pred.shape[1] == 3
        self.net.eval()
        # 显式关闭外层 autocast，防止 CUDA bf16 把 VGG 卷积降精度导致 LPIPS 量化失真。
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            score: torch.Tensor = self.net(pred.float(), target.float())
        return score.mean()


def sample_fm_inputs(
    x1: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按线性插值 `x_t = (1 - t) x_0 + t x_1` 构造 Flow Matching 带噪输入与时间步 `t ~ U(0, 1)`。"""
    assert x1.ndim == 4 and x1.shape[1] == 3, f"x1 形状应为 (B,3,H,W)，实际 {tuple(x1.shape)}"
    batch = x1.shape[0]
    device = x1.device
    if generator is not None and generator.device != device:
        t = torch.rand((batch,), generator=generator, device=generator.device, dtype=torch.float32).to(device)
        noise = torch.randn(x1.shape, generator=generator, device=generator.device, dtype=torch.float32).to(device)
    else:
        t = torch.rand((batch,), generator=generator, device=device, dtype=torch.float32)
        noise = torch.randn(x1.shape, generator=generator, device=device, dtype=torch.float32)
    t_view = t.view(batch, 1, 1, 1)
    x_t = (1.0 - t_view) * noise + t_view * x1.float()
    return x_t, t


def x_prediction_loss_breakdown(
    x_hat: torch.Tensor,
    x1: torch.Tensor,
    *,
    lpips_fn: LpipsFn | None = None,
    lpips_weight: float = 0.0,
) -> tuple[torch.Tensor, float, float]:
    """返回 `(总损失, mse 标量, lpips 标量)`，供 regress 与 fm 的 x-prediction 共用。"""
    assert x_hat.shape == x1.shape and x_hat.ndim == 4 and x_hat.shape[1] == 3, (
        f"损失输入形状不匹配：x_hat={tuple(x_hat.shape)}, x1={tuple(x1.shape)}"
    )
    if lpips_weight < 0.0:
        raise ValueError(f"lpips_weight 必须非负：{lpips_weight}")
    x_hat_f = x_hat.float()
    x1_f = x1.float()
    mse = F.mse_loss(x_hat_f, x1_f)
    if lpips_fn is not None and lpips_weight > 0.0:
        perceptual = lpips_fn(x_hat_f, x1_f).mean()
        total = mse + lpips_weight * perceptual
        return total, float(mse.detach().item()), float(perceptual.detach().item())
    return mse, float(mse.detach().item()), 0.0


def regress_loss_breakdown(
    x_hat: torch.Tensor,
    x1: torch.Tensor,
    *,
    lpips_fn: LpipsFn | None = None,
    lpips_weight: float = 0.0,
) -> tuple[torch.Tensor, float, float]:
    """返回 `(总损失, mse 标量, lpips 标量)`，避免训练日志重复计算 LPIPS。"""
    return x_prediction_loss_breakdown(x_hat, x1, lpips_fn=lpips_fn, lpips_weight=lpips_weight)


def loss_regress(
    x_hat: torch.Tensor,
    x1: torch.Tensor,
    *,
    lpips_fn: LpipsFn | None = None,
    lpips_weight: float = 0.0,
) -> torch.Tensor:
    """SPEC §6.1 regress 阶段损失：MSE + λ·LPIPS。"""
    total, _, _ = x_prediction_loss_breakdown(x_hat, x1, lpips_fn=lpips_fn, lpips_weight=lpips_weight)
    return total


def loss_fm(
    gen: Generator,
    x1: torch.Tensor,
    cond: GeneratorCondition,
    *,
    lpips_fn: LpipsFn | None = None,
    lpips_weight: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """SPEC §6.1 fm 阶段损失：线性插值构造 x_t，在 x 空间计算 MSE + λ·LPIPS。"""
    x_t, t = sample_fm_inputs(x1, generator=generator)
    x_hat = gen(x_t, t, cond.ctx_feats, cond.rays_ctx, cond.rays_tgt, prev=cond.prev)
    total, _, _ = x_prediction_loss_breakdown(x_hat, x1, lpips_fn=lpips_fn, lpips_weight=lpips_weight)
    return total
