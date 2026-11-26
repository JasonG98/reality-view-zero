"""各训练阶段损失函数（ARCH §3 规则 4：train 自行引用 lpips，不依赖 evaluate）。"""

from typing import Protocol

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


class LpipsFn(Protocol):
    """感知损失调用协议，支持在默认离线测试中注入轻量替身。"""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


class LpipsLoss(nn.Module):
    """封装 VGG LPIPS；在 fp32 下计算感知距离，骨干参数保持冻结。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = lpips.LPIPS(net="vgg", verbose=False)
        self.net.requires_grad_(False)
        self.net.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape and pred.ndim == 4 and pred.shape[1] == 3
        self.net.eval()
        # LPIPS 的 VGG 骨干统一用 fp32，避免 CUDA bf16 下数值不稳定。
        score: torch.Tensor = self.net(pred.float(), target.float())
        return score.mean()


def regress_loss_breakdown(
    x_hat: torch.Tensor,
    x1: torch.Tensor,
    *,
    lpips_fn: LpipsFn | None = None,
    lpips_weight: float = 0.0,
) -> tuple[torch.Tensor, float, float]:
    """返回 `(总损失, mse 标量, lpips 标量)`，避免训练日志重复计算 LPIPS。"""
    assert x_hat.shape == x1.shape and x_hat.ndim == 4 and x_hat.shape[1] == 3, (
        f"损失输入形状不匹配：x_hat={tuple(x_hat.shape)}, x1={tuple(x1.shape)}"
    )
    if lpips_weight < 0.0:
        raise ValueError(f"lpips_weight 必须非负：{lpips_weight}")
    mse = F.mse_loss(x_hat.float(), x1.float())
    if lpips_fn is not None and lpips_weight > 0.0:
        perceptual = lpips_fn(x_hat, x1).mean()
        total = mse + lpips_weight * perceptual
        return total, float(mse.detach().item()), float(perceptual.detach().item())
    return mse, float(mse.detach().item()), 0.0


def loss_regress(
    x_hat: torch.Tensor,
    x1: torch.Tensor,
    *,
    lpips_fn: LpipsFn | None = None,
    lpips_weight: float = 0.0,
) -> torch.Tensor:
    """SPEC §6.1 regress 阶段损失：MSE + λ·LPIPS。"""
    total, _, _ = regress_loss_breakdown(x_hat, x1, lpips_fn=lpips_fn, lpips_weight=lpips_weight)
    return total
