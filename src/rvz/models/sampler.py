"""与训练目标解耦的采样器：Flow Matching Euler ODE（M3）与 MeanFlow 接口预留（M5）（SPEC §5.4、ARCH §5.4）。"""

from typing import Literal

import torch

from rvz.device import autocast_ctx
from rvz.models.generator import Generator, GeneratorCondition

SamplerKind = Literal["fm", "meanflow"]
_DENOM_EPS = 1e-5


def _randn_like_shape(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if generator is not None and generator.device != device:
        return torch.randn(shape, generator=generator, device=generator.device, dtype=torch.float32).to(device)
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)


@torch.no_grad()
def sample(
    gen: Generator,
    cond: GeneratorCondition,
    steps: int = 50,
    kind: SamplerKind = "fm",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """按指定采样器积分生成 `[-1, 1]` 范围无人机图像 `(B, 3, Hd, Wd)`。"""
    if steps < 1:
        raise ValueError(f"steps 必须为正整数：{steps}")
    if kind == "meanflow":
        raise NotImplementedError("MeanFlow 少步采样将在 M5 实现")
    if kind != "fm":
        raise ValueError(f"不支持的采样类型：{kind!r}")

    batch = cond.ctx_feats.shape[0]
    device = cond.ctx_feats.device
    x = _randn_like_shape((batch, 3, gen.drone_res, gen.drone_res), device=device, generator=generator)

    # 线性插值 x_t = (1 - t) x_0 + t x_1 对应速度场 v = (x_hat - x_t) / (1 - t)，
    # t_i = i / steps ∈ [0, 1 - 1/steps] 永不触及 t=1，且下限截断避免数值除零。
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
    for idx in range(steps):
        t_curr = float(ts[idx].item())
        dt = float((ts[idx + 1] - ts[idx]).item())
        t_batch = torch.full((batch,), t_curr, device=device, dtype=torch.float32)
        with autocast_ctx(device):
            x_hat = gen(
                x,
                t_batch,
                cond.ctx_feats,
                cond.rays_ctx,
                cond.rays_tgt,
                prev=cond.prev,
            )
        velocity = (x_hat.float() - x) / max(1.0 - t_curr, _DENOM_EPS)
        x = x + dt * velocity

    return x.clamp(-1.0, 1.0)
