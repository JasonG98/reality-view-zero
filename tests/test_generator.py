"""DiT 生成器（x-prediction）与 Flow Matching 采样器单元测试（ARCH §5.4、§8）。"""

import pytest
import torch

from rvz.geometry import intrinsics_from_fovy
from rvz.models.generator import Generator, GeneratorCondition, build_condition, math_attention_ctx
from rvz.models.sampler import sample


def _make_tiny_inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, GeneratorCondition]:
    gen = torch.Generator().manual_seed(123)
    x_t = torch.randn(batch, 3, 16, 16, generator=gen)
    t = torch.linspace(0.1, 0.9, batch)
    ctx_feats = torch.randn(batch, 2, 4, 4, 12, generator=gen)
    ctx_T_tgt = torch.eye(4, dtype=torch.float32).view(1, 1, 1, 4, 4).expand(batch, 2, 4, 4, 4).clone()
    ctx_T_tgt[:, :, :, :3, 3] = torch.randn(batch, 2, 4, 3, generator=gen) * 0.2
    K_cam = torch.from_numpy(intrinsics_from_fovy(90.0, 32, 32))
    K_ctx = K_cam.view(1, 1, 3, 3).expand(batch, 4, 3, 3).clone()
    tgt_T = torch.eye(4, dtype=torch.float32).view(1, 4, 4).expand(batch, 4, 4).clone()
    tgt_T[:, 0, 3] = -1.5
    tgt_T[:, 2, 3] = 0.8
    K_tgt = torch.from_numpy(intrinsics_from_fovy(90.0, 16, 16)).view(1, 3, 3).expand(batch, 3, 3).clone()
    cond = build_condition(
        ctx_feats,
        ctx_T_tgt,
        K_ctx,
        tgt_T,
        K_tgt,
        encoder_res=32,
        drone_res=16,
        tgt_patch=8,
    )
    return x_t, t, cond


def test_generator_forward_backward_and_shape_assertions() -> None:
    model = Generator(
        feat_dim=12,
        num_ctx_patches=4,
        encoder_res=32,
        drone_res=16,
        patch=8,
        dim=16,
        depth=2,
        heads=4,
        p_drop=0.0,
    )
    x_t, t, cond = _make_tiny_inputs(batch=2)
    target = torch.randn_like(x_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # AdaLN-Zero 零初始化末层，需 2 步更新让所有隐藏层门控打开并收到非零梯度。
    for _ in range(2):
        optimizer.zero_grad()
        x_hat = model(x_t, t, cond.ctx_feats, cond.rays_ctx, cond.rays_tgt)
        assert x_hat.shape == (2, 3, 16, 16)
        assert torch.isfinite(x_hat).all()
        loss = (x_hat - target).square().mean()
        loss.backward()
        optimizer.step()

    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for name, p in model.named_parameters()
        if "dt_mlp" not in name
    )
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in model.blocks[0].parameters())

    # 入口张量形状断言。
    with pytest.raises(AssertionError, match="x_t"):
        model(torch.zeros(2, 3, 24, 24), t, cond.ctx_feats, cond.rays_ctx, cond.rays_tgt)
    with pytest.raises(AssertionError, match="rays_tgt"):
        model(x_t, t, cond.ctx_feats, cond.rays_ctx, cond.rays_tgt[:, :-1])
    with pytest.raises(AssertionError, match="r 形状"):
        model(x_t, t, cond.ctx_feats, cond.rays_ctx, cond.rays_tgt, r=torch.zeros(1))


def test_generator_math_attention_and_rt_interface() -> None:
    model = Generator(
        feat_dim=12,
        num_ctx_patches=4,
        encoder_res=32,
        drone_res=16,
        patch=8,
        dim=16,
        depth=2,
        heads=4,
        p_drop=0.0,
    )
    # 给零初始化的输出层与门控赋随机小权重，检验数值等价与 (r, t) 敏感性。
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "adaLN_modulation" in name or "final_layer.linear" in name or "dt_mlp" in name:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.1)
    model.eval()
    x_t, t, cond = _make_tiny_inputs(batch=2)
    r = t * 0.5

    out_sdpa = model(x_t, t, cond.ctx_feats, cond.rays_ctx, cond.rays_tgt, r=r)
    out_math_arg = model(x_t, t, cond.ctx_feats, cond.rays_ctx, cond.rays_tgt, r=r, use_math_attn=True)
    with math_attention_ctx(True):
        out_math_ctx = model(x_t, t, cond.ctx_feats, cond.rays_ctx, cond.rays_tgt, r=r)

    torch.testing.assert_close(out_sdpa, out_math_arg, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(out_sdpa, out_math_ctx, atol=1e-5, rtol=0.0)

    out_no_r = model(x_t, t, cond.ctx_feats, cond.rays_ctx, cond.rays_tgt)
    assert not torch.allclose(out_sdpa, out_no_r, atol=1e-5)


def test_generator_camera_dropout_active_only_in_train_and_never_nan() -> None:
    model = Generator(
        feat_dim=12,
        num_ctx_patches=4,
        encoder_res=32,
        drone_res=16,
        patch=8,
        dim=16,
        depth=1,
        heads=4,
        p_drop=0.99,
    )
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        model.final_layer.linear.weight.copy_(torch.randn(model.final_layer.linear.weight.shape, generator=gen) * 0.1)
    x_t, t, cond = _make_tiny_inputs(batch=4)

    # 极高丢弃率下仍至少保留 1 路相机，不得产生 NaN。
    model.train()
    out_train = model(
        x_t,
        t,
        cond.ctx_feats,
        cond.rays_ctx,
        cond.rays_tgt,
        dropout_generator=torch.Generator().manual_seed(0),
    )
    assert torch.isfinite(out_train).all()

    # eval 模式下不丢弃任何相机，与随机数无关。
    model.eval()
    out_eval_a = model(
        x_t,
        t,
        cond.ctx_feats,
        cond.rays_ctx,
        cond.rays_tgt,
        dropout_generator=torch.Generator().manual_seed(1),
    )
    out_eval_b = model(
        x_t,
        t,
        cond.ctx_feats,
        cond.rays_ctx,
        cond.rays_tgt,
        dropout_generator=torch.Generator().manual_seed(999),
    )
    torch.testing.assert_close(out_eval_a, out_eval_b, atol=0.0, rtol=0.0)
    assert not torch.allclose(out_train, out_eval_a, atol=1e-5)


@pytest.mark.parametrize("steps", [1, 5, 50])
def test_fm_sampler_exact_for_constant_oracle_x_prediction(steps: int) -> None:
    """回归：完美 x-prediction 预测器在任意步数 Euler ODE 下必须精确收敛到目标 x1，且端点无除零。"""
    _, _, cond = _make_tiny_inputs(batch=2)
    target_x1 = torch.linspace(-0.8, 0.8, 2 * 3 * 16 * 16).view(2, 3, 16, 16)

    class _OracleGenerator:
        drone_res = 16

        def __call__(self, x_t: torch.Tensor, t: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            assert torch.isfinite(x_t).all() and torch.isfinite(t).all()
            return target_x1.to(x_t.device)

    out = sample(
        _OracleGenerator(),  # type: ignore[arg-type]  # 极简理想预测器验证 ODE 公式闭环
        cond,
        steps=steps,
        kind="fm",
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(out, target_x1, atol=1e-5, rtol=0.0)


def test_fm_sampler_with_real_generator_is_deterministic() -> None:
    model = Generator(
        feat_dim=12,
        num_ctx_patches=4,
        encoder_res=32,
        drone_res=16,
        patch=8,
        dim=16,
        depth=1,
        heads=4,
        p_drop=0.15,
    ).eval()
    _, _, cond = _make_tiny_inputs(batch=2)
    a = sample(model, cond, steps=4, kind="fm", generator=torch.Generator().manual_seed(11))
    b = sample(model, cond, steps=4, kind="fm", generator=torch.Generator().manual_seed(11))
    assert a.shape == (2, 3, 16, 16)
    assert torch.isfinite(a).all()
    assert a.min().item() >= -1.0 and a.max().item() <= 1.0
    torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
