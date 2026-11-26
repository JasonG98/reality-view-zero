"""评估指标、copy_nearest_cam 基线、regress 训练与 ARCH §3 依赖边界测试。"""

import ast
import json
import math
from pathlib import Path

import pytest
import torch

from fakes import DeterministicFakeEncoder, write_synthetic_episode
from rvz.config import Config, DataConfig, EncoderConfig, ModelConfig, SimConfig, TrainConfig
from rvz.data.dataset import Sample
from rvz.data.features import cache_features, feature_cache_dir
from rvz.evaluate.metrics import (
    baseline_copy_nearest_cam,
    evaluate_checkpoint,
    evaluate_copy_nearest_cam,
    lpips_score,
    psnr,
    ssim,
)
from rvz.evaluate.report import sample_checkpoint
from rvz.models.regress import RegressBaseline
from rvz.train.loop import ExponentialMovingAverage
from rvz.train.loop import run as run_train
from rvz.train.losses import LpipsLoss


def _mock_lpips(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """轻量可微 LPIPS 替身，供默认离线测试使用（AGENTS 完成标准）。"""
    return (pred - target).abs().mean(dim=(1, 2, 3))


def _camera_pose_looking_along(direction: tuple[float, float, float]) -> torch.Tensor:
    z = torch.tensor(direction, dtype=torch.float32)
    z = z / z.norm()
    up_ref = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    if abs(float(torch.dot(z, up_ref))) > 0.99:
        up_ref = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    x = torch.linalg.cross(z, up_ref)
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    return pose


def test_metrics_properties_and_range_guard() -> None:
    gen = torch.Generator().manual_seed(7)
    clean = torch.rand(2, 3, 32, 32, generator=gen)
    small_noise = (clean + 0.05 * torch.randn(2, 3, 32, 32, generator=gen)).clamp(0.0, 1.0)
    large_noise = (clean + 0.30 * torch.randn(2, 3, 32, 32, generator=gen)).clamp(0.0, 1.0)

    torch.testing.assert_close(psnr(clean, clean), torch.full((2,), 100.0))
    torch.testing.assert_close(ssim(clean, clean), torch.ones(2), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(lpips_score(clean, clean, metric=_mock_lpips), torch.zeros(2))

    # 对称性与单调性。
    torch.testing.assert_close(ssim(clean, small_noise), ssim(small_noise, clean), atol=1e-6, rtol=0.0)
    assert bool((psnr(clean, small_noise) > psnr(clean, large_noise)).all())
    assert bool((ssim(clean, small_noise) > ssim(clean, large_noise)).all())
    assert bool(
        (
            lpips_score(clean, small_noise, metric=_mock_lpips) < lpips_score(clean, large_noise, metric=_mock_lpips)
        ).all()
    )

    # 误传 [-1, 1] 图像应直接报错，防止静默指标失真。
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        psnr(clean * 2.0 - 1.0, clean)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ssim(clean, clean * 2.0 - 1.0)


@pytest.mark.parametrize(
    ("target_dir", "expected_cam"),
    [
        ((1.0, 0.0, -0.2), 0),  # front
        ((0.0, 1.0, -0.1), 1),  # left
        ((0.0, -1.0, -0.1), 2),  # right
        ((-1.0, 0.0, 0.0), 3),  # back
    ],
)
def test_copy_nearest_cam_picks_smallest_optical_axis_angle(
    target_dir: tuple[float, float, float], expected_cam: int
) -> None:
    cam_dirs = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0))
    ctx_T = torch.stack([_camera_pose_looking_along(d) for d in cam_dirs], dim=0)
    ctx_T_tgt = torch.stack([ctx_T, ctx_T], dim=0)
    ctx_rgb = torch.zeros(2, 4, 3, 16, 16, dtype=torch.float32)
    colors = (0.1, 0.4, 0.7, 0.9)
    for cam, value in enumerate(colors):
        ctx_rgb[1, cam].fill_(value)
    sample = Sample(
        ctx_feats=None,
        ctx_rgb=ctx_rgb,
        ctx_T_tgt=ctx_T_tgt,
        K_ctx=torch.eye(3).expand(4, 3, 3).clone(),
        tgt_T=_camera_pose_looking_along(target_dir),
        K_tgt=torch.eye(3),
        tgt_rgb=torch.zeros(3, 24, 24),
        prev_rgb=torch.zeros(3, 24, 24),
        meta={"seed": 0, "t": 1, "m": 0},
    )
    out = baseline_copy_nearest_cam(sample)
    assert out.shape == (3, 24, 24)
    expected_val = colors[expected_cam] * 2.0 - 1.0
    torch.testing.assert_close(out, torch.full((3, 24, 24), expected_val), atol=1e-5, rtol=0.0)


def test_regress_baseline_forward_and_backward_tiny() -> None:
    model = RegressBaseline(feat_dim=12, num_patches=4, drone_res=24, dim=16, heads=4, num_layers=1, query_grid=4)
    ctx_feats = torch.randn(2, 2, 4, 4, 12)
    ctx_T_tgt = torch.eye(4).view(1, 1, 1, 4, 4).expand(2, 2, 4, 4, 4).clone()
    tgt_T = torch.eye(4).view(1, 4, 4).expand(2, 4, 4).clone()
    K_tgt = torch.eye(3).view(1, 3, 3).expand(2, 3, 3).clone()
    pred = model(ctx_feats, ctx_T_tgt, tgt_T, K_tgt)
    assert pred.shape == (2, 3, 24, 24)
    assert torch.isfinite(pred).all()
    assert pred.min().item() >= -1.0 and pred.max().item() <= 1.0
    pred.square().mean().backward()
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())


def test_regress_train_resume_and_baseline_eval_end_to_end(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_episode(raw_dir, seed=0, num_steps=4, num_drones=2, robot_res=32, drone_res=32)
    write_synthetic_episode(raw_dir, seed=10000, num_steps=3, num_drones=2, robot_res=32, drone_res=32)

    enc_cfg = EncoderConfig(id="fake/dinov3-tiny", res=32)
    cfg = Config(
        sim=SimConfig(robot_res=32, drone_res=32, steps=4, n_drones=2),
        data=DataConfig(raw_dir=str(raw_dir), feats_dir=str(tmp_path / "feats")),
        encoder=enc_cfg,
        model=ModelConfig(dim=16, depth=1, heads=4, patch=8),
        train=TrainConfig(batch=4, lr=1e-2, ema=0.9, seed=0, steps=4, log_every=2, save_every=2, lpips_weight=0.1),
    )
    cache_dir = feature_cache_dir(cfg.data.feats_dir, enc_cfg)
    encoder = DeterministicFakeEncoder(encoder_id=enc_cfg.id, input_res=enc_cfg.res)
    cache_features(raw_dir, cache_dir, encoder, enc_cfg, device=torch.device("cpu"))

    run_dir = tmp_path / "runs" / "regress_tiny"
    ckpt_path = run_train(
        cfg,
        "regress",
        run_dir,
        overfit_seed=0,
        lpips_fn=_mock_lpips,
        device=torch.device("cpu"),
    )
    assert ckpt_path.exists()
    assert (run_dir / "config.yaml").exists()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ckpt["stage"] == "regress"
    assert ckpt["step"] == 4
    assert ckpt["ema"] is not None
    assert {"python", "torch", "numpy", "rvz"} <= set(ckpt["versions"])

    # 验证 resume 从 step=4 继续到 step=6。
    cfg_resume = Config(
        sim=cfg.sim,
        data=cfg.data,
        encoder=cfg.encoder,
        model=cfg.model,
        train=TrainConfig(batch=4, lr=1e-2, ema=0.9, seed=0, steps=6, log_every=2, save_every=2, lpips_weight=0.1),
    )
    resumed_path = run_train(
        cfg_resume,
        "regress",
        run_dir,
        resume=True,
        overfit_seed=0,
        lpips_fn=_mock_lpips,
        device=torch.device("cpu"),
    )
    resumed_ckpt = torch.load(resumed_path, map_location="cpu", weights_only=True)
    assert resumed_ckpt["step"] == 6
    log_lines = [json.loads(line) for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log_lines[-1]["step"] == 6

    # 验证 copy_nearest_cam 与 regress 都能在 val 上算出全部指标并写 report.json。
    eval_nearest_dir = tmp_path / "eval_nearest"
    nearest_report = evaluate_copy_nearest_cam(
        cfg, split="val", out_dir=eval_nearest_dir, lpips_metric=_mock_lpips, device=torch.device("cpu")
    )
    assert nearest_report.num_samples == 4
    assert (eval_nearest_dir / "report.json").exists()
    assert all(
        math.isfinite(v) for v in (nearest_report.lpips.mean, nearest_report.psnr.mean, nearest_report.ssim.mean)
    )

    eval_reg_dir = tmp_path / "eval_regress"
    reg_report = evaluate_checkpoint(
        resumed_path,
        cfg,
        split="val",
        out_dir=eval_reg_dir,
        use_ema=True,
        lpips_metric=_mock_lpips,
        device=torch.device("cpu"),
    )
    assert reg_report.num_samples == 4
    assert reg_report.use_ema is True
    assert (eval_reg_dir / "report.json").exists()
    assert all(math.isfinite(v) for v in (reg_report.lpips.mean, reg_report.psnr.mean, reg_report.ssim.mean))


def test_ema_warmup_sheds_init_weights_and_caps_at_target_decay() -> None:
    """回归：高 decay（0.999）下短步数训练后 EMA 必须跟上已更新权重，且长步数上限不超过 decay。"""
    model = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.zeros_(model.weight)
    ema = ExponentialMovingAverage(model, decay=0.999)
    with torch.no_grad():
        model.weight.fill_(1.0)
    for step in range(1, 51):
        ema.update(model, step)
    assert float(ema.shadow["weight"].mean().item()) > 0.9

    # 大步数时预热因子趋近 1，实际衰减率应精确封顶于 0.999。
    with torch.no_grad():
        ema.shadow["weight"].zero_()
    ema.update(model, step=100_000)
    torch.testing.assert_close(ema.shadow["weight"], torch.full((2, 2), 0.001), atol=1e-6, rtol=0.0)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_arch_dependency_boundaries() -> None:
    """ARCH §3 硬规则：验证模块单向依赖与 sim 无 torch 约束。"""
    src_root = Path(__file__).resolve().parents[1] / "src" / "rvz"
    rules = {
        "sim": ("torch", "rvz.data", "rvz.models", "rvz.train", "rvz.evaluate"),
        "data": ("rvz.sim", "rvz.models", "rvz.train", "rvz.evaluate"),
        "models": ("rvz.sim", "rvz.data", "rvz.train", "rvz.evaluate"),
        "train": ("rvz.sim", "rvz.evaluate"),
        "evaluate": ("rvz.sim", "rvz.train"),
    }
    for pkg, forbidden in rules.items():
        for py_file in (src_root / pkg).glob("*.py"):
            imports = _imported_modules(py_file)
            for mod in imports:
                for bad in forbidden:
                    assert mod != bad and not mod.startswith(f"{bad}."), f"{py_file.name} 违规导入 {mod}"


def test_lpips_loss_runs_in_fp32_even_inside_outer_bf16_autocast() -> None:
    """回归：外层开启 bf16 autocast 时，LpipsLoss 内部仍须显式关闭 autocast 保持 fp32 卷积。"""
    observed: dict[str, object] = {}

    class _ProbeConvNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            out = self.conv(pred - target)
            observed["autocast_enabled"] = torch.is_autocast_enabled("cpu")
            observed["out_dtype"] = out.dtype
            return out.square().mean(dim=(1, 2, 3))

    loss_mod = LpipsLoss.__new__(LpipsLoss)
    torch.nn.Module.__init__(loss_mod)
    loss_mod.net = _ProbeConvNet()  # type: ignore[assignment]

    pred = torch.randn(2, 3, 16, 16, dtype=torch.bfloat16)
    target = torch.randn(2, 3, 16, 16, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        val = loss_mod(pred, target)
    assert observed["autocast_enabled"] is False
    assert observed["out_dtype"] == torch.float32
    assert val.dtype == torch.float32


def test_fm_train_resume_eval_and_sample_end_to_end(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_episode(raw_dir, seed=0, num_steps=4, num_drones=2, robot_res=32, drone_res=16)
    write_synthetic_episode(raw_dir, seed=10000, num_steps=3, num_drones=2, robot_res=32, drone_res=16)

    enc_cfg = EncoderConfig(id="fake/dinov3-tiny", res=32)
    cfg = Config(
        sim=SimConfig(robot_res=32, drone_res=16, steps=4, n_drones=2),
        data=DataConfig(raw_dir=str(raw_dir), feats_dir=str(tmp_path / "feats")),
        encoder=enc_cfg,
        model=ModelConfig(dim=16, depth=1, heads=4, patch=8, p_drop=0.15),
        train=TrainConfig(batch=4, lr=1e-2, ema=0.9, seed=0, steps=2, log_every=1, save_every=2, lpips_weight=0.1),
    )
    cache_dir = feature_cache_dir(cfg.data.feats_dir, enc_cfg)
    encoder = DeterministicFakeEncoder(encoder_id=enc_cfg.id, input_res=enc_cfg.res)
    cache_features(raw_dir, cache_dir, encoder, enc_cfg, device=torch.device("cpu"))

    run_dir = tmp_path / "runs" / "fm_tiny"
    ckpt_path = run_train(
        cfg,
        "fm",
        run_dir,
        overfit_seed=0,
        lpips_fn=_mock_lpips,
        device=torch.device("cpu"),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ckpt["stage"] == "fm" and ckpt["step"] == 2

    cfg_resume = Config(
        sim=cfg.sim,
        data=cfg.data,
        encoder=cfg.encoder,
        model=cfg.model,
        train=TrainConfig(batch=4, lr=1e-2, ema=0.9, seed=0, steps=4, log_every=2, save_every=2, lpips_weight=0.1),
    )
    resumed = run_train(
        cfg_resume,
        "fm",
        run_dir,
        resume=True,
        overfit_seed=0,
        lpips_fn=_mock_lpips,
        device=torch.device("cpu"),
    )
    assert torch.load(resumed, map_location="cpu", weights_only=True)["step"] == 4

    eval_dir = tmp_path / "eval_fm"
    report = evaluate_checkpoint(
        resumed,
        cfg,
        split="val",
        out_dir=eval_dir,
        use_ema=True,
        steps=2,
        lpips_metric=_mock_lpips,
        device=torch.device("cpu"),
    )
    assert report.name == "fm" and report.sample_steps == 2 and report.num_samples == 4
    assert (eval_dir / "report.json").exists()
    assert all(math.isfinite(v) for v in (report.lpips.mean, report.psnr.mean, report.ssim.mean))

    sample_dir = tmp_path / "sample_fm"
    outputs = sample_checkpoint(
        resumed,
        cfg,
        seed=10000,
        step=2,
        drone_index=1,
        steps=2,
        out_dir=sample_dir,
        lpips_metric=_mock_lpips,
        device=torch.device("cpu"),
    )
    for key in ("pred", "gt", "diff", "comparison", "metrics"):
        assert outputs[key].exists()
    metrics_json = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
    assert metrics_json["stage"] == "fm" and metrics_json["sample_steps"] == 2
