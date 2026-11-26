"""端到端小集成测试：真实 MuJoCo 生成 → 假编码器缓存 → 2 步训练 → 基线评估（不联网、不下载模型）。"""

import math
from pathlib import Path

import torch

from fakes import DeterministicFakeEncoder
from rvz.config import Config, DataConfig, EncoderConfig, ModelConfig, SimConfig, TrainConfig
from rvz.data.features import cache_features, feature_cache_dir
from rvz.evaluate.metrics import evaluate_checkpoint, evaluate_copy_nearest_cam
from rvz.sim.record import generate_many
from rvz.train.loop import run as run_train


def _mock_lpips(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean(dim=(1, 2, 3))


def test_pipeline_tiny_sim_cache_regress_eval(tmp_path: Path) -> None:
    """串联 sim.record → data.features → data.dataset → train(regress) → evaluate 全链路。"""
    raw_dir = tmp_path / "raw"
    feats_root = tmp_path / "feats"
    sim_cfg = SimConfig(robot_res=48, drone_res=32, steps=3, n_drones=2, max_resamples=30)
    enc_cfg = EncoderConfig(id="fake/dinov3-tiny", res=32)
    cfg = Config(
        sim=sim_cfg,
        data=DataConfig(raw_dir=str(raw_dir), feats_dir=str(feats_root)),
        encoder=enc_cfg,
        model=ModelConfig(dim=16, depth=1, heads=4, patch=8),
        train=TrainConfig(batch=2, lr=1e-2, ema=0.9, seed=0, steps=2, log_every=1, save_every=2, lpips_weight=0.1),
    )

    # 1. 真实 MuJoCo 录制 1 个 train 与 1 个 val episode。
    generate_many([0, 10000], sim_cfg, raw_dir, workers=1)
    assert (raw_dir / "ep_000000.npz").exists()
    assert (raw_dir / "ep_010000.npz").exists()

    # 2. 用假编码器离线缓存特征并验证幂等。
    cache_dir = feature_cache_dir(feats_root, enc_cfg)
    encoder = DeterministicFakeEncoder(encoder_id=enc_cfg.id, input_res=enc_cfg.res)
    first = cache_features(raw_dir, cache_dir, encoder, enc_cfg, batch_size=4, device=torch.device("cpu"))
    second = cache_features(raw_dir, cache_dir, encoder, enc_cfg, batch_size=4, device=torch.device("cpu"))
    assert first == {"total": 2, "cached": 2, "skipped": 0}
    assert second == {"total": 2, "cached": 0, "skipped": 2}

    # 3. 跑 2 步 regress 训练并导出 checkpoint。
    run_dir = tmp_path / "runs" / "regress_tiny"
    ckpt_path = run_train(cfg, "regress", run_dir, lpips_fn=_mock_lpips, device=torch.device("cpu"))
    assert ckpt_path.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "log.jsonl").exists()

    # 4. 在真实 MuJoCo 生成的 val episode 上评估两个 M2 基线。
    nearest_dir = tmp_path / "runs" / "eval_nearest"
    nearest = evaluate_copy_nearest_cam(
        cfg, split="val", out_dir=nearest_dir, lpips_metric=_mock_lpips, device=torch.device("cpu")
    )
    assert nearest.num_samples == (sim_cfg.steps - 1) * sim_cfg.n_drones
    assert (nearest_dir / "report.json").exists()
    assert all(math.isfinite(v) for v in (nearest.lpips.mean, nearest.psnr.mean, nearest.ssim.mean))

    regress_dir = tmp_path / "runs" / "eval_regress"
    reg = evaluate_checkpoint(
        ckpt_path,
        cfg,
        split="val",
        out_dir=regress_dir,
        use_ema=True,
        lpips_metric=_mock_lpips,
        device=torch.device("cpu"),
    )
    assert reg.num_samples == (sim_cfg.steps - 1) * sim_cfg.n_drones
    assert (regress_dir / "report.json").exists()
    assert all(math.isfinite(v) for v in (reg.lpips.mean, reg.psnr.mean, reg.ssim.mean))
