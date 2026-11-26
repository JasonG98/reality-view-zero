"""特征缓存幂等性、manifest 严格校验与 EpisodeDataset 位姿换算测试。"""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from fakes import DeterministicFakeEncoder, write_synthetic_episode
from rvz.config import Config, DataConfig, EncoderConfig
from rvz.data.dataset import EpisodeDataset, Sample, collate_samples, make_dataset
from rvz.data.features import MANIFEST_FILENAME, cache_features, feature_cache_dir, load_manifest
from rvz.geometry import scale_K


def test_cache_features_idempotent_and_manifest_rejects_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_synthetic_episode(raw_dir, seed=0)
    write_synthetic_episode(raw_dir, seed=10000)

    enc_cfg = EncoderConfig(id="fake/dinov3-tiny", res=32)
    out_dir = feature_cache_dir(tmp_path / "feats", enc_cfg)
    encoder = DeterministicFakeEncoder(encoder_id=enc_cfg.id, input_res=enc_cfg.res)

    first = cache_features(raw_dir, out_dir, encoder, enc_cfg, batch_size=4, device=torch.device("cpu"))
    assert first == {"total": 2, "cached": 2, "skipped": 0}
    first_calls = encoder.calls
    bytes_before = (out_dir / "ep_000000.npy").read_bytes()

    second = cache_features(raw_dir, out_dir, encoder, enc_cfg, batch_size=4, device=torch.device("cpu"))
    assert second == {"total": 2, "cached": 0, "skipped": 2}
    assert encoder.calls == first_calls
    assert (out_dir / "ep_000000.npy").read_bytes() == bytes_before

    # 校验编码器配置与 revision 不一致时必须拒绝加载或混写缓存。
    with pytest.raises(ValueError, match="不匹配"):
        load_manifest(out_dir, cfg=EncoderConfig(id="fake/dinov3-tiny", res=64))
    with pytest.raises(ValueError, match="不匹配"):
        load_manifest(out_dir, cfg=EncoderConfig(id="fake/other", res=32))
    wrong_rev = DeterministicFakeEncoder(encoder_id=enc_cfg.id, revision="rev-999", input_res=enc_cfg.res)
    with pytest.raises(ValueError, match="revision"):
        cache_features(raw_dir, out_dir, wrong_rev, enc_cfg, device=torch.device("cpu"))

    # 篡改 manifest 版本字段也必须被拒绝。
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["rvz_version"] = "0.0.0-bad"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    with pytest.raises(ValueError, match="rvz_version"):
        load_manifest(out_dir, cfg=enc_cfg)


def test_episode_dataset_pose_transforms_intrinsics_and_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    ep_train = write_synthetic_episode(raw_dir, seed=3, num_steps=4, num_drones=2, robot_res=64, drone_res=32)
    write_synthetic_episode(raw_dir, seed=10002, num_steps=3, num_drones=2, robot_res=64, drone_res=32)

    enc_cfg = EncoderConfig(id="fake/dinov3-tiny", res=32)
    cfg = Config(data=DataConfig(raw_dir=str(raw_dir), feats_dir=str(tmp_path / "feats")), encoder=enc_cfg)
    cache_dir = feature_cache_dir(cfg.data.feats_dir, enc_cfg)
    encoder = DeterministicFakeEncoder(encoder_id=enc_cfg.id, input_res=enc_cfg.res)
    cache_features(raw_dir, cache_dir, encoder, enc_cfg, device=torch.device("cpu"))

    train_ds = make_dataset(cfg, split="train", use_cache=True)
    val_ds = make_dataset(cfg, split="val", use_cache=True)
    assert train_ds.seeds == [3]
    assert val_ds.seeds == [10002]
    assert len(train_ds) == (4 - 1) * 2
    assert len(val_ds) == (3 - 1) * 2

    sample = train_ds[3]  # t=2, m=1
    assert sample.meta == {"seed": 3, "t": 2, "m": 1}
    assert sample.ctx_feats is not None and sample.ctx_feats.shape == (2, 4, 4, 12)
    assert sample.ctx_rgb is None
    assert sample.tgt_rgb.min().item() >= -1.0 and sample.tgt_rgb.max().item() <= 1.0

    # 验证 t-1 与 t 的上下文相机变换均满足世界位姿一致性。
    t, m = 2, 1
    T_W_Bt = ep_train["T_W_B"][t]
    T_W_Bprev = ep_train["T_W_B"][t - 1]
    for cam in range(4):
        expected_prev_W = T_W_Bprev @ ep_train["T_B_Crobot"][cam]
        actual_prev_W = T_W_Bt @ sample.ctx_T_tgt[0, cam].numpy()
        np.testing.assert_allclose(actual_prev_W, expected_prev_W, atol=1e-5)
        np.testing.assert_allclose(sample.ctx_T_tgt[1, cam].numpy(), ep_train["T_B_Crobot"][cam], atol=1e-6)
    np.testing.assert_allclose(T_W_Bt @ sample.tgt_T.numpy(), ep_train["T_W_Cdrone"][t, m], atol=1e-5)

    # 验证 K_ctx 从 64x64 同步缩放到 encoder.res=32x32，且逐相机独立保留。
    for cam in range(4):
        expected_K = scale_K(ep_train["K_robot"][cam], 0.5, 0.5)
        np.testing.assert_allclose(sample.K_ctx[cam].numpy(), expected_K, atol=1e-5)
    assert not np.allclose(sample.K_ctx[0].numpy(), sample.K_ctx[1].numpy())

    # 验证 collate_samples 与在线 RGB 模式。
    batch = collate_samples([train_ds[0], train_ds[1]])
    assert batch.ctx_feats is not None and batch.ctx_feats.shape == (2, 2, 4, 4, 12)
    online_ds = make_dataset(cfg, split="val", use_cache=False)
    online_sample = online_ds[0]
    assert online_sample.ctx_feats is None
    assert online_sample.ctx_rgb is not None and online_sample.ctx_rgb.shape == (2, 4, 3, 64, 64)
    assert 0.0 <= online_sample.ctx_rgb.min().item() <= online_sample.ctx_rgb.max().item() <= 1.0


def test_sample_shape_assertions_and_bad_cache_rejected(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="恰好提供一个"):
        Sample(
            ctx_feats=None,
            ctx_rgb=None,
            ctx_T_tgt=torch.zeros(2, 4, 4, 4),
            K_ctx=torch.zeros(4, 3, 3),
            tgt_T=torch.zeros(4, 4),
            K_tgt=torch.zeros(3, 3),
            tgt_rgb=torch.zeros(3, 8, 8),
            prev_rgb=torch.zeros(3, 8, 8),
            meta={"seed": 0, "t": 1, "m": 0},
        )

    raw_dir = tmp_path / "raw"
    write_synthetic_episode(raw_dir, seed=0)
    enc_cfg = EncoderConfig(id="fake/dinov3-tiny", res=32)
    cache_dir = tmp_path / "missing_cache"
    with pytest.raises(FileNotFoundError, match=MANIFEST_FILENAME):
        EpisodeDataset(raw_dir, enc_cfg, split="train", use_cache=True, cache_dir=cache_dir)


def test_cache_features_moves_module_encoder_to_target_device(tmp_path: Path) -> None:
    """回归：cache_features 必须把 nn.Module 编码器迁移到目标 device，避免 MPS/CUDA 设备不匹配。"""

    class _ModuleEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_id = "fake/mod"
            self.revision = "rev-1"
            self.input_res = 32
            self.patch_size = 16
            self.pool_2x2 = False
            self.num_patches = 4
            self.embed_dim = 8
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def encode(self, rgb: torch.Tensor) -> torch.Tensor:
            return torch.zeros(rgb.shape[0], self.num_patches, self.embed_dim, device=rgb.device)

    raw_dir = tmp_path / "raw"
    write_synthetic_episode(raw_dir, seed=0, num_steps=2, num_drones=1, robot_res=32, drone_res=32)
    enc_cfg = EncoderConfig(id="fake/mod", res=32)
    encoder = _ModuleEncoder()
    tracked_to = Mock(wraps=encoder.to)
    encoder.to = tracked_to  # type: ignore[method-assign]  # 仅观测对 nn.Module.to 的调用
    target = torch.device("cpu")
    cache_features(raw_dir, tmp_path / "feats", encoder, enc_cfg, device=target)
    tracked_to.assert_called_once_with(target)
