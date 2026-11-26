"""Episode 数据集、Sample 结构与目标 Base 系相对位姿换算（SPEC §2.3–2.5、§3.2、ARCH §5.3）。"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from rvz.config import Config, EncoderConfig
from rvz.data.features import Manifest, feature_cache_dir, load_manifest, parse_episode_seed
from rvz.data.schema import validate_episode
from rvz.geometry import relative, scale_K

SplitName = Literal["train", "val", "test"]

# SPEC §3.2 规定的 half-open seed 划分。
SPLIT_SEED_RANGES: dict[SplitName, tuple[int, int]] = {
    "train": (0, 10000),
    "val": (10000, 10100),
    "test": (20000, 20100),
}


@dataclass(frozen=True)
class Sample:
    """单条训练/评估样本 `(episode, t, m)`，所有位姿统一在目标时刻 `t` 的 Base 系下。"""

    ctx_feats: torch.Tensor | None  # (2, 4, N, C) 缓存模式
    ctx_rgb: torch.Tensor | None  # (2, 4, 3, H, W) [0, 1] 在线模式
    ctx_T_tgt: torch.Tensor  # (2, 4, 4, 4) 上下文相机 -> 目标时刻 Base 系
    K_ctx: torch.Tensor  # (4, 3, 3) 已按编码器分辨率缩放
    tgt_T: torch.Tensor  # (4, 4) 无人机相机 -> 目标时刻 Base 系
    K_tgt: torch.Tensor  # (3, 3)
    tgt_rgb: torch.Tensor  # (3, Hd, Wd) [-1, 1]
    prev_rgb: torch.Tensor  # (3, Hd, Wd) [-1, 1] 同一台无人机的 t-1 图像
    meta: dict[str, int]  # seed, t, m

    def __post_init__(self) -> None:
        if (self.ctx_feats is None) == (self.ctx_rgb is None):
            raise AssertionError("ctx_feats 与 ctx_rgb 必须恰好提供一个")
        if self.ctx_feats is not None:
            assert self.ctx_feats.ndim == 4 and self.ctx_feats.shape[:2] == (2, 4), (
                f"ctx_feats 形状应为 (2,4,N,C)，实际 {tuple(self.ctx_feats.shape)}"
            )
        if self.ctx_rgb is not None:
            assert self.ctx_rgb.ndim == 5 and self.ctx_rgb.shape[:3] == (2, 4, 3), (
                f"ctx_rgb 形状应为 (2,4,3,H,W)，实际 {tuple(self.ctx_rgb.shape)}"
            )
        assert self.ctx_T_tgt.shape == (2, 4, 4, 4), f"ctx_T_tgt 形状应为 (2,4,4,4)，实际 {tuple(self.ctx_T_tgt.shape)}"
        assert self.K_ctx.shape == (4, 3, 3), f"K_ctx 形状应为 (4,3,3)，实际 {tuple(self.K_ctx.shape)}"
        assert self.tgt_T.shape == (4, 4), f"tgt_T 形状应为 (4,4)，实际 {tuple(self.tgt_T.shape)}"
        assert self.K_tgt.shape == (3, 3), f"K_tgt 形状应为 (3,3)，实际 {tuple(self.K_tgt.shape)}"
        assert self.tgt_rgb.ndim == 3 and self.tgt_rgb.shape[0] == 3, (
            f"tgt_rgb 形状应为 (3,Hd,Wd)，实际 {tuple(self.tgt_rgb.shape)}"
        )
        assert self.prev_rgb.shape == self.tgt_rgb.shape, (
            f"prev_rgb 形状 {tuple(self.prev_rgb.shape)} 必须与 tgt_rgb {tuple(self.tgt_rgb.shape)} 一致"
        )


@dataclass(frozen=True)
class SampleBatch:
    """按 batch 堆叠后的样本容器，供训练循环与评估复用。"""

    ctx_feats: torch.Tensor | None  # (B, 2, 4, N, C)
    ctx_rgb: torch.Tensor | None  # (B, 2, 4, 3, H, W)
    ctx_T_tgt: torch.Tensor  # (B, 2, 4, 4, 4)
    K_ctx: torch.Tensor  # (B, 4, 3, 3)
    tgt_T: torch.Tensor  # (B, 4, 4)
    K_tgt: torch.Tensor  # (B, 3, 3)
    tgt_rgb: torch.Tensor  # (B, 3, Hd, Wd) [-1, 1]
    prev_rgb: torch.Tensor  # (B, 3, Hd, Wd) [-1, 1]
    meta: list[dict[str, int]]

    def __post_init__(self) -> None:
        batch = self.tgt_rgb.shape[0]
        assert self.tgt_rgb.ndim == 4 and self.tgt_rgb.shape[1] == 3
        assert self.prev_rgb.shape == self.tgt_rgb.shape
        assert self.ctx_T_tgt.shape == (batch, 2, 4, 4, 4)
        assert self.K_ctx.shape == (batch, 4, 3, 3)
        assert self.tgt_T.shape == (batch, 4, 4)
        assert self.K_tgt.shape == (batch, 3, 3)
        if (self.ctx_feats is None) == (self.ctx_rgb is None):
            raise AssertionError("SampleBatch 的 ctx_feats 与 ctx_rgb 必须恰好提供一个")
        if self.ctx_feats is not None:
            assert self.ctx_feats.ndim == 5 and self.ctx_feats.shape[:3] == (batch, 2, 4)
        if self.ctx_rgb is not None:
            assert self.ctx_rgb.ndim == 6 and self.ctx_rgb.shape[:4] == (batch, 2, 4, 3)
        assert len(self.meta) == batch

    def to(self, device: torch.device) -> "SampleBatch":
        return SampleBatch(
            ctx_feats=self.ctx_feats.to(device) if self.ctx_feats is not None else None,
            ctx_rgb=self.ctx_rgb.to(device) if self.ctx_rgb is not None else None,
            ctx_T_tgt=self.ctx_T_tgt.to(device),
            K_ctx=self.K_ctx.to(device),
            tgt_T=self.tgt_T.to(device),
            K_tgt=self.K_tgt.to(device),
            tgt_rgb=self.tgt_rgb.to(device),
            prev_rgb=self.prev_rgb.to(device),
            meta=self.meta,
        )


def collate_samples(samples: Sequence[Sample]) -> SampleBatch:
    """将样本列表堆叠为 SampleBatch。"""
    if not samples:
        raise ValueError("samples 不能为空")
    has_feats = samples[0].ctx_feats is not None
    return SampleBatch(
        ctx_feats=torch.stack([s.ctx_feats for s in samples if s.ctx_feats is not None], dim=0) if has_feats else None,
        ctx_rgb=torch.stack([s.ctx_rgb for s in samples if s.ctx_rgb is not None], dim=0) if not has_feats else None,
        ctx_T_tgt=torch.stack([s.ctx_T_tgt for s in samples], dim=0),
        K_ctx=torch.stack([s.K_ctx for s in samples], dim=0),
        tgt_T=torch.stack([s.tgt_T for s in samples], dim=0),
        K_tgt=torch.stack([s.K_tgt for s in samples], dim=0),
        tgt_rgb=torch.stack([s.tgt_rgb for s in samples], dim=0),
        prev_rgb=torch.stack([s.prev_rgb for s in samples], dim=0),
        meta=[s.meta for s in samples],
    )


@dataclass(frozen=True)
class _EpisodeMeta:
    seed: int
    npz_path: Path
    feat_path: Path | None
    num_steps: int
    num_drones: int
    T_W_B: np.ndarray  # (T, 4, 4)
    T_B_Crobot: np.ndarray  # (4, 4, 4)
    K_ctx_scaled: np.ndarray  # (4, 3, 3)
    T_W_Cdrone: np.ndarray  # (T, M, 4, 4)
    K_drone: np.ndarray  # (3, 3)


def discover_episodes(
    raw_dir: str | Path,
    *,
    split: SplitName | None = None,
    seeds: Sequence[int] | None = None,
) -> list[tuple[int, Path]]:
    """按 split 范围或显式 seeds 发现原始 episode npz 文件。"""
    directory = Path(raw_dir)
    if not directory.exists():
        raise FileNotFoundError(f"原始数据目录不存在：{directory}")
    found: dict[int, Path] = {}
    for entry in directory.iterdir():
        if entry.is_file():
            seed = parse_episode_seed(entry)
            if seed is not None:
                found[seed] = entry
    if seeds is not None:
        missing = [seed for seed in seeds if seed not in found]
        if missing:
            raise FileNotFoundError(f"目录 {directory} 缺少指定 seed：{missing[:10]}")
        return [(seed, found[seed]) for seed in sorted(set(seeds))]
    if split is not None:
        start, stop = SPLIT_SEED_RANGES[split]
        selected = [(seed, path) for seed, path in sorted(found.items()) if start <= seed < stop]
        if not selected:
            raise FileNotFoundError(f"目录 {directory} 中没有 split={split!r} ({start}:{stop}) 的 episode")
        return selected
    if not found:
        raise FileNotFoundError(f"目录 {directory} 中没有 ep_*.npz 文件")
    return sorted(found.items())


def _scale_robot_intrinsics(K_robot: np.ndarray, robot_hw: tuple[int, int], encoder_res: int) -> np.ndarray:
    height, width = robot_hw
    sx = encoder_res / float(width)
    sy = encoder_res / float(height)
    return np.stack([scale_K(K_robot[c], sx, sy) for c in range(4)], axis=0).astype(np.float32)


def _uint8_to_minus_one_one(rgb_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb_hwc.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def _uint8_batch_to_unit(rgb_tchw_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb_tchw_hwc.copy()).permute(0, 1, 4, 2, 3).float().div_(255.0)


class EpisodeDataset(Dataset[Sample]):
    """基于 episode npz 与离线特征缓存（或在线 RGB）的样本数据集。"""

    def __init__(
        self,
        raw_dir: str | Path,
        encoder_cfg: EncoderConfig,
        *,
        split: SplitName | None = None,
        use_cache: bool = True,
        cache_dir: str | Path | None = None,
        seeds: Sequence[int] | None = None,
        max_cached_episodes: int = 32,
    ) -> None:
        if max_cached_episodes <= 0:
            raise ValueError("max_cached_episodes 必须为正整数")
        self.use_cache = use_cache
        self.encoder_cfg = encoder_cfg
        self.manifest: Manifest | None = None
        resolved_cache_dir: Path | None = None
        if use_cache:
            if cache_dir is None:
                raise ValueError("use_cache=True 时必须提供 cache_dir")
            resolved_cache_dir = Path(cache_dir)
            self.manifest = load_manifest(resolved_cache_dir, cfg=encoder_cfg)

        discovered = discover_episodes(raw_dir, split=split, seeds=seeds)
        self._episodes: list[_EpisodeMeta] = []
        self._index: list[tuple[int, int, int]] = []
        for ep_idx, (seed, npz_path) in enumerate(discovered):
            with np.load(npz_path, allow_pickle=False) as loaded:
                episode = {name: loaded[name] for name in loaded.files}
            validate_episode(episode)
            robot_rgb = episode["robot_rgb"]
            drone_rgb = episode["drone_rgb"]
            num_steps = int(robot_rgb.shape[0])
            num_drones = int(drone_rgb.shape[1])
            if num_steps < 2:
                raise ValueError(f"episode {seed} 步数 {num_steps} < 2，无法构造 {{t-1, t}} 上下文（SPEC §2.3）")

            feat_path: Path | None = None
            if use_cache:
                assert resolved_cache_dir is not None and self.manifest is not None
                feat_path = resolved_cache_dir / f"ep_{seed:06d}.npy"
                if not feat_path.exists():
                    raise FileNotFoundError(f"缺少特征缓存文件：{feat_path}")
                feats_mmap = np.load(feat_path, mmap_mode="r", allow_pickle=False)
                expected_shape = (num_steps, 4, self.manifest.num_patches, self.manifest.embed_dim)
                if feats_mmap.dtype != np.float16 or tuple(feats_mmap.shape) != expected_shape:
                    raise ValueError(
                        f"特征缓存 {feat_path} 形状/类型异常：actual=({feats_mmap.shape}, {feats_mmap.dtype}) "
                        f"!= expected=({expected_shape}, float16)"
                    )

            robot_hw = (int(robot_rgb.shape[2]), int(robot_rgb.shape[3]))
            self._episodes.append(
                _EpisodeMeta(
                    seed=seed,
                    npz_path=npz_path,
                    feat_path=feat_path,
                    num_steps=num_steps,
                    num_drones=num_drones,
                    T_W_B=np.array(episode["T_W_B"], dtype=np.float32, copy=True),
                    T_B_Crobot=np.array(episode["T_B_Crobot"], dtype=np.float32, copy=True),
                    K_ctx_scaled=_scale_robot_intrinsics(episode["K_robot"], robot_hw, encoder_cfg.res),
                    T_W_Cdrone=np.array(episode["T_W_Cdrone"], dtype=np.float32, copy=True),
                    K_drone=np.array(episode["K_drone"], dtype=np.float32, copy=True),
                )
            )
            for t in range(1, num_steps):
                for m in range(num_drones):
                    self._index.append((ep_idx, t, m))

        self._max_cached = max_cached_episodes
        self._rgb_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray | None]] = OrderedDict()

    @property
    def seeds(self) -> list[int]:
        return [ep.seed for ep in self._episodes]

    def __len__(self) -> int:
        return len(self._index)

    def _load_rgb(self, ep_idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        cached = self._rgb_cache.get(ep_idx)
        if cached is not None:
            self._rgb_cache.move_to_end(ep_idx)
            return cached
        ep = self._episodes[ep_idx]
        with np.load(ep.npz_path, allow_pickle=False) as loaded:
            drone_rgb = np.array(loaded["drone_rgb"], dtype=np.uint8, copy=True)
            robot_rgb = None if self.use_cache else np.array(loaded["robot_rgb"], dtype=np.uint8, copy=True)
        entry = (drone_rgb, robot_rgb)
        self._rgb_cache[ep_idx] = entry
        if len(self._rgb_cache) > self._max_cached:
            self._rgb_cache.popitem(last=False)
        return entry

    def __getitem__(self, index: int) -> Sample:
        ep_idx, t, m = self._index[index]
        ep = self._episodes[ep_idx]
        drone_rgb, robot_rgb = self._load_rgb(ep_idx)

        # 上下文步 {t-1, t} 的 4 路相机位姿统一换算到目标时刻 t 的 Base 系（SPEC §2.5）。
        T_Bt_Bprev = relative(ep.T_W_B[t], ep.T_W_B[t - 1])
        ctx_prev = np.einsum("ij,cjk->cik", T_Bt_Bprev, ep.T_B_Crobot).astype(np.float32)
        ctx_curr = ep.T_B_Crobot.copy()
        ctx_T_tgt = np.stack([ctx_prev, ctx_curr], axis=0)
        tgt_T = relative(ep.T_W_B[t], ep.T_W_Cdrone[t, m])

        ctx_feats: torch.Tensor | None = None
        ctx_rgb: torch.Tensor | None = None
        if self.use_cache:
            assert ep.feat_path is not None
            feats_mmap = np.load(ep.feat_path, mmap_mode="r", allow_pickle=False)
            ctx_feats = torch.from_numpy(np.array(feats_mmap[t - 1 : t + 1], dtype=np.float32, copy=True))
        else:
            assert robot_rgb is not None
            ctx_rgb = _uint8_batch_to_unit(robot_rgb[t - 1 : t + 1])

        return Sample(
            ctx_feats=ctx_feats,
            ctx_rgb=ctx_rgb,
            ctx_T_tgt=torch.from_numpy(ctx_T_tgt),
            K_ctx=torch.from_numpy(ep.K_ctx_scaled.copy()),
            tgt_T=torch.from_numpy(tgt_T),
            K_tgt=torch.from_numpy(ep.K_drone.copy()),
            tgt_rgb=_uint8_to_minus_one_one(drone_rgb[t, m]),
            prev_rgb=_uint8_to_minus_one_one(drone_rgb[t - 1, m]),
            meta={"seed": ep.seed, "t": t, "m": m},
        )


def make_dataset(
    cfg: Config,
    split: SplitName | None = "train",
    use_cache: bool = True,
    *,
    seeds: Sequence[int] | None = None,
    raw_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> EpisodeDataset:
    """按 Config 构建 EpisodeDataset；缓存模式自动解析并校验 manifest。"""
    resolved_raw = Path(raw_dir) if raw_dir is not None else Path(cfg.data.raw_dir)
    resolved_cache = (
        Path(cache_dir) if cache_dir is not None else (feature_cache_dir(cfg.data.feats_dir, cfg.encoder) if use_cache else None)
    )
    return EpisodeDataset(
        resolved_raw,
        cfg.encoder,
        split=None if seeds is not None else split,
        use_cache=use_cache,
        cache_dir=resolved_cache,
        seeds=seeds,
    )
