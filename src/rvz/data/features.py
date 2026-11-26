"""离线编码器特征缓存与 manifest 严格校验（SPEC §3.4、ARCH §5.3）。"""

import importlib.metadata
import json
import logging
import math
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from rvz.config import SCHEMA_VERSION, EncoderConfig
from rvz.data.schema import validate_episode
from rvz.device import get_device

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
_EPISODE_PATTERN = re.compile(r"^ep_(\d+)\.npz$")


class PatchEncoder(Protocol):
    """特征缓存依赖的最小编码器协议，避免 data 反向依赖 models（ARCH §3）。"""

    encoder_id: str
    revision: str
    input_res: int
    patch_size: int
    pool_2x2: bool
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @property
    def num_patches(self) -> int: ...

    @property
    def embed_dim(self) -> int: ...

    def encode(self, rgb: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class Manifest:
    """特征缓存元数据；与当前配置或编码器不一致时必须拒绝加载（SPEC §3.4）。"""

    encoder_id: str
    revision: str
    input_res: int
    patch_size: int
    pool_2x2: bool
    num_patches: int
    embed_dim: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    rvz_version: str
    schema_version: int


def current_rvz_version() -> str:
    return importlib.metadata.version("reality-view-zero")


def parse_episode_seed(path: Path) -> int | None:
    """从 ``ep_XXXXXX.npz`` 文件名解析非负整数 seed。"""
    match = _EPISODE_PATTERN.match(path.name)
    return int(match.group(1)) if match else None


def feature_cache_dir(feats_dir: str | Path, cfg: EncoderConfig) -> Path:
    """生成不含路径分隔符的确定性缓存目录：``{feats_dir}/{encoder_id}_{res}[_p2]``。"""
    safe_id = cfg.id.replace("/", "__")
    suffix = "_p2" if cfg.pool_2x2 else ""
    return Path(feats_dir) / f"{safe_id}_{cfg.res}{suffix}"


def manifest_from_encoder(encoder: PatchEncoder) -> Manifest:
    return Manifest(
        encoder_id=encoder.encoder_id,
        revision=encoder.revision,
        input_res=encoder.input_res,
        patch_size=encoder.patch_size,
        pool_2x2=encoder.pool_2x2,
        num_patches=encoder.num_patches,
        embed_dim=encoder.embed_dim,
        mean=tuple(float(v) for v in encoder.mean),  # type: ignore[arg-type]  # 固定 3 通道均值
        std=tuple(float(v) for v in encoder.std),  # type: ignore[arg-type]  # 固定 3 通道标准差
        rvz_version=current_rvz_version(),
        schema_version=SCHEMA_VERSION,
    )


def _float_tuple_close(actual: tuple[float, float, float], expected: tuple[float, float, float]) -> bool:
    return all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6) for a, b in zip(actual, expected, strict=True))


def _check_manifest_equal(actual: Manifest, expected: Manifest) -> None:
    mismatches: list[str] = []
    comparisons = (
        ("encoder_id", actual.encoder_id, expected.encoder_id),
        ("revision", actual.revision, expected.revision),
        ("input_res", actual.input_res, expected.input_res),
        ("patch_size", actual.patch_size, expected.patch_size),
        ("pool_2x2", actual.pool_2x2, expected.pool_2x2),
        ("num_patches", actual.num_patches, expected.num_patches),
        ("embed_dim", actual.embed_dim, expected.embed_dim),
        ("rvz_version", actual.rvz_version, expected.rvz_version),
        ("schema_version", actual.schema_version, expected.schema_version),
    )
    for field_name, left, right in comparisons:
        if left != right:
            mismatches.append(f"{field_name}: actual={left!r} != expected={right!r}")
    if not _float_tuple_close(actual.mean, expected.mean):
        mismatches.append(f"mean: actual={actual.mean} != expected={expected.mean}")
    if not _float_tuple_close(actual.std, expected.std):
        mismatches.append(f"std: actual={actual.std} != expected={expected.std}")
    if mismatches:
        raise ValueError(f"特征缓存 manifest 不匹配：{'; '.join(mismatches)}")


def _parse_manifest_json(path: Path) -> Manifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest 顶层应为对象：{path}")
    required = {
        "encoder_id",
        "revision",
        "input_res",
        "patch_size",
        "pool_2x2",
        "num_patches",
        "embed_dim",
        "mean",
        "std",
        "rvz_version",
        "schema_version",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"manifest 缺少字段：{', '.join(missing)}")
    mean = raw["mean"]
    std = raw["std"]
    if not isinstance(mean, list) or len(mean) != 3 or not isinstance(std, list) or len(std) != 3:
        raise ValueError("manifest.mean 与 manifest.std 必须为长度为 3 的数值列表")
    return Manifest(
        encoder_id=str(raw["encoder_id"]),
        revision=str(raw["revision"]),
        input_res=int(raw["input_res"]),
        patch_size=int(raw["patch_size"]),
        pool_2x2=bool(raw["pool_2x2"]),
        num_patches=int(raw["num_patches"]),
        embed_dim=int(raw["embed_dim"]),
        mean=(float(mean[0]), float(mean[1]), float(mean[2])),
        std=(float(std[0]), float(std[1]), float(std[2])),
        rvz_version=str(raw["rvz_version"]),
        schema_version=int(raw["schema_version"]),
    )


def load_manifest(
    cache_dir: str | Path,
    expected: Manifest | None = None,
    *,
    cfg: EncoderConfig | None = None,
) -> Manifest:
    """加载并校验 manifest.json；版本、配置或编码器不匹配时抛出 ValueError。"""
    manifest_path = Path(cache_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"特征缓存缺少 {MANIFEST_FILENAME}：{manifest_path}")
    manifest = _parse_manifest_json(manifest_path)

    rvz_version = current_rvz_version()
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"特征缓存 schema_version 不匹配：actual={manifest.schema_version} != expected={SCHEMA_VERSION}"
        )
    if manifest.rvz_version != rvz_version:
        raise ValueError(f"特征缓存 rvz_version 不匹配：actual={manifest.rvz_version!r} != expected={rvz_version!r}")
    if expected is not None:
        _check_manifest_equal(manifest, expected)
    if cfg is not None:
        mismatches: list[str] = []
        if manifest.encoder_id != cfg.id:
            mismatches.append(f"encoder_id: actual={manifest.encoder_id!r} != expected={cfg.id!r}")
        if manifest.input_res != cfg.res:
            mismatches.append(f"input_res: actual={manifest.input_res} != expected={cfg.res}")
        if manifest.pool_2x2 != cfg.pool_2x2:
            mismatches.append(f"pool_2x2: actual={manifest.pool_2x2} != expected={cfg.pool_2x2}")
        if mismatches:
            raise ValueError(f"特征缓存与 EncoderConfig 不匹配：{'; '.join(mismatches)}")
    return manifest


def _write_manifest(cache_dir: Path, manifest: Manifest) -> None:
    path = cache_dir / MANIFEST_FILENAME
    tmp_path = cache_dir / f".{MANIFEST_FILENAME}.tmp"
    payload = asdict(manifest)
    payload["mean"] = list(manifest.mean)
    payload["std"] = list(manifest.std)
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.npy")
    np.save(tmp_path, array)
    os.replace(tmp_path, path)


def _existing_cache_valid(path: Path, expected_shape: tuple[int, int, int, int]) -> bool:
    if not path.exists():
        return False
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return arr.dtype == np.float16 and tuple(arr.shape) == expected_shape


def _list_raw_episodes(raw_dir: Path, seeds: Sequence[int] | None) -> list[tuple[int, Path]]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"原始 episode 目录不存在：{raw_dir}")
    discovered: dict[int, Path] = {}
    for entry in raw_dir.iterdir():
        if entry.is_file():
            seed = parse_episode_seed(entry)
            if seed is not None:
                discovered[seed] = entry
    if seeds is not None:
        missing = [seed for seed in seeds if seed not in discovered]
        if missing:
            raise FileNotFoundError(f"原始目录 {raw_dir} 缺少 seed：{missing[:10]}")
        return [(seed, discovered[seed]) for seed in sorted(set(seeds))]
    return sorted(discovered.items())


def cache_features(
    raw_dir: str | Path,
    out_dir: str | Path,
    encoder: PatchEncoder,
    cfg: EncoderConfig,
    *,
    seeds: Sequence[int] | None = None,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> dict[str, int]:
    """幂等缓存 4 路小车相机 patch 特征（fp16，[T, 4, N, C]）并写入 manifest.json。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if encoder.encoder_id != cfg.id or encoder.input_res != cfg.res or encoder.pool_2x2 != cfg.pool_2x2:
        raise ValueError(
            "编码器属性与 EncoderConfig 不一致："
            f"encoder=({encoder.encoder_id}, {encoder.input_res}, {encoder.pool_2x2}), "
            f"cfg=({cfg.id}, {cfg.res}, {cfg.pool_2x2})"
        )

    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    expected_manifest = manifest_from_encoder(encoder)
    manifest_path = out_path / MANIFEST_FILENAME
    if manifest_path.exists():
        load_manifest(out_path, expected=expected_manifest)
    else:
        _write_manifest(out_path, expected_manifest)

    episodes = _list_raw_episodes(raw_path, seeds)
    if not episodes:
        raise FileNotFoundError(f"未在 {raw_path} 找到任何 ep_*.npz 文件")

    target_device = device if device is not None else get_device()
    if isinstance(encoder, torch.nn.Module):
        encoder.to(target_device)
    cached_count = 0
    skipped_count = 0
    for seed, ep_path in episodes:
        feat_path = out_path / f"ep_{seed:06d}.npy"
        with np.load(ep_path, allow_pickle=False) as loaded:
            episode = {name: loaded[name] for name in loaded.files}
        validate_episode(episode)
        robot_rgb = episode["robot_rgb"]
        num_steps = int(robot_rgb.shape[0])
        expected_shape = (num_steps, 4, encoder.num_patches, encoder.embed_dim)
        if _existing_cache_valid(feat_path, expected_shape):
            skipped_count += 1
            continue

        flat_rgb = robot_rgb.reshape(num_steps * 4, robot_rgb.shape[2], robot_rgb.shape[3], 3)
        chunks: list[np.ndarray] = []
        for start in range(0, len(flat_rgb), batch_size):
            batch_np = flat_rgb[start : start + batch_size]
            batch_tensor = (
                torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device=target_device, dtype=torch.float32) / 255.0
            )
            tokens = encoder.encode(batch_tensor)
            assert tokens.shape == (len(batch_np), encoder.num_patches, encoder.embed_dim), (
                f"编码输出形状异常：{tuple(tokens.shape)}"
            )
            assert torch.isfinite(tokens).all(), f"episode {seed} 特征含非有限值"
            chunks.append(tokens.cpu().numpy().astype(np.float16))
        features = np.concatenate(chunks, axis=0).reshape(expected_shape)
        _atomic_save_npy(feat_path, features)
        cached_count += 1
        logger.info("已缓存特征 ep_%06d.npy，shape=%s", seed, expected_shape)

    return {"total": len(episodes), "cached": cached_count, "skipped": skipped_count}
