"""评估指标（LPIPS/PSNR/SSIM）与 M2 基线（copy_nearest_cam / regress）评估（SPEC §7.1–7.2、ARCH §5.6）。"""

import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rvz.config import Config
from rvz.data.dataset import Sample, SampleBatch, SplitName, collate_samples, make_dataset
from rvz.device import autocast_ctx, get_device
from rvz.models.regress import RegressBaseline

SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_C1 = 0.01**2
SSIM_C2 = 0.03**2
PSNR_MSE_FLOOR = 1e-10
_UNIT_RANGE_TOL = 1e-3


class LpipsEvaluator(Protocol):
    """LPIPS 评估协议（输入 `[-1, 1]` 范围的 `(B, 3, H, W)`，输出 `(B,)` 或可压平的每样本分数）。"""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float


@dataclass(frozen=True)
class EvalReport:
    name: str
    split: str
    num_samples: int
    lpips: MetricSummary
    psnr: MetricSummary
    ssim: MetricSummary
    platform: str
    git_hash: str | None
    timestamp: str
    checkpoint: str | None = None
    use_ema: bool | None = None
    rvz_version: str = ""


def to_unit_range(images: torch.Tensor) -> torch.Tensor:
    """将生成器/样本的 `[-1, 1]` 图像转换到评估所需的 `[0, 1]` 范围。"""
    return ((images.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def _prepare_unit_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """校验指标输入形状与 `[0, 1]` 值域，防止误传 `[-1, 1]`。"""
    if a.shape != b.shape:
        raise AssertionError(f"输入形状必须一致：{tuple(a.shape)} vs {tuple(b.shape)}")
    if a.ndim == 3:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    assert a.ndim == 4 and a.shape[1] == 3, f"期望 (B,3,H,W) 或 (3,H,W)，实际 {tuple(a.shape)}"
    a_f = a.float()
    b_f = b.float()
    assert torch.isfinite(a_f).all() and torch.isfinite(b_f).all(), "指标输入含非有限值"
    if float(a_f.min().item()) < -_UNIT_RANGE_TOL or float(a_f.max().item()) > 1.0 + _UNIT_RANGE_TOL:
        raise ValueError(f"a 必须在 [0, 1] 范围内，实际 [{a_f.min().item():.4f}, {a_f.max().item():.4f}]")
    if float(b_f.min().item()) < -_UNIT_RANGE_TOL or float(b_f.max().item()) > 1.0 + _UNIT_RANGE_TOL:
        raise ValueError(f"b 必须在 [0, 1] 范围内，实际 [{b_f.min().item():.4f}, {b_f.max().item():.4f}]")
    return a_f.clamp(0.0, 1.0), b_f.clamp(0.0, 1.0)


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """在 `[0, 1]` 范围计算逐样本 PSNR（dB），形状 `(B,)`。"""
    x, y = _prepare_unit_pair(a, b)
    mse = (x - y).square().mean(dim=(1, 2, 3)).clamp(min=PSNR_MSE_FLOOR)
    return -10.0 * torch.log10(mse)


def _gaussian_window(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(SSIM_WINDOW_SIZE, device=device, dtype=dtype) - (SSIM_WINDOW_SIZE - 1) / 2.0
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * SSIM_SIGMA * SSIM_SIGMA))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, SSIM_WINDOW_SIZE, SSIM_WINDOW_SIZE).expand(3, 1, SSIM_WINDOW_SIZE, SSIM_WINDOW_SIZE)


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """按 SPEC §7.1（Gaussian 11×11，σ=1.5，RGB 通道均值）计算逐样本 SSIM，形状 `(B,)`。"""
    x, y = _prepare_unit_pair(a, b)
    window = _gaussian_window(x.device, x.dtype)
    pad = SSIM_WINDOW_SIZE // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=3)
    mu_y = F.conv2d(y, window, padding=pad, groups=3)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x2 = (F.conv2d(x * x, window, padding=pad, groups=3) - mu_x2).clamp(min=0.0)
    sigma_y2 = (F.conv2d(y * y, window, padding=pad, groups=3) - mu_y2).clamp(min=0.0)
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=3) - mu_xy
    numerator = (2.0 * mu_xy + SSIM_C1) * (2.0 * sigma_xy + SSIM_C2)
    denominator = (mu_x2 + mu_y2 + SSIM_C1) * (sigma_x2 + sigma_y2 + SSIM_C2)
    return (numerator / denominator).mean(dim=(1, 2, 3))


def make_lpips_metric(device: torch.device) -> LpipsEvaluator:
    """构建冻结的 VGG LPIPS 评估器。"""
    net = lpips.LPIPS(net="vgg", verbose=False).to(device)
    net.requires_grad_(False)
    net.eval()
    return net


def lpips_score(a: torch.Tensor, b: torch.Tensor, *, metric: LpipsEvaluator | None = None) -> torch.Tensor:
    """在 `[0, 1]` 范围计算逐样本 VGG LPIPS，形状 `(B,)`。"""
    x, y = _prepare_unit_pair(a, b)
    evaluator = metric if metric is not None else make_lpips_metric(x.device)
    with torch.no_grad():
        raw = evaluator(x * 2.0 - 1.0, y * 2.0 - 1.0)
    return raw.reshape(x.shape[0]).float()


def baseline_copy_nearest_cam(sample: Sample) -> torch.Tensor:
    """SPEC §7.2 基线：选当前时刻光轴与无人机夹角最小的小车相机，缩放到无人机分辨率并输出 `[-1, 1]`。"""
    assert sample.ctx_rgb is not None, "copy_nearest_cam 需要 use_cache=False 的 ctx_rgb"
    # OpenCV 相机系光轴为 +z，即位姿矩阵旋转部分的第 3 列。
    ctx_axes = sample.ctx_T_tgt[1, :, :3, 2]
    tgt_axis = sample.tgt_T[:3, 2]
    cos_sim = ctx_axes @ tgt_axis
    nearest_cam = int(torch.argmax(cos_sim).item())
    selected = sample.ctx_rgb[1, nearest_cam].unsqueeze(0)
    target_hw = (int(sample.tgt_rgb.shape[1]), int(sample.tgt_rgb.shape[2]))
    if selected.shape[-2:] != target_hw:
        selected = F.interpolate(selected, size=target_hw, mode="bilinear", align_corners=False)
    return selected.squeeze(0).mul(2.0).sub(1.0).clamp(-1.0, 1.0)


def _git_hash() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _summarize(values: list[float]) -> MetricSummary:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    assert math.isfinite(mean) and math.isfinite(std)
    return MetricSummary(mean=round(mean, 6), std=round(std, 6))


def _write_report(report: EvalReport, out_dir: str | Path | None) -> None:
    if out_dir is None:
        return
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def evaluate_copy_nearest_cam(
    cfg: Config,
    *,
    split: SplitName = "val",
    seeds: Sequence[int] | None = None,
    out_dir: str | Path | None = None,
    max_samples: int | None = None,
    lpips_metric: LpipsEvaluator | None = None,
    device: torch.device | None = None,
) -> EvalReport:
    """在指定划分上计算 `copy_nearest_cam` 基线的全部指标并可选写入 report.json。"""
    target_device = device if device is not None else get_device()
    dataset = make_dataset(cfg, split=split, use_cache=False, seeds=seeds)
    metric = lpips_metric if lpips_metric is not None else make_lpips_metric(target_device)

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    batch_size = min(cfg.train.batch, max(1, total))
    lpips_vals: list[float] = []
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    preds_buf: list[torch.Tensor] = []
    tgts_buf: list[torch.Tensor] = []

    for idx in range(total):
        sample = dataset[idx]
        preds_buf.append(baseline_copy_nearest_cam(sample))
        tgts_buf.append(sample.tgt_rgb)
        if len(preds_buf) == batch_size or idx == total - 1:
            pred_unit = to_unit_range(torch.stack(preds_buf, dim=0).to(target_device))
            tgt_unit = to_unit_range(torch.stack(tgts_buf, dim=0).to(target_device))
            lpips_vals.extend(lpips_score(pred_unit, tgt_unit, metric=metric).cpu().tolist())
            psnr_vals.extend(psnr(pred_unit, tgt_unit).cpu().tolist())
            ssim_vals.extend(ssim(pred_unit, tgt_unit).cpu().tolist())
            preds_buf.clear()
            tgts_buf.clear()

    report = EvalReport(
        name="copy_nearest_cam",
        split=split if seeds is None else f"seeds({len(seeds)})",
        num_samples=total,
        lpips=_summarize(lpips_vals),
        psnr=_summarize(psnr_vals),
        ssim=_summarize(ssim_vals),
        platform=f"{sys.platform}-{platform.machine()}",
        git_hash=_git_hash(),
        timestamp=datetime.now(UTC).isoformat(),
        rvz_version=importlib.metadata.version("reality-view-zero"),
    )
    _write_report(report, out_dir)
    return report


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    cfg: Config,
    *,
    split: SplitName = "val",
    seeds: Sequence[int] | None = None,
    out_dir: str | Path | None = None,
    use_ema: bool = True,
    max_samples: int | None = None,
    lpips_metric: LpipsEvaluator | None = None,
    device: torch.device | None = None,
) -> EvalReport:
    """加载 checkpoint（默认用 EMA 权重）并在指定划分上计算 LPIPS/PSNR/SSIM。"""
    ckpt_file = Path(checkpoint_path)
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    stage = str(checkpoint["stage"])
    if stage != "regress":
        raise ValueError(f"当前仅支持评估 regress checkpoint，收到 stage={stage!r}")

    target_device = device if device is not None else get_device()
    model_meta = checkpoint["model_meta"]
    model = RegressBaseline(**model_meta).to(target_device)
    ema_state = checkpoint["ema"]
    using_ema = bool(use_ema and ema_state is not None)
    weights = ema_state if using_ema else checkpoint["model"]
    model.load_state_dict(weights)
    model.eval()

    dataset = make_dataset(cfg, split=split, use_cache=True, seeds=seeds)
    metric = lpips_metric if lpips_metric is not None else make_lpips_metric(target_device)
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    loader: DataLoader[SampleBatch] = DataLoader(
        dataset,  # type: ignore[arg-type]  # collate_samples 将 Sample 聚合为 SampleBatch
        batch_size=min(cfg.train.batch, max(1, total)),
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_samples,
    )

    lpips_vals: list[float] = []
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= total:
                break
            remaining = total - seen
            batch = batch.to(target_device)
            assert batch.ctx_feats is not None
            with autocast_ctx(target_device):
                pred = model(
                    batch.ctx_feats[:remaining],
                    batch.ctx_T_tgt[:remaining],
                    batch.tgt_T[:remaining],
                    batch.K_tgt[:remaining],
                )
            pred_unit = to_unit_range(pred)
            tgt_unit = to_unit_range(batch.tgt_rgb[:remaining])
            lpips_vals.extend(lpips_score(pred_unit, tgt_unit, metric=metric).cpu().tolist())
            psnr_vals.extend(psnr(pred_unit, tgt_unit).cpu().tolist())
            ssim_vals.extend(ssim(pred_unit, tgt_unit).cpu().tolist())
            seen += int(pred_unit.shape[0])

    report = EvalReport(
        name=stage,
        split=split if seeds is None else f"seeds({len(seeds)})",
        num_samples=seen,
        lpips=_summarize(lpips_vals),
        psnr=_summarize(psnr_vals),
        ssim=_summarize(ssim_vals),
        platform=f"{sys.platform}-{platform.machine()}",
        git_hash=_git_hash(),
        timestamp=datetime.now(UTC).isoformat(),
        checkpoint=str(ckpt_file),
        use_ema=using_ema,
        rvz_version=importlib.metadata.version("reality-view-zero"),
    )
    _write_report(report, out_dir)
    return report
