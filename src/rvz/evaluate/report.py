"""采样对比图导出（上下文 4 图 | 预测 | 真值 | 差异，SPEC §7.6、§9）。"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from rvz.config import Config
from rvz.data.dataset import collate_samples, make_dataset
from rvz.data.schema import CAM_ORDER
from rvz.device import get_device
from rvz.evaluate.metrics import (
    LpipsEvaluator,
    load_checkpoint_model,
    lpips_score,
    make_lpips_metric,
    predict_batch,
    psnr,
    ssim,
    to_unit_range,
)


def _tensor_to_pil(unit_chw: torch.Tensor) -> Image.Image:
    arr = (unit_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _render_context_grid(robot_rgb_step: np.ndarray, tile_size: int) -> Image.Image:
    grid = Image.new("RGB", (tile_size * 2, tile_size * 2), "black")
    for cam_idx, cam_name in enumerate(CAM_ORDER):
        img = Image.fromarray(robot_rgb_step[cam_idx])
        if img.size != (tile_size, tile_size):
            img = img.resize((tile_size, tile_size), Image.Resampling.NEAREST)
        x = (cam_idx % 2) * tile_size
        y = (cam_idx // 2) * tile_size
        grid.paste(img, (x, y))
        ImageDraw.Draw(grid).text((x + 4, y + 4), cam_name, fill="white", stroke_width=1, stroke_fill="black")
    return grid


def sample_checkpoint(
    checkpoint_path: str | Path,
    cfg: Config,
    *,
    seed: int,
    step: int = 1,
    drone_index: int = 0,
    steps: int = 50,
    out_dir: str | Path = "runs/samples/latest",
    use_ema: bool = True,
    sample_seed: int = 0,
    lpips_metric: LpipsEvaluator | None = None,
    device: torch.device | None = None,
) -> dict[str, Path]:
    """用指定 checkpoint 对 `(seed, step, drone_index)` 采样并导出对比图与指标 JSON。"""
    if step < 1:
        raise ValueError(f"step 必须 >= 1（需要 {{t-1, t}} 上下文）：{step}")
    if drone_index < 0:
        raise ValueError(f"drone_index 必须非负：{drone_index}")

    target_device = device if device is not None else get_device()
    stage, model, using_ema = load_checkpoint_model(checkpoint_path, use_ema=use_ema, device=target_device)

    dataset = make_dataset(cfg, split=None, use_cache=True, seeds=[seed])
    sample_idx = dataset.find_index(seed, step, drone_index)
    sample_item = dataset[sample_idx]
    batch = collate_samples([sample_item]).to(target_device)
    sample_gen = torch.Generator().manual_seed(sample_seed)
    with torch.no_grad():
        pred = predict_batch(stage, model, batch, steps=steps, generator=sample_gen)

    pred_unit = to_unit_range(pred)
    tgt_unit = to_unit_range(batch.tgt_rgb)
    diff_unit = (pred_unit - tgt_unit).abs().mul(2.0).clamp(0.0, 1.0)

    metric = lpips_metric if lpips_metric is not None else make_lpips_metric(target_device)
    lpips_val = float(lpips_score(pred_unit, tgt_unit, metric=metric).item())
    psnr_val = float(psnr(pred_unit, tgt_unit).item())
    ssim_val = float(ssim(pred_unit, tgt_unit).item())

    npz_path = Path(cfg.data.raw_dir) / f"ep_{seed:06d}.npz"
    with np.load(npz_path, allow_pickle=False) as loaded:
        robot_rgb_step = np.array(loaded["robot_rgb"][step], dtype=np.uint8, copy=True)

    pred_img = _tensor_to_pil(pred_unit[0])
    gt_img = _tensor_to_pil(tgt_unit[0])
    diff_img = _tensor_to_pil(diff_unit[0])

    display_size = max(256, pred_img.width, pred_img.height)
    context_img = _render_context_grid(robot_rgb_step, display_size // 2)
    pred_disp = pred_img.resize((display_size, display_size), Image.Resampling.NEAREST)
    gt_disp = gt_img.resize((display_size, display_size), Image.Resampling.NEAREST)
    diff_disp = diff_img.resize((display_size, display_size), Image.Resampling.NEAREST)

    header_h = 36
    comparison = Image.new("RGB", (display_size * 4, display_size + header_h), (24, 24, 24))
    draw = ImageDraw.Draw(comparison)
    pred_label = f"2. Pred ({stage}, {steps} steps)" if stage == "fm" else f"2. Pred ({stage})"
    panels = (
        (f"1. Context 4 cams (seed={seed}, t={step})", context_img),
        (pred_label, pred_disp),
        (f"3. Ground Truth (m={drone_index})", gt_disp),
        ("4. Abs Diff (x2)", diff_disp),
    )
    for panel_idx, (title, img) in enumerate(panels):
        x = panel_idx * display_size
        draw.text((x + 6, 4), title, fill="white")
        comparison.paste(img, (x, header_h))
    draw.text(
        (6, 19),
        f"LPIPS={lpips_val:.4f} | PSNR={psnr_val:.2f} dB | SSIM={ssim_val:.4f} | ema={using_ema}",
        fill=(200, 230, 255),
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pred": out_path / "pred.png",
        "gt": out_path / "gt.png",
        "diff": out_path / "diff.png",
        "comparison": out_path / "comparison.png",
        "metrics": out_path / "metrics.json",
    }
    pred_img.save(outputs["pred"])
    gt_img.save(outputs["gt"])
    diff_img.save(outputs["diff"])
    comparison.save(outputs["comparison"])
    metrics_payload = {
        "checkpoint": str(Path(checkpoint_path)),
        "stage": stage,
        "use_ema": using_ema,
        "seed": seed,
        "step": step,
        "drone_index": drone_index,
        "sample_steps": steps if stage == "fm" else None,
        "sample_seed": sample_seed,
        "lpips": round(lpips_val, 6),
        "psnr": round(psnr_val, 6),
        "ssim": round(ssim_val, 6),
    }
    outputs["metrics"].write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs
