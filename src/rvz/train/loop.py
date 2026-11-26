"""统一训练循环：所有阶段共用同一循环，支持 overfit、resume、EMA 与 JSONL 日志（SPEC §6.4、ARCH §5.5）。"""

import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rvz.config import Config, dump_config
from rvz.data.dataset import SampleBatch, collate_samples, make_dataset
from rvz.device import autocast_ctx, get_device, seed_everything
from rvz.models.regress import RegressBaseline
from rvz.train.losses import LpipsFn, LpipsLoss, regress_loss_breakdown

logger = logging.getLogger(__name__)

StageName = Literal["regress"]
CheckpointValue = (
    None | bool | int | float | str | torch.Tensor | list["CheckpointValue"] | dict[str, "CheckpointValue"]
)


class ExponentialMovingAverage:
    """维护模型参数的指数移动平均权重（FR-TRN-3）。"""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay 必须在 (0, 1) 内：{decay}")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().float().clone() for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        # 早期按 (1 + step) / (10 + step) 预热，避免短步数训练时 EMA 长期停留在随机初始化权重。
        effective_decay = min(self.decay, (1.0 + step) / (10.0 + step))
        for name, param in model.named_parameters():
            self.shadow[name].mul_(effective_decay).add_(param.detach().float(), alpha=1.0 - effective_decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu().clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor], device: torch.device) -> None:
        expected = set(self.shadow)
        actual = set(state)
        if actual != expected:
            raise ValueError(f"EMA 权重键不匹配：缺少 {sorted(expected - actual)}，多余 {sorted(actual - expected)}")
        for name, tensor in state.items():
            self.shadow[name] = tensor.to(device=device, dtype=torch.float32).clone()


def _git_hash() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "rvz": importlib.metadata.version("reality-view-zero"),
    }


def _atomic_save_checkpoint(path: Path, payload: dict[str, CheckpointValue]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _infinite_batches(loader: DataLoader[SampleBatch]) -> Iterator[SampleBatch]:
    while True:
        yield from loader


def run(
    cfg: Config,
    stage: StageName,
    run_dir: str | Path,
    *,
    resume: bool = False,
    overfit_seed: int | None = None,
    lpips_fn: LpipsFn | None = None,
    device: torch.device | None = None,
) -> Path:
    """运行指定阶段训练，写入 config.yaml、log.jsonl 与 ckpt/latest.pt。"""
    if stage != "regress":
        raise ValueError(f"当前里程碑仅支持 stage='regress'，收到：{stage!r}")

    run_path = Path(run_dir)
    ckpt_dir = run_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_yaml = dump_config(cfg)
    (run_path / "config.yaml").write_text(config_yaml, encoding="utf-8")

    seed_everything(cfg.train.seed)
    target_device = device if device is not None else get_device()

    seeds = [overfit_seed] if overfit_seed is not None else None
    dataset = make_dataset(cfg, split="train", use_cache=True, seeds=seeds)
    assert dataset.manifest is not None

    model = RegressBaseline(
        feat_dim=dataset.manifest.embed_dim,
        num_patches=dataset.manifest.num_patches,
        drone_res=cfg.sim.drone_res,
        dim=cfg.model.dim,
        heads=cfg.model.heads,
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    ema = ExponentialMovingAverage(model, cfg.train.ema) if cfg.train.ema > 0.0 else None

    active_lpips: LpipsFn | None = lpips_fn
    if active_lpips is None and cfg.train.lpips_weight > 0.0:
        active_lpips = LpipsLoss().to(target_device)

    start_step = 0
    latest_path = ckpt_dir / "latest.pt"
    if resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"无法 resume，缺少 checkpoint：{latest_path}")
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        if checkpoint["stage"] != stage:
            raise ValueError(f"checkpoint stage={checkpoint['stage']!r} 与当前 stage={stage!r} 不一致")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if ema is not None and checkpoint["ema"] is not None:
            ema.load_state_dict(checkpoint["ema"], target_device)
        start_step = int(checkpoint["step"])
        logger.info("从 %s 恢复训练，起始步 %d", latest_path, start_step)

    generator = torch.Generator().manual_seed(cfg.train.seed + start_step)
    batch_size = min(cfg.train.batch, len(dataset))
    loader: DataLoader[SampleBatch] = DataLoader(
        dataset,  # type: ignore[arg-type]  # collate_samples 将 Sample 聚合为 SampleBatch
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_samples,
        generator=generator,
    )
    batches = _infinite_batches(loader)
    log_path = run_path / "log.jsonl"
    git_hash = _git_hash()
    versions = _runtime_versions()

    model.train()
    for step in range(start_step + 1, cfg.train.steps + 1):
        step_start = time.perf_counter()
        batch = next(batches).to(target_device)
        assert batch.ctx_feats is not None

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(target_device):
            pred = model(batch.ctx_feats, batch.ctx_T_tgt, batch.tgt_T, batch.K_tgt)
            loss, mse_val, lpips_val = regress_loss_breakdown(
                pred,
                batch.tgt_rgb,
                lpips_fn=active_lpips,
                lpips_weight=cfg.train.lpips_weight,
            )
        assert torch.isfinite(loss), f"step {step} 出现非有限损失：{loss.item()}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model, step)

        step_time = time.perf_counter() - step_start
        if step == 1 or step % cfg.train.log_every == 0 or step == cfg.train.steps:
            record = {
                "step": step,
                "stage": stage,
                "loss": float(loss.detach().item()),
                "mse": mse_val,
                "lpips": lpips_val,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "step_time_s": round(step_time, 4),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                "step %d/%d loss=%.4f mse=%.4f lpips=%.4f (%.3fs)",
                step,
                cfg.train.steps,
                record["loss"],
                mse_val,
                lpips_val,
                step_time,
            )

        if step % cfg.train.save_every == 0 or step == cfg.train.steps:
            payload = {
                "stage": stage,
                "step": step,
                "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                "ema": ema.state_dict() if ema is not None else None,
                "optimizer": optimizer.state_dict(),
                "model_meta": model.model_meta(),
                "config": config_yaml,
                "git_hash": git_hash,
                "versions": versions,
            }
            _atomic_save_checkpoint(ckpt_dir / f"step_{step:07d}.pt", payload)
            _atomic_save_checkpoint(latest_path, payload)

    return latest_path
