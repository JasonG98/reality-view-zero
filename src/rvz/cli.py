"""rvz 命令行入口（ARCH §9）；只做参数解析与调度，不含业务逻辑。"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import tyro

# MuJoCo EGL 必须在 import mujoco 之前设置（ARCH §7），故入口先设环境再导入业务模块。
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from rvz.config import load_config  # 须晚于 MUJOCO_GL 设置（import 链会拉入 mujoco）
from rvz.data.dataset import SplitName
from rvz.data.features import cache_features, feature_cache_dir
from rvz.data.viz import viz_episode
from rvz.doctor import run_doctor  # 须晚于 MUJOCO_GL 设置
from rvz.evaluate.metrics import EvalReport, evaluate_checkpoint, evaluate_copy_nearest_cam
from rvz.evaluate.report import sample_checkpoint
from rvz.models.encoder import FrozenEncoder
from rvz.sim.record import generate_many
from rvz.train.loop import StageName
from rvz.train.loop import run as run_train
from rvz.viewer import run_viewer

_STATUS_ICON = {"pass": "✅", "fail": "❌", "skip": "⚠️"}
BaselineName = Literal["copy_nearest_cam"]


@dataclass(frozen=True)
class DoctorCommand:
    """M0 环境/可行性检查，写 runs/doctor_<platform>.json。

    Args:
        config: YAML 配置路径；缺省用内置默认值。
        overrides: 覆盖配置项。
        download: 启用需联网的检查（DINOv3 前向、lpips 权重下载）。
    """

    config: str | None = None
    overrides: Annotated[
        tyro.conf.UseAppendAction[list[str]],
        tyro.conf.arg(name="set", metavar="a.b.c=value"),
    ] = field(default_factory=list)
    download: bool = False


@dataclass(frozen=True)
class GenDataCommand:
    """生成 half-open seed 范围内的 episode。

    Args:
        seeds: 范围 a:b，例如 0:20 生成 seed 0 到 19。
        config: YAML 配置路径。
        out: 输出目录，缺省为 data.raw_dir。
        workers: 进程数，每个进程独立渲染器。
        overrides: 覆盖配置项。
    """

    seeds: str
    config: str | None = None
    out: str | None = None
    workers: int = 1
    overrides: Annotated[
        tyro.conf.UseAppendAction[list[str]],
        tyro.conf.arg(name="set", metavar="a.b.c=value"),
    ] = field(default_factory=list)


@dataclass(frozen=True)
class VizEpisodeCommand:
    """导出一个 episode 的上下文接触表与位姿重投影检查图。

    Args:
        episode: episode npz 路径。
        config: YAML 配置路径。
        out: 图片输出目录。
        step: 时刻下标。
        drone_index: 无人机下标。
        overrides: 覆盖配置项。
    """

    episode: str
    config: str | None = None
    out: str = "runs/viz/latest"
    step: int = 0
    drone_index: int = 0
    overrides: Annotated[
        tyro.conf.UseAppendAction[list[str]],
        tyro.conf.arg(name="set", metavar="a.b.c=value"),
    ] = field(default_factory=list)


@dataclass(frozen=True)
class ViewerCommand:
    """实时显示左侧小车四路相机和右侧无人机视角。

    Args:
        seed: 仓库布局与小车轨迹的非负随机种子。
        config: YAML 配置路径，默认使用 configs/viewer.yaml。
        drone_index: 无人机下标，从 0 开始。
        overrides: 覆盖配置项。
    """

    seed: int = 0
    config: str = str(Path(__file__).with_name("viewer.yaml").resolve())
    drone_index: int = 0
    overrides: Annotated[
        tyro.conf.UseAppendAction[list[str]],
        tyro.conf.arg(name="set", metavar="a.b.c=value"),
    ] = field(default_factory=list)


@dataclass(frozen=True)
class CacheFeatsCommand:
    """离线提取并幂等缓存冻结编码器 patch 特征（写 manifest.json）。

    Args:
        config: YAML 配置路径。
        raw_dir: 原始 episode 目录，缺省为 data.raw_dir。
        out: 特征缓存目录，缺省为 data.feats_dir/{encoder}_{res}。
        seeds: 可选 half-open seed 范围 a:b，仅缓存子集。
        batch_size: 编码器批大小。
        overrides: 覆盖配置项。
    """

    config: str | None = None
    raw_dir: str | None = None
    out: str | None = None
    seeds: str | None = None
    batch_size: int = 32
    overrides: Annotated[
        tyro.conf.UseAppendAction[list[str]],
        tyro.conf.arg(name="set", metavar="a.b.c=value"),
    ] = field(default_factory=list)


@dataclass(frozen=True)
class TrainCommand:
    """统一训练入口（SPEC §6、§9）。

    Args:
        stage: 训练阶段（regress / fm）。
        run: 运行名或目录（单个名称自动写入 runs/<name>）。
        config: YAML 配置路径。
        resume: 从 <run_dir>/ckpt/latest.pt 恢复训练。
        overfit_episode: 仅使用指定 seed 的单个 episode 训练。
        overrides: 覆盖配置项。
    """

    stage: StageName = "regress"
    run: str = "regress"
    config: str | None = None
    resume: bool = False
    overfit_episode: int | None = None
    overrides: Annotated[
        tyro.conf.UseAppendAction[list[str]],
        tyro.conf.arg(name="set", metavar="a.b.c=value"),
    ] = field(default_factory=list)


@dataclass(frozen=True)
class SampleCommand:
    """用 checkpoint 对指定 episode/step/drone 采样并导出对比图（SPEC §9）。

    Args:
        checkpoint: checkpoint 路径。
        seed: episode 随机种子。
        step: 时刻下标（>= 1）。
        drone_index: 无人机下标（从 0 开始）。
        steps: fm 采样步数（默认 50）。
        out: 输出目录。
        use_ema: 优先使用 EMA 权重。
        sample_seed: 初始高斯噪声随机种子。
        config: YAML 配置路径。
        overrides: 覆盖配置项。
    """

    checkpoint: str
    seed: int
    step: int = 1
    drone_index: int = 0
    steps: int = 50
    out: str = "runs/samples/latest"
    use_ema: bool = True
    sample_seed: int = 0
    config: str | None = None
    overrides: Annotated[
        tyro.conf.UseAppendAction[list[str]],
        tyro.conf.arg(name="set", metavar="a.b.c=value"),
    ] = field(default_factory=list)


@dataclass(frozen=True)
class EvalCommand:
    """评估基线或 checkpoint，在指定划分上计算 LPIPS/PSNR/SSIM 并写 report.json。

    Args:
        config: YAML 配置路径。
        baseline: 非训练基线名称（与 checkpoint 二选一）。
        checkpoint: 待评估 checkpoint 路径（与 baseline 二选一）。
        split: 评估划分（默认 val）。
        seeds: 可选 half-open seed 范围 a:b，覆盖 split。
        out: report.json 输出目录，缺省为 runs/eval_<name>。
        use_ema: 评估 checkpoint 时优先使用 EMA 权重。
        steps: fm 采样步数（默认 50）。
        max_samples: 可选最大评估样本数。
        overrides: 覆盖配置项。
    """

    config: str | None = None
    baseline: BaselineName | None = None
    checkpoint: str | None = None
    split: SplitName = "val"
    seeds: str | None = None
    out: str | None = None
    use_ema: bool = True
    steps: int = 50
    max_samples: int | None = None
    overrides: Annotated[
        tyro.conf.UseAppendAction[list[str]],
        tyro.conf.arg(name="set", metavar="a.b.c=value"),
    ] = field(default_factory=list)


def _print_report(report: dict[str, object]) -> None:  # type: ignore[explicit-any]  # doctor 报告为异构 JSON 字典
    checks = report["checks"]
    assert isinstance(checks, list)
    for check in checks:
        assert isinstance(check, dict)
        icon = _STATUS_ICON[str(check["status"])]
        print(f"{icon} {check['name']}: {check['detail']}")
    print(f"\nG0 {'通过' if report['passed'] else '未通过'}（{report['platform']}，跳过项需另一平台或 --download）")
    print(f"报告：runs/doctor_{report['platform']}.json；门槛：runs/gates/G0_{report['platform']}.json")


def _print_eval_report(report: EvalReport, out_dir: Path) -> None:
    print(
        f"[{report.name}] split={report.split} n={report.num_samples} | "
        f"LPIPS={report.lpips.mean:.4f}±{report.lpips.std:.4f} | "
        f"PSNR={report.psnr.mean:.2f}±{report.psnr.std:.2f} dB | "
        f"SSIM={report.ssim.mean:.4f}±{report.ssim.std:.4f}"
    )
    print(f"报告：{out_dir / 'report.json'}")


def _parse_seed_range(text: str) -> list[int]:
    try:
        start_text, stop_text = text.split(":")
        start, stop = int(start_text), int(stop_text)
    except ValueError as exc:
        raise ValueError(f"seeds 应为 half-open 范围 a:b：{text!r}") from exc
    if start < 0 or stop <= start:
        raise ValueError(f"seeds 应满足 0 <= a < b：{text!r}")
    return list(range(start, stop))


def _resolve_run_dir(run: str) -> Path:
    path = Path(run)
    if len(path.parts) == 1 and not path.is_absolute():
        return Path("runs") / path
    return path


def main(argv: list[str] | None = None) -> int:
    command = tyro.extras.subcommand_cli_from_dict(
        {
            "doctor": DoctorCommand,
            "gen-data": GenDataCommand,
            "viz-episode": VizEpisodeCommand,
            "viewer": ViewerCommand,
            "cache-feats": CacheFeatsCommand,
            "train": TrainCommand,
            "sample": SampleCommand,
            "eval": EvalCommand,
        },
        prog="rvz",
        description="reality-view-zero：多视角相机 → 无人机视角生成",
        args=argv,
    )
    cfg = load_config(command.config, command.overrides)
    if isinstance(command, DoctorCommand):
        report = run_doctor(cfg, download=command.download)
        _print_report(report)
        return 0 if report["passed"] else 1
    if isinstance(command, GenDataCommand):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s：%(message)s")
        seeds = _parse_seed_range(command.seeds)
        out_dir = command.out or cfg.data.raw_dir
        generate_many(seeds, cfg.sim, out_dir, command.workers)
        print(f"已生成 {len(seeds)} 个 episode：{Path(out_dir)}")
        return 0
    if isinstance(command, ViewerCommand):
        run_viewer(command.seed, cfg.sim, drone_index=command.drone_index)
        return 0
    if isinstance(command, CacheFeatsCommand):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s：%(message)s")
        raw_dir = command.raw_dir or cfg.data.raw_dir
        out_dir = Path(command.out) if command.out else feature_cache_dir(cfg.data.feats_dir, cfg.encoder)
        seeds = _parse_seed_range(command.seeds) if command.seeds is not None else None
        encoder = FrozenEncoder.from_config(cfg.encoder)
        stats = cache_features(raw_dir, out_dir, encoder, cfg.encoder, seeds=seeds, batch_size=command.batch_size)
        print(f"特征缓存完成：{out_dir}（总计 {stats['total']}，新缓存 {stats['cached']}，跳过 {stats['skipped']}）")
        return 0
    if isinstance(command, TrainCommand):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s：%(message)s")
        run_dir = _resolve_run_dir(command.run)
        ckpt_path = run_train(
            cfg,
            command.stage,
            run_dir,
            resume=command.resume,
            overfit_seed=command.overfit_episode,
        )
        print(f"训练完成：{ckpt_path}")
        return 0
    if isinstance(command, SampleCommand):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s：%(message)s")
        outputs = sample_checkpoint(
            command.checkpoint,
            cfg,
            seed=command.seed,
            step=command.step,
            drone_index=command.drone_index,
            steps=command.steps,
            out_dir=command.out,
            use_ema=command.use_ema,
            sample_seed=command.sample_seed,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0
    if isinstance(command, EvalCommand):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s：%(message)s")
        if (command.baseline is None) == (command.checkpoint is None):
            raise ValueError("eval 必须且只能指定 --baseline 或 --checkpoint 之一")
        seeds = _parse_seed_range(command.seeds) if command.seeds is not None else None
        if command.baseline is not None:
            out_dir = Path(command.out) if command.out else Path("runs") / f"eval_{command.baseline}"
            eval_report = evaluate_copy_nearest_cam(
                cfg,
                split=command.split,
                seeds=seeds,
                out_dir=out_dir,
                max_samples=command.max_samples,
            )
        else:
            assert command.checkpoint is not None
            out_dir = Path(command.out) if command.out else Path("runs") / f"eval_{Path(command.checkpoint).stem}"
            eval_report = evaluate_checkpoint(
                command.checkpoint,
                cfg,
                split=command.split,
                seeds=seeds,
                out_dir=out_dir,
                use_ema=command.use_ema,
                steps=command.steps,
                max_samples=command.max_samples,
            )
        _print_eval_report(eval_report, out_dir)
        return 0
    assert isinstance(command, VizEpisodeCommand)
    outputs = viz_episode(command.episode, command.out, step=command.step, drone_index=command.drone_index)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
