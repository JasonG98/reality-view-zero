"""rvz 命令行入口（ARCH §9）；只做参数解析与调度，不含业务逻辑。"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

# MuJoCo EGL 必须在 import mujoco 之前设置（ARCH §7），故入口先设环境再导入业务模块。
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from rvz.config import load_config  # 须晚于 MUJOCO_GL 设置（import 链会拉入 mujoco）
from rvz.data.viz import viz_episode
from rvz.doctor import run_doctor  # 须晚于 MUJOCO_GL 设置
from rvz.sim.record import generate_many
from rvz.viewer import run_viewer

_STATUS_ICON = {"pass": "✅", "fail": "❌", "skip": "⚠️"}


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


def _print_report(report: dict) -> None:
    for check in report["checks"]:
        icon = _STATUS_ICON[check["status"]]
        print(f"{icon} {check['name']}: {check['detail']}")
    print(f"\nG0 {'通过' if report['passed'] else '未通过'}（{report['platform']}，跳过项需另一平台或 --download）")
    print(f"报告：runs/doctor_{report['platform']}.json；门槛：runs/gates/G0_{report['platform']}.json")


def _parse_seed_range(text: str) -> list[int]:
    try:
        start_text, stop_text = text.split(":")
        start, stop = int(start_text), int(stop_text)
    except ValueError as exc:
        raise ValueError(f"seeds 应为 half-open 范围 a:b：{text!r}") from exc
    if start < 0 or stop <= start:
        raise ValueError(f"seeds 应满足 0 <= a < b：{text!r}")
    return list(range(start, stop))


def main(argv: list[str] | None = None) -> int:
    command = tyro.extras.subcommand_cli_from_dict(
        {
            "doctor": DoctorCommand,
            "gen-data": GenDataCommand,
            "viz-episode": VizEpisodeCommand,
            "viewer": ViewerCommand,
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
    assert isinstance(command, VizEpisodeCommand)
    outputs = viz_episode(command.episode, command.out, step=command.step, drone_index=command.drone_index)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
