"""rvz 命令行入口（ARCH §9）；只做参数解析与调度，不含业务逻辑。"""

import os
import sys
from dataclasses import dataclass, field
from typing import Annotated

import tyro

# MuJoCo EGL 必须在 import mujoco 之前设置（ARCH §7），故入口先设环境再导入业务模块。
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from rvz.config import load_config  # 须晚于 MUJOCO_GL 设置（import 链会拉入 mujoco）
from rvz.doctor import run_doctor  # 须晚于 MUJOCO_GL 设置

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


def _print_report(report: dict) -> None:
    for check in report["checks"]:
        icon = _STATUS_ICON[check["status"]]
        print(f"{icon} {check['name']}: {check['detail']}")
    print(f"\nG0 {'通过' if report['passed'] else '未通过'}（{report['platform']}，跳过项需另一平台或 --download）")
    print(f"报告：runs/doctor_{report['platform']}.json；门槛：runs/gates/G0_{report['platform']}.json")


def main(argv: list[str] | None = None) -> int:
    command = tyro.extras.subcommand_cli_from_dict(
        {"doctor": DoctorCommand},
        prog="rvz",
        description="reality-view-zero：多视角相机 → 无人机视角生成",
        args=argv,
    )
    assert isinstance(command, DoctorCommand)

    cfg = load_config(command.config, command.overrides)
    report = run_doctor(cfg, download=command.download)
    _print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
