"""CLI 参数解析与子命令调度的契约测试。"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from rvz import cli
from rvz.config import Config


@pytest.mark.parametrize(("passed", "expected_exit_code"), [(True, 0), (False, 1)])
def test_doctor_arguments_and_exit_code(monkeypatch: pytest.MonkeyPatch, passed: bool, expected_exit_code: int) -> None:
    config = Config()
    load_config = Mock(return_value=config)
    run_doctor = Mock(return_value={"checks": [], "passed": passed, "platform": "test"})
    monkeypatch.setattr(cli, "load_config", load_config)
    monkeypatch.setattr(cli, "run_doctor", run_doctor)

    exit_code = cli.main(
        [
            "doctor",
            "--config",
            "configs/mac_debug.yaml",
            "--set",
            "sim.drone_res=64",
            "--set",
            "train.batch=8",
            "--download",
        ]
    )

    assert exit_code == expected_exit_code
    load_config.assert_called_once_with("configs/mac_debug.yaml", ["sim.drone_res=64", "train.batch=8"])
    run_doctor.assert_called_once_with(config, download=True)


def test_gen_data_arguments_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config()
    load_config = Mock(return_value=config)
    generate_many = Mock()
    monkeypatch.setattr(cli, "load_config", load_config)
    monkeypatch.setattr(cli, "generate_many", generate_many)

    exit_code = cli.main(
        [
            "gen-data",
            "--seeds",
            "2:5",
            "--workers",
            "2",
            "--out",
            "data/test",
            "--set",
            "sim.steps=2",
        ]
    )

    assert exit_code == 0
    load_config.assert_called_once_with(None, ["sim.steps=2"])
    generate_many.assert_called_once_with([2, 3, 4], config.sim, "data/test", 2)


@pytest.mark.parametrize("seeds", ["2", "2:", "3:3", "-1:2", "a:b"])
def test_gen_data_rejects_invalid_seed_range(monkeypatch: pytest.MonkeyPatch, seeds: str) -> None:
    monkeypatch.setattr(cli, "load_config", Mock(return_value=Config()))
    generate_many = Mock()
    monkeypatch.setattr(cli, "generate_many", generate_many)
    with pytest.raises(ValueError, match="seeds"):
        cli.main(["gen-data", "--seeds", seeds])
    generate_many.assert_not_called()


def test_viz_episode_arguments_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config()
    load_config = Mock(return_value=config)
    viz_episode = Mock(return_value={"overview": "runs/viz/overview.png"})
    monkeypatch.setattr(cli, "load_config", load_config)
    monkeypatch.setattr(cli, "viz_episode", viz_episode)

    exit_code = cli.main(
        [
            "viz-episode",
            "--episode",
            "data/raw/ep_000002.npz",
            "--out",
            "runs/viz/test",
            "--step",
            "2",
            "--drone-index",
            "1",
        ]
    )

    assert exit_code == 0
    load_config.assert_called_once_with(None, [])
    viz_episode.assert_called_once_with("data/raw/ep_000002.npz", "runs/viz/test", step=2, drone_index=1)


def test_viewer_arguments_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config()
    load_config = Mock(return_value=config)
    run_viewer = Mock()
    monkeypatch.setattr(cli, "load_config", load_config)
    monkeypatch.setattr(cli, "run_viewer", run_viewer)

    exit_code = cli.main(
        ["viewer", "--config", "configs/mac_debug.yaml", "--seed", "6", "--drone-index", "1", "--set", "sim.dt=0.05"]
    )

    assert exit_code == 0
    load_config.assert_called_once_with("configs/mac_debug.yaml", ["sim.dt=0.05"])
    run_viewer.assert_called_once_with(6, config.sim, drone_index=1)


@pytest.mark.parametrize(("overrides", "drone_res"), [([], 768), (["--set", "sim.drone_res=640"], 640)])
def test_viewer_uses_high_resolution_warehouse_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overrides: list[str], drone_res: int
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "viewer.yaml"
    load_config = Mock(wraps=cli.load_config)
    run_viewer = Mock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_config", load_config)
    monkeypatch.setattr(cli, "run_viewer", run_viewer)

    assert cli.main(["viewer", *overrides]) == 0

    load_config.assert_called_once_with(str(config_path), overrides[1:])
    config = run_viewer.call_args.args[1]
    assert config.scene == "small_warehouse"
    assert (config.robot_res, config.drone_res) == (512, drone_res)
