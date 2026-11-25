"""CLI 参数解析与 doctor 调度的契约测试。"""

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
