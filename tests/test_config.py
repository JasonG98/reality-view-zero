"""配置加载、覆盖与校验的契约测试。"""

import dataclasses
from pathlib import Path

import pytest

from rvz.config import Config, EncoderConfig, ModelConfig, SimConfig, TrainConfig, dump_config, load_config


def test_defaults_construct_without_file() -> None:
    cfg = load_config()
    assert cfg == Config()
    assert cfg.sim.scene == "small_warehouse"
    assert cfg.sim.n_drones == 3
    assert cfg.eval.steps == [1, 2, 4, 50]


def test_yaml_partial_sections_keep_defaults(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("sim:\n  drone_res: 64\n  fovy_deg: 75.0\nmodel:\n  dim: 256\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.sim.drone_res == 64
    assert cfg.sim.fovy_deg == 75.0
    assert cfg.model.dim == 256
    assert cfg.model.depth == 12


def test_override_nested_fields_and_value_types() -> None:
    cfg = load_config(None, ["sim.drone_res=64", "train.lr=0.001", "train.batch=8", "eval.steps=[1, 2]"])
    assert cfg.sim.drone_res == 64
    assert cfg.train.lr == pytest.approx(0.001)
    assert cfg.train.batch == 8
    assert cfg.eval.steps == [1, 2]


def test_unknown_yaml_field_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("sim:\n  no_such_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知配置字段"):
        load_config(path)


def test_unknown_override_rejected() -> None:
    with pytest.raises(ValueError, match="未知配置字段"):
        load_config(None, ["sim.nope=1"])
    with pytest.raises(ValueError, match="覆盖格式"):
        load_config(None, ["sim.drone_res"])


def test_config_is_frozen() -> None:
    cfg = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.sim.drone_res = 64  # type: ignore[misc]  # 故意验证 frozen 行为


def test_invalid_sim_ranges_rejected() -> None:
    with pytest.raises(ValueError, match="未知场景"):
        SimConfig(scene="procedural")  # type: ignore[arg-type]  # 故意验证非法场景拒绝
    with pytest.raises(ValueError, match="无人机距离范围"):
        SimConfig(drone_distance_min=4.0, drone_distance_max=2.0)
    with pytest.raises(ValueError, match="robot_side_fovy_deg"):
        SimConfig(robot_side_fovy_deg=180.0)
    with pytest.raises(ValueError, match="robot_side_down_deg"):
        SimConfig(robot_side_down_deg=-1.0)


def test_shipped_configs_load() -> None:
    root = Path(__file__).resolve().parents[1]
    mac = load_config(root / "configs" / "mac_debug.yaml")
    assert mac.sim.robot_res == 256
    assert mac.sim.drone_res == 256
    assert mac.model.dim == 256
    linux = load_config(root / "configs" / "linux_main.yaml")
    assert linux.sim.drone_res == 128
    assert linux.model.dim == 512


def test_dump_config_roundtrip_and_new_validations(tmp_path: Path) -> None:
    cfg = load_config(None, ["encoder.pool_2x2=true", "train.steps=25", "train.lpips_weight=0.5"])
    snapshot = tmp_path / "snapshot.yaml"
    snapshot.write_text(dump_config(cfg), encoding="utf-8")
    assert load_config(snapshot) == cfg

    with pytest.raises(ValueError, match=r"encoder\.res"):
        EncoderConfig(res=0)
    with pytest.raises(ValueError, match="整除"):
        ModelConfig(dim=10, heads=3)
    with pytest.raises(ValueError, match=r"train\.ema"):
        TrainConfig(ema=1.0)
