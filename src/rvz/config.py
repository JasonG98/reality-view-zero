"""配置系统：frozen dataclass + YAML 加载 + ``a.b.c=value`` 覆盖（ARCH §5.1、§6）。"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar, get_args, get_type_hints

import yaml

# YAML 解析结果的递归类型，避免 Any。
YamlValue = None | bool | int | float | str | list["YamlValue"] | dict[str, "YamlValue"]
SCHEMA_VERSION = 2
SceneName = Literal["small_warehouse"]


@dataclass(frozen=True)
class SimConfig:
    """仿真与渲染；角度在配置边界用度（SPEC §2.1）。"""

    robot_res: int = 128  # 小车相机分辨率（正方形）
    drone_res: int = 128  # 无人机相机分辨率（正方形）
    fovy_deg: float = 90.0  # 前后固定相机与自由相机的垂直 FOV
    robot_side_fovy_deg: float = 120.0  # 左右侧方相机广角垂直 FOV
    robot_side_down_deg: float = 15.0  # 左右侧方相机下倾角
    dt: float = 0.1  # 步长，秒（初值）
    steps: int = 40  # 每 episode 步数（初值，D1）
    n_drones: int = 3  # 每步无人机相机数 M（FR-SIM-6）
    grid_resolution: float = 0.25
    robot_radius: float = 0.35
    robot_speed: float = 0.5  # 米/秒；路径按弧长匀速采样
    drone_distance_min: float = 1.4
    drone_distance_max: float = 2.2
    drone_height_min: float = 0.3
    drone_height_max: float = 1.5
    drone_elevation_min_deg: float = -30.0
    drone_elevation_max_deg: float = -15.0
    max_resamples: int = 100
    scene: SceneName = "small_warehouse"

    def __post_init__(self) -> None:
        if self.robot_res <= 0 or self.drone_res <= 0 or self.steps <= 0 or self.n_drones <= 0:
            raise ValueError("分辨率、步数和无人机数必须为正整数")
        if not 0.0 < self.fovy_deg < 180.0 or self.dt <= 0.0:
            raise ValueError("fovy_deg 必须在 (0, 180) 内，dt 必须为正")
        if not 0.0 < self.robot_side_fovy_deg < 180.0:
            raise ValueError("robot_side_fovy_deg 必须在 (0, 180) 内")
        if not 0.0 <= self.robot_side_down_deg < 90.0:
            raise ValueError("robot_side_down_deg 必须在 [0, 90) 内")
        if self.grid_resolution <= 0.0 or self.robot_radius <= 0.0 or self.robot_speed <= 0.0:
            raise ValueError("栅格分辨率、机器人半径和速度必须为正")
        if not 0.0 < self.drone_distance_min <= self.drone_distance_max:
            raise ValueError("无人机距离范围无效")
        if not 0.0 < self.drone_height_min <= self.drone_height_max:
            raise ValueError("无人机高度范围无效")
        if not -90.0 < self.drone_elevation_min_deg <= self.drone_elevation_max_deg < 90.0:
            raise ValueError("无人机俯仰角范围无效")
        if self.max_resamples <= 0:
            raise ValueError("max_resamples 必须为正整数")
        if self.scene not in get_args(SceneName):
            raise ValueError(f"未知场景：{self.scene}")


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw"
    feats_dir: str = "data/feats"


@dataclass(frozen=True)
class EncoderConfig:
    # DINOv3-B（patch16）；HF id 放配置不写死，可用性由 doctor --download 验证。
    id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    res: int = 128  # 编码器输入分辨率（SPEC §5.1）
    pool_2x2: bool = False  # 可选 2×2 平均池化（SPEC §5.1）

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("encoder.id 不能为空")
        if self.res <= 0:
            raise ValueError("encoder.res 必须为正整数")


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 512
    depth: int = 12
    heads: int = 8
    patch: int = 8
    p_drop: float = 0.15  # 相机丢弃概率（SPEC §5.3 初值）

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.depth <= 0 or self.heads <= 0 or self.patch <= 0:
            raise ValueError("model.dim/depth/heads/patch 必须为正整数")
        if self.dim % self.heads != 0:
            raise ValueError("model.dim 必须能被 model.heads 整除")
        if not 0.0 <= self.p_drop < 1.0:
            raise ValueError("model.p_drop 必须在 [0, 1) 内")


@dataclass(frozen=True)
class TrainConfig:
    batch: int = 32
    lr: float = 1e-4  # 初值，首次实测后可修订
    ema: float = 0.999  # FR-TRN-3 初值
    seed: int = 0
    steps: int = 1000
    log_every: int = 50
    save_every: int = 500
    lpips_weight: float = 0.2  # 感知损失权重 λ（SPEC §6.1）
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.batch <= 0 or self.steps <= 0 or self.log_every <= 0 or self.save_every <= 0:
            raise ValueError("train.batch/steps/log_every/save_every 必须为正整数")
        if self.lr <= 0.0:
            raise ValueError("train.lr 必须为正数")
        if not 0.0 <= self.ema < 1.0:
            raise ValueError("train.ema 必须在 [0, 1) 内")
        if self.lpips_weight < 0.0:
            raise ValueError("train.lpips_weight 必须非负")
        if self.num_workers < 0:
            raise ValueError("train.num_workers 必须非负")


@dataclass(frozen=True)
class EvalConfig:
    steps: list[int] = field(default_factory=lambda: [1, 2, 4, 50])
    warmup: int = 10  # SPEC §7.5
    iters: int = 50


@dataclass(frozen=True)
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


T = TypeVar("T")


def _build(cls: type[T], raw: Mapping[str, YamlValue]) -> T:
    """按 dataclass 字段递归构造，未知字段直接报错。"""
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{cls.__name__} 存在未知配置字段：{', '.join(unknown)}")
    kwargs: dict[str, YamlValue] = {}
    for name, value in raw.items():
        target = hints[name]
        if dataclasses.is_dataclass(target):
            if not isinstance(value, Mapping):
                raise ValueError(f"配置段 {cls.__name__}.{name} 应为映射，实际为 {type(value).__name__}")
            kwargs[name] = _build(target, value)
        elif target is float:
            # PyYAML 按 YAML 1.1 解析，会把无小数点的科学计数法（如 3e-4）读成字符串。
            if isinstance(value, bool) or not isinstance(value, int | float | str):
                raise ValueError(f"配置项 {cls.__name__}.{name} 应为浮点数，实际为 {value!r}")
            try:
                kwargs[name] = float(value)
            except ValueError as exc:
                raise ValueError(f"配置项 {cls.__name__}.{name} 应为浮点数，实际为 {value!r}") from exc
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[arg-type]  # kwargs 已由字段名与 hints 过滤


def _apply_override(raw: dict[str, YamlValue], item: str) -> None:
    key, sep, text = item.partition("=")
    if not sep or not key:
        raise ValueError(f"覆盖格式应为 a.b.c=value：{item!r}")
    parts = key.split(".")
    node = raw
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        child = node[part]
        if not isinstance(child, dict):
            raise ValueError(f"覆盖路径 {key!r} 经过非映射字段 {part!r}")
        node = child
    node[parts[-1]] = yaml.safe_load(text)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """读 YAML 并应用覆盖；path 为 None 时返回全默认配置。"""
    raw: dict[str, YamlValue] = {}
    if path is not None:
        loaded: YamlValue = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"配置文件顶层应为映射：{path}")
        raw = loaded
    for item in overrides or []:
        _apply_override(raw, item)
    return _build(Config, raw)


def dump_config(cfg: Config) -> str:
    """把配置序列化为 YAML 快照文本（ARCH §6）。"""
    return yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False, allow_unicode=True)
