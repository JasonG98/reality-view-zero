# ARCH.md — reality-view-zero 架构

> 配套：SPEC.md（做什么）、AGENTS.md（怎么协作）。本文回答"代码怎么组织、模块之间怎么说话"。
> 约定：所有接口名是**建议签名**，实现时允许微调，但模块边界与依赖规则（§3）不得破坏。改动请同步更新本文。

---

## 1. 设计原则

1. **简洁**：一个包、少量模块、最小依赖；不预建空文件，不做过度抽象。
2. **单向依赖**：仿真 → 数据 → 模型 → 训练 → 评估，无回边。
3. **仿真与深度学习解耦**：`rvz.sim` 不导入 torch，只通过 npz 文件把数据交给下游。
4. **缓存优先**：编码器冻结时特征离线缓存；训练只读缓存。
5. **显式 schema**：episode 文件、特征缓存、checkpoint 都有版本和校验，坏数据尽早失败。
6. **一套代码，两套配置**：Mac 与 Linux 的差异只写在 YAML 里，不写在代码分支里。

---

## 2. 项目布局（uv + src-layout）

```
reality-view-zero/
├── pyproject.toml
├── uv.lock
├── .python-version              # 3.11
├── .gitignore                   # data/ runs/ .venv/ __pycache__/
├── README.md
├── AGENTS.md
├── docs/
│   ├── SPEC.md
│   └── ARCH.md
├── configs/
│   ├── mac_debug.yaml
│   └── linux_main.yaml
├── src/
│   └── rvz/
│       ├── __init__.py
│       ├── cli.py               # 子命令入口（tyro）
│       ├── config.py            # frozen dataclass 配置 + YAML 加载 + --set 覆盖
│       ├── device.py            # 设备/精度/随机种子/同步
│       ├── geometry.py          # 位姿、相机约定转换、重投影（纯 numpy）
│       ├── doctor.py            # M0 可行性检查
│       ├── sim/                 # 不得导入 torch
│       │   ├── __init__.py
│       │   ├── layout.py        # 布局采样、空闲栅格、A*
│       │   ├── scene.py         # 布局 → MJCF 字符串
│       │   ├── rig.py           # 小车相机、无人机自由相机、轨迹
│       │   └── record.py        # episode 录制与多进程生成 → npz
│       ├── data/
│       │   ├── __init__.py
│       │   ├── schema.py        # 字段/形状常量、validate_episode
│       │   ├── viz.py           # episode 接触表与位姿重投影检查图
│       │   ├── features.py      # 编码器特征缓存（含 manifest）
│       │   └── dataset.py       # torch Dataset、采样、Sample 结构
│       ├── models/
│       │   ├── __init__.py
│       │   ├── encoder.py       # 冻结 DINOv3（可选 LoRA）
│       │   ├── rays.py          # Plücker 射线嵌入（torch）
│       │   ├── generator.py     # DiT 生成器（x-prediction）
│       │   └── sampler.py       # flow matching / MeanFlow 采样
│       ├── train/
│       │   ├── __init__.py
│       │   ├── losses.py        # 各阶段损失
│       │   └── loop.py          # 训练循环、checkpoint、EMA、日志
│       └── evaluate/            # 注意不叫 eval（避免遮蔽内建）
│           ├── __init__.py
│           ├── metrics.py       # LPIPS/PSNR/SSIM、基线、分桶、延迟
│           └── report.py        # report.json 与对比图
├── tests/
│   ├── test_geometry.py
│   ├── test_layout.py
│   ├── test_schema.py
│   ├── test_record.py
│   ├── test_viz.py
│   ├── test_rays.py
│   ├── test_generator.py
│   └── test_pipeline_tiny.py    # 端到端小集成
├── data/                        # gitignore：raw/ feats/
└── runs/                        # gitignore：日志、checkpoint、报告
```

- 文件**按里程碑需要时再创建**（对应关系见 SPEC §11.2）。
- 不设 `scripts/`、`notebooks/`；一切入口走 `rvz <子命令>`。

### 2.1 `pyproject.toml` 关键片段

先用 `uv init --package rvz` 生成骨架（src-layout、构建后端以 uv 实际生成为准），再按下面补充：

```toml
[project]
name = "reality-view-zero"
version = "0.1.0"
description = "Multi-view robot cameras -> drone-view generation (pixel-space MeanFlow) in MuJoCo"
readme = "README.md"
requires-python = ">=3.11,<3.16"
dependencies = [
  "numpy",
  "mujoco",
  "torch>=2.7",          # Blackwell(sm_120) 需要 PyTorch>=2.7 且 CUDA>=12.8 构建
  "torchvision",
  "transformers",
  "peft",
  "lpips",
  "einops",
  "pillow",
  "pyyaml",
  "tqdm",
  "tyro",
]

[project.scripts]
rvz = "rvz.cli:main"

[dependency-groups]
dev = ["pytest", "ruff"]

[build-system]
requires = ["uv_build>=0.13.10,<0.14"]
build-backend = "uv_build"

[tool.uv.build-backend]
# 发布名与导入包名不同，显式指定 src/rvz。
module-name = "rvz"

# 默认 PyPI 与 PyTorch 均走南京大学镜像；macOS 的 torch 用默认索引（带 MPS 的 arm64 轮子）
[[tool.uv.index]]
name = "nju-pypi"
url = "https://mirrors.nju.edu.cn/pypi/web/simple"
default = true

# Linux 走 CUDA 12.8 的 PyTorch 构建（Blackwell sm_120 需要）
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://mirrors.nju.edu.cn/pytorch/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]

# 让 uv.lock 同时覆盖两个目标平台
[tool.uv]
required-environments = [
  "sys_platform == 'darwin' and platform_machine == 'arm64'",
  "sys_platform == 'linux' and platform_machine == 'x86_64'",
]

[tool.ruff]
line-length = 120
target-version = "py311"
src = ["src", "tests"]
extend-exclude = ["data", "runs"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "C4", "SIM", "RUF"]
# 中文文档与输出使用全角标点，关闭 RUF001–003 的同形字歧义检查。
ignore = ["RUF001", "RUF002", "RUF003"]

[tool.ruff.format]
docstring-code-format = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config"
markers = ["slow: 耗时测试", "gpu: 需要 CUDA 或 MPS"]
```

- `uv_build` 版本范围采用 uv 官方构建后端文档建议；发布名 `reality-view-zero` 与导入包名 `rvz` 不同，因此必须保留 `module-name = "rvz"`。
- 这套"平台标记 + explicit 索引"的写法是 uv 官方 PyTorch 集成指南的常见做法；`required-environments` 是我在实际项目里见到的写法，**以 `uv lock` 能否通过为准**（未核实）。
- 常用命令：`uv sync`（装依赖，含 dev 组）、`uv run rvz doctor`、`uv run pytest -m "not slow and not gpu"`、`uv run ruff check . && uv run ruff format --check .`。
- 数据与 checkpoint 不进版本库；`uv.lock` 要提交。

---

## 3. 分层与依赖规则

```
            ┌────────────── cli.py ──────────────┐
            ▼              ▼            ▼        ▼
         doctor      evaluate ◄──── train ──► models
                         │             │         │
                         └──────► data ◄─────────┘
                                   ▲
                                   │ (只读 npz / npy 文件)
                                  sim
        叶子模块（任何人都可导入）：config  device  geometry
```

**硬规则**
1. `sim` 只依赖 `config`、`geometry`、`numpy`、`mujoco`；**禁止导入 `torch`、`data`、`models`**。
2. `data` 依赖 `config`、`geometry`、`schema`；只在 `features.py` 与 `dataset.py` 中使用 torch；通过**文件**接收 `sim` 的输出，不直接调用 `sim`。
3. `models` 不依赖 `data`、`train`、`evaluate`；它只认张量。
4. `train` 与 `evaluate` 可依赖 `models`、`data`；二者互不依赖，共享的指标函数放 `evaluate.metrics`，`train` 需要的 LPIPS 通过 `train.losses` 自行引用 `lpips` 包。
5. `cli` 是唯一允许依赖所有模块的地方，且只做参数解析与调度，不含业务逻辑。
6. 叶子模块 `config`、`device`、`geometry` 不得反向依赖其他业务模块。

这样 `sim` 可以单独在**另一个只装了 numpy + mujoco 的环境**里运行（例如以后接 RoboCasa 适配器），产物通过 npz 交换。

---

## 4. 数据流与产物

```
 seed ──► sim.layout ──► sim.scene(MJCF) ──► sim.rig ──► sim.record
                                                            │
                                    data/raw/ep_XXXXXX.npz  ▼   (schema v2)
                                                 data.schema.validate_episode
                                                            │
                          models.encoder (冻结) ─► data/feats/{enc}_{res}/ep_XXXXXX.npy + manifest.json
                                                            │
                         data.dataset ──► Sample ──► models.generator ◄─ models.rays
                                                            │
                                      train.loop ──► runs/<name>/{config.yaml, log.jsonl, ckpt/, samples/}
                                                            │
                           evaluate ◄──────────────────────┘──► runs/<name>/report.json + 图
```

| 产物 | 路径 | 来源 | 版本控制 |
|---|---|---|---|
| episode | `data/raw/ep_{seed:06d}.npz` | `gen-data` | 否 |
| 特征缓存 | `data/feats/{encoder_id}_{res}/` | `cache-feats` | 否 |
| 运行目录 | `runs/<name>/` | `train`/`eval`/`bench` | 否 |
| 配置快照 | `runs/<name>/config.yaml` | 每次运行自动写 | 否 |

---

## 5. 模块职责与关键接口

### 5.1 叶子模块

**`config.py`**
- 用 `@dataclass(frozen=True)` 定义 `SimConfig`、`DataConfig`、`EncoderConfig`、`ModelConfig`、`TrainConfig`、`EvalConfig`，顶层 `Config` 聚合。
- `load_config(path, overrides: list[str]) -> Config`：读 YAML，应用 `a.b.c=value` 覆盖；未知字段直接报错。
- 不引入 Hydra/OmegaConf。

**`device.py`**
```python
def get_device() -> torch.device            # cuda > mps > cpu
def autocast_ctx(device) -> ContextManager  # cuda: bf16；其他: 空上下文（fp32）
def seed_everything(seed: int) -> None
def sync(device) -> None                    # 计时前后同步
```
- 其他模块**不得**出现 `.cuda()`、`"cuda"` 字面量。

**`geometry.py`**（纯 numpy）
```python
GL_TO_CV = np.diag([1, -1, -1, 1]).astype(np.float32)      # 唯一的约定转换点
CAM_ORDER = ("front", "left", "right", "back")            # sim/data 共享，避免反向依赖
SIDE_CAMERAS = ("left", "right")                         # 独立广角与下倾参数
MARKER_LOCAL = (0.20, 0.0, 0.18)                          # 车体朝向标记的 Base 系位置
def gl_cam_to_cv_pose(pos, forward, up) -> np.ndarray       # -> T_W_Ccv (4,4)
def invert(T) -> np.ndarray
def relative(T_W_a, T_W_b) -> np.ndarray                    # T_a_b
def intrinsics_from_fovy(fovy_deg, width, height) -> np.ndarray
def project(K, T_C_W, pts_W) -> tuple[np.ndarray, np.ndarray]  # (uv, depth)
def scale_K(K, sx, sy) -> np.ndarray
```

**`doctor.py`**（M0）
- `run_doctor(cfg: Config, download: bool = False) -> dict`：逐项检查 SPEC §12.2，写 `runs/doctor_<platform>.json` 与 `runs/gates/G0_<platform>.json`。DINOv3 默认用本地随机初始化小模型离线验证（真实权重推迟到 M2）；lpips 在 VGG 权重未缓存时需 `--download` 启用下载验证。
- 相机检查在 **GL 约定内闭环**：投影公式与 MuJoCo 官方 tutorial 的 `compute_camera_matrix` 一致，不引入 GL→CV 转换，因此 M0 不需要提前创建 `geometry.py`。
- 跨平台像素对比：各平台写 `runs/doctor_assets/ref_<platform>.npz`（固定场景固定位姿），发现另一平台参考文件时自动报告差异量级。

### 5.2 `sim`（无 torch）

```python
# layout.py
@dataclass
class Layout: ...                                   # 货架/箱子/托盘/墙/灯 + 空闲栅格
def sample_layout(seed: int, cfg: SimConfig) -> Layout
def plan_path(layout: Layout, rng) -> np.ndarray    # 栅格上的 A* 路径，(K,2) 米

# scene.py
def layout_to_mjcf(layout: Layout, cfg: SimConfig) -> str   # 含 4 个小车固定相机

# rig.py
def robot_poses(path, n_steps, dt, speed) -> np.ndarray       # (T,4,4) T_W_B，按弧长匀速采样
def drone_trajectories(rng, layout, robot_poses, cfg) -> list[DroneTraj]
def read_gl_camera(renderer) -> tuple[pos, forward, up]      # 读取单目实际使用的双目平均相机

# record.py
def generate_episode(seed: int, cfg: SimConfig) -> dict[str, np.ndarray | str]
def generate_many(seeds, cfg, out_dir, workers: int) -> None   # 多进程，逐 episode 落盘
```
要点：
- 小车每步：设位姿 → `mj_forward` → 渲染 4 路 → 对每台无人机设自由相机 → `update_scene` → 渲染 → 读出 GL 相机位姿 → `gl_cam_to_cv_pose`。
- 小车前后相机使用 `fovy_deg`；左右相机使用 `robot_side_fovy_deg=120`、`robot_side_down_deg=15`，MJCF 中直接设置相机 XY 轴实现无滚转下倾，4 路内参分别写入 `K_robot`。
- 无人机参数轨迹由 `rig` 生成为严格正后方的近低机位第三人称视角；`record` 先用射线计算机位前方的安全距离，通过双向限幅平滑缩短弹簧臂，再从单目实际使用的双目平均 GL 相机读取位姿；遮挡物通过 `mj_ray` 找出，仅在当前无人机画面用 geom group 隐藏，随后立即恢复模型分组；几何校验不通过时整段重采样。
- 无人机相机位置检查（不在几何体内、小车在视野内）使用 `Layout` 里的几何体 AABB，不依赖物理引擎碰撞。
- 重采样有最大次数，超出抛异常；实际次数与发光标记重投影误差记录在 `meta` 中。

### 5.3 `data`

```python
# schema.py（SCHEMA_VERSION 由 config.py 定义并在此重导出，避免 sim 反向依赖 data）
SCHEMA_VERSION = 2
CAM_ORDER = ("front", "left", "right", "back")  # 从 geometry.py 重导出
def validate_episode(ep: Mapping[str, np.ndarray | str]) -> None  # 形状、dtype、旋转正交性

# viz.py
def viz_episode(episode_path, out_dir, step=0, drone_index=0) -> dict[str, Path]

# features.py
def cache_features(raw_dir, out_dir, encoder, cfg) -> None   # 幂等；写 manifest.json
def load_manifest(dir) -> Manifest                           # 不匹配则 raise

# dataset.py
@dataclass
class Sample:
    ctx_feats: Tensor | None      # (2, 4, N, C)  步 {t-1, t}，缓存模式
    ctx_rgb: Tensor | None        # (2, 4, 3, H, W)  在线编码（LoRA）模式
    ctx_T_tgt: Tensor             # (2, 4, 4, 4)  各上下文相机 → 目标时刻 Base 系 的变换
    K_ctx: Tensor                 # (3, 3)
    tgt_T: Tensor                 # (4, 4)  无人机相机 → 目标时刻 Base 系
    K_tgt: Tensor                 # (3, 3)
    tgt_rgb: Tensor              # (3, Hd, Wd)  [-1, 1]
    prev_rgb: Tensor              # (3, Hd, Wd)  同一台无人机的 t-1 图像
    meta: dict                    # seed, t, m
class EpisodeDataset(torch.utils.data.Dataset): ...
def make_dataset(cfg, split, use_cache: bool) -> EpisodeDataset
```
- 位姿换算全部在 `dataset.py` 里用 `geometry` 完成，**模型侧只拿到"已在目标 Base 系下"的变换**。

### 5.4 `models`

```python
# encoder.py
class FrozenEncoder(nn.Module):
    def encode(self, rgb: Tensor) -> Tensor        # (B,3,H,W) -> (B,N,C)，no_grad
    def enable_lora(self, r: int, alpha: int) -> None
# rays.py
def plucker_rays(K, T_B_C, h, w, patch) -> Tensor  # (N_patch, 6)，patch 中心像素
# generator.py
class Generator(nn.Module):
    def forward(self, x_t, t, r, ctx, rays_tgt, prev=None) -> Tensor   # x_hat，x-prediction
# sampler.py
def sample(gen, cond, steps: int, kind: Literal["fm", "meanflow"], generator=None) -> Tensor
```

**生成器内部张量流**（默认 128×128，patch 8，dim 512）

```
x_t (B,3,128,128) ── patchify(8) ──► (B,256,192) ─ Linear→512 ─ + pos + plucker(tgt) ─┐
prev (B,3,128,128) ─ 同上 ─► prev tokens (B,256,512)（带噪；缺省则无）                    │
ctx feats (B,2,4,N,768) ─ Linear→512 ─ + plucker(ctx) + cam/time emb ─► (B,≤512,512)    │
t / (r,t) ─ sinusoidal ─ MLP ─► c (B,512)  → AdaLN 调制                                  │
                                                                                         ▼
          [ DiT block × depth ]  self-attn(目标[+prev]) → cross-attn(ctx) → MLP   （AdaLN-Zero）
                                                                                         │
          Linear → (B,256,192) ─ unpatchify ─► x_hat (B,3,128,128)    # x-prediction
```
- 注意力统一用 `F.scaled_dot_product_attention`；MeanFlow 的 JVP 路径通过一个模块级开关切到数学注意力（见 ADR-4）。
- 训练目标由 `train/losses.py` 把 `x_hat` 与目标图像换算成各阶段损失，**生成器本身不含损失**。

### 5.5 `train`

```python
# losses.py
def loss_regress(x_hat, x1) -> Tensor
def loss_fm(gen, batch, ...) -> Tensor          # x 空间 MSE + λ·LPIPS
def loss_meanflow(gen, batch, variant: Literal["pmf", "emf"], ...) -> Tensor
def loss_aux_cross_cam(...) -> Tensor
# loop.py
def run(cfg: Config, stage: str, run_dir: Path, resume: bool, overfit_seed: int | None) -> None
```
- `loop.py` 负责：构建模型/数据、优化器、EMA、AMP（经 `device.autocast_ctx`）、日志、checkpoint、resume。
- 阶段差异只体现在"用哪个损失 + 哪些条件被启用"，循环本身不分叉。
- checkpoint 内容：`model`、`ema`、`optimizer`、`step`、`config`（YAML 字符串）、`git_hash`、`versions`。

### 5.6 `evaluate`

```python
# metrics.py
def lpips_score(a, b) -> Tensor;  def psnr(a, b) -> Tensor;  def ssim(a, b) -> Tensor
def baseline_copy_nearest_cam(sample) -> Tensor
def bucketize(meta, T_tgt) -> dict[str, str]        # 距离/高度/方位象限
def measure_latency(gen, cond, steps, device, warmup=10, iters=50) -> dict
# report.py
def build_report(results, out_dir) -> None           # report.json + 对比图
```

---

## 6. 配置系统

- 单一 YAML → `Config` dataclass；`--set a.b.c=value` 覆盖；每次运行把最终配置快照写进 `runs/<name>/config.yaml`。
- 顶层分区：`sim`、`data`、`encoder`、`model`、`train`、`eval`。
- 机器差异只放在 `configs/mac_debug.yaml` 与 `configs/linux_main.yaml`（见 SPEC §8.1）。
- 路径配置用相对仓库根的路径，避免写死机器路径。

---

## 7. 平台抽象

| 关注点 | 做法 |
|---|---|
| 设备选择 | 只经 `device.get_device()` |
| 精度 | 只经 `device.autocast_ctx()`；cuda=bf16，mps/cpu=fp32 |
| 缺算子（MPS） | 设 `PYTORCH_ENABLE_MPS_FALLBACK=1`；日志中记录回退的算子，避免悄悄变慢 |
| MuJoCo 后端 | Linux 在启动时设置 `MUJOCO_GL=egl`（`doctor`/`gen-data` 入口里设置，**必须在 `import mujoco` 之前**）；Mac 不设 |
| 注意力 | 统一 SDPA；不依赖 flash-attn/xformers；不用 `torch.compile` |
| 计时 | 计时前后调用 `device.sync()` |

---

## 8. 测试策略

| 层级 | 内容 | 标记 |
|---|---|---|
| 单元 | `geometry`：GL→CV 转换、`relative`/`invert` 往返、重投影已知点 | 默认 |
| 单元 | `layout`：随机 100 个 seed 全部有效；A* 路径不穿几何体 | 默认 |
| 单元 | `schema`：好/坏 episode 的校验结果 | 默认 |
| 集成 | `record`：tiny episode 确定性、真实 GL 重投影、可见性、禁止 `mj_step`、spawn 多进程 | 默认 |
| 集成 | `viz`：上下文/无人机/重投影图片导出与下标校验 | 默认 |
| 单元 | `rays`：torch 实现与 numpy 参考一致（容差 1e-5） | 默认 |
| 单元 | `generator`：tiny 配置（CPU）前向/反向、形状、有限性 | 默认 |
| 集成 | `test_pipeline_tiny`：生成 2 个 episode → 用**假编码器**缓存特征 → 跑 2 个训练步 → 采样 | 默认（不下载大文件） |
| 慢 | 真实 DINOv3 前向；单 episode 过拟合（G3①） | `slow` |
| 设备 | MPS/CUDA 上的前向与 `jvp` 探测 | `gpu` |

- 默认测试必须能在 Mac CPU 上跑完，且不联网下载模型。
- "假编码器"是一个返回随机/确定性 token 的 `nn.Module`，与 `FrozenEncoder` 同接口。

---

## 9. 关键设计决策（ADR-lite）

| # | 决策 | 理由 | 代价 |
|---|---|---|---|
| ADR-1 | `sim` 不依赖 torch，仅以 npz 交付数据 | 仿真可在独立轻量环境运行；多进程 worker 更轻；可接入别的仿真器 | 多一层文件 IO |
| ADR-2 | 小车只做运动学，不调用 `mj_step` | 更快、无卡死/穿模；本任务不需要物理 | 不能研究接触动力学 |
| ADR-3 | 像素空间 + x-prediction，不用 VAE/RAE | 去掉解码器与域差距；pMF 证明可行；128×128 token 数小 | 分辨率提高时计算增长快 |
| ADR-4 | MeanFlow 训练提供 `pmf` 与 `emf` 两种；JVP 路径走数学注意力 | JVP 与原生 Flash Attention 不兼容；`emf` 免 JVP，适合 Mac | 两套实现要维护 |
| ADR-5 | 特征离线缓存；LoRA 阶段改在线编码 | 冻结时训练快、省显存；微调必须有梯度 | 两条数据路径（`ctx_feats` / `ctx_rgb`） |
| ADR-6 | episode 只存世界位姿，相对位姿加载时算 | 避免重复存储与不一致；递归需要跨步换算 | 加载端有换算代码，须测试 |
| ADR-7 | 最终数据集只在 Linux 生成 | 跨平台渲染有细微像素差异，避免混合分布 | Mac 不能产出最终数据 |
| ADR-8 | 配置用 dataclass + YAML，不用 Hydra | 依赖少、可读 | 功能少，复杂覆盖需手写 |
| ADR-9 | 单一包 + `rvz <子命令>` 入口，无 scripts/ | 简洁，agent 好找入口 | 子命令多了需整理 |
| ADR-10 | 仅用世界位姿的 4×4 矩阵，不存四元数 | 消除四元数顺序/符号歧义 | 略占空间 |

---

## 10. 扩展点

- **换仿真器**：写一个适配器，输出符合 `schema v2` 的 npz 即可；下游不改。RoboCasa 需单独环境（要求特定 Python 版本和分支），只通过 npz 交换。
- **换编码器**：`EncoderConfig.id` + `encoder.py` 里的注册表；特征缓存 manifest 自动区分。
- **换生成器变体**：`Generator` 的构造参数（dim/depth/heads/patch/条件开关）；新变体放同一文件，不新增顶层模块。
- **输出多帧小片段**（M8）：在 `Sample` 增加 `tgt_rgb_seq`，生成器输出 token 维度扩展；采样器接口不变。

---

## 11. 已知限制

- 无人机能看到小车相机看不到的区域（被货架遮挡），这部分只能靠猜；通过限制无人机距离/朝向、时间上下文缓解，通过分桶评估暴露。
- 程序化仓库由基本体拼成，画面朴素、几何规整，可能引入"捷径"；靠随机化与未见布局验证缓解。
- 自由相机无滚转；本任务不需要。
- Mac 上 bf16、JVP、部分算子可能不可用（见 SPEC §8、§12.2），Mac 只承担调试角色。
