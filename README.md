# reality-view-zero

在 MuJoCo 仿真里，用小车 4 路相机的帧序列 + 无人机相机位姿，端到端生成无人机视角画面
（冻结 DINOv3 特征 → 像素空间 MeanFlow）。目标平台：Mac M1 Pro 与 Linux + RTX 5060 Ti 16G。

## 文档

- `docs/SPEC.md`：做什么、验收标准
- `docs/ARCH.md`：模块边界、接口
- `AGENTS.md`：协作协议（里程碑、平台规则、提交规则）

## 快速开始

```bash
uv sync
uv run rvz doctor                       # M0 环境/可行性检查
uv run rvz doctor --download            # 真实 DINOv3-B 权重（推迟到 M2）与 lpips 权重下载验证（需联网）
uv run rvz gen-data --config configs/mac_debug.yaml --seeds 0:20 --workers 4
uv run rvz gen-data --config configs/mac_debug.yaml --seeds 10000:10004 --workers 4
uv run rvz viz-episode --episode data/raw/ep_000000.npz --out runs/viz/ep_000000
uv run rvz viewer --seed 0
uv run rvz cache-feats --config configs/mac_debug.yaml
uv run rvz train --config configs/mac_debug.yaml --stage regress --run regress_d0
uv run rvz eval --config configs/mac_debug.yaml --baseline copy_nearest_cam --split val --out runs/eval_copy_nearest
uv run rvz eval --config configs/mac_debug.yaml --checkpoint runs/regress_d0/ckpt/latest.pt --split val --out runs/eval_regress
uv run pytest -m "not slow and not gpu"
uv run ruff check . && uv run ruff format --check .
```

## 实时预览与目检样例

主线训练、验证与实时预览统一使用 AWS RoboMaker 的无屋顶小型仓库（`sim.scene=small_warehouse`，`train` 0–9999，`val` 10000–10099），使用原场景的货架、纸箱、托盘车和地面材质。固定目检样例选用 `val` 中的 `seed=10000`（通道内平滑 90° 转弯）与 `seed=10008`（贴近货架连续平滑转弯并触发弹簧臂缩短）：

```bash
uv run rvz viewer --seed 10000
uv run rvz viewer --seed 10008
```

`rvz viewer` 默认配置 `configs/viewer.yaml` 使用高分辨率预览：左侧四宫格按前视、左视、右视、后视排列，每路原生渲染 512×512；右侧是正后方跟随的无人机视角，原生渲染 768×768。五路画面按 `sim.dt` 同步刷新，走完当前路线后从起点继续。关闭窗口或按 `Esc` 退出。[场景来源和授权](src/rvz/sim/assets/README.md)

在图形桌面会话中运行，Python 需支持 Tk。用 `--drone-index 1` 切换到第二个无人机机位。默认加载 `configs/viewer.yaml`，也可用 `--config` 选用其他配置，或用 `--set` 覆盖单项。

用同一份配置录制高清仓库画面。

```bash
uv run rvz gen-data --config configs/viewer.yaml --seeds 0:1 --out data/warehouse_preview
```

数据、运行产物一律不进版本库（`data/`、`runs/`）。
