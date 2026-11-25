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
uv run pytest -m "not slow and not gpu"
uv run ruff check . && uv run ruff format --check .
```

数据、运行产物一律不进版本库（`data/`、`runs/`）。
