# rvz

## 项目

reality-view-zero（rvz）：在 MuJoCo 仿真里，用小车 4 路相机的帧序列和无人机相机位姿，端到端生成无人机视角画面（冻结 DINOv3 特征，像素空间 MeanFlow）。目标平台为 Mac M1 Pro 与 Linux + RTX 5060 Ti 16G。

## 文档

- `docs/SPEC.md` 管需求、数据 schema 和验收：做里程碑看 §11.2 对应行和 G 项，改 episode 看 §3.1，变更记入 §12.1。
- `docs/ARCH.md` 管模块边界、接口和已否决方案：改依赖方向或接口看 §3、§5，重提旧方案先看 §9。
- 安装、CLI 示例和 viewer 用法见 `README.md`，其余子命令见 `uv run rvz --help`。

## 红线

- 不提交 `data/`、`runs/`、checkpoint、特征缓存和 TOKEN。`HF_TOKEN` 等只走环境变量或 `huggingface-cli login`。
- 提交和推送各需当次明确授权，不改写已推送历史。提交信息格式：`M<n>: <模块> <做了什么> (FR-XXX)`，末尾加实际模型的 Co-authored-by 行。
- 验收阈值只能通过修改 SPEC 并写变更记录来调整；不静默放宽，不为让测试通过而删断言。
- 更换仿真器、编码器默认值或改变任务定义，先更新 SPEC 并问人。

## 约定

- `rvz.sim` 不 import torch，只通过 npz 交付数据；`models` 不依赖 `data`、`train`、`evaluate`，`train` 与 `evaluate` 互不依赖。完整依赖方向见 ARCH §3。
- 设备只经 `device.get_device()`，精度只经 `device.autocast_ctx()`；业务代码不写设备字面量或平台分支。CUDA 用 bf16，MPS/CPU 用 fp32，平台差异只放 YAML。
- 注意力用 `F.scaled_dot_product_attention`；不用 `torch.compile`、flash-attn、xformers。MeanFlow 的 JVP 与原生 Flash Attention 不兼容，JVP 路径走数学注意力。
- 位姿用 4×4 float32，`T_a_b` 表示把 b 系的点变换到 a 系。GL 到 OpenCV 的转换只在 `geometry.py` 一处；无人机位姿从渲染后的 GL 相机读出，不手写方位角/俯仰角公式。
- 小车只做运动学：设位姿、`mj_forward`、渲染，不调 `mj_step`。改相机或位姿代码后跑重投影测试，误差不得超过 2 像素。
- 生成器输出去噪图像 `x_hat`（x-prediction），不直接预测像素空间速度场。显式 3D 表示和 VAE/RAE 解码器已被否决；想重提，先用小实验拿证据。
- 冻结编码器只读特征缓存，manifest 不匹配必须拒绝加载。所有训练阶段共用一个训练循环，差异只在损失和启用的条件。
- 改 episode 字段必须升 `SCHEMA_VERSION`，并同步 SPEC §3.1 和 `validate_episode`。
- 最终训练和评估数据只在 Linux 生成，Mac 的 D0 只用于调试，不与最终数据混用，避免跨平台渲染差异混入数据分布。
- 配置用 frozen dataclass，随机性显式传 seed 或 Generator，`Sample` 和生成器入口 assert 张量形状。文件按需创建，不预建空模块。
- 加依赖用 `uv add`，新增运行依赖在 SPEC §12.1 记录理由。

## 平台已知坑

- Linux 必须在 import mujoco 前设 `MUJOCO_GL=egl`。EGL 依赖 NVIDIA 的 EGL/GLVND 用户态库，`nvidia-smi` 正常不代表 EGL 正常。
- RTX 5060 Ti 是 Blackwell（sm_120），需要 PyTorch ≥2.7 的 CUDA 12.8+ 构建。报 `no kernel image is available` 时先检查 torch 是否带 `+cu128`。
- Mac 的 MPS 要求 macOS 15+。缺算子时设 `PYTORCH_ENABLE_MPS_FALLBACK=1`，并在日志记录回退的算子，避免静默变慢。

## 自主范围

- 规格未写清的实现细节自行判断。主线一次只做一个里程碑，对应 G 门槛通过再开始下一个；后续里程碑的探索实验可以先行，但不合入主线。
- SPEC/ARCH 的细节不一致时，选更合理的一边并在同一改动中同步；冲突影响验收或接口时先问人。
- 遇到以下情形才停下来问：SPEC/ARCH 与实际 API 不符且影响接口或验收；验收阈值明显不合理；显存或内存不足，需要缩小配置或改变阶段分工；需要偏离 ARCH 模块边界；需要引入 Hydra、Lightning、wandb 等框架级依赖。
- 提问时给出现象、已排查内容、倾向的 1–2 个方案，以及对 SPEC/ARCH 的影响。

## 完成标准

- 做完整个里程碑再停，不做完第一版就等确认；未命中上述问人情形时，先自行排查和尝试。
- 里程碑交付物必须可通过 CLI 使用；门槛结果写入 `runs/gates/G<n>_<platform>.json`，包含指标、是否通过和 git hash。
- SPEC/ARCH 与代码保持一致，变更记入 SPEC §12.1。Mac（D0/tiny 配置）与 Linux（主线配置）都做最小验证；只涉及单一平台时在门槛文件说明。
- 默认测试使用假编码器和合成数据，不联网，可直接运行并修复本次改动引起的失败，不必逐步确认。lpips 在默认测试里 mock；真实 DINOv3 前向和单 episode 过拟合标 `slow`，MPS/CUDA 相关标 `gpu`；每个 bug 修复附回归测试。

```bash
uv run pytest -m "not slow and not gpu"
uv run ruff check . && uv run ruff format --check .
```

- G3① 单 episode 过拟合不通过，不得推进后续主线；先查位姿约定、归一化、条件注入，再怀疑模型容量。
- 只报告实际运行过的验证。未运行的平台或检查标“待运行”并给出可复制命令，不把计划说成已完成。

## 沟通

- 与人交流用中文，技术术语、库名、API、命令和报错保留英文。注释、docstring、文档以中文为主，少而有用，解释为什么。
- 不确定的内容标“未核实”，不编造。
