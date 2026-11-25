# AGENTS.md — 给编码 agent 的协作说明

> 本文是你（agent）开工前必读的操作手册。
> 阅读顺序：**docs/SPEC.md（做什么、验收标准）→ docs/ARCH.md（模块边界、接口）→ 本文（怎么干活）**。
> 三份文档冲突时：SPEC 管“做什么”，ARCH 管“怎么组织”，本文管“怎么协作”。发现冲突先停下来问人，不要自行裁决。

## 1. 项目一句话
**reality-view-zero**（缩写 **rvz**）：在 MuJoCo 仿真里，用小车 4 路相机的帧序列 + 无人机相机位姿，端到端生成无人机视角画面（冻结 DINOv3 特征 → 像素空间 MeanFlow）。要同时在 **Mac M1 Pro** 与 **Linux + RTX 5060 Ti 16G** 上跑。

## 2. 环境与常用命令（uv）
```bash
uv sync                                   # 安装依赖（含 dev 组）
uv run rvz doctor                         # 环境/可行性检查（M0）
uv run rvz gen-data --config configs/mac_debug.yaml --seeds 0:20 --workers 2
uv run rvz train --config configs/mac_debug.yaml --stage fm --run dbg
uv run pytest -m "not slow and not gpu"   # 默认测试（必须能在 Mac CPU 上跑完）
uv run ruff check . && uv run ruff format --check .
```
- 需要自动格式化时运行 `uv run ruff format .`。
- 加依赖：`uv add <pkg>`（开发依赖加 `--dev`）。**新增运行依赖前先问人**，并在 SPEC §12.1 记录理由。
- 不要手动 `pip install`；不要提交 `.venv/`、`data/`、`runs/`、checkpoint。

## 3. 目录地图
详见 ARCH §2。要点：
- 源码在 `src/rvz/`：`sim/`（无 torch）、`data/`、`models/`、`train/`、`evaluate/`，叶子模块 `config.py`、`device.py`、`geometry.py`。
- 开发文档在 `docs/`；入口只有 `rvz <子命令>`（`cli.py`），不要新增 `scripts/` 或 notebook。
- 文件**按里程碑需要时再创建**，不要预先建空模块。

## 4. 工作协议（每个里程碑都照此执行）
1. **读**：SPEC §11.2 中该里程碑那一行，以及它引用的 FR 编号和 ARCH 对应模块。
2. **列计划**：先用几句话写出要新增/修改的文件和接口（与 ARCH §5 对照），有偏离先问人。
3. **先写测试**：`geometry`、`schema`、`layout`、`rays` 这类纯函数必须先有测试。
4. **最小实现**：先让最小路径跑通，再补功能；规格没写清的实现细节选最简方案，不自行增加功能。
5. **跑门槛**：按 SPEC §11.1 对应的 G 项验证，把结果写到 `runs/gates/G<n>_<platform>.json`（含指标、通过与否、git hash）。
6. **同步文档**：接口、schema、阈值有任何变化，同一次提交里改 SPEC/ARCH，并在 SPEC §12.1 加一行变更记录。
7. **提交**：获得当次授权后小步提交，格式见 §12。

**一次只做一个里程碑**。上一个里程碑的门槛没过，不要开始下一个。

## 5. 代码规范
- Python ≥ 3.11、< 3.16；公开函数有完整类型标注；配置用 `@dataclass(frozen=True)`，其他结构化数据用 `dataclass`。
- 格式与 lint 用 `ruff`（行宽 120）；导入放在模块顶层；库代码用 `logging`，不用 `print`。
- 优先纯函数，不使用全局可变状态；随机性通过显式 `seed` / `np.random.Generator` / `torch.Generator` 传递。
- 张量形状写在注释或 docstring 里，并在模块边界处 `assert` 形状（尤其是 `Sample` 与生成器入口）。
- 单文件尽量 ≤ 400 行；超出先考虑按既有模块边界拆分，不为此新增抽象层。
- 注释与 docstring 使用中文，少而有用，解释“为什么”，不复述代码在做什么。

## 6. 平台规则（最容易出错的地方）
- **禁止硬编码设备**：不出现 `.cuda()`、`"cuda"`、`"mps"` 字面量；一律经 `device.get_device()`。
- **精度**：CUDA 用 bf16（`device.autocast_ctx`）；**MPS/CPU 用 fp32**（M1 不支持 bf16，未核实）。
- **不用 `torch.compile`、flash-attn、xformers**；注意力统一 `F.scaled_dot_product_attention`。
- **MeanFlow 的 JVP**：与原生 Flash Attention 不兼容，JVP 路径要走数学注意力；MPS 上 JVP 是否可用未核实，先在 `doctor` 里测，不可用就用 `emf` 变体。
- **MuJoCo 后端**：Linux 在 `import mujoco` **之前**设置 `MUJOCO_GL=egl`；Mac 不设。Linux EGL 需要 NVIDIA 的 EGL/GLVND 用户态库（`nvidia-smi` 正常不代表 EGL 正常）。
- **5060 Ti**：Blackwell（sm_120）需要 PyTorch ≥ 2.7 的 CUDA 12.8+ 构建；出现 `no kernel image is available` 说明装错了 torch 构建，检查锁定的 torch 是否带 `+cu128`。
- **Mac**：最新 PyTorch 的 MPS 要求 macOS 15+ 与 Python 3.11+；缺算子时设 `PYTORCH_ENABLE_MPS_FALLBACK=1`，并在日志里记录回退的算子。
- **跨平台数据**：同一场景在两个平台渲染可能有细微像素差异。**最终训练/评估数据只在 Linux 生成**，Mac 的 D0 只用于调试，不混用。

## 7. 几何与仿真规则（pose 相关 bug 的重灾区）
- 位姿一律 4×4 float32，命名 `T_a_b`（a←b）；不存四元数。
- **GL→OpenCV 相机约定转换只写在 `geometry.py` 一处**（`diag(1,-1,-1,1)`）；其他地方不得再翻转轴。
- 无人机位姿**从渲染后场景里的 GL 相机读出**（位置/前向/上向量），**禁止手写方位角/俯仰角公式**。
- 小车**只做运动学**：设位姿 → `mj_forward` → 渲染；**不要调用 `mj_step`**。
- `rvz.sim` **不得导入 torch**（硬规则，见 ARCH §3）。
- 改相机/位姿相关代码后，必须跑重投影测试（已知 3D 点重投影误差 ≤ 2 像素）。

## 8. 模型与训练规则
- 生成器输出是**去噪后的图像 x̂（x-prediction）**；**不得直接预测像素空间的速度场**（会严重失败）。
- 不要引入显式 3D 表示（3DGS、占据栅格），不要引入 VAE/RAE 解码器——这些都是已被否决的方案（见 SPEC §1.3、ARCH ADR-3）。
- 冻结编码器时只读特征缓存；LoRA 阶段才在线编码。缓存 manifest 不匹配必须拒绝加载。
- 所有阶段共用同一个训练循环，阶段差异只在“用哪个损失 + 启用哪些条件”。
- **G3① 单 episode 过拟合**是最重要的体检：不通过不要往后做。先查位姿约定、归一化、条件注入，再怀疑模型容量。

## 9. 数据与密钥规则
- `data/`、`runs/`、checkpoint、特征缓存**一律不提交**。
- DINOv3 在 Hugging Face 上需要先同意协议（未核实）；令牌只通过环境变量（如 `HF_TOKEN`）或 `huggingface-cli login` 提供，**不要写进代码、配置或日志**。
- `lpips` 首次使用会下载骨干权重：默认测试里要 mock 或跳过，不让默认测试联网。
- 修改 episode 字段 = 升 `SCHEMA_VERSION`，同时更新 SPEC §3.1 和 `validate_episode`。

## 10. 测试与验证
- 默认测试（`-m "not slow and not gpu"`）必须在 Mac CPU 上跑完、**不下载大文件**、不依赖真实 DINOv3。
- 集成测试用“假编码器”（与 `FrozenEncoder` 同接口，返回确定性 token）。
- 真实 DINOv3 前向、单 episode 过拟合标 `slow`；MPS/CUDA 相关标 `gpu`。
- 每个 bug 修复附一个能复现它的测试；不得为了让测试通过而放宽误差阈值、删断言或 stub 被测代码。
- 只报告实际运行过的验证。没有 GPU 时不得声称做过 GPU 验证；待训练机执行的验收要标为“待运行”，并给出可直接复制的命令。

## 11. 沟通准则
- 凡能用中文表达的内容一律使用中文，包括注释、docstring、文档、agent 回复、日志、错误信息和 CLI 帮助。
- 代码标识符、公共 API、命令、路径、配置键和机器读取的字段名保持原文；第三方原始输出保留原样，必要时补充中文说明。
- 表达清楚、简洁、有依据；不臆测、不编造，不把计划或推测说成已完成的事实。

## 12. Git 提交与推送
- 提交和推送分别需要当次明确授权，未经授权不得执行。
- 不改写已推送历史：不 rebase 已推送提交、不 amend 已推送提交、不 force push。
- 提交前核对 `user.name` / `user.email`，并确认暂存内容不包含数据、checkpoint 或密钥。
- 小步提交，信息格式：`M<n>: <模块> <做了什么> (FR-XXX)`。
- 提交信息末尾添加：`Co-Authored-By: Claude Code <noreply@anthropic.com>`。

## 13. 禁止事项
- ❌ 静默修改 SPEC 里的验收阈值（必须改文档并写变更记录）。
- ❌ 新增依赖（Hydra、Lightning、wandb 等）而不先问。
- ❌ 在代码里按平台分支写逻辑（`if platform == "mac"`）；平台差异只放配置。
- ❌ 预先创建空模块或“以后会用到”的抽象。
- ❌ 在 `sim` 里 `import torch`，或在 `models` 里依赖 `data`/`train`/`evaluate`。
- ❌ 提交数据、checkpoint、令牌。
- ❌ 换仿真器、换编码器默认值、改变任务定义，而不先更新 SPEC 并问人。

## 14. 什么时候停下来问人
- SPEC/ARCH 之间或与现实不符（例如某个 API 在实际版本里不存在）。
- M0 的“未核实”项得出了与假设相反的结论（例如 MPS 上 JVP 不可用、EGL 配不通、DINOv3 无法下载）。
- 验收阈值明显不合理，需要校准。
- 显存/内存不足，需要缩小配置或改变阶段分工。
- 需要新增依赖，或需要偏离 ARCH 的模块边界。

提问时给出：现象、你已排查的内容、你倾向的方案（1–2 个）、对 SPEC/ARCH 的影响。

## 15. 完成定义（DoD）
一个里程碑算完成，需要同时满足：
1. SPEC §11.2 列出的交付物都已存在并可通过 CLI 使用；
2. 对应 G 门槛通过，结果文件已写入 `runs/gates/`；
3. 默认测试全绿，`ruff check` 与 `ruff format --check` 通过；
4. SPEC/ARCH 与代码一致，变更已记入 SPEC §12.1；
5. 在 Mac 与 Linux 上都做过最小验证（Mac 可用 D0 与 tiny 配置，Linux 用主线配置；该里程碑只涉及单一平台的除外，需在门槛文件里说明）。
