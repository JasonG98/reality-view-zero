# SPEC.md — reality-view-zero 项目规格

> 状态：v1（初稿）　项目名：`reality-view-zero`　包名：`rvz`　Python ≥ 3.11、< 3.16　布局：uv + src-layout
> 阅读顺序：**SPEC.md（做什么）→ ARCH.md（怎么组织）→ AGENTS.md（怎么协作）**
> 标注约定：「（初值）」= 起始取值，首次实测后允许修订并记入 §12；「（未核实）」= 凭已有知识写的，必须在 M0 验证。

---

## 1. 目的与范围

### 1.1 一句话
在 MuJoCo 仿真里，用**小车 4 路相机的帧序列 + 无人机相机位姿**，端到端（像素进、像素出）生成**无人机视角画面**；特征提取器为冻结的预训练模型，生成器为像素空间的 MeanFlow。

这是对 Tesla Reality View 公开描述的**低成本复刻验证**：冻结编码器特征 → 小型扩散/流模型 → 上一步输出加噪后去噪 → 少步生成。Tesla 的训练目标与数据来源未公开，本项目的对应关系是推断。

### 1.2 目标
1. 在仿真中证明"多视角特征 + 位姿条件 → 新视角像素"这条端到端管线能跑通、能收敛。
2. 验证时间递归（上一步输出加噪作条件）能抑制漂移。
3. 验证像素空间 MeanFlow 能把采样压到 1–4 步，并在两台机器上测出延迟。
4. 验证冻结编码器上叠加 LoRA 的自监督/端到端微调是否带来稳定提升。

### 1.3 非目标
- 不做显式 3D 表示（3DGS、占据栅格、点云）。
- 不做真实数据（nuScenes 等）与真机部署。
- 不做物理仿真（小车只做运动学）。
- 不追求画质上限；追求**管线正确、指标可比较、双平台可复现**。
- 不实现 Tesla 的 4 周期分摊、层间交错更新（记为后续工作）。

---

## 2. 约定（所有模块必须遵守）

### 2.1 单位与类型
- 长度：米；时间：秒；内部角度：弧度（仅在配置文件边界使用度）。
- 位姿一律用 **4×4 float32 矩阵**，不存四元数（避免顺序歧义）。
- 命名：`T_a_b` 表示把 b 系下的点变换到 a 系（a←b），即 `p_a = T_a_b @ p_b`。

### 2.2 坐标系
| 名称 | 定义 |
|---|---|
| World `W` | MuJoCo 世界系，z 向上 |
| Base `B` | 小车系：x 前、y 左、z 上；原点为车体中心 |
| Camera `C` | **OpenCV 约定**：x 右、y 下、z 前（光轴） |

- **MuJoCo 的 GL 相机**前向为 −Z、上为 +Y。转 OpenCV：`T_W_Ccv = T_W_Cgl @ diag(1, −1, −1, 1)`。这个转换只允许出现在 `geometry.py` 中一处。
- 小车 4 相机固定顺序：`front, left, right, back`；相对 B 的偏航角分别为 `0°, +90°, −90°, 180°`（绕 z，逆时针为正）；左右相机额外下倾 15°，安装高度约 0.3 m。
- 每个相机使用独立内参 `K`（3×3，像素单位）；4 路小车相机按固定顺序存为 `[4,3,3]`。图像缩放后 `K` 必须同步缩放。

### 2.3 时间与采样
- 一个 episode 有 `T` 步，步长 `Δt`；步下标 `t = 0..T−1`。
- 训练样本为 `(episode, t, m)`：`t ≥ 1`（需要 `t−1` 作上下文和递归条件），`m` 为无人机相机下标。
- 上下文步集合固定为 `{t, t−1}`。

### 2.4 图像与张量
- 存储：uint8，HWC。
- 生成器输入输出：float32，CHW，范围 `[−1, 1]`。
- 编码器输入：按所选预训练模型的归一化（均值/方差）处理；写入特征缓存 manifest。

### 2.5 射线嵌入（Plücker）
对目标相机（无人机）和每个上下文相机的每个 patch 中心像素 `(u, v)`：
```
d_C = normalize(K^-1 [u, v, 1]^T)
d_B = R_B_C d_C ，  o_B = t_B_C
plucker = [d_B, o_B × d_B]   # 6 维
```
- 所有射线统一表示在**目标时刻 `t` 的 Base 系**里；上下文步 `t−1` 的相机位姿通过世界位姿换算到 `t` 的 Base 系，以补偿小车运动。
- 必须有 numpy 参考实现，torch 实现与其数值一致（测试覆盖）。

---

## 3. 数据规格

### 3.1 Episode 文件
路径：`data/raw/ep_{seed:06d}.npz`。字段（`T` 步，`H/W` 为小车图像分辨率，`Hd/Wd` 为无人机图像分辨率，`M` 为无人机数）：

| 字段 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `robot_rgb` | `[T, 4, H, W, 3]` | uint8 | 小车 4 路图像，顺序 front/left/right/back |
| `T_W_B` | `[T, 4, 4]` | float32 | 每步小车位姿（World←Base） |
| `T_B_Crobot` | `[4, 4, 4]` | float32 | 4 路小车相机相对 Base 的固定外参（OpenCV 约定） |
| `K_robot` | `[4, 3, 3]` | float32 | 4 路小车相机独立内参，顺序与 `robot_rgb` 一致；schema v2 |
| `drone_rgb` | `[T, M, Hd, Wd, 3]` | uint8 | 无人机图像 |
| `T_W_Cdrone` | `[T, M, 4, 4]` | float32 | 无人机相机世界位姿（OpenCV 约定），**由渲染后的场景相机读出** |
| `K_drone` | `[3, 3]` | float32 | 无人机内参 |
| `meta_json` | 标量字符串 | str | seed、布局参数、`dt`、`fovy`、MuJoCo 版本、平台、git hash、schema 版本，以及轻量诊断统计 `diagnostics`（`max_step_yaw_deg`、`total_yaw_change_deg`、弹簧臂缩短步数/最小跟随距离、遮挡隐藏步数/总隐藏几何体数） |

- 只存世界位姿；Base 系下的相对位姿在加载时计算。
- `schema_version` 写在 `meta_json` 里；**改字段必须升版本号**。v2 将 `K_robot` 从 `[3,3]` 改为 `[4,3,3]`，v1 文件不得按 v2 加载。
- 校验器 `data.schema.validate_episode` 必须检查形状、dtype、旋转矩阵正交性（容差 1e-4）、位姿最后一行为 `[0,0,0,1]`。

### 3.2 划分
主线训练、验证与预览统一使用 AWS RoboMaker 无屋顶小型仓库（`sim.scene=small_warehouse`）。该仓库为单一固定布局，按 `seed` 确定性控制小车平滑轨迹与无人机跟拍机位；`train` / `val` / `test` 划分用于检验模型在**未见轨迹与未见机位**上的新视角生成泛化能力。

| 划分 | seed 范围（初值） |
|---|---|
| train | 0–9999 |
| val | 10000–10099 |
| test | 20000–20099 |

固定目检样例从 `val` 划分选取：
- `seed=10000`：通道内平滑 90° 转弯（40 步累计转向约 90.0°，最大单步偏航约 8.94°，全程标称跟随距离，未触发弹簧臂缩短）。
- `seed=10008`：贴近货架连续平滑转弯（40 步累计转向约 180.0°，最大单步偏航约 11.62°，触发弹簧臂平滑缩短至约 1.37 m）。

### 3.3 数据集规模（初值，估计）
| 名称 | 规模 | 用途 |
|---|---|---|
| D0 | train 20 episode × 30 步；val 4 episode | Mac 调试 |
| D1 | train 300 episode × 40 步；val 30 episode | 主线首轮（Linux 生成） |
| D2 | train 2,000 episode × 50 步；val 100 episode | 扩大版 |

### 3.4 特征缓存
路径：`data/feats/{encoder_id}_{res}/ep_{seed:06d}.npy`，fp16，形状 `[T, 4, N, C]`（`N=(res/16)²`，`C=768`）。
同目录 `manifest.json` 记录：编码器 id 与 revision、输入分辨率、归一化参数、`rvz` 版本。manifest 不匹配时必须拒绝加载。

---

## 4. 仿真规格（`rvz.sim`）

| 编号 | 需求 |
|---|---|
| FR-SIM-1 | 全仓库统一使用 `sim.scene=small_warehouse`（AWS RoboMaker 无屋顶小型仓库，约 14.0×20.9 m，**无天花板**），保留原模型摆放、货架、纸箱与托盘车几何，并按导入包围盒确定性构建导航布局（原简化程序化盒体仓库已废弃移除） |
| FR-SIM-2 | 使用小型仓库资产自带的墙面、地面、货架与纸箱贴图和光照配置；由 `seed` 确定性控制小车轨迹与无人机机位 |
| FR-SIM-3 | 布局同时产出 2D 空闲栅格（为正后方跟拍留出规划余量）；小车路径采用 8 邻接 A* + 视线剪枝 + 防 180° 折返 + 二次 Bézier 圆角平滑 + 起点后向间隙检查，无碰撞，定步弦长匀速 |
| FR-SIM-4 | 小车为**运动学刚体**，不调用 `mj_step`：每步直接设位姿，调用 `mj_forward` 后渲染。车体有倒角车壳、轮组、相机外壳和前后灯，车头灯标用于辨认朝向和校验重投影 |
| FR-SIM-5 | 小车 4 路固定相机（挂车体，安装高度约 0.3 m）：前后 FOV 90°；左右为 120° 垂直 FOV 广角并下倾 15°，覆盖近车侧地面。无人机为 **MuJoCo 自由相机**（看向点、距离、方位角、俯仰角），每步每台更新 |
| FR-SIM-6 | 每步 `M=3` 台无人机相机，方位角严格等于小车当前 `yaw(t)`，相机位于小车**正后方**，不使用随机方位偏置、环绕角速度或侧向避障；各台标称距离 1.4–2.2 m、离地 0.3–1.5 m、俯仰角 −30° 至 −15°（均为初值），看向小车中心。正后方有墙体/货架时保持方向不变，平滑缩短跟随距离（最短 0.45 m，相邻步变化 ≤0.12 m），不得穿模 |
| FR-SIM-7 | 无人机相机位置**不得在任何几何体内**，且小车必须在其视野内（含朝向标记）；违反则重采样。仅对当前无人机画面，用射线找出相机到小车之间的环境几何体并自动隐藏；小车四路相机、其他无人机和碰撞布局保持不变 |
| FR-SIM-8 | 前后固定相机与自由相机使用 `fovy_deg`；左右相机使用独立 `robot_side_fovy_deg` 与 `robot_side_down_deg`，`K_robot` 按 4 路分别计算；确保离屏缓冲尺寸不小于渲染尺寸 |
| FR-SIM-9 | 位姿从渲染后场景里的 GL 相机（位置、前向、上向量）读出，再转 OpenCV 约定；单目模式读取左右 GL 相机的平均值，与 MuJoCo 实际单目渲染一致，避免默认 IPD 的半眼距横移。**禁止手写方位角/俯仰角公式**。注意 MuJoCo `mjvCamera.elevation` 的符号约定：**负值 = 相机在上方俯视**（M0 实测确认） |
| FR-SIM-10 | 多进程并行生成，每进程独立渲染器；Linux 使用 `MUJOCO_GL=egl`，Mac 用默认后端 |
| FR-SIM-11 | `rvz.sim` **不得导入 torch** |
| FR-SIM-12 | 已合并至 FR-SIM-1/FR-SIM-2：`small_warehouse` 为唯一主线与预览场景，原简化程序化场景已移除 |

录制和预览按同一顺序选择无人机机位。整段轨迹须逐帧通过几何检查，且渲染画面中能检出车头灯标质心，重投影误差为有限值。

---

## 5. 模型规格

### 5.1 特征提取器（FR-MOD-1）
- 默认：冻结 **DINOv3-B（patch 16）**；HF 模型 id 放配置，不写死（未核实：需在 Hugging Face 同意协议）。
- 输入分辨率可配：默认 128（8×8=64 token/相机），后期可试 224（14×14=196）。
- 输出 patch token `[B, N, 768]`；可选 2×2 平均池化。
- 冻结模式：`no_grad`，使用特征缓存。LoRA 模式：在线编码（不使用缓存），仅 LoRA 参数可训。

### 5.2 生成器（FR-MOD-2）
- DiT 风格 Transformer，**像素空间，不用潜空间/解码器**；目标分辨率 `Hd×Wd`（默认 128；Mac D0 为 256），patch 8。
- 输入：带噪图像 patch token、时间嵌入（flow matching 用 `t`，MeanFlow 用 `(r, t)`）、上下文 token、射线嵌入、递归条件 token。
- 上下文 token：4 相机 × 步 `{t, t−1}`，每个 token = 投影后的特征 + 该 patch 的 Plücker 嵌入 + 相机/时间嵌入。
- 注入方式：目标 token 对上下文 token 做 cross-attention；时间步用 AdaLN。
- **输出为去噪后的图像 x̂（x-prediction）。不得直接预测像素空间的速度场**（pMF 论文指出会严重失败）。
- 参数量：Linux 主线约 30–40M，Mac 调试约 3–5M（均为估计）。

### 5.3 条件（FR-MOD-3）
- **射线嵌入**：见 §2.5，目标（无人机）与每个上下文 token 都带。
- **时间递归**：上一步无人机输出（同一台无人机相机的 `t−1` 图像）加噪后作为额外条件 token。训练时用真值加噪（噪声强度随机采样），推理时用自己上一步的输出；首步不带该条件。
- **相机丢弃**：训练时以概率 `p_drop`（初值 0.15）随机丢弃部分小车相机的上下文 token。

### 5.4 采样（FR-MOD-4）
- flow matching：Euler，步数可配（默认 50）。
- MeanFlow：少步（1/2/4），按平均速度场迭代。
- 采样器与具体训练目标解耦，对外只暴露 `sample(generator, cond, steps)`。

---

## 6. 训练规格

### 6.1 阶段
| 阶段 `--stage` | 内容 | 损失（默认） |
|---|---|---|
| `regress` | 无生成模型，直接回归无人机图像；作为下限基线 | MSE + LPIPS |
| `fm` | flow matching + x-prediction，多步采样 | x 空间 MSE + λ·LPIPS |
| `fm_rec` | 在 `fm` 上加入时间递归条件与噪声增强 | 同上 |
| `meanflow` | pMF（或 Euler Mean Flows）训练，1–4 步生成 | 见 §6.2 |
| `lora` | 冻结编码器 → LoRA 微调（编码器 + 生成器联合） | 同 `meanflow` 或 `fm_rec`，加可选辅助损失 |

### 6.2 MeanFlow 变体（`meanflow_variant`）
| 值 | 说明 |
|---|---|
| `pmf` | pixel MeanFlow（arXiv:2601.22158）：x-prediction + MeanFlow 损失 + 感知损失；需要 JVP |
| `emf` | Euler Mean Flows（arXiv:2602.02571）：不需要 JVP，含像素空间 x₁-prediction 变体 |
- JVP 与 PyTorch 原生 Flash Attention 不兼容；JVP 路径需走数学注意力。
- Mac 上 JVP 是否可用**未核实**（M0 检查）；不可用则 Mac 只跑 `fm`/`fm_rec`，或改用 `emf`。

### 6.3 LoRA 与自监督辅助损失（`lora` 阶段）
- 仿真中无人机画面是免费真值，整套训练无需人工标注。
- LoRA 微调 DINOv3（`peft`），学习率低于生成器；保留冻结版作对照。
- 可选辅助损失 `aux_cross_cam`：丢掉一路小车相机，用其余相机预测该相机的（冻结教师）特征。
- 若验证集无稳定提升，**保留冻结版为最终结果**。

### 6.4 通用要求（FR-TRN）
- FR-TRN-1：设备无关（见 §8）；`--overfit-episode <seed>` 模式：只用一个 episode 训练。
- FR-TRN-2：支持 resume；checkpoint 含权重、配置快照、git hash、库版本。
- FR-TRN-3：可选 EMA（初值 0.999），评估默认用 EMA 权重。
- FR-TRN-4：日志写 `runs/<name>/log.jsonl`；默认不依赖外部实验跟踪服务。

---

## 7. 评估规格（`rvz.evaluate`）

### 7.1 指标（对无人机图像真值，在 `[0,1]` 范围计算）
- **LPIPS**（主指标）、**PSNR**、**SSIM**（Gaussian 11×11，σ=1.5，RGB 通道均值）。
- D1 规模不报 FID/KID。

### 7.2 基线
| 名称 | 定义 |
|---|---|
| `regress` | §6.1 回归基线 |
| `copy_nearest_cam` | 选光轴与无人机光轴夹角最小的小车相机，缩放到无人机分辨率后直接输出 |
| 消融 | 去位姿条件 / 去时间上下文 / 去递归 / 去 camera dropout / 冻结 vs LoRA |

### 7.3 分桶
因 FR-SIM-6 将无人机固定在严格正后方（无前/左/右象限变化），按以下维度分桶并分别报告：
- **跟随距离**：近 / 中 / 远（含弹簧臂缩短段，例如 `<1.2 m`、`1.2–1.8 m`、`≥1.8 m`，初值）。
- **无人机高度**：低 / 中 / 高（例如 `<0.6 m`、`0.6–1.0 m`、`≥1.0 m`，初值）。
- **小车转弯角速度（步级 `|Δyaw|`）**：直行 / 缓弯 / 急弯（例如 `<2°/步`、`2–6°/步`、`≥6°/步`，初值；episode 级可参考 `meta_json.diagnostics.max_step_yaw_deg`）。
- **弹簧臂与遮挡状态**：正常跟随 / 触发弹簧臂缩短 / 触发遮挡物隐藏（对应 `meta_json.diagnostics`）。

### 7.4 递归漂移
对同一无人机轨迹滚动推理 `K=10` 步，报告各步 LPIPS 曲线。

### 7.5 延迟（FR-EVL）
batch=1；预热 10 次，测 50 次；计时前后做设备同步；报告 1/2/4/50 步的均值±标准差；CUDA 额外报告峰值显存。**每台机器单独报告**。

### 7.6 产物
`runs/<name>/report.json` + 对比图（上下文 4 图 | 预测 | 真值 | 差异）。

---

## 8. 平台与性能需求

| 项 | Linux + RTX 5060 Ti 16G | Mac M1 Pro |
|---|---|---|
| 角色 | 最终数据集生成、特征缓存、全部训练 | 开发调试、D0、`fm`/`fm_rec`、推理与可视化 |
| MuJoCo 渲染 | EGL（需 NVIDIA EGL/GLVND 用户态库） | 默认后端（cgl）；MuJoCo 原生被动查看器需 `mjpython` |
| 精度 | bf16 autocast | **fp32**（M0 实测 MPS bf16 matmul 可运行；fp32 为既定策略，不变） |
| PyTorch | **CUDA 12.8+ 构建**（Blackwell sm_120 需要 PyTorch ≥ 2.7） | MPS；最新版要求 macOS 15+、Python 3.11+ |
| 注意力 | `scaled_dot_product_attention` | 同左；不依赖 flash-attn/xformers |
| `torch.compile` | 可选 | **不使用** |

- **Python 版本**：PyPI 上 MuJoCo 4.0.1 与 `mujoco-core` 4.0.1 的元数据声明支持 Python 3.11–3.15；`mujoco-core` 已为 macOS arm64 和 Linux x86_64 提供 CPython 3.15 wheel。项目配置限制为 `>=3.11,<3.16`。M0 实测（darwin arm64）：Python 3.11.15 + MuJoCo 4.0.1 + torch 2.15.0 + transformers 5.24.1 安装运行通过，`uv.lock` 双平台解析成功；Linux 侧安装待运行。
- 同一场景同一位姿在两个平台渲染的像素可能略有差异。**最终训练/评估数据只在 Linux 生成**；Mac 的 D0 仅用于调试，不与最终数据混用。
- 配置按机器分两份（`configs/mac_debug.yaml`、`configs/linux_main.yaml`）；差异只允许出现在配置里，不允许出现在代码分支里。

### 8.1 配置差异（初值，估计）
| 项 | `mac_debug` | `linux_main` |
|---|---|---|
| 小车相机分辨率 | 256 | 128 → 224 |
| 无人机分辨率 | 256 | 128 |
| 生成器 dim/depth/heads | 约 256/4/4（3–5M） | 512/12/8（30–40M） |
| batch | 8–16 | 32–64 |
| 精度 | fp32 | bf16 |
| 数据 | D0 | D1 → D2 |

---

## 9. 命令行（FR-CLI）

入口：`rvz <子命令>`（`uv run rvz ...`）。

| 子命令 | 作用 |
|---|---|
| `doctor` | M0 可行性检查，输出 ✅/❌ 表，写 `runs/doctor_<platform>.json` |
| `gen-data` | 生成 episode（`--seeds a:b --workers N --out data/raw`） |
| `viz-episode` | 把一个 episode 导出为图片/视频，并画出位姿重投影检查图 |
| `viewer` | 按 `sim.dt` 实时预览指定 `--seed` 的仿真，左侧小车相机四宫格，右侧 `--drone-index` 指定的无人机视角 |
| `cache-feats` | 缓存编码器特征（幂等，manifest 校验） |
| `train` | 训练（`--stage`、`--run`、`--resume`、`--overfit-episode`） |
| `sample` | 用 checkpoint 对指定 episode 采样并保存图片 |
| `eval` | 评估（`--steps 1,2,4,50`），输出 `report.json` |
| `bench` | 延迟与显存测试 |

所有子命令统一接收 `--config`，并支持 `--set a.b.c=value` 覆盖配置。

`viewer` 使用 Tk 桌面窗口，默认 `--seed 0 --drone-index 0`。默认配置 `configs/viewer.yaml` 选用导入的小型仓库，小车每路原生渲染 512×512，无人机原生渲染 768×768。小车相机按 `front, left, right, back` 从左到右、从上到下排列，与无人机画面同步渲染。沿路径循环，关闭窗口或按 `Esc` 退出。

---

## 10. 非功能需求
- **确定性**：给定 `(seed, 代码版本)`，数据生成可复现；训练固定随机种子并在 `meta` 中记录。
- **最小依赖**：运行依赖仅 `numpy, mujoco, torch, torchvision, transformers, peft, lpips, einops, pillow, pyyaml, tqdm, tyro`；开发依赖仅 `pytest, ruff`。新增依赖需先在 §12 记录理由。
- **可测试**：`pytest -m "not slow and not gpu"` 在 Mac CPU 上跑完，且不下载任何大文件。
- **日志**：库代码用 `logging`，不用 `print`。
- **文件体量**：单文件尽量不超过约 400 行；超出先考虑拆分。

---

## 11. 验收门槛与里程碑

验收阈值均为**初值**，首次实测后可修订，修订需记入 §12，**不得静默改动**。

### 11.1 门槛
| 门槛 | 通过条件 |
|---|---|
| **G0** | `doctor` 在两台机器上全部 ✅，或每个 ❌ 项有记录在案的替代方案 |
| **G1** | 随机 100 个 seed 的仓库布局与空闲栅格有效、平滑路径无碰撞；位姿重投影误差 ≤ 2 像素；无人机相机无穿模且小车在视野内；`validate_episode` 全通过；Linux 吞吐已记录；画面门槛：人眼判断为小型仓库且转弯跟随平滑 |
| **G2** | 特征缓存可重复生成；`regress` 与 `copy_nearest_cam` 基线能在 val 上算出全部指标 |
| **G3** | ① 单 episode 过拟合：train LPIPS ≤ 0.10（初值）；② 未见轨迹/机位（val）上 LPIPS 优于 `regress` 与 `copy_nearest_cam` |
| **G4** | 10 步滚动推理的 LPIPS 漂移：第 10 步 ≤ 1.3 × 第 1 步（初值）；且优于无递归版本 |
| **G5** | 4 步 MeanFlow 的 val LPIPS 与 `fm` 50 步相比，退化 ≤ 10%（相对，初值）；两台机器延迟已记录 |
| **G6** | LoRA 版 val LPIPS 相对冻结版提升 ≥ 3%（相对，初值）；否则冻结版为最终结果 |
| **G7** | 完整对比表 + 分桶报告 + 漂移曲线 + 延迟表 + 定性图齐全 |

**最重要的体检是 G3① 单 episode 过拟合**：过不了说明管线有 bug（位姿约定、归一化、条件注入），不得进入后续里程碑。

### 11.2 里程碑（不含时间，按依赖顺序）

| 里程碑 | 交付物 | 退出条件 | 依赖 |
|---|---|---|---|
| **M0 环境与可行性** | `pyproject.toml`（uv，平台标记的 torch 源）、`config.py`、`device.py`、`doctor.py` | G0；§12.2 的未决问题逐项有结论 | — |
| **M1 仿真与数据管线** | `geometry.py`、`sim/*`、`data/schema.py`、`gen-data`、`viz-episode`、`viewer`、对应测试 | G1 | M0 |
| **M2 特征缓存与基线** | `models/encoder.py`、`data/features.py`、`data/dataset.py`、`cache-feats`、`regress` 阶段、`evaluate/metrics.py` 与基线 | G2 | M1 |
| **M3 flow matching** | `models/rays.py`、`models/generator.py`、`models/sampler.py`、`train/*`（`fm`）、`sample` | G3 | M2 |
| **M4 时间递归** | `fm_rec` 阶段、滚动推理、漂移评估 | G4 | M3 |
| **M5 MeanFlow** | `meanflow` 阶段（`pmf`/`emf`）、少步采样、`bench` | G5 | M4 |
| **M6 自监督微调** | `lora` 阶段、`aux_cross_cam`、冻结 vs LoRA 对照 | G6 | M5 |
| **M7 评估与报告** | `eval` 全功能、分桶、报告与图 | G7 | M6 |
| **M8（可选扩展）** | RoboCasa 适配器（独立环境，输出同 schema）、224 分辨率、2–4 帧小片段输出、Self-Forcing 式滚动训练 | 各自单独立项 | M7 |

---

## 12. 变更记录与未决问题

### 12.1 变更记录
| 日期 | 章节 | 变更 | 原因 |
|---|---|---|---|
| 2026-11-25 | 全文 | 初稿 | — |
| 2026-11-25 | 全文、§9 | 项目更名为 `reality-view-zero`，Python 包与 CLI 更名为 `rvz` | 统一项目命名 |
| 2026-11-25 | §8；ARCH §2.1 | Python 范围改为 `>=3.11,<3.16`；补充 `uv_build`、Ruff 与 pytest 配置 | 依据 MuJoCo PyPI 元数据并统一工程检查 |
| 2026-11-25 | M0；§4 FR-SIM-9；§8；§12.2；ARCH §5.1 | M0 交付：uv 骨架、`config`/`device`/`doctor`/`cli`、双平台配置与测试；G0（darwin）通过；记录 MuJoCo elevation 符号约定、MPS JVP 可用、MPS bf16 可运行（fp32 策略不变）等实测结论 | 环境与可行性验证 |
| 2026-11-25 | §12.2-7；`doctor` | 真实 DINOv3-B 权重与 HF 协议验证推迟到 M2；M0 改用本地随机初始化小 DINOv3-ViT 离线验证架构支持与前向 | 模型体积大且本机代理网络受限，M0 不需要真实权重 |
| 2026-11-25 | pyproject；ARCH §2.1 | 默认 PyPI 索引与 `pytorch-cu128` 索引均改为南京大学镜像（`mirrors.nju.edu.cn/pypi/web/simple`、`mirrors.nju.edu.cn/pytorch/whl/cu128`） | 国内网络环境下提升解析与下载速度 |
| 2026-11-25 | §9、§10；ARCH §2、§2.1 | CLI 从 `argparse` 迁移到 `tyro`，新增 `tyro` 运行依赖；保留现有子命令、参数与退出码 | 用类型标注生成参数解析，减少手工解析与调度样板 |
| 2026-11-25 | M1；§2、§3、§4、§9；ARCH §5 | 实现几何转换、确定性布局/A*、运动学录制、schema v1、`gen-data` 与 `viz-episode`；`project` 同时返回深度，`robot_poses` 显式接收速度；Mac 生成 D0 train 20 + val 4，24 个 episode 校验通过，最大标记重投影误差 0.898 px，默认测试通过；G1 的 Linux 吞吐与人工画面判断待验收 | 完成仿真与数据管线；阈值和 schema v1 字段不变 |
| 2026-11-25 | §4 FR-SIM-6；ARCH §5.2 | 无人机改为沿路径后方跟随的低机位第三人称视角：高度初值 1.2–2.5 m、俯仰 −35° 至 −20°、初始方位偏置 ±15°；遮挡时全局选择平滑方位修正，相邻步方位差 ≤15°；可视化增加 `Drone`/`Pose check` 标题和图例，投影叠加在放大后绘制；最终 Mac D0 最大重投影误差 1.195 px，默认测试 51 passed；G1 的 Linux 吞吐与人工画面判断待验收 | 高俯视机位可读性差，固定后方机位在部分布局与转弯处被货架遮挡；G1 的 2 px 阈值和 schema v1 不变 |
| 2026-11-25 | §4 FR-SIM-6/7/9、§5.2、§8.1；ARCH §5.2 | 按目标视角改为严格正后方：移除随机偏置、角速度配置和侧向避让；仅对当前无人机画面用 `mj_ray` 查找并隐藏遮挡物，其他相机与布局不变；单目位姿改读左右 GL 相机平均；Mac D0 和预览升至 256×256；24 个 episode 校验通过，最大重投影误差 0.626 px，正后方横向偏差 ≤5.4e-7 m，默认测试 51 passed | 用户要求正后方而非斜后方，并要求遮挡物自动隐藏和更高图像分辨率；G1 阈值与 schema v1 不变 |
| 2026-11-25 | §4 FR-SIM-6；ARCH §5.2 | 无人机标称距离缩至 1.4–2.2 m，高度范围降至 0.3–1.5 m、俯仰改为 −30° 至 −15°；加入正后方弹簧臂，遇几何体时平滑缩短距离（最短 0.45 m、相邻步 ≤0.12 m），不穿模且不转斜后方；最终 24 个 256×256 episode 最大重投影误差 0.829 px，默认测试 52 passed | 用户要求相机更近、更低；固定近距离在少数转弯布局会穿墙；G1 阈值与 schema v1 不变 |
| 2026-11-25 | §2、§3.1、§4 FR-SIM-5/8；ARCH §2、§4、§5 | 左右小车相机改为 120° 垂直 FOV、下倾 15°，前后和无人机保持 90°；`K_robot` 改为逐相机 `[4,3,3]`，schema 升至 v2；24 个 256×256 episode 全部通过，最大重投影误差 0.829 px，默认测试 53 passed | 参考量产车侧后视镜/环视相机的侧装与广角覆盖，增加近车侧地面可见性；不模拟鱼眼畸变，G1 的 2 px 阈值不变 |
| 2026-11-25 | pyproject；§9、§12.2-7 | 新增运行依赖 `socksio`（`uv add socksio`）；Linux 上 `rvz doctor --download` 完成真实 DINOv3-B 下载与 CUDA 前向（tokens shape=(1,201,768)），HF gated 协议经 huggingface-cli 凭据通过；G0（linux）通过 | 本机经 SOCKS 代理（socks5://127.0.0.1:7890）访问 Hugging Face，huggingface_hub 的 httpx 客户端需要 `socksio` 才能走该代理；见 §12.2-7 修正 |
| 2026-11-25 | §12.2-6/10/11；`doctor` | G0 双平台通过：跨平台像素自动对比平均绝对差 0.138/255（最大 15，相机位置差 0 m）；npz 跨机拷贝实测读入成功；双平台 `uv sync` 均通过；`doctor` 的 `uv_lock`/`npz_roundtrip` 待复核平台消息改为按平台生成（新增 `_pending_platform()` 与回归测试，默认测试 54 passed） | Mac 补齐 G0 剩余项；修复 Linux 上"待 Linux 运行"的自相矛盾消息 |
| 2026-11-25 | §8、§9、§11.2；ARCH §2、§5.2、§7 | 增加 `rvz viewer`，同步预览四路小车相机与一路无人机视角 | 运行时检查相机画面与无人机跟随效果 |
| 2026-11-26 | §3.2、§4 FR-SIM-4/12、§5.2、§9；ARCH §2、§5.2 | 导入 AWS RoboMaker 无屋顶小型仓库作为默认预览场景，小车相机升至 512×512，无人机升至 768×768。细化小车外观并调整广角相机近裁剪面 | 使用现成仓库资产改善画面，提升预览清晰度 |
| 2026-11-26 | ARCH §5.2 | 场景组装收敛到 `scene.build_scene`（record 不再持有布局→模型逻辑）；新增 `layout.required_path_length`、`geometry.marker_world`、`rig.LOOKAT_HEIGHT_OFFSET_M` 消除重复公式与魔法数；仓库模型按 `cfg` 缓存；`SimConfig.scene` 用 `SceneName` Literal 单一来源 | `/simplify` 复审：去重与抬高抽象层次，行为不变，默认测试 61 passed |
| 2026-11-26 | §4 FR-SIM-7；ARCH §5.2 | 录制与预览共用无人机单帧校验和整段轨迹重采样 | 前序机位重采样时，预览与录制的机位选择仍逐帧一致 |
| 2026-11-26 | ARCH §5.2 | 公开 `rvz.sim.record.marker_centroid` | 通过公开入口覆盖灯标检测的回归测试 |
| 2026-11-26 | §4 FR-SIM-2；ARCH §5.2 | `sample_layout` 分别返回 `Layout` 和 `ProceduralColors`，`layout_to_mjcf` 显式接收颜色 | 程序化纹理颜色随布局采样，导入仓库直接使用资产材质 |
| 2026-11-26 | §9；ARCH §2、§5.2 | `viewer` 默认加载随包分发的 `configs/viewer.yaml`，子命令统一加载配置 | 高清预览和录制示例共用一份小型仓库配置 |
| 2026-11-26 | §4 FR-SIM-3；ARCH §5.2 | `plan_path` 改用 8 邻接 A*、视线剪枝、防 180° 折返与二次 Bézier 圆角平滑；`robot_poses` 改用定步弦长采样和平滑切线朝向 | 消除 90°/180° 航向与跟随机位瞬时跳变，提高短序列转弯覆盖率；G1 阈值与 schema v2 不变 |
| 2026-11-26 | §3.1/§3.2、§4 FR-SIM-1/2/3/12、§7.3、§11.1；ARCH §2/§5.2/§5.6/§11；`config.py`/`layout.py`/`scene.py`/`record.py` | ① 按用户最新需求全仓库统一改用 `small_warehouse`（AWS RoboMaker 无屋顶小型仓库），移除原简化 `procedural` 场景及相关配置项（`warehouse_min_size`/`warehouse_max_size`/`min_aisles`/`max_aisles`、`sample_layout`、`layout_to_mjcf`、`ProceduralColors`），`SimConfig.scene` 默认并限定为 `"small_warehouse"`；② 固定 `val` 目检样例更新为 `seed=10000` 与 `seed=10008`；③ 在 `meta_json` 中增加轻量 `diagnostics` 统计（不改数组字段，不升 `SCHEMA_VERSION`），同步修订 §7.3 分桶维度对齐 FR-SIM-6 严格正后方机位（距离/高度/步级转弯角速度/弹簧臂与遮挡状态）；④ `plan_path` 起点增加正后方间隙检查，收紧 `marker_centroid` 品红比例阈值；⑤ 产出 `runs/gates/G1_darwin.json`：100/100 `small_warehouse val`（10000:10100，40 步 128×128）全通过，最大重投影误差 0.493 px（均值 0.295 px），最大单步偏航 16.68°（p95=12.55°，93/100 含 ≥15° 转弯），零重采样，Mac 4-worker 吞吐 952.4 帧/s（3.40 ep/s），默认测试 61 passed；Linux EGL 吞吐与 D1 生成待 Linux 运行 | 用户明确要求以后统一使用 `small_warehouse` 并全仓库同步适配，同时完成 M1 诊断与评估规范对齐；G1 2 px 阈值与 schema v2 不变 |
| 2026-11-26 | M2；§3.4、§5.1、§6.1、§6.4、§7.1–7.2、§9、§12.2-7；ARCH §2、§5.3–5.6、§8 | M2 交付：① 新增 `models/encoder.py`（冻结 DINOv3-B、可选 `pool_2x2` 与 `enable_lora`）、`models/regress.py`（`RegressBaseline` 直接回归基线）、`data/features.py`（幂等 fp16 特征缓存与严格 `manifest.json` 校验，通过 `PatchEncoder` Protocol 保持 `data` 不反向依赖 `models`）、`data/dataset.py`（`Sample`/`SampleBatch`、目标 Base 系位姿换算与 schema v2 逐相机内参 `K_ctx` `(4,3,3)` 同步缩放）、`train/losses.py`、`train/loop.py`（统一训练循环，支持 `regress`、`--overfit-episode`、`--resume`、带步数预热的 EMA 与 JSONL 日志）、`evaluate/metrics.py`（`[0,1]` 范围 LPIPS/PSNR/SSIM、`copy_nearest_cam` 与 checkpoint 评估）及 `cache-feats`/`train`/`eval` CLI；② 补齐 SPEC §12.2-7：真实 DINOv3-B 权重（revision `5931719e`）在 Mac M1 Pro MPS 上前向通过；③ 产出 `runs/gates/G2_darwin.json`：Mac D0（train 20 + val 4，256×256）特征缓存幂等且重复提取逐字节一致（`max_abs_diff=0.0`），val（`n=348`）指标 `copy_nearest_cam`（LPIPS 0.6325 / PSNR 14.57 dB / SSIM 0.4916）、`regress` 120 步 EMA（LPIPS 0.6527 / PSNR 17.11 dB / SSIM 0.6492），默认测试 83 passed、`ruff` 全通过；Linux D1 缓存与基线评估待 Linux 运行 | 完成特征缓存与基线里程碑并通过 G2（darwin）；同步 ARCH §5.3 的 `K_ctx` 形状对齐 schema v2；为 EMA 增加步数预热 `min(decay, (1+step)/(10+step))` 解决短步数训练 EMA 滞后初始化权重的问题；无新增运行依赖，验收阈值与 schema v2 不变 |
| 2026-11-26 | M2 补齐 + M3；§2.5、§5.2–5.4、§6.1、§7.6、§9、§11.1 G3；ARCH §5.3–5.6 | ① Linux G2 落盘 `runs/gates/G2_linux.json`（D1 train 300 + val 30，DINOv3-B 缓存 330/330 幂等通过；val `n=3510`：`copy_nearest_cam` LPIPS 0.6049 / PSNR 14.70 dB / SSIM 0.4376，`regress` 3000 步 EMA LPIPS 0.4221 / PSNR 20.87 dB / SSIM 0.6287）；② 前置修补：`DataConfig.max_cached_episodes` 增至 512、`EpisodeDataset` 延迟缓存 `.npy` mmap 句柄、`linux_main.yaml` 设 `train.num_workers=4` 且训练 DataLoader 启用 `persistent_workers`；`loop.py` 将损失移出 `autocast_ctx` 且 `LpipsLoss` 内显式关闭外层 autocast 确保 VGG LPIPS 真正运行在 fp32；③ M3 交付：新增 `models/rays.py`（`plucker_rays` + numpy 参考）、`models/generator.py`（像素空间 DiT Transformer `Generator`，x-prediction、QK-Norm、共享 Plücker 射线嵌入、AdaLN-Zero、随机相机丢弃 `p_drop`、预留 `(r, t)` 与 `math_attention_ctx` 接口；Linux `512/12/8` 约 71.7M 参数，Mac `256/4/4` 约 6.8M 参数）、`models/sampler.py`（Flow Matching Euler ODE 采样）、`train` 接入 `--stage fm`、`evaluate/report.py` 与 `rvz sample` 对比图导出及 `rvz eval --steps` 支持；④ 产出 `runs/gates/G3_linux.json`（RTX 5060 Ti 16G）：G3① `seed=2`（99.5° 转弯）2000 步过拟合 50 步 ODE 采样 EMA LPIPS `0.0474`（≤ 0.10）、PSNR `29.93 dB`、SSIM `0.9162`，目检车体品红灯标与货架对齐；G3② D1 全量 6000 步（`0.347s/step`）在 `val`（`n=3510`）50 步 EMA LPIPS `0.2454`（较 `regress` 降低 41.85%、较 `copy_nearest_cam` 降低 59.42%）、PSNR `21.27 dB`、SSIM `0.6722`，已导出并目检 `seed=10000`、`seed=10008` 对比图；默认测试 99 passed、`ruff` 全通过；Mac D0 复验待运行 | 完成 M3 像素空间 Flow Matching 里程碑并通过 G3（linux），同时消除 M2 实测暴露的 D1 I/O 与 bf16 LPIPS 量化问题；无新增运行依赖，验收阈值与 schema v2 不变 |

### 12.2 M0 必须解决的未决问题（均对应 `doctor` 检查项）

> 结论标注：「✅ darwin」= 2026-11-25 在 Mac M1 Pro（macOS 26，arm64）实测通过，数据见 `runs/gates/G0_darwin.json`；「待运行」= 尚未在该平台执行，不得视为已验证。Linux 统一复核命令：`uv sync && uv run rvz doctor --download`。

1. MuJoCo 离屏渲染在 Mac（cgl）与 Linux（EGL）上能否各出一张图。
   → ✅ darwin：固定相机与自由相机均出图，像素统计正常。Linux（EGL）：待运行。
2. 自由相机方位角扫 0–360° 无花屏；位姿读出与重投影误差 ≤ 2 像素。
   → ✅ darwin：24 个视角（12 方位 × 2 俯仰）无花屏，最大重投影误差 **0.885 像素**（GL 约定闭环，与官方 `compute_camera_matrix` 公式一致）。附带发现：`mjvCamera.elevation` **负值 = 俯视**（已记入 FR-SIM-9）。Linux：待运行。
3. M0 最小场景的固定相机与自由相机内参一致；离屏缓冲尺寸足够（M1 左右侧方相机的独立广角参数另由 schema v2 与集成测试覆盖）。
   → ✅ darwin：M0 最小场景 `fovy` 均为 90°；离屏缓冲 640×480 ≥ 128×128。M1 前后/自由相机 90°、左右相机 120° 下倾 15° 已实测；Linux：待运行。
4. 128×128 渲染吞吐（单进程 / 多进程，两台机器）。
   → ✅ darwin：单进程 **约 517 fps**（50 帧，含 `update_scene`）；spawn 2 worker 各约 88–92 fps。Linux：待运行。
5. 多进程渲染在两个平台上都稳定。
   → ✅ darwin：spawn 2 worker × 10 帧正常出图。Linux：待运行。
6. 同一场景同一位姿跨平台像素差异的量级。
   → ✅ 双平台（2026-11-25）：Mac（cgl）与 Linux（EGL）参考图自动对比，平均绝对差 **0.138/255**，最大 **15/255**，相机位置差 **0 m**；差异集中在个别像素。
7. DINOv3-B 前向在 MPS 与 CUDA 上可用；`transformers` 版本支持；HF 协议已同意。
   → ✅ 双平台（darwin + linux）：transformers 5.24.1 提供 `DINOv3ViTModel`/`DINOv3ViTConfig`。**真实 DINOv3-B 权重与 HF 协议验证** ✅ linux（2026-11-25）：`rvz doctor --download` 下载 gated 仓库 `facebook/dinov3-vitb16-pretrain-lvd1689m` 成功，CUDA 前向正常，tokens shape=(1,201,768)（cls + 4 register + 196 patch）；实测本机 SOCKS 代理需 `socksio`（已列入依赖）。**MPS 真实权重前向** ✅ darwin（2026-11-26，M2 实测）：本地缓存 `facebook/dinov3-vitb16-pretrain-lvd1689m`（revision `5931719e`）在 Mac M1 Pro MPS 上 128×128 前向通过（patch tokens `(2, 64, 768)` 全有限），并完成 D0 24 个 episode 离线特征缓存。
8. 5060 Ti 上 PyTorch CUDA 算子可运行（无 `no kernel image` 错误）。
   → 待运行：本机无 CUDA。Linux 需确认 torch 为 `+cu128` 构建（锁文件已含 cu128 索引条目）。
9. MPS 上 `torch.func.jvp` 是否可用。
   → ✅ **可用**：MPS 上 jvp 与中心差分最大偏差 6.86e-04（fp32）。`pmf` 变体在 Mac 上不被 JVP 否决；`emf` 仍保留为备选（ADR-4）。
10. 数据文件在两台机器间可无痛拷贝并读取。
    → ✅ 双平台：npz 压缩写入/读回一致（uint8/float32/字符串）；跨机实际拷贝已验证（`ref_linux.npz` 拷至 Mac 后读入并对比成功）。
11. `uv sync` 在两个平台上都能解析并安装锁定的依赖。
    → ✅ 双平台：darwin（Python 3.11.15，torch 2.15.0、MuJoCo 4.0.1、transformers 5.24.1）与 linux（Python 3.11.16，torch 2.11.0+cu128、MuJoCo 4.0.1、transformers 5.24.1）`uv sync` 均成功；`uv.lock` 含 darwin arm64 与 linux x86_64 双平台解析。
12. `lpips` 首次使用会下载权重：确认两台机器的下载与缓存路径。
    → ✅ darwin：VGG 骨干下载与前向验证通过，缓存目录 `~/.cache/torch/hub/checkpoints`；已缓存时 `rvz doctor` 离线复验，未缓存需 `--download`。Linux：待运行。

附加项（§8 未核实）：✅ darwin：`mps_bf16` 检查实测 **MPS bf16 matmul 可运行**（原"不支持"假设不成立）；MPS/CPU 训练策略保持 fp32 不变。
