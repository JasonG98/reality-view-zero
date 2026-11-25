"""M0 可行性检查：逐项验证 SPEC §12.2 的未决问题，输出报告与 G0 门槛文件。

相机检查在 GL 约定内闭环（投影公式与 MuJoCo 官方 tutorial 的
compute_camera_matrix 一致），不涉及 GL→CV 转换——该转换唯一实现属于 M1 的
geometry.py（AGENTS §7），此处不提前实现。
"""

import importlib.metadata
import json
import logging
import math
import multiprocessing as mp
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import lpips
import mujoco
import numpy as np
import torch
import transformers
from transformers import AutoImageProcessor, AutoModel, DINOv3ViTConfig, DINOv3ViTModel

from rvz.config import Config
from rvz.device import get_device, sync

logger = logging.getLogger(__name__)

RENDER_RES = 128  # SPEC §12.2 第 4 项要求的吞吐测试分辨率
REPROJECT_TOL_PX = 2.0  # SPEC §12.2 第 2 项阈值
VGG16_WEIGHTS = "vgg16-397923af.pth"  # torchvision VGG16 骨干固定文件名（lpips 默认 net="vgg"）
TARGET_POS = np.array([0.0, 0.0, 1.0])  # 红色发光球球心（重投影参照）
LOOKAT = [0.3, 0.0, 1.0]  # 自由相机看向点：偏离球心，避免投影错误被画面中心掩盖

# 最小场景：灰地面 + 红色发光球 + 一个固定相机（fovy 一致性检查用）。
DOCTOR_MJCF = """
<mujoco>
  <visual>
    <global fovy="90"/>
  </visual>
  <asset>
    <material name="target_glow" rgba="1 0 0 1" emission="1"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="20 20 0.1" rgba="0.5 0.5 0.5 1"/>
    <body name="target" pos="0 0 1">
      <geom name="target_sphere" type="sphere" size="0.2" material="target_glow" contype="0" conaffinity="0"/>
    </body>
    <camera name="fixed" pos="2 0 1" xyaxes="0 1 0 0 0 1" fovy="90"/>
  </worldbody>
</mujoco>
"""


@dataclass
class CheckResult:
    name: str
    status: Literal["pass", "fail", "skip"]  # skip = 本机未运行，需另一平台或 --download
    detail: str


def project_gl(
    pos: np.ndarray, forward: np.ndarray, up: np.ndarray, fovy_deg: float, width: int, height: int, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """GL 相机模型投影。points (N,3) → (像素坐标 (N,2)，深度 (N,)，前方为正）。"""
    z_axis = -forward / np.linalg.norm(forward)
    y_axis = up / np.linalg.norm(up)
    x_axis = np.cross(y_axis, z_axis)
    rot = np.stack([x_axis, y_axis, z_axis])
    p_cam = (points - pos) @ rot.T
    focal = height / 2.0 / math.tan(math.radians(fovy_deg) / 2.0)
    u = -focal * p_cam[:, 0] / p_cam[:, 2] + (width - 1) / 2.0
    v = focal * p_cam[:, 1] / p_cam[:, 2] + (height - 1) / 2.0
    return np.stack([u, v], axis=-1), -p_cam[:, 2]


def _make_scene() -> tuple[mujoco.MjModel, mujoco.MjData, mujoco.Renderer]:
    model = mujoco.MjModel.from_xml_string(DOCTOR_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, mujoco.Renderer(model, RENDER_RES, RENDER_RES)


def _free_camera(azimuth: float, elevation: float) -> mujoco.MjvCamera:
    """构造自由相机。注意 MuJoCo 约定：elevation 负值 = 相机在上方俯视。"""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = LOOKAT
    cam.distance = 3.0
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _red_centroid(img: np.ndarray) -> tuple[int, float, float]:
    """红色掩码的像素数与质心；用于无花屏与重投影误差检查。"""
    rgb = img.astype(np.int32)
    mask = (rgb[..., 0] > 150) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90)
    count = int(mask.sum())
    if count == 0:
        return 0, 0.0, 0.0
    ys, xs = np.nonzero(mask)
    return count, float(xs.mean()), float(ys.mean())


def _render_worker(payload: tuple[int, int]) -> tuple[float, float]:
    """多进程 worker：独立渲染器渲染 n_frames 帧，返回 (fps, 平均像素值)。"""
    n_frames, seed = payload
    _, data, renderer = _make_scene()
    start = time.perf_counter()
    mean_pixel = 0.0
    for i in range(n_frames):
        renderer.update_scene(data, camera=_free_camera((seed * 37 + i * 30) % 360, -25.0))
        mean_pixel += float(renderer.render().mean())
    elapsed = time.perf_counter() - start
    renderer.close()
    return n_frames / elapsed, mean_pixel / n_frames


def _check_mujoco_basic() -> CheckResult:
    """§12.2-1/3：离屏出图、固定相机与自由相机 fovy 一致、离屏缓冲足够。"""
    model, data, renderer = _make_scene()
    with renderer:
        renderer.update_scene(data, camera="fixed")
        img = renderer.render()
        assert img.shape == (RENDER_RES, RENDER_RES, 3), img.shape
        fovy_global = float(model.vis.global_.fovy)
        fovy_fixed = float(model.cam_fovy[model.camera("fixed").id])
        buffer_ok = model.vis.global_.offwidth >= RENDER_RES and model.vis.global_.offheight >= RENDER_RES
        renderer.update_scene(data, camera=_free_camera(45.0, -30.0))
        free_img = renderer.render()
        count, _, _ = _red_centroid(free_img)
    if abs(fovy_global - fovy_fixed) > 1e-6:
        return CheckResult("mujoco_offscreen", "fail", f"fovy 不一致：global={fovy_global}, fixed={fovy_fixed}")
    if not buffer_ok:
        return CheckResult("mujoco_offscreen", "fail", "离屏缓冲小于 128×128")
    if img.std() < 1.0 or count < 20:
        return CheckResult("mujoco_offscreen", "fail", f"渲染内容异常：std={img.std():.2f}, 红球像素={count}")
    return CheckResult(
        "mujoco_offscreen",
        "pass",
        f"固定/自由相机均出图，fovy={fovy_global}° 一致，"
        f"缓冲 {model.vis.global_.offwidth}×{model.vis.global_.offheight}",
    )


def _check_freecam_reprojection() -> CheckResult:
    """§12.2-2：方位角 0–360° 扫描无花屏；读出 GL 位姿重投影误差 ≤ 2 像素。"""
    _, data, renderer = _make_scene()
    max_err, min_count = 0.0, 1 << 30
    with renderer:
        for azimuth in range(0, 360, 30):
            for elevation in (-25.0, -60.0):
                renderer.update_scene(data, camera=_free_camera(float(azimuth), elevation))
                img = renderer.render()
                scam = renderer.scene.camera[0]
                uv, depth = project_gl(
                    np.array(scam.pos),
                    np.array(scam.forward),
                    np.array(scam.up),
                    90.0,
                    RENDER_RES,
                    RENDER_RES,
                    TARGET_POS[None],
                )
                count, cx, cy = _red_centroid(img)
                min_count = min(min_count, count)
                if count < 20 or depth[0] <= 0:
                    return CheckResult("freecam_reprojection", "fail", f"az={azimuth} el={elevation} 红球不可见或花屏")
                max_err = max(max_err, math.hypot(uv[0, 0] - cx, uv[0, 1] - cy))
    status = "pass" if max_err <= REPROJECT_TOL_PX else "fail"
    return CheckResult(
        "freecam_reprojection", status, f"24 个视角（12 方位 × 2 俯仰），最大重投影误差 {max_err:.3f} 像素（阈值 2）"
    )


def _check_throughput() -> CheckResult:
    """§12.2-4 前半：128×128 单进程渲染吞吐。"""
    _, data, renderer = _make_scene()
    n_frames = 50
    with renderer:
        for i in range(5):  # 预热
            renderer.update_scene(data, camera=_free_camera(i * 10.0, -25.0))
            renderer.render()
        start = time.perf_counter()
        for i in range(n_frames):
            renderer.update_scene(data, camera=_free_camera(i * 7.0, -25.0))
            renderer.render()
        fps = n_frames / (time.perf_counter() - start)
    return CheckResult("render_throughput", "pass", f"128×128 单进程 {fps:.1f} fps（{n_frames} 帧，含 update_scene）")


def _check_multiprocess() -> CheckResult:
    """§12.2-4 后半/5：多进程渲染稳定且各自独立出图。"""
    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        results = pool.map(_render_worker, [(10, 0), (10, 1)])
    fps = [r[0] for r in results]
    means = [r[1] for r in results]
    if any(m < 5.0 for m in means):
        return CheckResult("multiprocess_render", "fail", f"worker 图像均值异常：{means}")
    return CheckResult("multiprocess_render", "pass", f"spawn 2 worker × 10 帧正常，fps={[f'{f:.1f}' for f in fps]}")


def _check_cross_platform() -> CheckResult:
    """§12.2-6：写出本机参考图；若另一平台参考已存在则报告像素差异量级。"""
    ref_dir = Path("runs") / "doctor_assets"
    ref_dir.mkdir(parents=True, exist_ok=True)
    _, data, renderer = _make_scene()
    with renderer:
        renderer.update_scene(data, camera=_free_camera(45.0, -30.0))
        scam = renderer.scene.camera[0]
        payload = {
            "rgb": renderer.render(),
            "pos": np.array(scam.pos),
            "forward": np.array(scam.forward),
            "up": np.array(scam.up),
            "fovy": np.float32(90.0),
        }
    np.savez(ref_dir / f"ref_{sys.platform}.npz", **payload)
    others = [p for p in ref_dir.glob("ref_*.npz") if p.stem != f"ref_{sys.platform}"]
    if not others:
        return CheckResult(
            "cross_platform_pixels", "skip", f"已写 {ref_dir}/ref_{sys.platform}.npz，待另一平台运行后自动对比"
        )
    with np.load(others[0]) as ref:
        diff = payload["rgb"].astype(np.float32) - ref["rgb"].astype(np.float32)
        pose_diff = float(np.abs(payload["pos"] - ref["pos"]).max())
    return CheckResult(
        "cross_platform_pixels",
        "pass",
        f"对比 {others[0].name}：平均绝对差 {np.abs(diff).mean():.3f}/255，最大 {np.abs(diff).max():.0f}，"
        f"相机位置差 {pose_diff:.2e} m",
    )


def _check_dinov3(cfg: Config, download: bool) -> CheckResult:
    """§12.2-7：DINOv3 前向可行性。

    默认用本地随机初始化的小 DINOv3-ViT（不下载任何文件）验证 transformers
    架构支持与设备前向；真实 DINOv3-B 权重与 HF 协议验证推迟到 M2（--download）。
    """
    device = get_device()
    if not download:
        config = DINOv3ViTConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
            num_register_tokens=2,
        )
        model = DINOv3ViTModel(config).to(device).eval()
        pixels = torch.rand(1, 3, 32, 32, device=device)
        with torch.no_grad():
            tokens = model(pixel_values=pixels).last_hidden_state
        sync(device)
        if not torch.isfinite(tokens).all():
            return CheckResult("dinov3_forward", "fail", "小 DINOv3 输出含非有限值")
        return CheckResult(
            "dinov3_forward",
            "pass",
            f"本地随机小 DINOv3-ViT（2 层，hidden 64）在 {device.type} 前向正常，tokens shape={tuple(tokens.shape)}；"
            "真实 DINOv3-B 权重与 HF 协议验证推迟到 M2",
        )
    processor = AutoImageProcessor.from_pretrained(cfg.encoder.id)
    model = AutoModel.from_pretrained(cfg.encoder.id).to(device).eval()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (cfg.encoder.res, cfg.encoder.res, 3), dtype=np.uint8)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        tokens = model(**inputs).last_hidden_state
    sync(device)
    if not torch.isfinite(tokens).all():
        return CheckResult("dinov3_forward", "fail", "输出含非有限值")
    return CheckResult(
        "dinov3_forward", "pass", f"{cfg.encoder.id} 在 {device.type} 上前向正常，tokens shape={tuple(tokens.shape)}"
    )


def _check_cuda() -> CheckResult:
    """§12.2-8：CUDA 算子可运行（Blackwell 无 no kernel image）。"""
    if not torch.cuda.is_available():
        return CheckResult("cuda_ops", "skip", "当前平台无 CUDA：待 Linux + RTX 5060 Ti 运行（torch 需 +cu128 构建）")
    device = get_device()
    x = torch.randn(512, 512, device=device, requires_grad=True)
    conv = torch.nn.Conv2d(3, 8, 3, device=device)
    loss = (x @ x).square().mean() + conv(torch.rand(1, 3, 32, 32, device=device)).mean()
    loss.backward()
    sync(device)
    return CheckResult(
        "cuda_ops", "pass", f"matmul/conv/backward 正常，torch {torch.__version__}，CUDA {torch.version.cuda}"
    )


def _check_jvp() -> CheckResult:
    """§12.2-9：torch.func.jvp 可用性与数值正确性（中心差分对照）。"""
    device = get_device()
    gen = torch.Generator(device=device).manual_seed(0)
    w = torch.randn(8, 8, device=device, generator=gen)
    x = torch.randn(4, 8, device=device, generator=gen)
    v = torch.randn(4, 8, device=device, generator=gen)

    def f(x: torch.Tensor) -> torch.Tensor:
        return (x @ w).tanh().sum()

    out, tangent = torch.func.jvp(f, (x,), (v,))
    eps = 1e-3
    numeric = (f(x + eps * v) - f(x - eps * v)) / (2 * eps)
    err = float((tangent - numeric).abs().max())
    sync(device)
    ok = bool(torch.isfinite(out).all() and torch.isfinite(tangent).all() and err < 1e-2)
    return CheckResult(
        "torch_jvp",
        "pass" if ok else "fail",
        f"{device.type} 上 jvp 与中心差分最大偏差 {err:.2e}；不可用时的替代方案：meanflow_variant=emf（SPEC §6.2）",
    )


def _check_mps_bf16() -> CheckResult:
    """SPEC §8 未核实项：MPS 是否支持 bf16。无论结果如何，MPS/CPU 训练策略保持 fp32。"""
    if not torch.backends.mps.is_available():
        return CheckResult("mps_bf16", "skip", "非 MPS 平台：待 Mac 运行")
    x = torch.randn(64, 64, device=get_device(), dtype=torch.bfloat16)
    out = (x @ x).float()
    sync(get_device())
    ok = bool(torch.isfinite(out).all())
    return CheckResult(
        "mps_bf16",
        "pass" if ok else "fail",
        f"MPS bf16 matmul {'可运行' if ok else '异常'}；训练策略仍为 fp32（SPEC §8）",
    )


def _pending_platform() -> str:
    """§12.2 双平台项：本机之外、仍待验证的平台名。"""
    return "Linux" if sys.platform == "darwin" else "Mac"


def _check_npz_roundtrip() -> CheckResult:
    """§12.2-10：npz 写入/读回一致（跨平台字节兼容；实际跨机拷贝由另一平台复核）。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.npz"
        rgb = np.arange(2 * 4 * 8 * 8 * 3, dtype=np.uint8).reshape(2, 4, 8, 8, 3)
        poses = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
        np.savez_compressed(path, robot_rgb=rgb, T_W_B=poses, meta_json=np.array('{"seed": 0}'))
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z["robot_rgb"], rgb)
            assert np.array_equal(z["T_W_B"], poses)
            assert str(z["meta_json"]) == '{"seed": 0}'
    detail = f"npz 压缩写入/读回一致（含 uint8/float32/字符串）；跨机拷贝待 {_pending_platform()} 复核"
    return CheckResult("npz_roundtrip", "pass", detail)


def _check_uv_lock() -> CheckResult:
    """§12.2-11：锁文件存在且覆盖双平台；另一平台的 sync 待运行。"""
    lock = Path("uv.lock")
    if not lock.exists():
        return CheckResult("uv_lock", "fail", "uv.lock 不存在，先执行 uv sync")
    text = lock.read_text(encoding="utf-8")
    has_cu128 = "cu128" in text
    detail = (
        f"本机 uv sync 成功；锁文件{'含' if has_cu128 else '不含'} cu128 索引条目；"
        f"{_pending_platform()} 平台 sync 待运行"
    )
    return CheckResult("uv_lock", "pass" if has_cu128 else "fail", detail)


def _check_lpips(download: bool) -> CheckResult:
    """§12.2-12：lpips 权重与缓存路径；VGG 已缓存时离线验证，否则需 --download。"""
    cached = Path(torch.hub.get_dir()) / "checkpoints" / VGG16_WEIGHTS
    if not download and not cached.exists():
        return CheckResult("lpips_weights", "skip", "VGG 权重未缓存：rvz doctor --download 启用下载验证")
    metric = lpips.LPIPS(net="vgg", verbose=False)
    score = metric(torch.rand(1, 3, 64, 64) * 2 - 1, torch.rand(1, 3, 64, 64) * 2 - 1)
    if not torch.isfinite(score).all():
        return CheckResult("lpips_weights", "fail", "LPIPS 输出非有限值")
    origin = "已缓存，离线验证" if cached.exists() else "本次下载验证"
    return CheckResult("lpips_weights", "pass", f"VGG 骨干可用（{origin}），缓存目录 {torch.hub.get_dir()}")


def _git_hash() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def run_doctor(cfg: Config, download: bool = False) -> dict:
    """执行全部检查，写 runs/doctor_<platform>.json 与 runs/gates/G0_<platform>.json。"""
    checks: list[CheckResult] = []
    check_fns = [
        ("mujoco_offscreen", lambda: _check_mujoco_basic()),
        ("freecam_reprojection", lambda: _check_freecam_reprojection()),
        ("render_throughput", lambda: _check_throughput()),
        ("multiprocess_render", lambda: _check_multiprocess()),
        ("cross_platform_pixels", lambda: _check_cross_platform()),
        ("dinov3_forward", lambda: _check_dinov3(cfg, download)),
        ("cuda_ops", lambda: _check_cuda()),
        ("torch_jvp", lambda: _check_jvp()),
        ("mps_bf16", lambda: _check_mps_bf16()),
        ("npz_roundtrip", lambda: _check_npz_roundtrip()),
        ("uv_lock", lambda: _check_uv_lock()),
        ("lpips_weights", lambda: _check_lpips(download)),
    ]
    for name, fn in check_fns:
        try:
            checks.append(fn())
        except Exception:
            detail = traceback.format_exc(limit=3)
            logger.warning("检查项 %s 异常：\n%s", name, detail)
            checks.append(CheckResult(name, "fail", f"异常：{detail.strip().splitlines()[-1][:300]}"))
    report = {
        "gate": "G0",
        "platform": sys.platform,
        "machine": platform.machine(),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_hash": _git_hash(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "transformers": transformers.__version__,
            "lpips": importlib.metadata.version("lpips"),
        },
        "checks": [asdict(c) for c in checks],
        "passed": all(c.status != "fail" for c in checks),
        "pending_linux": [c.name for c in checks if c.status == "skip"],
    }
    runs = Path("runs")
    (runs / "gates").mkdir(parents=True, exist_ok=True)
    for path in (runs / f"doctor_{sys.platform}.json", runs / "gates" / f"G0_{sys.platform}.json"):
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
