"""doctor 的 GL 投影纯函数测试：已知相机位姿下的解析像素坐标。"""

import sys

import numpy as np
import pytest

from rvz.doctor import _check_npz_roundtrip, _check_uv_lock, project_gl


def test_project_gl_center_and_offsets() -> None:
    uv, depth = project_gl(
        pos=np.zeros(3),
        forward=np.array([0.0, 0.0, -1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fovy_deg=90.0,
        width=128,
        height=128,
        points=np.array([[0.0, 0.0, -2.0], [0.5, 0.0, -2.0], [0.0, 0.5, -2.0]]),
    )
    np.testing.assert_allclose(depth, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(uv[0], [63.5, 63.5])  # 光轴点 → 画面中心
    np.testing.assert_allclose(uv[1], [79.5, 63.5])  # +x → 右移 16 像素
    np.testing.assert_allclose(uv[2], [63.5, 47.5])  # +y → 上移 16 像素


def test_project_gl_camera_looks_down_x() -> None:
    # 与 doctor 场景固定相机同位姿：球心应投影到画面中心。
    uv, depth = project_gl(
        pos=np.array([2.0, 0.0, 1.0]),
        forward=np.array([-1.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0]),
        fovy_deg=90.0,
        width=128,
        height=128,
        points=np.array([[0.0, 0.0, 1.0]]),
    )
    np.testing.assert_allclose(uv[0], [63.5, 63.5], atol=1e-6)
    assert depth[0] > 0


def test_pending_platform_messages_follow_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """§12.2-10/11 回归：待复核平台必须指向本机之外（Linux 上不能写"待 Linux 运行"）。"""
    for host, other in (("darwin", "Linux"), ("linux", "Mac")):
        monkeypatch.setattr(sys, "platform", host)
        assert f"{other} 平台 sync 待运行" in _check_uv_lock().detail
        assert f"跨机拷贝待 {other} 复核" in _check_npz_roundtrip().detail
