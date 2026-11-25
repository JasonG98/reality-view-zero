"""设备抽象测试：CPU 路径可复现、精度策略正确。"""

import numpy as np
import torch

from rvz.device import autocast_ctx, get_device, seed_everything, sync


def test_get_device_valid_type() -> None:
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}


def test_autocast_fp32_on_cpu() -> None:
    with autocast_ctx(torch.device("cpu")):
        out = torch.mm(torch.ones(2, 2), torch.ones(2, 2))
    assert out.dtype == torch.float32


def test_seed_everything_reproducible() -> None:
    seed_everything(123)
    torch_first, numpy_first = torch.rand(3), np.random.rand(3)
    seed_everything(123)
    torch_second, numpy_second = torch.rand(3), np.random.rand(3)
    assert torch.equal(torch_first, torch_second)
    assert np.array_equal(numpy_first, numpy_second)


def test_sync_cpu_noop() -> None:
    sync(torch.device("cpu"))
