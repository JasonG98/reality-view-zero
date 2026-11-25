"""设备/精度/随机种子/同步；其他模块不得出现设备字面量（ARCH §5.1、§7）。"""

import contextlib
import random
from contextlib import AbstractContextManager

import numpy as np
import torch


def get_device() -> torch.device:
    """按 cuda > mps > cpu 优先级选择设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device) -> AbstractContextManager[None]:
    """CUDA 用 bf16 autocast；MPS/CPU 用空上下文（fp32，SPEC §8）。"""
    if device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    """计时前后同步设备，避免异步执行影响计时。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
