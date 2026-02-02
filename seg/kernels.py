from __future__ import annotations

import math

import torch


def _auto_ksize(sigma: float) -> int:
    radius = int(math.ceil(3.0 * float(sigma)))
    return 2 * radius + 1


def gaussian_kernel2d(
    sigma: float,
    ksize: int | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not (sigma > 0):
        raise ValueError("sigma must be > 0")

    if ksize is None or ksize <= 0:
        ksize = _auto_ksize(sigma)

    if ksize < 3:
        raise ValueError("ksize must be >= 3")
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")

    half = ksize // 2
    xs = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g1 = torch.exp(-(xs * xs) / (2.0 * float(sigma) * float(sigma)))
    g1 = g1 / g1.sum()
    k2 = g1[:, None] * g1[None, :]
    k2 = k2 / k2.sum()
    return k2.view(1, 1, ksize, ksize)

