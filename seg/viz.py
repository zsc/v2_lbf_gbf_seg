from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def _png_base64_from_pil(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _to_u8_gray(img01: np.ndarray) -> np.ndarray:
    arr = np.clip(img01, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    mp = np.pad(m, 1, mode="edge")
    c = (
        (mp[1:-1, 1:-1] != mp[:-2, 1:-1])
        | (mp[1:-1, 1:-1] != mp[2:, 1:-1])
        | (mp[1:-1, 1:-1] != mp[1:-1, :-2])
        | (mp[1:-1, 1:-1] != mp[1:-1, 2:])
    )
    return c


def overlay_base64(gray01: np.ndarray, contour: np.ndarray) -> str:
    g = _to_u8_gray(gray01)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[contour] = np.array([255, 40, 40], dtype=np.uint8)
    return _png_base64_from_pil(Image.fromarray(rgb, mode="RGB"))


def mask_base64(mask: np.ndarray) -> str:
    img = (mask.astype(np.uint8) * 255).astype(np.uint8)
    return _png_base64_from_pil(Image.fromarray(img, mode="L"))


def gray_base64(gray01: np.ndarray, *, normalize: bool = False) -> str:
    x = gray01.astype(np.float32)
    if normalize:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            x = (x - lo) / (hi - lo)
    return _png_base64_from_pil(Image.fromarray(_to_u8_gray(x), mode="L"))

