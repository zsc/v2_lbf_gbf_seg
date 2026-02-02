from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .kernels import gaussian_kernel2d
from .viz import contour_from_mask, gray_base64, mask_base64, overlay_base64


@dataclass(frozen=True)
class LevelSetParams:
    model: str = "coupled"  # "lbf" | "gbf" | "coupled"
    sigma: float = 3.0
    ksize: int = 0
    epsilon: float = 1.5
    nu: float = 0.3
    mu: float = 0.3
    omega: float = 1.0
    dt: float = 0.1
    iters: int = 120
    viz_stride: int = 5
    init_type: str = "otsu"  # "otsu" | "circle" | "rect"
    init_params: dict[str, Any] | None = None
    mode: str = "alt"  # "alt" | "e2e" (optional)
    eps0: float = 1e-6
    eta: float = 1e-8
    padding_mode: str = "reflect"
    device: str = "auto"  # "auto" | "cpu" | "mps"

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "LevelSetParams":
        if not isinstance(payload, dict):
            raise ValueError("params must be a JSON object")

        init_params = payload.get("init_params", None)
        if init_params is None:
            init_params_dict: dict[str, Any] | None = None
        elif isinstance(init_params, dict):
            init_params_dict = init_params
        else:
            raise ValueError("init_params must be an object")

        return cls(
            model=str(payload.get("model", cls.model)),
            sigma=float(payload.get("sigma", cls.sigma)),
            ksize=int(payload.get("ksize", cls.ksize)),
            epsilon=float(payload.get("epsilon", cls.epsilon)),
            nu=float(payload.get("nu", cls.nu)),
            mu=float(payload.get("mu", cls.mu)),
            omega=float(payload.get("omega", cls.omega)),
            dt=float(payload.get("dt", cls.dt)),
            iters=int(payload.get("iters", cls.iters)),
            viz_stride=int(payload.get("viz_stride", cls.viz_stride)),
            init_type=str(payload.get("init_type", cls.init_type)),
            init_params=init_params_dict,
            mode=str(payload.get("mode", cls.mode)),
            device=str(payload.get("device", cls.device)),
        )

    def validate(self) -> None:
        if self.model not in {"lbf", "gbf", "coupled"}:
            raise ValueError('model must be one of: "lbf", "gbf", "coupled"')
        if self.mode not in {"alt", "e2e"}:
            raise ValueError('mode must be one of: "alt", "e2e"')
        if self.mode == "e2e":
            raise ValueError('mode="e2e" is not implemented in this demo')
        if not (self.sigma > 0):
            raise ValueError("sigma must be > 0")
        if self.ksize != 0:
            if self.ksize < 3 or self.ksize % 2 == 0:
                raise ValueError("ksize must be 0 (auto) or an odd integer >= 3")
        if not (self.epsilon > 0):
            raise ValueError("epsilon must be > 0")
        if not (self.dt > 0):
            raise ValueError("dt must be > 0")
        if self.iters < 1:
            raise ValueError("iters must be >= 1")
        if self.viz_stride < 1:
            raise ValueError("viz_stride must be >= 1")
        if self.init_type not in {"otsu", "circle", "rect"}:
            raise ValueError('init_type must be one of: "otsu", "circle", "rect"')
        if self.padding_mode not in {"reflect", "replicate"}:
            raise ValueError('padding_mode must be "reflect" or "replicate"')
        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError('device must be one of: "auto", "cpu", "mps"')
        if self.device == "mps" and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise ValueError('device="mps" requested but MPS is not available in this PyTorch build')

    def resolve_device(self) -> torch.device:
        if self.device == "cpu":
            return torch.device("cpu")
        if self.device == "mps":
            return torch.device("mps")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")


def _heaviside(phi: torch.Tensor, eps: float) -> torch.Tensor:
    return 0.5 * (1.0 + (2.0 / math.pi) * torch.atan(phi / float(eps)))


def _dirac(phi: torch.Tensor, eps: float) -> torch.Tensor:
    return (1.0 / math.pi) * (float(eps) / (float(eps) * float(eps) + phi * phi))


def _conv2d_same(x: torch.Tensor, k: torch.Tensor, *, padding_mode: str) -> torch.Tensor:
    pad = k.shape[-1] // 2
    xpad = F.pad(x, (pad, pad, pad, pad), mode=padding_mode)
    return F.conv2d(xpad, k)


def _gradient(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = F.pad(phi, (1, 1, 1, 1), mode="replicate")
    dx = 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
    dy = 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
    return dx, dy


def _divergence(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    pxp = F.pad(px, (1, 1, 1, 1), mode="replicate")
    pyp = F.pad(py, (1, 1, 1, 1), mode="replicate")
    ddx = 0.5 * (pxp[..., 1:-1, 2:] - pxp[..., 1:-1, :-2])
    ddy = 0.5 * (pyp[..., 2:, 1:-1] - pyp[..., :-2, 1:-1])
    return ddx + ddy


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("scipy is required for signed distance initialization") from exc

    mask_bool = mask.astype(bool)
    dt_in = ndimage.distance_transform_edt(mask_bool)
    dt_out = ndimage.distance_transform_edt(~mask_bool)
    sdf = dt_in - dt_out
    return sdf.astype(np.float32)


def _otsu_threshold_u8(img_u8: np.ndarray) -> int:
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return int(np.argmax(sigma_b2))


def _init_mask(image01: np.ndarray, init_type: str, init_params: dict[str, Any] | None) -> np.ndarray:
    h, w = int(image01.shape[0]), int(image01.shape[1])
    p = init_params or {}
    if init_type == "otsu":
        img_u8 = np.clip(image01 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        thr = _otsu_threshold_u8(img_u8)
        hi = img_u8 > thr
        lo = ~hi
        if int(hi.sum()) == 0 or int(hi.sum()) == h * w:
            r_frac = float(p.get("radius_frac", 0.25))
            r_frac = max(0.01, min(0.49, r_frac))
            r = r_frac * min(h, w)
            yy, xx = np.ogrid[:h, :w]
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            return (dist2 <= r * r).astype(np.uint8)

        fg = str(p.get("foreground", "auto")).lower()
        if fg == "bright":
            mask = hi
        elif fg == "dark":
            mask = lo
        else:
            mask = hi if int(hi.sum()) <= int(lo.sum()) else lo

        if int(mask.sum()) == 0 or int(mask.sum()) == h * w:
            r_frac = float(p.get("radius_frac", 0.25))
            r_frac = max(0.01, min(0.49, r_frac))
            r = r_frac * min(h, w)
            yy, xx = np.ogrid[:h, :w]
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            return (dist2 <= r * r).astype(np.uint8)
        return mask.astype(np.uint8)
    if init_type == "circle":
        r_frac = float(p.get("radius_frac", 0.25))
        r_frac = max(0.01, min(0.49, r_frac))
        r = r_frac * min(h, w)
        yy, xx = np.ogrid[:h, :w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        return (dist2 <= r * r).astype(np.uint8)
    if init_type == "rect":
        m_frac = float(p.get("margin_frac", 0.2))
        m_frac = max(0.0, min(0.49, m_frac))
        my = int(round(m_frac * h))
        mx = int(round(m_frac * w))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[my : h - my, mx : w - mx] = 1
        return mask
    raise ValueError(f"Unsupported init_type: {init_type}")


def run_levelset(image01: np.ndarray, params: LevelSetParams) -> list[dict[str, Any]]:
    params.validate()

    if image01.ndim != 2:
        raise ValueError("image01 must be a 2D grayscale array")

    img = np.clip(image01.astype(np.float32), 0.0, 1.0)
    h, w = img.shape

    init_mask = _init_mask(img, params.init_type, params.init_params)
    phi0 = _signed_distance(init_mask)

    device = params.resolve_device()
    I = torch.from_numpy(img).to(device=device, dtype=torch.float32).view(1, 1, h, w)
    phi = torch.from_numpy(phi0).to(device=device, dtype=torch.float32).view(1, 1, h, w)

    k = gaussian_kernel2d(params.sigma, params.ksize, device=device, dtype=torch.float32)

    frames: list[dict[str, Any]] = []

    def t2np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()[0, 0]

    def capture(
        iter_idx: int,
        *,
        phi_t: torch.Tensor,
        stats: dict[str, Any],
        f1_t: torch.Tensor | None = None,
        f2_t: torch.Tensor | None = None,
    ) -> None:
        phi_np = t2np(phi_t)
        mask = phi_np > 0
        contour = contour_from_mask(mask)
        frame = {
            "iter": int(iter_idx),
            "overlay_png_base64": overlay_base64(img, contour),
            "mask_png_base64": mask_base64(mask),
            "f1_png_base64": gray_base64(t2np(f1_t), normalize=True) if f1_t is not None else None,
            "f2_png_base64": gray_base64(t2np(f2_t), normalize=True) if f2_t is not None else None,
            "stats": stats,
        }
        frames.append(frame)

    with torch.no_grad():
        want_lbf = params.model in {"lbf", "coupled"}
        want_gbf = params.model in {"gbf", "coupled"}
        gbf_weight = float(params.omega) if params.model == "coupled" else 1.0

        for it in range(0, params.iters + 1):
            H = _heaviside(phi, params.epsilon)
            delta = _dirac(phi, params.epsilon)

            f1 = f2 = None
            e1 = e2 = None

            if want_lbf:
                IH = I * H
                I1H = I * (1.0 - H)
                KH = _conv2d_same(H, k, padding_mode=params.padding_mode)
                K1H = _conv2d_same(1.0 - H, k, padding_mode=params.padding_mode)
                f1 = _conv2d_same(IH, k, padding_mode=params.padding_mode) / (KH + params.eps0)
                f2 = _conv2d_same(I1H, k, padding_mode=params.padding_mode) / (K1H + params.eps0)

                Kf1 = _conv2d_same(f1, k, padding_mode=params.padding_mode)
                Kf1_2 = _conv2d_same(f1 * f1, k, padding_mode=params.padding_mode)
                Kf2 = _conv2d_same(f2, k, padding_mode=params.padding_mode)
                Kf2_2 = _conv2d_same(f2 * f2, k, padding_mode=params.padding_mode)

                I2 = I * I
                e1 = I2 - 2.0 * I * Kf1 + Kf1_2
                e2 = I2 - 2.0 * I * Kf2 + Kf2_2

            c1 = c2 = None
            gbf_term = None

            if want_gbf:
                sumH = H.sum()
                sum1H = (1.0 - H).sum()
                c1 = float(((I * H).sum() / (sumH + params.eps0)).item())
                c2 = float(((I * (1.0 - H)).sum() / (sum1H + params.eps0)).item())
                gbf_term = (I - c1) ** 2 - (I - c2) ** 2

            data_force = torch.zeros_like(phi)
            if want_lbf and e1 is not None and e2 is not None:
                data_force = data_force + (e1 - e2)
            if want_gbf and gbf_term is not None:
                data_force = data_force + gbf_weight * gbf_term

            phi_x, phi_y = _gradient(phi)
            s = torch.sqrt(phi_x * phi_x + phi_y * phi_y + params.eta)

            nx = phi_x / (s + params.eta)
            ny = phi_y / (s + params.eta)
            kappa = _divergence(nx, ny)

            q = (s - 1.0) / (s + params.eta)
            rx = q * phi_x
            ry = q * phi_y
            rphi = _divergence(rx, ry)

            if it % params.viz_stride == 0 or it == params.iters:
                energy = 0.0
                e_lbf = None
                e_gbf = None
                e_len = None
                e_dist = None
                if want_lbf and e1 is not None and e2 is not None:
                    e_lbf_t = (e1 * H + e2 * (1.0 - H)).sum()
                    e_lbf = float(e_lbf_t.item())
                    energy += e_lbf
                if want_gbf and c1 is not None and c2 is not None:
                    eg_t = (((I - c1) ** 2) * H + ((I - c2) ** 2) * (1.0 - H)).sum()
                    e_gbf = float((gbf_weight * eg_t).item())
                    energy += e_gbf

                e_len = float((float(params.nu) * (delta * s).sum()).item())
                energy += e_len

                e_dist = float(((float(params.mu) / 2.0) * ((s - 1.0) ** 2).sum()).item())
                energy += e_dist

                capture(
                    it,
                    phi_t=phi,
                    f1_t=f1 if want_lbf else None,
                    f2_t=f2 if want_lbf else None,
                    stats={
                        "c1": c1,
                        "c2": c2,
                        "energy": energy,
                        "energy_lbf_data": e_lbf,
                        "energy_gbf_data": e_gbf,
                        "energy_len": e_len,
                        "energy_dist": e_dist,
                    },
                )

            if it == params.iters:
                break

            rhs = -delta * data_force + float(params.nu) * delta * kappa + float(params.mu) * rphi
            phi = phi + float(params.dt) * rhs

            if not torch.isfinite(phi).all():
                raise RuntimeError("phi became NaN/Inf (try smaller dt or larger epsilon)")

    return frames
