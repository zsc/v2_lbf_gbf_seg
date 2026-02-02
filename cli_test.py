import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from seg.levelset import LevelSetParams, run_levelset


def load_gray01(path: Path, *, max_side: int = 512) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")

    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def maybe_save_frames(frames: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fr in frames:
        it = fr.get("iter", 0)
        for key, suffix in [
            ("overlay_png_base64", "overlay"),
            ("mask_png_base64", "mask"),
            ("f1_png_base64", "f1"),
            ("f2_png_base64", "f2"),
        ]:
            b64 = fr.get(key)
            if not b64:
                continue
            data = base64.b64decode(b64)
            (out_dir / f"iter_{it:04d}_{suffix}.png").write_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="CLI smoke test for level set demo (defaults to test.png).")
    ap.add_argument("--image", type=Path, default=Path("test.png"))
    ap.add_argument("--model", choices=["lbf", "gbf", "coupled"], default="coupled")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--ksize", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=1.5)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--viz-stride", type=int, default=20)
    ap.add_argument("--init-type", choices=["otsu", "circle", "rect"], default="otsu")
    ap.add_argument("--radius-frac", type=float, default=0.25)
    ap.add_argument("--margin-frac", type=float, default=0.20)
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--export-json", type=Path, default=Path("static/cli_run.json"))
    ap.add_argument("--no-export-json", action="store_true", help="Disable exporting /static/cli_run.json")
    args = ap.parse_args()

    img = load_gray01(args.image)
    if args.init_type == "circle":
        init_params = {"radius_frac": args.radius_frac}
    elif args.init_type == "rect":
        init_params = {"margin_frac": args.margin_frac}
    else:
        init_params = {}
    p = LevelSetParams(
        model=args.model,
        device=args.device,
        sigma=args.sigma,
        ksize=args.ksize,
        epsilon=args.epsilon,
        nu=args.nu,
        mu=args.mu,
        omega=args.omega,
        dt=args.dt,
        iters=args.iters,
        viz_stride=args.viz_stride,
        init_type=args.init_type,
        init_params=init_params,
    )

    print(f"image={args.image} shape={img.shape} model={p.model} device={p.resolve_device().type}")
    t0 = time.perf_counter()
    frames = run_levelset(img, p)
    t1 = time.perf_counter()

    e0 = frames[0]["stats"]["energy"]
    emin = min(fr["stats"]["energy"] for fr in frames)
    eend = frames[-1]["stats"]["energy"]
    print("iter    energy        lbf_data     gbf_data     len        dist")
    for fr in frames:
        s = fr["stats"]
        print(
            f"{fr['iter']:4d}  {s['energy']:10.2f}  {s.get('energy_lbf_data') or 0:10.2f}  {s.get('energy_gbf_data') or 0:10.2f}  {s.get('energy_len') or 0:8.2f}  {s.get('energy_dist') or 0:8.2f}"
        )
    print(
        f"E0={e0:.2f}  Emin={emin:.2f}  Eend={eend:.2f}  d_end={eend - e0:.2f}  rel={((eend - e0) / e0) * 100:.3f}%"
    )
    if args.save_dir is not None:
        maybe_save_frames(frames, args.save_dir)
        print(f"saved frames -> {args.save_dir}")

    if not args.no_export_json and args.export_json is not None:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frames": frames,
            "final": frames[-1] if frames else None,
            "timing_ms": int(round((t1 - t0) * 1000)),
            "device": p.resolve_device().type,
            "params": {
                "model": p.model,
                "device": p.device,
                "sigma": p.sigma,
                "ksize": p.ksize,
                "epsilon": p.epsilon,
                "nu": p.nu,
                "mu": p.mu,
                "omega": p.omega,
                "dt": p.dt,
                "iters": p.iters,
                "viz_stride": p.viz_stride,
                "init_type": p.init_type,
                "init_params": p.init_params,
            },
        }
        args.export_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"exported json -> {args.export_json}")


if __name__ == "__main__":
    main()
