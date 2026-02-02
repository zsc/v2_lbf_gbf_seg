import json
import time
from io import BytesIO
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from seg.levelset import LevelSetParams, run_levelset


app = FastAPI(title="Coupled LBF+GBF Level Set Segmentation Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


def _read_image_to_gray01(image_bytes: bytes, max_side: int = 512) -> np.ndarray:
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to open image: {exc}") from exc

    img = ImageOps.exif_transpose(img)
    img = img.convert("L")

    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _parse_params(params_json: str) -> LevelSetParams:
    try:
        payload = json.loads(params_json)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid params JSON: {exc}") from exc

    try:
        return LevelSetParams.from_json(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/run")
async def api_run(image: UploadFile = File(...), params: str = Form(...)) -> dict[str, Any]:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    img01 = _read_image_to_gray01(image_bytes)
    p = _parse_params(params)

    t0 = time.perf_counter()
    try:
        frames = run_levelset(img01, p)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc
    t1 = time.perf_counter()

    return {
        "frames": frames,
        "final": frames[-1] if frames else None,
        "timing_ms": int(round((t1 - t0) * 1000)),
        "device": p.resolve_device().type,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
