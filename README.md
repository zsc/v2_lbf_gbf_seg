# Coupled LBF+GBF Level Set Segmentation Demo

实现内容以 `CLAUDE.md` 的 Demo SPEC 为准：FastAPI 后端 + 纯 HTML/JS 前端，支持 LBF / GBF / Coupled 三种水平集分割，并返回迭代帧（overlay / mask / f1,f2 / energy）。

## 运行

```bash
python app.py
```

浏览器打开：

`http://127.0.0.1:8000/`

## 命令行 Smoke Test（默认 test.png）

```bash
python cli_test.py
```

会输出每隔 `viz_stride` 的能量（loss）与分项，并显示实际使用的设备（`auto` 会优先 MPS）。

默认初始化使用 `Otsu`（更适合 `test.png` 这类图；如果想用居中圆/矩形框可传 `--init-type circle|rect`）。

命令行运行后会额外导出 `static/cli_run.json`（可用 `--no-export-json` 关闭）。前端页面可点击 `Load CLI Run` 直接播放这次的帧序列。

## MPS 加速（Apple Silicon）

前端参数里 `device` 选择 `auto` 或 `mps`，后端会把主要卷积/迭代放到 `torch.device("mps")` 上运行（若当前 PyTorch 支持且可用）。

## 安装依赖（可选）

环境里如果缺依赖：

```bash
pip install -r requirements.txt
```
