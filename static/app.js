const $ = (id) => document.getElementById(id);

const els = {
  file: $("file"),
  model: $("model"),
  initType: $("initType"),
  radiusFrac: $("radiusFrac"),
  rectMarginFrac: $("rectMarginFrac"),
  sigma: $("sigma"),
  ksize: $("ksize"),
  epsilon: $("epsilon"),
  nu: $("nu"),
  mu: $("mu"),
  omega: $("omega"),
  device: $("device"),
  dt: $("dt"),
  iters: $("iters"),
  vizStride: $("vizStride"),
  runBtn: $("runBtn"),
  loadCliBtn: $("loadCliBtn"),
  status: $("status"),
  frameSlider: $("frameSlider"),
  playBtn: $("playBtn"),
  pauseBtn: $("pauseBtn"),
  iterLabel: $("iterLabel"),
  c1Label: $("c1Label"),
  c2Label: $("c2Label"),
  energyLabel: $("energyLabel"),
  overlayCanvas: $("overlayCanvas"),
  maskCanvas: $("maskCanvas"),
  f1Img: $("f1Img"),
  f2Img: $("f2Img"),
  energyCanvas: $("energyCanvas"),
};

let frames = [];
let playTimer = null;
let defaultTestBlob = null;

function setStatus(msg) {
  els.status.textContent = msg || "";
}

async function getImageFileOrDefault() {
  const file = els.file.files?.[0];
  if (file) return file;

  if (!defaultTestBlob) {
    try {
      const resp = await fetch("/static/test.png", { cache: "no-store" });
      if (resp.ok) defaultTestBlob = await resp.blob();
    } catch (_) {
      // ignore
    }
  }

  if (!defaultTestBlob) return null;
  return new File([defaultTestBlob], "test.png", { type: defaultTestBlob.type || "image/png" });
}

function syncUI() {
  const initType = els.initType.value;
  els.radiusFrac.disabled = initType !== "circle";
  els.rectMarginFrac.disabled = initType !== "rect";

  const model = els.model.value;
  els.omega.disabled = model !== "coupled";
}

function clampNumber(value, fallback) {
  const v = Number(value);
  return Number.isFinite(v) ? v : fallback;
}

function buildParams() {
  const initType = els.initType.value;
  const initParams = {};
  if (initType === "circle") {
    initParams.radius_frac = clampNumber(els.radiusFrac.value, 0.25);
  } else if (initType === "rect") {
    initParams.margin_frac = clampNumber(els.rectMarginFrac.value, 0.2);
  }

  return {
    model: els.model.value,
    device: els.device.value,
    sigma: clampNumber(els.sigma.value, 3.0),
    ksize: Math.floor(clampNumber(els.ksize.value, 0)),
    epsilon: clampNumber(els.epsilon.value, 1.5),
    nu: clampNumber(els.nu.value, 0.3),
    mu: clampNumber(els.mu.value, 0.3),
    omega: clampNumber(els.omega.value, 1.0),
    dt: clampNumber(els.dt.value, 0.1),
    iters: Math.floor(clampNumber(els.iters.value, 120)),
    viz_stride: Math.floor(clampNumber(els.vizStride.value, 5)),
    init_type: initType,
    init_params: initParams,
    mode: "alt",
  };
}

function decodeAndDrawToCanvas(canvas, base64Png) {
  if (!base64Png) return;
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
  };
  img.src = "data:image/png;base64," + base64Png;
}

function drawEnergyPlot(canvas, xs) {
  const ctx = canvas.getContext("2d");
  const w = Math.max(200, canvas.clientWidth || 0);
  const h = canvas.height || 140;
  canvas.width = w;
  canvas.height = h;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(0,0,0,0.2)";
  ctx.fillRect(0, 0, w, h);

  if (!xs || xs.length < 2) return;

  const finite = xs.filter((v) => Number.isFinite(v));
  if (finite.length < 2) return;

  const minV = Math.min(...finite);
  const maxV = Math.max(...finite);
  const denom = maxV - minV || 1.0;

  ctx.strokeStyle = "rgba(110,168,254,0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const v = xs[i];
    if (!Number.isFinite(v)) continue;
    const x = (i / (xs.length - 1)) * (w - 16) + 8;
    const y = h - 10 - ((v - minV) / denom) * (h - 20);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(231,236,255,0.75)";
  ctx.font = "12px ui-sans-serif, system-ui";
  ctx.fillText(`min ${minV.toFixed(4)} / max ${maxV.toFixed(4)}`, 10, 16);
}

function showFrame(i) {
  if (!frames.length) return;
  const idx = Math.max(0, Math.min(i, frames.length - 1));
  const fr = frames[idx];

  els.frameSlider.value = String(idx);
  els.iterLabel.textContent = fr.iter ?? "-";
  els.c1Label.textContent = fr.stats?.c1 != null ? fr.stats.c1.toFixed(4) : "-";
  els.c2Label.textContent = fr.stats?.c2 != null ? fr.stats.c2.toFixed(4) : "-";
  els.energyLabel.textContent = fr.stats?.energy != null ? fr.stats.energy.toFixed(6) : "-";

  decodeAndDrawToCanvas(els.overlayCanvas, fr.overlay_png_base64);
  decodeAndDrawToCanvas(els.maskCanvas, fr.mask_png_base64);

  if (fr.f1_png_base64) els.f1Img.src = "data:image/png;base64," + fr.f1_png_base64;
  else els.f1Img.removeAttribute("src");
  if (fr.f2_png_base64) els.f2Img.src = "data:image/png;base64," + fr.f2_png_base64;
  else els.f2Img.removeAttribute("src");

  const energies = frames.map((f) => f.stats?.energy);
  drawEnergyPlot(els.energyCanvas, energies);
}

function setFramesFromPayload(payload, sourceLabel) {
  const next = payload?.frames || [];
  if (!Array.isArray(next) || !next.length) throw new Error("Empty frames");
  frames = next;
  els.frameSlider.max = String(frames.length - 1);
  const meta = payload?.device ? ` device=${payload.device}` : "";
  setStatus(`Loaded ${sourceLabel}. ${frames.length} frames.${meta}`);
  showFrame(0);
}

async function loadCliRun(silent = false) {
  try {
    const resp = await fetch("/static/cli_run.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const payload = await resp.json();
    setFramesFromPayload(payload, "CLI run");
  } catch (e) {
    if (!silent) setStatus(`Load CLI run failed: ${e.message || e}`);
  }
}

function stopPlayback() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
}

function startPlayback() {
  stopPlayback();
  if (!frames.length) return;
  playTimer = setInterval(() => {
    const next = Number(els.frameSlider.value) + 1;
    if (next >= frames.length) {
      stopPlayback();
      return;
    }
    showFrame(next);
  }, 150);
}

els.playBtn.addEventListener("click", () => startPlayback());
els.pauseBtn.addEventListener("click", () => stopPlayback());
els.frameSlider.addEventListener("input", (e) => showFrame(Number(e.target.value)));
els.initType.addEventListener("change", () => syncUI());
els.model.addEventListener("change", () => syncUI());
els.loadCliBtn.addEventListener("click", () => loadCliRun(false));
syncUI();
loadCliRun(true);

els.runBtn.addEventListener("click", async () => {
  stopPlayback();
  const file = await getImageFileOrDefault();
  if (!file) {
    setStatus("请先选择一张图片（或确保 /static/test.png 存在）");
    return;
  }

  const params = buildParams();
  const form = new FormData();
  form.append("image", file);
  form.append("params", JSON.stringify(params));

  els.runBtn.disabled = true;
  setStatus("Running...");

  try {
    const resp = await fetch("/api/run", { method: "POST", body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    setFramesFromPayload(data, `API run (${data.timing_ms} ms)`);
  } catch (e) {
    setStatus(`Error: ${e.message || e}`);
  } finally {
    els.runBtn.disabled = false;
  }
});
