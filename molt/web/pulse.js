// Pulse dock: a scrolling record of what the model perceives, remembers and learns.
// One column per event, newest on the right:
//   感知  surprise of each token read or written (-log p, nats)
//   记忆  per-layer size of the WKV state change (a "spectrogram" of the working memory)
//   学习  loss of each optimizer step
//   权重  per-layer size of the LoRA update after each step
const COLW = 3, GUTTER = 64;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function startPulse(h) {
  const cv = document.getElementById("pulse"), ctx = cv.getContext("2d");
  const cols = [];
  let dirty = true;
  const push = (c) => {
    cols.push(c);
    const cap = Math.ceil((cv.clientWidth - GUTTER) / COLW) + 4;
    if (cols.length > cap) cols.splice(0, cols.length - cap);
    dirty = true;
  };

  const es = new EventSource("/api/pulse");
  es.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e.k === "tok") push({ kind: "说", s: e.s, d: e.d, t: e.t });
    else if (e.k === "listen") {
      if (e.ss && e.ss.length) {
        const per = e.d ? e.d.map((x) => x / e.ss.length) : null;   // spread the chunk's state change over its tokens
        e.ss.forEach((s) => push({ kind: "听", s, d: per }));
      } else push({ kind: "听", s: null, d: e.d, n: e.n });
    } else if (e.k === "step") push({ kind: "学", loss: e.loss, kl: e.kl, g: e.gnorm, lora: e.lora });
    else if (e.k === "activity") h.onActivity(e.state, e.label);
    else if (e.k === "progress") h.onProgress(e);
    else if (e.k === "hb") h.onHeartbeat(e);
    else if (e.k === "model") h.onModel(e);
  };

  const strips = [
    { key: "s", name: "感知", sub: "意外度", frac: 0.24 },
    { key: "d", name: "记忆", sub: "各层 Δstate", frac: 0.32 },
    { key: "loss", name: "学习", sub: "损失", frac: 0.18 },
    { key: "lora", name: "权重", sub: "各层 ΔLoRA", frac: 0.26 },
  ];
  let layout = [];

  function draw() {
    if (!dirty || document.body.classList.contains("dock-min")) return;
    dirty = false;
    const dpr = devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const s1 = css("--series-1"), s2 = css("--series-2"), heat = css("--heat"), muted = css("--muted"), grid = css("--grid");
    const pad = 6, avail = H - pad * 2 - 3 * 4;
    let y = pad;
    layout = strips.map((st) => { const hh = Math.round(avail * st.frac); const r = { ...st, y, h: hh }; y += hh + 4; return r; });
    const x0 = W - cols.length * COLW;
    const vis = cols;
    const maxD = Math.max(1e-9, ...vis.flatMap((c) => c.d || []));
    const maxL = Math.max(1e-9, ...vis.flatMap((c) => c.lora || []));
    const maxLoss = Math.max(1, ...vis.map((c) => c.loss ?? 0));
    ctx.font = "11px system-ui, 'Microsoft YaHei', sans-serif";
    for (const r of layout) {
      ctx.fillStyle = grid; ctx.fillRect(GUTTER, r.y + r.h, W - GUTTER, 1);
      ctx.fillStyle = muted; ctx.fillText(r.name, 8, r.y + 12); ctx.fillText(r.sub, 8, r.y + 25);
    }
    const [rs, rd, rl, rw] = layout;
    vis.forEach((c, i) => {
      const x = x0 + i * COLW;
      if (x < GUTTER) return;
      if (c.s != null) {                                   // surprise: bars from the baseline, capped at 8 nats
        const hh = Math.min(c.s, 8) / 8 * rs.h;
        ctx.fillStyle = s1; ctx.globalAlpha = 0.35 + 0.65 * Math.min(c.s, 8) / 8;
        ctx.fillRect(x, rs.y + rs.h - hh, COLW - 1, hh);
      }
      ctx.globalAlpha = 1;
      if (c.d) paintLayers(c.d, maxD, rd, x, heat);
      if (c.loss != null) {
        const hh = Math.min(c.loss, maxLoss) / maxLoss * rl.h;
        ctx.fillStyle = s2; ctx.fillRect(x, rl.y + rl.h - hh, COLW - 1, Math.max(hh, 1));
      }
      if (c.lora) paintLayers(c.lora, maxL, rw, x, heat);
    });
    ctx.globalAlpha = 1;
    if (!cols.length) { ctx.fillStyle = muted; ctx.fillText("等待信号：对话、学习或运行实验时，这里会实时滚动", GUTTER + 12, H / 2); }
  }
  function paintLayers(vals, max, r, x, color) {
    const n = vals.length, rh = r.h / n;
    ctx.fillStyle = color;
    vals.forEach((v, l) => {
      const t = Math.log1p(v) / Math.log1p(max);
      if (t <= 0.02) return;
      ctx.globalAlpha = 0.08 + 0.92 * t;
      ctx.fillRect(x, r.y + l * rh, COLW - 1, Math.max(rh - 0.5, 0.5));
    });
    ctx.globalAlpha = 1;
  }
  (function loop() { draw(); requestAnimationFrame(loop); })();
  new ResizeObserver(() => (dirty = true)).observe(cv);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => (dirty = true));
  new MutationObserver(() => (dirty = true)).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  cv.addEventListener("mousemove", (e) => {
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const i = Math.floor((mx - (r.width - cols.length * COLW)) / COLW);
    const c = cols[i];
    if (!c || mx < GUTTER) return h.hideTip();
    const strip = layout.find((s) => my >= s.y && my <= s.y + s.h);
    const top = (arr) => arr ? arr.map((v, l) => [l, v]).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([l, v]) => `第 ${l} 层 ${v.toFixed(3)}`).join("<br>") : "";
    let html = `<b>${c.kind}</b>`;
    if (c.t != null) html += ` “${String(c.t).replace(/\n/g, "↵")}”`;
    if (c.s != null) html += `<br>意外度 ${c.s.toFixed(2)}（p ≈ ${(Math.exp(-c.s) * 100).toFixed(1)}%）`;
    if (c.loss != null) html += `<br>损失 ${c.loss.toFixed(3)}${c.kl != null ? ` · 锚定 KL ${c.kl.toFixed(3)}` : ""} · 梯度范数 ${c.g?.toFixed(2)}`;
    if (strip?.key === "d" && c.d) html += `<div class="k">state 变化最大的层</div>${top(c.d)}`;
    if (strip?.key === "lora" && c.lora) html += `<div class="k">LoRA 更新最大的层</div>${top(c.lora)}`;
    h.showTip(e, html);
  });
  cv.addEventListener("mouseleave", () => h.hideTip());
  return { redraw: () => (dirty = true) };
}

// ---------- jobs with progress and ETA ----------
const jobs = new Map();   // job -> {phase, done, total, t0, last}
const renderers = new Set();
export function onJobs(fn) { renderers.add(fn); fn(jobs); return () => renderers.delete(fn); }
export function trackProgress(e) {
  const prev = jobs.get(e.job);
  const t0 = prev && prev.phase === e.phase && e.done >= prev.done ? prev.t0 : performance.now();
  jobs.set(e.job, { phase: e.phase, done: e.done, total: e.total, t0, last: performance.now() });
  renderers.forEach((fn) => fn(jobs));
}
export function jobsView(filter = () => true) {
  const out = [];
  const now = performance.now();
  for (const [name, j] of jobs) {
    if (!filter(name) || now - j.last > 120000) continue;
    const frac = j.total ? Math.min(1, j.done / j.total) : 0;
    const el = (now - j.t0) / 1000;
    const eta = j.done > 0 && j.done < j.total ? (el / j.done) * (j.total - j.done) : null;
    const d = document.createElement("div");
    d.className = "job" + (j.done >= j.total ? " done" : "");
    d.innerHTML = `<div class="meta"><span>${j.phase}</span><span>${j.done}/${j.total}${eta != null ? ` · 约剩 ${fmtSec(eta)}` : j.done >= j.total ? " · 完成" : ""}</span></div><div class="bar2"><i style="width:${(frac * 100).toFixed(1)}%"></i></div>`;
    out.push(d);
  }
  return out;
}
const fmtSec = (s) => (s < 60 ? `${Math.round(s)} 秒` : `${Math.floor(s / 60)} 分 ${Math.round(s % 60)} 秒`);
