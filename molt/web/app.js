// Molt web UI — no framework, no build step. Talks to molt.server over JSON + Server-Sent Events.
const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k);
  return n;
};
const SVG = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}) => { const n = document.createElementNS(SVG, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; };
const fmt = (x, d = 3) => (x == null ? "—" : Number(x).toFixed(d));

// ---------- API ----------
const api = {
  get: (u) => fetch(u).then((r) => r.json()),
  post: (u, b) => fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then((r) => r.json()),
  async stream(u, b, onEvent) {
    const r = await fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) });
    if (!r.ok) { onEvent({ error: `${r.status} ${await r.text()}` }); return; }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 2);
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
      }
    }
  },
};

// ---------- tooltip ----------
const tip = $("#tooltip");
function showTip(evt, html) {
  tip.innerHTML = html; tip.hidden = false;
  const pad = 12, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > innerWidth - 4) x = evt.clientX - w - pad;
  if (y + h > innerHeight - 4) y = evt.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => (tip.hidden = true);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]).replace(/\n/g, "↵");

// ---------- color encodings ----------
// Surprise (−log p, nats): one hue, light → dark. Alpha over the surface keeps text in primary ink in both themes.
const SURPRISE_BINS = [[0.5, 0, "很确定"], [1.5, 18, ""], [3, 34, ""], [5, 52, ""], [Infinity, 70, "很意外"]];
const surpriseBg = (loss) => { const b = SURPRISE_BINS.find(([hi]) => loss < hi); return b[1] ? `color-mix(in srgb, var(--seq-400) ${b[1]}%, transparent)` : "transparent"; };
// Δloss after − before: diverging blue (more familiar) ↔ gray ↔ red (less familiar).
const DELTA_BINS = [[-2, "var(--div-neg)", 60, "明显变熟"], [-0.5, "var(--div-neg)", 28, "变熟"], [0.5, null, 0, "几乎不变"], [2, "var(--div-pos)", 28, "变生"], [Infinity, "var(--div-pos)", 60, "明显变生"]];
const deltaBg = (d) => { const b = DELTA_BINS.find(([hi]) => d < hi); return b[1] ? `color-mix(in srgb, ${b[1]} ${b[2]}%, transparent)` : "transparent"; };

function legend(node, items) {
  node.replaceChildren(...items.map(([bg, label]) => el("span", {}, Object.assign(el("span", { class: "sw" }), { style: `background:${bg}` }), label)));
}
legend($("#tint-legend"), [[surpriseBg(0.1), "很确定 (p>0.6)"], [surpriseBg(1), ""], [surpriseBg(2), ""], [surpriseBg(4), ""], [surpriseBg(9), "很意外 (p<0.7%)"]]
  .map(([b, l], i) => [b, l || ["", "p 0.2–0.6", "p 0.05–0.2", "p 0.007–0.05"][i]]));

function tokenSpan(t, bg) {
  const s = el("span", { class: "tok" }, t.text);
  s.style.background = bg;
  s.addEventListener("mousemove", (e) => {
    const alts = (t.top || []).map(([w, p]) => `${esc(w)} <span class="k">${(p * 100).toFixed(1)}%</span>`).join("<br>");
    const extra = t.delta != null ? `<div>Δloss ${t.delta > 0 ? "+" : ""}${t.delta.toFixed(2)}（${t.before.toFixed(2)} → ${t.loss.toFixed(2)}）</div>` : "";
    showTip(e, `<div>“${esc(t.text)}” p=${t.p == null ? "—" : (t.p * 100).toFixed(1) + "%"}</div>${extra}<div class="k">模型的前几名候选</div>${alts}`);
  });
  s.addEventListener("mouseleave", hideTip);
  return s;
}

// ---------- line chart (one series, crosshair + tooltip) ----------
function lineChart(node, pts, { color = "var(--series-1)", name = "" } = {}) {
  node.replaceChildren();
  if (!pts.length) { node.append(el("div", { class: "empty" }, "还没有数据")); return; }
  const W = 420, H = 130, L = 34, R = 8, T = 8, B = 18;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1), y1 = Math.max(...ys) * 1.08 || 1, y0 = 0;
  const sx = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R), sy = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": name });
  for (let k = 0; k <= 3; k++) {
    const v = y0 + ((y1 - y0) * k) / 3, y = sy(v);
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: y, y2: y, class: k ? "gridline" : "baseline" }));
    const t = svgEl("text", { x: L - 4, y: y + 3, "text-anchor": "end", class: "axis-label" }); t.textContent = v.toFixed(v < 1 ? 2 : 1); svg.append(t);
  }
  const xl = svgEl("text", { x: W - R, y: H - 4, "text-anchor": "end", class: "axis-label" }); xl.textContent = `第 ${x1} 段`; svg.append(xl);
  svg.append(svgEl("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(""), fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round" }));
  const last = pts[pts.length - 1];
  svg.append(svgEl("circle", { cx: sx(last.x), cy: sy(last.y), r: 4, fill: color, stroke: "var(--surface)", "stroke-width": 2 }));
  const cross = svgEl("line", { y1: T, y2: H - B, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  const dot = svgEl("circle", { r: 4, fill: color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" });
  svg.append(cross, dot);
  const hit = svgEl("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" });
  hit.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect(), mx = ((e.clientX - r.left) / r.width) * W;
    const p = pts.reduce((a, b) => (Math.abs(sx(b.x) - mx) < Math.abs(sx(a.x) - mx) ? b : a));
    cross.setAttribute("x1", sx(p.x)); cross.setAttribute("x2", sx(p.x)); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", sx(p.x)); dot.setAttribute("cy", sy(p.y)); dot.setAttribute("visibility", "visible");
    showTip(e, `第 ${p.x} 段<br>${name} <b>${fmt(p.y)}</b>${p.label ? `<br><span class="k">${esc(p.label)}</span>` : ""}`);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); hideTip(); });
  svg.append(hit);
  node.append(svg);
}

// ---------- heatmap (rows = layers) ----------
function heatmap(node, grid, { cols = null, colName = "头", unit = "" } = {}) {
  node.replaceChildren();
  if (!grid || !grid.length) { node.append(el("div", { class: "empty" }, "无（未启用）")); return; }
  const nr = grid.length, nc = grid[0].length, max = Math.max(...grid.flat(), 1e-9);
  const W = 300, L = 26, T = cols ? 34 : 4, cw = (W - L) / nc, ch = Math.max(5, Math.min(12, 220 / nr)), H = T + nr * ch + 14;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  grid.forEach((row, i) => {
    row.forEach((v, j) => {
      const c = svgEl("rect", { x: L + j * cw + 1, y: T + i * ch + 1, width: Math.max(cw - 2, 1), height: ch - 2, rx: 1.5,
        fill: "var(--heat)", "fill-opacity": (0.06 + 0.94 * (v / max)).toFixed(3) });
      c.addEventListener("mousemove", (e) => showTip(e, `第 ${i} 层 · ${colName} ${cols ? cols[j] : j}<br><b>${fmt(v, 4)}</b>${unit}<br><span class="k">最大 ${fmt(max, 4)}</span>`));
      c.addEventListener("mouseleave", hideTip);
      svg.append(c);
    });
    if (i % Math.ceil(nr / 6) === 0) { const t = svgEl("text", { x: L - 4, y: T + i * ch + ch - 2, "text-anchor": "end", class: "axis-label" }); t.textContent = i; svg.append(t); }
  });
  if (cols) cols.forEach((c, j) => { const t = svgEl("text", { x: L + j * cw + cw / 2, y: T - 4, "text-anchor": "middle", class: "axis-label" }); t.textContent = c.replace("att.", "").replace("ffn.", "f.").replace("receptance", "r"); svg.append(t); });
  const lab = svgEl("text", { x: L, y: H - 2, class: "axis-label" }); lab.textContent = `↓ 层（0–${nr - 1}）  颜色越深数值越大，悬停看数值`; svg.append(lab);
  node.append(svg);
}

function renderMemory(m) {
  if (!m) return;
  heatmap($("#hm-working"), m.working);
  heatmap($("#hm-drift"), m.drift);
  heatmap($("#hm-lora"), m.lora, { cols: m.lora_modules, colName: "模块" });
}

// ---------- state ----------
const S = { lossPts: [], klPts: [], chunk: 0, info: null, rounds: [], probes: [], baseline: null };

function addLearnPoints(d, label) {
  if (d.tokens) { S.chunk += 1; S.lossPts.push({ x: S.chunk, y: d.loss, label }); }
  if (d.kl != null) S.klPts.push({ x: S.chunk, y: d.kl, label });
}
function renderLearnCharts() {
  lineChart($("#loss-chart"), S.lossPts, { name: "损失" });
  $("#kl-figure").hidden = !S.klPts.length;
  if (S.klPts.length) lineChart($("#kl-chart"), S.klPts, { name: "锚定 KL", color: "var(--series-2)" });
}

async function learn(body, label) {
  $("#learn-status").textContent = "学习中…";
  body.epochs = +$("#epochs").value || 1;
  body.label = label;
  await api.stream("/api/learn", body, (e) => {
    if (e.chunk) { addLearnPoints(e.chunk, label); renderLearnCharts(); }
    if (e.done) $("#learn-status").textContent = `完成：${e.done.tokens} token，损失 ${fmt(e.done.loss_first, 2)} → ${fmt(e.done.loss_last, 2)}，${e.done.seconds}s`;
    if (e.round) { S.rounds.push(e.round); renderProbes(); }
    if (e.memory) renderMemory(e.memory);
    if (e.error) $("#learn-status").textContent = "出错：" + e.error;
  });
}

// ---------- chat ----------
const tint = () => $("#tint").checked;
function addUser(text) { const m = el("div", { class: "msg user" }, text); $("#messages").append(m); m.scrollIntoView({ block: "end" }); }

async function send(text) {
  addUser(text);
  const body = el("div", { class: "body" }), meta = el("div", { class: "meta" });
  const msg = el("div", { class: "msg bot" }, body, meta);
  $("#messages").append(msg);
  const toks = [];
  $("#send").disabled = true;
  let pending = [];
  await api.stream("/api/chat", { prompt: text, think: $("#think").checked }, (e) => {
    if (e.memory) return renderMemory(e.memory);
    if (e.done) {
      meta.append(`${toks.length} token · 平均意外度 ${fmt(toks.reduce((a, t) => a - Math.log(Math.max(t.p || 1, 1e-9)), 0) / Math.max(toks.length, 1), 2)}`,
        el("button", { class: "ghost", onclick: () => fixUI(msg, text, e.text) }, "改写并学习"),
        el("button", { class: "ghost", onclick: () => learn({ user: text, assistant: e.text }, "强化：" + text.slice(0, 20)) }, "强化这条"));
      return;
    }
    if (e.error) { meta.append("出错：" + e.error); return; }
    // a token without visible text (first half of a split UTF-8 character) is merged into the token that completes it
    pending.push(e);
    if (!e.text) return;
    const t = { text: e.text, p: pending.reduce((a, x) => a * (x.p ?? 1), 1), top: pending[0].top };
    pending = [];
    toks.push(t);
    body.append(tokenSpan(t, tint() ? surpriseBg(-Math.log(Math.max(t.p ?? 1, 1e-9))) : "transparent"));
    msg.scrollIntoView({ block: "end" });
  });
  msg._toks = toks;
  $("#send").disabled = false;
}

function fixUI(msg, user, reply) {
  if (msg.querySelector(".fix")) return;
  const ta = el("textarea", { rows: 3 }); ta.value = reply;
  const box = el("div", { class: "fix" }, ta, el("div", { class: "row" },
    el("button", { onclick: async () => { box.remove(); await learn({ user, assistant: ta.value }, "纠正：" + user.slice(0, 20)); } }, "用改写后的回答学习"),
    el("button", { class: "ghost", onclick: () => box.remove() }, "取消")));
  msg.append(box);
  ta.focus();
}

$("#tint").addEventListener("change", () => {
  document.querySelectorAll(".msg.bot").forEach((m) => {
    (m._toks || []).forEach((t, i) => { const s = m.querySelectorAll(".tok")[i]; if (s) s.style.background = tint() ? surpriseBg(-Math.log(Math.max(t.p ?? 1, 1e-9))) : "transparent"; });
  });
});
$("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#prompt").value.trim(); if (!t) return; $("#prompt").value = ""; send(t); });
$("#prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); } });
$("#btn-reset").addEventListener("click", async () => { renderMemory(await api.post("/api/reset", { mode: "ego" })); $("#messages").append(el("div", { class: "muted hint" }, "— 工作记忆已清空，学到的东西还在 —")); });
$("#btn-teach").addEventListener("click", () => { const t = $("#teach-text").value.trim(); if (t) learn({ text: t }, "文字：" + t.slice(0, 20)); });

// ---------- probes ----------
function sparkline(vals) {
  const v = vals.filter((x) => x != null);
  if (v.length < 2) return el("span", { class: "muted" }, "—");
  const W = 70, H = 20, max = Math.max(...v), min = Math.min(...v), sx = (i) => 2 + (i / (v.length - 1)) * (W - 4), sy = (y) => H - 3 - ((y - min) / (max - min || 1)) * (H - 6);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  svg.append(svgEl("path", { d: v.map((y, i) => `${i ? "L" : "M"}${sx(i)},${sy(y)}`).join(""), fill: "none", stroke: "var(--series-1)", "stroke-width": 1.5 }));
  svg.append(svgEl("circle", { cx: sx(v.length - 1), cy: sy(v.at(-1)), r: 2.5, fill: "var(--series-1)" }));
  const t = svgEl("title"); t.textContent = v.map((x) => x.toFixed(2)).join(" → "); svg.append(t);
  return svg;
}
function renderProbes() {
  const rows = S.probes.map((p) => {
    const hist = S.rounds.map((r) => r.results.find((x) => x.id === p.id)).filter(Boolean);
    const last = hist.at(-1), first = hist[0];
    const losses = hist.map((h) => h.loss);
    let delta = "";
    if (last?.loss != null && first?.loss != null && hist.length > 1) {
      const d = last.loss - first.loss;
      delta = Math.abs(d) < 0.05 ? " ≈" : d < 0 ? ` ↓${(-d).toFixed(2)}（更熟）` : ` ↑${d.toFixed(2)}（被干扰）`;
    }
    const changed = first && last && hist.length > 1 && first.answer !== last.answer;
    return el("tr", {},
      el("td", {}, p.prompt, p.expect ? el("div", { class: "muted" }, "期望：" + p.expect) : ""),
      el("td", { class: "num" }, last?.loss != null ? fmt(last.loss, 2) + delta : "—"),
      el("td", {}, sparkline(losses)),
      el("td", { class: "ans" }, last ? last.answer : "—", changed ? el("span", { class: "badge", title: "首轮回答：" + first.answer }, "回答变了") : ""),
      el("td", {}, el("button", { class: "ghost", title: "删除", onclick: async () => { const r = await api.post("/api/probes", { action: "remove", id: p.id }); S.probes = r.probes; renderProbes(); } }, "×")));
  });
  $("#probe-rows").replaceChildren(...rows);
}
$("#probe-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const r = await api.post("/api/probes", { action: "add", prompt: $("#probe-q").value, expect: $("#probe-a").value });
  $("#probe-q").value = $("#probe-a").value = ""; S.probes = r.probes; renderProbes();
});
$("#btn-eval").addEventListener("click", async () => {
  $("#btn-eval").disabled = true;
  const r = await api.post("/api/probes", { action: "eval" }); S.probes = r.probes; S.rounds = r.rounds; renderProbes();
  $("#btn-eval").disabled = false;
});

// ---------- microscope ----------
async function analyze() {
  const text = $("#scope-text").value;
  if (!text.trim()) return null;
  return api.post("/api/tokens", { text, from: "ego" });
}
function renderScope(toks, compare) {
  const out = $("#scope-out"); out.replaceChildren();
  for (const t of toks) out.append(tokenSpan(t, compare ? deltaBg(t.delta) : surpriseBg(t.loss)));
  const mean = toks.reduce((a, t) => a + t.loss, 0) / Math.max(toks.length, 1);
  out.append(el("div", { class: "muted hint" }, compare
    ? `平均损失 ${fmt(toks.reduce((a, t) => a + t.before, 0) / toks.length, 2)} → ${fmt(mean, 2)}`
    : `平均损失 ${fmt(mean, 2)}（${toks.length} token）`));
  legend($("#scope-legend"), compare ? DELTA_BINS.map(([hi, c, a, l]) => [c ? `color-mix(in srgb, ${c} ${a}%, transparent)` : "transparent", l])
    : [[surpriseBg(0.1), "很确定"], [surpriseBg(2), "一般"], [surpriseBg(9), "很意外"]]);
}
$("#btn-scope").addEventListener("click", async () => { const t = await analyze(); if (t) renderScope(t, false); });
$("#btn-baseline").addEventListener("click", async () => {
  const t = await analyze(); if (!t) return;
  S.baseline = { text: $("#scope-text").value, toks: t }; renderScope(t, false);
  $("#btn-compare").disabled = false; $("#btn-baseline").textContent = "基线已设";
});
$("#btn-compare").addEventListener("click", async () => {
  if (!S.baseline || S.baseline.text !== $("#scope-text").value) { alert("文字和基线不一致，请先重新设为基线"); return; }
  const t = await analyze();
  t.forEach((x, i) => { x.before = S.baseline.toks[i].loss; x.delta = x.loss - x.before; });
  renderScope(t, true);
});

// ---------- settings ----------
$("#btn-settings").addEventListener("click", () => { const s = $("#settings"); s.hidden = !s.hidden; $("#btn-settings").setAttribute("aria-expanded", String(!s.hidden)); });
$("#btn-theme").addEventListener("click", () => {
  const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  r.dataset.theme = dark ? "light" : "dark";
});
document.querySelectorAll("[data-cfg]").forEach((inp) => inp.addEventListener("change", async () => {
  const [grp, key] = inp.dataset.cfg.split(".");
  S.info = await api.post("/api/config", { [grp]: { [key]: +inp.value } });
}));
$("#btn-anchors").addEventListener("click", async () => {
  $("#anchor-status").textContent = "生成中…";
  const r = await api.post("/api/anchors", { prompts: $("#anchor-prompts").value.split("\n") });
  $("#anchor-status").textContent = `${r.anchors} 个锚点`;
});
$("#btn-save").addEventListener("click", async () => { await api.post("/api/save", { path: $("#mem-path").value }); alert("已保存"); });
$("#btn-load").addEventListener("click", async () => { renderMemory(await api.post("/api/load", { path: $("#mem-path").value })); alert("已载入"); });
$("#btn-mem").addEventListener("click", async () => renderMemory(await api.get("/api/memory")));

// ---------- boot ----------
(async () => {
  const info = (S.info = await api.get("/api/info"));
  $("#model-info").textContent = `${info.variant} · ${info.n_layer} 层 × ${info.n_embd} · LoRA r=${info.lora_r} · ego ${info.tune_ego ? "可学" : "冻结"} · 锚点 ${info.anchors}`;
  document.querySelectorAll("[data-cfg]").forEach((inp) => { const [g, k] = inp.dataset.cfg.split("."); inp.value = info[g][k]; });
  S.probes = info.probes; S.rounds = info.rounds;
  for (const d of info.curve || []) addLearnPoints(d, d.label);
  $("#anchor-status").textContent = `${info.anchors} 个锚点`;
  renderProbes(); renderLearnCharts(); renderMemory(await api.get("/api/memory"));
})();
