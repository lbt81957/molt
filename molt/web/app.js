// Molt labs UI — no framework, no build step. Each lab's knobs are generated from its parameter schema.
import { jobsView, onJobs, startPulse, trackProgress } from "./pulse.js";
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) n.append(k);
  return n;
};
const SVG = "http://www.w3.org/2000/svg";
const svgEl = (tag, attrs = {}) => { const n = document.createElementNS(SVG, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); return n; };
const svgText = (attrs, s) => { const t = svgEl("text", attrs); t.textContent = s; return t; };
const fmt = (x, d = 3) => (x == null || Number.isNaN(+x) ? String(x ?? "—") : Math.abs(+x) >= 1000 ? (+x).toFixed(0) : (+x).toFixed(d));
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]).replace(/\n/g, "↵");

const api = {
  get: (u) => fetch(u).then((r) => r.json()),
  post: (u, b) => fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) }).then((r) => r.json()),
  async stream(u, b, on) {
    const r = await fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) });
    if (!r.ok) return on({ error: `${r.status} ${await r.text()}` });
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) { const line = buf.slice(0, i); buf = buf.slice(i + 2); if (line.startsWith("data: ")) on(JSON.parse(line.slice(6))); }
    }
  },
};

// ---------- tooltip ----------
const tip = $("#tooltip");
function showTip(evt, html) {
  tip.innerHTML = html; tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + 12, y = evt.clientY + 12;
  if (x + w > innerWidth - 4) x = evt.clientX - w - 12;
  if (y + h > innerHeight - 4) y = evt.clientY - h - 12;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => (tip.hidden = true);
const hover = (node, html) => { node.addEventListener("mousemove", (e) => showTip(e, typeof html === "function" ? html() : html)); node.addEventListener("mouseleave", hideTip); };

// ---------- charts ----------
const SERIES = ["var(--series-1)", "var(--series-2)"];
function legend(names) {
  return names.length < 2 ? null : el("div", { class: "legend" }, names.map((n, i) => el("span", {}, el("span", { class: "sw", style: `background:${SERIES[i]}` }), n)));
}

function lineChart(v) {
  const wrap = el("div", { class: "chart" }, el("h3", {}, v.title || ""));
  const series = (v.series || []).filter((s) => s.points && s.points.length);
  if (!series.length) { wrap.append(el("div", { class: "muted small" }, "还没有数据")); return wrap; }
  wrap.append(legend(series.map((s) => s.name)) || "");
  const W = 560, H = 170, L = 40, R = 10, T = 8, B = 24;
  const all = series.flatMap((s) => s.points);
  const xs = all.map((p) => +p[0]), ys = all.map((p) => +p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
  const ymin = Math.min(0, ...ys), ymax = Math.max(...ys) === ymin ? ymin + 1 : Math.max(...ys) * 1.06;
  const sx = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R), sy = (y) => T + (1 - (y - ymin) / (ymax - ymin)) * (H - T - B);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": v.title || "" });
  for (let k = 0; k <= 4; k++) {
    const yv = ymin + ((ymax - ymin) * k) / 4, y = sy(yv);
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: y, y2: y, class: Math.abs(yv) < 1e-12 ? "baseline" : "gridline" }));
    svg.append(svgText({ x: L - 4, y: y + 3, "text-anchor": "end", class: "axis-label" }, fmt(yv, Math.abs(ymax) < 2 ? 2 : 1)));
  }
  const xd = xs.every((x) => Number.isInteger(x)) ? 0 : 2;
  svg.append(svgText({ x: L, y: H - 6, class: "axis-label" }, fmt(x0, xd)), svgText({ x: W - R, y: H - 6, "text-anchor": "end", class: "axis-label" }, `${fmt(x1, xd)}  ${v.x || ""}`));
  series.forEach((s, i) => {
    svg.append(svgEl("path", { d: s.points.map((p, j) => `${j ? "L" : "M"}${sx(+p[0])},${sy(+p[1])}`).join(""), fill: "none", stroke: SERIES[i], "stroke-width": 2, "stroke-linejoin": "round" }));
    if (s.points.length < 30) s.points.forEach((p) => svg.append(svgEl("circle", { cx: sx(+p[0]), cy: sy(+p[1]), r: 3, fill: SERIES[i], stroke: "var(--surface)", "stroke-width": 1.5 })));
  });
  const cross = svgEl("line", { y1: T, y2: H - B, stroke: "var(--axis)", visibility: "hidden" });
  svg.append(cross);
  const hit = svgEl("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" });
  hit.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect(), mx = ((e.clientX - r.left) / r.width) * W;
    const xsAll = [...new Set(all.map((p) => +p[0]))];
    const xv = xsAll.reduce((a, b) => (Math.abs(sx(b) - mx) < Math.abs(sx(a) - mx) ? b : a));
    cross.setAttribute("x1", sx(xv)); cross.setAttribute("x2", sx(xv)); cross.setAttribute("visibility", "visible");
    const rows = series.map((s, i) => { const p = s.points.find((q) => +q[0] === xv); return p ? `<div><span style="color:${SERIES[i]}">●</span> ${esc(s.name)} <b>${fmt(p[1])}</b></div>` : ""; }).join("");
    showTip(e, `<div class="k">${esc(v.x || "x")} = ${fmt(xv, 2)}</div>${rows}`);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  svg.append(hit);
  wrap.append(svg);
  return wrap;
}

function barsChart(v) {
  const wrap = el("div", { class: "chart" }, el("h3", {}, v.title || ""));
  const items = v.items || [];
  if (!items.length) return wrap;
  const W = 560, H = 150, L = 40, R = 6, T = 8, B = 20;
  const vals = items.map((x) => +x[1]), lo = Math.min(0, ...vals), hi = Math.max(0, ...vals) || 1;
  const sy = (y) => T + (1 - (y - lo) / (hi - lo || 1)) * (H - T - B), bw = (W - L - R) / items.length;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  [lo, (lo + hi) / 2, hi].forEach((yv) => svg.append(svgText({ x: L - 4, y: sy(yv) + 3, "text-anchor": "end", class: "axis-label" }, fmt(yv, 2))));
  svg.append(svgEl("line", { x1: L, x2: W - R, y1: sy(0), y2: sy(0), class: "baseline" }));
  const signed = lo < 0;
  items.forEach(([label, val], i) => {
    const y0 = sy(0), y1 = sy(+val), x = L + i * bw + 1;
    const r = svgEl("rect", { x, y: Math.min(y0, y1), width: Math.max(bw - 2, 1), height: Math.max(Math.abs(y1 - y0), 1), rx: Math.min(3, bw / 4),
      fill: signed ? (+val < 0 ? "var(--div-pos)" : "var(--div-neg)") : "var(--series-1)" });
    hover(r, `${esc(label)}<br><b>${fmt(val)}</b>`);
    svg.append(r);
    if (items.length <= 16 || i % Math.ceil(items.length / 12) === 0) svg.append(svgText({ x: x + bw / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" }, String(label).slice(0, 8)));
  });
  wrap.append(svg);
  return wrap;
}

function heatmapChart(v, onClick) {
  const wrap = el("div", { class: "chart" }, el("h3", {}, v.title || ""));
  const grid = v.grid || [];
  if (!grid.length) return wrap;
  const nr = grid.length, nc = grid[0].length, flat = grid.flat();
  const amax = Math.max(...flat.map(Math.abs), 1e-9);
  const W = 560, L = 26, T = 4, cw = (W - L) / nc, ch = Math.max(6, Math.min(14, 300 / nr)), H = T + nr * ch + 16;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const marks = new Set((v.marks || []).map(([a, b]) => `${a},${b}`));
  grid.forEach((row, i) => {
    row.forEach((val, j) => {
      const t = Math.abs(val) / amax;
      const fill = v.diverging ? (val < 0 ? "var(--div-pos)" : "var(--div-neg)") : "var(--heat)";
      const c = svgEl("rect", { x: L + j * cw + 1, y: T + i * ch + 1, width: Math.max(cw - 2, 1), height: ch - 2, rx: 1.5, fill, "fill-opacity": (0.05 + 0.95 * t).toFixed(3), class: onClick ? "cell" : "" });
      if (marks.has(`${i},${j}`)) { c.setAttribute("stroke", "var(--ink)"); c.setAttribute("stroke-width", 1.5); }
      hover(c, () => `第 ${i} ${v.rows || "行"} · ${v.cols || "列"} ${j}<br><b>${fmt(val)}</b>${marks.has(`${i},${j}`) ? "<br>已关闭" : ""}${onClick ? '<br><span class="k">点击：开/关</span>' : ""}`);
      if (onClick) c.addEventListener("click", () => onClick(i, j));
      svg.append(c);
    });
    if (i % Math.ceil(nr / 8) === 0) svg.append(svgText({ x: L - 4, y: T + i * ch + ch - 2, "text-anchor": "end", class: "axis-label" }, i));
  });
  svg.append(svgText({ x: L, y: H - 3, class: "axis-label" }, `↓ ${v.rows || "行"} 0–${nr - 1}   → ${v.cols || "列"} 0–${nc - 1}   ${v.diverging ? "红 = 变低（这里重要）  蓝 = 变高" : "颜色越深数值越大"}`));
  wrap.append(svg);
  return wrap;
}

function tilesView(v) {
  return el("div", { class: "tiles" }, (v.items || []).map(([k, val, note]) => el("div", { class: "tile" }, el("div", { class: "k" }, k), el("div", { class: String(val).length > 14 ? "v long" : "v" }, String(val)), el("div", { class: "n" }, note || ""))));
}
function tableView(v) {
  return el("div", { class: "chart" }, el("h3", {}, v.title || ""), el("div", { class: "tablewrap" }, el("table", { class: "t" },
    el("thead", {}, el("tr", {}, (v.columns || []).map((c) => el("th", {}, c)))),
    el("tbody", {}, (v.rows || []).map((r) => el("tr", {}, r.map((c) => el("td", { class: typeof c === "number" ? "num" : "" }, typeof c === "number" ? fmt(c) : String(c ?? "")))))))));
}
function textsView(v) {
  return el("div", { class: "chart" }, el("h3", {}, v.title || ""), el("div", { class: "texts" },
    (v.items || []).map((t) => el("div", { class: "tx" }, el("b", {}, t.title || ""), t.text || "", el("span", { class: "n" }, t.note || "")))));
}

function renderViz(list, ctx) {
  return (list || []).map((v) => {
    const card = el("div", { class: "card" });
    if (v.type === "metrics") card.append(tilesView(v));
    else if (v.type === "line") card.append(lineChart(v));
    else if (v.type === "bars") card.append(barsChart(v));
    else if (v.type === "table") card.append(tableView(v));
    else if (v.type === "texts") card.append(textsView(v));
    else if (v.type === "heatmap") card.append(heatmapChart(v, v.click ? (i, j) => ctx.act(v.click, { row: i, col: j }, v, card) : null));
    return card;
  });
}

// ---------- labs ----------
const S = { labs: [], current: null };
const store = (id) => JSON.parse(localStorage.getItem("molt.lab." + id) || "{}");
const save = (id, v) => localStorage.setItem("molt.lab." + id, JSON.stringify(v));

function knob(p, values, onLive) {
  const id = "k-" + p.name, wrap = el("div", { class: "knob" + (p.live ? " live" : "") });
  const val = el("span", { class: "val" });
  let input;
  const show = (x) => { val.textContent = p.kind === "float" ? fmt(x, Math.abs(x) < 0.01 ? 5 : 3) : p.kind === "int" ? String(x) : ""; };
  if (p.kind === "float" || p.kind === "int") {
    const toS = (x) => (p.log ? Math.log(x) : x), fromS = (s) => (p.log ? Math.exp(s) : s);
    const lo = toS(p.min), hi = toS(p.max);
    input = el("input", { id, type: "range", min: lo, max: hi, step: p.log ? (hi - lo) / 200 : p.step || (hi - lo) / 200 });
    input.value = toS(values[p.name]);
    input.addEventListener("input", () => { let x = fromS(+input.value); if (p.kind === "int") x = Math.round(x); values[p.name] = x; show(x); if (p.live) onLive(); });
  } else if (p.kind === "choice") {
    input = el("select", { id }, p.choices.map((c) => el("option", { value: c, selected: c === values[p.name] }, c)));
    input.addEventListener("change", () => { values[p.name] = input.value; if (p.live) onLive(); });
  } else if (p.kind === "bool") {
    input = el("input", { id, type: "checkbox", checked: !!values[p.name] });
    input.addEventListener("change", () => { values[p.name] = input.checked; });
  } else {
    input = el("textarea", { id, rows: Math.min(10, String(values[p.name] ?? "").split("\n").length + 1) });
    input.value = values[p.name] ?? "";
    input.addEventListener("input", () => { values[p.name] = input.value; });
  }
  show(values[p.name]);
  const srch = (p.kind === "float" || p.kind === "int" || p.kind === "choice") && !p.live
    ? el("label", { class: "srch", title: "自动优化时搜索这个旋钮" }, el("input", { type: "checkbox", "data-search": p.name, checked: p.search }), "搜索") : null;
  wrap.append(el("label", { for: id }, el("span", {}, p.label), el("span", {}, val, srch ? " " : "", srch || "")), input, p.help ? el("div", { class: "help" }, p.help) : "");
  return wrap;
}

const EXTRAS = {
  inject: [["LoRA 强度扫描", "dial_sweep"]],
  surgery: [["复位所有头", "reset"]],
};

function labView(schema) {
  const values = Object.assign(Object.fromEntries(schema.params.map((p) => [p.name, p.default])), store(schema.id));
  const jobBox = el("div", { class: "progress-inline jobs" });
  const progress = el("div", { class: "card", hidden: true }, el("h3", {}, "进度"), jobBox, el("div", { class: "log" }), el("div", { class: "pchart" }));
  const mine = (job) => job.startsWith(schema.id) || job === "learn" || job === "strategy";
  onJobs(() => { if (document.body.contains(jobBox)) jobBox.replaceChildren(...jobsView(mine)); });
  const liveBox = el("div", { class: "results" }), optBox = el("div"), resBox = el("div", { class: "results" });
  const ctx = {
    async act(action, payload, viz, card) {
      const r = await api.post(`/api/labs/${schema.id}/act`, { action, values, payload });
      if (r.marks && viz) { viz.marks = r.marks; card.replaceChildren(heatmapChart(viz, (i, j) => ctx.act(action, { row: i, col: j }, viz, card))); }
      if (r.viz) liveBox.replaceChildren(...renderViz(r.viz, ctx));
      refreshIV();
    },
  };
  let liveTimer = null, liveBusy = false;
  const onLive = () => {
    save(schema.id, values);
    clearTimeout(liveTimer);
    liveTimer = setTimeout(async () => {
      if (liveBusy) return onLive();
      liveBusy = true;
      try { const r = await api.post(`/api/labs/${schema.id}/live`, { values }); if (r.viz) liveBox.replaceChildren(...renderViz(r.viz, ctx)); refreshIV(); }
      finally { liveBusy = false; }
    }, 250);
  };
  const knobs = el("div", { class: "card knobs" }, schema.params.map((p) => knob(p, values, onLive)));
  const trials = el("input", { type: "number", value: 8, min: 2, max: 64, title: "试验次数" });
  const runBtn = el("button", {}, "运行"), optBtn = el("button", { class: "ghost" }, "自动优化");
  const extras = (EXTRAS[schema.id] || []).map(([label, action]) => el("button", { class: "ghost", onclick: () => ctx.act(action, {}) }, label));
  knobs.append(el("div", { class: "actions" }, runBtn, optBtn, trials, el("span", { class: "muted small" }, "次"), extras),
    el("div", { class: "objective" }, "优化目标：" + schema.objective));

  const busy = (b) => { runBtn.disabled = optBtn.disabled = b; };
  const startProgress = () => {
    progress.hidden = false;
    const log = $(".log", progress), pchart = $(".pchart", progress);
    log.textContent = ""; pchart.replaceChildren();
    const curve = [];
    return (e) => {
      if (e.log) { log.textContent += e.log + "\n"; log.scrollTop = 1e9; }
      if (e.trial_progress?.log) { log.textContent += `[试验 ${e.trial_progress.trial}] ${e.trial_progress.log}\n`; log.scrollTop = 1e9; }
      if (e.chunk && e.chunk.tokens) { curve.push([curve.length + 1, e.chunk.loss]); pchart.replaceChildren(lineChart({ title: "训练损失（实时）", x: "步", series: [{ name: "loss", points: curve }] })); }
      if (e.error) log.textContent += "出错：" + e.error + "\n";
      if (e.interventions) showIV(e.interventions);
    };
  };
  runBtn.onclick = async () => {
    save(schema.id, values); busy(true); optBox.replaceChildren();
    const on = startProgress();
    await api.stream(`/api/labs/${schema.id}/run`, { values }, (e) => { on(e); if (e.result) resBox.replaceChildren(...renderViz(e.result.viz, ctx)); });
    busy(false);
  };
  optBtn.onclick = async () => {
    save(schema.id, values);
    const search = [...knobs.querySelectorAll("[data-search]")].filter((x) => x.checked).map((x) => x.dataset.search);
    if (!search.length) { alert("先勾选至少一个要搜索的旋钮"); return; }
    busy(true);
    const on = startProgress(), rows = [];
    const chart = el("div"), table = el("div");
    optBox.replaceChildren(el("div", { class: "card" }, el("h3", {}, `自动优化：搜索 ${search.join("、")}（前一半随机探索，后一半在最优附近细化）`), chart, table));
    const draw = (bestTrial) => {
      chart.replaceChildren(lineChart({ title: "每次试验的得分", x: "试验", series: [{ name: "得分", points: rows.map((r) => [r.trial, r.score]) }] }));
      table.replaceChildren(el("div", { class: "tablewrap" }, el("table", { class: "t" },
        el("thead", {}, el("tr", {}, el("th", {}, "#"), search.map((s) => el("th", {}, s)), el("th", {}, "得分"), el("th", {}, "指标"))),
        el("tbody", {}, rows.map((r) => el("tr", { class: r.trial === bestTrial ? "best" : "" }, el("td", {}, r.trial),
          search.map((s) => el("td", { class: "num" }, typeof r.values[s] === "number" ? fmt(r.values[s], 4) : r.values[s])),
          el("td", { class: "num" }, fmt(r.score)), el("td", { class: "small muted" }, Object.entries(r.metrics || {}).map(([k, v]) => `${k} ${fmt(v, 2)}`).join("  "))))))));
    };
    await api.stream(`/api/labs/${schema.id}/optimize`, { values, search, trials: +trials.value }, (e) => {
      on(e);
      if (e.trial) { rows.push(e.trial); draw(e.best?.trial); }
      if (e.result) {
        resBox.replaceChildren(...renderViz(e.result.viz, ctx));
        const best = e.result.optimized?.best || {};
        Object.assign(values, best); save(schema.id, values);
        knobs.replaceWith(labView(schema).querySelector(".knobs"));  // move the sliders to the best values
        on({ log: "最优设置已应用到模型：" + Object.entries(best).map(([k, v]) => `${k}=${typeof v === "number" ? fmt(v, 4) : v}`).join("，") });
      }
    });
    busy(false);
  };
  if (schema.last?.viz) resBox.replaceChildren(...renderViz(schema.last.viz, ctx));
  return el("div", { class: "lab" }, el("div", { class: "lab-head" }, el("h2", {}, schema.title), el("p", {}, schema.summary)),
    knobs, el("div", { class: "results" }, progress, liveBox, optBox, resBox));
}

// ---------- chat lab ----------
const SURPRISE = [[0.5, 0], [1.5, 18], [3, 34], [5, 52], [Infinity, 70]];
const surpriseBg = (loss) => { const a = SURPRISE.find(([hi]) => loss < hi)[1]; return a ? `color-mix(in srgb, var(--seq-400) ${a}%, transparent)` : "transparent"; };
function chatView() {
  const msgs = el("div", { class: "messages" });
  const prompt = el("textarea", { rows: 2, placeholder: "说点什么…（Enter 发送，Shift+Enter 换行）" });
  const think = el("input", { type: "checkbox" });
  const dialVal = el("span", { class: "val" }, "1.00");
  const dial = el("input", { type: "range", min: 0, max: 2, step: 0.05, value: 1 });
  dial.addEventListener("input", async () => { dialVal.textContent = (+dial.value).toFixed(2); await api.post("/api/dial", { value: +dial.value }); refreshIV(); });
  const teach = el("textarea", { rows: 3, placeholder: "教它一段文字" });
  const status = el("div", { class: "muted small" });
  const learn = async (body, label) => {
    status.textContent = "学习中…";
    await api.stream("/api/learn", { ...body, epochs: 3, label }, (e) => {
      const d = e.result?.done;
      if (d) status.textContent = `学完：${d.tokens} token，损失 ${fmt(d.loss_first, 2)} → ${fmt(d.loss_last, 2)}`;
      if (e.error) status.textContent = "出错：" + e.error;
    });
  };
  const send = async (text) => {
    msgs.append(el("div", { class: "msg user" }, text));
    const body = el("div"), meta = el("div", { class: "meta" }), msg = el("div", { class: "msg bot" }, body, meta);
    msgs.append(msg);
    let pending = [], n = 0, sur = 0;
    await api.stream("/api/chat", { prompt: text, think: think.checked }, (e) => {
      if (e.done) {
        meta.append(`${n} token · 平均意外度 ${fmt(sur / Math.max(n, 1), 2)}`,
          el("button", { class: "ghost", onclick: () => {
            const ta = el("textarea", { rows: 3 }); ta.value = e.text;
            msg.append(el("div", { class: "composer" }, ta, el("div", { class: "row" }, el("button", { onclick: () => learn({ user: text, assistant: ta.value }, "纠正") }, "用改写后的回答学习"))));
          } }, "改写并学习"));
        return;
      }
      if (e.text == null) return;
      pending.push(e);
      if (!e.text) return;
      const p = pending.reduce((a, x) => a * (x.p ?? 1), 1), top = pending[0].top || [];
      pending = [];
      const s = el("span", { class: "tok" }, e.text);
      s.style.background = surpriseBg(-Math.log(Math.max(p, 1e-9)));
      hover(s, `“${esc(e.text)}” p=${(p * 100).toFixed(1)}%<div class="k">模型的前几名候选</div>${top.map(([w, q]) => `${esc(w)} <span class="k">${(q * 100).toFixed(1)}%</span>`).join("<br>")}`);
      body.append(s); n++; sur += -Math.log(Math.max(p, 1e-9));
      msgs.scrollTop = 1e9;
    });
  };
  prompt.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); const t = prompt.value.trim(); if (t) { prompt.value = ""; send(t); } } });
  return el("div", { class: "lab" },
    el("div", { class: "lab-head" }, el("h2", {}, "自由对话"), el("p", {}, "和当前模型（含所有干预与学到的东西）对话。token 底色越深越出乎模型意料；任何回答都能改写后立即学习。")),
    el("div", { class: "card knobs" },
      el("div", { class: "knob live" }, el("label", {}, el("span", {}, "LoRA 强度（全局）"), dialVal), dial, el("div", { class: "help" }, "0 = 学习前的模型，1 = 学到的样子")),
      el("label", { class: "small" }, think, " 思考模式"),
      el("button", { class: "ghost", onclick: async () => { await api.post("/api/reset", { mode: "ego" }); msgs.append(el("div", { class: "muted small" }, "— 工作记忆已清空 —")); } }, "清空工作记忆"),
      el("h3", {}, "教它一段文字"), teach, el("button", { onclick: () => teach.value.trim() && learn({ text: teach.value.trim() }, "文字") }, "学习"), status),
    el("div", { class: "card" }, msgs, el("div", { class: "composer" }, prompt)));
}

// ---------- interventions status ----------
function showIV(iv) {
  const off = iv.head_scale.flat().filter((x) => x === 0).length;
  const dec = iv.decay_scale.flat().filter((x) => Math.abs(x - 1) > 1e-3).length;
  const st = iv.steer_norm.filter((x) => x > 0).length;
  $("#iv-status").textContent = off || dec || st ? `当前干预：关闭头 ${off} · 改衰减的头 ${dec} · 操控层 ${st}` : "无干预";
}
const refreshIV = async () => showIV(await api.get("/api/interventions"));

// ---------- shell ----------
function select(id) {
  S.current = id;
  document.querySelectorAll(".nav button").forEach((b) => b.classList.toggle("on", b.dataset.id === id));
  $("#view").replaceChildren(id === "chat" ? chatView() : labView(S.labs.find((l) => l.id === id)));
  location.hash = id;
}
$("#btn-theme").addEventListener("click", () => {
  const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  r.dataset.theme = dark ? "light" : "dark";
});
$("#btn-iv-reset").addEventListener("click", async () => showIV(await api.post("/api/interventions/reset")));
$("#btn-forget").addEventListener("click", async () => { if (confirm("把 LoRA 和 ego 恢复到从未训练的状态？")) { await api.post("/api/memory/restore"); alert("已恢复"); } });

// ---------- pulse dock, activity, jobs, models ----------
const ACT = { idle: "空闲", listening: "在听", speaking: "在说", learning: "在学", lab: "实验中" };
startPulse({
  showTip, hideTip,
  onActivity(state, label) {
    const dot = $("#act-dot");
    dot.className = "dot" + (state === "idle" ? "" : " on") + (state === "learning" ? " learning" : "");
    $("#act-label").textContent = (ACT[state] || state) + (label ? `：${label}` : "");
  },
  onProgress: trackProgress,
  onHeartbeat(e) {
    if (e.vram_total) $("#vram").textContent = `显存 ${e.vram_used} / ${e.vram_total} GB`;
    if (e.activity === "idle") this.onActivity("idle", "");
  },
  onModel() {},
});
onJobs(() => $("#jobs").replaceChildren(...jobsView()));
setInterval(() => $("#jobs").replaceChildren(...jobsView()), 1000);   // keep ETAs ticking
$("#btn-dock").addEventListener("click", () => { document.body.classList.toggle("dock-min"); $("#btn-dock").textContent = document.body.classList.contains("dock-min") ? "▴" : "▾"; });

async function loadModels(current) {
  const r = await api.get("/api/models");
  const sel = $("#model-select");
  sel.replaceChildren(...r.models.map((m) => el("option", { value: m.path, selected: m.path === r.current }, `${m.name.replace(".pth", "")}  (${m.gb} GB)`)));
  sel.onchange = async () => {
    if (!confirm(`切换到 ${sel.selectedOptions[0].textContent.trim()}？\n当前模型、学到的东西和所有干预都会被释放。`)) { sel.value = r.current; return; }
    sel.disabled = true;
    await api.stream("/api/models/load", { path: sel.value }, (e) => { if (e.error) alert("加载失败：" + e.error); });
    location.reload();
  };
}

(async () => {
  const [info, labs] = await Promise.all([api.get("/api/info"), api.get("/api/labs")]);
  S.labs = labs;
  loadModels();
  $("#model-info").textContent = `${info.variant} · ${info.n_layer} 层 × ${info.n_embd} · LoRA r=${info.lora_r}`;
  const short = { inject: "学习策略对比，测涌现", horizon: "调遗忘速度，拉长记忆", surgery: "关头、清 state，找知识", steer: "残差方向，操控行为" };
  $("#nav").append(...labs.map((l) => el("button", { "data-id": l.id, onclick: () => select(l.id) }, l.title.split("：")[0], el("span", { class: "muted" }, short[l.id] || ""))),
    el("button", { "data-id": "chat", onclick: () => select("chat") }, "自由对话", el("span", { class: "muted" }, "对话、改写、即时学习")));
  const pick = () => { const want = location.hash.slice(1); return want === "chat" || labs.some((l) => l.id === want) ? want : labs[0].id; };
  select(pick());
  addEventListener("hashchange", () => { if (pick() !== S.current) select(pick()); });
  refreshIV();
})();
