"""Steering lab: push the residual stream along a direction found from contrasting examples.

The direction is mean(hidden | positive examples) - mean(hidden | negative examples) at one layer;
during generation alpha * |typical hidden| * unit direction is added after that layer. Drag alpha
and the test prompt is regenerated at once; the metric (e.g. share of Chinese characters) tells
whether the behaviour moved, and the fluency check tells what it cost.
"""
from __future__ import annotations

import re

import torch

from ..sampling import SamplingConfig
from .base import Lab, Param

POS = """我喜欢在周末去公园散步。
这个模型的效果比我预想的好。
请帮我把这段代码改得更简洁。
今天的天气非常适合出门。"""
NEG = """I like to walk in the park on weekends.
This model works better than I expected.
Please help me make this code more concise.
The weather today is perfect for going out."""

METRICS = {
    "中文字符占比": lambda t: len(re.findall(r"[一-鿿]", t)) / max(1, len(re.sub(r"\s", "", t))),
    "感叹号数量": lambda t: float(t.count("!") + t.count("！")),
    "平均句长（字符）": lambda t: len(t) / max(1, len(re.findall(r"[.!?。！？]", t))),
}


class SteerLab(Lab):
    id = "steer"
    title = "行为操控：残差流上的方向"
    summary = ("给两组对照例句（比如中文 vs 英文），在某一层取它们隐状态的差作为方向；生成时把这个方向按强度 α 加到残差流上。"
               "拖动 α 立刻重新生成，指标显示行为变了多少，流畅度显示代价。")
    objective = "综合得分 = 指标 − 流畅度惩罚"
    params = [
        Param("positive", "正例（每行一句）", "text", POS),
        Param("negative", "反例（每行一句）", "text", NEG),
        Param("prompt", "测试问题", "text", "Tell me about your favourite food."),
        Param("layer", "加在第几层之后", "int", 12, 0, 47, step=1, search=True),
        Param("alpha", "强度 α（实时）", "float", 4.0, -12.0, 12.0, step=0.5, live=True, search=True),
        Param("metric", "指标", "choice", "中文字符占比", choices=list(METRICS)),
        Param("fluency_penalty", "流畅度惩罚系数", "float", 0.1, 0.0, 2.0, step=0.05),
        Param("max_tokens", "生成长度", "int", 60, 16, 200, step=8),
    ]

    def __init__(self, robot):
        super().__init__(robot)
        self.dir = None   # (layer, unit vector, typical norm, direction norm)
        L = robot.model.cfg.n_layer
        self.params = [Param(**{**p.__dict__, "max": L - 1, "default": L // 2}) if p.name == "layer" else p for p in self.params]

    @torch.no_grad()
    def _direction(self, v):
        m, tok = self.robot.model, self.robot.tok
        layer = min(v["layer"], m.cfg.n_layer - 1)
        grab = {}
        h = m.blocks[layer].register_forward_hook(lambda mod, inp, out: grab.__setitem__("x", out[0]))
        try:
            def mean_hidden(lines):
                hs = []
                for line in [l for l in lines.splitlines() if l.strip()]:
                    m(torch.tensor([tok.encode(line.strip())], device=self.robot.device))
                    hs.append(grab["x"][0].float().mean(0))
                return torch.stack(hs).mean(0), torch.stack(hs).norm(dim=-1).mean()
            pos, n1 = mean_hidden(v["positive"])
            neg, n2 = mean_hidden(v["negative"])
        finally:
            h.remove()
        d = pos - neg
        self.dir = (layer, d / d.norm(), float((n1 + n2) / 2), float(d.norm()))
        return self.dir

    def _generate(self, v, alpha):
        m, r = self.robot.model, self.robot
        layer, unit, typ, _ = self.dir
        m.reset_interventions()
        m.set_steer(layer, unit * (alpha * typ / 10.0))  # alpha 10 = one typical hidden-state norm
        text = r.ask(v["prompt"], max_tokens=v["max_tokens"])
        m.set_steer(layer, None)
        fl = r.token_losses(" " + text, prefix=f"User: {v['prompt']}\n\nAssistant: <think></think>", state=r.ego) if text else []
        return text, METRICS[v["metric"]](text), (sum(x["loss"] for x in fl) / len(fl) if fl else 0.0)

    def run(self, v, emit):
        emit({"log": "计算操控方向…"})
        layer, _, typ, dn = self._direction(v)
        items, pts = [], []
        for a in sorted({0.0, -abs(v["alpha"]), abs(v["alpha"]) / 2, abs(v["alpha"])}):
            text, met, flu = self._generate(v, a)
            items.append({"title": f"α = {a:+.1f}", "text": text, "note": f"{v['metric']} {met:.2f} · 不流畅度 {flu:.2f}"})
            pts.append((a, met, flu))
            emit({"log": f"α={a:+.1f}：{v['metric']} {met:.2f}"})
        text, met, flu = self._generate(v, v["alpha"])
        score = met - v["fluency_penalty"] * flu
        res = {"score": round(score, 4), "summary": {"metric": round(met, 3), "disfluency": round(flu, 3), "score": round(score, 4)},
               "viz": [{"type": "metrics", "items": [[v["metric"], f"{met:.2f}", f"α={v['alpha']:+.1f}，第 {layer} 层"],
                                                     ["不流畅度", f"{flu:.2f}", "生成文本在未操控模型下的平均损失"],
                                                     ["方向长度", f"{dn:.1f}", f"典型隐状态范数 {typ:.1f}"]]},
                       {"type": "line", "title": f"{v['metric']} 随 α 变化", "x": "α", "y": v["metric"],
                        "series": [{"name": v["metric"], "points": [[a, m] for a, m, _ in pts]}]},
                       {"type": "texts", "title": "不同强度下的回答", "items": items}]}
        self.last = res
        return res

    def live(self, v):
        if self.dir is None or self.dir[0] != min(v["layer"], self.robot.model.cfg.n_layer - 1):
            self._direction(v)
        text, met, flu = self._generate(v, v["alpha"])
        return {"viz": [{"type": "texts", "title": f"α = {v['alpha']:+.1f} 时的回答",
                         "items": [{"title": f"{v['metric']} {met:.2f} · 不流畅度 {flu:.2f}", "text": text, "note": ""}]}]}
