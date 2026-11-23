"""Memory horizon lab: how far back can the RNN state still retrieve something, and can we stretch it?

RWKV forgets through a per-head decay. This lab rescales that decay layer group by layer group
(the decay_scale intervention: <1 = forget more slowly) and runs a pass-key test: a random number is
stated once, followed by N tokens of filler, then the model is asked for it. A reference paragraph's
loss is tracked too, so you see what stretching memory costs in ordinary language modelling.
"""
from __future__ import annotations

import random

import torch

from .base import Lab, Param

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
WORDS = ("river city market winter engine garden teacher letter mountain window station paper silver morning "
         "village orange signal harbor forest planet violin ladder thunder candle museum bridge island pocket").split()
VERBS = "carries finds builds watches opens follows paints repairs counts remembers".split()


def random_filler(rng, n_words):
    out = []
    while len(out) < n_words:
        out += f"The {rng.choice(WORDS)} {rng.choice(VERBS)} the {rng.choice(WORDS)} near the {rng.choice(WORDS)}.".split()
    return " ".join(out[:n_words]) + " "
REFERENCE = ("Rivers shape the land they cross. Over thousands of years, water carries sand and silt downstream, "
             "cutting valleys in some places and building deltas in others. Cities grew along rivers because they "
             "offered water, food and a road for trade. Today engineers manage rivers with dams and levees, but "
             "floods still remind us that the river has its own memory of where it used to flow.")


class HorizonLab(Lab):
    id = "horizon"
    title = "记忆跨度：拉长 state 的记忆"
    summary = ("先说一个随机口令，接着是 N 个 token 的无关填充，最后问口令。分四组调各层的遗忘速度（衰减缩放 <1 记得更久），"
               "看不同距离上的找回率，同时盯住普通文本的损失有没有变差。")
    objective = "综合得分 = 各距离平均找回率 − 普通文本损失上升的惩罚"
    params = [Param(f"g{i}", f"第 {i + 1} 组层的衰减缩放", "float", 1.0, 0.2, 2.0, step=0.05, log=True, search=True,
                    help="<1 遗忘更慢、记得更久；>1 遗忘更快") for i in range(4)] + [
        Param("distances", "距离（token，逗号分隔）", "text", "2048,8192,32768"),
        Param("trials", "每个距离的试次", "int", 3, 1, 10, step=1),
        Param("filler", "填充", "choice", "随机句", choices=["随机句", "重复句"], help="重复句几乎不含信息，RNN 很容易跳过；随机句更难"),
        Param("decoys", "干扰口令数", "int", 4, 0, 12, step=1, help="在填充中插入作废的旧口令，模型必须找回真正的那个"),
        Param("ppl_penalty", "普通文本损失惩罚系数", "float", 2.0, 0.0, 10.0, step=0.5),
    ]

    def __init__(self, robot):
        super().__init__(robot)
        self._base = {}

    def _apply(self, v):
        m = self.robot.model
        L = m.cfg.n_layer
        for i in range(L):
            m.set_decay_scale(i, None, v[f"g{min(3, i * 4 // L)}"])

    @torch.no_grad()
    def _measure(self, dists, trials, emit=None, filler_kind="随机句", decoys=0):
        r, m, tok = self.robot, self.robot.model, self.robot.tok
        rng = random.Random(1234)
        filler = tok.encode(FILLER)
        acc = {}
        for d in dists:
            ok = 0
            for t in range(trials):
                key = str(rng.randint(10000, 99999))
                head = tok.encode(f"There is a pass key hidden in a lot of irrelevant text. Remember it. The pass key is {key}. "
                                  f"Remember it. {key} is the pass key. ")
                if filler_kind == "重复句":
                    body = (filler * (d // len(filler) + 1))[:d]
                else:
                    body = tok.encode(random_filler(rng, int(d * 0.75)))[:d]
                for j in range(decoys):  # spread superseded keys through the filler
                    pos = (j + 1) * len(body) // (decoys + 1)
                    fake = tok.encode(f" (An old pass key was {rng.randint(10000, 99999)}, but it is no longer valid.) ")
                    body = body[:pos] + fake + body[pos:]
                tail = tok.encode(" What is the pass key? The pass key is")
                st = r.listen(head + body + tail[:-1], state=m.new_state())
                saved, r.state = r.state, st
                from ..sampling import SamplingConfig
                out = r.speak(tok.decode(tail[-1:]), cfg=SamplingConfig(temperature=0, max_tokens=8, stop=("\n",)))
                r.state = saved
                ok += key in out
            acc[d] = ok / trials
            if emit:
                emit({"log": f"距离 {d}：找回 {ok}/{trials}"})
        ref = r.token_losses(REFERENCE, state=m.new_state())
        return acc, sum(x["loss"] for x in ref) / len(ref)

    def run(self, v, emit):
        dists = [int(x) for x in str(v["distances"]).replace("，", ",").split(",") if x.strip()]
        key = (tuple(dists), v["trials"], v["filler"], v["decoys"])
        m = self.robot.model
        if key not in self._base:
            emit({"log": "测基线（不做干预）…"})
            m.reset_interventions()
            self._base[key] = self._measure(dists, v["trials"], None, v["filler"], v["decoys"])
        b_acc, b_ref = self._base[key]
        self._apply(v)
        acc, ref = self._measure(dists, v["trials"], emit, v["filler"], v["decoys"])
        mean_acc = sum(acc.values()) / len(acc)
        dref = ref - b_ref
        score = mean_acc - v["ppl_penalty"] * max(0.0, dref)
        L = m.cfg.n_layer
        res = {"score": round(score, 4),
               "summary": {**{f"acc@{d}": a for d, a in acc.items()}, "ref_loss_delta": round(dref, 4), "score": round(score, 4)},
               "viz": [
                   {"type": "metrics", "items": [[f"距离 {d}", f"{a:.0%}", f"基线 {b_acc[d]:.0%}"] for d, a in acc.items()]
                    + [["普通文本损失", f"{ref:.3f}", f"基线 {b_ref:.3f}（{dref:+.3f}）"], ["综合得分", f"{score:.3f}", ""]]},
                   {"type": "line", "title": "口令找回率 vs 距离", "x": "距离（token）", "y": "找回率",
                    "series": [{"name": "当前设置", "points": [[d, a] for d, a in acc.items()]},
                               {"name": "基线", "points": [[d, a] for d, a in b_acc.items()]}]},
                   {"type": "bars", "title": "各层的衰减缩放（<1 记得更久）",
                    "items": [[str(i), round(v[f"g{min(3, i * 4 // L)}"], 3)] for i in range(L)]},
               ]}
        self.last = res
        return res
