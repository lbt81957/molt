"""Knowledge injection lab: which learning strategy turns new facts into usable knowledge?

Each fact is tested five ways, from the model's ego state (no working memory):
  recall     the direct question            ("What is the capital of Quellia?")
  paraphrase the same question reworded
  reverse    asked from the other side       ("Zorvath is the capital of which planet?")
  compose    needs two facts learned separately ("Which language is spoken in Lumen Bay's country?")
  control    unrelated questions: does the model still give its old answers? (interference)
Training only ever sees the fact statements; the test questions are never trained on.
"""
from __future__ import annotations

import re

import torch

from ..lora import set_dial
from ..strategies import STRATEGIES, learn_facts
from .base import Lab, Param

DEFAULT_PACK = """\
F: Zorvath is the capital of the planet Quellia.
Q: What is the capital of Quellia? => Zorvath
P: Which city serves as the capital of planet Quellia? => Zorvath
R: Zorvath is the capital of which planet? => Quellia
F: The hydroglass engine was invented by Dr. Mira Okonkwo in 2031.
Q: Who invented the hydroglass engine? => Mira Okonkwo
P: Which scientist created the hydroglass engine? => Mira Okonkwo
R: What did Dr. Mira Okonkwo invent? => hydroglass engine
F: Lumen Bay is a port city in the country of Tarsis.
Q: In which country is Lumen Bay? => Tarsis
P: Lumen Bay belongs to which country? => Tarsis
R: Name a port city in Tarsis. => Lumen Bay
F: The official language of Tarsis is Veltric.
Q: What is the official language of Tarsis? => Veltric
P: Which language is official in Tarsis? => Veltric
R: Veltric is the official language of which country? => Tarsis
F: 青岚学院的校训是"知行若水"。
Q: 青岚学院的校训是什么？ => 知行若水
P: "知行若水"是哪所学院的校训？ => 青岚学院
C: What is the official language of the country where Lumen Bay is located? => Veltric
C: Which language would people speak in Lumen Bay? => Veltric
"""
DEFAULT_CONTROLS = """What is the capital of France? => Paris
Who wrote Romeo and Juliet? => Shakespeare
What is 12 times 7? => 84
What gas do plants absorb from the air? => carbon dioxide
中国的首都是哪里？ => 北京
Translate 'good morning' into French. => Bonjour
What is the largest planet in the solar system? => Jupiter
水的化学式是什么？ => H2O"""

KIND_NAMES = {"Q": "recall", "P": "paraphrase", "R": "reverse", "C": "compose"}


def parse_pack(text: str):
    facts, tests = [], []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^([FQPRC])\s*[:：]\s*(.+)$", line)
        if not m:
            continue
        tag, body = m.groups()
        if tag == "F":
            facts.append({"fact": body.strip()})
        else:
            q, _, a = body.partition("=>")
            tests.append({"kind": KIND_NAMES[tag], "q": q.strip(), "a": a.strip(), "fact": len(facts) - 1})
    return facts, tests


def parse_controls(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        if line.strip():
            q, _, a = line.partition("=>")
            out.append((q.strip(), a.strip()))
    return out


def norm(s: str) -> str:
    return re.sub(r"[\s*_`\"'“”。，.,!！?？]", "", s).lower()


class InjectLab(Lab):
    id = "inject"
    title = "知识注入：学习策略对比"
    summary = ("教模型一组虚构事实，用不同学习策略写入长期记忆，再从五个角度考它：原题、换问法、反问、"
               "跨事实组合、无关问题的干扰。训练只见过事实陈述，考题从未训练过。")
    objective = "综合得分 = 四类考题平均正确率 − 干扰惩罚"
    params = [
        Param("strategy", "学习策略", "choice", "augment", choices=list(STRATEGIES), search=True,
              help="sft 逐条原句训练 · augment 先改写成多种问答再学 · distill 把读进 state 的知识蒸馏进权重 · sleep 批量交错巩固"),
        Param("lr", "LoRA 学习率", "float", 3e-4, 1e-5, 3e-3, log=True, search=True),
        Param("epochs", "轮数", "int", 3, 1, 12, step=1, search=True),
        Param("anchor_weight", "自我锚定权重", "float", 3.0, 0.0, 10.0, step=0.5, search=True,
              help="学习时约束模型在锚点问题上不偏离学习前的自己"),
        Param("n_aug", "每条事实的改写数", "int", 6, 2, 12, step=1),
        Param("interference_penalty", "干扰惩罚系数", "float", 0.5, 0.0, 3.0, step=0.1),
        Param("dial", "LoRA 强度（实时）", "float", 1.0, 0.0, 2.0, step=0.05, live=True,
              help="0 = 学习前，1 = 学到的样子，>1 = 放大；拖动时实时重测各类考题的答案损失"),
        Param("pack", "事实与考题", "text", DEFAULT_PACK,
              help="F: 事实；Q: 原题；P: 换问法；R: 反问；C: 组合题；格式 问题 => 答案"),
        Param("controls", "干扰对照题（每行：问题 => 关键词）", "text", DEFAULT_CONTROLS,
              help="学习前后都考；答对率下降和答案损失上升都算干扰"),
    ]

    def __init__(self, robot):
        super().__init__(robot)
        self._baseline = {}

    # ---- measuring ---------------------------------------------------------
    def _prefix(self, q):
        return f"User: {q.strip()}\n\nAssistant: <think></think>"

    def _loss(self, q, a):
        t = self.robot.token_losses(" " + a, state=self.robot.ego, prefix=self._prefix(q))
        return sum(x["loss"] for x in t) / max(len(t), 1)

    def _baseline_for(self, controls):
        """Answers and answer losses of the untrained model on the control questions (cached)."""
        key = tuple(controls)
        if key not in self._baseline:
            snap = self.robot.snapshot()
            self.robot.restore("fresh")
            from ..lora import set_dial
            self.progress("建立基线（未训练模型的回答）", 0, 1)
            ans = {q: self.robot.ask(q, max_tokens=40) for q, _ in controls}
            self._baseline[key] = {"ans": ans, "loss": {q: self._loss(q, ans[q]) for q, _ in controls},
                                   "ok": {q: (norm(k) in norm(ans[q])) if k else None for q, k in controls}}
            self.robot.restore(snap)
            self.progress("建立基线（未训练模型的回答）", 1, 1)
        return self._baseline[key]

    def evaluate(self, tests, controls, answers=True) -> dict:
        base = self._baseline_for(controls)
        rows, by = [], {}
        total = len(tests) + len(controls)
        for n, t in enumerate(tests):
            self.progress("考试" if answers else "重测损失", n, total)
            loss = self._loss(t["q"], t["a"])
            ans = self.robot.ask(t["q"], max_tokens=40) if answers else ""
            ok = norm(t["a"]) in norm(ans) if answers else None
            rows.append([t["kind"], t["q"], t["a"], ans, ok, round(loss, 3)])
            by.setdefault(t["kind"], []).append((ok, loss))
        ctrl = []
        for n, (q, k) in enumerate(controls):
            self.progress("考试：对照题" if answers else "重测损失", len(tests) + n, total)
            loss = self._loss(q, base["ans"][q])            # how likely the model still finds its old answer
            ans = self.robot.ask(q, max_tokens=40) if answers else ""
            ok = (norm(k) in norm(ans)) if (answers and k) else None
            ctrl.append([q, k, base["ans"][q], ans, ok, round(loss - base["loss"][q], 3)])
        acc = {k: (sum(1 for ok, _ in v if ok) / len(v) if answers else None) for k, v in by.items()}
        loss = {k: sum(l for _, l in v) / len(v) for k, v in by.items()}
        keyed = [c for c in controls if c[1]]
        base_ok = sum(1 for q, _ in keyed if base["ok"][q]) / max(len(keyed), 1)
        now_ok = (sum(1 for r in ctrl if r[4]) / max(len(keyed), 1)) if answers else None
        return {"rows": rows, "ctrl": ctrl, "acc": acc, "loss": loss,
                "ctrl_dloss": sum(r[5] for r in ctrl) / max(len(ctrl), 1),
                "ctrl_acc": now_ok, "ctrl_acc_base": base_ok}

    # ---- lab interface -----------------------------------------------------
    def run(self, v, emit):
        r = self.robot
        facts, tests = parse_pack(v["pack"])
        controls = parse_controls(v["controls"])
        emit({"log": f"{len(facts)} 条事实，{len(tests)} 道考题，{len(controls)} 道对照题；建立基线…"})
        self._baseline_for(controls)
        r.restore("fresh")
        set_dial(r.model, 1.0)
        r.set_learn_config(lr=v["lr"], anchor_weight=v["anchor_weight"])
        emit({"log": f"策略 {v['strategy']}：学习中…"})
        curve = []

        def on(e):
            if "chunk" in e:
                curve.append(e["chunk"])
            emit(e)
        learn_facts(r, facts, v["strategy"], epochs=v["epochs"], emit=on, n_aug=v["n_aug"])
        emit({"log": "考试中…"})
        ev = self.evaluate(tests, controls)
        interf = max(0.0, ev["ctrl_acc_base"] - ev["ctrl_acc"]) + max(0.0, ev["ctrl_dloss"]) / 4
        kinds = [k for k in ("recall", "paraphrase", "reverse", "compose") if k in ev["acc"]]
        mean_acc = sum(ev["acc"][k] for k in kinds) / max(len(kinds), 1)
        score = mean_acc - v["interference_penalty"] * interf
        res = {"score": round(score, 4),
               "summary": {**{f"acc_{k}": round(ev["acc"][k], 3) for k in kinds}, "interference": round(interf, 3),
                           "ctrl_acc": round(ev["ctrl_acc"], 3), "score": round(score, 4)},
               "viz": self._viz(v, ev, interf, score, curve, kinds)}
        self.last = res
        return res

    def _viz(self, v, ev, interf, score, curve, kinds):
        name = {"recall": "原题", "paraphrase": "换问法", "reverse": "反问", "compose": "组合"}
        return [
            {"type": "metrics", "items": [[name[k], f"{ev['acc'][k]:.0%}", f"损失 {ev['loss'][k]:.2f}"] for k in kinds]
             + [["干扰", f"{interf:.3f}", f"对照题答对 {ev['ctrl_acc_base']:.0%} → {ev['ctrl_acc']:.0%}，旧答案损失 {ev['ctrl_dloss']:+.2f}"], ["综合得分", f"{score:.3f}", v["strategy"]]]},
            {"type": "line", "title": "训练损失", "x": "步", "y": "损失",
             "series": [{"name": "loss", "points": [[i + 1, c["loss"]] for i, c in enumerate(curve) if c.get("tokens")]}]},
            {"type": "table", "title": "考题明细", "columns": ["类型", "问题", "期望", "回答", "对", "损失"],
             "rows": [[name[r[0]]] + r[1:4] + ["✓" if r[4] else "✗", r[5]] for r in ev["rows"]]},
            {"type": "table", "title": "干扰对照（学习前 vs 现在）", "columns": ["问题", "关键词", "学习前", "现在", "对", "旧答案损失变化"],
             "rows": [r[:4] + [("✓" if r[4] else "✗") if r[4] is not None else "—", r[5]] for r in ev["ctrl"]]},
        ]

    def live(self, v):
        """Dial the LoRA strength and re-measure answer losses (teacher-forced: fast enough for a slider)."""
        set_dial(self.robot.model, v["dial"])
        facts, tests = parse_pack(v["pack"])
        controls = parse_controls(v["controls"])
        ev = self.evaluate(tests, controls, answers=False)
        name = {"recall": "原题", "paraphrase": "换问法", "reverse": "反问", "compose": "组合"}
        return {"viz": [{"type": "bars", "title": f"LoRA 强度 {v['dial']:.2f} 时的答案损失（越低越会）",
                         "items": [[name[k], round(l, 3)] for k, l in ev["loss"].items()] + [["对照题旧答案（损失变化）", round(ev["ctrl_dloss"], 3)]]}]}

    def act(self, action, payload):
        if action == "dial_sweep":
            v = self.values(payload)
            facts, tests = parse_pack(v["pack"])
            controls = parse_controls(v["controls"])
            pts_f, pts_c = [], []
            for d in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
                set_dial(self.robot.model, d)
                ev = self.evaluate(tests, controls, answers=False)
                new = [l for k, l in ev["loss"].items()]
                pts_f.append([d, sum(new) / len(new)]); pts_c.append([d, ev["ctrl_dloss"]])
            set_dial(self.robot.model, v["dial"])
            return {"viz": [{"type": "line", "title": "LoRA 强度扫描：新知识损失（降）与对照题损失（升）", "x": "LoRA 强度", "y": "损失",
                             "series": [{"name": "新知识考题的答案损失", "points": pts_f}, {"name": "对照题旧答案的损失变化", "points": pts_c}]}]}
        return None
