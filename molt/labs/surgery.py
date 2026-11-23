"""State surgery lab: where does the model keep what it just read?

The model reads a context, then a question; we track log p(answer). A scan switches off each
time-mix head in turn (head_scale = 0) and wipes each layer's WKV state in turn, and records how much
the answer's log-probability drops. The head heatmap is clickable: a click toggles that head off/on
and the answer is re-measured at once, so you can disable a few heads and watch the memory break.
"""
from __future__ import annotations

import torch

from .base import Lab, Param

CTX = ("Mira keeps her spare key under the blue flowerpot next to the back door. Her neighbour Tom waters the "
       "plants on Sundays. The front door has a brass knocker shaped like a fox.")


class SurgeryLab(Lab):
    id = "surgery"
    title = "状态手术：知识存在哪儿"
    summary = ("模型先读一段文字再回答问题。逐个关掉注意力头、逐层清空 state，看答案的对数概率掉多少，就知道信息存在哪些头、哪些层。"
               "热力图可以点：点一下就关掉或恢复那个头，立刻重测。")
    objective = "（扫描型实验，无需优化）"
    params = [
        Param("context", "上下文", "text", CTX),
        Param("question", "问题", "text", "Where is the spare key?"),
        Param("answer", "答案", "text", "under the blue flowerpot"),
        Param("scan_heads", "扫描全部头（较慢）", "bool", True),
    ]

    def __init__(self, robot):
        super().__init__(robot)
        self.v = None

    def _logp(self, v, state_edit=None):
        r, m = self.robot, self.robot.model
        st = r.listen(v["context"].strip() + "\n\n", state=m.new_state())
        if state_edit:
            st = state_edit(st)
        t = r.token_losses(" " + v["answer"].strip(), state=st, prefix=f"User: {v['question'].strip()}\n\nAssistant:")
        return -sum(x["loss"] for x in t)

    def _answer(self, v):
        r, m = self.robot, self.robot.model
        st = r.listen(v["context"].strip() + "\n\n", state=m.new_state())
        return r.ask(v["question"], state=st, max_tokens=48)

    @torch.no_grad()
    def run(self, v, emit):
        m = self.robot.model
        self.v = v
        L, H = m.cfg.n_layer, m.cfg.n_head
        user_off = [[float(x) for x in b.att.head_scale.tolist()] for b in m.blocks]
        base = self._logp(v)
        emit({"log": f"当前 log p(答案) = {base:.2f}；逐层清空 state…"})
        wipe = []
        for i in range(L):
            self.progress("逐层清空 state", i, L)

            def edit(st, i=i):
                st = st.clone(); st.wkv[i].zero_(); return st
            wipe.append(round(self._logp(v, edit) - base, 3))
        grid = None
        if v["scan_heads"]:
            grid = []
            for i in range(L):
                self.progress("逐头扫描", i, L)
                row = []
                for h in range(H):
                    old = m.blocks[i].att.head_scale[h].item()
                    m.set_head_scale(i, h, 0.0)
                    row.append(round(self._logp(v) - base, 3))
                    m.set_head_scale(i, h, old)
                grid.append(row)
                if i % 4 == 3:
                    emit({"log": f"头扫描：{i + 1}/{L} 层"})
        res = {"score": 0.0, "summary": {"logp": round(base, 3)}, "viz": self._viz(v, base, wipe, grid, user_off)}
        self.last = res
        return res

    def _viz(self, v, base, wipe, grid, off):
        out = [{"type": "metrics", "items": [["log p(答案)", f"{base:.2f}", "越接近 0 越确定"],
                                             ["当前回答", self._answer(v)[:60], ""],
                                             ["已关闭的头", str(sum(1 for row in off for x in row if x == 0)), "点热力图格子开关"]]},
               {"type": "bars", "title": "逐层清空 WKV state 后 log p 的变化（越负，这层越在存这段记忆）",
                "items": [[str(i), d] for i, d in enumerate(wipe)]}]
        if grid is not None:
            out.append({"type": "heatmap", "title": "逐个关掉注意力头后 log p 的变化（点格子：关掉/恢复这个头）",
                        "grid": grid, "rows": "层", "cols": "头", "click": "toggle_head", "diverging": True,
                        "marks": [[i, h] for i, row in enumerate(off) for h, x in enumerate(row) if x == 0]})
        return out

    @torch.no_grad()
    def act(self, action, payload):
        m = self.robot.model
        if action == "toggle_head":
            i, h = int(payload["row"]), int(payload["col"])
            cur = m.blocks[i].att.head_scale[h].item()
            m.set_head_scale(i, h, 1.0 if cur == 0 else 0.0)
        elif action == "reset":
            m.reset_interventions()
        v = self.v or self.defaults()
        off = [[float(x) for x in b.att.head_scale.tolist()] for b in m.blocks]
        base = self._logp(v)
        return {"viz": [{"type": "metrics", "items": [["log p(答案)", f"{base:.2f}", ""], ["当前回答", self._answer(v)[:60], ""],
                                                      ["已关闭的头", str(sum(1 for row in off for x in row if x == 0)), ""]]}],
                "marks": [[i, h] for i, row in enumerate(off) for h, x in enumerate(row) if x == 0]}
