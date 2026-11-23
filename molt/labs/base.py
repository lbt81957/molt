"""Lab framework.

A lab is a small, self-contained experiment on the live model:
  params    knobs, described by a schema the web page turns into sliders / menus / text boxes
  run()     do the experiment, stream progress through `emit`, return metrics + visualisations
  score()   one number to maximise, so `optimize()` can search the knobs automatically
  live()    optional: knobs that apply instantly (no retraining), with a fast re-measure
  act()     optional: clicks on a chart (e.g. switch a head off) mapped to model edits

Visualisations are plain dicts the page knows how to draw:
  {"type": "metrics", "items": [[label, value, note], ...]}
  {"type": "line", "title", "x", "y", "series": [{"name", "points": [[x, y], ...]}]}
  {"type": "bars", "title", "items": [[label, value], ...], "compare": [[label, value], ...] | None}
  {"type": "heatmap", "title", "grid", "rows", "cols", "click": action name | None, "diverging": bool}
  {"type": "table", "title", "columns", "rows"}
  {"type": "texts", "title", "items": [{"title", "text", "note"}]}
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field


@dataclass
class Param:
    name: str
    label: str
    kind: str = "float"          # float | int | choice | bool | text
    default: object = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    log: bool = False
    choices: list | None = None
    live: bool = False           # applied instantly through Lab.live()
    search: bool = False         # searched by optimize() by default
    help: str = ""

    def sample(self, rng: random.Random):
        if self.kind == "choice":
            return rng.choice(self.choices)
        if self.kind == "bool":
            return rng.random() < 0.5
        lo, hi = self.min, self.max
        v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if self.log else rng.uniform(lo, hi)
        return int(round(v)) if self.kind == "int" else v

    def perturb(self, v, rng: random.Random, scale=0.25):
        if self.kind in ("choice", "bool"):
            return v if rng.random() < 0.7 else self.sample(rng)
        lo, hi = self.min, self.max
        if self.log:
            v = math.exp(min(math.log(hi), max(math.log(lo), math.log(v) + rng.gauss(0, scale * (math.log(hi) - math.log(lo))))))
        else:
            v = min(hi, max(lo, v + rng.gauss(0, scale * (hi - lo))))
        return int(round(v)) if self.kind == "int" else v


class Lab:
    id = "lab"
    title = "Lab"
    summary = ""
    params: list[Param] = []
    objective = "score"          # label of what score() measures

    def __init__(self, robot):
        self.robot = robot
        self.last: dict | None = None

    def schema(self) -> dict:
        return {"id": self.id, "title": self.title, "summary": self.summary, "objective": self.objective,
                "params": [asdict(p) for p in self.params], "last": self.last}

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def values(self, given: dict | None) -> dict:
        v = self.defaults()
        for p in self.params:
            if given and p.name in given and given[p.name] is not None:
                x = given[p.name]
                v[p.name] = {"float": float, "int": lambda t: int(float(t)), "bool": bool}.get(p.kind, lambda t: t)(x)
        return v

    def progress(self, phase: str, done: int, total: int, job: str | None = None):
        """Report progress of a long step (shown as a bar with an ETA in the page)."""
        self.robot.bus.publish("progress", job=job or self.id, phase=phase, done=done, total=total)

    def run(self, values: dict, emit) -> dict:
        raise NotImplementedError

    def score(self, result: dict) -> float:
        return float(result.get("score", 0.0))

    def live(self, values: dict) -> dict | None:
        return None

    def act(self, action: str, payload: dict) -> dict | None:
        return None


def optimize(lab: Lab, values: dict, search: list[str], trials: int, emit, seed: int = 0) -> dict:
    """Random search, then local refinement around the best so far (half the budget each).
    Every trial is a full lab.run(); the best values are re-run at the end so the model is left in the best state."""
    rng = random.Random(seed)
    space = [p for p in lab.params if p.name in search]
    best, history = None, []
    for t in range(trials):
        lab.progress(f"自动优化：试验 {t + 1}/{trials}", t, trials, job=lab.id + ":opt")
        cand = dict(values)
        for p in space:
            cand[p.name] = p.sample(rng) if (best is None or t < trials // 2) else p.perturb(best["values"][p.name], rng)
        res = lab.run(cand, lambda e: emit({"trial_progress": {"trial": t + 1, **e}}) if "log" in e else None)
        s = lab.score(res)
        rec = {"trial": t + 1, "values": {p.name: cand[p.name] for p in space}, "metrics": res.get("summary", {}), "score": s}
        history.append(rec)
        if best is None or s > best["score"]:
            best = {**rec, "values": cand}
        emit({"trial": rec, "best": {"trial": best["trial"], "score": best["score"]}})
    lab.progress("自动优化：应用最优设置", trials, trials, job=lab.id + ":opt")
    final = lab.run(best["values"], emit)
    final["optimized"] = {"best": {k: best["values"][k] for k in search}, "trials": history}
    lab.last = final
    return final
