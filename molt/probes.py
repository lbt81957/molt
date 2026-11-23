"""Probes: a fixed exam re-run after every learning step.

Each probe is a question, optionally with the expected answer. After each learn() the whole set
is re-evaluated from the ego state (no working memory), so you see at once what was acquired
(loss on new facts goes down) and what was damaged (old answers change, their loss goes up).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .robot import Robot, clean
from .sampling import SamplingConfig


@dataclass
class Probe:
    id: int
    prompt: str
    expect: str = ""


@dataclass
class ProbeSet:
    probes: list[Probe] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)
    max_tokens: int = 48

    def add(self, prompt: str, expect: str = "") -> Probe:
        p = Probe(max((x.id for x in self.probes), default=0) + 1, prompt.strip(), expect.strip())
        self.probes.append(p)
        return p

    def remove(self, pid: int):
        self.probes = [p for p in self.probes if p.id != pid]

    def evaluate(self, robot: Robot, label: str = "") -> dict:
        saved = robot.state
        cfg = SamplingConfig(temperature=0, max_tokens=self.max_tokens)
        res = []
        for p in self.probes:
            q = f"User: {clean(p.prompt)}\n\nAssistant:"
            loss = None
            if p.expect:
                toks = robot.token_losses(" " + p.expect, state=robot.ego, prefix=q)
                loss = round(sum(t["loss"] for t in toks) / max(len(toks), 1), 4)
            robot.state = robot.ego
            answer = robot.speak(q, cfg=cfg)
            res.append({"id": p.id, "loss": loss, "answer": answer})
        robot.state = saved
        rnd = {"step": len(self.rounds), "label": label, "time": time.time(), "results": res}
        self.rounds.append(rnd)
        return rnd
