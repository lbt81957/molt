"""Ways to turn new information into lasting knowledge.

Training on the raw message is immediate but shallow: the model learns to repeat the sentence,
yet often cannot answer a rephrased or reversed question or combine the fact with another one.
Each strategy here takes the same input - a list of facts (plain statements) - and writes them
into the slow tiers differently:

  sft      online: each fact, one after another, trained as a Q/A or plain sentence
  augment  rewrite each fact into many questions and answers (incl. reversed ones) first,
           by the model itself or by a teacher LLM, then train on all of them
  distill  context distillation, state -> weights: read the fact into the RNN state (the model
           "knows" it in context), then train the context-free model to match the in-context
           model's full output distribution on questions about the fact
  sleep    collect the day's facts, then consolidate in shuffled batches interleaved with
           replay of earlier facts (augmented), like hippocampal replay during sleep

All strategies respect robot.learn_cfg (lr, anchor_weight, ...) and report progress through emit.
"""
from __future__ import annotations

import random
import re

import torch
import torch.nn.functional as F

from .model import token_loss
from .robot import Robot, chat_segments, clean

STRATEGIES = ("sft", "augment", "distill", "sleep")

AUG_PROMPT = ("User: Fact: {fact}\nWrite {n} different question-answer pairs that test this fact. "
              "Vary the wording, and include questions asked in the reverse direction. "
              "Use exactly the format:\nQ: ...\nA: ...\n\nAssistant: <think></think>")


def parse_qa(text: str) -> list[tuple[str, str]]:
    pairs = re.findall(r"Q[:：]\s*(.+?)\s*\n\s*A[:：]\s*(.+?)(?=\n\s*Q[:：]|\Z)", text.strip() + "\n", re.S)
    return [(q.strip(), a.strip().split("\n")[0]) for q, a in pairs if q.strip() and a.strip()]


def augment_fact(robot: Robot, fact: str, n: int = 6, teacher=None) -> list[tuple[str, str]]:
    """Question/answer rewrites of `fact`. teacher: optional callable(prompt) -> text (an external LLM)."""
    if teacher is not None:
        text = teacher(f"Fact: {fact}\nWrite {n} different question-answer pairs that test this fact. Vary the "
                       f"wording, include reversed questions, answer briefly. Format:\nQ: ...\nA: ...")
    else:
        saved = robot.state
        robot.state = robot.ego
        from .sampling import SamplingConfig
        text = robot.speak(AUG_PROMPT.format(fact=fact, n=n), cfg=SamplingConfig(temperature=0.7, top_p=0.7, max_tokens=60 * n))
        robot.state = saved
    pairs = parse_qa(text)[:n]
    return pairs or [(f"Tell me what you know: {fact.split()[0]}", fact)]


def _sft_items(items, robot: Robot, epochs: int, emit, label: str, shuffle: bool):
    """items: list of segment lists (or strings). Trains `epochs` passes; one optimizer step per item."""
    order = list(items)
    for ep in range(epochs):
        if shuffle:
            random.shuffle(order)
        rec = robot.learn(order, on_chunk=lambda d: emit({"chunk": {**d, "label": f"{label} ep{ep + 1}"}}) if emit else None)
    return rec


def as_dialogue(q: str, a: str):
    return chat_segments([("user", q), ("assistant", a)])


def run_sft(robot: Robot, facts: list[dict], epochs: int, emit=None, **_):
    """Online: one fact at a time, all its epochs, then the next (how a chat stream arrives)."""
    for f in facts:
        item = as_dialogue(f["q"], f["fact"]) if f.get("q") else f["fact"]
        _sft_items([item], robot, epochs, emit, "sft", shuffle=False)


def run_augment(robot: Robot, facts: list[dict], epochs: int, emit=None, n_aug: int = 6, teacher=None, **_):
    items = []
    for f in facts:
        pairs = augment_fact(robot, f["fact"], n_aug, teacher)
        if emit:
            emit({"log": f"augment: {f['fact'][:40]}… -> {len(pairs)} Q/A", "pairs": pairs})
        items += [as_dialogue(q, a) for q, a in pairs] + [f["fact"]]
    _sft_items(items, robot, epochs, emit, "augment", shuffle=True)


def run_distill(robot: Robot, facts: list[dict], epochs: int, emit=None, n_aug: int = 6, teacher=None, **_):
    """Teacher = the model with the fact in its state; student = the model from ego (no context).
    Questions come from augment_fact; the teacher answers them in context, and the student is trained
    on the teacher's full next-token distribution over that answer (KL), not just the sampled tokens."""
    m, dev = robot.model, robot.device
    data = []
    for f in facts:
        ctx = robot.listen(f["fact"] + "\n\n", state=robot.ego)
        qs = [q for q, _ in augment_fact(robot, f["fact"], n_aug, teacher)] + ([f["q"]] if f.get("q") else [])
        for q in qs:
            a = robot.ask(q, state=ctx, max_tokens=48)
            prompt = robot.tok.encode(f"User: {clean(q)}\n\nAssistant: <think></think>")
            ans = robot.tok.encode(" " + a) + [0]
            with torch.no_grad():
                t_logits, _ = m(torch.tensor([prompt + ans[:-1]], device=dev), ctx)
            data.append((prompt, ans, torch.log_softmax(t_logits[0, len(prompt) - 1:].float(), -1)))
        if emit:
            emit({"log": f"distill: {f['fact'][:40]}… -> {len(qs)} questions answered in context"})
    for ep in range(epochs):
        random.shuffle(data)
        with robot.training():
            for i, (prompt, ans, t_lp) in enumerate(data):
                hid, _ = m(torch.tensor([prompt + ans[:-1]], device=dev), robot._ego_for_training(), return_hidden=True)
                s_logits = m.head(hid[0, len(prompt) - 1:])
                loss = F.kl_div(torch.log_softmax(s_logits.float(), -1), t_lp, log_target=True, reduction="batchmean")
                kl = robot.train_step(loss)
                if emit:
                    emit({"chunk": {"i": ep * len(data) + i + 1, "loss": loss.item(), "tokens": len(ans), "kl": kl,
                                    "label": f"distill ep{ep + 1}"}})


def run_sleep(robot: Robot, facts: list[dict], epochs: int, emit=None, n_aug: int = 6, teacher=None, replay=None, **_):
    """Consolidate everything at once: augmented Q/A for all facts plus replay of earlier ones,
    shuffled together every epoch, so no single fact dominates consecutive updates."""
    items = []
    for f in facts + list(replay or []):
        items += [as_dialogue(q, a) for q, a in augment_fact(robot, f["fact"], n_aug, teacher)] + [f["fact"]]
    if emit:
        emit({"log": f"sleep: {len(items)} items ({len(facts)} new facts, {len(replay or [])} replayed)"})
    _sft_items(items, robot, epochs, emit, "sleep", shuffle=True)


RUNNERS = {"sft": run_sft, "augment": run_augment, "distill": run_distill, "sleep": run_sleep}


def learn_facts(robot: Robot, facts: list[dict], strategy: str = "sft", epochs: int = 3, emit=None, **kw):
    return RUNNERS[strategy](robot, facts, epochs, emit=emit, **kw)
