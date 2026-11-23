"""Robot: an RWKV-7 model that listens, speaks and learns while it is being used.

Three memory tiers, from fast to slow:
  working  `self.state`   the running RNN state; every listen/speak writes to it
  ego      `self.ego`     a learned initial state (state tuning); `reset()` returns to it
  weights  LoRA adapters   float32 adapters over a frozen bf16 base

`learn()` updates ego and/or adapters by gradient descent. Long texts are cut into chunks
and the state is carried across chunk boundaries (truncated BPTT), so the model learns to
use what it read earlier in the same text instead of seeing each chunk in isolation.
"""
from __future__ import annotations

import json
import math
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decode import GraphDecoder
from .lora import add_lora, lora_state_dict, merged_state_dict
from .model import RWKV7, token_loss
from .sampling import SamplingConfig, sample
from .state import State
from .telemetry import NullBus
from .tokenizer import RWKVTokenizer

Segment = tuple[str | list[int], bool]  # (text or token ids, train_on_it)
END_OF_DOC = 0  # G1 models close a finished reply with token 0


def clean(txt: str) -> str:
    return re.sub(r"\n{2,}", "\n", txt.replace("\r\n", "\n")).strip()


def chat_segments(turns: list[tuple[str, str]], system: str | None = None) -> list[Segment]:
    """[("user", ...), ("assistant", ...), ...] -> G1 chat template; loss only on assistant replies.

    The last reply is closed with end_of_doc (token 0), so the model also learns to stop.
    """
    segs: list[Segment] = [(f"System: {clean(system)}\n\n", False)] if system else []
    for i, (role, text) in enumerate(turns):
        if role == "user":
            segs.append((f"User: {clean(text)}\n\nAssistant:", False))
        elif i == len(turns) - 1:
            segs += [(f" {text.strip()}", True), ([END_OF_DOC], True)]
        else:
            segs.append((f" {text.strip()}\n\n", True))
    return segs


@dataclass
class LearnConfig:
    lr: float = 1e-4          # LoRA learning rate
    ego_lr: float = 1e-2      # state-tuning learning rate (states tolerate far larger steps)
    chunk_len: int = 1024     # tokens per forward
    bptt: int = 2             # chunks per backward; the state is detached between groups
    min_loss: float = 0.0     # surprise gate: chunks whose loss is below this are scaled by gate_scale
    gate_scale: float = 0.05
    grad_clip: float = 1.0
    grad_checkpoint: bool = True  # recompute activations in backward: ~1/3 slower, far less VRAM
    weight_decay: float = 0.0
    anchor_weight: float = 0.0  # >0: KL(frozen self || current) on anchor transcripts


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *e): return False


@torch.no_grad()
def layer_delta(cur, prev) -> list[float] | None:
    """Per-layer size of a WKV state change, |S_cur - S_prev| (Frobenius, summed over heads)."""
    if prev is None:
        return None
    return [round(x, 4) for x in (cur[:, 0].float() - prev[:, 0].float()).flatten(1).norm(dim=1).tolist()]


class Robot:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16, lora_r: int = 0, lora_alpha: float = 32.0,
                 tune_ego: bool = False, learn_cfg: LearnConfig | None = None):
        self.model: RWKV7 = RWKV7.from_pretrained(model_path, device=device, dtype=dtype)
        self.tok = RWKVTokenizer()
        self.device = device
        self.learn_cfg = learn_cfg or LearnConfig()
        self.lora_modules = add_lora(self.model, lora_r, lora_alpha) if lora_r > 0 else []
        self.lora_r, self.lora_alpha = lora_r, lora_alpha
        # ego: learnable float32 WKV state; token-shift parts start at zero and are not tuned
        z = self.model.new_state()
        self.ego_wkv = nn.Parameter(z.wkv.clone().float(), requires_grad=tune_ego)
        self.state: State = self.ego
        self.sampling = SamplingConfig()
        self._decoder: GraphDecoder | None = None
        self._opt = None
        self.history: list[dict] = []
        self.anchors: list[list[int]] = []
        self._fresh = self.snapshot()  # the untrained slow tiers, for restore("fresh")
        self.bus = NullBus()           # set to a telemetry.Bus to stream what the model perceives and learns

    # ---- memory tiers -----------------------------------------------------
    @property
    def ego(self) -> State:
        z = self.model.new_state()
        return State(z.att_x, self.ego_wkv.detach().clone(), z.ffn_x)

    def _ego_for_training(self) -> State:
        z = self.model.new_state()
        return State(z.att_x, self.ego_wkv, z.ffn_x)

    def reset(self, mode: str = "ego", ratio: float = 0.5):
        """ego: back to the learned self; zero: blank slate; decay: fade working memory; blend: ego/working mix."""
        if mode == "ego":
            self.state = self.ego
        elif mode == "zero":
            self.state = self.model.new_state()
        elif mode == "decay":
            self.state = self.state.decay(ratio)
        elif mode == "blend":
            self.state = State.mix(self.ego, self.state, ratio, 1 - ratio)
        else:
            raise ValueError(mode)
        return self

    def lora_parameters(self):
        return [p for n, p in self.model.named_parameters() if "lora_" in n]

    # ---- snapshots of the slow tiers (labs restore them between trials) ----
    @torch.no_grad()
    def snapshot(self) -> dict:
        return {"lora": {n: p.detach().clone() for n, p in self.model.named_parameters() if "lora_" in n},
                "ego": self.ego_wkv.detach().clone()}

    @torch.no_grad()
    def restore(self, snap: dict | str = "fresh"):
        """Put adapters and ego back to a snapshot ("fresh" = never trained); also resets the optimizer."""
        snap = self._fresh if snap == "fresh" else snap
        params = dict(self.model.named_parameters())
        for n, t in snap["lora"].items():
            params[n].copy_(t)
        self.ego_wkv.copy_(snap["ego"])
        self._opt = None
        return self

    @contextmanager
    def training(self):
        """Gradients on for the trainable tiers, model in train mode with checkpointing; undone on exit."""
        for p in self.lora_parameters():
            p.requires_grad_(True)
        self.model.train()
        self.model.grad_checkpoint = self.learn_cfg.grad_checkpoint
        try:
            yield self._optimizer()
        finally:
            self.model.eval()
            self.model.grad_checkpoint = False
            self.model.requires_grad_(False)

    def train_step(self, loss) -> float | None:
        """One optimizer step on a loss built inside `with robot.training():` (anchor KL added if enabled)."""
        return self._step(loss, self._optimizer())

    @torch.no_grad()
    def ask(self, question: str, state: State | None = None, max_tokens: int = 64, think: bool = False) -> str:
        """Greedy answer from `state` (default: ego), leaving the working state untouched."""
        saved = self.state
        self.state = (state if state is not None else self.ego).clone()
        try:
            q = f"User: {clean(question)}\n\nAssistant:" + (" <think" if think else " <think></think>")
            return self.speak(q, cfg=SamplingConfig(temperature=0, max_tokens=max_tokens)).strip()
        finally:
            self.state = saved

    # ---- perception / action ---------------------------------------------
    @torch.no_grad()
    def listen(self, text: str | list[int], state: State | None = None) -> State:
        """Read text into the working state (no learning). Returns the new state."""
        ids = self.tok.encode(text) if isinstance(text, str) else list(text)
        st = self.state if state is None else state
        live = state is None and not isinstance(self.bus, NullBus)   # only the working memory is shown
        with self.bus.doing("listening") if live else _nullctx():
            for i in range(0, len(ids), 4096):
                chunk = ids[i:i + 4096]
                short = live and len(chunk) <= 1024
                before = st.wkv if live else None
                logits, st = self.model(torch.tensor([chunk], device=self.device), st, last_only=not short)
                if live:
                    ss = None
                    if short and len(chunk) > 1:
                        lp = torch.log_softmax(logits[0, :-1].float(), -1)
                        ss = [round(x, 3) for x in (-lp.gather(1, torch.tensor(chunk[1:], device=self.device)[:, None])[:, 0]).tolist()]
                    self.bus.publish("listen", n=len(chunk), d=layer_delta(st.wkv, before), ss=ss)
        if state is None:
            self.state = st
        return st

    @torch.no_grad()
    def speak_iter(self, prompt: str = "", cfg: SamplingConfig | None = None, topk: int = 5):
        """Generator version of speak(). Yields one dict per sampled token:
        {"id", "text" (the new characters, may be ""), "p" (model probability of the token),
         "top": [(text, p), ...] (the model's top-k alternatives)}; probabilities are the raw
        model distribution (temperature 1, no penalties), so they explain the model, not the sampler.
        The final yield is {"done": True, "text": reply}."""
        cfg = cfg or self.sampling
        ids = self.tok.encode(prompt) if prompt else []
        if len(ids) > 1:
            self.listen(ids[:-1])
        first = ids[-1] if ids else 261  # "\n\n"
        step = self._stepper()
        live = not isinstance(self.bus, NullBus)
        prev = None
        occ = torch.zeros(self.model.cfg.vocab_size, device=self.device)
        prev_act = self.bus.activity
        self.bus.activity = "speaking"
        self.bus.publish("activity", state="speaking", label="")
        out, text, tok = [], "", first
        hold = max((len(s) for s in cfg.stop), default=0)
        queue, attributed = [], 0  # tokens not yet sent; characters already given to some token
        for _ in range(cfg.max_tokens):
            logits, wkv = step(tok)
            probs = torch.softmax(logits.float(), -1)
            tok = sample(logits, cfg, occ)
            if live:
                d = layer_delta(wkv, prev) if prev is not None else None
                prev = wkv.clone() if prev is None else prev.copy_(wkv)
                self.bus.publish("tok", s=round(-math.log(max(float(probs[tok]), 1e-12)), 3), d=d,
                                 t=self.tok.decode([tok]) if tok else "")
            if tok == 0:  # end_of_doc
                break
            out.append(tok)
            text = self.tok.decode(out)
            tp, ti = probs.topk(topk)
            # a token's own text: the characters it completes (a split UTF-8 character goes to the token finishing it)
            stable = len(text) - (1 if text.endswith("�") else 0)
            queue.append({"id": tok, "p": float(probs[tok]), "text": text[attributed:stable],
                          "top": [(self.tok.decode([i]), round(float(p), 4)) for p, i in zip(tp.tolist(), ti.tolist())]})
            attributed = max(attributed, stable)
            hit = next((s for s in cfg.stop if s in text), None)
            if hit:
                text = text[: text.index(hit)]
                break
            # send tokens whose text lies entirely before the hold-back zone (a stop string may still form there)
            pos = attributed - sum(len(d["text"]) for d in queue)
            while queue and pos + len(queue[0]["text"]) <= len(text) - hold:
                pos += len(queue[0]["text"])
                yield queue.pop(0)
        pos = attributed - sum(len(d["text"]) for d in queue)
        for d in queue:  # flush, trimming whatever lies past a cut stop string
            keep = max(0, min(len(d["text"]), len(text) - pos))
            pos += len(d["text"])
            if keep:
                d["text"] = d["text"][:keep]
                yield d
        # Re-read the reply as kept (stop string cut, turn closed with "\n\n") so the working
        # state matches the transcript exactly rather than whatever was sampled past the stop.
        self.listen([first] + self.tok.encode(text + "\n\n"))
        self.bus.activity = prev_act
        self.bus.publish("activity", state=prev_act, label="")
        yield {"done": True, "text": text}

    def speak(self, prompt: str = "", cfg: SamplingConfig | None = None, stream=None) -> str:
        """Continue from the working state (after reading `prompt`). `stream(piece)` gets text as it is produced."""
        for d in self.speak_iter(prompt, cfg):
            if d.get("done"):
                return d["text"]
            if stream and d["text"]:
                stream(d["text"])

    def _stepper(self):
        """One-token decode function starting from the working state (which it does not modify).
        CUDA graph when possible; eager for CPU and for attention variants (their cache grows)."""
        if self.device != "cpu" and str(self.device).startswith("cuda") and not self.model.has_attention:
            if self._decoder is None:
                self._decoder = GraphDecoder(self.model)
            self._decoder.set_state(self.state)
            return lambda t: (self._decoder.step(t)[0], self._decoder.state.wkv)
        box = {"st": self.state}

        def step(t):
            logits, box["st"] = self.model(torch.tensor([[t]], device=self.device), box["st"], last_only=True)
            return logits[0, -1], box["st"].wkv
        return step

    def chat(self, user: str, think: bool = False, **kw) -> str:
        return self.speak(f"User: {clean(user)}\n\nAssistant:" + (" <think" if think else ""), **kw)

    @torch.no_grad()
    def reflect(self, text: str, state: State | None = None) -> dict:
        """Surprise of `text` under the current memory: mean loss and perplexity; nothing is updated."""
        ids = self.tok.encode(text)
        st = self.state if state is None else state
        tot, n = 0.0, 0
        for i in range(0, len(ids) - 1, 4096):
            x = ids[i:i + 4097]
            logits, st = self.model(torch.tensor([x[:-1]], device=self.device), st)
            tot += F.cross_entropy(logits[0].float(), torch.tensor(x[1:], device=self.device), reduction="sum").item()
            n += len(x) - 1
        return {"loss": tot / max(n, 1), "ppl": float(torch.tensor(tot / max(n, 1)).exp()), "tokens": n}

    @torch.no_grad()
    def token_losses(self, text: str, state: State | None = None, prefix: str = "") -> list[dict]:
        """Per-token surprise of `text` (after reading `prefix`) under the current memory:
        [{"text", "loss", "p", "top": [(text, p)] }]. Nothing is updated."""
        st = self.state if state is None else state
        pre = self.tok.encode(prefix) if prefix else []
        ids = self.tok.encode(text)
        if pre[:-1]:
            st = self.listen(pre[:-1], state=st)
        seq = (pre[-1:] or [261]) + ids
        logits, _ = self.model(torch.tensor([seq[:-1]], device=self.device), st)
        lp = torch.log_softmax(logits[0].float(), -1)
        tgt = torch.tensor(seq[1:], device=self.device)
        loss = -lp.gather(1, tgt[:, None])[:, 0]
        tp, ti = lp.exp().topk(3, -1)
        return [{"text": self.tok.decode([t]), "loss": round(float(l), 3), "p": round(math.exp(-l), 4),
                 "top": [(self.tok.decode([i]), round(p, 3)) for p, i in zip(a, b)]}
                for t, l, a, b in zip(ids, loss.tolist(), tp.tolist(), ti.tolist())]

    # ---- introspection ----------------------------------------------------
    @torch.no_grad()
    def memory_stats(self) -> dict:
        """Numbers behind the memory heatmaps: per layer x head WKV norms of the working state,
        its distance from ego, ego norms, and per layer x module LoRA update norms ||scale*B@A||."""
        w = self.state.wkv[:, 0].float()                      # [L, H, N, N]
        e = self.ego_wkv.detach()[:, 0].float()
        norm = lambda t: t.flatten(2).norm(dim=-1).cpu().tolist()
        out = {"working": norm(w), "drift": norm(w - e), "ego": norm(e), "lora": None, "lora_modules": []}
        from .lora import LoRALinear
        mods = [(n, m) for n, m in self.model.named_modules() if isinstance(m, LoRALinear)]
        if mods:
            names = sorted({n.split(".", 2)[2] for n, _ in mods})
            grid = [[0.0] * len(names) for _ in range(self.model.cfg.n_layer)]
            for n, m in mods:
                layer = int(n.split(".")[1])
                grid[layer][names.index(n.split(".", 2)[2])] = float((m.lora_B @ m.lora_A).norm() * m.scale)
            out["lora"], out["lora_modules"] = grid, names
        return out

    # ---- learning ---------------------------------------------------------
    @torch.no_grad()
    def make_anchors(self, prompts: list[str], max_tokens: int = 96):
        """Ouroboros-style anchors: the model answers `prompts` itself and keeps the transcripts.
        With anchor_weight > 0, learn() adds KL to the frozen self on them, limiting collateral damage."""
        saved = self.state
        cfg = SamplingConfig(temperature=0, max_tokens=max_tokens)
        for p in prompts:
            self.state = self.model.new_state()
            q = f"User: {clean(p)}\n\nAssistant:"
            a = self.speak(q, cfg=cfg)
            self.anchors.append(self.tok.encode(q + " " + a.strip()))
        self.state = saved
        return self

    def _anchor_loss(self):
        from .lora import lora_disabled
        ids = self.anchors[torch.randint(len(self.anchors), (1,)).item()]
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad(), lora_disabled():
            ref, _ = self.model(x)
        cur, _ = self.model(x, self._ego_for_training())
        return F.kl_div(F.log_softmax(cur[0].float(), -1), F.log_softmax(ref[0].float(), -1),
                        log_target=True, reduction="batchmean")

    def _encode_segments(self, item) -> tuple[list[int], list[int]]:
        segs = [(item, True)] if isinstance(item, str) else item
        ids, mask = [], []
        for text, train in segs:
            t = self.tok.encode(text) if isinstance(text, str) else list(text)
            ids += t; mask += [int(train)] * len(t)
        return ids, mask

    def _optimizer(self):
        if self._opt is None:
            c = self.learn_cfg
            groups = []
            lora = self.lora_parameters()
            if lora:
                groups.append({"params": lora, "lr": c.lr})
            if self.ego_wkv.requires_grad:
                groups.append({"params": [self.ego_wkv], "lr": c.ego_lr})
            if not groups:
                raise RuntimeError("nothing to train: enable lora_r > 0 or tune_ego=True")
            self._opt = torch.optim.AdamW(groups, weight_decay=c.weight_decay, betas=(0.9, 0.99))
        return self._opt

    def learn(self, data, epochs: int = 1, absorb: bool = False, on_chunk=None) -> dict:
        """Gradient-update the slow memory tiers on `data`.

        data: a string, a list of strings, or a list of segment lists [(text, train?), ...]
              (use chat_segments() for dialogues: loss only on the assistant's words).
        absorb: afterwards also read the data into the working state.
        on_chunk: called after every trained chunk with {"i", "loss", "tokens", "kl" (anchor KL, at optimizer steps)}.
        """
        items = [data] if isinstance(data, str) or (data and isinstance(data[0], tuple)) else list(data)
        c, opt, m = self.learn_cfg, self._optimizer(), self.model
        for p in self.lora_parameters():
            p.requires_grad_(True)
        m.train()
        m.grad_checkpoint = c.grad_checkpoint
        losses, n_tok, t0 = [], 0, time.time()
        encoded = [self._encode_segments(it) for it in items]
        total = epochs * sum(max(1, -(-(len(ids) - 1) // c.chunk_len)) for ids, _ in encoded)
        done = 0
        prev_act = self.bus.activity
        self.bus.activity = "learning"
        self.bus.publish("activity", state="learning", label="")
        for _ in range(epochs):
            for ids, mask in encoded:
                st = self._ego_for_training()
                acc, k = 0.0, 0
                for i in range(0, len(ids) - 1, c.chunk_len):
                    x = ids[i:i + c.chunk_len + 1]
                    mk = torch.tensor(mask[i + 1:i + len(x)], device=self.device, dtype=torch.float32)
                    if len(x) < 2 or mk.sum() == 0:
                        with torch.no_grad():
                            _, st = m(torch.tensor([x[:-1]], device=self.device), st.detach())
                        continue
                    hid, st = m(torch.tensor([x[:-1]], device=self.device), st, return_hidden=True)
                    ce = token_loss(m, hid[0], torch.tensor(x[1:], device=self.device))
                    loss = (ce * mk).sum() / mk.sum()
                    losses.append(loss.item()); n_tok += int(mk.sum())
                    scale = c.gate_scale if loss.item() < c.min_loss else 1.0
                    acc = acc + loss * scale; k += 1
                    kl = None
                    if k == c.bptt:
                        kl = self._step(acc / k, opt); acc, k = 0.0, 0
                        st = st.detach()
                    done += 1
                    self.bus.publish("progress", job="learn", phase="学习", done=done, total=total)
                    if on_chunk:
                        on_chunk({"i": len(losses), "loss": losses[-1], "tokens": int(mk.sum()), "kl": kl})
                if k:
                    kl = self._step(acc / k, opt)
                    if on_chunk and kl is not None:
                        on_chunk({"i": len(losses), "loss": losses[-1], "tokens": 0, "kl": kl})
        m.eval()
        m.grad_checkpoint = False
        self.model.requires_grad_(False)
        self.bus.activity = prev_act
        self.bus.publish("activity", state=prev_act, label="")
        if absorb:
            for item in items:
                self.listen(self._encode_segments(item)[0])
        rec = {"chunks": len(losses), "tokens": n_tok, "loss_first": losses[0] if losses else None,
               "loss_last": losses[-1] if losses else None, "loss_mean": sum(losses) / max(len(losses), 1),
               "seconds": round(time.time() - t0, 2)}
        self.history.append({"event": "learn", **rec})
        return rec

    def _step(self, loss, opt):
        """One optimizer step; returns the anchor KL (None if anchoring is off)."""
        opt.zero_grad(set_to_none=True)
        kl = None
        if self.learn_cfg.anchor_weight > 0 and self.anchors:
            kl_t = self._anchor_loss()
            loss = loss + self.learn_cfg.anchor_weight * kl_t
            kl = kl_t.item()
        loss.backward()
        params = [p for g in opt.param_groups for p in g["params"]]
        gnorm = torch.nn.utils.clip_grad_norm_(params, self.learn_cfg.grad_clip)
        opt.step()
        if not isinstance(self.bus, NullBus):
            from .lora import lora_layer_norms
            self.bus.publish("step", loss=round(float(loss.detach()), 4), kl=kl, gnorm=round(float(gnorm), 4),
                             lora=lora_layer_norms(self.model))
        return kl

    def set_learn_config(self, **kw):
        """Change learning hyper-parameters on the fly (learning rates apply to the live optimizer)."""
        for k, v in kw.items():
            if hasattr(self.learn_cfg, k):
                setattr(self.learn_cfg, k, type(getattr(self.learn_cfg, k))(v))
        if self._opt is not None:
            for g in self._opt.param_groups:
                g["lr"] = self.learn_cfg.ego_lr if any(p is self.ego_wkv for p in g["params"]) else self.learn_cfg.lr

    # ---- persistence ------------------------------------------------------
    def save(self, path):
        """Save the learned tiers (ego + adapters) and the working state, not the base weights."""
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        torch.save({"ego_wkv": self.ego_wkv.detach().cpu(), "lora": lora_state_dict(self.model)}, p / "memory.pth")
        self.state.save(p / "working_state.pth")
        (p / "meta.json").write_text(json.dumps({"lora_r": self.lora_r, "lora_alpha": self.lora_alpha,
                                                 "learn": asdict(self.learn_cfg), "history": self.history},
                                                ensure_ascii=False, indent=1), encoding="utf-8")

    def load(self, path):
        p = Path(path)
        d = torch.load(p / "memory.pth", map_location=self.device, weights_only=True)
        with torch.no_grad():
            self.ego_wkv.copy_(d["ego_wkv"])
        if d["lora"]:
            self.model.load_state_dict(d["lora"], strict=False)
        if (p / "working_state.pth").exists():
            self.state = State.load(p / "working_state.pth", self.device)
        return self

    def export_merged(self, path):
        """Write an official-format .pth with adapters merged (ego is saved separately via save())."""
        torch.save(merged_state_dict(self.model), path)
