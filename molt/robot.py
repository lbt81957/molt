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
import re
import time
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


class Robot:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16, lora_r: int = 0, lora_alpha: float = 32.0,
                 tune_ego: bool = True, learn_cfg: LearnConfig | None = None):
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

    # ---- perception / action ---------------------------------------------
    @torch.no_grad()
    def listen(self, text: str | list[int], state: State | None = None) -> State:
        """Read text into the working state (no learning). Returns the new state."""
        ids = self.tok.encode(text) if isinstance(text, str) else list(text)
        st = self.state if state is None else state
        for i in range(0, len(ids), 4096):
            _, st = self.model(torch.tensor([ids[i:i + 4096]], device=self.device), st, last_only=True)
        if state is None:
            self.state = st
        return st

    @torch.no_grad()
    def speak(self, prompt: str = "", cfg: SamplingConfig | None = None, stream=None) -> str:
        """Continue from the working state (after reading `prompt`). `stream(piece)` gets text as it is produced."""
        cfg = cfg or self.sampling
        ids = self.tok.encode(prompt) if prompt else []
        if len(ids) > 1:
            self.listen(ids[:-1])
        first = ids[-1] if ids else 261  # "\n\n"
        step = self._stepper()
        occ = torch.zeros(self.model.cfg.vocab_size, device=self.device)
        out, text, emitted, tok = [], "", 0, first
        for _ in range(cfg.max_tokens):
            logits = step(tok)
            tok = sample(logits, cfg, occ)
            if tok == 0:  # end_of_doc
                break
            out.append(tok)
            text = self.tok.decode(out)
            hit = next((s for s in cfg.stop if s in text), None)
            if hit:
                text = text[: text.index(hit)]
                break
            safe = len(text) - max(len(s) for s in cfg.stop) if cfg.stop else len(text)
            if stream and safe > emitted and "�" not in text[emitted:safe]:
                stream(text[emitted:safe]); emitted = safe
        if stream and len(text) > emitted:
            stream(text[emitted:])
        # Re-read the reply as kept (stop string cut, turn closed with "\n\n") so the working
        # state matches the transcript exactly rather than whatever was sampled past the stop.
        self.listen([first] + self.tok.encode(text + "\n\n"))
        return text

    def _stepper(self):
        """One-token decode function starting from the working state (which it does not modify).
        CUDA graph when possible; eager for CPU and for attention variants (their cache grows)."""
        if self.device != "cpu" and str(self.device).startswith("cuda") and not self.model.has_attention:
            if self._decoder is None:
                self._decoder = GraphDecoder(self.model)
            self._decoder.set_state(self.state)
            return lambda t: self._decoder.step(t)[0]
        box = {"st": self.state}

        def step(t):
            logits, box["st"] = self.model(torch.tensor([[t]], device=self.device), box["st"], last_only=True)
            return logits[0, -1]
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

    def learn(self, data, epochs: int = 1, absorb: bool = False) -> dict:
        """Gradient-update the slow memory tiers on `data`.

        data: a string, a list of strings, or a list of segment lists [(text, train?), ...]
              (use chat_segments() for dialogues: loss only on the assistant's words).
        absorb: afterwards also read the data into the working state.
        """
        items = [data] if isinstance(data, str) or (data and isinstance(data[0], tuple)) else list(data)
        c, opt, m = self.learn_cfg, self._optimizer(), self.model
        for p in self.lora_parameters():
            p.requires_grad_(True)
        m.train()
        m.grad_checkpoint = c.grad_checkpoint
        losses, n_tok, t0 = [], 0, time.time()
        for _ in range(epochs):
            for item in items:
                ids, mask = self._encode_segments(item)
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
                    if k == c.bptt:
                        self._step(acc / k, opt); acc, k = 0.0, 0
                        st = st.detach()
                if k:
                    self._step(acc / k, opt)
        m.eval()
        m.grad_checkpoint = False
        self.model.requires_grad_(False)
        if absorb:
            for item in items:
                self.listen(self._encode_segments(item)[0])
        rec = {"chunks": len(losses), "tokens": n_tok, "loss_first": losses[0] if losses else None,
               "loss_last": losses[-1] if losses else None, "loss_mean": sum(losses) / max(len(losses), 1),
               "seconds": round(time.time() - t0, 2)}
        self.history.append({"event": "learn", **rec})
        return rec

    def _step(self, loss, opt):
        opt.zero_grad(set_to_none=True)
        if self.learn_cfg.anchor_weight > 0 and self.anchors:
            loss = loss + self.learn_cfg.anchor_weight * self._anchor_loss()
        loss.backward()
        params = [p for g in opt.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, self.learn_cfg.grad_clip)
        opt.step()

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
