"""RWKV-7 "Goose" (x070) as a trainable nn.Module.

Parameter names and shapes match BlinkDL's .pth checkpoints exactly, so official weights
load with load_state_dict and anything we save loads back into official tools.

The FFN (channel-mix) is pluggable: experimental generations (v7a, v8-rc00, ...) keep the
x070 time-mix and change only the FFN; see molt/variants.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .ops import wkv7
from .state import State

EXP_HALF = 0.6065306597126334  # exp(-0.5): max per-step log-decay magnitude


@dataclass
class RWKVConfig:
    vocab_size: int = 65536
    n_layer: int = 12
    n_embd: int = 768
    head_size: int = 64
    dim_ffn: int = 3072
    dim_w: int = 64     # low-rank sizes of the decay / in-context-lr / value-residual / gate LoRAs
    dim_a: int = 64
    dim_v: int = 32
    dim_g: int = 128
    variant: str = "x070"
    extra: dict = field(default_factory=dict)

    @property
    def n_head(self) -> int:
        return self.n_embd // self.head_size

    @staticmethod
    def from_state_dict(sd: dict) -> "RWKVConfig":
        from .variants import detect_variant
        n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        V, C = sd["head.weight"].shape
        H, N = sd["blocks.0.att.r_k"].shape
        return RWKVConfig(
            vocab_size=V, n_layer=n_layer, n_embd=C, head_size=N,
            dim_ffn=sd["blocks.0.ffn.key.weight"].shape[0],
            dim_w=sd["blocks.0.att.w1"].shape[1], dim_a=sd["blocks.0.att.a1"].shape[1],
            dim_v=sd["blocks.1.att.v1"].shape[1] if n_layer > 1 else 32, dim_g=sd["blocks.0.att.g1"].shape[1],
            variant=detect_variant(sd),
        )


def _p(*shape):
    return nn.Parameter(torch.zeros(*shape))


class TimeMix(nn.Module):
    def __init__(self, cfg: RWKVConfig, layer_id: int):
        super().__init__()
        self.layer_id, self.H, self.N = layer_id, cfg.n_head, cfg.head_size
        C = cfg.n_embd
        for n in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "v0", "k_k", "k_a"):
            setattr(self, n, _p(1, 1, C))
        self.w1, self.w2 = _p(C, cfg.dim_w), _p(cfg.dim_w, C)
        self.a1, self.a2 = _p(C, cfg.dim_a), _p(cfg.dim_a, C)
        self.v1, self.v2 = _p(C, cfg.dim_v), _p(cfg.dim_v, C)   # unused in layer 0, kept for checkpoint compat
        self.g1, self.g2 = _p(C, cfg.dim_g), _p(cfg.dim_g, C)
        self.r_k = _p(self.H, self.N)
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(self.H, C, eps=64e-5)
        self._init_interventions()

    def _init_interventions(self, device=None):
        # decay_scale multiplies the per-head log decay: <1 remembers longer, >1 forgets faster.
        # head_scale multiplies each head's output: 0 switches the head off.
        self.register_buffer("decay_scale", torch.ones(self.H, device=device), persistent=False)
        self.register_buffer("head_scale", torch.ones(self.H, device=device), persistent=False)

    def forward(self, x, x_prev, wkv, v_first, backend="auto"):
        B, T, C = x.shape
        H, N = self.H, self.N
        xx = torch.cat((x_prev.unsqueeze(1), x[:, :-1]), dim=1) - x
        xr, xw, xk, xv, xa, xg = (x + xx * m for m in (self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g))

        r = self.receptance(xr)
        w = -EXP_HALF * torch.sigmoid((self.w0 + torch.tanh(xw @ self.w1) @ self.w2).float())  # log decay
        k = self.key(xk)
        v = self.value(xv)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(B, T, H, N), dim=-1, p=2.0)
        k = k * (1 + (a - 1) * self.k_a)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        hv = lambda t: t.view(B, T, H, N)
        w = hv(w) * self.decay_scale.float().view(1, 1, H, 1)
        o, wkv = wkv7(hv(r), w.to(r.dtype), hv(k), hv(v), -kk, kk * hv(a), wkv, backend=backend)

        o = self.ln_x(o.reshape(B * T, C)).view(B, T, C)
        o = o + ((hv(r) * hv(k) * self.r_k).sum(-1, keepdim=True) * hv(v)).view(B, T, C)
        o = (hv(o) * self.head_scale.view(1, 1, H, 1).to(o.dtype)).view(B, T, C)
        return self.output(o * g), x[:, -1], wkv, v_first


class ChannelMix(nn.Module):
    """x070 FFN: squared-ReLU MLP with token shift."""

    def __init__(self, cfg: RWKVConfig, layer_id: int):
        super().__init__()
        C = cfg.n_embd
        self.x_k = _p(1, 1, C)
        self.key = nn.Linear(C, cfg.dim_ffn, bias=False)
        self.value = nn.Linear(cfg.dim_ffn, C, bias=False)

    def hidden(self, x, x_prev):
        xx = torch.cat((x_prev.unsqueeze(1), x[:, :-1]), dim=1) - x
        return torch.relu(self.key(x + xx * self.x_k)) ** 2

    def forward(self, x, x_prev, tok=None):
        """tok: {"idx": [B,T] ids, "x0": [B,T,C] = ln0(emb(idx)), ...}, used by token-conditioned variants."""
        return self.value(self.hidden(x, x_prev)), x[:, -1]


class Block(nn.Module):
    def __init__(self, cfg: RWKVConfig, layer_id: int, ffn_cls):
        super().__init__()
        C = cfg.n_embd
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(C)
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.att = TimeMix(cfg, layer_id)
        self.ffn = ffn_cls(cfg, layer_id)

    def forward(self, x, att_x, wkv, ffn_x, v_first, tok, cache=None, backend="auto"):
        h, att_x, wkv, v_first = self.att(self.ln1(x), att_x, wkv, v_first, backend)
        x = x + h
        h, ffn_x = self.ffn(self.ln2(x), ffn_x, tok)
        return x + h, att_x, wkv, ffn_x, v_first, None


class RWKV7(nn.Module):
    def __init__(self, cfg: RWKVConfig):
        super().__init__()
        from .variants import block_class, ffn_class
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        blk = block_class(cfg.variant)
        self.blocks = nn.ModuleList(blk(cfg, i, ffn_class(cfg.variant)) for i in range(cfg.n_layer))
        self.has_attention = cfg.variant == "x070b"
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.grad_checkpoint = False
        self.backend = "auto"
        self.register_buffer("steer", torch.zeros(cfg.n_layer, cfg.n_embd), persistent=False)

    # ---- interventions -----------------------------------------------------
    # All are tensors changed in place, so a captured CUDA graph sees every change.
    @torch.no_grad()
    def reset_interventions(self):
        for b in self.blocks:
            b.att.decay_scale.fill_(1.0)
            b.att.head_scale.fill_(1.0)
        self.steer.zero_()

    @torch.no_grad()
    def set_decay_scale(self, layer, head=None, value=1.0):
        t = self.blocks[layer].att.decay_scale
        (t if head is None else t[head:head + 1]).fill_(value)

    @torch.no_grad()
    def set_head_scale(self, layer, head=None, value=1.0):
        t = self.blocks[layer].att.head_scale
        (t if head is None else t[head:head + 1]).fill_(value)

    @torch.no_grad()
    def set_steer(self, layer, vector=None):
        """Add `vector` [C] to the residual stream after block `layer` (None clears it)."""
        if vector is None:
            self.steer[layer].zero_()
        else:
            self.steer[layer].copy_(vector.to(self.steer))

    def interventions(self) -> dict:
        return {"decay_scale": torch.stack([b.att.decay_scale for b in self.blocks]).float().cpu().tolist(),
                "head_scale": torch.stack([b.att.head_scale for b in self.blocks]).float().cpu().tolist(),
                "steer_norm": self.steer.float().norm(dim=-1).cpu().tolist()}

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_pretrained(cls, path, device="cuda", dtype=torch.bfloat16) -> "RWKV7":
        sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        cfg = RWKVConfig.from_state_dict(sd)
        with torch.device("meta"):
            model = cls(cfg)
        model.load_state_dict(sd, strict=True, assign=True)
        model._init_interventions_cpu()
        return model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    def _init_interventions_cpu(self):
        """Buffers built on the meta device during from_pretrained are materialised here."""
        for b in self.blocks:
            b.att._init_interventions("cpu")
        self.register_buffer("steer", torch.zeros(self.cfg.n_layer, self.cfg.n_embd), persistent=False)

    def new_state(self, batch: int = 1) -> State:
        p = self.head.weight
        c = self.cfg
        return State.zeros(c.n_layer, batch, c.n_embd, c.n_head, c.head_size, device=p.device, dtype=p.dtype)

    # ---- forward ---------------------------------------------------------
    def forward(self, idx: torch.Tensor, state: State | None = None, last_only: bool = False,
                return_hidden: bool = False):
        """idx: [B, T] token ids. Returns (logits [B, T or 1, V], new State).
        return_hidden: return the final normed hidden states [B, T, C] instead of logits
        (see token_loss: the vocab projection is then done in slices, saving memory).

        Gradients flow into `state` if it requires grad (state tuning / BPTT across chunks).
        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        B = idx.shape[0]
        if state is None:
            state = self.new_state(B)
        x = self.blocks[0].ln0(self.emb(idx))
        tok = {"idx": idx, "x0": x}
        kv = state.kv
        if self.has_attention:  # the attention path sees the whole context: cached ids + new ids
            ctx = idx if kv is None else torch.cat((kv["idx"], idx), 1)
            tok.update(ctx_idx=ctx, ctx_x0=self.blocks[0].ln0(self.emb(ctx)), n_prev=ctx.shape[1] - idx.shape[1])
        v_first = None
        att_xs, wkvs, ffn_xs, caches = [], [], [], []
        for i, blk in enumerate(self.blocks):
            cache = None if kv is None else (kv["k"][i], kv["v"][i], kv["q"][i])
            args = (x, state.att_x[i], state.wkv[i], state.ffn_x[i], v_first, tok, cache)
            if self.grad_checkpoint and torch.is_grad_enabled():
                out = checkpoint(blk, *args, self.backend, use_reentrant=False)
            else:
                out = blk(*args, backend=self.backend)
            x, ax, s, fx, v_first, c = out
            x = x + self.steer[i].to(x.dtype)
            att_xs.append(ax); wkvs.append(s); ffn_xs.append(fx); caches.append(c)
        if last_only:
            x = x[:, -1:]
        x = self.ln_out(x)
        logits = x if return_hidden else self.head(x)
        new_kv = None
        if self.has_attention:
            new_kv = {"idx": tok["ctx_idx"], "k": torch.stack([c[0] for c in caches]),
                      "v": torch.stack([c[1] for c in caches]), "q": torch.stack([c[2] for c in caches])}
        return logits, State(torch.stack(att_xs), torch.stack(wkvs), torch.stack(ffn_xs), new_kv)


def _ce_slice(head_w, h, y):
    return F.cross_entropy((h @ head_w.t()).float(), y, reduction="none")


def token_loss(model: "RWKV7", hidden: torch.Tensor, targets: torch.Tensor, slice_len: int = 512) -> torch.Tensor:
    """Per-token cross-entropy from hidden states [T, C], projecting to the vocabulary in
    checkpointed slices so the full [T, V] float32 logits never exist at once."""
    w = model.head.weight
    outs = [checkpoint(_ce_slice, w, hidden[i:i + slice_len], targets[i:i + slice_len], use_reentrant=False)
            for i in range(0, hidden.shape[0], slice_len)]
    return torch.cat(outs)
