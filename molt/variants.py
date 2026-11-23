"""Experimental RWKV generations built on the x070 time-mix.

  x070   RWKV-7 "Goose" (G1 series, production)
  x070a  RWKV-7a: FFN gated by a per-token 32x32 "deep embedding"        (BlinkDL rwkv_v7a_demo.py)
  x070b  RWKV-7b: x070a + DeepEmbedAttention (DEA), a small-KV-cache
         attention path added in parallel with time-mix                (BlinkDL rwkv_v7b_demo.py)
  x080   RWKV-8 rc00 "Heron" preview: FFN output scaled per token (enn) (BlinkDL rwkv_v8_rc00_demo.py)

Formulas follow BlinkDL/RWKV-LM (Apache-2.0); tests/official_harness.py checks them against
the official demo code itself.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Block, ChannelMix, RWKVConfig, _p


def detect_variant(sd: dict) -> str:
    if any(k.startswith("blocks.0.qkv.") for k in sd):
        return "x070b"
    if "blocks.0.ffn.s_emb.weight" in sd:
        return "x070a"
    if "blocks.0.ffn.enn.weight" in sd:
        return "x080"
    return "x070"


class ChannelMixDeepEmb(ChannelMix):
    """x070a/b: k <- k * (((x s1) @ S[idx]) s2 + s0), S[idx] = s_emb[idx] + x0 s_emb_x^T as 32x32."""

    def __init__(self, cfg: RWKVConfig, layer_id: int):
        super().__init__(cfg, layer_id)
        C, Fd, D = cfg.n_embd, cfg.dim_ffn, 32
        self.s1 = _p(C, D)
        self.s2 = _p(D, Fd)
        self.s0 = _p(1, 1, Fd)
        self.s_emb = nn.Embedding(cfg.vocab_size, D * D)
        self.s_emb_x = nn.Linear(C, D * D, bias=False)

    def forward(self, x, x_prev, tok=None):
        B, T, _ = x.shape
        k = self.hidden(x, x_prev)
        semb = (self.s_emb(tok["idx"]) + self.s_emb_x(tok["x0"])).view(B, T, 32, 32)
        ss = ((x @ self.s1).unsqueeze(-2) @ semb).squeeze(-2)
        k = k * (ss @ self.s2 + self.s0)
        return self.value(k), x[:, -1]


class ChannelMixEnn(ChannelMix):
    """x080 rc00: out = value(k) * enn[idx]."""

    def __init__(self, cfg: RWKVConfig, layer_id: int):
        super().__init__(cfg, layer_id)
        self.enn = nn.Embedding(cfg.vocab_size, cfg.n_embd)

    def forward(self, x, x_prev, tok=None):
        return self.value(self.hidden(x, x_prev)) * self.enn(tok["idx"]), x[:, -1]


class DeepEmbedAttention(nn.Module):
    """DEA (x070b). Caches only 32-dim projections of k and v per token and layer; full keys and
    values are rebuilt from them times per-token embeddings. Soft-capped softmax attention."""

    DQ, DR = 256, 32

    def __init__(self, cfg: RWKVConfig, layer_id: int):
        super().__init__()
        C, DQ, DR, V = cfg.n_embd, self.DQ, self.DR, cfg.vocab_size
        self.qq = nn.Linear(C, DQ, bias=False)
        self.x_q, self.x_k, self.x_v = _p(1, 1, DQ), _p(1, 1, DQ), _p(1, 1, C)
        self.k1, self.k2 = _p(C, DR), _p(DR, DQ)
        self.v1, self.v2 = _p(C, DR), _p(DR, C)
        self.k_emb, self.k_emb_x = nn.Embedding(V, DQ), nn.Linear(C, DQ, bias=False)
        self.v_emb, self.v_emb_x = nn.Embedding(V, C), nn.Linear(C, C, bias=False)
        self.lnq, self.lnk, self.lnv = nn.LayerNorm(DQ), nn.LayerNorm(DQ), nn.LayerNorm(C)

    def forward(self, x, tok, cache):
        B, T, C = x.shape
        q = self.qq(x)
        ks, vs = x @ self.k1, x @ self.v1
        if cache is not None:
            ks, vs, q_prev = torch.cat((cache[0], ks), 1), torch.cat((cache[1], vs), 1), cache[2]
        else:
            q_prev = torch.zeros(B, self.DQ, device=x.device, dtype=x.dtype)
        ctx, cx0 = tok["ctx_idx"], tok["ctx_x0"]
        k = (ks @ self.k2) * (self.k_emb(ctx) + self.k_emb_x(cx0))
        v = torch.tanh(vs @ self.v2) * (self.v_emb(ctx) + self.v_emb_x(cx0))
        q_last = q[:, -1]
        q = q + (torch.cat((q_prev.unsqueeze(1), q[:, :-1]), 1) - q) * self.x_q
        k = k + (F.pad(k, (0, 0, 1, -1)) - k) * self.x_k
        v = v + (F.pad(v, (0, 0, 1, -1)) - v) * self.x_v
        q, k, v = self.lnq(q), self.lnk(k), self.lnv(v)
        scores = 64 * torch.tanh((q @ k.mT) * (1.0 / 1024.0))  # soft cap
        if T > 1:
            pos_q = torch.arange(T, device=x.device).unsqueeze(1) + tok["n_prev"]
            pos_k = torch.arange(ks.shape[1], device=x.device).unsqueeze(0)
            scores = scores.masked_fill(pos_k > pos_q, float("-inf"))
        return scores.softmax(-1) @ v, (ks, vs, q_last)


class BlockDEA(Block):
    def __init__(self, cfg: RWKVConfig, layer_id: int, ffn_cls):
        super().__init__(cfg, layer_id, ffn_cls)
        self.qkv = DeepEmbedAttention(cfg, layer_id)

    def forward(self, x, att_x, wkv, ffn_x, v_first, tok, cache=None, backend="auto"):
        a, cache = self.qkv(x, tok, cache)
        h, att_x, wkv, v_first = self.att(self.ln1(x), att_x, wkv, v_first, backend)
        x = x + h + a
        h, ffn_x = self.ffn(self.ln2(x), ffn_x, tok)
        return x + h, att_x, wkv, ffn_x, v_first, cache


_FFN = {"x070": ChannelMix, "x070a": ChannelMixDeepEmb, "x070b": ChannelMixDeepEmb, "x080": ChannelMixEnn}
_BLOCK = {"x070b": BlockDEA}


def ffn_class(variant: str):
    if variant not in _FFN:
        raise NotImplementedError(f"variant {variant} is not supported (supported: {sorted(_FFN)})")
    return _FFN[variant]


def block_class(variant: str):
    return _BLOCK.get(variant, Block)
