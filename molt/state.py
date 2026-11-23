"""Recurrent state of an RWKV-7 model: the model's working memory.

Per layer: the previous token's input to time-mix (att_x), the WKV matrix (wkv) and the
previous token's input to channel-mix (ffn_x). The WKV matrix is always float32: it
accumulates over arbitrarily long streams, and bf16 storage makes it drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class State:
    att_x: torch.Tensor  # [L, B, C]
    wkv: torch.Tensor    # [L, B, H, N, N] float32, [k, v] layout (see ops.py)
    ffn_x: torch.Tensor  # [L, B, C]
    # Only for hybrid variants with attention (x070b / DEA): token ids [B, T], per-layer compressed
    # key/value caches [L, B, T, 32] and the query token-shift [L, B, 256]. Grows with context.
    kv: dict | None = None

    @staticmethod
    def zeros(n_layer, batch, n_embd, n_head, head_size, device="cpu", dtype=torch.bfloat16):
        return State(
            torch.zeros(n_layer, batch, n_embd, device=device, dtype=dtype),
            torch.zeros(n_layer, batch, n_head, head_size, head_size, device=device, dtype=torch.float32),
            torch.zeros(n_layer, batch, n_embd, device=device, dtype=dtype),
        )

    # ---- tensor plumbing -------------------------------------------------
    def _map(self, fn) -> "State":
        kv = None if self.kv is None else {k: fn(v) for k, v in self.kv.items()}
        return State(fn(self.att_x), fn(self.wkv), fn(self.ffn_x), kv)

    def to(self, device=None, dtype=None) -> "State":
        """dtype applies to the token-shift parts only; wkv stays float32."""
        kv = None if self.kv is None else {k: v.to(device=device) for k, v in self.kv.items()}
        return State(self.att_x.to(device=device, dtype=dtype or self.att_x.dtype),
                     self.wkv.to(device=device),
                     self.ffn_x.to(device=device, dtype=dtype or self.ffn_x.dtype), kv)

    def clone(self) -> "State":
        return self._map(torch.clone)

    def detach(self) -> "State":
        return self._map(torch.Tensor.detach)

    @property
    def batch(self) -> int:
        return self.wkv.shape[1]

    def expand(self, batch: int) -> "State":
        assert self.batch == 1 and self.kv is None, "expand: batch-1 recurrent state only"
        return self._map(lambda t: t.expand(t.shape[0], batch, *t.shape[2:]).contiguous())

    # ---- memory operations -----------------------------------------------
    def decay(self, ratio: float) -> "State":
        """Fade the recurrent memory towards zero (returns a new State; an attention cache is kept as is)."""
        return State(self.att_x * ratio, self.wkv * ratio, self.ffn_x * ratio, self.kv)

    @staticmethod
    def mix(s1: "State", s2: "State", w1: float = 0.5, w2: float = 0.5) -> "State":
        return State(s1.att_x * w1 + s2.att_x * w2, s1.wkv * w1 + s2.wkv * w2, s1.ffn_x * w1 + s2.ffn_x * w2)

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        d = {"att_x": self.att_x.cpu(), "wkv": self.wkv.cpu(), "ffn_x": self.ffn_x.cpu()}
        if self.kv is not None:
            d["kv"] = {k: v.cpu() for k, v in self.kv.items()}
        torch.save(d, path)

    @staticmethod
    def load(path, device="cpu") -> "State":
        d = torch.load(path, map_location=device, weights_only=True)
        return State(d["att_x"], d["wkv"], d["ffn_x"], d.get("kv"))
