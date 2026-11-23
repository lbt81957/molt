"""Run BlinkDL's official demo model code as the reference, without compiling its CUDA kernel.

We parse a vendored demo file (tests/official/), keep only its imports, functions and model
classes, and execute them with RWKV7_OP replaced by a float32 PyTorch recurrence in the
official layout. Everything else - weight preprocessing, time-mix, FFN variants, DEA - is the
official code, so molt is compared against the source of truth rather than our transcription.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

OFFICIAL = Path(__file__).parent / "official"
DEMOS = {"x070": "rwkv_v7_demo_fast.py", "x070a": "rwkv_v7a_demo.py",
         "x070b": "rwkv_v7b_demo.py", "x080": "rwkv_v8_rc00_demo.py"}


def _rwkv7_op_reference(head_size):
    def op(state, r, w, k, v, a, b):
        """Official kernel semantics: w is pre-activation, decay = exp(-exp(w)); state[h] is S[v, k], updated in place."""
        T, C = r.shape
        H, N = C // head_size, head_size
        y = torch.empty(T, C, device=r.device, dtype=r.dtype)
        S = state
        for t in range(T):
            rt, wt, kt, vt, at, bt = (z[t].float().view(H, N) for z in (r, w, k, v, a, b))
            decay = torch.exp(-torch.exp(wt))
            S.copy_(S * decay.view(H, 1, N) + (S @ at.view(H, N, 1)) @ bt.view(H, 1, N) + vt.view(H, N, 1) @ kt.view(H, 1, N))
            y[t] = (S @ rt.view(H, N, 1)).view(C).to(r.dtype)
        return y
    return op


def load_official(variant: str, model_path: str, n_layer: int, n_embd: int, head_size: int = 64,
                  vocab_size: int = 65536):
    src = (OFFICIAL / DEMOS[variant]).read_text(encoding="utf-8")
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (isinstance(n, ast.ClassDef) and n.name.startswith("RWKV_x0"))]
    keep = [n for n in keep if not (isinstance(n, ast.FunctionDef) and n.name == "RWKV7_OP")]
    mod = ast.Module(body=keep, type_ignores=[])
    args = types.SimpleNamespace(MODEL_NAME=str(model_path).removesuffix(".pth"), n_layer=n_layer,
                                 n_embd=n_embd, vocab_size=vocab_size, head_size=head_size)
    ident = lambda f: f
    g = {"torch": torch, "nn": nn, "F": F, "List": List, "args": args, "DTYPE": torch.float32,
         "HEAD_SIZE": head_size, "MyModule": nn.Module, "MyFunction": ident, "MyStatic": ident,
         "RWKV7_OP": _rwkv7_op_reference(head_size), "__name__": "official_reference"}
    exec(compile(mod, DEMOS[variant], "exec"), g)
    cls = next(v for k, v in g.items() if k.startswith("RWKV_x0") and isinstance(v, type))
    return cls(args)
