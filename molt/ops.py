"""WKV-7 operator backends.

All backends share one convention (the one used by flash-linear-attention):
  r, w, k, v, a, b : [B, T, H, N]   (w is the *log* decay, i.e. decay = exp(w) in (0, 1))
  state            : [B, H, N_k, N_v], float32
  returns          : (o [B, T, H, N], new_state [B, H, N_k, N_v])

Recurrence per head, with S in [k, v] layout:
  S_t = diag(exp(w_t)) S_{t-1} + b_t (a_t^T S_{t-1}) + k_t v_t^T
  o_t = S_t^T r_t
(The official RWKV-7 code stores the transpose, S[v, k].)

Backends:
  "fla"    Triton chunked kernel; differentiable w.r.t. every input *and* the initial state.
           Works on Windows via triton-windows, no MSVC needed. Used for training and prefill.
  "fla_rec" Triton fused recurrent kernel; fastest for decoding a few tokens.
  "step"   Single-token PyTorch update; default for decoding (T == 1), CUDA-graph friendly.
  "torch"  Pure PyTorch reference (float32). Slow; for CPU, tests and debugging.
"""
from __future__ import annotations

import torch

_BACKENDS = {}


def register(name):
    def deco(fn):
        _BACKENDS[name] = fn
        return fn
    return deco


def available_backends() -> list[str]:
    out = ["torch"]
    try:
        import fla.ops.rwkv7  # noqa: F401
        out += ["fla", "fla_rec"]
    except Exception:
        pass
    return out


def wkv7(r, w, k, v, a, b, state, backend: str = "auto"):
    if backend == "auto":
        if r.shape[1] == 1 and not torch.is_grad_enabled():
            backend = "step"
        elif r.is_cuda and "fla" in available_backends():
            backend = "fla" if r.shape[1] > 16 or torch.is_grad_enabled() else "fla_rec"
        else:
            backend = "torch"
    return _BACKENDS[backend](r, w, k, v, a, b, state)


@register("torch")
def wkv7_torch(r, w, k, v, a, b, state):
    dt = r.dtype
    r, w, k, v, a, b = (x.float() for x in (r, w, k, v, a, b))
    S = state.float()
    decay = w.exp()
    outs = []
    for t in range(r.shape[1]):
        aS = torch.einsum("bhk,bhkv->bhv", a[:, t], S)
        S = decay[:, t, :, :, None] * S + b[:, t, :, :, None] * aS[:, :, None, :] \
            + k[:, t, :, :, None] * v[:, t, :, None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", S, r[:, t]))
    return torch.stack(outs, 1).to(dt), S


@register("step")
def wkv7_step(r, w, k, v, a, b, state):
    """Single-token update with a handful of small kernels (decode path; CUDA-graph friendly)."""
    dt = r.dtype
    r, w, k, v, a, b = (x[:, 0].float() for x in (r, w, k, v, a, b))   # [B, H, N]
    S = state.float()
    aS = (a.unsqueeze(-1) * S).sum(-2)                                # [B, H, V]
    S = w.exp().unsqueeze(-1) * S + b.unsqueeze(-1) * aS.unsqueeze(-2) + k.unsqueeze(-1) * v.unsqueeze(-2)
    o = (S * r.unsqueeze(-1)).sum(-2)
    return o.unsqueeze(1).to(dt), S


@register("fla")
def wkv7_fla_chunk(r, w, k, v, a, b, state):
    from fla.ops.rwkv7 import chunk_rwkv7
    o, s = chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=state.float(), output_final_state=True)
    return o, s.float()


@register("fla_rec")
def wkv7_fla_recurrent(r, w, k, v, a, b, state):
    from fla.ops.rwkv7 import fused_recurrent_rwkv7
    o, s = fused_recurrent_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=state.float(), output_final_state=True)
    return o, s.float()
